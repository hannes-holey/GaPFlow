#
# Copyright 2026 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# flake8: noqa: W503
"""Taylor-Hood Q2Q1 FEM Solver.

Wires GridIndexManager, QuadFieldManager, Assembly, and linear solver into a
Newton iteration loop.
"""

import time
from typing import Callable, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..parallel import MPI
from .elements import TaylorHoodQ2Q1
from .grid_index import GridIndexManager
from .quad_fields import QuadFieldManager
from .assembly import Assembly
from .fieldspec import FieldSpec, VAR_GRID, RES_GRID, patch_registry_for_gp
from .scipy_system import ScipySystem

from ..bc import (GhostUpdater, BoundarySpec, BND_IDX, sample_bc_spec,
                  translate_bc_rho_to_p, resolve_pressure_bcs)
from .solution_guards import clamp_solution
from .terms import get_active_terms
from .scaling import build_scaling, build_scaling_from_blocks
from . import checkpoint
from ..logging import get_logger

logger = get_logger("gapflow.run")

if TYPE_CHECKING:
    from ..problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver:
    """FEM Solver for 2D Taylor-Hood Q2Q1 problems.

    Parameters
    ----------
    fem_spec : dict
        fem_solver config block from the Problem.
    problem : Problem
        Problem instance.
    """

    def __init__(self, fem_spec: dict, problem: "Problem") -> None:

        self.fem_spec = fem_spec
        self.problem = problem
        self.R_norm_history: List[List[float]] = []
        self.rank = problem.decomp.rank
        self.cb_list: List[Callable[["FEMSolver"], None]] = []

        self._build_variable_and_residual_lists()

        self.elements = TaylorHoodQ2Q1(problem.grid['dx'], problem.grid['dy'])
        self.grid_idx = GridIndexManager(problem.decomp, self)

        self._get_active_terms()

    def _build_variable_and_residual_lists(self) -> None:
        """Check active equations and build lists."""

        p = self.problem
        self.energy = p.fem_solver['equations']['energy']
        self.cavitation = p.fem_solver['equations']['cavitation']
        self.oss = p.fem_solver['stabilization']['oss']
        self.reynolds = p.fem_solver['equations']['reynolds']

        if self.reynolds:
            self.variables = ['p']
            self.residuals = ['mass']
        else:
            self.variables = ['jx', 'jy', 'p']
            self.residuals = ['momentum_x', 'momentum_y', 'mass']

        self.add_fields = []

        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')
            self.add_fields.append('e')
        if self.cavitation:
            self.variables.append('theta')
            self.residuals.append('fb')
            self.add_fields.append('theta')
        if self.cavitation and self.oss and not self.reynolds:
            self.variables.append('xi')
            self.residuals.append('R_oss')
            self.add_fields.append('xi')

        self.var_specs = [FieldSpec(name=v, grid=VAR_GRID[v], idx=i)
                          for i, v in enumerate(self.variables)]
        self.res_specs = [FieldSpec(name=r, grid=RES_GRID[r], idx=i)
                          for i, r in enumerate(self.residuals)]

        nb_inner = {
            'P1': np.prod(p.decomp.nb_subdomain_grid_pts),
            'P2': np.prod(p.decomp.nb_subdomain_grid_pts_P2),
        }
        self.res_size = sum(nb_inner[spec.grid] for spec in self.res_specs)

    # =========================================================================
    # Boundary conditions
    # =========================================================================

    def _apply_bc_callbacks(self, var_name, bc_type):
        """Override bc_type/bc_functions at boundaries with a registered
        problem.set_bc_function callback."""
        callbacks = self.problem._bc_callbacks.get(var_name, {})
        bc_type = list(bc_type)
        bc_functions = [None] * 4
        for bnd, fn in callbacks.items():
            idx = BND_IDX[bnd]
            bc_type[idx] = 'F'
            bc_functions[idx] = fn
        return bc_type, bc_functions

    def build_boundary_conditions(self):
        """Build the list of BoundarySpec objects for BC application.
        Note: bc_spec needs to be sampled in the same order as variables."""

        resolve_pressure_bcs(self.problem.grid, self.problem.prop, problem=self.problem)
        specs = []
        no_fun = [None] * 4

        if 'jx' in self.variables:
            field = self.quad_mgr.nodal_fields['jx']
            jx_bc_type, jx_bc_vals = sample_bc_spec(self.problem.grid, 1)
            jx_bc_type, jx_bc_fun = self._apply_bc_callbacks('jx', jx_bc_type)
            specs.append(BoundarySpec(field, 'P2', jx_bc_type,
                                      jx_bc_vals, jx_bc_fun, self.problem.decomp))

        if 'jy' in self.variables:
            field = self.quad_mgr.nodal_fields['jy']
            jy_bc_type, jy_bc_vals = sample_bc_spec(self.problem.grid, 2)
            jy_bc_type, jy_bc_fun = self._apply_bc_callbacks('jy', jy_bc_type)
            specs.append(BoundarySpec(field, 'P2', jy_bc_type,
                                      jy_bc_vals, jy_bc_fun, self.problem.decomp))

        field = self.quad_mgr.nodal_fields['p']
        rho_bc_type, rho_bc_vals = sample_bc_spec(self.problem.grid, 0)
        p_bc_vals = translate_bc_rho_to_p(rho_bc_type, rho_bc_vals, self.problem)
        rho_bc_type, rho_bc_fun = self._apply_bc_callbacks('rho', rho_bc_type)
        specs.append(BoundarySpec(field, 'P1', rho_bc_type,
                                  p_bc_vals, rho_bc_fun, self.problem.decomp))

        if self.cavitation:
            field = self.quad_mgr.nodal_fields['theta']
            specs.append(BoundarySpec(field, 'P1', rho_bc_type,
                                      [0, 0, 0, 0], no_fun, self.problem.decomp))

        if self.cavitation and self.oss and not self.reynolds:
            field = self.quad_mgr.nodal_fields['xi']
            specs.append(BoundarySpec(field, 'P1', ['D', 'D', 'D', 'D'],
                                      [0, 0, 0, 0], no_fun, self.problem.decomp))

        if self.energy:
            pass

        self.ghost_updater = GhostUpdater(
            decomp=self.problem.decomp,
            problem=self.problem,
            specs=specs
        )

        self.bc_specs = specs

    # =========================================================================
    # Initialisation
    # =========================================================================

    @property
    def _res_slices(self):
        return self.assembly._res_slices

    @property
    def _sol_slices(self):
        return self.assembly._sol_slices

    @property
    def nb_sol(self):
        return len(self.variables)

    def _get_active_terms(self) -> None:
        self.terms = get_active_terms(self.fem_spec)

    def _init_quad_fields(self) -> None:
        self.quad_mgr = QuadFieldManager(
            problem=self.problem,
            energy=self.energy,
            cavitation=self.cavitation,
            variables=self.variables,
            add_fields=self.add_fields,
            elements=self.elements,
            decomp=self.problem.decomp,
            terms=self.terms,
        )

    def _build_assembly(self) -> None:
        self.assembly = Assembly(
            grid_idx=self.grid_idx,
            element=self.elements,
            var_specs=self.var_specs,
            res_specs=self.res_specs,
            terms=self.terms,
        )

    def _build_jit_functions(self) -> None:
        p = self.problem
        p.pressure.build_grad()
        p.wall_stress_xz.build_grad()
        p.wall_stress_yz.build_grad()
        if self.energy:
            p.energy.build_grad()

    def _build_terms(self) -> None:
        p = self.problem

        def make_getter(name):
            return lambda: self.quad_mgr.get_quad_sq(name)

        _, quad_fields, _ = self.quad_mgr._needed_fields()
        for term in self.terms:

            # fields
            ctx = {name: make_getter(name) for name in quad_fields}

            # scalars
            ctx['dt'] = lambda: p.numerics['dt']
            ctx['ad_alpha'] = lambda: p.fem_solver['stabilization']['ad_alpha']
            ctx['fc_beta'] = lambda: p.fem_solver['stabilization']['fc_beta']
            ctx['p_cav'] = lambda: p.prop['p_cav']
            ctx['fb_p_ref'] = lambda: 1e05
            ctx['dx'] = lambda: p.grid['dx']
            ctx['dy'] = lambda: p.grid['dy']

            if self.energy:
                ctx['k'] = lambda: p.energy.k
            term.build(ctx)

    def _init_linear_solver(self) -> None:
        petsc_info = self.assembly.get_petsc_info(res_size=self.res_size)
        solver_type = self.problem.fem_solver['linear_solver']

        try:
            from .. import HAS_PETSC
        except ImportError:
            HAS_PETSC = False

        if HAS_PETSC:
            from .petsc_system import PETScSystem
            self.linear_solver = PETScSystem(petsc_info,
                                             solver_type=solver_type,
                                             print_mumps_diagnostics=self.problem.fem_solver.get(
                                                 'print_mumps_diagnostics', False))
        else:
            if MPI.COMM_WORLD.size > 1:
                raise RuntimeError(
                    "PETSc required for parallel execution.")
            self.linear_solver = ScipySystem(petsc_info,
                                             solver_type=solver_type,
                                             assembly=self.assembly)

        self.scaling = build_scaling(
            self.problem, self.energy, self.variables, self.assembly,
            cavitation=self.cavitation)

    # =========================================================================
    # Quadrature field update
    # =========================================================================

    def update_quad(self, it: int = 0, **kwargs) -> dict:
        """Update models and quadrature fields for the current solution guess."""
        self.quad_mgr.update_physics(it, **kwargs)
        self.quad_mgr.update_quad_fields()
        return self.quad_mgr.collect_quad_fields()

    def update_prev_quad(self) -> None:
        """Update 'previous' quadrature fields for time-dependent terms."""
        self.quad_mgr.store_prev_values()

    # =========================================================================
    # Newton scatter / gather
    # =========================================================================

    def get_q_nodal(self) -> NDArray:
        """Gather inner nodal values into a flat solution vector."""
        q = np.zeros(self.res_size)
        for var in self.variables:
            q[self._sol_slices[var]] = self.quad_mgr.get_nodal_sol_val(var)
        return q

    def update_q_nodal(self, q: NDArray) -> None:
        """Scatter flat solution vector back to nodal fields and update ghosts."""
        for var in self.variables:
            self.quad_mgr.set_nodal_sol_val(var, q[self._sol_slices[var]])
        self.ghost_updater.update()

    # =========================================================================
    # Assembly
    # =========================================================================

    def get_M(self, qf: dict = None) -> NDArray:
        """Assemble Jacobian COO values."""
        if qf is None:
            qf = self.quad_mgr.collect_quad_fields()
        return self.assembly.assemble_matrix(qf)

    def get_M_dense(self) -> NDArray:
        """Debug/test helper: assemble Jacobian as a dense (res_size, res_size) matrix in block ordering."""
        coo = self.get_M()
        n_nnz = len(coo)
        block_rows = np.empty(n_nnz, dtype=np.int64)
        block_cols = np.empty(n_nnz, dtype=np.int64)

        for (res, var), block in self.assembly.block_order.items():
            s = block['nnz_idx_start']
            n = block['nb_nnz']
            row_off = self.assembly._res_slices[res].start
            col_off = self.assembly._sol_slices[var].start
            block_rows[s:s + n] = row_off + self.assembly.nnz_local_to[s:s + n]
            block_cols[s:s + n] = col_off + self.assembly.nnz_local_from[s:s + n]

        M = np.zeros((self.res_size, self.res_size), dtype=np.float64)
        np.add.at(M, (block_rows, block_cols), coo)
        return M

    def get_R_(self, qf: dict = None) -> NDArray:
        """Assemble residual vector."""
        if qf is None:
            qf = self.quad_mgr.collect_quad_fields()
        return self.assembly.assemble_rhs(qf)

    # =========================================================================
    # Solver step
    # =========================================================================

    def update_quad_and_assemble(self, it: int = 0) -> Tuple[NDArray, NDArray]:
        """Update models, quadrature fields, and assemble M and R."""
        qf = self.update_quad(it)
        M = self.get_M(qf)
        R = self.get_R_(qf).copy()
        return M, R

    def get_R(self, q_guess: NDArray, it: int = 1) -> float:
        """Compute the local residual vector for a given solution guess."""
        self.update_q_nodal(q_guess)
        qf = self.update_quad(it)
        R = self.get_R_(qf).copy()
        return R

    def get_R_norm_global(self, R: NDArray) -> float:
        """Compute the L2 norm of the global residual vector."""
        p = self.problem
        comm = p.decomp._mpi_comm
        R_norm_local_sq = float(np.linalg.norm(R) ** 2)
        R_norm_global_sq = comm.allreduce(R_norm_local_sq, op=MPI.SUM)
        R_norm_global = float(np.sqrt(R_norm_global_sq))
        return R_norm_global

    # =========================================================================
    # Output helpers
    # =========================================================================

    def update_output_fields(self) -> None:
        """Update stress models to ensure output fields are up to date before writing."""
        p = self.problem
        self.quad_mgr.sync_to_problem_q()
        p.pressure.update(residuals=p.residual_buffer)
        if 'jx' in self.variables:
            p.wall_stress_xz.update(residuals=p.residual_buffer)
        if 'jy' in self.variables:
            p.wall_stress_yz.update(residuals=p.residual_buffer)
        if hasattr(p, 'bulk_stress') and 'jx' in self.variables and 'jy' in self.variables:
            p.bulk_stress.update()

    # =========================================================================
    # Pre-run setup
    # =========================================================================

    def pre_run(self, **kwargs) -> None:
        """Initialize assembly, jit functions, and solver. Load initial state and
        update quadrature fields. Fetch solver parameters."""

        p = self.problem

        self._init_quad_fields()
        self._build_assembly()
        self._build_jit_functions()
        if (p.pressure.is_gp_model
                or p.wall_stress_xz.is_gp_model
                or p.wall_stress_yz.is_gp_model):
            patch_registry_for_gp(
                pressure=p.pressure.is_gp_model,
                wall_stress_xz=p.wall_stress_xz.is_gp_model,
                wall_stress_yz=p.wall_stress_yz.is_gp_model,
            )

        self.build_boundary_conditions()
        self._build_terms()
        self.assembly.build_assembly_templates()
        if p.topo.elastic and p.fem_solver['physics']['einm']:
            self.assembly.build_elastic_templates(p.topo.ElasticDeformation)
        self._init_linear_solver()

        self.quad_mgr.sync_from_problem_q()

        self.update_quad(**kwargs)
        self.update_prev_quad()
        self.update_output_fields()

        self.time_inner = 0.0
        self.inner_iterations = 0

        self.tol = p.fem_solver['R_norm_tol']
        self.alpha = p.fem_solver['newton_relax']
        self.max_iter = p.fem_solver['max_iter']

    # =========================================================================
    # Time step
    # =========================================================================

    def check_residual(self, R: NDArray, it: int) -> float:
        """Check residual norm and return True if converged."""
        R_norm = self.get_R_norm_global(R)
        if self.rank == 0:
            self.R_norm_history[-1].append(R_norm)
        if R_norm < self.tol and it > 0:
            return True
        return False

    def post_solve(self, q: NDArray, dq: NDArray) -> None:
        """Post-process obtained solution update: Newton step and cavitation clamping."""
        q = q + self.alpha * dq
        q = clamp_solution(q, self)
        return q

    def run_callbacks(self) -> None:
        """Run per-Newton-iteration callbacks."""
        for cb in self.cb_list:
            cb(self)

    def wrap_up_timestep(self, it: int, tic: float) -> None:
        """Wrap up the finished timstep."""

        toc = time.time()
        self.time_inner = toc - tic
        self.inner_iterations = it + 1

    def update(self) -> None:
        """Perform one time step with Newton iteration."""

        p = self.problem
        tic = time.time()

        self.update_prev_quad()
        q = self.get_q_nodal().copy()

        if self.rank == 0:
            self.R_norm_history.append([])

        # Inner Newton loop
        for it in range(self.max_iter):

            M, R = self.update_quad_and_assemble(it)
            if self.check_residual(R, it):
                break
            dq, M_scaled = self.solve_linear_system(M, R, it)
            q = self.post_solve(q, dq)
            self.run_callbacks()
            self.update_q_nodal(q)

        self.wrap_up_timestep(it, tic)
        self.update_output_fields()
        p._post_update()

    def solve_linear_system(self, M: NDArray, R: NDArray, it: int = 0) -> tuple:
        """Scale the system, solve for dq, and unscale. Returns M_scaled for debugging.

        Returns
        -------
        dq : ndarray
            Unscaled Newton update.
        M_scaled : ndarray
            The matrix passed to the linear solver (M itself if scaling is off).
        """
        fem_solver = self.problem.fem_solver
        if fem_solver['scaling']:
            scale_interval = fem_solver['scaling_update_interval']
            step = self.problem.step
            if (it == 0
                    and (step == 0 or step % scale_interval == 0)):
                self.scaling = build_scaling_from_blocks(
                    M, self.variables, self.residuals, self.assembly,
                    n_iter=fem_solver['scaling_ruiz_iter'])
            M_scaled, R_scaled = self.scaling.scale_system(M, R)
            self.linear_solver.assemble(M_scaled, R_scaled)
            dq = self.scaling.unscale_solution(self.linear_solver.solve())
        else:
            M_scaled = M
            self.linear_solver.assemble(M, R)
            dq = self.linear_solver.solve()

        return dq, M_scaled

    # =========================================================================
    # Checkpointing (warm restart)
    # =========================================================================

    def save_state(self, path: str) -> None:
        """Save a full checkpoint of the current simulation state. See `Problem.save_state`."""
        checkpoint.save_state(self, path)

    def load_state(self, path: str) -> None:
        """Restore a checkpoint written by `save_state`. See `Problem.load_state`."""
        checkpoint.load_state(self, path)

    # =========================================================================
    # Status / diagnostics
    # =========================================================================

    def print_status_header(self) -> None:
        p = self.problem
        if p.options.get('print_progress') and p.decomp.rank == 0:
            logger.info(78 * '-')
            logger.info(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} "
                        f"{'Iter':<6s} {'Conv. Time':<12s} {'Residual':<12s} {'|R| Newton':<14s}")
            logger.info(78 * '-')
        self.print_status()
        if p.options.get('save_output'):
            p.write()

    def print_status(self) -> None:
        """
        Log the current status line, if enabled.
        """
        p = self.problem
        if p.options.get('print_progress') and p.decomp.rank == 0:
            history = self.R_norm_history
            R_newton = history[-1][-1] if history and history[-1] else float('nan')
            logger.info(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} "
                        f"{self.inner_iterations:<6d} "
                        f"{self.time_inner:<12.4e} {p.residual:<12.4e} {R_newton:<14.4e}")

    def status_record(self) -> dict:
        """
        Scalar values recorded to history.csv for the current step.
        """
        p = self.problem
        history = self.R_norm_history
        R_newton = history[-1][-1] if history and history[-1] else float('nan')
        return {
            "step": p.step,
            "dt": p.dt,
            "time": p.simtime,
            "inner_iterations": self.inner_iterations,
            "time_inner": self.time_inner,
            "residual": p.residual,
            "R_newton": R_newton,
        }
