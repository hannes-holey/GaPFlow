#
# Copyright 2025 Christoph Huber
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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from . import HAS_PETSC
from .fem_2d.elements import TriangleQuadrature
from .fem_2d.terms import NonLinearTerm, get_active_terms
from .fem_2d.assembly import Assembly
from .fem_2d.scaling import build_scaling
from .fem_2d.grid_index import GridIndexManager
from .fem_2d.quad_fields import QuadFieldManager
from .fem_2d.solution_guards import apply_guards
from .fem_2d.residual_analysis import (
    create_residual_analysis_plot,
    plot_residual_history,
    print_residual_analysis,
)

import numpy as np
import time
from muGrid import Timer
from mpi4py import MPI

import numpy.typing as npt
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from .problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver2D:
    """FEM Solver for 2D problems using triangular elements."""

    def __init__(self, fem_spec: dict, problem: "Problem") -> None:
        self.fem_spec = fem_spec
        self.problem = problem
        self.timer = Timer()
        self.R_norm_history = []  # Nested list: [[step1_iters], [step2_iters], ...]

    def _init_convenience_accessors(self) -> None:
        """Initialize convenience accessors for problem and grid properties."""
        p = self.problem
        self.energy = p.fem_solver['equations']['energy']

        nb_subdomain_pts = p.decomp.nb_subdomain_grid_pts
        self.Nx_inner = nb_subdomain_pts[0]
        self.Ny_inner = nb_subdomain_pts[1]

        self.dx = p.grid['dx']
        self.dy = p.grid['dy']

        self.variables = ['rho', 'jx', 'jy']
        self.residuals = ['mass', 'momentum_x', 'momentum_y']
        self.field_map = {'rho': p.q[0], 'jx': p.q[1], 'jy': p.q[2]}
        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')
            self.field_map['E'] = p.energy.energy

        # Grid index manager (handles masks, connectivity, stencils, BCs)
        energy_spec = p.energy_spec if self.energy else None
        self.grid_idx = GridIndexManager(
            decomp=p.decomp,
            variables=self.variables,
            energy_spec=energy_spec,
        )

        self.nb_inner_pts = self.grid_idx.nb_inner_pts

        self.res_size = len(self.residuals) * self.nb_inner_pts
        self.var_size = len(self.variables) * self.grid_idx.nb_contributors

        # Quadrature data and element tensors for triangular elements
        self.quad = TriangleQuadrature(self.dx, self.dy)

    def _get_active_terms(self) -> None:
        """Initialize list of active terms from problem fem_solver config."""
        self.terms = get_active_terms(self.fem_spec)

    def _init_quad_fields(self) -> None:
        """Initialize quadrature field manager with operators and fields."""
        self.quad_mgr = QuadFieldManager(
            problem=self.problem,
            energy=self.energy,
            variables=self.variables,
            quad=self.quad,
        )

    def _build_jit_functions(self) -> None:
        """Build JIT-compiled gradient functions for all physical models."""
        p = self.problem

        p.pressure.build_grad()
        p.wall_stress_xz.build_grad()
        p.wall_stress_yz.build_grad()
        if self.energy:
            p.energy.build_grad()

    def _build_terms(self) -> None:
        """Build term contexts with callable references to quad_fields."""
        p = self.problem

        def make_getter(name):
            return lambda: self.quad_mgr.get(name)

        quad_field_names = self.quad_mgr.get_needed_fields()

        for term in self.terms:
            term_ctx = {name: make_getter(name) for name in quad_field_names}

            # Non-quad values
            term_ctx['dt'] = p.numerics['dt']

            if self.energy:
                term_ctx['k'] = lambda: p.energy.k

            term.build(term_ctx)

    # =========================================================================
    # Quadrature Field Update (delegates to quad_mgr)
    # =========================================================================

    def update_quad(self) -> None:
        """Full quadrature update (nodal fields, interpolation, derived quantities)."""
        self.quad_mgr.update_nodal_fields()
        self.quad_mgr.update_quad_nodal()
        self.quad_mgr.update_quad_computed()

    def update_prev_quad(self) -> None:
        """Store current quad values for time derivatives."""
        self.quad_mgr.store_prev_values()

    # =========================================================================
    # Assembly Layout (unified COO pattern and term mappings)
    # =========================================================================

    def _build_assembly(self) -> None:
        """Precompute assembly layout for O(nnz) matrix/vector assembly."""
        self.assembly = Assembly(
            grid_idx=self.grid_idx,
            quad=self.quad,
            terms=self.terms,
            residuals=self.residuals,
            variables=self.variables
        )

    # =========================================================================
    # Sparse Assembly Methods (O(nnz) direct COO value computation)
    # =========================================================================

    def get_tang_matrix_sparse(self) -> NDArray:
        """Assemble tangent matrix directly to COO values - O(nnz) memory."""
        coo_values = np.zeros(self.assembly.nnz, dtype=np.float64)

        for term in self.terms:
            for dep_var in term.dep_vars:
                self._accumulate_term_sparse(term, dep_var, coo_values)

        return coo_values

    def _accumulate_term_sparse(self, term: NonLinearTerm, dep_var: str,
                                coo_values: NDArray):
        """Dispatch to appropriate sparse assembly method based on term type."""
        if term.d_dx_resfun:
            self._accumulate_term_deriv_sparse(term, dep_var, coo_values, 'x')
        elif term.d_dy_resfun:
            self._accumulate_term_deriv_sparse(term, dep_var, coo_values, 'y')
        else:
            self._accumulate_term_zero_der_sparse(term, dep_var, coo_values)

    def _accumulate_term_zero_der_sparse(self, term: NonLinearTerm, dep_var: str,
                                         coo_values: NDArray):
        """Sparse assembly for zero-derivative term using template."""
        key = (term.name, term.res, dep_var)
        ref = self.assembly.term_templates[key]
        template = ref.template

        if len(template.template_coo_idx) == 0:
            return

        res_deriv = self.get_res_deriv_vals(term, dep_var)
        res_vals = res_deriv.ravel()[template.flat_field_idx]

        # Compute actual COO indices from template + block offset
        actual_coo_idx = template.template_coo_idx + ref.block_offset

        contrib = template.weights * res_vals
        np.add.at(coo_values, actual_coo_idx, contrib)

    def _accumulate_term_deriv_sparse(self, term: NonLinearTerm, dep_var: str,
                                      coo_values: NDArray, direction: str):
        """Sparse assembly for derivative term with ±1/d* stencil using template."""
        key = (term.name, term.res, dep_var)
        ref = self.assembly.term_templates[key]
        template = ref.template

        if len(template.template_coo_idx) == 0:
            return

        res_deriv = self.get_res_deriv_vals(term, dep_var)
        res_vals = res_deriv.ravel()[template.flat_field_idx]
        inv_d = 1.0 / (self.dx if direction == 'x' else self.dy)

        # Compute actual COO indices from template + block offset
        actual_coo_idx = template.template_coo_idx + ref.block_offset

        contrib = template.weights * res_vals * template.signs * inv_d
        np.add.at(coo_values, actual_coo_idx, contrib)

    # =========================================================================
    # Residual Vector Assembly
    # =========================================================================

    def get_res_deriv_vals(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Evaluate derivative of residual function w.r.t. dep_var at quadrature points."""
        dep_var_vals = {v: self.quad_mgr.get(v) for v in term.dep_vars}
        return term.evaluate_deriv(dep_var, *[dep_var_vals[v] for v in term.dep_vars])

    def get_res_vals(self, term: NonLinearTerm) -> NDArray:
        """Evaluate residual function at quadrature points."""
        dep_var_vals = {v: self.quad_mgr.get(v) for v in term.dep_vars}
        return term.evaluate(*[dep_var_vals[v] for v in term.dep_vars])

    def residual_vector_term(self, term: NonLinearTerm) -> NDArray:
        """Wrapper for different spatial derivatives (nb_inner_pts,)."""
        if term.d_dx_resfun:
            return self._residual_vector_term_deriv(term, 'x')
        elif term.d_dy_resfun:
            return self._residual_vector_term_deriv(term, 'y')
        elif term.der_testfun in ('x', 'y'):
            return self._residual_vector_term_testfun_deriv(term)
        else:
            return self._residual_vector_term_zero_der(term)

    def _assemble_residual_contributions(self, contrib_left: NDArray,
                                         contrib_right: NDArray) -> NDArray:
        """Assemble triangle contributions into residual vector."""
        R = np.zeros((self.nb_inner_pts,), dtype=float)
        TO = self.grid_idx.sq_TO_inner

        for t, contrib in enumerate([contrib_left, contrib_right]):
            TO_tri = TO[:, self.quad.TRI_PTS[t]]
            for i in range(3):
                valid = TO_tri[:, i] != -1
                np.add.at(R, TO_tri[valid, i], contrib[valid, i])
        return R

    def _residual_vector_term_zero_der(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term without derivative in residual function."""
        res_fun_vals = self.get_res_vals(term)  # (6, sq_per_row, sq_per_col)

        sx, sy = self.grid_idx.sq_x_arr, self.grid_idx.sq_y_arr
        # Quad layout: [0:3] = left triangle, [3:6] = right triangle
        res_left = res_fun_vals[0:3, sx, sy].T   # (nb_sq, 3)
        res_right = res_fun_vals[3:6, sx, sy].T

        # contrib[sq, i] = sum_q(N_i[q] * res[sq, q] * w[q]) * A
        contrib_left = np.einsum('iq,sq->si', self.quad.test_wA[0], res_left)
        contrib_right = np.einsum('iq,sq->si', self.quad.test_wA[1], res_right)

        return self._assemble_residual_contributions(contrib_left, contrib_right)

    def _residual_vector_term_testfun_deriv(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for test-function-derivative-only term (PSPG).

        Computes INT (dN_i/dx_k) * F(q) dOmega where F has no spatial derivative.
        Uses test function gradient weights instead of standard test function weights.
        """
        res_fun_vals = self.get_res_vals(term)  # (6, sq_per_row, sq_per_col)

        sx, sy = self.grid_idx.sq_x_arr, self.grid_idx.sq_y_arr
        res_left = res_fun_vals[0:3, sx, sy].T   # (nb_sq, 3)
        res_right = res_fun_vals[3:6, sx, sy].T

        test_wA = self.quad.test_wA_dx if term.der_testfun == 'x' \
            else self.quad.test_wA_dy

        contrib_left = np.einsum('iq,sq->si', test_wA[0], res_left)
        contrib_right = np.einsum('iq,sq->si', test_wA[1], res_right)

        return self._assemble_residual_contributions(contrib_left, contrib_right)

    def _residual_vector_term_deriv(self, term: NonLinearTerm, direction: str) -> NDArray:
        """Get residual vector for term with spatial derivative in residual function."""
        # Field derivative direction (for chain rule computation)
        get_field_deriv = self.quad_mgr.get_deriv_dx if direction == 'x' \
            else self.quad_mgr.get_deriv_dy

        # Test function weights (decoupled from field derivative direction)
        if term.der_testfun is True:
            # Legacy (PSPG pressure): test weight dir matches field deriv dir
            test_wA = self.quad.test_wA_dx if direction == 'x' \
                else self.quad.test_wA_dy
        elif term.der_testfun == 'x':
            test_wA = self.quad.test_wA_dx
        elif term.der_testfun == 'y':
            test_wA = self.quad.test_wA_dy
        else:
            test_wA = self.quad.test_wA

        # Compute dF/d* using chain rule: (6, sq_per_row, sq_per_col)
        dF = np.zeros((6, self.grid_idx.sq_per_row, self.grid_idx.sq_per_col))
        for dep_var in term.dep_vars:
            dF_dvar = self.get_res_deriv_vals(term, dep_var)  # (6, X, Y)
            dvar_2 = get_field_deriv(dep_var)                 # (2, X, Y)
            dvar_6 = np.repeat(dvar_2, 3, axis=0)             # expand to (6, X, Y)
            dF += dF_dvar * dvar_6

        sx, sy = self.grid_idx.sq_x_arr, self.grid_idx.sq_y_arr
        # Quad layout: [0:3] = left triangle, [3:6] = right triangle
        res_left = dF[0:3, sx, sy].T   # (nb_sq, 3)
        res_right = dF[3:6, sx, sy].T

        contrib_left = np.einsum('iq,sq->si', test_wA[0], res_left)
        contrib_right = np.einsum('iq,sq->si', test_wA[1], res_right)

        return self._assemble_residual_contributions(contrib_left, contrib_right)

    def get_residual_vec(self) -> NDArray:
        """Assemble full residual vector from all terms (res_size,)."""
        res_vec = np.zeros(self.res_size)
        for term in self.terms:
            sl = self._res_slice(term.res)
            res_vec[sl] += self.residual_vector_term(term)
        return res_vec

    # =========================================================================
    # System Assembly Entry Points
    # =========================================================================

    def _res_slice(self, res_name: str) -> slice:
        """Slice for residual block in local arrays."""
        i = self.residuals.index(res_name)
        return slice(i * self.nb_inner_pts, (i + 1) * self.nb_inner_pts)

    def get_M(self) -> NDArray:
        """Get tangent matrix (COO values for sparse assembly)."""
        return self.get_tang_matrix_sparse()

    def get_M_dense(self) -> NDArray:
        """Get tangent matrix as dense array (for testing/debugging)."""
        coo_values = self.get_tang_matrix_sparse()
        rows = self.assembly.local_rows
        cols = self.assembly.local_cols

        M = np.zeros((self.res_size, self.var_size), dtype=np.float64)
        np.add.at(M, (rows, cols), coo_values)
        return M

    def get_R(self) -> NDArray:
        """Get residual vector."""
        return self.get_residual_vec()

    # =========================================================================
    # Solution Vector Management
    # =========================================================================

    def _sol_slice(self, var_name: str) -> slice:
        """Slice for solution vector (uses nb_inner_pts, not nb_contributors)."""
        i = self.variables.index(var_name)
        return slice(i * self.nb_inner_pts, (i + 1) * self.nb_inner_pts)

    def get_nodal_val(self, field_name: str) -> NDArray:
        """Returns the inner nodal values of a field in shape (nb_inner_pts,)."""
        return self.field_map[field_name][1:-1, 1:-1].flatten(order='F')

    def get_q_nodal(self) -> NDArray:
        """Returns the full solution vector q in nodal values shape (nb_vars*nb_inner_pts,)."""
        q_nodal = np.zeros(self.res_size)
        for var in self.variables:
            q_nodal[self._sol_slice(var)] = self.get_nodal_val(var)
        return q_nodal

    def set_q_nodal(self, q_nodal: NDArray) -> None:
        """Sets the full solution vector q from nodal values shape (nb_vars*nb_inner_pts,).
        """
        for var in self.variables:
            var_nodal = q_nodal[self._sol_slice(var)]
            self.field_map[var][1:-1, 1:-1] = var_nodal.reshape(
                (self.Nx_inner, self.Ny_inner), order='F')

    # =========================================================================
    # Solver Interface
    # =========================================================================

    def solver_step_fun(self, q_guess: NDArray) -> Tuple[NDArray, NDArray]:
        """Newton solver step: set guess, update fields, return (M, R)."""
        self.set_q_nodal(q_guess)
        with self.timer("update_quad"):
            self.update_quad()
        with self.timer("jacobian"):
            M = self.get_M()
        with self.timer("residual"):
            R = self.get_R()
        return M, R

    def update_output_fields(self) -> None:
        """Update nodal output fields (wall stress, bulk stress) for plotting/output."""
        p = self.problem
        p.wall_stress_xz.update()
        p.wall_stress_yz.update()
        if hasattr(p, 'bulk_stress'):
            p.bulk_stress.update()

    def update_dynamic(self) -> None:
        """Do a single dynamic time step update using PETSc."""
        p = self.problem
        fem_solver = p.fem_solver

        with self.timer("timestep"):
            self.update_prev_quad()

            tic = time.time()

            q = self.get_q_nodal().copy()
            max_iter = fem_solver['max_iter']
            tol = fem_solver['R_norm_tol']
            alpha = fem_solver['newton_relax']

            # Start new timestep history (rank 0 only)
            if p.decomp.rank == 0:
                self.R_norm_history.append([])

            for it in range(max_iter):
                with self.timer("newton_iteration"):
                    M, R = self.solver_step_fun(q)
                    # Compute global residual norm via MPI allreduce
                    R_norm_local_sq = np.linalg.norm(R)**2
                    R_norm_global_sq = p.decomp._mpi_comm.allreduce(R_norm_local_sq, op=MPI.SUM)
                    R_norm = np.sqrt(R_norm_global_sq)

                    # Track residual history (rank 0 only)
                    if p.decomp.rank == 0:
                        self.R_norm_history[-1].append(R_norm)

                    if (R_norm < tol) and it > 0:
                        break

                    # Scale system for better conditioning
                    M_scaled, R_scaled = self.scaling.scale_system(M, R)

                    # Assemble and solve
                    with self.timer("petsc_assemble"):
                        self.linear_solver.assemble(M_scaled, R_scaled)
                    with self.timer("petsc_solve"):
                        dq_scaled = self.linear_solver.solve(
                            self.nb_inner_pts, len(self.variables))

                    # Unscale solution
                    dq = self.scaling.unscale_solution(dq_scaled)

                    q = apply_guards(q, alpha * dq, self)

                    # Update solver state
                    self.set_q_nodal(q)
                    p.decomp.communicate_ghost_buffers(p)

            toc = time.time()
            self.time_inner = toc - tic
            self.inner_iterations = it + 1  # Store number of iterations

            self.update_output_fields()

        p._post_update()

    def update(self) -> None:
        """Top-level solver update function."""
        self.update_dynamic()

    def print_status_header(self) -> None:
        """Print header for simulation status output."""
        p = self.problem
        if p.options['print_progress'] and p.decomp.rank == 0:
            print(75 * '-')
            print(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} {'Iter':<6s} {'Conv. Time':<12s} {'Residual':<12s}")
            print(75 * '-')
        if p.options['save_output']:
            p.write(params=False)

    def print_status(self, scalars=None) -> None:
        """Print status line for simulation."""
        p = self.problem
        if scalars and p.options['print_progress'] and p.decomp.rank == 0:
            print(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} "
                  f"{self.inner_iterations:<6d} {self.time_inner:<12.4e} {p.residual:<12.4e}")

    def print_timer_summary(self, save_json: str = "") -> None:
        """Print timer summary and optionally save to JSON.

        Parameters
        ----------
        save_json : str, optional
            Path to save JSON output. If empty, no file is written.
        """
        if self.problem.decomp.rank == 0:
            self.timer.print_summary()
            if save_json:
                with open(save_json, 'w') as f:
                    f.write(self.timer.to_json())

    def plot_residual_history(self, max_separators: int = 50):
        """Plot R_norm evolution across all Newton iterations.

        Returns (fig, ax) or (None, None) if not on rank 0.
        """
        if self.problem.decomp.rank != 0:
            return None, None
        return plot_residual_history(self.R_norm_history, max_separators)

    def run_residual_analysis(self) -> None:
        """
        Run residual analysis and generate output if enabled.

        Called from _post_run when residual_analysis option is True.
        Requires solver to be initialized with updated quad fields.
        """
        import os
        p = self.problem

        # Ensure quad fields are up-to-date
        self.update_quad()

        # Print text summary
        print_residual_analysis(self)

        # Generate plot if output directory exists
        if hasattr(p, 'outdir'):
            plot_path = os.path.join(p.outdir, 'residual_analysis.png')
            create_residual_analysis_plot(self, plot_path)

    def pre_run(self, **kwargs) -> None:
        """Initialize solver before running."""

        with self.timer("preparation"):
            self._init_convenience_accessors()
            self._init_quad_fields()
            self._get_active_terms()
            self._build_assembly()
            self._build_jit_functions()
            self._build_terms()
            with self.timer("init_petsc"):
                self._init_linear_solver()

            # Initial quad update
            self.update_quad()
            self.update_prev_quad()

            # Update output fields for initial frame
            self.update_output_fields()

        self.time_inner = 0.0
        self.inner_iterations = 0

    def _init_linear_solver(self):
        """Initialize linear solver and scaling for sparse system solves."""
        solver_type = self.problem.fem_solver['linear_solver']
        nb_global_pts = np.prod(self.problem.decomp.nb_domain_grid_pts)
        petsc_info = self.assembly.get_petsc_info(
            nb_vars=len(self.variables),
            nb_inner_pts=self.nb_inner_pts,
            nb_global_pts=nb_global_pts,
        )

        if HAS_PETSC:
            from .fem_2d.petsc_system import PETScSystem
            self.linear_solver = PETScSystem(petsc_info, solver_type=solver_type)
        else:
            from mpi4py import MPI
            if MPI.COMM_WORLD.size > 1:
                raise RuntimeError(
                    "PETSc is required for parallel execution. "
                    "Install petsc4py or run in serial mode."
                )
            from .fem_2d.scipy_system import ScipySystem
            self.linear_solver = ScipySystem(petsc_info, solver_type=solver_type)
            print("Note: Using SciPy sparse solver (PETSc not available). "
                  "Install petsc4py for better performance and parallel support.")

        # Build scaling for linear system conditioning
        self.scaling = build_scaling(
            self.problem, self.energy, self.variables, self.assembly)
