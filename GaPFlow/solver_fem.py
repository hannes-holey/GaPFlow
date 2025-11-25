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
from .fem.num_solver import Solver
from .fem.utils import (
    NonLinearTerm,
    create_quad_fields,
    get_active_terms,
    get_norm_quad_pts,
    get_norm_quad_wts,
    print_matrix
)

from muGrid import GlobalFieldCollection
import numpy as np
import time

import numpy.typing as npt
from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from .problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver1D:

    def __init__(self, problem: "Problem") -> None:
        self.problem = problem
        self.num_solver = Solver(problem.fem_solver)

    def _build_terms(self) -> None:
        """Linking functions of initialized models into abstract term functions"""
        p = self.problem

        # build model gradient functions
        p.pressure.build_grad()
        p.wall_stress_xz.build_grad()
        # p.energy.build_grad()

        # build terms with context
        for term in self.terms:
            term_ctx = {}
            term_ctx['p'] = lambda nbq=term.nb_quad_pts: p.pressure.p_quad(nbq)
            term_ctx['dp_drho'] = lambda nbq=term.nb_quad_pts: p.pressure.dp_drho_quad(nbq)
            term_ctx['tau_xz'] = lambda nbq=term.nb_quad_pts: p.wall_stress_xz.tau_xz_quad(nbq)
            term_ctx['dtau_xz_drho'] = lambda nbq=term.nb_quad_pts: p.wall_stress_xz.dtau_xz_drho_quad(nbq)
            term_ctx['dtau_xz_djx'] = lambda nbq=term.nb_quad_pts: p.wall_stress_xz.dtau_xz_djx_quad(nbq)
            term_ctx['tau_xz_bot'] = lambda nbq=term.nb_quad_pts: p.wall_stress_xz.tau_xz_bot_quad(nbq)
            term_ctx['dtau_xz_bot_drho'] = lambda nbq=term.nb_quad_pts: p.wall_stress_xz.dtau_xz_bot_drho_quad(nbq)
            term_ctx['dtau_xz_bot_djx'] = lambda nbq=term.nb_quad_pts: p.wall_stress_xz.dtau_xz_bot_djx_quad(nbq)
            term_ctx['T'] = lambda nbq=term.nb_quad_pts: p.energy.T_quad(nbq)
            term_ctx['dT_drho'] = lambda nbq=term.nb_quad_pts: p.energy.dT_drho_quad(nbq)
            term_ctx['dT_djx'] = lambda nbq=term.nb_quad_pts: p.energy.dT_djx_quad(nbq)
            term_ctx['dT_dE'] = lambda nbq=term.nb_quad_pts: p.energy.dT_dE_quad(nbq)
            term_ctx['S'] = lambda nbq=term.nb_quad_pts: p.energy.S_quad(nbq)
            term_ctx['dS_drho'] = lambda nbq=term.nb_quad_pts: p.energy.dS_drho_quad(nbq)
            term_ctx['dS_djx'] = lambda nbq=term.nb_quad_pts: p.energy.dS_djx_quad(nbq)
            term_ctx['dS_dE'] = lambda nbq=term.nb_quad_pts: p.energy.dS_dE_quad(nbq)
            term_ctx['h'] = lambda nbq=term.nb_quad_pts: p.topo.h_quad(nbq)
            term_ctx['dh_dx'] = lambda nbq=term.nb_quad_pts: p.topo.dh_dx_quad(nbq)
            term_ctx['U'] = lambda nbq=term.nb_quad_pts: p.topo.U_quad(nbq)
            term_ctx['rho_prev'] = lambda nbq=term.nb_quad_pts: self.get_quad_field('rho_prev', nbq)
            term_ctx['jx_prev'] = lambda nbq=term.nb_quad_pts: self.get_quad_field('jx_prev', nbq)
            term_ctx['E_prev'] = lambda nbq=term.nb_quad_pts: self.get_quad_field('E_prev', nbq)
            term_ctx['dt'] = p.numerics['dt']
            term.build(term_ctx)

    def _init_convenience_accessors(self) -> None:
        """Initialize convenience accessors for problem and grid properties."""
        p = self.problem

        self.periodic = p.grid['bc_xE_P'][0]  # periodic in x
        self.energy = p.fem_solver['equations']['energy']
        self.dynamic = p.fem_solver['dynamic']

        self.nb_pts = p.grid['Nx']
        self.nb_ele = self.nb_pts if self.periodic else self.nb_pts - 1
        self.dx = p.grid['Lx'] / self.nb_ele

        if self.energy:
            # equation system
            self.variables = ['rho', 'jx', 'E']
            self.residuals = ['mass', 'momentum_x', 'energy']
            # quadrature point fields
            self.field_list = ['rho', 'jx', 'jy', 'E']
            self.field_val_list = [p.q[0], p.q[1], p.q[2], p.energy.energy]
            self.prev_field_list = ['rho_prev', 'jx_prev', 'E_prev']
        else:
            # equation system
            self.variables = ['rho', 'jx']
            self.residuals = ['mass', 'momentum_x']
            # quadrature point fields
            self.field_list = ['rho', 'jx', 'jy']
            self.field_val_list = [p.q[0], p.q[1], p.q[2]]
            self.prev_field_list = ['rho_prev', 'jx_prev']

        self.res_size = len(self.residuals) * self.nb_pts
        self.mat_size = (self.res_size, self.res_size)

    def _get_active_terms(self) -> None:
        """Initialize list of active terms from problem fem_solver config."""
        self.terms = get_active_terms(self.problem.fem_solver)

    def _get_quad_list(self, **kwargs) -> None:
        """Get list of occuring quadrature point numbers from active terms."""
        if 'enforced_quad_list' in kwargs:
            self.quad_list = kwargs['enforced_quad_list']
            return
        nb_quad_pts_set = set()
        for term in self.terms:
            nb_quad_pts_set.add(term.nb_quad_pts)
        self.quad_list = sorted(list(nb_quad_pts_set))

    def _init_quad_fun(self) -> None:
        """Init quadrature point evaluation function."""

        def quad_fun(var: NDArray, nb_quad_pts: int) -> NDArray:
            """Returns quadrature point values in 1d array (nb_ele*nb_quad_pts,)"""
            vals = np.append(var, var[0]) if self.periodic else var
            if vals.ndim != 1:
                raise ValueError(f"quad_fun expects a 1D array, got shape {vals.shape}")
            xi = get_norm_quad_pts(nb_quad_pts)
            i = np.arange(self.nb_ele)[:, None]
            x_quad = i + xi[None, :]
            return np.interp(x_quad.ravel(), np.arange(len(vals)), vals)

        self.quad_fun = quad_fun

    def _init_dx_fun(self) -> None:
        """Init quadrature point derivative evaluation function."""

        def dx_fun(var: NDArray, nb_quad_pts: int) -> NDArray:
            vals = np.append(var, var[0]) if self.periodic else var
            diff = np.diff(vals) / self.dx
            return np.repeat(diff, nb_quad_pts)

        self.dx_fun = dx_fun

    def _init_fc_fem(self) -> None:
        self.fc_fem = GlobalFieldCollection((self.nb_ele, ))

    def _res_slice(self, res_name) -> slice:
        i = self.residuals.index(res_name)
        return slice(i * self.nb_pts, (i + 1) * self.nb_pts)

    def _var_slice(self, var_name) -> slice:
        i = self.variables.index(var_name)
        return slice(i * self.nb_pts, (i + 1) * self.nb_pts)

    def _block_slices(self, res_name, var_name) -> tuple[slice, slice]:
        return (self._res_slice(res_name), self._var_slice(var_name))

    def get_N_quad(self, nb_quad_pts: int) -> tuple[NDArray, NDArray]:
        xi = get_norm_quad_pts(nb_quad_pts)
        N1 = 1 - xi
        N2 = xi
        return N1, N2

    def get_N_quad_w(self, nb_quad_pts: int) -> tuple[NDArray, NDArray]:
        """These N-function quad values already include the quadrature weights."""
        xi = get_norm_quad_pts(nb_quad_pts)
        wi = get_norm_quad_wts(nb_quad_pts)
        N1 = (1 - xi) * wi
        N2 = xi * wi
        return N1, N2

    def get_dN_dx_quad_w(self, nb_quad_pts: int) -> tuple[NDArray, NDArray]:
        """These dN/dx-function quad values already include the quadrature weights."""
        wi = get_norm_quad_wts(nb_quad_pts)
        dN1_dx = -1.0 * wi
        dN2_dx = 1.0 * wi
        return dN1_dx, dN2_dx

    def _get_quad_deriv(self, vec: NDArray, nb_quad_pts: int) -> NDArray:
        """Takes 1D nodal values and returns quadrature derivative values."""
        vals = vec
        if self.periodic:
            vals = np.append(vals, vals[0])
        diff = np.diff(vals) / self.dx
        return np.repeat(diff[None, :], nb_quad_pts, axis=0)

    def tang_matrix_term(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        M = np.zeros((self.nb_pts, self.nb_pts))

        if not term.d_dx_testfun:
            N1_quad_w, N2_quad_w = self.get_N_quad_w(term.nb_quad_pts)
            N1_quad, N2_quad = self.get_N_quad(term.nb_quad_pts)
        else:
            N1_quad_w, N2_quad_w = self.get_dN_dx_quad_w(term.nb_quad_pts)  # scaled later by dx_ele

        dep_var_vals = {dep_var: self.get_quad_field(dep_var, term.nb_quad_pts)
                        for dep_var in term.dep_vars}
        res_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
        res_deriv_vals = res_deriv_vals.T  # get shape (nb_ele, nb_quad_pts)

        if not term.d_dx_resfun:  # res = f(a)
            for pt_test in range(self.nb_pts):

                # apply wrap-around by default, sort out non-periodic later
                pt_mid = pt_test % self.nb_pts
                pt_left = (pt_test - 1) % self.nb_pts
                pt_right = (pt_test + 1) % self.nb_pts

                for rel_element in [-1, 0]:
                    element = pt_test + rel_element

                    if not self.periodic and (element < 0 or element == self.nb_ele):
                        continue

                    if term.d_dx_testfun:
                        raise ValueError("d_dx_testfun=True not compatible with d_dx_resfun=False")

                    res_deriv_local = res_deriv_vals[element % self.nb_ele, :]
                    dx_ele = self.dx

                    if rel_element == -1:  # left side
                        M[pt_mid, pt_mid] += np.sum(N2_quad_w * res_deriv_local * N2_quad) * dx_ele
                        M[pt_mid, pt_left] += np.sum(N2_quad_w * res_deriv_local * N1_quad) * dx_ele
                    else:  # right side
                        M[pt_mid, pt_mid] += np.sum(N1_quad_w * res_deriv_local * N1_quad) * dx_ele
                        M[pt_mid, pt_right] += np.sum(N1_quad_w * res_deriv_local * N2_quad) * dx_ele

        if term.d_dx_resfun:  # res = df/da * da/dx
            for pt_test in range(self.nb_pts):

                # apply wrap-around by default, sort out non-periodic later
                pt_mid = pt_test % self.nb_pts
                pt_left = (pt_test - 1) % self.nb_pts
                pt_right = (pt_test + 1) % self.nb_pts

                for rel_element in [-1, 0]:
                    element = pt_test + rel_element

                    # prevent wrap-around for non-periodic
                    if not self.periodic and (element < 0 or element == self.nb_ele):
                        continue

                    res_deriv_local = res_deriv_vals[element % self.nb_ele, :]

                    if term.d_dx_testfun:
                        N1_quad_w_scaled = np.copy(N1_quad_w) / self.dx
                        N2_quad_w_scaled = np.copy(N2_quad_w) / self.dx
                    else:
                        N1_quad_w_scaled = N1_quad_w
                        N2_quad_w_scaled = N2_quad_w

                    if rel_element == -1:  # left side
                        M[pt_mid, pt_mid] += np.sum(N2_quad_w_scaled * res_deriv_local * 1.0)
                        M[pt_mid, pt_left] += np.sum(N2_quad_w_scaled * res_deriv_local * -1.0)
                    else:  # right side
                        M[pt_mid, pt_mid] += np.sum(N1_quad_w_scaled * res_deriv_local * -1.0)
                        M[pt_mid, pt_right] += np.sum(N1_quad_w_scaled * res_deriv_local * 1.0)
        return M

    def residual_vector_term(self, term: NonLinearTerm) -> NDArray:
        res_vec = np.zeros(self.nb_pts)

        dep_var_vals = {dep_var: self.get_quad_field(dep_var, term.nb_quad_pts)
                        for dep_var in term.dep_vars}

        if not term.d_dx_resfun:  # res = f(a)
            res_fun_vals = term.evaluate(*[dep_var_vals[dep_var] for dep_var in term.dep_vars])
            res_fun_vals = res_fun_vals.T  # get shape (nb_ele, nb_quad_pts)
        else:  # res = df/da * da/dx
            res_fun_vals = np.zeros((self.nb_ele, term.nb_quad_pts))
            for dep_var in term.dep_vars:
                fun_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
                dep_var_deriv_vals = self.get_quad_deriv(dep_var, term.nb_quad_pts)
                res_fun_vals += (fun_deriv_vals * dep_var_deriv_vals).T  # get shape (nb_ele, nb_quad_pts)

        if not term.d_dx_testfun:
            N1_quad_w, N2_quad_w = self.get_N_quad_w(term.nb_quad_pts)
        else:
            N1_quad_w, N2_quad_w = self.get_dN_dx_quad_w(term.nb_quad_pts)  # scaled later by dx_ele

        assert res_fun_vals.shape == (self.nb_ele, term.nb_quad_pts)

        # for each test function, get left and right element integral
        for pt_test in range(self.nb_pts):

            for rel_element in [-1, 0]:
                element = pt_test + rel_element
                if not self.periodic and (element < 0 or element == self.nb_ele):
                    continue
                test_fun_val = N2_quad_w if rel_element == -1 else N1_quad_w

                if term.d_dx_testfun:
                    test_fun_val_scaled = np.copy(test_fun_val) / self.dx
                else:
                    test_fun_val_scaled = test_fun_val

                res_fun_local = res_fun_vals[element % self.nb_ele, :]
                res = np.sum(test_fun_val_scaled * res_fun_local) * self.dx
                res_vec[pt_test] += res
        return res_vec

    def get_tang_matrix(self) -> NDArray:
        tang_matrix = np.zeros(self.mat_size)
        for term in self.terms:
            if (not self.dynamic) and 'T' in term.name:
                continue
            for dep_var in term.dep_vars:
                bl = self._block_slices(term.res, dep_var)
                tang_matrix[bl] += self.tang_matrix_term(term, dep_var)
        return tang_matrix

    def get_residual_vec(self) -> NDArray:
        res_vec = np.zeros(self.res_size)
        for term in self.terms:
            if (not self.dynamic) and 'T' in term.name:
                continue
            sl = self._res_slice(term.res)
            res_vec[sl] += self.residual_vector_term(term)
        return res_vec

    def boundary_condition_M(self, M: NDArray) -> NDArray:
        p = self.problem

        if p.grid['bc_xW_D'][0]:
            M[0, :] = 0.0
            M[0, 0] = 1.0
        if p.grid['bc_xE_D'][0]:
            M[-1, :] = 0.0
            M[-1, self.nb_pts - 1] = 1.0
        return M

    def boundary_condition_R(self, R: NDArray) -> NDArray:
        p = self.problem

        # check Dirichlet
        if p.grid['bc_xW_D'][0]:
            target = p.grid['bc_xW_D_val']
            guess = self.get_nodal_val('rho')[0]
            R[0] = guess - target
        if p.grid['bc_xE_D'][0]:
            target = p.grid['bc_xE_D_val']
            guess = self.get_nodal_val('rho')[-1]
            R[-1] = guess - target
        return R

    def get_M(self) -> NDArray:
        M = self.get_tang_matrix()
        M = self.boundary_condition_M(M)
        return M

    def get_R(self) -> NDArray:
        R = self.get_residual_vec()
        R = self.boundary_condition_R(R)
        return R

    def _inner_1d(self, field: NDArray) -> NDArray:
        """Extract inner 1D field from 2D field array with ghost cells."""
        assert field.shape[1] == 3, "Not a 1D problem: {}".format(field.shape)
        return field[1:-1, 1:-1].ravel()

    def init_quad(self) -> None:
        """Initialize quadrature fields for 'q' and 'q_prev'"""
        create_quad_fields(self, self.fc_fem, self.field_list, self.quad_list)
        create_quad_fields(self, self.fc_fem, self.prev_field_list, self.quad_list)

    def get_nodal_val(self, field_name: str) -> NDArray:
        """Returns the nodal values of a field in shape (nb_pts,)."""
        var_idx = self.field_list.index(field_name)
        return self._inner_1d(self.field_val_list[var_idx])

    def get_q_nodal(self) -> NDArray:
        """Returns the full solution vector q in nodal values shape (nb_vars*nb_pts,)."""
        q_nodal = np.zeros(self.res_size)
        for var in self.variables:
            var_slice = self._var_slice(var)
            q_nodal[var_slice] = self.get_nodal_val(var)
        assert q_nodal.shape == (self.res_size, )
        return q_nodal

    def set_q_nodal(self, q_nodal: NDArray) -> None:
        """Sets the full solution vector q from nodal values shape (nb_vars*nb_pts,)."""
        for var in self.variables:
            var_slice = self._var_slice(var)
            var_nodal = q_nodal[var_slice]
            var_idx = self.field_list.index(var)
            self.field_val_list[var_idx][1:-1, 1:-1] = var_nodal.reshape((self.nb_pts, 1))

    def update_sol_quad(self) -> None:
        """Update all quadrature fields from 'q' nodal solution."""
        for nb_quad in self.quad_list:
            for field, val in zip(self.field_list, self.field_val_list):
                fieldname = f'_{field}_quad_{nb_quad}'
                quad_vals = self.quad_fun(self._inner_1d(val), nb_quad)
                getattr(self, fieldname).p = quad_vals.reshape(-1, nb_quad).T
                # reshape(-1, nb_quad) gives shape (nb_ele, nb_quad)
                # transpose to (nb_quad, nb_ele) for muGrid storage

    def get_quad_field(self, field_name: str, nb_quad: int) -> NDArray:
        """Returns the quadrature values of a field in shape (nb_quad, nb_ele)."""
        fieldname = f'_{field_name}_quad_{nb_quad}'
        return getattr(self, fieldname).p

    def get_quad_deriv(self, field_name: str, nb_quad: int) -> NDArray:
        """Returns the quadrature derivative values of a field in shape (nb_quad, nb_ele)."""
        vec = self.get_nodal_val(field_name)
        return self._get_quad_deriv(vec, nb_quad)

    def update_quad(self) -> None:
        """Update all quadrature fields from current nodal solution."""
        p = self.problem
        self.update_sol_quad()
        p.topo.update_quad(self.quad_fun, self.dx_fun, self._inner_1d)
        p.pressure.update_quad(self.quad_fun, self._inner_1d, self.get_quad_field)
        p.wall_stress_xz.update_quad(self.quad_fun, self._inner_1d, self.get_quad_field, p.topo)
        if self.energy:
            p.energy.update_quad(self.quad_fun, self._inner_1d, self.get_quad_field)

    def update_prev_quad(self) -> None:
        """Write current quad values into 'previous' quad fields."""
        for nb_quad in self.quad_list:
            for field in self.variables:
                curr_fieldname = f'_{field}_quad_{nb_quad}'
                prev_fieldname = f'_{field}_prev_quad_{nb_quad}'
                curr_vals = getattr(self, curr_fieldname).p
                getattr(self, prev_fieldname).p = np.copy(curr_vals)

    def pre_run(self, **kwargs) -> None:
        p = self.problem

        p.dt = p.numerics['dt']
        p.tol = p.numerics['tol']
        p.max_it = p.numerics['max_it']

        self._get_active_terms()
        self._get_quad_list(**kwargs)
        self._init_convenience_accessors()
        self._init_quad_fun()
        self._init_dx_fun()
        self._init_fc_fem()

        print("FEM Solver 1D intialized with the following settings:")
        print(f"  Periodic boundary conditions in x-direction: {self.periodic}")
        print(f"  Energy equation included: {self.energy}")
        print(f"  Dynamic solver enabled: {self.dynamic} \n")

        self.init_quad()
        p.topo.init_quad(self.fc_fem, p.geo, self.quad_list)
        p.pressure.init_quad(self.fc_fem, self.quad_list, create_quad_fields)
        p.pressure.build_grad()
        p.wall_stress_xz.init_quad(self.fc_fem, self.quad_list)
        p.wall_stress_xz.build_grad()
        if self.energy:
            p.energy.init_quad(self.fc_fem, self.quad_list, create_quad_fields)
            p.energy.build_grad()

        self._build_terms()
        self.update_quad()

        if self.dynamic:
            self.update_prev_quad()
        self.time_inner = 0.0

    def solver_step_fun(self, q_guess: NDArray) -> Tuple[NDArray, NDArray]:
        self.set_q_nodal(q_guess)
        self.update_quad()
        M = self.get_M()
        R = self.get_R()
        return M, R

    def print_system(self) -> None:
        assert self.nb_pts < 15, "System too large to print."
        M = self.get_M()
        R = self.get_R()
        print_matrix(M)
        print(R)

    def steady_state(self) -> None:
        p = self.problem

        print(61 * '-')
        print(f"{'Iteration':<10s} {'Residual':<12s}")
        print(61 * '-')

        self.num_solver.sol_dict.q0 = self.get_q_nodal()
        self.num_solver.get_MR_fun = self.solver_step_fun

        tic = time.time()
        sol_dict = self.num_solver.solve()
        toc = time.time()
        if sol_dict.success:
            print(f"Converged. Solving took {toc - tic:.2f} seconds.")

        p._stop = True

    def update_dynamic(self) -> None:
        """Do a single dynamic time step update, then return to problem main loop.
        R_norm_tol and fem_solver['max_iter'] apply to inner convergence loop.
        """
        p = self.problem
        self.update_prev_quad()

        self.num_solver.sol_dict.reset()
        self.num_solver.sol_dict.q0 = self.get_q_nodal()
        self.num_solver.get_MR_fun = self.solver_step_fun

        tic = time.time()
        self.num_solver.solve(silent=True)
        toc = time.time()
        self.time_inner = toc - tic

        p.post_update()

    def update(self) -> None:
        """Top-level solver update function"""
        if self.dynamic:
            self.update_dynamic()
        else:
            self.steady_state()

    def print_status_header(self) -> None:
        p = self.problem
        if not p.options['silent'] and self.dynamic:
            print(61 * '-')
            print(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} {'Convergence Time':<18s} {'Residual':<12s}")
            print(61 * '-')
            p.write(params=False)

    def print_status(self, scalars=None) -> None:
        p = self.problem
        if not p.options['silent'] and self.dynamic:
            print(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} {self.time_inner:<18.4e} {p.residual:<12.4e}")
