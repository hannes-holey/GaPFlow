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
from .fem.utils import NonLinearTerm, get_active_terms, get_norm_quad_pts, get_norm_quad_wts, create_quad_fields

from muGrid import GlobalFieldCollection
import numpy as np

import numpy.typing as npt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver1D:

    def __init__(self, problem: "Problem") -> None:
        self.problem = problem

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
            term_ctx['h'] = lambda nbq=term.nb_quad_pts: p.topo.h_quad(nbq)
            term_ctx['dh_dx'] = lambda nbq=term.nb_quad_pts: p.topo.dh_dx_quad(nbq)
            term_ctx['U'] = lambda nbq=term.nb_quad_pts: self.h_wrapper.quad_U(nbq)
            term_ctx['rho_prev'] = lambda nbq=term.nb_quad_pts: self.get_quad_vals(self.a_prev, 'rho', nbq)
            term_ctx['jx_prev'] = lambda nbq=term.nb_quad_pts: self.get_quad_vals(self.a_prev, 'jx', nbq)
            term_ctx['E_prev'] = lambda nbq=term.nb_quad_pts: self.get_quad_vals(self.a_prev, 'E', nbq)
            term.build(term_ctx)

    def _init_convenience_accessors(self) -> None:
        """Initialize convenience accessors for problem and grid properties."""
        p = self.problem

        self.periodic = p.grid['bc_xE_P'][0]  # periodic in x
        self.nb_pts = p.grid['Nx']
        self.nb_ele = self.nb_pts if self.periodic else self.nb_pts - 1

        if p.fem_solver['equations']['energy']:
            self.variables = ['rho', 'jx', 'E']
            self.residuals = ['mass', 'momentum_x', 'energy']
        else:
            self.variables = ['rho', 'jx']
            self.residuals = ['mass', 'momentum_x']

        self.res_size = len(self.residuals) * self.nb_pts
        self.mat_size = (self.res_size, self.res_size)

    def _get_active_terms(self) -> None:
        """Initialize list of active terms from problem fem_solver config."""
        self.terms = get_active_terms(self.problem.fem_solver)

    def _get_quad_list(self) -> None:
        """Get list of occuring quadrature point numbers from active terms."""
        nb_quad_pts_set = set()
        for term in self.terms:
            nb_quad_pts_set.add(term.nb_quad_pts)
        self.quad_list = sorted(list(nb_quad_pts_set))

    def _init_quad_fun(self) -> None:
        """Init quadrature point evaluation function."""

        def quad_fun(var: NDArray, nb_quad_pts: int) -> NDArray:
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
            diff = np.diff(vals) / self.problem.grid['dx']
            return np.repeat(diff, nb_quad_pts)

        self.dx_fun = dx_fun

    def _init_fc_fem(self) -> None:
        self.fc_fem = GlobalFieldCollection((self.nb_ele, ))

    def _res_slice(self, res_name) -> slice:
        i = self.residuals.index(res_name)
        return slice(i * self.config.N, (i + 1) * self.config.N)

    def _var_slice(self, var_name) -> slice:
        i = self.variables.index(var_name)
        return slice(i * self.config.N, (i + 1) * self.config.N)

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

    def get_quad_vals(self, a: NDArray, var_name: str, nb_quad_pts: int) -> NDArray:
        vals = a[self._var_slice(var_name)]
        if self.periodic:
            vals = np.append(vals, vals[0])  # for periodicity, nb_ele is already increased
        xi = get_norm_quad_pts(nb_quad_pts)
        i = np.arange(self.nb_ele)[:, None]
        x_quad = i + xi[None, :]
        return np.interp(x_quad.ravel(), np.arange(len(vals)), vals)

    def get_quad_deriv(self, a: NDArray, var_name: str, nb_quad_pts: int) -> NDArray:
        vals = a[self._var_slice(var_name)]
        if self.periodic:
            vals = np.append(vals, vals[0])
        diff = np.diff(vals) / self.config.dx
        return np.repeat(diff, nb_quad_pts)

    def tang_matrix_term(self, term: NonLinearTerm, dep_var: str, a: NDArray) -> NDArray:
        M = np.zeros((self.nb_pts, self.nb_pts))

        if not term.d_dx_testfun:
            N1_quad_w, N2_quad_w = self.get_N_quad_w(term.nb_quad_pts)
            N1_quad, N2_quad = self.get_N_quad(term.nb_quad_pts)
        else:
            N1_quad_w, N2_quad_w = self.get_dN_dx_quad_w(term.nb_quad_pts)  # scaled later by dx_ele

        dep_var_vals = {dep_var: self.get_quad_vals(a, dep_var, term.nb_quad_pts)
                        for dep_var in term.dep_vars}
        res_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
        res_deriv_vals = res_deriv_vals.reshape(self.nb_ele, term.nb_quad_pts)

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
                    dx_ele = self.config.dx_ele(element % self.nb_ele)

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
                        N1_quad_w_scaled = np.copy(N1_quad_w) / (self.config.dx_ele(element % self.nb_ele))
                        N2_quad_w_scaled = np.copy(N2_quad_w) / (self.config.dx_ele(element % self.nb_ele))
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

    def residual_vector_term(self, term: NonLinearTerm, a: NDArray) -> NDArray:
        res_vec = np.zeros(self.nb_pts)

        if not term.d_dx_resfun:  # res = f(a)
            dep_var_vals = {dep_var: self.get_quad_vals(a, dep_var, term.nb_quad_pts)
                            for dep_var in term.dep_vars}
            res_fun_vals = term.evaluate(*[dep_var_vals[dep_var] for dep_var in term.dep_vars])
            res_fun_vals = res_fun_vals.reshape(self.nb_ele, term.nb_quad_pts)

        if term.d_dx_resfun:  # res = df/da * da/dx
            dep_var_vals = {dep_var: self.get_quad_vals(a, dep_var, term.nb_quad_pts)
                            for dep_var in term.dep_vars}
            res_fun_vals = np.zeros((self.nb_ele, term.nb_quad_pts))
            for dep_var in term.dep_vars:
                fun_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
                dep_var_deriv_vals = self.get_quad_deriv(a, dep_var, term.nb_quad_pts)
                res_fun_vals += (fun_deriv_vals * dep_var_deriv_vals).reshape(self.nb_ele, term.nb_quad_pts)

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
                    test_fun_val_scaled = np.copy(test_fun_val) / (self.config.dx_ele(element % self.nb_ele))
                else:
                    test_fun_val_scaled = test_fun_val

                res_fun_local = res_fun_vals[element % self.nb_ele, :]
                res = np.sum(test_fun_val_scaled * res_fun_local) * self.config.dx_ele(element % self.nb_ele)
                res_vec[pt_test] += res
        return res_vec

    def get_tang_matrix(self, a) -> NDArray:
        tang_matrix = np.zeros(self.mat_size)
        for term in self.terms:
            if (not self.dynamic) and 'T' in term.name:
                continue
            for dep_var in term.dep_vars:
                bl = self._block_slices(term.res, dep_var)
                tang_matrix[bl] += self.tang_matrix_term(term, dep_var, a)
        return tang_matrix

    def get_residual_vec(self, a) -> NDArray:
        res_vec = np.zeros(self.res_size)
        for term in self.terms:
            if (not self.dynamic) and 'T' in term.name:
                continue
            sl = self._res_slice(term.res)
            res_vec[sl] += self.residual_vector_term(term, a)
        return res_vec

    def boundary_conditions(self, M, R, a) -> tuple[NDArray, NDArray]:
        pass
        return M, R

    def get_MR(self, sol) -> tuple[NDArray, NDArray]:
        a = sol.a
        M = self.get_tang_matrix(a)
        R = self.get_residual_vec(a)
        M, R = self.boundary_conditions(M, R, a)
        return M, R

    def get_M(self, a) -> NDArray:
        pass

    def get_R(self, a) -> NDArray:
        pass

    def pre_run(self) -> None:
        p = self.problem

        self.num_solver = Solver(p.fem_solver)

        self._get_active_terms()
        self._get_quad_list()
        self._build_terms()
        self._init_convenience_accessors()
        self._init_quad_fun()
        self._init_dx_fun()
        self._init_fc_fem()

        self.init_quad()
        p.topo.init_quad(self.fc_fem, p.geo, self.quad_list)
        p.pressure.init_quad(self.fc_fem, self.quad_list)
        p.pressure.build_grad()
        p.wall_stress_xz.init_quad(self.fc_fem, self.quad_list)
        p.wall_stress_xz.build_grad()
        # p.energy.build_grad()

    def _inner_1d(self, field: NDArray) -> NDArray:
        """Extract inner 1D field from 2D field array with ghost cells."""
        assert field.shape[1] == 3, "Not a 1D problem: {}".format(field.shape)
        return field[1:-1, 1:-1].ravel()

    def _get_sol_quad_fields(self) -> dict:
        sol_inner = self._get_inner_sol_fields()
        sol_quad = {}
        for field in sol_inner:
            sol_quad[field] = {}
            for nb_quad in self.quad_list:
                quad_vec = self.quad_fun(sol_inner[field], nb_quad)
                sol_quad[field][nb_quad] = quad_vec
        return sol_quad

    def init_quad(self) -> None:
        p = self.problem
        self.field_list = ['rho', 'jx', 'jy']
        self.field_val_list = [p.q[0], p.q[1], p.q[2]]
        create_quad_fields(self, self.fc_fem, self.field_list, self.quad_list)

    def update_sol_quad(self) -> None:
        for nb_quad in self.quad_list:
            for field, val in zip(self.field_list, self.field_val_list):
                fieldname = f'_{field}_quad_{nb_quad}'
                quad_vals = self.quad_fun(self._inner_1d(val), nb_quad)
                getattr(self, fieldname).p = quad_vals.reshape(nb_quad, -1)

    def get_quad_field(self, field_name: str, nb_quad: int) -> NDArray:
        """Returns the quadrature values of a field in shape (nb_quad, nb_ele)."""
        fieldname = f'_{field_name}_quad_{nb_quad}'
        return getattr(self, fieldname).p

    def update_quad(self) -> None:
        p = self.problem

        # update value and gradient quadrature values
        self.update_sol_quad()
        p.topo.update_quad(self.quad_fun, self.dx_fun, self._inner_1d)
        p.pressure.update_quad(self.quad_fun, self._inner_1d, self.get_quad_field)
        p.wall_stress_xz.update_quad(self.quad_fun, self._inner_1d, self.get_quad_field, p.topo)
        # p.energy.update_quad(self.quad_fun)

    def update(self) -> None:
        self.update_quad()
