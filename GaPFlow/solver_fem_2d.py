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
from .fem.utils2d import (
    NonLinearTerm,
    get_triangle_3_operator,
    get_triangle_2_operator_dx,
    get_triangle_2_operator_dy,
)
from .fem.utils2d import *

from functools import cached_property, lru_cache
import numpy as np
import time

import numpy.typing as npt
from typing import TYPE_CHECKING, Tuple, Dict, Any
if TYPE_CHECKING:
    from .problem import Problem

NDArray = npt.NDArray[np.floating]


BASE_FIELDS = {
    'rho', 'jx', 'jy',
    'p', 'h', 'dh_dx', 'dh_dy', 'eta',
    'U', 'V', 'Ls',
    'dp_drho',
    'rho_prev', 'jx_prev', 'jy_prev',
}

STRESS_XZ_FIELDS = {
    'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
    'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx',
}

STRESS_YZ_FIELDS = {
    'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
    'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy',
}

ENERGY_FIELDS = {
    'E', 'Tb_top', 'Tb_bot',
    'T', 'dT_drho', 'dT_djx', 'dT_djy', 'dT_dE',
    'S', 'dS_drho', 'dS_djx', 'dS_djy', 'dS_dE',
    'E_prev',
}


class FEMSolver2D:
    """FEM Solver for 2D problems using triangular elements."""

    def __init__(self, fem_spec: dict, problem: "Problem") -> None:
        self.fem_spec = fem_spec
        self.problem = problem
        self.num_solver = Solver(problem.fem_solver)
        self.nodal_fields: Dict[str, Any] = {}
        self.quad_fields: Dict[str, Any] = {}

    def _init_convenience_accessors(self) -> None:
        """Initialize convenience accessors for problem and grid properties."""
        p = self.problem

        self.per_x = p.grid['bc_xE_P'][0]  # TODO periodicity information from decomp
        self.per_y = p.grid['bc_yN_P'][0]
        self.energy = p.fem_solver['equations']['energy']
        self.dynamic = p.fem_solver['dynamic']

        self.decomp = p.decomp
        self.global_coords = p.decomp.icoordsg  # shape (2, Nx_padded, Ny_padded)

        nb_subdomain_pts = p.decomp.nb_subdomain_grid_pts
        self.Nx_inner = nb_subdomain_pts[0]
        self.Ny_inner = nb_subdomain_pts[1]
        self.Nx_padded = self.Nx_inner + 2
        self.Ny_padded = self.Ny_inner + 2

        self.bc_at_W = p.decomp.is_at_xW
        self.bc_at_E = p.decomp.is_at_xE
        self.bc_at_S = p.decomp.is_at_yS
        self.bc_at_N = p.decomp.is_at_yN

        self.dx = p.grid['dx']
        self.dy = p.grid['dy']
        self.A = self.dx * self.dy

        self.variables = ['rho', 'jx', 'jy']
        self.residuals = ['mass', 'momentum_x', 'momentum_y']
        self.field_map = {'rho': p.q[0], 'jx': p.q[1], 'jy': p.q[2]}
        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')
            self.field_map['E'] = p.energy.energy

        self.nb_inner_pts = self.Nx_inner * self.Ny_inner
        self.res_size = len(self.residuals) * self.nb_inner_pts
        self.var_size = len(self.variables) * self.nb_contributors

        self.mat_size = (self.res_size, self.var_size)

        self.sq_per_row = self.Nx_inner + 1
        self.sq_per_col = self.Ny_inner + 1
        self.nb_sq = self.sq_per_row * self.sq_per_col

        # Test function values for FEM assembly
        self.test_fun_vals_left = get_N_left_test_vals()
        self.test_fun_vals_right = get_N_right_test_vals()
        self.test_fun_vals_left_dx = get_N_left_test_vals_dx(self.dx)
        self.test_fun_vals_right_dx = get_N_right_test_vals_dx(self.dx)
        self.test_fun_vals_left_dy = get_N_left_test_vals_dy(self.dy)
        self.test_fun_vals_right_dy = get_N_right_test_vals_dy(self.dy)
        self.quad_weights = get_quad_weights()

    def _get_active_terms(self) -> None:
        """Initialize list of active terms from problem fem_solver config."""
        self.terms = get_active_terms(self.fem_spec)

    def _init_quad_operator(self) -> None:
        """Init quadrature point evaluation and derivative operators."""
        self.quad_operator = get_triangle_3_operator()
        self.dx_operator = get_triangle_2_operator_dx(self.dx)
        self.dy_operator = get_triangle_2_operator_dy(self.dy)

    def _init_quad_fields(self) -> None:
        """Initialize quadrature infrastructure."""
        fc = self.problem.fc
        fc.set_nb_sub_pts("quad", 6)
        fc.set_nb_sub_pts("quad_deriv", 2)  # for derivatives

        # multi-component sources need to be copied to single-component fields
        nodal_names = ['rho', 'jx', 'jy', 'h', 'dh_dx', 'dh_dy']
        if self.energy:
            nodal_names.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in nodal_names:
            self.nodal_fields[name] = fc.real_field(f'{name}_nodal', 1, 'pixel')

        # existing single-component fields
        self.nodal_fields['p'] = fc.get_real_field('pressure')
        self.nodal_fields['eta'] = fc.get_real_field('shear_viscosity')

        # quad output fields
        for name in self._get_needed_quad_fields():
            self.quad_fields[name] = fc.real_field(f'{name}_q', 1, 'quad')

        # placeholder field for derivative computation (reused)
        self.deriv_placeholder = fc.real_field('deriv_placeholder', 1, 'quad_deriv')

    def _get_needed_quad_fields(self) -> set:
        """Get set of needed quadrature fields based on active terms."""
        needed = (BASE_FIELDS | STRESS_XZ_FIELDS | STRESS_YZ_FIELDS)
        if self.energy:
            needed |= ENERGY_FIELDS

        return needed

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
            return lambda: self.get_quad(name)

        quad_fields = self._get_needed_quad_fields()

        for term in self.terms:
            term_ctx = {
                name: make_getter(name)
                for name in quad_fields
            }

            # Non-quad values
            term_ctx['dt'] = p.numerics['dt']

            if self.energy:
                term_ctx['k'] = lambda: p.energy.k

            term.build(term_ctx)

    # =========================================================================
    # Quadrature Field Access and Update
    # =========================================================================

    def get_quad(self, name: str) -> NDArray:
        """Get quadrature field values for internal squares.
        Returns shape (6, sq_per_row, sq_per_col).
        """
        return self.quad_fields[name].pg[..., :-1, :-1]

    def get_field_deriv_dx(self, name: str) -> NDArray:
        """Compute d(field)/dx at quad points (2, sq_per_row, sq_per_col)."""
        self.dx_operator.apply(self.nodal_fields[name], self.deriv_placeholder)

        return self.deriv_placeholder.pg[..., :-1, :-1].copy()

    def get_field_deriv_dy(self, name: str) -> NDArray:
        """Compute d(field)/dy at quad points (2, sq_per_row, sq_per_col)."""
        self.dy_operator.apply(self.nodal_fields[name], self.deriv_placeholder)
        return self.deriv_placeholder.pg[..., :-1, :-1].copy()

    def update_nodal_fields(self) -> None:
        """Step 1: Update nodal fields after q update."""
        p = self.problem

        p.pressure.update()
        p.topo.update()
        dp_dx = np.gradient(p.pressure.pressure, p.grid['dx'], axis=0)
        dp_dy = np.gradient(p.pressure.pressure, p.grid['dy'], axis=1)
        p.viscosity.update(p.pressure.pressure, dp_dx, dp_dy, p.topo.h, p.geo['U'], p.geo['V'])
        if self.energy:
            p.energy.update_temperature()

    def update_quad_nodal(self) -> None:
        """Step 2: Interpolate nodal fields to quadrature points."""
        p = self.problem

        # copy multi-component sources to single-component fields
        self.nodal_fields['rho'].pg[0] = p.q[0]
        self.nodal_fields['jx'].pg[0] = p.q[1]
        self.nodal_fields['jy'].pg[0] = p.q[2]
        self.nodal_fields['h'].pg[0] = p.topo.h
        self.nodal_fields['dh_dx'].pg[0] = p.topo.dh_dx
        self.nodal_fields['dh_dy'].pg[0] = p.topo.dh_dy

        if self.energy:
            self.nodal_fields['E'].pg[0] = p.energy.energy
            self.nodal_fields['Tb_top'].pg[0] = p.energy.Tb_top
            self.nodal_fields['Tb_bot'].pg[0] = p.energy.Tb_bot

        # interpolate nodal → quad using operator
        interpolated = ['rho', 'jx', 'jy', 'p', 'h', 'dh_dx', 'dh_dy', 'eta']
        if self.energy:
            interpolated.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in interpolated:
            self.quad_operator.apply(self.nodal_fields[name], self.quad_fields[name])

        # broadcast constants
        self.quad_fields['U'].pg[:] = p.geo['U']
        self.quad_fields['V'].pg[:] = p.geo['V']
        self.quad_fields['Ls'].pg[:] = p.prop.get('slip_length', 0.0)

    def _apply_2d_vmap(self, func, *args):
        """Reshape args for 2D vmap application and reshape result back."""
        shape = args[0].shape  # (6, Ny, Nx)
        args_2d = [a.reshape(shape[0], -1) for a in args]
        result_2d = func(*args_2d)  # shape (nb_quad, Ny * Nx)

        return result_2d.reshape(shape)

    def update_quad_computed(self) -> None:
        """Step 3: Compute derived quantities at quadrature points."""
        p = self.problem
        q = lambda name: self.quad_fields[name].pg
        apply = self._apply_2d_vmap

        # Pressure gradient
        self.quad_fields['dp_drho'].pg = apply(p.pressure.dp_drho, q('rho'))

        # Wall stress xz
        args_xz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dx'), q('U'), q('V'), q('Ls'))
        for name in ['tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
                     'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx']:
            self.quad_fields[name].pg = apply(getattr(p.wall_stress_xz, name), *args_xz)

        # Wall stress yz
        args_yz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dy'), q('U'), q('V'), q('Ls'))
        for name in ['tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
                     'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy']:
            self.quad_fields[name].pg = apply(getattr(p.wall_stress_yz, name), *args_yz)

        if self.energy:
            # Temperature
            args_T = (q('rho'), q('jx'), q('jy'), q('E'))
            for name, func in [('T', 'T_func'), ('dT_drho', 'T_grad_rho'), ('dT_djx', 'T_grad_jx'),
                               ('dT_djy', 'T_grad_jy'), ('dT_dE', 'T_grad_E')]:
                self.quad_fields[name].pg = getattr(p.energy, func)(*args_T)

            # Wall heat flux
            args_S = (q('h'), p.energy.h_Robin, p.energy.k, p.energy.cv, q('eta'),
                      q('rho'), q('E'), q('jx'), q('jy'), q('U'), q('V'), q('Tb_top'), q('Tb_bot'), None)
            for name, func in [('S', 'q_wall_sum'), ('dS_drho', 'q_wall_grad_rho'), ('dS_djx', 'q_wall_grad_jx'),
                               ('dS_djy', 'q_wall_grad_jy'), ('dS_dE', 'q_wall_grad_E')]:
                self.quad_fields[name].pg = getattr(p.energy, func)(*args_S)

    def update_quad(self) -> None:
        """Full quadrature update (Steps 1-3)."""
        self.update_nodal_fields()
        self.update_quad_nodal()
        self.update_quad_computed()

    def update_prev_quad(self) -> None:
        """Store current quad values for time derivatives."""
        for var in self.variables:
            curr_key = f'{var}'
            prev_key = f'{var}_prev'
            if curr_key in self.quad_fields and prev_key in self.quad_fields:
                self.quad_fields[prev_key].pg = np.copy(self.quad_fields[curr_key].pg)

    # =========================================================================
    # Solution Vector Management
    # =========================================================================

    def isBC(self, x: int, y: int) -> bool:
        """Check if the point at (x, y) is a boundary condition point."""
        if (x == 0 and self.bc_at_W) or (x == self.Nx_padded - 1 and self.bc_at_E):
            return True
        if (y == 0 and self.bc_at_S) or (y == self.Ny_padded - 1 and self.bc_at_N):
            return True
        return False

    @cached_property
    def index_mask_inner_local(self) -> NDArray:
        """Only residual (TO) points, local indices, with periodic wrapping"""
        mask = np.full((self.Nx_padded, self.Ny_padded), -1, dtype=int)
        inner_shape = (self.Nx_inner, self.Ny_inner)
        mask[1:-1, 1:-1] = np.arange(np.prod(inner_shape)).reshape(inner_shape, order='F')

        if self.decomp.periodic_x and self.decomp.has_full_x:
            mask[0, :] = mask[self.Nx_padded - 2, :]
            mask[self.Nx_padded - 1, :] = mask[1, :]

        if self.decomp.periodic_y and self.decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_padded - 2]
            mask[:, self.Ny_padded - 1] = mask[:, 1]

        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local(self, var: str = '') -> NDArray:
        """All non-BC points, local indices, with periodic wrapping, with Neumann forwarding."""
        mask = self.index_mask_inner_local.copy()
        cur_val = self.Nx_inner * self.Ny_inner
        for x in range(self.Nx_padded):
            for y in range(self.Ny_padded):
                if mask[x, y] == -1 and not self.isBC(x, y):
                    mask[x, y] = cur_val
                    cur_val += 1
        # Neumann BC forwarding (only if var specified)
        if var:
            var_idx = self.variables.index(var)
            if self.bc_at_W and self.problem.grid['bc_xW_N'][var_idx]:
                mask[0, :] = mask[1, :]
            if self.bc_at_E and self.problem.grid['bc_xE_N'][var_idx]:
                mask[self.Nx_padded - 1, :] = mask[self.Nx_padded - 2, :]
            if self.bc_at_S and self.problem.grid['bc_yS_N'][var_idx]:
                mask[:, 0] = mask[:, 1]
            if self.bc_at_N and self.problem.grid['bc_yN_N'][var_idx]:
                mask[:, self.Ny_padded - 1] = mask[:, self.Ny_padded - 2]
        return mask

    @cached_property
    def nb_contributors(self) -> int:
        """Number of contributors (inner + non-BC padded points)"""
        count = self.Nx_inner * self.Ny_inner
        # Count non-BC padded points
        for x in range(self.Nx_padded):
            for y in range(self.Ny_padded):
                if (x == 0 or x == self.Nx_padded - 1 or
                    y == 0 or y == self.Ny_padded - 1):
                    if not self.isBC(x, y):
                        count += 1
        return count

    def _res_slice(self, res_name) -> slice:
        i = self.residuals.index(res_name)
        return slice(i * self.nb_inner_pts, (i + 1) * self.nb_inner_pts)

    def _var_slice(self, var_name) -> slice:
        i = self.variables.index(var_name)
        return slice(i * self.nb_contributors, (i + 1) * self.nb_contributors)

    def _block_slices(self, res_name, var_name) -> tuple[slice, slice]:
        return (self._res_slice(res_name), self._var_slice(var_name))

    # =========================================================================
    # Matrix and Residual Assembly
    # =========================================================================

    def get_local_nodes_from_sq(self, idx_sq: int, var: str = ''):
        """Returns node info for each corner of the square.

        Returns:
            a_bl, a_br, a_tl, a_tr: Each is 3-tuple:
                (inner_idx, padded_idx, pos)
                - inner_idx: int, residual (TO) nodes with local index
                - padded_idx: int, all contributor nodes with local index
                - pos: tuple(x, y) position in padded grid
            sq_x, sq_y: Square position
        """
        sq_x = idx_sq % self.sq_per_row
        sq_y = idx_sq // self.sq_per_row

        m_inner = self.index_mask_inner_local
        m_padded = self.index_mask_padded_local(var)

        def node_info(x, y):
            inner_idx = m_inner[x, y]
            padded_idx = m_padded[x, y]
            return (inner_idx, padded_idx, (x, y))

        a_bl = node_info(sq_x,     sq_y)
        a_br = node_info(sq_x + 1, sq_y)
        a_tl = node_info(sq_x,     sq_y + 1)
        a_tr = node_info(sq_x + 1, sq_y + 1)

        return a_bl, a_br, a_tl, a_tr, sq_x, sq_y

    def get_res_deriv_vals(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Evaluate derivative of residual function w.r.t. dep_var at quadrature points."""
        dep_var_vals = {dep_var: self.get_quad(dep_var) for dep_var in term.dep_vars}
        res_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
        return res_deriv_vals

    def get_res_vals(self, term: NonLinearTerm) -> NDArray:
        """Evaluate residual function at quadrature points."""
        dep_var_vals = {var: self.get_quad(var) for var in term.dep_vars}
        res_vals = term.evaluate(*[dep_var_vals[var] for var in term.dep_vars])
        return res_vals

    def tang_matrix_term(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Wrapper for different spatial derivatives (nb_inner_pts, nb_contributors)."""
        if term.d_dx_resfun:
            return self.tang_matrix_term_dx(term, dep_var)
        elif term.d_dy_resfun:
            return self.tang_matrix_term_dy(term, dep_var)
        else:
            return self.tang_matrix_term_zero_der(term, dep_var)

    def tang_matrix_term_zero_der(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Get tangent matrix for term without derivative in residual function."""
        M = np.zeros((self.nb_inner_pts, self.nb_contributors), dtype=float)
        res_deriv_vals = self.get_res_deriv_vals(term, dep_var)  # (6, q_per_row, q_per_col)

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq, dep_var)
            res_quad = res_deriv_vals[:, sq_x, sq_y]  # (6,)

            # Left triangle: nodes [bl, tl, br]
            ele_points = [a_bl, a_tl, a_br]
            test_fun_vals = self.test_fun_vals_left  # (3, 3): [node][quad_pt]
            trial_fun_vals = self.test_fun_vals_left  # Same shape functions for trial

            for pt_idx, (pt_TO, _, _) in enumerate(ele_points):
                if pt_TO == -1:
                    continue
                test_vals = test_fun_vals[pt_idx]  # N_i at quad points (3,)

                for trial_idx, (_, pt_FROM, _) in enumerate(ele_points):
                    if pt_FROM == -1:
                        continue
                    trial_vals = trial_fun_vals[trial_idx]  # N_j at quad points (3,)
                    area = np.sum(test_vals * res_quad[0:3] * trial_vals * self.quad_weights) * self.A
                    M[pt_TO, pt_FROM] += area

            # Right triangle: nodes [tr, br, tl]
            ele_points = [a_tr, a_br, a_tl]
            test_fun_vals = self.test_fun_vals_right
            trial_fun_vals = self.test_fun_vals_right

            for pt_idx, (pt_TO, _, _) in enumerate(ele_points):
                if pt_TO == -1:
                    continue
                test_vals = test_fun_vals[pt_idx]

                for trial_idx, (_, pt_FROM, _) in enumerate(ele_points):
                    if pt_FROM == -1:
                        continue
                    trial_vals = trial_fun_vals[trial_idx]
                    area = np.sum(test_vals * res_quad[3:6] * trial_vals * self.quad_weights) * self.A
                    M[pt_TO, pt_FROM] += area

        return M

    def tang_matrix_term_dx(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Get tangent matrix for term with x-derivative in residual function."""
        M = np.zeros((self.nb_inner_pts, self.nb_contributors), dtype=float)
        res_deriv_vals = self.get_res_deriv_vals(term, dep_var)  # (6, q_per_row, q_per_col)

        if term.der_testfun:
            test_fun_vals_left = self.test_fun_vals_left_dx
            test_fun_vals_right = self.test_fun_vals_right_dx
        else:
            test_fun_vals_left = self.test_fun_vals_left
            test_fun_vals_right = self.test_fun_vals_right

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq, dep_var)
            res_quad = res_deriv_vals[:, sq_x, sq_y]  # (6,)

            # Unpack padded indices for dx derivative (bl <-> br, tl <-> tr)
            bl_FROM, br_FROM = a_bl[1], a_br[1]
            tl_FROM, tr_FROM = a_tl[1], a_tr[1]

            # Left triangle: nodes [bl, tl, br], dx involves bl <-> br
            ele_points = [a_bl, a_tl, a_br]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_TO == -1:
                    continue
                base = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A
                if bl_FROM != -1:
                    M[pt_TO, bl_FROM] += base * (-1 / self.dx)
                if br_FROM != -1:
                    M[pt_TO, br_FROM] += base * (1 / self.dx)

            # Right triangle: nodes [tr, br, tl], dx involves tl <-> tr
            ele_points = [a_tr, a_br, a_tl]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_TO == -1:
                    continue
                base = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A
                if tl_FROM != -1:
                    M[pt_TO, tl_FROM] += base * (-1 / self.dx)
                if tr_FROM != -1:
                    M[pt_TO, tr_FROM] += base * (1 / self.dx)

        return M

    def tang_matrix_term_dy(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Get tangent matrix for term with y-derivative in residual function."""
        M = np.zeros((self.nb_inner_pts, self.nb_contributors), dtype=float)
        res_deriv_vals = self.get_res_deriv_vals(term, dep_var)  # (6, q_per_row, q_per_col)

        if term.der_testfun:
            test_fun_vals_left = self.test_fun_vals_left_dy
            test_fun_vals_right = self.test_fun_vals_right_dy
        else:
            test_fun_vals_left = self.test_fun_vals_left
            test_fun_vals_right = self.test_fun_vals_right

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq, dep_var)
            res_quad = res_deriv_vals[:, sq_x, sq_y]  # (6,)

            # Unpack padded indices for dy derivative (bl <-> tl, br <-> tr)
            bl_FROM, tl_FROM = a_bl[1], a_tl[1]
            br_FROM, tr_FROM = a_br[1], a_tr[1]

            # Left triangle: nodes [bl, tl, br], dy involves bl <-> tl
            ele_points = [a_bl, a_tl, a_br]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_TO == -1:
                    continue
                base = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A
                if bl_FROM != -1:
                    M[pt_TO, bl_FROM] += base * (-1 / self.dy)
                if tl_FROM != -1:
                    M[pt_TO, tl_FROM] += base * (1 / self.dy)

            # Right triangle: nodes [tr, br, tl], dy involves br <-> tr
            ele_points = [a_tr, a_br, a_tl]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_TO == -1:
                    continue
                base = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A
                if br_FROM != -1:
                    M[pt_TO, br_FROM] += base * (-1 / self.dy)
                if tr_FROM != -1:
                    M[pt_TO, tr_FROM] += base * (1 / self.dy)

        return M

    def residual_vector_term(self, term: NonLinearTerm) -> NDArray:
        """Wrapper for different spatial derivatives (nb_inner_pts,)."""
        if term.d_dx_resfun:
            return self.residual_vector_term_dx(term)
        elif term.d_dy_resfun:
            return self.residual_vector_term_dy(term)
        else:
            return self.residual_vector_term_zero_der(term)

    def residual_vector_term_zero_der(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term without derivative in residual function."""
        R = np.zeros((self.nb_inner_pts,), dtype=float)
        res_fun_vals = self.get_res_vals(term)  # (6, sq_per_row, sq_per_col)

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)
            res_quad = res_fun_vals[:, sq_x, sq_y]  # (6,)

            # Left triangle: nodes [bl, tl, br]
            ele_points = [a_bl, a_tl, a_br]
            for (pt_TO, _, _), test_vals in zip(ele_points, self.test_fun_vals_left):
                if pt_TO == -1:
                    continue
                R[pt_TO] += np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A

            # Right triangle: nodes [tr, br, tl]
            ele_points = [a_tr, a_br, a_tl]
            for (pt_TO, _, _), test_vals in zip(ele_points, self.test_fun_vals_right):
                if pt_TO == -1:
                    continue
                R[pt_TO] += np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A

        return R

    def residual_vector_term_dx(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term with d/dx in residual function."""
        R = np.zeros((self.nb_inner_pts,), dtype=float)

        # Shape: (6, sq_per_row, sq_per_col)
        dF_dx = np.zeros((6, self.sq_per_row, self.sq_per_col))
        for dep_var in term.dep_vars:
            dF_dvar = self.get_res_deriv_vals(term, dep_var)  # (6, X, Y)
            dvar_dx_2 = self.get_field_deriv_dx(dep_var)      # (2, X, Y)
            dvar_dx_6 = np.repeat(dvar_dx_2, 3, axis=0)       # expand to (6, X, Y)
            dF_dx += dF_dvar * dvar_dx_6

        if term.der_testfun:
            test_fun_vals_left = self.test_fun_vals_left_dx
            test_fun_vals_right = self.test_fun_vals_right_dx
        else:
            test_fun_vals_left = self.test_fun_vals_left
            test_fun_vals_right = self.test_fun_vals_right

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)
            res_quad = dF_dx[:, sq_x, sq_y]

            # Left triangle: nodes [bl, tl, br]
            ele_points = [a_bl, a_tl, a_br]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_TO == -1:
                    continue
                R[pt_TO] += np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A

            # Right triangle: nodes [tr, br, tl]
            ele_points = [a_tr, a_br, a_tl]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_TO == -1:
                    continue
                R[pt_TO] += np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A

        return R

    def residual_vector_term_dy(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term with d/dy in residual function."""
        R = np.zeros((self.nb_inner_pts,), dtype=float)

        # Shape: (6, sq_per_row, sq_per_col)
        dF_dy = np.zeros((6, self.sq_per_row, self.sq_per_col))
        for dep_var in term.dep_vars:
            dF_dvar = self.get_res_deriv_vals(term, dep_var)  # (6, X, Y)
            dvar_dy_2 = self.get_field_deriv_dy(dep_var)      # (2, X, Y)
            dvar_dy_6 = np.repeat(dvar_dy_2, 3, axis=0)       # expand to (6, X, Y)
            dF_dy += dF_dvar * dvar_dy_6

        if term.der_testfun:
            test_fun_vals_left = self.test_fun_vals_left_dy
            test_fun_vals_right = self.test_fun_vals_right_dy
        else:
            test_fun_vals_left = self.test_fun_vals_left
            test_fun_vals_right = self.test_fun_vals_right

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)
            res_quad = dF_dy[:, sq_x, sq_y]

            # Left triangle: nodes [bl, tl, br]
            ele_points = [a_bl, a_tl, a_br]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_TO == -1:
                    continue
                R[pt_TO] += np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A

            # Right triangle: nodes [tr, br, tl]
            ele_points = [a_tr, a_br, a_tl]
            for (pt_TO, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_TO == -1:
                    continue
                R[pt_TO] += np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A

        return R

    def get_tang_matrix(self) -> NDArray:
        """Assemble full tangent matrix from all terms (res_size, var_size)."""
        tang_matrix = np.zeros(self.mat_size)
        for term in self.terms:
            if (not self.dynamic) and 'T' in term.name:
                continue
            for dep_var in term.dep_vars:
                bl = self._block_slices(term.res, dep_var)
                tang_matrix[bl] += self.tang_matrix_term(term, dep_var)
        return tang_matrix

    def get_residual_vec(self) -> NDArray:
        """Assemble full residual vector from all terms (res_size,)."""
        res_vec = np.zeros(self.res_size)
        for term in self.terms:
            if (not self.dynamic) and 'T' in term.name:
                continue
            sl = self._res_slice(term.res)
            res_vec[sl] += self.residual_vector_term(term)
        return res_vec

    def get_M(self) -> NDArray:
        M = self.get_tang_matrix()
        return M

    def get_R(self) -> NDArray:
        R = self.get_residual_vec()
        return R

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
        self.update_quad()
        M = self.get_M()
        R = self.get_R()
        return M, R

    def update_output_fields(self) -> None:
        """Update nodal output fields (wall stress, bulk stress) for plotting/output."""
        p = self.problem
        p.wall_stress_xz.update()
        p.wall_stress_yz.update()
        if hasattr(p, 'bulk_stress'):
            p.bulk_stress.update()

    def steady_state(self) -> None:
        """Solve steady-state problem using Newton iteration."""
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

        # Update output fields for plotting
        self.update_output_fields()

        p._stop = True

    def update_dynamic(self) -> None:
        """Do a single dynamic time step update, then return to problem main loop."""
        p = self.problem
        self.update_prev_quad()

        self.num_solver.sol_dict.reset()
        self.num_solver.sol_dict.q0 = self.get_q_nodal()
        self.num_solver.get_MR_fun = self.solver_step_fun

        tic = time.time()
        self.num_solver.solve(silent=True)
        toc = time.time()
        self.time_inner = toc - tic

        # Update output fields for plotting
        self.update_output_fields()

        p._post_update()

    def update(self) -> None:
        """Top-level solver update function."""
        if self.dynamic:
            self.update_dynamic()
        else:
            self.steady_state()

    def print_status_header(self) -> None:
        """Print header for dynamic simulation status output."""
        p = self.problem
        if not p.options['silent'] and self.dynamic:
            print(61 * '-')
            print(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} {'Convergence Time':<18s} {'Residual':<12s}")
            print(61 * '-')
            p.write(params=False)

    def print_status(self, scalars=None) -> None:
        """Print status line for dynamic simulation."""
        p = self.problem
        if not p.options['silent'] and self.dynamic:
            print(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} {self.time_inner:<18.4e} {p.residual:<12.4e}")

    def pre_run(self, **kwargs) -> None:
        """Initialize solver before running."""
        self._init_convenience_accessors()
        self._init_quad_operator()
        self._init_quad_fields()
        self._get_active_terms()
        self._build_jit_functions()
        self._build_terms()


class PETScTranslator():
    pass
