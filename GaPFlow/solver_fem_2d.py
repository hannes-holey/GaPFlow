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

from functools import cached_property
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
        if self.energy:
            self.variables.append('E')
            self.residuals.append('energy')

        self.nb_inner_pts = self.Nx_inner * self.Ny_inner
        self.res_size = len(self.residuals) * self.nb_inner_pts
        self.var_size = len(self.variables) * self.nb_contributors

        self.mat_size = (self.res_size, self.var_size)

        assert self.nb_inner_pts == len(self.indices_inner_local)
        assert self.nb_contributors == len(self.indices_padded_local)

        self.sq_per_row = self.Nx_inner + 1
        self.sq_per_col = self.Ny_inner + 1
        self.nb_sq = self.sq_per_row * self.sq_per_col

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

    def isBC(self, i: int, j: int) -> bool:
        """Check if the point at (i, j) is a boundary condition point."""
        if (i == 0 and self.bc_at_S) or (i == self.Ny_padded - 1 and self.bc_at_N):
            return True
        if (j == 0 and self.bc_at_W) or (j == self.Nx_padded - 1 and self.bc_at_E):
            return True
        return False

    @cached_property
    def index_mask_inner_local(self) -> NDArray:
        """2D Index mask for inner points with local indices."""
        mask = np.full((self.Ny_padded, self.Nx_padded), -1, dtype=int)
        inner_shape = (self.Ny_inner, self.Nx_inner)
        mask[1:-1, 1:-1] = np.arange(np.prod(inner_shape)).reshape(inner_shape)
        return mask

    @cached_property
    def index_mask_inner_with_periodic(self) -> NDArray:
        """2D Index mask for inner points including periodic ghost mappings.

        For periodic BCs, ghost positions map to the corresponding inner node
        on the opposite boundary (all variables have the same periodic BCs).
        Only applies if there's 1 process in that direction (local wrapping).
        Used for test function handling in Jacobian/residual assembly.
        """
        mask = self.index_mask_inner_local.copy()

        # Add periodic ghost mappings (using var 0 as reference - all vars same)
        # Only if there's 1 process in that direction (local wrapping possible)
        # Decomposition is (1, size) - always 1 in x, size in y
        p = self.problem
        decomp = p.decomp

        # X direction: always 1 process, so local wrapping valid
        local_periodic_x = True
        # Y direction: local wrapping only if single rank
        local_periodic_y = (decomp.size == 1)

        # South periodic: y=0 maps to y=Ny_padded-2 (top inner row)
        if p.grid['bc_yS_P'][0] and local_periodic_y:
            mask[0, 1:-1] = mask[self.Ny_padded - 2, 1:-1]
        # North periodic: y=Ny_padded-1 maps to y=1 (bottom inner row)
        if p.grid['bc_yN_P'][0] and local_periodic_y:
            mask[self.Ny_padded - 1, 1:-1] = mask[1, 1:-1]
        # West periodic: x=0 maps to x=Nx_padded-2 (right inner column)
        if p.grid['bc_xW_P'][0] and local_periodic_x:
            mask[1:-1, 0] = mask[1:-1, self.Nx_padded - 2]
        # East periodic: x=Nx_padded-1 maps to x=1 (left inner column)
        if p.grid['bc_xE_P'][0] and local_periodic_x:
            mask[1:-1, self.Nx_padded - 1] = mask[1:-1, 1]

        # Handle corners for fully periodic cases (both directions must be local)
        if p.grid['bc_yS_P'][0] and p.grid['bc_xW_P'][0] and local_periodic_y and local_periodic_x:
            mask[0, 0] = mask[self.Ny_padded - 2, self.Nx_padded - 2]
        if p.grid['bc_yS_P'][0] and p.grid['bc_xE_P'][0] and local_periodic_y and local_periodic_x:
            mask[0, self.Nx_padded - 1] = mask[self.Ny_padded - 2, 1]
        if p.grid['bc_yN_P'][0] and p.grid['bc_xW_P'][0] and local_periodic_y and local_periodic_x:
            mask[self.Ny_padded - 1, 0] = mask[1, self.Nx_padded - 2]
        if p.grid['bc_yN_P'][0] and p.grid['bc_xE_P'][0] and local_periodic_y and local_periodic_x:
            mask[self.Ny_padded - 1, self.Nx_padded - 1] = mask[1, 1]

        return mask

    @cached_property
    def indices_inner_local(self) -> NDArray:
        """1D index list for inner points with local indices"""
        inner_shape = (self.Nx_inner, self.Ny_inner)
        return np.arange(np.prod(inner_shape))

    @cached_property
    def index_mask_padded_local(self) -> NDArray:
        """2D Index mask for padded points with local indices"""
        mask = self.index_mask_inner_local.copy()
        cur_val = self.Nx_inner * self.Ny_inner
        for i in range(self.Ny_padded):
            for j in range(self.Nx_padded):
                if mask[i, j] == -1 and not self.isBC(i, j):
                    mask[i, j] = cur_val
                    cur_val += 1
        return mask

    @cached_property
    def indices_padded_local(self) -> NDArray:
        """1D index list for padded points with local indices"""
        indices = self.index_mask_padded_local.copy()
        return indices[indices != -1]

    @cached_property
    def index_mask_all_global(self) -> NDArray:
        """2D Index mask for all points with global indices"""
        mask = np.zeros((self.Ny_padded, self.Nx_padded), dtype=int)
        mask = self.global_coords[1, :] * self.Ny_inner + self.global_coords[0, :]
        return mask

    @cached_property
    def index_mask_inner_global(self) -> NDArray:
        """2D Index mask for inner points with global indices"""
        mask = self.index_mask_all_global.copy()
        mask[[0, -1], :] = -1
        mask[:, [0, -1]] = -1
        return mask

    @cached_property
    def indices_inner_global(self) -> NDArray:
        """1D index list for inner points with global indices"""
        indices = self.index_mask_inner_global.copy()
        return indices[indices != -1]

    @cached_property
    def index_mask_padded_global(self) -> NDArray:
        """2D Index mask for padded points with global indices"""
        mask = self.index_mask_all_global.copy()
        if self.bc_at_W:
            mask[:, 0] = -1
        if self.bc_at_E:
            mask[:, -1] = -1
        if self.bc_at_S:
            mask[0, :] = -1
        if self.bc_at_N:
            mask[-1, :] = -1
        return mask

    @cached_property
    def indices_padded_global(self) -> NDArray:
        """1D index list for padded points with global indices"""
        indices = self.index_mask_padded_global.copy()
        return indices[indices != -1]

    def _build_neumann_mask(self, var_idx: int) -> NDArray:
        """Build Neumann mask for variable at index var_idx.

        Returns bool array of shape (Ny_padded, Nx_padded) where True indicates
        a Neumann boundary for this variable.
        """
        p = self.problem
        mask = np.zeros((self.Ny_padded, self.Nx_padded), dtype=bool)

        if p.grid['bc_xW_N'][var_idx]:
            mask[:, 0] = True           # West column
        if p.grid['bc_xE_N'][var_idx]:
            mask[:, -1] = True          # East column
        if p.grid['bc_yS_N'][var_idx]:
            mask[0, :] = True           # South row
        if p.grid['bc_yN_N'][var_idx]:
            mask[-1, :] = True          # North row

        return mask

    @cached_property
    def index_mask_Neumann_rho(self) -> NDArray:
        """Neumann BC mask for rho: True at Neumann boundaries, False elsewhere."""
        return self._build_neumann_mask(0)

    @cached_property
    def index_mask_Neumann_jx(self) -> NDArray:
        """Neumann BC mask for jx: True at Neumann boundaries, False elsewhere."""
        return self._build_neumann_mask(1)

    @cached_property
    def index_mask_Neumann_jy(self) -> NDArray:
        """Neumann BC mask for jy: True at Neumann boundaries, False elsewhere."""
        return self._build_neumann_mask(2)

    @cached_property
    def index_mask_Neumann_E(self) -> NDArray:
        """Neumann BC mask for E: True at Neumann boundaries, False elsewhere."""
        return self._build_neumann_mask(3)

    @cached_property
    def neumann_masks(self) -> list:
        """List of Neumann masks in variable order [rho, jx, jy] or [rho, jx, jy, E]."""
        masks = [self.index_mask_Neumann_rho,
                 self.index_mask_Neumann_jx,
                 self.index_mask_Neumann_jy]
        if self.energy:
            masks.append(self.index_mask_Neumann_E)
        return masks

    def _build_periodic_mask(self, var_idx: int) -> NDArray:
        """Build periodic mask for variable at index var_idx.

        Returns bool array of shape (Ny_padded, Nx_padded) where True indicates
        a periodic boundary for this variable.
        """
        p = self.problem
        mask = np.zeros((self.Ny_padded, self.Nx_padded), dtype=bool)

        if p.grid['bc_xW_P'][var_idx]:
            mask[:, 0] = True           # West column
        if p.grid['bc_xE_P'][var_idx]:
            mask[:, -1] = True          # East column
        if p.grid['bc_yS_P'][var_idx]:
            mask[0, :] = True           # South row
        if p.grid['bc_yN_P'][var_idx]:
            mask[-1, :] = True          # North row

        return mask

    @cached_property
    def periodic_masks(self) -> list:
        """List of periodic masks in variable order [rho, jx, jy] or [rho, jx, jy, E]."""
        masks = [self._build_periodic_mask(i) for i in range(len(self.variables))]
        return masks

    @cached_property
    def nb_contributors(self) -> int:
        """Number of contributors (inner + non-BC padded points)"""
        count = self.Nx_inner * self.Ny_inner
        # Count non-BC padded points
        for i in range(self.Ny_padded):
            for j in range(self.Nx_padded):
                if (i == 0 or i == self.Ny_padded - 1 or
                    j == 0 or j == self.Nx_padded - 1):
                    if not self.isBC(i, j):
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

    def get_neumann_adjacent_inner(self, x: int, y: int, var_idx: int) -> int:
        """Get the inner node index that a Neumann ghost at (x, y) copies from.

        For Neumann BC, ghost = adjacent inner value. This returns the inner node
        index that the ghost at position (x, y) would copy from.

        For corners with mixed BCs (one direction periodic, other Neumann),
        only shift in the Neumann direction - the periodic direction is handled
        by index_mask_inner_with_periodic.

        Returns -1 if position is not a Neumann ghost for this variable.
        """
        masks = self.neumann_masks
        if not masks[var_idx][y, x]:
            return -1  # Not a Neumann ghost

        # Determine which boundaries this position is on
        is_west = (x == 0)
        is_east = (x == self.Nx_padded - 1)
        is_south = (y == 0)
        is_north = (y == self.Ny_padded - 1)

        # Check which directions are truly Neumann (not periodic)
        p = self.problem
        x_is_neumann = not (is_west and p.grid['bc_xW_P'][var_idx]) and \
                       not (is_east and p.grid['bc_xE_P'][var_idx])
        y_is_neumann = not (is_south and p.grid['bc_yS_P'][var_idx]) and \
                       not (is_north and p.grid['bc_yN_P'][var_idx])

        # Calculate the target position - only shift in Neumann directions
        target_x = x
        target_y = y

        if x_is_neumann:
            if is_west:
                target_x = 1
            elif is_east:
                target_x = self.Nx_padded - 2

        if y_is_neumann:
            if is_south:
                target_y = 1
            elif is_north:
                target_y = self.Ny_padded - 2

        # If position didn't change, it's not a boundary ghost (or purely periodic)
        if target_x == x and target_y == y:
            return -1

        # Use index_mask_inner_with_periodic to handle cases where the target
        # position is on a periodic boundary
        return self.index_mask_inner_with_periodic[target_y, target_x]

    def get_periodic_adjacent_inner(self, x: int, y: int, var_idx: int) -> int:
        """Get the inner node index that a periodic ghost at (x, y) maps to.

        For periodic BC, ghost at one boundary maps to inner node at opposite boundary.
        - South ghost (y=0) maps to inner node at y=Ny_padded-2 (top inner row)
        - North ghost (y=Ny_padded-1) maps to inner node at y=1 (bottom inner row)
        - West ghost (x=0) maps to inner node at x=Nx_padded-2 (right inner column)
        - East ghost (x=Nx_padded-1) maps to inner node at x=1 (left inner column)

        For corners where one direction is periodic and the other is not (e.g., Dirichlet),
        returns -1 since the contribution should be handled by the non-periodic BC.

        Returns -1 if position is not a periodic ghost for this variable,
        or if the target position is not an inner node.
        """
        masks = self.periodic_masks
        if not masks[var_idx][y, x]:
            return -1  # Not a periodic ghost

        # Determine which boundaries this position is on
        is_west = (x == 0)
        is_east = (x == self.Nx_padded - 1)
        is_south = (y == 0)
        is_north = (y == self.Ny_padded - 1)

        # Calculate the target position (wrap to opposite side ONLY in periodic direction)
        target_x = x
        target_y = y

        # Only wrap x if x boundary is periodic for this variable
        p = self.problem
        if is_west and p.grid['bc_xW_P'][var_idx]:
            target_x = self.Nx_padded - 2  # Right inner column
        elif is_east and p.grid['bc_xE_P'][var_idx]:
            target_x = 1  # Left inner column

        # Only wrap y if y boundary is periodic for this variable
        if is_south and p.grid['bc_yS_P'][var_idx]:
            target_y = self.Ny_padded - 2  # Top inner row
        elif is_north and p.grid['bc_yN_P'][var_idx]:
            target_y = 1  # Bottom inner row

        # If position didn't change, it's not a purely periodic boundary ghost
        if target_x == x and target_y == y:
            return -1

        # If target is still in ghost region (corner case), return -1
        target_inner = self.index_mask_inner_local[target_y, target_x]
        if target_inner == -1:
            return -1

        return target_inner

    def get_trial_target(self, tr_inner: int, tr_padded: int, tr_neu: tuple,
                         tr_pos: tuple, var_idx: int) -> int:
        """Get the target column index for a trial node in Jacobian assembly.

        For inner nodes, returns the inner index.
        For ghost nodes, redirects based on BC type:
        - Periodic: redirect to opposite boundary inner node
        - Neumann: redirect to adjacent inner node
        - Dirichlet: skip (return -1)

        Args:
            tr_inner: Inner grid index (-1 for ghost)
            tr_padded: Padded grid index (-1 for BC ghost)
            tr_neu: Tuple of Neumann flags per variable
            tr_pos: Position (x, y) in padded grid
            var_idx: Variable index

        Returns:
            Target column index, or -1 to skip
        """
        if tr_inner != -1:
            # Inner node
            return tr_inner

        # Ghost node - check BC type
        # 1. Check periodic first
        target = self.get_periodic_adjacent_inner(*tr_pos, var_idx)
        if target != -1:
            return target

        # 2. Check Neumann
        if tr_neu[var_idx]:
            return self.get_neumann_adjacent_inner(*tr_pos, var_idx)

        # 3. Padded but not BC (shouldn't happen for boundary ghosts)
        if tr_padded != -1:
            return tr_padded

        # 4. Dirichlet - skip
        return -1

    def get_local_nodes_from_sq(self, idx_sq: int):
        """Returns node info for each corner of the square.

        Returns:
            a_bl, a_br, a_tl, a_tr: Each is 4-tuple:
                (inner_idx, padded_idx, neumann_tuple, pos)
                - inner_idx: int, index in inner grid (-1 for ghost)
                - padded_idx: int, index in padded grid (-1 for BC)
                - neumann_tuple: tuple(bool, ...) one per variable
                - pos: tuple(x, y) position in padded grid
            sq_x, sq_y: Square position
        """
        sq_x = idx_sq % self.sq_per_row
        sq_y = idx_sq // self.sq_per_row
        m_inner = self.index_mask_inner_with_periodic
        m_padded = self.index_mask_padded_local
        masks = self.neumann_masks

        def node_info(x, y):
            inner_idx = m_inner[y, x]
            padded_idx = m_padded[y, x]
            neumann = tuple(mask[y, x] for mask in masks)
            return (inner_idx, padded_idx, neumann, (x, y))

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

    @cached_property
    def test_fun_vals_left(self) -> NDArray:
        """Test function values at left element quadrature points (3, 3) [bl, tl, br]."""
        return get_N_left_test_vals()

    @cached_property
    def test_fun_vals_right(self) -> NDArray:
        """Test function values at right element quadrature points (3, 3) [tr, br, tl]."""
        return get_N_right_test_vals()

    @cached_property
    def test_fun_vals_left_dx(self) -> NDArray:
        """Test function dx values at left element quadrature points (3, 3) [bl, tl, br]."""
        return get_N_left_test_vals_dx(self.dx)
    
    @cached_property
    def test_fun_vals_right_dx(self) -> NDArray:
        """Test function dx values at right element quadrature points (3, 3) [tr, br, tl]."""
        return get_N_right_test_vals_dx(self.dx)

    @cached_property
    def test_fun_vals_left_dy(self) -> NDArray:
        """Test function dy values at left element quadrature points (3, 3) [bl, tl, br]."""
        return get_N_left_test_vals_dy(self.dy)

    @cached_property
    def test_fun_vals_right_dy(self) -> NDArray:
        """Test function dy values at right element quadrature points (3, 3) [tr, br, tl]."""
        return get_N_right_test_vals_dy(self.dy)

    @cached_property
    def quad_weights(self) -> NDArray:
        """Quadrature weights for triangle with 3 quadrature points (3,)."""
        return get_quad_weights()

    def tang_matrix_term(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Wrapper for different spatial derivatives (nb_inner_pts, nb_contributors)."""
        if term.d_dx_resfun:
            return self.tang_matrix_term_dx(term, dep_var)
        elif term.d_dy_resfun:
            return self.tang_matrix_term_dy(term, dep_var)
        else:
            return self.tang_matrix_term_zero_der(term, dep_var)

    def tang_matrix_term_zero_der(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Get tangent matrix for term without derivative in residual function.

        For zero-order terms: R_i = ∫ N_i * F(q) dΩ
        Jacobian: M[i,j] = ∫ N_i * (dF/dq) * N_j dΩ

        Where N_i is test function, N_j is trial function (same shape functions).

        For boundary ghosts, redirects contributions based on BC type:
        - Periodic: redirect to opposite boundary inner node
        - Neumann: redirect to adjacent inner node
        - Dirichlet: skip
        """
        M = np.zeros((self.nb_inner_pts, self.nb_contributors), dtype=float)
        res_deriv_vals = self.get_res_deriv_vals(term, dep_var)  # (6, q_per_row, q_per_col)
        var_idx = self.variables.index(dep_var)

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)
            res_quad = res_deriv_vals[:, sq_x, sq_y]  # (6,)

            # Left triangle: nodes [bl, tl, br]
            ele_points = [a_bl, a_tl, a_br]
            test_fun_vals = self.test_fun_vals_left  # (3, 3): [node][quad_pt]
            trial_fun_vals = self.test_fun_vals_left  # Same shape functions for trial

            for pt_idx, (pt_inner, _, _, _) in enumerate(ele_points):
                if pt_inner not in self.indices_inner_local:
                    continue
                test_vals = test_fun_vals[pt_idx]  # N_i at quad points (3,)

                for trial_idx, (tr_inner, tr_padded, tr_neu, tr_pos) in enumerate(ele_points):
                    target = self.get_trial_target(tr_inner, tr_padded, tr_neu, tr_pos, var_idx)
                    if target == -1:
                        continue
                    trial_vals = trial_fun_vals[trial_idx]  # N_j at quad points (3,)

                    # M[i,j] = ∫ N_i * (dF/dq) * N_j dΩ
                    area = np.sum(test_vals * res_quad[0:3] * trial_vals * self.quad_weights) * self.A
                    M[pt_inner, target] += area

            # Right triangle: nodes [tr, br, tl]
            ele_points = [a_tr, a_br, a_tl]
            test_fun_vals = self.test_fun_vals_right
            trial_fun_vals = self.test_fun_vals_right

            for pt_idx, (pt_inner, _, _, _) in enumerate(ele_points):
                if pt_inner not in self.indices_inner_local:
                    continue
                test_vals = test_fun_vals[pt_idx]

                for trial_idx, (tr_inner, tr_padded, tr_neu, tr_pos) in enumerate(ele_points):
                    target = self.get_trial_target(tr_inner, tr_padded, tr_neu, tr_pos, var_idx)
                    if target == -1:
                        continue
                    trial_vals = trial_fun_vals[trial_idx]

                    area = np.sum(test_vals * res_quad[3:6] * trial_vals * self.quad_weights) * self.A
                    M[pt_inner, target] += area

        return M

    def tang_matrix_term_dx(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Get tangent matrix for term with first derivative in residual function.

        For boundary ghosts, redirects contributions based on BC type:
        - Periodic: redirect to opposite boundary inner node
        - Neumann: redirect to adjacent inner node
        - Dirichlet: skip
        """
        M = np.zeros((self.nb_inner_pts, self.nb_contributors), dtype=float)
        res_deriv_vals = self.get_res_deriv_vals(term, dep_var)  # (6, q_per_row, q_per_col)
        var_idx = self.variables.index(dep_var)

        if term.der_testfun:
            test_fun_vals_left = self.test_fun_vals_left_dx
            test_fun_vals_right = self.test_fun_vals_right_dx
        else:
            test_fun_vals_left = self.test_fun_vals_left
            test_fun_vals_right = self.test_fun_vals_right

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)

            # Unpack node info (4-tuple now includes position)
            bl_inner, bl_padded, bl_neu, bl_pos = a_bl
            br_inner, br_padded, br_neu, br_pos = a_br
            tl_inner, tl_padded, tl_neu, tl_pos = a_tl
            tr_inner, tr_padded, tr_neu, tr_pos = a_tr

            # Field indexing: [:, x, y] to match muGrid's (sub_pts, X, Y) convention
            res_quad = res_deriv_vals[:, sq_x, sq_y]  # (6,)

            # Left triangle: dx involves bl <-> br
            bl_target = self.get_trial_target(bl_inner, bl_padded, bl_neu, bl_pos, var_idx)
            br_target = self.get_trial_target(br_inner, br_padded, br_neu, br_pos, var_idx)

            ele_points = [(bl_inner, bl_padded, bl_pos), (tl_inner, tl_padded, tl_pos),
                          (br_inner, br_padded, br_pos)]
            for (pt_inner, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_inner not in self.indices_inner_local:
                    continue  # not in residual

                base = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A

                # dx derivative: d/dx = (br - bl) / dx
                if bl_target != -1:
                    M[pt_inner, bl_target] += base * (-1 / self.dx)
                if br_target != -1:
                    M[pt_inner, br_target] += base * (1 / self.dx)

            # Right triangle: dx involves tl <-> tr
            tl_target = self.get_trial_target(tl_inner, tl_padded, tl_neu, tl_pos, var_idx)
            tr_target = self.get_trial_target(tr_inner, tr_padded, tr_neu, tr_pos, var_idx)

            ele_points = [(tr_inner, tr_padded, tr_pos), (br_inner, br_padded, br_pos),
                          (tl_inner, tl_padded, tl_pos)]
            for (pt_inner, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_inner not in self.indices_inner_local:
                    continue  # not in residual

                base = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A

                # dx derivative: d/dx = (tr - tl) / dx
                if tl_target != -1:
                    M[pt_inner, tl_target] += base * (-1 / self.dx)
                if tr_target != -1:
                    M[pt_inner, tr_target] += base * (1 / self.dx)

        return M

    def tang_matrix_term_dy(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        """Get tangent matrix for term with first derivative in residual function.

        For boundary ghosts, redirects contributions based on BC type:
        - Periodic: redirect to opposite boundary inner node
        - Neumann: redirect to adjacent inner node
        - Dirichlet: skip
        """
        M = np.zeros((self.nb_inner_pts, self.nb_contributors), dtype=float)
        res_deriv_vals = self.get_res_deriv_vals(term, dep_var)  # (6, q_per_row, q_per_col)
        var_idx = self.variables.index(dep_var)

        if term.der_testfun:
            test_fun_vals_left = self.test_fun_vals_left_dy
            test_fun_vals_right = self.test_fun_vals_right_dy
        else:
            test_fun_vals_left = self.test_fun_vals_left
            test_fun_vals_right = self.test_fun_vals_right

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)

            # Unpack node info (4-tuple now includes position)
            bl_inner, bl_padded, bl_neu, bl_pos = a_bl
            br_inner, br_padded, br_neu, br_pos = a_br
            tl_inner, tl_padded, tl_neu, tl_pos = a_tl
            tr_inner, tr_padded, tr_neu, tr_pos = a_tr

            # Field indexing: [:, x, y] to match muGrid's (sub_pts, X, Y) convention
            res_quad = res_deriv_vals[:, sq_x, sq_y]  # (6,)

            # Left triangle: dy involves bl <-> tl
            bl_target = self.get_trial_target(bl_inner, bl_padded, bl_neu, bl_pos, var_idx)
            tl_target = self.get_trial_target(tl_inner, tl_padded, tl_neu, tl_pos, var_idx)

            ele_points = [(bl_inner, bl_padded, bl_pos), (tl_inner, tl_padded, tl_pos),
                          (br_inner, br_padded, br_pos)]
            for (pt_inner, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_inner not in self.indices_inner_local:
                    continue  # not in residual

                base = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A

                # dy derivative: d/dy = (tl - bl) / dy
                if bl_target != -1:
                    M[pt_inner, bl_target] += base * (-1 / self.dy)
                if tl_target != -1:
                    M[pt_inner, tl_target] += base * (1 / self.dy)

            # Right triangle: dy involves br <-> tr
            br_target = self.get_trial_target(br_inner, br_padded, br_neu, br_pos, var_idx)
            tr_target = self.get_trial_target(tr_inner, tr_padded, tr_neu, tr_pos, var_idx)

            ele_points = [(tr_inner, tr_padded, tr_pos), (br_inner, br_padded, br_pos),
                          (tl_inner, tl_padded, tl_pos)]
            for (pt_inner, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_inner not in self.indices_inner_local:
                    continue  # not in residual

                base = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A

                # dy derivative: d/dy = (tr - br) / dy
                if br_target != -1:
                    M[pt_inner, br_target] += base * (-1 / self.dy)
                if tr_target != -1:
                    M[pt_inner, tr_target] += base * (1 / self.dy)

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

        R = np.zeros((self.nb_inner_pts,), dtype=float)  # (nb_inner_pts,)
        res_fun_vals = self.get_res_vals(term)  # (6, sq_per_row, sq_per_col)

        for idx_sq in range(self.nb_sq):
            a_bl, a_br, a_tl, a_tr, sq_x, sq_y = self.get_local_nodes_from_sq(idx_sq)
            # Field indexing: [:, x, y] to match muGrid's (sub_pts, X, Y) convention
            res_quad = res_fun_vals[:, sq_x, sq_y]  # (6,)

            # element 1 (left triangle)
            ele_points = [a_bl, a_tl, a_br]
            for (pt_inner, _, _, _), test_vals in zip(ele_points, self.test_fun_vals_left):  # test_vals (3,)
                if pt_inner not in self.indices_inner_local:
                    continue  # not in residual
                area = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A
                R[pt_inner] += area

            # element 2 (right triangle)
            ele_points = [a_tr, a_br, a_tl]
            for (pt_inner, _, _, _), test_vals in zip(ele_points, self.test_fun_vals_right):  # test_vals (3,)
                if pt_inner not in self.indices_inner_local:
                    continue  # not in residual
                area = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A
                R[pt_inner] += area
        return R

    def residual_vector_term_dx(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term with d/dx in residual function."""

        R = np.zeros((self.nb_inner_pts,), dtype=float)

        # Shape: (6, sq_per_row, sq_per_col) to match muGrid's (sub_pts, X, Y) convention
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
            # Field indexing: [:, x, y] to match muGrid's (sub_pts, X, Y) convention
            res_quad = dF_dx[:, sq_x, sq_y]

            # Left triangle
            ele_points = [a_bl, a_tl, a_br]
            for (pt_inner, _, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_inner not in self.indices_inner_local:
                    continue
                area = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A
                R[pt_inner] += area

            # Right triangle
            ele_points = [a_tr, a_br, a_tl]
            for (pt_inner, _, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_inner not in self.indices_inner_local:
                    continue
                area = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A
                R[pt_inner] += area

        return R

    def residual_vector_term_dy(self, term: NonLinearTerm) -> NDArray:
        """Get residual vector for term with d/dy in residual function."""

        R = np.zeros((self.nb_inner_pts,), dtype=float)

        # Shape: (6, sq_per_row, sq_per_col) to match muGrid's (sub_pts, X, Y) convention
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
            # Field indexing: [:, x, y] to match muGrid's (sub_pts, X, Y) convention
            res_quad = dF_dy[:, sq_x, sq_y]

            # Left triangle
            ele_points = [a_bl, a_tl, a_br]
            for (pt_inner, _, _, _), test_vals in zip(ele_points, test_fun_vals_left):
                if pt_inner not in self.indices_inner_local:
                    continue
                area = np.sum(test_vals * res_quad[0:3] * self.quad_weights) * self.A
                R[pt_inner] += area

            # Right triangle
            ele_points = [a_tr, a_br, a_tl]
            for (pt_inner, _, _, _), test_vals in zip(ele_points, test_fun_vals_right):
                if pt_inner not in self.indices_inner_local:
                    continue
                area = np.sum(test_vals * res_quad[3:6] * self.quad_weights) * self.A
                R[pt_inner] += area
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
        """Returns the inner nodal values of a field in shape (nb_inner_pts,).

        Note: Fields have shape (Nx_padded, Ny_padded) with [X, Y] indexing.
        The index_mask assigns nodes row-by-row in (Y, X) order, so we need
        Fortran order (column-major) to match: X varies fastest in the flattened array.
        """
        p = self.problem
        field_map = {
            'rho': p.q[0],
            'jx': p.q[1],
            'jy': p.q[2],
        }
        if self.energy:
            field_map['E'] = p.energy.energy
        # Extract inner region and flatten with Fortran order to match index_mask node ordering
        return field_map[field_name][1:-1, 1:-1].flatten(order='F')

    def get_q_nodal(self) -> NDArray:
        """Returns the full solution vector q in nodal values shape (nb_vars*nb_inner_pts,)."""
        q_nodal = np.zeros(self.res_size)
        for var in self.variables:
            q_nodal[self._sol_slice(var)] = self.get_nodal_val(var)
        return q_nodal

    def set_q_nodal(self, q_nodal: NDArray) -> None:
        """Sets the full solution vector q from nodal values shape (nb_vars*nb_inner_pts,).

        Note: Fields have shape (Nx_padded, Ny_padded) with [X, Y] indexing.
        The index_mask assigns nodes row-by-row in (Y, X) order, so we need
        Fortran order (column-major) to match: X varies fastest in the nodal array.
        """
        p = self.problem
        field_map = {
            'rho': p.q[0],
            'jx': p.q[1],
            'jy': p.q[2],
        }
        if self.energy:
            field_map['E'] = p.energy.energy

        for var in self.variables:
            var_nodal = q_nodal[self._sol_slice(var)]
            # Reshape with Fortran order to match index_mask node ordering
            # Field shape is (Nx_padded, Ny_padded)
            field_map[var][1:-1, 1:-1] = var_nodal.reshape((self.Nx_inner, self.Ny_inner), order='F')

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
