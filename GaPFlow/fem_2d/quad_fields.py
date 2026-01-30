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
"""Quadrature field management for FEM assembly.

Handles field storage, interpolation operators, update methods, and access
for nodal and quadrature point fields on triangular elements.
"""
from typing import Any, Dict, List, Set, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from muGrid import Field

from .elements import TriangleQuadrature

if TYPE_CHECKING:
    from ..problem import Problem

NDArray = npt.NDArray[np.floating]


# Field name sets for different physics
BASE_FIELDS = {
    'rho', 'jx', 'jy',
    'p', 'h', 'dh_dx', 'dh_dy', 'eta',
    'U', 'V', 'Ls',
    'dp_drho',
    'rho_prev', 'jx_prev', 'jy_prev',
    'tau_mass',
    'tau_mom',
    'force_x', 'force_y',  # Body force components
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
    'tau_energy',
}


class QuadFieldManager:
    """Manages quadrature fields, operators, and updates for FEM.

    Handles:
    - Field storage (nodal and quadrature point fields)
    - Interpolation operators (nodal → quad, derivatives)
    - Field access methods (get values, compute derivatives)
    - Physics updates (nodal fields, interpolation, derived quantities)
    - Time stepping (store previous values)

    Parameters
    ----------
    problem : Problem
        Problem instance for physics model access.
    energy : bool
        Whether energy equation is active.
    variables : list of str
        Variable names for time stepping (e.g., ['rho', 'jx', 'jy']).
    """

    def __init__(self, problem: "Problem", energy: bool, variables: List[str]):
        self.problem = problem
        self.energy = energy
        self.variables = variables

        # Grid spacing from problem
        self.dx = problem.grid['dx']
        self.dy = problem.grid['dy']

        # Field storage
        self.nodal_fields: Dict[str, Any] = {}
        self.quad_fields: Dict[str, Any] = {}

        # Cache for boundary distance (computed on first use)
        self._boundary_distance_cache = None

        # Initialize operators and fields
        self._init_operators()
        self._init_fields()

    def _init_operators(self) -> None:
        """Initialize interpolation and derivative operators."""
        quad = TriangleQuadrature(self.dx, self.dy)
        self.quad_operator = quad.interpolation_operator
        self.dx_operator = quad.dx_operator
        self.dy_operator = quad.dy_operator

    def _init_fields(self) -> None:
        """Initialize nodal and quadrature point fields."""
        fc = self.problem.fc
        fc.set_nb_sub_pts("quad", 6)
        fc.set_nb_sub_pts("quad_deriv", 2)

        # Multi-component sources need single-component fields for interpolation
        nodal_names = ['rho', 'jx', 'jy', 'h', 'dh_dx', 'dh_dy']
        if self.energy:
            nodal_names.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in nodal_names:
            self.nodal_fields[name] = fc.real_field(f'{name}_nodal', 1, 'pixel')

        # Existing single-component fields (pressure, viscosity)
        self.nodal_fields['p'] = Field(fc.get_real_field('pressure'))
        self.nodal_fields['eta'] = Field(fc.get_real_field('shear_viscosity'))

        # Quadrature output fields
        for name in self.get_needed_fields():
            self.quad_fields[name] = fc.real_field(f'{name}_q', 1, 'quad')

        # Placeholder for derivative computation (reused)
        self.deriv_placeholder = fc.real_field('deriv_placeholder', 1, 'quad_deriv')

    def get_needed_fields(self) -> Set[str]:
        """Get set of needed quadrature field names based on physics."""
        needed = BASE_FIELDS | STRESS_XZ_FIELDS | STRESS_YZ_FIELDS
        if self.energy:
            needed |= ENERGY_FIELDS
        return needed

    # =========================================================================
    # Field Access
    # =========================================================================

    def get(self, name: str) -> NDArray:
        """Get quadrature field values for internal squares.

        Returns shape (6, sq_per_row, sq_per_col).
        """
        return self.quad_fields[name].pg[..., :-1, :-1]

    def get_deriv_dx(self, name: str) -> NDArray:
        """Compute d(field)/dx at quad points.

        Returns shape (2, sq_per_row, sq_per_col).
        """
        self.dx_operator.apply(self.nodal_fields[name], self.deriv_placeholder)
        return self.deriv_placeholder.pg[..., :-1, :-1].copy()

    def get_deriv_dy(self, name: str) -> NDArray:
        """Compute d(field)/dy at quad points.

        Returns shape (2, sq_per_row, sq_per_col).
        """
        self.dy_operator.apply(self.nodal_fields[name], self.deriv_placeholder)
        return self.deriv_placeholder.pg[..., :-1, :-1].copy()

    def interpolate_nodal_to_quad(self, name: str) -> None:
        """Interpolate a single nodal field to quadrature points."""
        self.quad_operator.apply(self.nodal_fields[name], self.quad_fields[name])

    # =========================================================================
    # Field Updates (physics-dependent)
    # =========================================================================

    def update_nodal_fields(self) -> None:
        """Update nodal fields from problem state (pressure, viscosity, etc.)."""
        p = self.problem
        timer = p.solver.timer

        p.pressure.update()
        with timer("topography_update"):
            p.topo.update()
        dp_dx = np.gradient(p.pressure.pressure, p.grid['dx'], axis=0)
        dp_dy = np.gradient(p.pressure.pressure, p.grid['dy'], axis=1)
        with timer("viscosity_update"):
            p.viscosity.update(p.pressure.pressure, dp_dx, dp_dy,
                               p.topo.h, p.geo['U'], p.geo['V'])
        if self.energy:
            p.energy.update_temperature()

    def update_quad_nodal(self) -> None:
        """Copy problem state to nodal fields and interpolate to quad points."""
        p = self.problem

        # Copy multi-component sources to single-component fields
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

        # Interpolate nodal → quad using operator
        interpolated = ['rho', 'jx', 'jy', 'p', 'h', 'dh_dx', 'dh_dy', 'eta']
        if self.energy:
            interpolated.extend(['E', 'Tb_top', 'Tb_bot'])
        for name in interpolated:
            self.interpolate_nodal_to_quad(name)

        # Broadcast constants
        self.quad_fields['U'].pg[:] = p.geo['U']
        self.quad_fields['V'].pg[:] = p.geo['V']
        self.quad_fields['Ls'].pg[:] = p.prop.get('slip_length', 0.0)

    def _apply_2d_vmap(self, func, *args):
        """Reshape args for 2D vmap application and reshape result back."""
        shape = args[0].shape  # (6, Ny, Nx)
        args_2d = [a.reshape(shape[0], -1) for a in args]
        result_2d = func(*args_2d)
        return result_2d.reshape(shape)

    def _compute_boundary_distance(self, target_shape: tuple) -> NDArray:
        """Compute distance (in grid cells) from each quad point to nearest Dirichlet boundary.

        Parameters
        ----------
        target_shape : tuple
            Target shape (6, nx, ny) to match the quad field slice shape.

        Returns array of shape target_shape representing distance in cell units.
        Points away from any Dirichlet boundary get a large value.
        """
        p = self.problem
        grid = p.grid
        decomp = p.decomp

        Lx, Ly = grid['Lx'], grid['Ly']
        h = np.sqrt(self.dx * self.dy)  # characteristic element size

        # Grid dimensions from target shape
        _, nx, ny = target_shape

        # Physical coordinates for each grid point
        x_coords = np.arange(nx) * self.dx
        y_coords = np.arange(ny) * self.dy

        # Broadcast to (nx, ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        # Distance to each boundary (in grid cells)
        # Only consider non-periodic boundaries (Dirichlet)
        # Use full_like to ensure consistent array shapes
        dist_W = X / h if not decomp.periodic_x else np.full_like(X, np.inf)
        dist_E = (Lx - X) / h if not decomp.periodic_x else np.full_like(X, np.inf)
        dist_S = Y / h if not decomp.periodic_y else np.full_like(Y, np.inf)
        dist_N = (Ly - Y) / h if not decomp.periodic_y else np.full_like(Y, np.inf)

        # Minimum distance to any Dirichlet boundary
        dist_boundary = np.minimum(np.minimum(dist_W, dist_E), np.minimum(dist_S, dist_N))

        # Broadcast to (6, nx, ny) for all quad points
        return np.broadcast_to(dist_boundary[np.newaxis, :, :], target_shape).copy()

    def _compute_tau_stabilization(self, s, q) -> None:
        """Compute stabilization parameters at quadrature points.

        Uses user-specified alpha values as tau, with P0 normalization for
        pressure/energy and optional boundary enhancement for Dirichlet BCs.
        """
        p = self.problem
        rho = q('rho')
        P0 = p.prop.get('P0', 1.0)

        # Base tau values: pressure/energy normalized by P0, momentum direct
        tau_mass_base = p.fem_solver['pressure_stab_alpha'] / P0
        tau_mom_base = p.fem_solver['momentum_stab_alpha']

        # Optional boundary enhancement
        boundary_factor = p.fem_solver.get('boundary_stab_factor', 1.0)
        boundary_decay = p.fem_solver.get('boundary_stab_decay', 2.0)

        if boundary_factor > 1.0 and (not p.decomp.periodic_x or not p.decomp.periodic_y):
            if self._boundary_distance_cache is None:
                self._boundary_distance_cache = self._compute_boundary_distance(rho.shape)
            dist = self._boundary_distance_cache
            enhancement = 1.0 + (boundary_factor - 1.0) * np.exp(-dist / boundary_decay)
            tau_mass = tau_mass_base * enhancement
            tau_mom = tau_mom_base * enhancement
        else:
            tau_mass = np.full_like(rho, tau_mass_base)
            tau_mom = np.full_like(rho, tau_mom_base)

        self.quad_fields['tau_mass'].pg[s] = tau_mass
        self.quad_fields['tau_mom'].pg[s] = tau_mom

        # Energy stabilization (if enabled)
        if self.energy:
            tau_energy = p.fem_solver['energy_stab_alpha']
            self.quad_fields['tau_energy'].pg[s] = np.full_like(rho, tau_energy)

    def update_quad_computed(self) -> None:
        """Compute derived quantities at quadrature points.

        Only computes on interior cells (excludes last row/column) where
        interpolation produced valid data. Ghost cells are left as zero.
        """
        p = self.problem
        # Interior slice accessor (excludes ghost cells at x=Nx+1, y=Ny+1)
        s = np.s_[..., :-1, :-1]
        q = lambda name: self.quad_fields[name].pg[s]
        apply = self._apply_2d_vmap

        # Pressure gradient
        self.quad_fields['dp_drho'].pg[s] = apply(p.pressure.dp_drho, q('rho'))

        # Stabilization parameter computation
        self._compute_tau_stabilization(s, q)

        # Wall stress xz
        args_xz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dx'),
                   q('U'), q('V'), q('Ls'))
        for name in ['tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
                     'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx']:
            self.quad_fields[name].pg[s] = apply(
                getattr(p.wall_stress_xz, name), *args_xz)

        # Wall stress yz
        args_yz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dy'),
                   q('U'), q('V'), q('Ls'))
        for name in ['tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
                     'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy']:
            self.quad_fields[name].pg[s] = apply(
                getattr(p.wall_stress_yz, name), *args_yz)

        # Body force (constant fields from properties, default 0)
        force_x = p.prop.get('force_x', 0.0)
        force_y = p.prop.get('force_y', 0.0)
        self.quad_fields['force_x'].pg[s] = np.full_like(q('rho'), force_x)
        self.quad_fields['force_y'].pg[s] = np.full_like(q('rho'), force_y)

        if self.energy:
            # Temperature
            args_T = (q('rho'), q('jx'), q('jy'), q('E'))
            for name, func in [('T', 'T_func'), ('dT_drho', 'T_grad_rho'),
                               ('dT_djx', 'T_grad_jx'), ('dT_djy', 'T_grad_jy'),
                               ('dT_dE', 'T_grad_E')]:
                self.quad_fields[name].pg[s] = getattr(p.energy, func)(*args_T)

            # Wall heat flux (constants captured in closure, only arrays passed)
            args_S = (q('h'), q('eta'), q('rho'), q('E'), q('jx'), q('jy'),
                      q('U'), q('V'), q('Tb_top'), q('Tb_bot'))
            for name, func in [('S', 'q_wall_sum'), ('dS_drho', 'q_wall_grad_rho'),
                               ('dS_djx', 'q_wall_grad_jx'),
                               ('dS_djy', 'q_wall_grad_jy'),
                               ('dS_dE', 'q_wall_grad_E')]:
                self.quad_fields[name].pg[s] = apply(
                    getattr(p.energy, func), *args_S)

    def store_prev_values(self) -> None:
        """Store current quad values for time derivatives."""
        for var in self.variables:
            curr_key = var
            prev_key = f'{var}_prev'
            if curr_key in self.quad_fields and prev_key in self.quad_fields:
                self.quad_fields[prev_key].pg[:] = np.copy(
                    self.quad_fields[curr_key].pg)
