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
from .fem_1d.num_solver import Solver
from .fem_1d.utils import (
    NonLinearTerm,
    get_active_terms,
    get_norm_quad_pts,
    get_norm_quad_wts,
    print_matrix
)

from muGrid import GlobalFieldCollection
import numpy as np
import time

import numpy.typing as npt
from typing import TYPE_CHECKING, Tuple, Dict, Any
if TYPE_CHECKING:
    from .problem import Problem

NDArray = npt.NDArray[np.floating]


class FEMSolver1D:
    """1D FEM solver with centralized quadrature field management.

    All quadrature fields are stored in self.quad_fields dict with keys (field_name, nb_quad).
    Physical models only provide nodal data and JIT-compiled functions.
    """

    def __init__(self, problem: "Problem") -> None:
        self.problem = problem
        self.num_solver = Solver(problem.fem_solver)

        # Centralized quadrature field storage: {(field_name, nb_quad): muGrid_field}
        self.quad_fields: Dict[Tuple[str, int], Any] = {}

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_convenience_accessors(self) -> None:
        """Initialize convenience accessors for problem and grid properties."""
        p = self.problem

        self.periodic = p.grid['bc_xE'][0] == 'P'  # periodic in x
        self.energy = p.fem_solver['equations']['energy']
        self.dynamic = p.fem_solver['dynamic']

        self.nb_pts = p.grid['Nx']
        self.nb_ele = self.nb_pts if self.periodic else self.nb_pts - 1
        self.dx = p.grid['Lx'] / self.nb_ele

        if self.energy:
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

    def _get_quad_list(self, **kwargs) -> None:
        """Get list of occurring quadrature point numbers from active terms."""
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

    def _init_quad_field_storage(self) -> None:
        """Create all quadrature fields in fc_fem with centralized storage."""
        # Determine which fields are needed
        needed_fields = self._get_needed_quad_fields()

        for field_name in needed_fields:
            for nb_quad in self.quad_list:
                full_name = f'{field_name}_quad_{nb_quad}'
                field = self.fc_fem.real_field(full_name, nb_quad)
                self.quad_fields[(field_name, nb_quad)] = field

    def _get_needed_quad_fields(self) -> set:
        """Determine which quad fields are needed based on active terms and energy flag."""
        # Always needed: primary solution variables
        needed = {'rho', 'jx', 'jy'}

        # Nodal fields that need interpolation
        nodal_fields = {'p', 'h', 'dh_dx', 'dh_dy', 'eta'}

        # Constants (broadcast)
        constant_fields = {'U', 'V', 'Ls'}

        # Computed fields (pressure gradient)
        pressure_computed = {'dp_drho'}

        # Wall stress xz computed fields
        stress_xz_fields = {'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx',
                            'tau_xz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx'}

        # Wall stress yz computed fields (unused for now but kept for symmetry)
        _ = {'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy',
             'tau_yz_bot', 'dtau_yz_bot_drho', 'dtau_yz_bot_djy'}

        # Previous timestep fields
        prev_fields = {'rho_prev', 'jx_prev'}

        needed.update(nodal_fields)
        needed.update(constant_fields)
        needed.update(pressure_computed)
        needed.update(stress_xz_fields)
        needed.update(prev_fields)

        if self.energy:
            # Energy nodal fields
            energy_nodal = {'E', 'Tb_top', 'Tb_bot'}
            # Temperature computed fields
            temp_fields = {'T', 'dT_drho', 'dT_djx', 'dT_djy', 'dT_dE'}
            # Wall heat flux computed fields
            heat_flux_fields = {'S', 'dS_drho', 'dS_djx', 'dS_djy', 'dS_dE'}
            # Previous energy
            energy_prev = {'E_prev'}

            needed.update(energy_nodal)
            needed.update(temp_fields)
            needed.update(heat_flux_fields)
            needed.update(energy_prev)

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
        """Build term contexts with direct quad_fields references."""
        p = self.problem

        for term in self.terms:
            nbq = term.nb_quad_pts

            # Context directly references quad_fields via get_quad method
            term_ctx = {
                # Pressure
                'p': lambda nbq=nbq: self.get_quad('p', nbq),
                'dp_drho': lambda nbq=nbq: self.get_quad('dp_drho', nbq),

                # Wall stress xz
                'tau_xz': lambda nbq=nbq: self.get_quad('tau_xz', nbq),
                'dtau_xz_drho': lambda nbq=nbq: self.get_quad('dtau_xz_drho', nbq),
                'dtau_xz_djx': lambda nbq=nbq: self.get_quad('dtau_xz_djx', nbq),
                'tau_xz_bot': lambda nbq=nbq: self.get_quad('tau_xz_bot', nbq),
                'dtau_xz_bot_drho': lambda nbq=nbq: self.get_quad('dtau_xz_bot_drho', nbq),
                'dtau_xz_bot_djx': lambda nbq=nbq: self.get_quad('dtau_xz_bot_djx', nbq),

                # Topography
                'h': lambda nbq=nbq: self.get_quad('h', nbq),
                'dh_dx': lambda nbq=nbq: self.get_quad('dh_dx', nbq),
                'U': lambda nbq=nbq: self.get_quad('U', nbq),

                # Previous timestep
                'rho_prev': lambda nbq=nbq: self.get_quad('rho_prev', nbq),
                'jx_prev': lambda nbq=nbq: self.get_quad('jx_prev', nbq),
                'E_prev': lambda nbq=nbq: self.get_quad('E_prev', nbq),

                # Time step
                'dt': p.numerics['dt'],
            }

            # Energy terms (only if energy equation enabled)
            if self.energy:
                term_ctx.update({
                    'T': lambda nbq=nbq: self.get_quad('T', nbq),
                    'dT_drho': lambda nbq=nbq: self.get_quad('dT_drho', nbq),
                    'dT_djx': lambda nbq=nbq: self.get_quad('dT_djx', nbq),
                    'dT_dE': lambda nbq=nbq: self.get_quad('dT_dE', nbq),
                    'S': lambda nbq=nbq: self.get_quad('S', nbq),
                    'dS_drho': lambda nbq=nbq: self.get_quad('dS_drho', nbq),
                    'dS_djx': lambda nbq=nbq: self.get_quad('dS_djx', nbq),
                    'dS_dE': lambda nbq=nbq: self.get_quad('dS_dE', nbq),
                    'k': lambda: p.energy.k,
                })

            term.build(term_ctx)

    # =========================================================================
    # Quadrature Field Access and Update
    # =========================================================================

    def get_quad(self, name: str, nb_quad: int) -> NDArray:
        """Get quadrature field values."""
        return self.quad_fields[(name, nb_quad)].p

    def _inner_1d(self, field: NDArray) -> NDArray:
        """Extract inner 1D field from 2D field array with ghost cells."""
        assert field.shape[1] == 3, "Not a 1D problem: {}".format(field.shape)
        return field[1:-1, 1:-1].ravel()

    def update_nodal_fields(self) -> None:
        """Step 1: Update nodal data from physical models."""
        p = self.problem

        # Pressure from EOS
        p.pressure.update()

        # Topography (elastic deformation if enabled)
        p.topo.update()

        # Viscosity (piezoviscosity + shear-thinning)
        # Need pressure gradient for shear-thinning
        dp_dx = np.gradient(p.pressure.pressure, p.grid['dx'], axis=0)
        dp_dy = np.gradient(p.pressure.pressure, p.grid['dy'], axis=1)
        p.viscosity.update(p.pressure.pressure, dp_dx, dp_dy, p.topo.h, p.geo['U'], p.geo['V'])

        # Energy temperature
        if self.energy:
            p.energy.update_temperature()

    def update_quad_nodal(self) -> None:
        """Step 2: Interpolate nodal fields to quadrature points."""
        p = self.problem

        # Map of field names to their nodal data sources
        nodal_sources = {
            'rho': self._inner_1d(p.q[0]),
            'jx': self._inner_1d(p.q[1]),
            'jy': self._inner_1d(p.q[2]),
            'p': self._inner_1d(p.pressure.pressure),
            'h': self._inner_1d(p.topo.h),
            'dh_dx': self._inner_1d(p.topo.dh_dx),
            'dh_dy': self._inner_1d(p.topo.dh_dy),
            'eta': self._inner_1d(p.viscosity.eta),
        }

        if self.energy:
            nodal_sources.update({
                'E': self._inner_1d(p.energy.energy),
                'Tb_top': self._inner_1d(p.energy.Tb_top),
                'Tb_bot': self._inner_1d(p.energy.Tb_bot),
            })

        # Constants
        Ls = p.prop.get('slip_length', 0.0)
        constants = {
            'U': p.geo['U'],
            'V': p.geo['V'],
            'Ls': Ls,
        }

        for nb_quad in self.quad_list:
            # Interpolate nodal fields
            for name, nodal_vals in nodal_sources.items():
                if (name, nb_quad) in self.quad_fields:
                    quad_vals = self.quad_fun(nodal_vals, nb_quad)
                    self.quad_fields[(name, nb_quad)].p[:] = quad_vals.reshape(-1, nb_quad).T

            # Broadcast constants
            for name, value in constants.items():
                if (name, nb_quad) in self.quad_fields:
                    self.quad_fields[(name, nb_quad)].p[:] = value

    def update_quad_computed(self) -> None:
        """Step 3: Compute derived quantities at quadrature points."""
        p = self.problem

        for nb_quad in self.quad_list:
            # Helper to get quad field
            def q(name):
                return self.quad_fields[(name, nb_quad)].p

            # Pressure gradient: dp_drho(rho)
            if ('dp_drho', nb_quad) in self.quad_fields:
                self.quad_fields[('dp_drho', nb_quad)].p[:] = p.pressure.dp_drho(q('rho'))

            # Wall stress xz
            # Args: rho, jx, jy, h, hx, U, V, Ls
            args_xz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dx'),
                       q('U'), q('V'), q('Ls'))

            if ('tau_xz', nb_quad) in self.quad_fields:
                self.quad_fields[('tau_xz', nb_quad)].p[:] = p.wall_stress_xz.tau_xz(*args_xz)
            if ('dtau_xz_drho', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_xz_drho', nb_quad)].p[:] = p.wall_stress_xz.dtau_xz_drho(*args_xz)
            if ('dtau_xz_djx', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_xz_djx', nb_quad)].p[:] = p.wall_stress_xz.dtau_xz_djx(*args_xz)
            if ('tau_xz_bot', nb_quad) in self.quad_fields:
                self.quad_fields[('tau_xz_bot', nb_quad)].p[:] = p.wall_stress_xz.tau_xz_bot(*args_xz)
            if ('dtau_xz_bot_drho', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_xz_bot_drho', nb_quad)].p[:] = p.wall_stress_xz.dtau_xz_bot_drho(*args_xz)
            if ('dtau_xz_bot_djx', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_xz_bot_djx', nb_quad)].p[:] = p.wall_stress_xz.dtau_xz_bot_djx(*args_xz)

            # Wall stress yz (uses dh_dy instead of dh_dx, and U is constant instead of V)
            # Args: rho, jx, jy, h, hy, U, V, Ls
            args_yz = (q('rho'), q('jx'), q('jy'), q('h'), q('dh_dy'),
                       q('U'), q('V'), q('Ls'))

            if ('tau_yz', nb_quad) in self.quad_fields:
                self.quad_fields[('tau_yz', nb_quad)].p[:] = p.wall_stress_yz.tau_yz(*args_yz)
            if ('dtau_yz_drho', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_yz_drho', nb_quad)].p[:] = p.wall_stress_yz.dtau_yz_drho(*args_yz)
            if ('dtau_yz_djy', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_yz_djy', nb_quad)].p[:] = p.wall_stress_yz.dtau_yz_djy(*args_yz)
            if ('tau_yz_bot', nb_quad) in self.quad_fields:
                self.quad_fields[('tau_yz_bot', nb_quad)].p[:] = p.wall_stress_yz.tau_yz_bot(*args_yz)
            if ('dtau_yz_bot_drho', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_yz_bot_drho', nb_quad)].p[:] = p.wall_stress_yz.dtau_yz_bot_drho(*args_yz)
            if ('dtau_yz_bot_djy', nb_quad) in self.quad_fields:
                self.quad_fields[('dtau_yz_bot_djy', nb_quad)].p[:] = p.wall_stress_yz.dtau_yz_bot_djy(*args_yz)

            # Energy fields
            if self.energy:
                # Temperature: T(rho, jx, jy, E)
                args_T = (q('rho'), q('jx'), q('jy'), q('E'))

                if ('T', nb_quad) in self.quad_fields:
                    self.quad_fields[('T', nb_quad)].p[:] = p.energy.T_func(*args_T)
                if ('dT_drho', nb_quad) in self.quad_fields:
                    self.quad_fields[('dT_drho', nb_quad)].p[:] = p.energy.T_grad_rho(*args_T)
                if ('dT_djx', nb_quad) in self.quad_fields:
                    self.quad_fields[('dT_djx', nb_quad)].p[:] = p.energy.T_grad_jx(*args_T)
                if ('dT_djy', nb_quad) in self.quad_fields:
                    self.quad_fields[('dT_djy', nb_quad)].p[:] = p.energy.T_grad_jy(*args_T)
                if ('dT_dE', nb_quad) in self.quad_fields:
                    self.quad_fields[('dT_dE', nb_quad)].p[:] = p.energy.T_grad_E(*args_T)

                # Wall heat flux (constants captured in closure, only arrays passed)
                args_S = (q('h'), q('eta'), q('rho'), q('E'), q('jx'), q('jy'),
                          q('U'), q('V'), q('Tb_top'), q('Tb_bot'))

                if ('S', nb_quad) in self.quad_fields:
                    self.quad_fields[('S', nb_quad)].p[:] = p.energy.q_wall_sum(*args_S)
                if ('dS_drho', nb_quad) in self.quad_fields:
                    self.quad_fields[('dS_drho', nb_quad)].p[:] = p.energy.q_wall_grad_rho(*args_S)
                if ('dS_djx', nb_quad) in self.quad_fields:
                    self.quad_fields[('dS_djx', nb_quad)].p[:] = p.energy.q_wall_grad_jx(*args_S)
                if ('dS_djy', nb_quad) in self.quad_fields:
                    self.quad_fields[('dS_djy', nb_quad)].p[:] = p.energy.q_wall_grad_jy(*args_S)
                if ('dS_dE', nb_quad) in self.quad_fields:
                    self.quad_fields[('dS_dE', nb_quad)].p[:] = p.energy.q_wall_grad_E(*args_S)

    def update_quad(self) -> None:
        """Full quadrature update (Steps 1-3)."""
        self.update_nodal_fields()
        self.update_quad_nodal()
        self.update_quad_computed()

    def update_prev_quad(self) -> None:
        """Store current quad values for time derivatives."""
        for nb_quad in self.quad_list:
            for var in self.variables:
                curr_key = (var, nb_quad)
                prev_key = (f'{var}_prev', nb_quad)
                if curr_key in self.quad_fields and prev_key in self.quad_fields:
                    self.quad_fields[prev_key].p[:] = np.copy(self.quad_fields[curr_key].p)

    # =========================================================================
    # Solution Vector Management
    # =========================================================================

    def _res_slice(self, res_name) -> slice:
        i = self.residuals.index(res_name)
        return slice(i * self.nb_pts, (i + 1) * self.nb_pts)

    def _var_slice(self, var_name) -> slice:
        i = self.variables.index(var_name)
        return slice(i * self.nb_pts, (i + 1) * self.nb_pts)

    def _block_slices(self, res_name, var_name) -> tuple[slice, slice]:
        return (self._res_slice(res_name), self._var_slice(var_name))

    def get_nodal_val(self, field_name: str) -> NDArray:
        """Returns the nodal values of a field in shape (nb_pts,)."""
        p = self.problem
        field_map = {
            'rho': p.q[0],
            'jx': p.q[1],
            'jy': p.q[2],
        }
        if self.energy:
            field_map['E'] = p.energy.energy
        return self._inner_1d(field_map[field_name])

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
        p = self.problem
        field_map = {
            'rho': p.q[0],
            'jx': p.q[1],
            'jy': p.q[2],
        }
        if self.energy:
            field_map['E'] = p.energy.energy

        for var in self.variables:
            var_slice = self._var_slice(var)
            var_nodal = q_nodal[var_slice]
            field_map[var][1:-1, 1:-1] = var_nodal.reshape((self.nb_pts, 1))

    # =========================================================================
    # Shape Functions and Quadrature
    # =========================================================================

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

    def get_quad_deriv(self, field_name: str, nb_quad: int) -> NDArray:
        """Returns the quadrature derivative values of a field in shape (nb_quad, nb_ele)."""
        vec = self.get_nodal_val(field_name)
        return self._get_quad_deriv(vec, nb_quad)

    # =========================================================================
    # Matrix and Residual Assembly
    # =========================================================================

    def tang_matrix_term(self, term: NonLinearTerm, dep_var: str) -> NDArray:
        M = np.zeros((self.nb_pts, self.nb_pts))

        if not term.d_dx_testfun:
            N1_quad_w, N2_quad_w = self.get_N_quad_w(term.nb_quad_pts)
            N1_quad, N2_quad = self.get_N_quad(term.nb_quad_pts)
        else:
            N1_quad_w, N2_quad_w = self.get_dN_dx_quad_w(term.nb_quad_pts)

        dep_var_vals = {dep_var: self.get_quad(dep_var, term.nb_quad_pts)
                        for dep_var in term.dep_vars}
        res_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
        res_deriv_vals = res_deriv_vals.T  # get shape (nb_ele, nb_quad_pts)

        if not term.d_dx_resfun:  # res = f(a)
            for pt_test in range(self.nb_pts):
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

                    if rel_element == -1:
                        M[pt_mid, pt_mid] += np.sum(N2_quad_w * res_deriv_local * N2_quad) * dx_ele
                        M[pt_mid, pt_left] += np.sum(N2_quad_w * res_deriv_local * N1_quad) * dx_ele
                    else:
                        M[pt_mid, pt_mid] += np.sum(N1_quad_w * res_deriv_local * N1_quad) * dx_ele
                        M[pt_mid, pt_right] += np.sum(N1_quad_w * res_deriv_local * N2_quad) * dx_ele

        if term.d_dx_resfun:  # res = df/da * da/dx
            for pt_test in range(self.nb_pts):
                pt_mid = pt_test % self.nb_pts
                pt_left = (pt_test - 1) % self.nb_pts
                pt_right = (pt_test + 1) % self.nb_pts

                for rel_element in [-1, 0]:
                    element = pt_test + rel_element

                    if not self.periodic and (element < 0 or element == self.nb_ele):
                        continue

                    res_deriv_local = res_deriv_vals[element % self.nb_ele, :]

                    if term.d_dx_testfun:
                        N1_quad_w_scaled = np.copy(N1_quad_w) / self.dx
                        N2_quad_w_scaled = np.copy(N2_quad_w) / self.dx
                    else:
                        N1_quad_w_scaled = N1_quad_w
                        N2_quad_w_scaled = N2_quad_w

                    if rel_element == -1:
                        M[pt_mid, pt_mid] += np.sum(N2_quad_w_scaled * res_deriv_local * 1.0)
                        M[pt_mid, pt_left] += np.sum(N2_quad_w_scaled * res_deriv_local * -1.0)
                    else:
                        M[pt_mid, pt_mid] += np.sum(N1_quad_w_scaled * res_deriv_local * -1.0)
                        M[pt_mid, pt_right] += np.sum(N1_quad_w_scaled * res_deriv_local * 1.0)
        return M

    def residual_vector_term(self, term: NonLinearTerm) -> NDArray:
        res_vec = np.zeros(self.nb_pts)

        dep_var_vals = {dep_var: self.get_quad(dep_var, term.nb_quad_pts)
                        for dep_var in term.dep_vars}

        if not term.d_dx_resfun:
            res_fun_vals = term.evaluate(*[dep_var_vals[dep_var] for dep_var in term.dep_vars])
            res_fun_vals = res_fun_vals.T
        else:
            res_fun_vals = np.zeros((self.nb_ele, term.nb_quad_pts))
            for dep_var in term.dep_vars:
                fun_deriv_vals = term.evaluate_deriv(dep_var, *[dep_var_vals[dep_var] for dep_var in term.dep_vars])
                dep_var_deriv_vals = self.get_quad_deriv(dep_var, term.nb_quad_pts)
                res_fun_vals += (fun_deriv_vals * dep_var_deriv_vals).T

        if not term.d_dx_testfun:
            N1_quad_w, N2_quad_w = self.get_N_quad_w(term.nb_quad_pts)
        else:
            N1_quad_w, N2_quad_w = self.get_dN_dx_quad_w(term.nb_quad_pts)

        assert res_fun_vals.shape == (self.nb_ele, term.nb_quad_pts)

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

        # Density (mass) BC
        if p.grid['bc_xW'][0] == 'D':
            M[0, :] = 0.0
            M[0, 0] = 1.0
        if p.grid['bc_xE'][0] == 'D':
            M[2 * self.nb_pts - 1, :] = 0.0
            M[2 * self.nb_pts - 1, self.nb_pts - 1] = 1.0

        # Energy BC
        if self.energy:
            E_row_W = 2 * self.nb_pts
            E_row_E = 3 * self.nb_pts - 1
            E_col_W = 2 * self.nb_pts
            E_col_E = 3 * self.nb_pts - 1

            if p.energy.bc_xW == 'D':
                M[E_row_W, :] = 0.0
                M[E_row_W, E_col_W] = 1.0
            if p.energy.bc_xE == 'D':
                M[E_row_E, :] = 0.0
                M[E_row_E, E_col_E] = 1.0

        return M

    def boundary_condition_R(self, R: NDArray) -> NDArray:
        p = self.problem

        # Density (mass) BC
        if p.grid['bc_xW'][0] == 'D':
            target = p.grid['bc_xW_D_val']
            guess = self.get_nodal_val('rho')[0]
            R[0] = guess - target
        if p.grid['bc_xE'][0] == 'D':
            target = p.grid['bc_xE_D_val']
            guess = self.get_nodal_val('rho')[-1]
            R[2 * self.nb_pts - 1] = guess - target

        # Energy BC
        if self.energy:
            E_row_W = 2 * self.nb_pts
            E_row_E = 3 * self.nb_pts - 1

            if p.energy.bc_xW == 'D':
                E_target = self._compute_E_from_T(p.energy.T_bc_xW, idx=0)
                E_guess = self.get_nodal_val('E')[0]
                R[E_row_W] = E_guess - E_target
            if p.energy.bc_xE == 'D':
                E_target = self._compute_E_from_T(p.energy.T_bc_xE, idx=-1)
                E_guess = self.get_nodal_val('E')[-1]
                R[E_row_E] = E_guess - E_target

        return R

    def _compute_E_from_T(self, T_bc: float, idx: int) -> float:
        """Compute target energy from temperature BC value."""
        p = self.problem
        rho = self.get_nodal_val('rho')[idx]
        jx = self.get_nodal_val('jx')[idx]
        ux = jx / rho
        kinetic = 0.5 * ux**2
        return rho * (p.energy.cv * T_bc + kinetic)

    def get_M(self) -> NDArray:
        M = self.get_tang_matrix()
        M = self.boundary_condition_M(M)
        return M

    def get_R(self) -> NDArray:
        R = self.get_residual_vec()
        R = self.boundary_condition_R(R)
        return R

    # =========================================================================
    # Solver Interface
    # =========================================================================

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

        # Centralized quad field initialization
        self._init_quad_field_storage()

        # Build JIT functions
        self._build_jit_functions()

        # Build term contexts
        self._build_terms()

        # Initial quad update
        self.update_quad()

        if self.dynamic:
            self.update_prev_quad()

        # Update output fields for initial frame
        self.update_output_fields()

        self.time_inner = 0.0

        print("FEM Solver 1D initialized with centralized quad field management:")
        print(f"  Periodic: {self.periodic}, Energy: {self.energy}, Dynamic: {self.dynamic}")
        print(f"  Quad fields: {len(self.quad_fields)} total")

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

    def update_output_fields(self) -> None:
        """Update nodal output fields (wall stress, bulk stress) for plotting/output."""
        p = self.problem
        p.wall_stress_xz.update()
        p.wall_stress_yz.update()
        p.bulk_stress.update()

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
        p = self.problem
        if p.options['print_progress'] and self.dynamic:
            print(61 * '-')
            print(f"{'Step':<6s} {'Timestep':<12s} {'Time':<12s} {'Convergence Time':<18s} {'Residual':<12s}")
            print(61 * '-')
        if p.options['save_output']:
            p.write(params=False)

    def print_status(self, scalars=None) -> None:
        p = self.problem
        if p.options['print_progress'] and self.dynamic:
            print(f"{p.step:<6d} {p.dt:<12.4e} {p.simtime:<12.4e} {self.time_inner:<18.4e} {p.residual:<12.4e}")
