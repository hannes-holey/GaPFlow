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

from typing import Dict, List, Set, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from muGrid import Field
from scipy.ndimage import zoom

from .elements import TaylorHoodQ2Q1
from .fieldspec import NODAL_P1, NODAL_P2, QUAD_FIELD_REGISTRY, resolve_source, categorize_registry_fields
from .terms import collect_required_fields


if TYPE_CHECKING:
    from ..problem import Problem
    from ..parallel import DomainDecomposition

NDArray = npt.NDArray[np.floating]


class QuadFieldManager:
    """Manages nodal and quadrature fields for Taylor-Hood Q2Q1 assembly.

    Parameters
    ----------
    problem : Problem
        Problem instance (coarse-grid state, physics models).
    energy : bool
        Whether the energy equation is active.
    variables : list of str
        Newton variables in order, e.g. ['jx', 'jy', 'rho'] or with 'E'.
    elements : TaylorHoodQ2Q1
        Element instance carrying P1 and P2 operators.
    decomp : DomainDecomposition
        Decomposition holding both coarse (_p) and fine (_P2) grid info.
    """

    def __init__(self, problem: "Problem",
                 energy: bool,
                 cavitation: bool,
                 variables: List[str],
                 add_fields: List[str],
                 elements: TaylorHoodQ2Q1,
                 decomp: "DomainDecomposition",
                 terms: list) -> None:

        self.problem = problem
        self.energy = energy
        self.cavitation = cavitation
        self.variables = variables
        self.add_fields = add_fields
        self.elements = elements
        self.decomp = decomp
        self.terms = terms

        self.dx = problem.grid['dx']
        self.dy = problem.grid['dy']

        self.nodal_fields: Dict[str, Field] = {}
        self.quad_fields: Dict[str, Field] = {}

        self.nf = lambda name: self.nodal_fields[name].pg[0]
        self.qf = lambda name: self.quad_fields[name].pg

        self._init_fields()

    # =========================================================================
    # Initialisation
    # =========================================================================

    def _init_fields(self) -> None:
        """Initialize nodal and quadrature fields based on active terms."""

        fc = self.decomp.fc
        self._reg_nodal, self._reg_scalar, self._reg_computed = categorize_registry_fields()
        self.nodal_field_keys, self.quad_field_keys, self.der_field_keys = self._needed_fields()

        # ---- P1 nodal fields ----
        for name in NODAL_P1 + self.add_fields + list(self.nodal_field_keys):
            self.nodal_fields[name] = fc.real_field(f'{name}_nodal', 1, 'pixel')

        # ---- P2 nodal fields ----
        fc_P2 = self.decomp.fc_P2
        for name in NODAL_P2:
            if name in self.variables:
                self.nodal_fields[name] = fc_P2.real_field(f'{name}_nodal', 1, 'pixel')

        # ---- Quadrature fields ----
        nb_quad_sq = self.elements.Quadrature.nb_points
        fc.set_nb_sub_pts('quad', nb_quad_sq)

        for name in self.quad_field_keys | set(self.variables):
            self.quad_fields[name] = fc.real_field(f'{name}_q', 1, 'quad')

        self._deriv_placeholder = fc.real_field('deriv_placeholder', 1, 'quad')

    def _add_dependent_fields(self, keys: Set[str]) -> Set[str]:
        """Recursively add dependent fields for computed fields in keys.
        E.g. U_bot is not directly needed by any term, but is an argument for tau_xz."""

        result = set(keys)
        frontier = set(keys)
        while frontier:
            new = set()
            for name in frontier:
                entry = QUAD_FIELD_REGISTRY.get(name)
                if entry and entry['type'] == 'computed':
                    new |= {a for a in entry.get('args', []) if a not in result}
            result |= new
            frontier = new
        return result

    def _needed_fields(self) -> tuple[Set[str], Set[str], Set[str]]:
        """Collect all keys of fields to initialize.
        Nodal fields are always in the registry while the derivatives,
        which are part of the overall quad fields, are not."""

        plain, der = collect_required_fields(self.terms)
        plain = self._add_dependent_fields(plain)

        nodal_fields = {name for name in plain if name in self._reg_nodal}
        quad_fields = plain | der

        return nodal_fields, quad_fields, der

    # =========================================================================
    # Field access  (assembly calls these)
    # transpose from (nb_quad_sq, sq_per_row, sq_per_col) to (nb_sq, nb_quad_sq)
    # =========================================================================

    def _element_for(self, name: str):
        if name in NODAL_P2:
            return self.elements.Q2
        return self.elements.Q1

    def _deriv_pg(self, name: str, axis: str) -> NDArray:
        """Derivative of nodal field `name` along `axis` ('x' or 'y') at quad points.
        Returns raw pg shape (nb_quad_sq, Nx-1, Ny-1), trimmed to inner squares."""
        el = self._element_for(name)
        getattr(el, f'd{axis}_operator').apply(self.nodal_fields[name], self._deriv_placeholder)
        return self._deriv_placeholder.pg[..., :-1, :-1] / getattr(self, f'd{axis}')

    def get_quad_sq(self, name: str) -> NDArray:
        """Quadrature values for all squares.
        Returns shape (nb_sq, nb_quad_sq): squares x-major, quad innermost.
        """
        sq = self.quad_fields[name].pg[:, :-1, :-1]
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def get_quad_dx_sq(self, name: str) -> NDArray:
        """d(field)/dx at quad points, shape (nb_sq, nb_quad_sq)."""
        sq = self._deriv_pg(name, 'x')
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def get_quad_dy_sq(self, name: str) -> NDArray:
        """d(field)/dy at quad points, shape (nb_sq, nb_quad_sq)."""
        sq = self._deriv_pg(name, 'y')
        return sq.transpose(2, 1, 0).reshape(-1, sq.shape[0])

    def interpolate_nodal_to_quad(self, name: str) -> None:
        """Interpolate a single nodal field to its quad output field.
        No return, but updates self.quad_fields[name] in-place."""
        el = self._element_for(name)
        el.interpolation_operator.apply(self.nodal_fields[name], self.quad_fields[name])

    # =========================================================================
    # Newton scatter / gather  (solver calls these)
    # =========================================================================

    def get_nodal_sol_val(self, var: str) -> NDArray:
        """Extract inner nodal values as a flat vector for the Newton solve.
        """
        return self.nodal_fields[var].p[0].flatten(order='F')

    def set_nodal_sol_val(self, var: str, values: NDArray) -> None:
        """Scatter Newton solution vector back to the nodal field inner region.
        """
        self.nodal_fields[var].p[0] = values.reshape(
            self.nodal_fields[var].p[0].shape, order='F')

    # =========================================================================
    # Sync gate: fine-grid jx/jy → problem.q  (for output / BCs)
    # =========================================================================

    def sync_to_problem_q(self) -> None:
        """Sync FEM-owned nodal fields back to problem.q."""
        p = self.problem
        p.q[0] = self.nf('rho')
        base_slots = len(self.variables) - len(self.add_fields)
        if 'jx' in self.variables:
            p.q[1] = self.nf('jx')[::2, ::2]
        if 'jy' in self.variables:
            p.q[2] = self.nf('jy')[::2, ::2]
        for i, name in enumerate(self.add_fields):
            p.q[base_slots + i] = self.nf(name)

    def sync_from_problem_q(self) -> None:
        """Initial state copy of problem.q to FEM-owned nodal fields."""
        p = self.problem

        # P1 fields
        self.nf('rho')[:] = p.q[0]
        # Populate nodal fields (e.g. h) needed as args for p_from_rho before calling it
        for name in self.nodal_field_keys:
            self.nf(name)[:] = resolve_source(p, QUAD_FIELD_REGISTRY[name]['source'])
        self.nf('p')[:] = self._call_computed('p_from_rho', self.nf)
        self.problem.pressure.pressure[:] = self.nf('p')

        base_slots = len(self.variables) - len(self.add_fields)
        for i, name in enumerate(self.add_fields):
            self.nf(name)[:] = p.q[base_slots + i]

        # P2 fields
        if 'jx' in self.variables or 'jy' in self.variables:
            Nx_p, Ny_p = self.decomp.local_shape_padded
            Nx_P2, Ny_P2 = self.decomp.local_shape_padded_P2
            zoom_factors = (Nx_P2 / Nx_p, Ny_P2 / Ny_p)
            if 'jx' in self.variables:
                self.nf('jx')[:] = zoom(p.q[1], zoom_factors, order=1)
            if 'jy' in self.variables:
                self.nf('jy')[:] = zoom(p.q[2], zoom_factors, order=1)

    # =========================================================================
    # Field updates  (called once per Newton step)
    # =========================================================================

    def update_physics(self, it: int = 0, **kwargs) -> None:
        """Update physics model fields after solution field update."""
        p = self.problem

        # sync p to pressure module
        p_nodal = self.nf('p')
        self.problem.pressure.pressure[:] = p_nodal

        # rho update
        self.nf('rho')[:] = self._call_computed('rho_from_p', self.nf)
        self.sync_to_problem_q()

        # height update (needs updated p in pressure module)
        p.topo.update(it, **kwargs)

        # viscosity
        dp_dx = np.gradient(p_nodal, self.dx, axis=0)
        dp_dy = np.gradient(p_nodal, self.dy, axis=1)
        p.viscosity.update(p_nodal, dp_dx, dp_dy,
                           p.topo.h,
                           p.geo['U_bot'], p.geo['V_bot'],
                           p.geo['U_top'], p.geo['V_top'])

        # energy
        if self.energy:
            p.energy.update_temperature()

    def _call_computed(self, name, field_getter):
        entry = QUAD_FIELD_REGISTRY[name]
        func = resolve_source(self.problem, entry['source'])
        args = tuple(field_getter(a) for a in entry['args'])
        return self._apply_2d_vmap(func, *args)

    def _apply_2d_vmap(self, func, *args):
        shape = args[0].shape
        args_2d = [a.reshape(shape[0], -1) for a in args]
        return func(*args_2d).reshape(shape)

    def update_quad_fields(self) -> None:
        """Update all quad fields from nodal fields and physics methods.
        - nodal fields: copy and interpolate
        - scalar fields: broadcast from source
        - derivative fields: compute from nodal values
        - computed fields: call physics method with quad field arguments
        """

        p = self.problem
        s = np.s_[..., :-1, :-1]
        q = lambda name: self.quad_fields[name].pg[s]
        apply = self._apply_2d_vmap

        # Nodal fields - update
        for name in self.nodal_field_keys:
            self.nf(name)[:] = resolve_source(p, QUAD_FIELD_REGISTRY[name]['source'])

        # Nodal fields - interpolate to quad
        for name in self.nodal_field_keys | set(self.variables):
            self.interpolate_nodal_to_quad(name)

        # Compute 'dh_dx'/'dh_dy' quad values directly from nodal 'h', if needed
        if 'dh_dx' in self.quad_fields:
            self.quad_fields['dh_dx'].pg[s] = self._deriv_pg('h', 'x')
        if 'dh_dy' in self.quad_fields:
            self.quad_fields['dh_dy'].pg[s] = self._deriv_pg('h', 'y')

        # Special case for: rho_quad <- p_quad: we need an initial guess for rho_quad
        self.interpolate_nodal_to_quad('rho')
        q('rho')[:] = self._call_computed('rho_from_p', q)

        # Scalar Broadcast
        for name in self.quad_field_keys:
            if name in self._reg_scalar:
                self.qf(name)[:] = resolve_source(p, QUAD_FIELD_REGISTRY[name]['source'])

        # Derivative fields
        for name in self.der_field_keys:
            if name.startswith('d_dx_'):
                q(name)[:] = self._deriv_pg(name[5:], 'x')
            elif name.startswith('d_dy_'):
                q(name)[:] = self._deriv_pg(name[5:], 'y')

        # rho_avg must be ready before the computed loop (d2p_drho2 depends on it)
        if 'rho_avg' in self.quad_fields:
            q('rho_avg')[:] = 0.5 * (q('rho') + q('rho_prev'))

        # Computed fields
        for name in self.quad_field_keys:
            if name in self._reg_computed:
                entry = QUAD_FIELD_REGISTRY[name]
                func = resolve_source(p, entry['source'])
                args = tuple(q(a) for a in entry['args'])
                q(name)[:] = apply(func, *args)

        # Hardcoded quad fields
        if 'dh_dt' in self.quad_field_keys:
            self._update_squeeze_quad_fields(q)
        if 'xi' in self.variables and not self.problem.fem_solver['equations']['reynolds']:
            self._update_oss_quad_fields(q)
        if self.problem.fem_solver['stabilization']['fc']:
            self._update_fc_quad_fields(q)
        if self.problem.fem_solver['equations']['reynolds']:
            self._update_reynolds_quad_fields(q)
            if self.problem.fem_solver['stabilization']['mass_supg']:
                self._update_reynolds_supg_quad_fields(q)
        elif self.problem.fem_solver['stabilization']['mass_supg']:
            self._update_mass_supg_quad_fields(q)

    def _update_squeeze_quad_fields(self, q) -> None:
        """Compute dh_dt at quad points from h and h_prev."""
        dt = self.problem.numerics['dt']
        q('dh_dt')[:] = (q('h') - q('h_prev')) / dt

    def _update_oss_quad_fields(self, q) -> None:
        """Compute OSS stabilisation fields (a_vec, tau, one_minus_theta) at quad points."""
        rho_l = self.problem.prop['rho_l']
        geo = self.problem.geo
        U_m = 0.5 * (geo['U_bot'] + geo['U_top'])
        V_m = 0.5 * (geo['V_bot'] + geo['V_top'])

        a_vec_x = rho_l * U_m
        a_vec_y = rho_l * V_m
        q('a_vec_x')[:] = a_vec_x
        q('a_vec_y')[:] = a_vec_y

        norm_a = np.sqrt(a_vec_x**2 + a_vec_y**2)
        h_elem = min(self.dx, self.dy)
        if self.problem.fem_solver.get('oss_tau_pointwise', True):
            tau = h_elem / (2.0 * np.maximum(norm_a, 0.1))
        else:
            tau = 1.0 / (2.0 / h_elem * max(float(norm_a.mean()), 0.1))

        alpha = self.problem.fem_solver['stabilization']['oss_alpha']
        q('tau_a_x')[:] = alpha * tau * a_vec_x
        q('tau_a_y')[:] = alpha * tau * a_vec_y

    def _update_fc_quad_fields(self, q) -> None:
        """Compute flux-capturing diffusion coefficient fc_tau at quad points."""
        h_elem = min(self.dx, self.dy)
        beta = self.problem.fem_solver['stabilization']['fc_beta']

        R_mass = (
            -(
                (1 - q('theta')) * (q('d_dx_jx') + q('d_dy_jy'))
                - q('jx') * q('d_dx_theta')
                - q('jy') * q('d_dy_theta')
                + (1 - q('theta')) / q('h') * (q('dh_dx') * q('jx') + q('dh_dy') * q('jy'))
            )
            - (q('rho') * (1 - q('theta')) - q('rho_prev') * (1 - q('theta_prev'))) / self.problem.numerics['dt']
        )

        q('fc_tau')[:] = (h_elem**2 / 2.0) * beta * np.abs(R_mass) * 1e05

    def _update_reynolds_quad_fields(self, q) -> None:
        """Compute mean surface velocity U_m at quad points."""
        geo = self.problem.geo
        q('U_m')[:] = 0.5 * (geo['U_bot'] + geo['U_top'])

    def _update_reynolds_supg_quad_fields(self, q) -> None:
        """Compute UPG stabilization coefficient tau_supg at quad points."""
        geo = self.problem.geo
        U = 0.5 * (geo['U_bot'] + geo['U_top'])
        alpha = self.problem.fem_solver['stabilization']['mass_supg_factor']
        q('tau_supg')[:] = 0.5 * alpha * self.dx * np.sign(U)

    def _update_mass_supg_quad_fields(self, q) -> None:
        """Compute SUPG stabilization coefficients f_x, f_y at quad points."""
        alpha = self.problem.fem_solver['stabilization']['mass_supg_factor']
        supg_type = self.problem.fem_solver['stabilization']['mass_supg_type']

        if supg_type == 'flux':
            q('f_x')[:] = 0.5 * alpha * self.dx * np.sign(q('jx'))
            q('f_y')[:] = 0.5 * alpha * self.dy * np.sign(q('jy'))
        else:
            geo = self.problem.geo
            U = 0.5 * (geo['U_bot'] + geo['U_top'])
            V = 0.5 * (geo['V_bot'] + geo['V_top'])
            q('f_x')[:] = 0.5 * alpha * self.dx * np.sign(U)
            q('f_y')[:] = 0.5 * alpha * self.dy * np.sign(V)

    def collect_quad_fields(self) -> dict:
        return {name: self.get_quad_sq(name) for name in self.quad_fields}

    def store_prev_values(self) -> None:
        """Store current quad values for time derivatives."""
        for var in self.variables:
            prev_key = f'{var}_prev'
            if var in self.quad_fields and prev_key in self.quad_fields:
                self.qf(prev_key)[:] = self.qf(var).copy()
        if 'h_prev' in self.quad_fields:
            self.qf('h_prev')[:] = self.qf('h').copy()
        if 'rho_prev' in self.quad_fields:
            self.qf('rho_prev')[:] = self.qf('rho').copy()
        if 'dp_drho_prev' in self.quad_fields:
            self.qf('dp_drho_prev')[:] = self.qf('dp_drho').copy()
        if 'p_prev' in self.quad_fields:
            self.qf('p_prev')[:] = self.qf('p').copy()
