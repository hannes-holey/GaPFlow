#
# Copyright 2025 Hannes Holey
#           2026 Christoph Huber
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

import numpy as np
import numpy.typing as npt
from scipy.ndimage import zoom

from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from .problem import Problem
    from .parallel import DomainDecomposition


NDArray = npt.NDArray[np.floating]

BND_IDX = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
BND_TO_KEY = {'N': 'yN', 'E': 'xE', 'S': 'yS', 'W': 'xW'}


def offset_to_slice(bnd: str, k: int):
    """Return a 2D index tuple selecting the k-th layer from boundary `bnd`."""
    s, sn = slice, slice(None)
    if bnd == 'W':
        return (s(k, k + 1), sn)
    if bnd == 'E':
        return (s(-(k + 1), -k or None), sn)
    if bnd == 'S':
        return (sn, s(k, k + 1))
    if bnd == 'N':
        return (sn, s(-(k + 1), -k or None))


def sample_bc_spec(grid: dict, var_idx: int):
    """Extract bc_type and bc_vals lists for a variable from the grid BC dict.

    Parameters
    ----------
    grid : dict
        Grid configuration dict (problem.grid).
    var_idx : int
        Variable index (0=rho, 1=jx, 2=jy).

    Returns
    -------
    bc_type : List[str]
        BC types in BND_IDX order ('D', 'N', 'P').
    bc_vals : List[float | None]
        BC values in BND_IDX order, None if not Dirichlet.
    """
    bc_type, bc_vals = [], []
    for bnd in BND_IDX:
        key = BND_TO_KEY[bnd]
        t = grid[f'bc_{key}'][var_idx]
        bc_type.append(t)
        if t == 'D':
            raw = grid.get(f'bc_{key}_D_val')
            bc_vals.append(raw[var_idx] if isinstance(raw, list) else raw)
        else:
            bc_vals.append(0.0)
    return bc_type, bc_vals


def resolve_pressure_bcs(grid: dict, prop: dict, problem: "Problem" = None) -> None:
    """Convert pressure Dirichlet BCs (bc_*_P_val) to density (bc_*_D_val) in-place."""
    for side in ('xW', 'xE', 'yS', 'yN'):
        p_val = grid.get(f'bc_{side}_P_val')
        if p_val is not None:
            if problem is not None and problem.pressure.is_gp_model:
                import warnings
                warnings.warn(
                    f"bc_{side}_P_val with GP pressure: rho inversion not yet supported, skipping.")
            else:
                grid[f'bc_{side}_D_val'] = float(
                    problem.pressure.rho_from_p(np.array([[float(p_val)]])).squeeze())


def translate_bc_rho_to_p(bc_type: List[str], bc_vals: List[float | None],
                          problem: "Problem") -> List[float | None]:
    """Translate Dirichlet rho BC values to pressure using the EoS."""
    if problem.pressure.is_gp_model:
        result = []
        for (bnd, _), type, rho in zip(BND_IDX.items(), bc_type, bc_vals):
            if type == 'D':
                h_bnd = float(problem.topo.h[offset_to_slice(bnd, 1)].mean())
                p = float(problem.pressure.p_from_rho(np.array([[rho]]), np.array([[h_bnd]])).squeeze())
                result.append(p)
            else:
                result.append(0.0)
        return result
    return [float(problem.pressure.p_from_rho(np.array([[rho]])).squeeze()) if type == 'D' else 0.0
            for type, rho in zip(bc_type, bc_vals)]


class BCContext_:
    """Mutable context object passed to BC functions. Updated in-place per boundary.

    Attributes
    ----------
    problem : Problem
        The GaPFlow Problem instance.
    required_shape : tuple
        Shape of the P1 ghost layer at the current boundary.
    slice_ghost : tuple
        Slice for the P1 ghost layer (offset 0) at the current boundary.
    slice_interior : tuple
        Slice for the P1 interior layer (offset 1) at the current boundary.
    x_norm : NDArray
        Normalized x-coordinates at the ghost layer (0 to 1).
    y_norm : NDArray
        Normalized y-coordinates at the ghost layer (0 to 1).
    """

    def __init__(self, problem: "Problem"):
        self.problem = problem
        self.required_shape = None
        self.slice_ghost = None
        self.slice_interior = None
        self.x_norm = None
        self.y_norm = None


class BoundarySpec:
    """Structured BC specification for a single field.

    Attributes
    ----------
    field : Field
        The muGrid field this BC spec applies to.
    grid_type : str
        'P1' or 'P2', determines ghost depth and interpolation needs.
    bc_type : List[str]
        BC types per boundary in BND_IDX order: 'D', 'N', 'F', or 'P'.
    bc_vals : List[float | None]
        BC values per boundary if bc_type is 'D', else None.
    bc_functions : List[Callable | None]
        BC functions per boundary if bc_type is 'F', else None.
    decomp : DomainDecomposition
        Used for array sizing and P1-P2 zoom factors.
    """

    def __init__(self,
                 field,
                 grid_type: str,
                 bc_type: List[str],
                 bc_vals: List[float | None],
                 bc_functions: List[Callable | None],
                 decomp: "DomainDecomposition",
                 do_exchange: bool = True):

        self.field = field
        self.grid_type = grid_type
        self.bc_type = bc_type
        self.bc_vals = bc_vals
        self.bc_functions = bc_functions
        self.decomp = decomp
        self.do_exchange = do_exchange

        if self.grid_type == 'P2':
            self._compute_zoom_factors()

        self._make_arrays()
        self._make_bnds()

    def _compute_zoom_factors(self):
        """Precompute zoom factors for function-based BCs on P2 grids."""
        Nx_p1, Ny_p1 = self.decomp.local_shape_padded
        Nx_p2, Ny_p2 = self.decomp.local_shape_padded_P2
        self.zoom_factors = {
            'W': (1.0, Ny_p2 / Ny_p1),
            'E': (1.0, Ny_p2 / Ny_p1),
            'S': (Nx_p2 / Nx_p1, 1.0),
            'N': (Nx_p2 / Nx_p1, 1.0),
        }

    def _make_arrays(self):
        """Create template arrays for uniform value broadcasting."""
        if self.grid_type == 'P2':
            Nx, Ny = self.decomp.local_shape_padded_P2
        else:
            Nx, Ny = self.decomp.local_shape_padded
        self.arr = {
            'W': np.zeros((1, Ny)),
            'E': np.zeros((1, Ny)),
            'S': np.zeros((Nx, 1)),
            'N': np.zeros((Nx, 1)),
        }

    def _make_bnds(self):
        """Create List[str] of boundaries this BC spec applies to."""
        self.bnds = [bnd for bnd, idx in BND_IDX.items() if self.bc_type[idx] != 'P']

    def _idx(self, bnd: str) -> int:
        return BND_IDX[bnd]

    def get_bc_type(self, bnd: str) -> str:
        return self.bc_type[self._idx(bnd)]

    def is_function(self, bnd: str) -> bool:
        return self.bc_type[self._idx(bnd)] == 'F'

    def get_bc_val(self, bnd: str) -> float:
        return self.bc_vals[self._idx(bnd)]

    def bc_function(self, bnd: str, ctx: BCContext_):
        """Evaluate BC function for the specified boundary."""
        func = self.bc_functions[self._idx(bnd)]
        res = func(ctx)
        assert res.shape == ctx.required_shape
        return res

    def get_zoom_factor(self, bnd: str):
        return self.zoom_factors[bnd]


class GhostUpdater:
    """Generalized ghost exchange + BC application.
    """

    def __init__(self, decomp: "DomainDecomposition",
                 problem: "Problem",
                 specs: List[BoundarySpec]):
        self.decomp = decomp
        self.problem = problem
        self.specs = specs
        self.dx = decomp.grid['dx']
        self.dy = decomp.grid['dy']
        self.ctx = BCContext_(problem)

    def _exchange_ghosts(self, field, grid_type):
        if grid_type == 'P1':
            self.decomp.exchange_ghosts(field)
        else:
            self.decomp.exchange_ghosts_P2(field)

    def _interpolate_to_P2(self, arr_, bnd: str, bc_spec: BoundarySpec):
        """Interpolate function-based BC array from P1 to P2."""
        zoom_factors = bc_spec.get_zoom_factor(bnd)
        return zoom(arr_, zoom_factors, order=1)

    def _grid_fit(self, arr_, bnd: str, bc_spec: BoundarySpec):
        """BC function is always evaluated on P1.
        Check if interpolation to P2 is necessary."""
        if bc_spec.grid_type == 'P2':
            return self._interpolate_to_P2(arr_, bnd, bc_spec)
        return arr_

    def _get_d_cell(self, bnd: str) -> float:
        return self.dx if bnd in ('W', 'E') else self.dy

    def _update_ctx(self, slice_interior, slice_outer):
        """Update mutable context object for BC function evaluation."""
        self.ctx.slice_ghost = slice_outer
        self.ctx.slice_interior = slice_interior
        self.ctx.required_shape = self.decomp.xx_norm[slice_outer].shape
        self.ctx.x_norm = self.decomp.xx_norm[slice_outer]
        self.ctx.y_norm = self.decomp.yy_norm[slice_outer]

    def _get_BC_array(self, bnd: str, bc_spec: BoundarySpec, slice_interior, slice_outer):
        """Get BC array for the specified boundary.
        Zoomed to P2 size if bc_spec is a function and grid_type is P2."""
        arr = bc_spec.arr[bnd]
        if bc_spec.is_function(bnd):
            self._update_ctx(slice_interior, slice_outer)
            arr_ = bc_spec.bc_function(bnd, self.ctx)
            arr = self._grid_fit(arr_, bnd, bc_spec)
        else:
            arr[:] = bc_spec.get_bc_val(bnd)
        return arr

    def _get_ghost_slices(self, bnd: str, grid_type: str):
        s_2, s_1, s_0 = [offset_to_slice(bnd, k) for k in [2, 1, 0]]
        if grid_type == 'P1':
            return s_1, s_1, s_0
        else:
            return s_2, s_1, s_0

    def update(self) -> None:
        """Perform ghost exchange and BC application for all specs."""
        for bc_spec in self.specs:
            field = bc_spec.field.pg[0] if hasattr(bc_spec.field, 'pg') else bc_spec.field
            grid_type = bc_spec.grid_type

            if bc_spec.do_exchange:
                self._exchange_ghosts(bc_spec.field, grid_type)

            for bnd in bc_spec.bnds:
                if not self.decomp.owns_boundary(bnd):
                    continue

                interior, ghost_middle, ghost_outer = self._get_ghost_slices(bnd, grid_type)
                bc_arr = self._get_BC_array(bnd, bc_spec, ghost_middle, ghost_outer)
                inner = field[interior]
                d_cell = self._get_d_cell(bnd)
                bc_type = bc_spec.get_bc_type(bnd)

                if bc_type == 'D' or bc_type == 'F':
                    if grid_type == 'P1':
                        field[ghost_outer] = bc_arr
                    else:
                        field[ghost_middle] = bc_arr
                        field[ghost_outer] = bc_arr

                elif bc_type == 'N':
                    if grid_type == 'P1':
                        field[ghost_outer] = inner + d_cell * bc_arr
                    else:
                        field[ghost_middle] = inner + d_cell / 2 * bc_arr
                        field[ghost_outer] = inner + d_cell * bc_arr
