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
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from .elements import TaylorHoodQ2Q1
from .grid_index import GridIndexManager
from .global_matrix import field_to_global
from .fieldspec import FieldSpec

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]


@dataclass
class GlobalIndexPattern:
    """Global indices for nnz values -> global matrix entries."""
    local_size: int
    mat_global_rows: "IntArray"
    mat_global_cols: "IntArray"
    rhs_global_rows: "IntArray"


@dataclass
class AssemblyTemplate:
    """Precomputed injection template for one (res, [var,] deriv_key) combination."""
    w: NDArray
    entries_per_quad: int
    nnz: IntArray


# Residual, Variable
BLOCK = (('P1', 'P1'), ('P1', 'P2'), ('P2', 'P1'), ('P2', 'P2'))


class Assembly:
    """Stencils, node connectivity, and assembly of rhs and tangential matrix.

    Parameters
    ----------
    grid_idx : GridIndexManager
    element : TaylorHoodQ2Q1
    var_specs : list of FieldSpec
        Active variables in block order.
    res_specs : list of FieldSpec
        Active residuals in block order.
    terms : list of Term
        All terms (including diagnostic).
    """

    def __init__(self,
                 grid_idx: GridIndexManager,
                 element: TaylorHoodQ2Q1,
                 var_specs: List[FieldSpec],
                 res_specs: List[FieldSpec],
                 terms: list,
                 ) -> None:

        self.grid_idx = grid_idx
        self.element = element
        self.var_specs = var_specs
        self.res_specs = res_specs
        self.assembly_terms = [t for t in terms if t.res != 'diagnostic']

        # Convenience views derived from specs — single source of truth
        self.variables = [s.name for s in var_specs]
        self.residuals = [s.name for s in res_specs]
        self.var_to_grid = {s.name: s.grid for s in var_specs}
        self.res_to_grid = {s.name: s.grid for s in res_specs}
        self.var_spec = {s.name: s for s in var_specs}
        self.res_spec = {s.name: s for s in res_specs}
        self.p1_factor = sum(1 for s in var_specs if s.grid == 'P1')
        self.p2_factor = sum(1 for s in var_specs if s.grid == 'P2')

        # Ordered list of all (res, var) block combinations
        # holds block-specific info: 'res_grid', 'var_grid', 'nnz_idx_start', 'nb_nnz'
        self.block_order = {
            (rs.name, vs.name): {'res_grid': rs.grid, 'var_grid': vs.grid}
            for rs in res_specs
            for vs in var_specs
        }

        # Build block-wise nnz list (inner_pts, contrib_pts) -> block-nnz
        self.conn = self.build_connectivity()

        # Build nnz list across all blocks and corresponding lookup
        # (block, (inner_pts, contrib_pts)) -> local_nnz / global_nnz
        self._build_coo_pattern(self.conn)
        self.coo_lookups = self._build_coo_lookups()

        # Build global row indices for RHS entries: (res, local_inner_idx) -> global_row_idx
        self._build_rhs_pattern()

        # Helpers
        self._build_slices()
        self._build_scaling_indices()

        self.res_size = self._res_slices[self.residuals[-1]].stop
        # +1 as trash can for non-valid entries
        self._nnz_buf = np.zeros(len(self.nnz_global_rows) + 1, dtype=np.float64)
        self._rhs_buf = np.zeros(self.res_size + 1, dtype=np.float64)

        # Weighting templates for the effective influence Newton method
        self.elastic_templates: Dict[str, AssemblyTemplate] = {}
        self.elastic_templates_dhdx: Dict[str, AssemblyTemplate] = {}
        self.elastic_templates_dhdy: Dict[str, AssemblyTemplate] = {}
        self._nnz_cache: Dict[Tuple[str, str], IntArray] = {}

    # ======================================================================
    # Block connectivity — stencil-based
    # ======================================================================

    @staticmethod
    def _compute_stencils(element: TaylorHoodQ2Q1):
        """Compute stencils for all four block types from the fine-grid stencils.
        Just a block-specific resampling of the stencils in elements.py.
        """

        stencil = {}
        stencil[3] = {  # P2-P2 has all
            (0, 0): np.array(element.stencil_even_even, dtype=np.int32),
            (1, 1): np.array(element.stencil_odd_odd, dtype=np.int32),
            (0, 1): np.array(element.stencil_even_odd, dtype=np.int32),
            (1, 0): np.array(element.stencil_odd_even, dtype=np.int32),
        }

        for idx, combination in enumerate(BLOCK[0:3]):

            stencil[idx] = {}
            res = combination[0]
            var = combination[1]

            for origin in stencil[3].keys():

                if res == 'P1' and origin != (0, 0):  # only use origin (0, 0) for P1 residuals
                    continue

                if var == 'P2':
                    stencil[idx][origin] = stencil[3][origin]  # use all
                    continue

                # var = 'P1', filter for even-even
                points = stencil[3][origin]
                stencil[idx][origin] = np.empty((0, 2), dtype=np.int32)
                for point in points:
                    x = origin[0] + point[0]
                    y = origin[1] + point[1]
                    if not (x % 2 == 0 and y % 2 == 0):
                        continue
                    stencil[idx][origin] = np.vstack([stencil[idx][origin], point])

        return stencil

    def build_connectivity(self) -> Dict[Tuple[str, str], Tuple[IntArray, IntArray, int]]:
        """Build connectivity for all four block types using fine-grid stencils.
        For each P1/P2 combination, returns (inner_pts, contrib_pts, nb_nnz) where
        indices follow index_mask_padded_local.
        """
        connectivity = {}
        stencil = self._compute_stencils(self.element)

        for block_index, block_type in enumerate(BLOCK):
            inner_pts, contrib_pts = self.apply_stencil(stencil[block_index], block_type)
            connectivity[block_type] = (inner_pts, contrib_pts, len(inner_pts))

        return connectivity

    def apply_stencil(self, stencil, block_type) -> Tuple[IntArray, IntArray]:
        """Apply the given stencil to compute (inner_pts, contrib_pts) for one block type."""

        res_grid, var_grid = block_type
        grid_idx = self.grid_idx

        m_inner_P2 = grid_idx.index_mask_inner_local_P2
        m_padded_P2 = grid_idx.index_mask_padded_local_P2()
        m_padded_P1 = grid_idx.index_mask_padded_local_P1()

        inner_pts_2d = np.argwhere(m_inner_P2 >= 0)          # (N, 2)
        inner_idx = m_inner_P2[inner_pts_2d[:, 0], inner_pts_2d[:, 1]]

        inner_list, contrib_list = [], []

        for origin, offsets in stencil.items():
            sel = ((inner_pts_2d[:, 0] % 2 == origin[0]) &
                   (inner_pts_2d[:, 1] % 2 == origin[1]))
            pts = inner_pts_2d[sel]
            idx = inner_idx[sel]

            # iterate through stencil offsets; compile all res-var pairs simultaneously
            # using grid index masks
            for dx, dy in offsets:
                nx = pts[:, 0] + dx
                ny = pts[:, 1] + dy

                if var_grid == 'P2':
                    contrib = m_padded_P2[nx, ny]
                else:
                    contrib = m_padded_P1[nx // 2, ny // 2]

                valid = contrib >= 0
                if res_grid == 'P2':
                    inner_list.append(idx[valid])
                else:
                    inner_list.append(m_padded_P1[pts[valid, 0] // 2, pts[valid, 1] // 2])
                contrib_list.append(contrib[valid])

        inner_all = np.concatenate(inner_list).astype(np.int32)
        contrib_all = np.concatenate(contrib_list).astype(np.int32)

        pairs = np.column_stack([inner_all, contrib_all])
        _, unique_idx = np.unique(pairs, axis=0, return_index=True)
        unique_idx.sort()
        return inner_all[unique_idx], contrib_all[unique_idx]

    # ======================================================================
    # COO pattern
    # ======================================================================

    def _build_coo_pattern(self, connectivity) -> None:
        """Build nnz list. Ordering determined by self._block_order.
        Stacks the block-specific connectivity pairs from apply_stencil into a single list.
        """
        self.nnz_local_to = np.empty((0,), dtype=np.int32)
        self.nnz_local_from = np.empty((0,), dtype=np.int32)
        self.nnz_global_rows = np.empty((0,), dtype=np.int32)
        self.nnz_global_cols = np.empty((0,), dtype=np.int32)

        nnz_idx_start = 0

        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            inner_pts, contrib_pts, nb_nnz = connectivity[(rg, vg)]

            # used for M_dense reconstruction
            self.nnz_local_to = np.concatenate([self.nnz_local_to, inner_pts])
            self.nnz_local_from = np.concatenate([self.nnz_local_from, contrib_pts])

            inner_pts_global_idx = self.apply_l2g(rg, inner_pts)
            contrib_pts_global_idx = self.apply_l2g(vg, contrib_pts)

            global_rows = field_to_global(inner_pts_global_idx, self.res_spec[res], self.grid_idx, self.p1_factor, self.p2_factor)
            global_cols = field_to_global(contrib_pts_global_idx, self.var_spec[var], self.grid_idx, self.p1_factor, self.p2_factor)
            self.nnz_global_rows = np.concatenate([self.nnz_global_rows, global_rows])
            self.nnz_global_cols = np.concatenate([self.nnz_global_cols, global_cols])

            block['nnz_idx_start'] = nnz_idx_start
            block['nb_nnz'] = nb_nnz
            nnz_idx_start += nb_nnz

    def apply_l2g(self, grid_type: str, local_indices: IntArray) -> IntArray:
        """Apply local-to-global mapping for the given grid type."""
        if grid_type == 'P2':
            return self.grid_idx.l2g_list_P2[local_indices]
        else:
            return self.grid_idx.l2g_list_P1[local_indices]

    # ======================================================================
    # COO lookup
    # ======================================================================

    def _build_coo_lookups(self):
        """Build one sorted-key lookup per (res, var) block."""
        lookups = {}
        for (res, var), block in self.block_order.items():
            rg, vg = block['res_grid'], block['var_grid']
            inner_pts, contrib_pts, nb_nnz = self.conn[(rg, vg)]
            start = block['nnz_idx_start']
            contrib_stride = int(contrib_pts.max()) + 1 if nb_nnz else 1
            keys = inner_pts.astype(np.int64) * contrib_stride + contrib_pts.astype(np.int64)
            values = start + np.arange(nb_nnz, dtype=np.int32)
            order = np.argsort(keys)
            lookups[(res, var)] = (keys[order], values[order], contrib_stride)
        return lookups

    def _lookup_nnz_vec(self, res: str, var: str, res_idx: IntArray, var_idx: IntArray) -> IntArray:
        """Vectorized lookup of absolute flat nnz index."""
        keys_sorted, values_sorted, contrib_stride = self.coo_lookups[(res, var)]
        both_valid = (res_idx >= 0) & (var_idx >= 0)
        res_safe = np.where(both_valid, res_idx, 0)
        var_safe = np.where(both_valid, var_idx, 0)
        query_keys = res_safe.astype(np.int64) * contrib_stride + var_safe.astype(np.int64)
        pos = np.searchsorted(keys_sorted, query_keys)
        pos_clipped = np.clip(pos, 0, len(keys_sorted) - 1)
        found = both_valid & (pos < len(keys_sorted)) & (keys_sorted[pos_clipped] == query_keys)
        return np.where(found, values_sorted[pos_clipped], -1).astype(np.int32)

    # ======================================================================
    # RHS pattern
    # ======================================================================

    def _build_rhs_pattern(self) -> None:
        """Build global RHS row indices for all residuals, which effectively
        allows mapping of (res, local_inner_idx) -> global_row_idx.

        For each residual, maps local inner-node indices to global DOF indices
        using field_to_global. Result is stored as self.rhs_global_rows.

        Implicitly assumes that local residual vector is ordered in the
        self.residuals order, and for each residual, the local entries are
        ordered by the inner node indices.
        """
        rows = []
        for spec in self.res_specs:
            nb_inner = (self.grid_idx.Nx_P2_inner * self.grid_idx.Ny_P2_inner
                        if spec.grid == 'P2'
                        else self.grid_idx.Nx_P1_inner * self.grid_idx.Ny_P1_inner)

            global_field_indices = self.apply_l2g(spec.grid, np.arange(nb_inner, dtype=np.int32))
            rows.append(field_to_global(global_field_indices, spec, self.grid_idx, self.p1_factor, self.p2_factor))

        self.rhs_global_rows: IntArray = np.concatenate(rows)

    # ======================================================================
    # Slices
    # ======================================================================

    def _build_slices(self) -> None:
        """Precompute contiguous slices for residual and solution vectors."""
        self._res_slices: Dict[str, slice] = {}
        offset = 0
        for res in self.residuals:
            n = (self.grid_idx.Nx_P2_inner * self.grid_idx.Ny_P2_inner
                 if self.res_to_grid[res] == 'P2'
                 else self.grid_idx.Nx_P1_inner * self.grid_idx.Ny_P1_inner)
            self._res_slices[res] = slice(offset, offset + n)
            offset += n

        self._sol_slices: Dict[str, slice] = {}
        offset = 0
        for var in self.variables:
            n = (self.grid_idx.Nx_P2_inner * self.grid_idx.Ny_P2_inner
                 if self.var_to_grid[var] == 'P2'
                 else self.grid_idx.Nx_P1_inner * self.grid_idx.Ny_P1_inner)
            self._sol_slices[var] = slice(offset, offset + n)
            offset += n

    def _build_scaling_indices(self) -> None:
        """Build index arrays mapping each nnz / rhs entry to its variable/residual index.

        These are needed by scaling.build_scaling for per-variable/residual scaling.
        """
        var_idx_list = []
        res_idx_list = []
        for (res, var), block in self.block_order.items():
            nb_nnz = block['nb_nnz']
            var_idx_list.append(np.full(nb_nnz, self.variables.index(var), dtype=np.int32))
            res_idx_list.append(np.full(nb_nnz, self.residuals.index(res), dtype=np.int32))
        self.var_block_idx: IntArray = np.concatenate(var_idx_list)
        self.res_block_idx: IntArray = np.concatenate(res_idx_list)

        rhs_idx_list = []
        for i, res in enumerate(self.residuals):
            nb_inner = self._res_slices[res].stop - self._res_slices[res].start
            rhs_idx_list.append(np.full(nb_inner, i, dtype=np.int32))
        self.rhs_res_block_idx: IntArray = np.concatenate(rhs_idx_list)

    # ======================================================================
    # Linear solver info
    # ======================================================================

    def get_petsc_info(self, res_size: int) -> "GlobalIndexPattern":
        """Return assembly info for the linear solver.

        Parameters
        ----------
        res_size : int
            Total residual vector length (sum of nb_inner over all residuals).
        """
        return GlobalIndexPattern(
            local_size=res_size,
            mat_global_rows=self.nnz_global_rows,
            mat_global_cols=self.nnz_global_cols,
            rhs_global_rows=self.rhs_global_rows,
        )

    # ======================================================================
    # Assembly templates
    # ======================================================================

    def build_assembly_templates(self) -> None:
        """Precompute injection templates for all (res, var, deriv_key) combinations.
        deriv_key = (depvar_deriv, test_deriv) from term.depvar_deriv_for(var).
        Shape weighting depends on deriv_key; nnz_index depends only on (res, var).
        """
        self.assembly_templates = {}

        # Collect all (res, var, dd, td) quadruples actually needed
        needed = set()
        for term in self.assembly_terms:
            td = term.test_deriv
            for var in term.dep_vars:
                dd = term.depvar_deriv_for(var)
                needed.add((term.res, var, dd, td))

        for res, var, dd, td in needed:
            w, entries_per_quad = self._build_weighting(res, var, dd, td)
            self.assembly_templates[(res, var, dd, td)] = AssemblyTemplate(
                w=w,
                entries_per_quad=entries_per_quad,
                nnz=self._get_nnz(res, var),
            )

        # Residual-only keys for assemble_rhs (one key per term: (res, dd, td))
        for term in self.assembly_terms:
            dd, td = term.deriv_key
            key = (term.res, dd, td)
            if key not in self.assembly_templates:
                w, entries_per_quad = self._build_res_weighting(term.res, dd, td)
                self.assembly_templates[key] = AssemblyTemplate(
                    w=w,
                    entries_per_quad=entries_per_quad,
                    nnz=self._build_nnz_res(term.res),
                )

    def build_elastic_templates(self, elastic_deformation) -> None:
        """Precompute injection templates for the Effective Influence Newton
        Method's elastic-influence Jacobian contribution

        Parameters
        ----------
        elastic_deformation : GaPFlow.topography.ElasticDeformation
        """
        G3 = self._build_greens_function(elastic_deformation)

        for res in ('mass', 'momentum_x', 'momentum_y'):
            if res not in self.residuals:
                continue
            w, entries_per_quad = self._build_weighting_elastic(res, G3)
            self.elastic_templates[res] = AssemblyTemplate(
                w=w,
                entries_per_quad=entries_per_quad,
                nnz=self._get_nnz(res, 'p'),
            )

        for deriv, templates in (('x', self.elastic_templates_dhdx),
                                    ('y', self.elastic_templates_dhdy)):
            w, entries_per_quad = self._build_weighting_elastic('mass', G3, deriv=deriv)
            templates['mass'] = AssemblyTemplate(
                w=w,
                entries_per_quad=entries_per_quad,
                nnz=self._get_nnz('mass', 'p'),
            )

    def _build_greens_function(self, elastic_deformation) -> NDArray:
        """Return the 3x3 P1 elastic Green's function block; dx, dy in {-1, 0, 1}."""
        decomp = self.grid_idx.decomp
        Nx_padded = self.grid_idx.Nx_P1_padded
        Ny_padded = self.grid_idx.Ny_P1_padded

        p_impulse = np.zeros((Nx_padded, Ny_padded))
        if decomp.rank == 0:
            ix, iy = Nx_padded // 2, Ny_padded // 2
            p_impulse[ix, iy] = 1.0

        disp = elastic_deformation.get_deformation(p_impulse)

        G3 = np.zeros((3, 3))
        if decomp.rank == 0:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    G3[dx + 1, dy + 1] = disp[ix + dx, iy + dy]

        decomp._mpi_comm.Bcast(G3, root=0)
        return G3

    def _interpolate_G_quad(self, G_sub: NDArray, coords: NDArray, deriv: str = None) -> float:
        """Bilinearly (Q1) interpolate a 2x2 block of G3 to a quad point."""
        x, y = coords
        if deriv is None:
            return ((1 - x) * (1 - y) * G_sub[0, 0] + x * (1 - y) * G_sub[1, 0]
                    + (1 - x) * y * G_sub[0, 1] + x * y * G_sub[1, 1])
        elif deriv == 'x':
            return (-(1 - y) * G_sub[0, 0] + (1 - y) * G_sub[1, 0]
                    - y * G_sub[0, 1] + y * G_sub[1, 1]) / self.element.dx
        elif deriv == 'y':
            return (-(1 - x) * G_sub[0, 0] - x * G_sub[1, 0]
                    + (1 - x) * G_sub[0, 1] + x * G_sub[1, 1]) / self.element.dy

    # G3 sub-block (2x2) to use for each Q1 corner's dh/dp interpolation,
    # picked so the corner's own physical position is included in the block.
    _G_SUB_SLICES = [
        (slice(1, 3), slice(1, 3)),  # bl
        (slice(0, 2), slice(1, 3)),  # br
        (slice(1, 3), slice(0, 2)),  # tl
        (slice(0, 2), slice(0, 2)),  # tr
    ]

    def _build_weighting_elastic(self, res: str, G3: NDArray, deriv: str = None):
        """Build the elastic-influence shape_weighting for a P1-grid."""
        Q1 = self.element.Q1
        n_var = Q1.nodes_per_element
        n_quad = self.element.Quadrature.nb_points

        elem_res = self.element.Q1 if self.res_to_grid[res] == 'P1' else self.element.Q2

        n_res = elem_res.nodes_per_element
        n_entries_per_quad = n_res * n_var
        res_N = elem_res.N  # shape (n_quad, n_res)

        quad_coords = self.element.Quadrature.coordinates  # shape (n_quad, 2)
        weights = self.element.Quadrature.weights  # shape (n_quad,)

        # Compute dh/dp_i at quad points
        dh_dp_quad = np.zeros((n_quad, n_var))

        for quad_idx in range(n_quad):
            coords = quad_coords[quad_idx]

            for node_idx in range(n_var):
                row_sl, col_sl = self._G_SUB_SLICES[node_idx]
                G_sub = G3[row_sl, col_sl]

                dh_dp_quad[quad_idx, node_idx] = self._interpolate_G_quad(
                    G_sub, coords, deriv=deriv)

        # Repeat for all residual nodes in the square
        dh_dp_quad_ = dh_dp_quad.repeat(n_res, axis=1).reshape(n_quad, n_var, n_res)

        # Assemble weighting and residual shape function vectors
        area = self.element.sq_area
        weights_vec = np.repeat(weights * area, n_entries_per_quad)
        res_N_vec = np.tile(res_N, (1, n_var)).ravel()  # shape (n_quad * n_res * n_var,)

        return dh_dp_quad_.ravel() * res_N_vec * weights_vec, n_entries_per_quad

    def _build_weighting(self, res: str, var: str,
                         depvar_deriv: str, test_deriv):
        """Build shape_weighting for one (res, var, depvar_deriv, test_deriv)
        combination for the tangential matrix.

        shape_weighting is applied to one square and repeated for all squares.
        """
        res_grid, var_grid = self.res_to_grid[res], self.var_to_grid[var]

        res_element = self.element.Q1 if res_grid == 'P1' else self.element.Q2
        var_element = self.element.Q1 if var_grid == 'P1' else self.element.Q2

        nodes_res = res_element.nodes_per_element
        nodes_var = var_element.nodes_per_element

        # --- Trial function (var) shape functions ---
        deriv_scale = 1.0
        if depvar_deriv == 'none':
            var_N = var_element.N
        else:
            var_N = var_element.dN_dx if depvar_deriv == 'x' else var_element.dN_dy
            d = self.element.dx if depvar_deriv == 'x' else self.element.dy
            deriv_scale *= 1.0 / d

        # --- Test function (res) shape functions ---
        if not test_deriv:
            res_N = res_element.N
        else:
            res_N = res_element.dN_dx if test_deriv == 'x' else res_element.dN_dy
            d = self.element.dx if test_deriv == 'x' else self.element.dy
            deriv_scale *= -1.0 / d

        entries_per_quad = nodes_res * nodes_var

        weights = self.element.Quadrature.weights  # shape (nb_quad,)
        area = self.element.sq_area
        weights_vec = np.repeat(weights * area * deriv_scale, entries_per_quad)

        var_N_vec = var_N.repeat(nodes_res, axis=1).ravel()
        res_N_vec = np.tile(res_N, (1, nodes_var)).ravel()

        return weights_vec * var_N_vec * res_N_vec, entries_per_quad

    def _get_nnz(self, res: str, var: str) -> IntArray:
        """Cached accessor for _build_nnz, keyed on (res, var) only since the
        result is identical across all derivative combos sharing that block.
        """
        key = (res, var)
        if key not in self._nnz_cache:
            self._nnz_cache[key] = self._build_nnz(res, var)
        return self._nnz_cache[key]

    def _build_nnz(self, res: str, var: str) -> IntArray:
        """Build nnz injection indices for one (res, var) block.
        Note: same for all derivative combos.

        Ordering: 0->0, 0->1, ... residual moves faster than variable
        (must match _build_weighting's entries_per_quad ordering exactly)
        """

        TO_Q2 = self.grid_idx.sq_TO_inner_P2  # (n_sq, 9)
        TO_Q1 = self.grid_idx.sq_TO_inner_P1  # (n_sq, 4)
        FROM_Q2 = self.grid_idx.sq_FROM_padded_P2(var)  # (n_sq, 9)
        FROM_Q1 = self.grid_idx.sq_FROM_padded_P1(var)  # (n_sq, 4)

        res_sq_to_nodes = TO_Q1 if self.res_to_grid[res] == 'P1' else TO_Q2  # (n_sq, nodes_res)
        var_sq_to_nodes = FROM_Q1 if self.var_to_grid[var] == 'P1' else FROM_Q2  # (n_sq, nodes_var)

        n_sq, nodes_res = res_sq_to_nodes.shape
        nodes_var = var_sq_to_nodes.shape[1]

        res_idx = np.broadcast_to(res_sq_to_nodes[:, None, :], (n_sq, nodes_var, nodes_res))
        var_idx = np.broadcast_to(var_sq_to_nodes[:, :, None], (n_sq, nodes_var, nodes_res))
        nnz = self._lookup_nnz_vec(res, var, res_idx, var_idx)

        return nnz.ravel().astype(np.int32)

    def _scatter_quad_contribution(self, res_quad_field: NDArray, tmpl: AssemblyTemplate) -> None:
        """Core of the assembly process; from quadrature values to matrix entries. Steps:
        - repeat quad values by entries per quad
        - tile shape weighting by number of squares
        - multiply and reduce from per-quad to per-square contributions
        - accumulate into the nnz buffer

        Parameters
        ----------
        res_quad_field : NDArray
            shape (n_sq, n_quad_sq)
        tmpl : AssemblyTemplate
            Template for the assembly process
        """
        nb_sq = self.grid_idx.nb_sq
        quad_per_sq = self.element.Quadrature.nb_points
        sw = tmpl.w
        entries_per_quad = tmpl.entries_per_quad

        quad_val_vec = np.repeat(res_quad_field.flatten(), entries_per_quad)
        sw_vec = np.tile(sw, nb_sq)
        q_vec = quad_val_vec * sw_vec

        ele_vec = q_vec.reshape(-1, quad_per_sq, entries_per_quad).sum(axis=1).reshape(-1)

        np.add.at(self._nnz_buf, tmpl.nnz, ele_vec)

    def assemble_matrix(self,
                        quad_fields: Dict[str, NDArray],
                        ) -> NDArray:
        """Accumulate Jacobian term contributions and return a view on the result.

        Returns
        -------
        NDArray
            View of shape (n_nnz,) into the internal buffer. Valid until the
            next call to assemble_matrix.
        """
        self._nnz_buf[:] = 0.0

        for term in self.assembly_terms:
            td = term.test_deriv
            dep_vars = [quad_fields[v] for v in term.dep_vars]
            res = term.res

            for var in term.dep_vars:
                dd = term.depvar_deriv_for(var)
                key = (res, var, dd, td)
                tmpl = self.assembly_templates[key]

                res_quad_field = term.evaluate_deriv(var, *dep_vars)
                self._scatter_quad_contribution(res_quad_field, tmpl)

            if term.der_h is not None and res in self.elastic_templates:
                dRdh = term.der_h(*dep_vars)
                self._scatter_quad_contribution(dRdh, self.elastic_templates[res])

            if term.der_h_dx is not None and res in self.elastic_templates_dhdx:
                dRdhdx = term.der_h_dx(*dep_vars)
                self._scatter_quad_contribution(dRdhdx, self.elastic_templates_dhdx[res])

            if term.der_h_dy is not None and res in self.elastic_templates_dhdy:
                dRdhdy = term.der_h_dy(*dep_vars)
                self._scatter_quad_contribution(dRdhdy, self.elastic_templates_dhdy[res])

        return self._nnz_buf[:-1]

    def _build_res_weighting(self, res: str, depvar_deriv: str, test_deriv):
        """Residual weighting arrays (n_quad_sq * nodes_res,)
        """
        res_grid = self.res_to_grid[res]

        res_element = self.element.Q1 if res_grid == 'P1' else self.element.Q2

        nodes_res = res_element.nodes_per_element

        if not test_deriv:
            res_N = res_element.N
            factor = 1.0
        else:
            res_N = res_element.dN_dx if test_deriv == 'x' else res_element.dN_dy
            factor = -1.0 / self.element.dx if test_deriv == 'x' else -1.0 / self.element.dy

        # shape (n_quad_sq * nodes_res,)
        # (q0, N0), (q0, N1), (q0, N2), (q1, N0), ...
        res_N_vec = res_N.reshape(-1)

        # weights compensate for area on square
        weights = self.element.Quadrature.weights  # shape (nb_quad,)
        area = self.element.sq_area

        weights_vec = np.repeat(weights * area * factor, nodes_res)

        res_weighting = weights_vec * res_N_vec

        return res_weighting, nodes_res

    def _build_nnz_res(self, res: str) -> IntArray:
        """Build nnz_index for residual-only weighting. Same for all derivative combos.

        Ordering: 0->0, 0->1, ... residual moves faster than variable
        (must match _build_res_weighting's ordering exactly)
        """

        TO_Q2 = self.grid_idx.sq_TO_inner_P2  # (n_sq, 9)
        TO_Q1 = self.grid_idx.sq_TO_inner_P1  # (n_sq, 4)
        n_sq = self.grid_idx.nb_sq

        res_sq_to_nodes = TO_Q1 if self.res_to_grid[res] == 'P1' else TO_Q2
        res_element = self.element.Q1 if self.res_to_grid[res] == 'P1' else self.element.Q2

        nb_nnz = n_sq * res_element.nodes_per_element
        nnz = np.empty((nb_nnz), dtype=np.int32)
        nnz_idx = 0

        res_slice_start = self._res_slices[res].start

        for sq_idx in range(n_sq):
            res_nodes_on_sq = res_sq_to_nodes[sq_idx]

            for res_node in res_nodes_on_sq:
                local_idx = res_slice_start + res_node if res_node >= 0 else -1
                nnz[nnz_idx] = local_idx
                nnz_idx += 1

        return nnz

    def assemble_rhs(self, quad_fields: Dict[str, NDArray]) -> NDArray:
        """Accumulate residual term contributions and return a view on the result.

        Returns
        -------
        NDArray
            View of shape (res_size,) into the internal buffer. Valid until the
            next call to assemble_rhs.
        """
        self._rhs_buf[:] = 0.0

        for term in self.assembly_terms:

            dd, td = term.deriv_key
            res = term.res
            nb_sq = self.grid_idx.nb_sq
            quad_per_tri = self.element.Quadrature.nb_points

            key_res = (res, dd, td)

            tmpl = self.assembly_templates[key_res]
            nnz = tmpl.nnz
            entries_per_quad = tmpl.entries_per_quad
            sw = tmpl.w

            dep_vars_rhs = [quad_fields[v] for v in term.dep_vars]
            quad_vals = term.evaluate(*dep_vars_rhs)

            quad_val_vec = np.repeat(quad_vals.flatten(), entries_per_quad)
            sw_vec = np.tile(sw, nb_sq)

            q_vec = quad_val_vec * sw_vec

            ele_vec = q_vec.reshape(-1, quad_per_tri, entries_per_quad).sum(axis=1).reshape(-1)

            np.add.at(self._rhs_buf, nnz, ele_vec)

        return self._rhs_buf[:-1]
