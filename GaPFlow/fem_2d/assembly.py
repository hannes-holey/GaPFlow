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

# flake8: noqa: W503

"""
FEM Assembly - unified sparse assembly structure.

Combines COO pattern, RHS pattern, and term template mappings into a single
Assembly class with all precomputation logic.

Architecture:
    Assembly (top-level container)
    ├── COO arrays           - local/global rows/cols, block indices
    ├── RHS arrays           - global rows, block indices
    ├── coo_lookup()         - Memory-efficient COO index lookup
    ├── generic_templates    - Dict[TemplateKey, AssemblyGenericTemplate]
    │                          Shared templates (typically 5-20)
    └── term_templates       - Dict[TermKey, AssemblyTermTemplate]
                               Lightweight refs (~80) pointing to templates

    PETScAssemblyInfo        - Extracted info for PETSc (references, not copies)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .elements import TriangleQuadrature
from .terms import NonLinearTerm
from .grid_index import GridIndexManager

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]
Int8Array = npt.NDArray[np.int8]

# Type aliases for template keys
TermKey = Tuple[str, str, str]      # (term_name, residual, dep_var)
TemplateKey = Tuple[int, str, bool]  # (from_pattern_hash, deriv_type, der_testfun)


@dataclass
class AssemblyGenericTemplate:
    """Shared index template for terms with identical (from_pattern, deriv_type, der_testfun).

    Multiple AssemblyTermTemplate objects can reference the same generic template
    when they share the same structural pattern (FROM indices and derivative type).
    The actual COO indices are computed at assembly time as:
        actual_coo_idx = template_coo_idx + block_offset

    Attributes
    ----------
    template_coo_idx : IntArray
        Base COO indices for block (res=0, var=0).
    weights : NDArray
        FEM integration weights.
    flat_field_idx : IntArray
        Flat index into quad field: quad_idx * nb_sq + sq_x * sq_per_col + sq_y.
        Use with res_deriv.ravel()[flat_field_idx] for field lookup.
    signs : NDArray or None
        Sign multipliers (+/-1) for derivative terms.
    nnz_per_block : int
        Number of non-zeros per (res, var) block, for computing block offsets.
    """
    template_coo_idx: IntArray
    weights: NDArray
    flat_field_idx: IntArray
    signs: Optional[NDArray]
    nnz_per_block: int


@dataclass
class AssemblyTermTemplate:
    """Lightweight reference to a shared AssemblyGenericTemplate.

    Stores the template reference and block offset for computing actual COO indices.

    Attributes
    ----------
    template : AssemblyGenericTemplate
        Shared template with base indices and weights.
    block_offset : int
        Offset into COO array: (res_idx * nb_vars + var_idx) * nnz_per_block.
    """
    template: AssemblyGenericTemplate
    block_offset: int


@dataclass
class PETScAssemblyInfo:
    """Minimal info for PETSc system creation and assembly.

    Attributes
    ----------
    local_size : int
        Local matrix/vector size for this rank.
    global_size : int
        Global matrix/vector size.
    mat_global_rows, mat_global_cols : IntArray
        Global COO indices for matrix preallocation.
    rhs_global_rows : IntArray
        Global row indices for RHS assembly.
    """
    local_size: int
    global_size: int
    mat_global_rows: IntArray
    mat_global_cols: IntArray
    rhs_global_rows: IntArray


class Assembly:
    """Unified FEM assembly structure.

    Holds all precomputed index structures for O(nnz) matrix/vector assembly:
    - COO pattern arrays (local/global rows/cols, block indices)
    - RHS pattern arrays
    - Shared generic templates and per-term template references
    - COO index lookup for template construction

    Parameters
    ----------
    grid_idx : GridIndexManager
        Grid index manager with masks, connectivity, BCs.
    quad : TriangleQuadrature
        Quadrature data with element tensors.
    terms : list of NonLinearTerm
        Active PDE terms.
    residuals : list of str
        Residual names in block order.
    variables : list of str
        Variable names in block order.
    decomp : DomainDecomposition
        Domain decomposition (for local-to-global mapping).
    """

    def __init__(self, grid_idx: GridIndexManager,
                 quad: TriangleQuadrature,
                 terms: List[NonLinearTerm],
                 residuals: List[str],
                 variables: List[str]
                 ) -> None:
        nb_vars = len(variables)
        nb_res = len(residuals)
        l2g_pts = grid_idx.l2g_list

        # 1. Build block connectivity
        # parallel arrays with all valid (inner_pt, contrib_pt) pairs using LOCAL indexing
        inner_pts, contrib_pts = self._build_block_connectivity(grid_idx)
        self._nnz_per_block = len(inner_pts)

        # 2. Build matrix COO pattern
        # building upon block connectivity to create full local/global COO arrays 
        # and block indices. These are not sorted in any particular order since
        # the COO lookup handles the matching during template construction
        self._build_coo_pattern(grid_idx, inner_pts, contrib_pts, nb_vars, nb_res)

        # 3. Build COO lookup for template construction
        self._build_coo_lookup()

        # 4. Build RHS pattern
        self._build_rhs_pattern(grid_idx, l2g_pts, nb_res)

        # 5. Build shared generic templates (grouped by FROM pattern)
        self.generic_templates, var_to_from_hash = self._build_generic_templates(
            grid_idx, quad, variables)

        # 6. Create per-term template references
        self.term_templates = self._build_term_templates(
            terms, residuals, variables, var_to_from_hash, nb_vars)

    # =========================================================================
    # COO Pattern Construction
    # =========================================================================

    @staticmethod
    def _build_block_connectivity(grid_idx: GridIndexManager
                                    ) -> Tuple[IntArray, IntArray]:
        """Build FEM block connectivity using elementary stencil.

        Parallel arrays with all valid (inner_pt, contrib_pt) pairs using 
        LOCAL indexing. No specific sorting done or required since COO lookup
        handles matching in the assembly template construction.
        """
        m_inner = grid_idx.index_mask_inner_local
        m_padded = grid_idx._index_mask_padded_local('')

        inner_list = []
        contrib_list = []

        inner_pts_2d = np.argwhere(m_inner >= 0)

        for dx, dy in TriangleQuadrature.STENCIL_OFFSETS:
            nx = inner_pts_2d[:, 0] + dx
            ny = inner_pts_2d[:, 1] + dy

            inner_idx = m_inner[inner_pts_2d[:, 0], inner_pts_2d[:, 1]]
            contrib_idx = m_padded[nx, ny]

            valid = contrib_idx >= 0
            inner_list.append(inner_idx[valid])
            contrib_list.append(contrib_idx[valid])

        return np.concatenate(inner_list), np.concatenate(contrib_list)

    def _build_coo_pattern(self,
                           grid_idx: GridIndexManager,
                           inner_pts: IntArray,
                           contrib_pts: IntArray,
                           nb_vars: int,
                           nb_res: int
                           ) -> None:
        """Build sparse matrix COO pattern arrays.
        
        Building upon block connectivity to create full local/global COO arrays 
        and block indices these are not sorted in any particular order since
        the COO lookup handles matching during template construction"""
        nnz_per_block = self._nnz_per_block
        total_nnz = nb_res * nb_vars * nnz_per_block

        local_rows = np.empty(total_nnz, dtype=np.int32)
        local_cols = np.empty(total_nnz, dtype=np.int32)
        global_rows = np.empty(total_nnz, dtype=np.int32)
        global_cols = np.empty(total_nnz, dtype=np.int32)
        res_block_idx = np.empty(total_nnz, dtype=np.int8)
        var_block_idx = np.empty(total_nnz, dtype=np.int8)

        global_inner = grid_idx.l2g_list[inner_pts]
        global_contrib = grid_idx.l2g_list[contrib_pts]

        idx = 0
        n = nnz_per_block

        for res_idx in range(nb_res):
            for var_idx in range(nb_vars):
                # Local block ordering (res/var major)
                local_rows[idx:idx + n] = inner_pts + res_idx * grid_idx.nb_inner_pts
                local_cols[idx:idx + n] = contrib_pts + var_idx * grid_idx.nb_contributors

                # Global point-interleaved ordering (point-major)
                global_rows[idx:idx + n] = global_inner * nb_res + res_idx
                global_cols[idx:idx + n] = global_contrib * nb_vars + var_idx

                # Block indices for scaling
                res_block_idx[idx:idx + n] = res_idx
                var_block_idx[idx:idx + n] = var_idx

                idx += n

        self.nnz: int = total_nnz
        self.local_rows: IntArray = local_rows
        self.local_cols: IntArray = local_cols
        self.global_rows: IntArray = global_rows
        self.global_cols: IntArray = global_cols
        self.res_block_idx: Int8Array = res_block_idx
        self.var_block_idx: Int8Array = var_block_idx

    def _build_coo_lookup(self) -> None:
        """Build memory-efficient COO index lookup using sorted arrays.

        Only indexes block (0,0) since all lookup queries use base indices;
        callers add block offsets separately via AssemblyTermTemplate.
        """
        n = self._nnz_per_block
        rows = self.local_rows[:n]
        cols = self.local_cols[:n]
        self._coo_max_col = int(cols.max()) + 1
        coo_keys = rows.astype(np.int64) * self._coo_max_col + cols
        sort_order = np.argsort(coo_keys)
        self._coo_sorted_keys = coo_keys[sort_order]
        self._coo_sorted_indices = sort_order.astype(np.int32)

    def coo_lookup(self, row_block: IntArray, col_block: IntArray) -> IntArray:
        """Vectorized COO index lookup.

        Parameters
        ----------
        row_block : IntArray
            Row indices to look up.
        col_block : IntArray
            Column indices to look up.

        Returns
        -------
        IntArray
            COO indices for each (row, col) pair, or -1 if not found.
        """
        query_keys = row_block.astype(np.int64) * self._coo_max_col + col_block
        positions = np.searchsorted(self._coo_sorted_keys, query_keys)
        positions = np.minimum(positions, len(self._coo_sorted_keys) - 1)
        valid = self._coo_sorted_keys[positions] == query_keys
        return np.where(valid, self._coo_sorted_indices[positions],
                        -1).astype(np.int32)

    # =========================================================================
    # RHS Pattern Construction
    # =========================================================================

    def _build_rhs_pattern(self, grid_idx: "GridIndexManager",
                           l2g_pts: IntArray, nb_res: int) -> None:
        """Build RHS vector assembly pattern."""
        nb_inner_pts = grid_idx.nb_inner_pts
        local_size = nb_res * nb_inner_pts
        global_rows = np.empty(local_size, dtype=np.int32)
        res_block_idx = np.empty(local_size, dtype=np.int8)
        global_pts = l2g_pts[:nb_inner_pts]

        for res_idx in range(nb_res):
            start = res_idx * nb_inner_pts
            end = start + nb_inner_pts
            global_rows[start:end] = global_pts * nb_res + res_idx
            res_block_idx[start:end] = res_idx

        self.rhs_global_rows: IntArray = global_rows
        self.rhs_res_block_idx: Int8Array = res_block_idx

    # =========================================================================
    # Template Construction
    # =========================================================================

    def _build_generic_templates(self, grid_idx: "GridIndexManager",
                                 quad: TriangleQuadrature,
                                 variables: List[str]
                                 ) -> Tuple[Dict[TemplateKey, AssemblyGenericTemplate], Dict[str, int]]:
        """Build shared templates grouped by unique FROM pattern.

        Returns
        -------
        generic_templates : dict
            Maps (from_hash, deriv_type, der_testfun) -> AssemblyGenericTemplate
        var_to_from_hash : dict
            Maps variable name -> FROM pattern hash
        """
        from_pattern_to_vars: Dict[int, List[str]] = {}
        var_to_from_hash: Dict[str, int] = {}

        for var in variables:
            from_arr = grid_idx.sq_FROM_padded(var)
            from_hash = hash(from_arr.tobytes())
            var_to_from_hash[var] = from_hash
            if from_hash not in from_pattern_to_vars:
                from_pattern_to_vars[from_hash] = []
            from_pattern_to_vars[from_hash].append(var)

        generic_templates: Dict[TemplateKey, AssemblyGenericTemplate] = {}
        for from_hash, vars_with_pattern in from_pattern_to_vars.items():
            representative_var = vars_with_pattern[0]

            for deriv_type in ['none', 'dx', 'dy']:
                for der_testfun in [False, True]:
                    if deriv_type == 'none' and der_testfun:
                        continue

                    template_key = (from_hash, deriv_type, der_testfun)
                    if deriv_type == 'none':
                        generic_templates[template_key] = self._build_template_zero_der(
                            grid_idx, quad, representative_var)
                    else:
                        direction = 'x' if deriv_type == 'dx' else 'y'
                        generic_templates[template_key] = self._build_template_deriv(
                            grid_idx, quad, representative_var,
                            direction, der_testfun)

            # Test-function-derivative-only templates (PSPG)
            for testfun_dir in ['x', 'y']:
                template_key = (from_hash, 'testfun_d' + testfun_dir, False)
                generic_templates[template_key] = self._build_template_testfun_deriv(
                    grid_idx, quad, representative_var, testfun_dir)

            # Cross-term templates: residual deriv in one dir, test fn in another
            for deriv_type in ['dx', 'dy']:
                for testfun_dir in ['x', 'y']:
                    template_key = (from_hash, deriv_type, testfun_dir)
                    direction = 'x' if deriv_type == 'dx' else 'y'
                    generic_templates[template_key] = self._build_template_deriv(
                        grid_idx, quad, representative_var,
                        direction, testfun_dir)

        return generic_templates, var_to_from_hash

    def _build_term_templates(self, terms: List[NonLinearTerm],
                              residuals: List[str], variables: List[str],
                              var_to_from_hash: Dict[str, int],
                              nb_vars: int
                              ) -> Dict[TermKey, AssemblyTermTemplate]:
        """Create lightweight refs for each (term, res, dep_var) combination."""
        term_templates: Dict[TermKey, AssemblyTermTemplate] = {}
        nnz_per_block = self._nnz_per_block

        for term in terms:
            for dep_var in term.dep_vars:
                term_key = (term.name, term.res, dep_var)
                deriv_type, der_testfun = self._get_template_key(term)
                from_hash = var_to_from_hash[dep_var]
                template_key = (from_hash, deriv_type, der_testfun)

                res_idx = residuals.index(term.res)
                var_idx = variables.index(dep_var)
                block_offset = (res_idx * nb_vars + var_idx) * nnz_per_block

                term_templates[term_key] = AssemblyTermTemplate(
                    template=self.generic_templates[template_key],
                    block_offset=block_offset,
                )

        return term_templates

    @staticmethod
    def _get_template_key(term: NonLinearTerm) -> Tuple[str, object]:
        """Get (deriv_type, der_testfun) template key from term.

        Returns (deriv_type, der_testfun) where:
        - deriv_type: 'none', 'dx', 'dy', 'testfun_dx', or 'testfun_dy'
        - der_testfun: False, True (legacy: same dir as residual deriv),
          or 'x'/'y' (explicit test fn derivative direction, for cross-terms)
        """
        # Resolve residual derivative direction first
        if term.d_dx_resfun:
            deriv_type = 'dx'
        elif term.d_dy_resfun:
            deriv_type = 'dy'
        else:
            deriv_type = 'none'
        # Test-function-derivative-only (PSPG temporal/wall-shear style)
        if deriv_type == 'none' and term.der_testfun in ('x', 'y'):
            return ('testfun_d' + term.der_testfun, False)
        return (deriv_type, term.der_testfun)

    @staticmethod
    def _flat_field_idx(nb_sq: int, sq_per_col: int,
                        quad_idx: int, sq_x: IntArray,
                        sq_y: IntArray) -> IntArray:
        """Flat index into quad field array: maps (quad_pt, sq_x, sq_y) -> 1D."""
        return (quad_idx * nb_sq + sq_x * sq_per_col + sq_y).astype(np.int32)

    def _build_template_zero_der(self, grid_idx: "GridIndexManager",
                                 quad: TriangleQuadrature,
                                 representative_var: str
                                 ) -> AssemblyGenericTemplate:
        """Build template for zero-derivative term (3x3 per triangle)."""
        sq_x, sq_y = grid_idx.sq_x_arr, grid_idx.sq_y_arr
        nb_sq, sq_per_col = grid_idx.nb_sq, grid_idx.sq_per_col

        TO = grid_idx.sq_TO_inner
        FROM = grid_idx.sq_FROM_padded(representative_var)

        coo_idx_list = []
        weights_list = []
        flat_field_idx_list = []

        for t in range(2):
            TO_tri = TO[:, quad.TRI_PTS[t]]
            FROM_tri = FROM[:, quad.TRI_PTS[t]]
            elem_tensor = quad.elem_tensor[t]
            quad_offset = t * 3

            for i in range(3):
                for j in range(3):
                    valid = (TO_tri[:, i] >= 0) & (FROM_tri[:, j] >= 0)
                    coo_idx = self.coo_lookup(TO_tri[valid, i],
                                              FROM_tri[valid, j])
                    sq_x_valid = sq_x[valid]
                    sq_y_valid = sq_y[valid]
                    for quad_pt in range(3):
                        coo_idx_list.append(coo_idx)
                        weights_list.append(np.full(
                            len(coo_idx), elem_tensor[i, j, quad_pt],
                            dtype=np.float64))
                        flat_field_idx_list.append(
                            self._flat_field_idx(nb_sq, sq_per_col,
                                                 quad_pt + quad_offset,
                                                 sq_x_valid, sq_y_valid))

        return AssemblyGenericTemplate(
            template_coo_idx=np.concatenate(coo_idx_list),
            weights=np.concatenate(weights_list),
            flat_field_idx=np.concatenate(flat_field_idx_list),
            signs=None,
            nnz_per_block=self._nnz_per_block,
        )

    def _build_template_testfun_deriv(self, grid_idx: "GridIndexManager",
                                      quad: TriangleQuadrature,
                                      representative_var: str,
                                      direction: str,
                                      ) -> AssemblyGenericTemplate:
        """Build template for test-function-derivative-only term (PSPG).

        Structurally identical to zero-derivative template but uses
        dN_i/dx_k * N_j * w * A element tensors instead of N_i * N_j * w * A.
        """
        sq_x, sq_y = grid_idx.sq_x_arr, grid_idx.sq_y_arr
        nb_sq, sq_per_col = grid_idx.nb_sq, grid_idx.sq_per_col

        TO = grid_idx.sq_TO_inner
        FROM = grid_idx.sq_FROM_padded(representative_var)

        et = quad.elem_tensor_testfun_dx if direction == 'x' \
            else quad.elem_tensor_testfun_dy

        coo_idx_list = []
        weights_list = []
        flat_field_idx_list = []

        for t in range(2):
            TO_tri = TO[:, quad.TRI_PTS[t]]
            FROM_tri = FROM[:, quad.TRI_PTS[t]]
            elem_tensor = et[t]
            quad_offset = t * 3

            for i in range(3):
                for j in range(3):
                    valid = (TO_tri[:, i] >= 0) & (FROM_tri[:, j] >= 0)
                    coo_idx = self.coo_lookup(TO_tri[valid, i],
                                              FROM_tri[valid, j])
                    sq_x_valid = sq_x[valid]
                    sq_y_valid = sq_y[valid]
                    for quad_pt in range(3):
                        coo_idx_list.append(coo_idx)
                        weights_list.append(np.full(
                            len(coo_idx), elem_tensor[i, j, quad_pt],
                            dtype=np.float64))
                        flat_field_idx_list.append(
                            self._flat_field_idx(nb_sq, sq_per_col,
                                                 quad_pt + quad_offset,
                                                 sq_x_valid, sq_y_valid))

        return AssemblyGenericTemplate(
            template_coo_idx=np.concatenate(coo_idx_list),
            weights=np.concatenate(weights_list),
            flat_field_idx=np.concatenate(flat_field_idx_list),
            signs=None,
            nnz_per_block=self._nnz_per_block,
        )

    def _build_template_deriv(self, grid_idx: "GridIndexManager",
                              quad: TriangleQuadrature,
                              representative_var: str,
                              direction: str, der_testfun
                              ) -> AssemblyGenericTemplate:
        """Build template for derivative term (+/-1/d* stencil)."""
        sq_x, sq_y = grid_idx.sq_x_arr, grid_idx.sq_y_arr
        nb_sq, sq_per_col = grid_idx.nb_sq, grid_idx.sq_per_col

        TO = grid_idx.sq_TO_inner
        FROM = grid_idx.sq_FROM_padded(representative_var)

        # Stencil direction: always from 'direction'
        deriv_nodes = quad.DERIV_NODES_DX if direction == 'x' \
            else quad.DERIV_NODES_DY

        # Test function weights: decoupled from stencil direction
        if der_testfun is True:
            # Legacy (PSPG pressure): test weight dir matches stencil dir
            test_wA_arr = quad.test_wA_dx if direction == 'x' \
                else quad.test_wA_dy
        elif der_testfun == 'x':
            test_wA_arr = quad.test_wA_dx
        elif der_testfun == 'y':
            test_wA_arr = quad.test_wA_dy
        else:
            test_wA_arr = quad.test_wA

        coo_idx_list = []
        weights_list = []
        flat_field_idx_list = []
        signs_list = []

        for t in range(2):
            TO_tri = TO[:, quad.TRI_PTS[t]]
            quad_offset = t * 3
            test_wA = test_wA_arr[t]

            # Derive neg/pos corners in square indexing
            c0, c1 = quad.TRI_PTS[t, deriv_nodes]
            if quad.DERIV_SIGNS[t] > 0:
                neg_corner, pos_corner = c0, c1
            else:
                neg_corner, pos_corner = c1, c0
            FROM_neg = FROM[:, neg_corner]
            FROM_pos = FROM[:, pos_corner]
            for i in range(3):
                for FROM_pts, sign in [(FROM_neg, -1.0), (FROM_pos, 1.0)]:
                    valid = (TO_tri[:, i] >= 0) & (FROM_pts >= 0)
                    coo_idx = self.coo_lookup(TO_tri[valid, i],
                                              FROM_pts[valid])
                    sq_x_valid = sq_x[valid]
                    sq_y_valid = sq_y[valid]
                    for quad_pt in range(3):
                        coo_idx_list.append(coo_idx)
                        weights_list.append(np.full(
                            len(coo_idx), test_wA[i, quad_pt],
                            dtype=np.float64))
                        flat_field_idx_list.append(
                            self._flat_field_idx(nb_sq, sq_per_col,
                                                 quad_pt + quad_offset,
                                                 sq_x_valid, sq_y_valid))
                        signs_list.append(np.full(len(coo_idx), sign,
                                                  dtype=np.float64))

        return AssemblyGenericTemplate(
            template_coo_idx=np.concatenate(coo_idx_list),
            weights=np.concatenate(weights_list),
            flat_field_idx=np.concatenate(flat_field_idx_list),
            signs=np.concatenate(signs_list),
            nnz_per_block=self._nnz_per_block,
        )

    # =========================================================================
    # PETSc Interface
    # =========================================================================

    def get_petsc_info(self, nb_vars: int, nb_inner_pts: int,
                       nb_global_pts: int) -> PETScAssemblyInfo:
        """Extract PETSc-specific assembly info.

        Parameters
        ----------
        nb_vars : int
            Number of variables (3 or 4 with energy).
        nb_inner_pts : int
            Number of inner points on this rank.
        nb_global_pts : int
            Total global grid points.
        """
        return PETScAssemblyInfo(
            local_size=nb_vars * nb_inner_pts,
            global_size=nb_vars * nb_global_pts,
            mat_global_rows=self.global_rows,
            mat_global_cols=self.global_cols,
            rhs_global_rows=self.rhs_global_rows,
        )
