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
FEM Assembly Layout - Precomputed index structures for O(nnz) assembly.

Architecture:
    FEMAssemblyLayout (top-level container)
    ├── MatrixCOOPattern      - Sparse matrix structure
    ├── RHSPattern            - RHS vector structure
    ├── jacobian_templates    - Dict[TemplateKey, JacobianTermTemplate]
    │                           Shared templates (typically 5-20)
    └── jacobian_terms        - Dict[TermKey, JacobianTermRef]
                                Lightweight refs (~80) pointing to templates

    PETScAssemblyInfo         - Extracted info for PETSc (references, not copies)

Memory optimization: Templates share structural index patterns across terms that
differ only in (res, var) block position. Actual COO indices are computed at
assembly time as: template_coo_idx + block_offset.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .scaling import ScalingInfo, build_scaling as _build_scaling

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]
Int8Array = npt.NDArray[np.int8]


@dataclass
class MatrixCOOPattern:
    """Sparse matrix COO structure with block metadata for scaling.

    Attributes
    ----------
    nnz : int
        Number of non-zero entries.
    local_rows, local_cols : IntArray
        Local indices for each COO entry. Used for dense reconstruction.
    global_rows, global_cols : IntArray
        Global indices for PETSc preallocation.
    res_block_idx : Int8Array
        Residual block index for each entry (0=mass, 1=mom_x, 2=mom_y, 3=energy).
    var_block_idx : Int8Array
        Variable block index for each entry (0=rho, 1=jx, 2=jy, 3=E).
    """
    nnz: int
    local_rows: IntArray
    local_cols: IntArray
    global_rows: IntArray
    global_cols: IntArray
    res_block_idx: Int8Array
    var_block_idx: Int8Array


@dataclass
class RHSPattern:
    """RHS vector structure for linear solver assembly and scaling.

    Maps local block-ordered RHS to global point-interleaved ordering
    required by PETSc/SciPy solvers.

    Attributes
    ----------
    global_rows : IntArray
        Global row indices for RHS vector assembly (PETSc/SciPy).
        Translates local block order to global point-interleaved order.
    res_block_idx : Int8Array
        Residual block index for diagonal scaling (0=mass, 1=mom_x, 2=mom_y, 3=energy).
    """
    global_rows: IntArray
    res_block_idx: Int8Array


@dataclass
class JacobianTermMap:
    """Contribution map for one (term, res, var) block in Jacobian.

    Attributes
    ----------
    coo_idx : IntArray
        Which COO entries receive contributions.
    weights : NDArray
        FEM integration weights.
    flat_field_idx : IntArray
        Flat index into quad field: quad_idx * nb_sq + sq_x + sq_y * sq_per_row.
        Use with res_deriv.ravel()[flat_field_idx] for field lookup.
    signs : NDArray or None
        Sign multipliers (+/-1) for derivative terms.
    """
    coo_idx: IntArray
    weights: NDArray
    flat_field_idx: IntArray
    signs: Optional[NDArray] = None


# Type alias for term keys
JacobianTermKey = Tuple[str, str, str]  # (term_name, residual, dep_var)

# Template key: (from_pattern_hash, deriv_type, der_testfun)
TemplateKey = Tuple[int, str, bool]


@dataclass
class JacobianTermTemplate:
    """Shared index template for terms with identical (from_pattern, deriv_type, der_testfun).

    Multiple JacobianTermRef objects can reference the same template when they
    share the same structural pattern (FROM indices and derivative type).
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
class JacobianTermRef:
    """Lightweight reference to a shared template (~24 bytes vs ~200 MB per JacobianTermMap).

    Stores the template reference and block offset for computing actual COO indices.

    Attributes
    ----------
    template : JacobianTermTemplate
        Shared template with base indices and weights.
    block_offset : int
        Offset into COO array: (res_idx * nb_vars + var_idx) * nnz_per_block.
    """
    template: JacobianTermTemplate
    block_offset: int


@dataclass
class COOLookup:
    """Memory-efficient COO index lookup using sorted arrays.

    Replaces Python dict with 16M+ entries (~2GB overhead) with numpy arrays
    and binary search (~0.3GB for same data).

    Attributes
    ----------
    sorted_keys : IntArray
        Composite keys (row * max_col + col), sorted for binary search.
    sorted_indices : IntArray
        COO indices corresponding to sorted_keys.
    max_col : int
        Maximum column index + 1, for computing composite keys.
    """
    sorted_keys: IntArray
    sorted_indices: IntArray
    max_col: int

    @classmethod
    def from_coo_pattern(cls, local_rows: IntArray, local_cols: IntArray) -> "COOLookup":
        """Build lookup structure from COO row/col arrays.

        Parameters
        ----------
        local_rows : IntArray
            Local row indices for COO entries.
        local_cols : IntArray
            Local column indices for COO entries.

        Returns
        -------
        COOLookup
            Lookup structure for efficient index queries.
        """
        max_col = int(local_cols.max()) + 1
        # Composite key: row * max_col + col (fits in int64 for large grids)
        coo_keys = local_rows.astype(np.int64) * max_col + local_cols
        sort_order = np.argsort(coo_keys)
        return cls(
            sorted_keys=coo_keys[sort_order],
            sorted_indices=sort_order.astype(np.int32),
            max_col=max_col,
        )

    def lookup(self, row_block: IntArray, col_block: IntArray) -> IntArray:
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
        query_keys = row_block.astype(np.int64) * self.max_col + col_block
        positions = np.searchsorted(self.sorted_keys, query_keys)
        # Handle out-of-bounds positions
        positions = np.minimum(positions, len(self.sorted_keys) - 1)
        valid = self.sorted_keys[positions] == query_keys
        return np.where(valid, self.sorted_indices[positions], -1).astype(np.int32)


@dataclass
class FEMAssemblyLayout:
    """Complete precomputed layout for FEM assembly.

    Single source of truth for all index mappings.

    Attributes
    ----------
    matrix_coo : MatrixCOOPattern
        Sparse matrix structure.
    rhs : RHSPattern
        RHS vector structure.
    jacobian_terms : dict
        Maps (term_name, residual, dep_var) -> JacobianTermRef.
        Each ref contains a lightweight reference to a shared template.
    jacobian_templates : dict
        Maps TemplateKey -> JacobianTermTemplate.
        Typically 5-20 templates shared across all ~80 term refs.
    """
    matrix_coo: MatrixCOOPattern
    rhs: RHSPattern
    jacobian_terms: Dict[JacobianTermKey, JacobianTermRef]
    jacobian_templates: Dict[TemplateKey, JacobianTermTemplate]

    def get_petsc_info(self, nb_vars: int, nb_inner_pts: int,
                       nb_global_pts: int) -> "PETScAssemblyInfo":
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
            mat_global_rows=self.matrix_coo.global_rows,
            mat_global_cols=self.matrix_coo.global_cols,
            rhs_global_rows=self.rhs.global_rows,
        )

    def build_scaling(self, char_scales: Dict[str, float],
                      variables: List[str]) -> ScalingInfo:
        """Build scaling factors from characteristic scales.

        Delegates to the standalone build_scaling function in the scaling module.

        Parameters
        ----------
        char_scales : dict
            Characteristic scale for each variable: {'rho': ρ_ref, 'jx': j_ref, ...}
        variables : list
            Variable names in block order: ['rho', 'jx', 'jy'] or with 'E'.

        Returns
        -------
        ScalingInfo
            Precomputed scaling factors for COO values, RHS, and solution.
        """
        return _build_scaling(char_scales, variables, self.matrix_coo, self.rhs)


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
