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
    ├── MatrixCOOPattern  - Sparse matrix structure
    ├── RHSPattern        - RHS vector structure
    ├── jacobian_terms    - Dict[TermKey, JacobianTermMap]
    └── rhs_terms         - Dict[TermKey, RHSTermMap]

    PETScAssemblyInfo     - Extracted info for PETSc (references, not copies)
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

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
    """RHS vector structure with block metadata for scaling.

    Attributes
    ----------
    size : int
        Local RHS size (nb_vars * nb_inner_pts).
    global_rows : IntArray
        Global row indices for PETSc assembly.
    res_block_idx : Int8Array
        Residual block index for each entry (0=mass, 1=mom_x, 2=mom_y, 3=energy).
    """
    size: int
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


@dataclass
class RHSTermMap:
    """Contribution map for one (term, res) block in RHS.

    Attributes
    ----------
    output_idx : IntArray
        Which R[i] entries receive contributions.
    weights : NDArray
        FEM integration weights.
    flat_field_idx : IntArray
        Flat index into quad field: quad_idx * nb_sq + sq_x + sq_y * sq_per_row.
        Use with res_vals.ravel()[flat_field_idx] for field lookup.
    """
    output_idx: IntArray
    weights: NDArray
    flat_field_idx: IntArray


# Type alias for term keys
JacobianTermKey = Tuple[str, str, str]  # (term_name, residual, dep_var)
RHSTermKey = Tuple[str, str]            # (term_name, residual)


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
class ScalingInfo:
    """Precomputed scaling factors for linear system conditioning.

    Transforms the linear system J·dq = -R into a well-conditioned
    scaled system J*·dq* = -R* where all matrix entries are O(1).

    The transformation is:
        J*[i,j] = J[i,j] · D_q[j] / D_R[i]
        R*[i] = R[i] / D_R[i]
        dq[j] = dq*[j] · D_q[j]

    Attributes
    ----------
    coo_scale : NDArray
        Scale factors for COO values, shape (nnz,).
    rhs_scale : NDArray
        Scale factors for residual vector, shape (rhs_size,).
    sol_scale : NDArray
        Scale factors for solution vector, shape (sol_size,).
    char_scales : dict
        Original characteristic scales for reference.
    """
    coo_scale: NDArray
    rhs_scale: NDArray
    sol_scale: NDArray
    char_scales: Dict[str, float]

    def scale_system(self, M_coo: NDArray, R: NDArray) -> Tuple[NDArray, NDArray]:
        """Scale Jacobian COO values and residual vector.

        Parameters
        ----------
        M_coo : NDArray, shape (nnz,)
            Unscaled Jacobian in COO value format.
        R : NDArray, shape (n,)
            Unscaled residual vector.

        Returns
        -------
        M_scaled : NDArray, shape (nnz,)
            Scaled Jacobian COO values.
        R_scaled : NDArray, shape (n,)
            Scaled residual vector.
        """
        return M_coo * self.coo_scale, R / self.rhs_scale

    def unscale_solution(self, dq_scaled: NDArray) -> NDArray:
        """Recover physical solution increment from scaled solution.

        Parameters
        ----------
        dq_scaled : NDArray, shape (n,)
            Solution of the scaled system.

        Returns
        -------
        dq : NDArray, shape (n,)
            Physical solution increment.
        """
        return dq_scaled * self.sol_scale


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
        Maps (term, res, var) -> JacobianTermMap.
    rhs_terms : dict
        Maps (term, res) -> RHSTermMap.
    dx, dy : float
        Grid spacing for derivative scaling.
    nb_vars : int
        Number of variables (3 or 4 with energy).
    nb_inner_pts : int
        Number of inner points on this rank.
    nb_global_pts : int
        Total global grid points.
    nb_sq : int
        Number of square elements (sq_per_row * sq_per_col).
    sq_per_row : int
        Squares per row (for flat index computation).
    """
    matrix_coo: MatrixCOOPattern
    rhs: RHSPattern
    jacobian_terms: Dict[JacobianTermKey, JacobianTermMap]
    rhs_terms: Dict[RHSTermKey, RHSTermMap]
    dx: float
    dy: float
    nb_vars: int
    nb_inner_pts: int
    nb_global_pts: int
    nb_sq: int
    sq_per_row: int
    sq_per_col: int

    def get_petsc_info(self) -> "PETScAssemblyInfo":
        """Extract PETSc-specific assembly info."""
        return PETScAssemblyInfo(
            local_size=self.nb_vars * self.nb_inner_pts,
            global_size=self.nb_vars * self.nb_global_pts,
            mat_global_rows=self.matrix_coo.global_rows,
            mat_global_cols=self.matrix_coo.global_cols,
            rhs_global_rows=self.rhs.global_rows,
        )

    def build_scaling(self, char_scales: Dict[str, float],
                      variables: List[str]) -> ScalingInfo:
        """Build scaling factors from characteristic scales.

        Uses the block indices stored in matrix_coo and rhs patterns
        to efficiently compute per-entry scale factors.

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

        Raises
        ------
        ValueError
            If char_scales is missing entries or contains non-positive values.
        """
        # Validation
        for var in variables:
            if var not in char_scales:
                raise ValueError(f"Missing characteristic scale for '{var}'")
            if char_scales[var] <= 0:
                raise ValueError(
                    f"Characteristic scale for '{var}' must be positive, "
                    f"got {char_scales[var]}"
                )

        # Per-block scale arrays
        q_scales = np.array([char_scales[v] for v in variables], dtype=np.float64)
        r_scales = q_scales  # Residual scales match corresponding variable

        # COO scaling factors: J*[k] = J[k] * D_q[var_idx] / D_R[res_idx]
        coo_scale = (q_scales[self.matrix_coo.var_block_idx]
                     / r_scales[self.matrix_coo.res_block_idx])

        # RHS scaling factors
        rhs_scale = r_scales[self.rhs.res_block_idx]

        # Solution scaling factors (same block structure as RHS)
        sol_scale = q_scales[self.rhs.res_block_idx]

        return ScalingInfo(
            coo_scale=coo_scale,
            rhs_scale=rhs_scale,
            sol_scale=sol_scale,
            char_scales=char_scales,
        )


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
