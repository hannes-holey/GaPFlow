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

# Structural key determines JacobianTermMap identity
# Terms with same structural key have identical index patterns
StructuralKey = Tuple[str, str, str, bool]  # (residual, dep_var, deriv_type, der_testfun)


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
class JacobianMapCache:
    """Cache of canonical JacobianTermMaps with deduplication.

    Multiple terms can share the same JacobianTermMap if they have identical
    structural properties (residual, variable, derivative type, der_testfun).
    This reduces memory usage by 12-40% depending on the term configuration.

    Attributes
    ----------
    _canonical_maps : dict
        Maps StructuralKey -> JacobianTermMap (the actual arrays).
    _term_refs : dict
        Maps JacobianTermKey -> StructuralKey (references to canonical maps).
    """
    _canonical_maps: Dict[StructuralKey, JacobianTermMap]
    _term_refs: Dict[JacobianTermKey, StructuralKey]

    @classmethod
    def empty(cls) -> "JacobianMapCache":
        """Create an empty cache."""
        return cls(_canonical_maps={}, _term_refs={})

    def register(self, term_key: JacobianTermKey, structural_key: StructuralKey,
                 map_or_builder) -> None:
        """Register a term, building canonical map if needed.

        Parameters
        ----------
        term_key : JacobianTermKey
            The (term_name, residual, dep_var) key for this term.
        structural_key : StructuralKey
            The structural key that determines map identity.
        map_or_builder : JacobianTermMap or callable
            Either the prebuilt map or a callable that builds it.
        """
        if structural_key not in self._canonical_maps:
            # First term with this structure - build/store the canonical map
            if callable(map_or_builder):
                self._canonical_maps[structural_key] = map_or_builder()
            else:
                self._canonical_maps[structural_key] = map_or_builder

        self._term_refs[term_key] = structural_key

    def get(self, term_key: JacobianTermKey) -> JacobianTermMap:
        """Get the JacobianTermMap for a term (may be shared).

        Parameters
        ----------
        term_key : JacobianTermKey
            The (term_name, residual, dep_var) key.

        Returns
        -------
        JacobianTermMap
            The term map (possibly shared with other terms).
        """
        structural_key = self._term_refs[term_key]
        return self._canonical_maps[structural_key]

    def __contains__(self, term_key: JacobianTermKey) -> bool:
        """Check if a term key is registered."""
        return term_key in self._term_refs

    def __getitem__(self, term_key: JacobianTermKey) -> JacobianTermMap:
        """Get JacobianTermMap by term key (dict-like access)."""
        return self.get(term_key)

    def __len__(self) -> int:
        """Return number of registered term keys."""
        return len(self._term_refs)

    def __iter__(self):
        """Iterate over term keys."""
        return iter(self._term_refs)

    def keys(self):
        """Return all registered term keys."""
        return self._term_refs.keys()

    def items(self):
        """Iterate over (term_key, JacobianTermMap) pairs."""
        for term_key, structural_key in self._term_refs.items():
            yield term_key, self._canonical_maps[structural_key]

    def values(self):
        """Iterate over JacobianTermMaps (with duplicates for shared maps)."""
        for structural_key in self._term_refs.values():
            yield self._canonical_maps[structural_key]

    def memory_stats(self) -> Dict[str, float]:
        """Report memory usage statistics.

        Returns
        -------
        dict
            Memory statistics including actual usage and savings.
        """
        canonical_mem = sum(
            m.coo_idx.nbytes + m.weights.nbytes + m.flat_field_idx.nbytes +
            (m.signs.nbytes if m.signs is not None else 0)
            for m in self._canonical_maps.values()
        )

        num_canonical = len(self._canonical_maps)
        num_refs = len(self._term_refs)

        # Estimate what memory would be without sharing
        if num_canonical > 0:
            avg_map_size = canonical_mem / num_canonical
            unshared_mem = avg_map_size * num_refs
        else:
            unshared_mem = 0

        return {
            'canonical_maps': num_canonical,
            'term_refs': num_refs,
            'actual_memory_mb': canonical_mem / 1024 / 1024,
            'unshared_memory_mb': unshared_mem / 1024 / 1024,
            'savings_mb': (unshared_mem - canonical_mem) / 1024 / 1024,
            'savings_percent': 100 * (1 - num_canonical / num_refs) if num_refs > 0 else 0,
        }


@dataclass
class FEMAssemblyLayout:
    """Complete precomputed layout for FEM assembly.

    Single source of truth for all index mappings. Uses JacobianMapCache
    for Jacobian term maps to deduplicate identical index structures,
    reducing memory usage by 12-40%.

    Attributes
    ----------
    matrix_coo : MatrixCOOPattern
        Sparse matrix structure.
    rhs : RHSPattern
        RHS vector structure.
    jacobian_terms : JacobianMapCache
        Cache mapping (term, res, var) -> JacobianTermMap with deduplication.
        Use jacobian_terms.get(key) to retrieve maps.
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
    jacobian_terms: JacobianMapCache
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
