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
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]


@dataclass
class MatrixCOOPattern:
    """Sparse matrix COO structure.

    Attributes
    ----------
    nnz : int
        Number of non-zero entries.
    local_rows, local_cols : IntArray
        Local indices for each COO entry. Used for dense reconstruction.
    global_rows, global_cols : IntArray
        Global indices for PETSc preallocation.
    """
    nnz: int
    local_rows: IntArray
    local_cols: IntArray
    global_rows: IntArray
    global_cols: IntArray


@dataclass
class RHSPattern:
    """RHS vector assembly structure.

    Attributes
    ----------
    size : int
        Local RHS size (nb_vars * nb_inner_pts).
    global_rows : IntArray
        Global row indices for PETSc assembly.
    """
    size: int
    global_rows: IntArray


@dataclass
class JacobianTermMap:
    """Contribution map for one (term, res, var) block in Jacobian.

    Attributes
    ----------
    coo_idx : IntArray
        Which COO entries receive contributions.
    weights : NDArray
        FEM integration weights.
    quad_idx : IntArray
        Quadrature point index (0-5).
    sq_x, sq_y : IntArray
        Square indices for field value lookup.
    signs : NDArray or None
        Sign multipliers (+/-1) for derivative terms.
    """
    coo_idx: IntArray
    weights: NDArray
    quad_idx: IntArray
    sq_x: IntArray
    sq_y: IntArray
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
    quad_idx : IntArray
        Quadrature point index (0-5).
    sq_x, sq_y : IntArray
        Square indices for field value lookup.
    """
    output_idx: IntArray
    weights: NDArray
    quad_idx: IntArray
    sq_x: IntArray
    sq_y: IntArray


# Type alias for term keys
JacobianTermKey = Tuple[str, str, str]  # (term_name, residual, dep_var)
RHSTermKey = Tuple[str, str]            # (term_name, residual)


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

    def get_petsc_info(self) -> "PETScAssemblyInfo":
        """Extract PETSc-specific assembly info."""
        return PETScAssemblyInfo(
            local_size=self.nb_vars * self.nb_inner_pts,
            global_size=self.nb_vars * self.nb_global_pts,
            mat_global_rows=self.matrix_coo.global_rows,
            mat_global_cols=self.matrix_coo.global_cols,
            rhs_global_rows=self.rhs.global_rows,
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
