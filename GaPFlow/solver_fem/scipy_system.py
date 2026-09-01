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
"""SciPy-based sparse linear solver for serial Taylor-Hood Q2Q1 execution."""

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import splu, gmres

if TYPE_CHECKING:
    from .assembly import GlobalIndexPattern, Assembly

NDArray = npt.NDArray[np.floating]


class ScipySystem:
    """Serial sparse linear solver using SciPy (fallback when PETSc unavailable).

    Provides the same .assemble() / .solve() interface as PETScSystem.
    Unlike the old version, solve() returns a flat 1-D solution vector so
    that the caller (FEMSolver) can unpack per-variable slices
    of unequal length.

    Parameters
    ----------
    info : GlobalIndexPattern
        Precomputed assembly info (sizes and global indices).
    solver_type : str, optional
        "direct" (SuperLU) or "iterative" (GMRES). Default: "direct".
    """

    def __init__(self, info: "GlobalIndexPattern", solver_type: str = "direct",
                 assembly: "Assembly" = None):
        self._info = info
        self._solver_type = solver_type
        self._size = info.local_size
        self._rhs_rows = info.rhs_global_rows
        self._assembly = assembly

        self._rhs: NDArray = np.zeros(self._size)
        self._iterations = 0
        self._converged = True

        # Build CSC structure once from the fixed sparsity pattern.
        dummy = csc_matrix(
            (np.ones(len(info.mat_global_rows), dtype=np.float64),
             (info.mat_global_rows, info.mat_global_cols)),
            shape=(self._size, self._size),
        )
        dummy.sum_duplicates()
        dummy.sort_indices()
        self._mat = dummy

        # Permutation: _mat.data[i] holds coo_values[_coo_to_csc[i]]
        coo = coo_matrix(
            (np.arange(len(info.mat_global_rows), dtype=np.int32),
             (info.mat_global_rows, info.mat_global_cols)),
            shape=(self._size, self._size),
        )
        csc_perm = coo.tocsc()
        csc_perm.sum_duplicates()
        csc_perm.sort_indices()
        self._coo_to_csc = csc_perm.data.copy()

    def assemble(self, coo_values: NDArray, R_local: NDArray) -> None:
        """Assemble sparse matrix and RHS vector.

        Parameters
        ----------
        coo_values : NDArray, shape (nnz,)
            COO Jacobian values.
        R_local : NDArray, shape (local_size,)
            Local residual vector (will be negated for Newton RHS).
        """
        self._mat.data[:] = coo_values[self._coo_to_csc]
        self._rhs[:] = 0.0
        np.add.at(self._rhs, self._rhs_rows, -R_local)

    def solve(self) -> NDArray:
        """Solve the assembled linear system.

        Returns
        -------
        NDArray, shape (local_size,)
            Solution vector in block ordering (all jx, all jy, all rho).
        """
        if self._solver_type == "iterative":
            sol, info = gmres(self._mat, self._rhs, rtol=1e-8, atol=1e-12,
                              maxiter=1000)
            self._converged = (info == 0)
            self._iterations = info if info > 0 else 0
            if not self._converged:
                print(f"WARNING: GMRES did not converge (stagnated at iteration {info})")
        else:
            lu = splu(self._mat, permc_spec='COLAMD', diag_pivot_thresh=0.1)
            sol = lu.solve(self._rhs)
            self._converged = True
            self._iterations = 0

        return sol[self._rhs_rows]

    def get_convergence_info(self) -> dict:
        return {
            'converged': self._converged,
            'iterations': self._iterations,
            'residual_norm': getattr(self, '_residual_norm', 0.0),
            'relative_residual_norm': getattr(self, '_relative_residual_norm', 0.0),
            'reason': 1 if self._converged else -1,
        }
