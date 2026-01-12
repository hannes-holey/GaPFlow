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
"""SciPy-based sparse linear solver for serial execution without PETSc."""

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, gmres

if TYPE_CHECKING:
    from .assembly_layout import PETScAssemblyInfo

NDArray = npt.NDArray[np.floating]


class ScipySystem:
    """Serial sparse linear solver using SciPy (fallback when PETSc unavailable).

    Provides the same interface as PETScSystem for drop-in replacement.

    Parameters
    ----------
    info : PETScAssemblyInfo
        Precomputed assembly info (sizes and global indices).
    solver_type : str, optional
        "direct" (SuperLU) or "iterative" (GMRES). Default: "direct".
    """

    def __init__(self, info: "PETScAssemblyInfo", solver_type: str = "direct"):
        self._info = info
        self._solver_type = solver_type
        self._size = info.local_size  # = global_size for serial
        self._rows = info.mat_global_rows
        self._cols = info.mat_global_cols
        self._rhs_rows = info.rhs_global_rows

        # Stored after assemble()
        self._mat: csr_matrix | None = None
        self._rhs: NDArray | None = None

        # Convergence info (for iterative solver)
        self._iterations = 0
        self._converged = True

    def assemble(self, coo_values: NDArray, R_local: NDArray):
        """Assemble sparse matrix and RHS vector.

        Parameters
        ----------
        coo_values : NDArray
            COO values array, shape (nnz,).
        R_local : NDArray
            Local residual vector, shape (local_size,).
        """
        # Build CSR matrix from COO data
        self._mat = csr_matrix(
            (coo_values, (self._rows, self._cols)),
            shape=(self._size, self._size)
        )

        # Build RHS vector (negated for Newton: solve M @ dx = -R)
        self._rhs = np.zeros(self._size)
        np.add.at(self._rhs, self._rhs_rows, -R_local)

    def solve(self, nb_inner_pts: int, nb_vars: int) -> NDArray:
        """Solve linear system, return solution in block layout.

        Parameters
        ----------
        nb_inner_pts : int
            Number of inner grid points.
        nb_vars : int
            Number of variables per point.

        Returns
        -------
        NDArray
            Solution vector reshaped to block layout.
        """
        if self._mat is None or self._rhs is None:
            raise RuntimeError("Must call assemble() before solve()")

        if self._solver_type == "iterative":
            sol, info = gmres(self._mat, self._rhs, rtol=1e-8, atol=1e-12, maxiter=1000)
            self._converged = (info == 0)
            self._iterations = info if info > 0 else 0
        else:
            # Direct solver (SuperLU)
            sol = spsolve(self._mat, self._rhs)
            self._converged = True
            self._iterations = 0

        return sol.reshape(nb_inner_pts, nb_vars).T.ravel()

    def get_convergence_info(self) -> dict:
        """Get information about the last solve.

        Returns
        -------
        dict
            Dictionary with convergence info:
            - converged: bool
            - iterations: int (0 for direct solver)
            - residual_norm: float (0.0, not computed for scipy)
            - reason: int (1 if converged, -1 if not)
        """
        return {
            'converged': self._converged,
            'iterations': self._iterations,
            'residual_norm': 0.0,  # Not computed for scipy
            'reason': 1 if self._converged else -1,
        }
