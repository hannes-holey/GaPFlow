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
from dataclasses import dataclass

from petsc4py import PETSc


NDArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class PETScLayout:
    """
    Explicit link between Solver/DomainDecomposition and PETSc:
    - l2g_pts maps local point indices to global point indices
    - mat_rows/cols define global matrix structure (COO format)
    - All arrays computed once at solver initialization.

    Attributes
    ----------
    nb_vars : int
        Number of variables (3 or 4 with energy).
    nb_inner_pts : int
        Number of inner points owned by this rank.
    nb_contributors : int
        Number of unique contributor indices (inner + ghost).
    nb_global_pts : int
        Total global grid points.
    l2g_pts : NDArray
        Local-to-global point mapping, shape (nb_contributors,).
    mat_rows : NDArray
        Global row indices for each non-zero, shape (nnz,).
    mat_cols : NDArray
        Global column indices for each non-zero, shape (nnz,).
    mat_local_idx : NDArray
        Local indices [row, col] in M_local for each non-zero, shape (nnz, 2).
        This attribute is used by sparse_layout to invert the mapping from
        coo_idx → (local_row, local_col) to (local_row, local_col) → coo_idx.
    rhs_rows : NDArray
        Global row indices for RHS assembly, shape (local_size,).
    """
    nb_vars: int
    nb_inner_pts: int
    nb_contributors: int
    nb_global_pts: int
    l2g_pts: NDArray
    mat_rows: NDArray
    mat_cols: NDArray
    mat_local_idx: NDArray
    rhs_rows: NDArray


class PETScSystem:
    """
    Manages distributed PETSc linear system and solves.
    1. PETSc matrix/vector creation with COO preallocation
    2. O(nnz) vectorized assembly from COO values
    3. KSP solver configuration and execution

    All index translation is handled by PETScLayout.

    Parameters
    ----------
    layout : PETScLayout
        Precomputed index arrays for assembly and solution extraction.
    solver_type : str, optional
        Solver type: "direct" (MUMPS LU) or "iterative" (BiCGSTAB + ILU).
        Default is "direct".

    Notes
    -----
    Iterative solver performance (vs MUMPS direct):
    - Grid ≤160×160: BiCGSTAB + ILU(1) → 4.7-5.3× faster
    - Grid ≥192×192: BiCGSTAB + ILU(2) → 1.9-4.5× faster
    ILU fill level is selected automatically based on matrix size.
    """

    # Threshold for switching from ILU(1) to ILU(2)
    # ~110k corresponds to 192×192 grid with 3 variables
    _ILU2_THRESHOLD = 110000

    def __init__(self, layout: PETScLayout, solver_type: str = "direct"):
        self._layout = layout
        self._solver_type = solver_type
        self.comm = PETSc.COMM_WORLD
        self._create_petsc_objects()

    def _create_petsc_objects(self):
        """
        Create distributed PETSc Mat, Vec, and KSP objects.

        Matrix size: (nb_vars * nb_global_pts) x (nb_vars * nb_global_pts)
        Each rank owns rows corresponding to its inner points.
        """
        L = self._layout
        local_size = L.nb_vars * L.nb_inner_pts
        global_size = L.nb_vars * L.nb_global_pts

        # create sparse matrix (aij = compressed sparse row)
        self.mat = PETSc.Mat().create(self.comm)
        self.mat.setSizes([(local_size, global_size), (local_size, global_size)])
        self.mat.setType('aij')
        self.mat.setFromOptions()

        # COO preallocation for O(nnz) assembly
        self.mat.setPreallocationCOO(self._layout.mat_rows, self._layout.mat_cols)
        self.mat.setUp()

        # create vectors
        self.vec_rhs = self.mat.createVecLeft()
        self.vec_sol = self.mat.createVecRight()

        # create KSP solver
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.mat)

        if self._solver_type == "iterative":
            # BiCGSTAB + ILU: faster than MUMPS for most grid sizes
            # ILU(1) for smaller grids, ILU(2) for larger (≥192×192)
            self.ksp.setType('bcgs')
            self.ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
            pc = self.ksp.getPC()
            pc.setType('ilu')
            fill_level = 2 if global_size > self._ILU2_THRESHOLD else 1
            pc.setFactorLevels(fill_level)
        else:
            # MUMPS direct solver (default)
            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')

        self.ksp.setFromOptions()

    def assemble(self, coo_values: NDArray, R_local: NDArray):
        """
        Assemble local system into distributed PETSc objects.

        Uses O(nnz) vectorized COO assembly with point-interleaved ordering:
        for point p, residual r, the global row is p * nb_res + r.

        Parameters
        ----------
        coo_values : NDArray
            COO values array from sparse assembly, shape (nnz,).
        R_local : NDArray
            Local residual vector, shape (res_size,).
        """
        L = self._layout

        # assemble matrix using coo values
        self.mat.setValuesCOO(coo_values, PETSc.InsertMode.INSERT_VALUES)

        # assemble RHS
        self.vec_rhs.zeroEntries()
        rhs_vals = -R_local  # Negate for Newton
        self.vec_rhs.setValues(L.rhs_rows, rhs_vals, PETSc.InsertMode.INSERT_VALUES)

        # finalize assembly
        self.mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
        self.vec_rhs.assemblyBegin()
        self.vec_rhs.assemblyEnd()

    def solve(self) -> NDArray:
        """Solve linear system, return solution in block layout."""
        self.ksp.solve(self.vec_rhs, self.vec_sol)
        L = self._layout
        return self.vec_sol.getArray().reshape(L.nb_inner_pts, L.nb_vars).T.ravel()

    def get_convergence_info(self) -> dict:
        """
        Get information about the last solve.

        Returns
        -------
        dict
            Dictionary with convergence info:
            - converged: bool
            - iterations: int
            - residual_norm: float
            - reason: int (PETSc convergence reason code)
        """
        reason = self.ksp.getConvergedReason()
        return {
            'converged': reason > 0,
            'iterations': self.ksp.getIterationNumber(),
            'residual_norm': self.ksp.getResidualNorm(),
            'reason': reason,
        }
