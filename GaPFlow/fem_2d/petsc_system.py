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
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

from .. import HAS_PETSC

if not HAS_PETSC:
    raise ImportError(
        "petsc4py is required for the 2D FEM solver but is not installed.\n"
        "See README.md for installation instructions."
    )

from petsc4py import PETSc

if TYPE_CHECKING:
    from .assembly_layout import PETScAssemblyInfo

NDArray = npt.NDArray[np.floating]


class PETScSystem:
    """Manages distributed PETSc linear system and solves.

    1. PETSc matrix/vector creation with COO preallocation
    2. O(nnz) vectorized assembly from COO values
    3. KSP solver configuration and execution

    Parameters
    ----------
    info : PETScAssemblyInfo
        Precomputed assembly info (sizes and global indices).
    solver_type : str, optional
        "direct" (MUMPS LU) or "iterative" (BiCGSTAB + ILU). Default: "direct".
    """

    # Threshold for switching from ILU(1) to ILU(2)
    _ILU2_THRESHOLD = 110000

    def __init__(self, info: "PETScAssemblyInfo", solver_type: str = "direct"):
        self._info = info
        self._solver_type = solver_type
        self.comm = PETSc.COMM_WORLD
        self._create_petsc_objects()

    def _create_petsc_objects(self):
        """Create distributed PETSc Mat, Vec, and KSP objects."""
        info = self._info
        local_size = info.local_size
        global_size = info.global_size

        # Create sparse matrix
        self.mat = PETSc.Mat().create(self.comm)
        self.mat.setSizes([(local_size, global_size), (local_size, global_size)])
        self.mat.setType('aij')
        self.mat.setFromOptions()

        # COO preallocation
        self.mat.setPreallocationCOO(info.mat_global_rows, info.mat_global_cols)
        self.mat.setUp()

        # Create vectors
        self.vec_rhs = self.mat.createVecLeft()
        self.vec_sol = self.mat.createVecRight()

        # Create KSP solver
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.mat)

        if self._solver_type == "iterative":
            self.ksp.setType('bcgs')
            self.ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
            pc = self.ksp.getPC()
            pc.setType('ilu')
            fill_level = 2 if global_size > self._ILU2_THRESHOLD else 1
            pc.setFactorLevels(fill_level)
        else:
            # MUMPS direct solver (default)
            if self.comm.getRank() == 0:
                print("Using MUMPS direct solver for PETSc KSP.")
            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')

        self.ksp.setFromOptions()

    def assemble(self, coo_values: NDArray, R_local: NDArray):
        """Assemble local system into distributed PETSc objects.

        Parameters
        ----------
        coo_values : NDArray
            COO values array, shape (nnz,).
        R_local : NDArray
            Local residual vector, shape (local_size,).
        """
        # Assemble matrix
        self.mat.setValuesCOO(coo_values, PETSc.InsertMode.INSERT_VALUES)

        # Assemble RHS
        self.vec_rhs.zeroEntries()
        rhs_vals = -R_local  # Negate for Newton
        self.vec_rhs.setValues(
            self._info.rhs_global_rows, rhs_vals, PETSc.InsertMode.INSERT_VALUES)

        # Finalize assembly
        self.mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
        self.vec_rhs.assemblyBegin()
        self.vec_rhs.assemblyEnd()

    def solve(self, nb_inner_pts: int, nb_vars: int) -> NDArray:
        """Solve linear system, return solution in block layout."""
        self.ksp.solve(self.vec_rhs, self.vec_sol)
        return self.vec_sol.getArray().reshape(nb_inner_pts, nb_vars).T.ravel()

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
