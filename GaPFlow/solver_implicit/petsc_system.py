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

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING

from .. import HAS_PETSC

if not HAS_PETSC:
    raise ImportError(
        "petsc4py is required for the Taylor-Hood FEM solver but is not installed.\n"
        "See README.md for installation instructions."
    )

from petsc4py import PETSc

if TYPE_CHECKING:
    from .assembly import GlobalIndexPattern

NDArray = npt.NDArray[np.floating]


class PETScSystem:
    """PETSc-based sparse linear solver for the Taylor-Hood Q2Q1 FEM solver.

    Provides the same .assemble() / .solve() interface as ScipySystem.
    solve() returns a flat 1-D solution vector; the caller unpacks per-variable
    slices of unequal length.

    Parameters
    ----------
    info : GlobalIndexPattern
        Precomputed assembly info (sizes and global indices).
    solver_type : str, optional
        "direct" (MUMPS LU) or "iterative" (GMRES). Default: "direct".
    """

    _ILU2_THRESHOLD = 110000

    def __init__(self, info: "GlobalIndexPattern", solver_type: str = "direct",
                 print_mumps_diagnostics: bool = False):
        self._info = info
        self._solver_type = solver_type
        self.print_mumps_diagnostics = print_mumps_diagnostics
        self.comm = PETSc.COMM_WORLD
        self._create_petsc_objects()

    def _create_petsc_objects(self):
        info = self._info
        local_size = info.local_size
        global_size = PETSc.DECIDE

        self.mat = PETSc.Mat().create(self.comm)
        self.mat.setSizes([(local_size, global_size), (local_size, global_size)])
        self.mat.setType('aij')
        self.mat.setFromOptions()

        # Pass copies: PETSc's setPreallocationCOO sorts the arrays in-place,
        # which would corrupt the assembly COO index arrays.
        self.mat.setPreallocationCOO(
            info.mat_global_rows.copy(), info.mat_global_cols.copy()
        )
        self.mat.setUp()

        self.vec_rhs = self.mat.createVecLeft()
        self.vec_sol = self.mat.createVecRight()

        self._sol_local = PETSc.Vec().createSeq(local_size, comm=PETSc.COMM_SELF)
        sol_is = PETSc.IS().createGeneral(info.rhs_global_rows, comm=self.comm)
        self._sol_scatter = PETSc.Scatter().create(self.vec_sol, sol_is, self._sol_local, None)

        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.mat)

        if self._solver_type == "iterative":
            self.ksp.setType('bcgs')
            self.ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=1000)
            self.ksp.setComputeSingularValues(True)
            pc = self.ksp.getPC()
            if self.comm.getSize() > 1:
                pc.setType('bjacobi')
            else:
                fill_level = 2  # if local_size > self._ILU2_THRESHOLD else 1
                pc.setType('ilu')
                pc.setFactorLevels(fill_level)
        else:
            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')

        self.ksp.setFromOptions()

    def assemble(self, coo_values: NDArray, R_local: NDArray) -> None:
        """Assemble sparse matrix and RHS vector.

        Parameters
        ----------
        coo_values : NDArray, shape (nnz,)
            COO Jacobian values.
        R_local : NDArray, shape (local_size,)
            Local residual vector (will be negated for Newton RHS).
        """
        self.mat.setValuesCOO(coo_values, PETSc.InsertMode.INSERT_VALUES)

        self.vec_rhs.zeroEntries()
        self.vec_rhs.setValues(
            self._info.rhs_global_rows, -R_local, PETSc.InsertMode.INSERT_VALUES)

        self.mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
        self.vec_rhs.assemblyBegin()
        self.vec_rhs.assemblyEnd()

    def solve(self) -> NDArray:
        """Solve the assembled linear system.

        Returns
        -------
        NDArray, shape (local_size,)
            Flat solution vector.  The caller unpacks per-variable slices.
        """
        self.ksp.solve(self.vec_rhs, self.vec_sol)
        reason = self.ksp.getConvergedReason()
        if self._solver_type == "iterative":
            smax, smin = self.ksp.computeExtremeSingularValues()
            cond = smax / smin if smin > 0 else float('inf')
            if reason < 0:
                print(f"WARNING: KSP did not converge (reason={reason}, "
                      f"iterations={self.ksp.getIterationNumber()}, cond~{cond:.2e})")
            elif cond > 1e8:
                print(f"WARNING: high condition number cond~{cond:.2e} — solution may be inaccurate")
        elif reason < 0:
            print(f"WARNING: KSP did not converge (reason={reason}, iterations={self.ksp.getIterationNumber()})")
            if self._solver_type == "direct":
                try:
                    F = self.ksp.getPC().getFactorMatrix()
                    info = F.getMumpsInfo(1)
                    info2 = F.getMumpsInfo(2)
                    print(f"  MUMPS INFOG(1)={info}, INFOG(2)={info2}")
                except Exception as e:
                    print(f"  (could not get MUMPS info: {e})")

        if self._solver_type == "direct" and self.print_mumps_diagnostics:
            try:
                F = self.ksp.getPC().getFactorMatrix()
                error_code = F.getMumpsInfo(1)
                n_negative_pivots = F.getMumpsInfo(12)
                n_tiny_pivots = F.getMumpsInfo(25)
                print(f"  [LinSolve] MUMPS: error_code={error_code} "
                      f"negative_pivots={n_negative_pivots} "
                      f"tiny_pivots_perturbed={n_tiny_pivots}")
                if error_code < 0:
                    print(f"  WARNING: MUMPS reported error INFOG(1)={error_code}")
            except Exception as e:
                print(f"  (could not get MUMPS diagnostics: {e})")

        self._sol_scatter.scatter(self.vec_sol, self._sol_local)
        return self._sol_local.getArray().copy()

    def get_convergence_info(self) -> dict:
        reason = self.ksp.getConvergedReason()
        return {
            'converged': reason > 0,
            'iterations': self.ksp.getIterationNumber(),
            'residual_norm': self.ksp.getResidualNorm(),
            'reason': reason,
        }
