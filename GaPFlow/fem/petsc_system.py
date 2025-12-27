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

from petsc4py import PETSc

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..solver_fem_2d import FEMSolver2D

NDArray = npt.NDArray[np.floating]


class PETScSystem:
    """
    Translates local FEM system to distributed PETSc and solves.

    This class manages:
    1. Local-to-global index translation
    2. PETSc matrix/vector creation and assembly
    3. KSP solver configuration and execution

    The translation is built by comparing index_mask_padded_local (local indices)
    with index_mask_padded_global (global indices) at each grid position.

    Parameters
    ----------
    solver : FEMSolver2D
        The FEM solver instance containing index masks and problem info.
    """

    def __init__(self, solver: "FEMSolver2D"):
        self.solver = solver
        self.decomp = solver.decomp
        self.comm = PETSc.COMM_WORLD

        # Build translation array
        self.l2g = self._build_l2g()

        # Global grid size
        Nx, Ny = self.decomp.nb_domain_grid_pts
        self.nb_global = Nx * Ny

        # Create PETSc objects
        self._create_petsc_objects()

    def _build_l2g(self) -> NDArray:
        """
        Build local-to-global translation array.

        Compares index_mask_padded_local with index_mask_padded_global
        to create a mapping from local indices to global indices.

        Returns
        -------
        NDArray
            Array of shape (nb_contributors,) mapping local index to global index.
        """
        mask_local = self.solver.index_mask_padded_local('')
        mask_global = self.decomp.index_mask_padded_global

        l2g = np.zeros(self.solver.nb_contributors, dtype=PETSc.IntType)
        valid = mask_local >= 0
        l2g[mask_local[valid]] = mask_global[valid]
        return l2g

    def _create_petsc_objects(self):
        """
        Create distributed PETSc Mat, Vec, and KSP objects.

        Matrix size: (nb_res * nb_global) x (nb_var * nb_global)
        Each rank owns rows corresponding to its inner points.
        """
        s = self.solver
        nb_res = len(s.residuals)
        nb_var = len(s.variables)

        # Local and global sizes (square system: rows = cols)
        local_size = nb_res * s.nb_inner_pts
        global_size = nb_res * self.nb_global

        # Create sparse matrix (AIJ = compressed sparse row)
        self.mat = PETSc.Mat().create(self.comm)
        # Square matrix: row and column distributions match
        self.mat.setSizes([(local_size, global_size), (local_size, global_size)])
        self.mat.setType('aij')
        self.mat.setFromOptions()
        # Preallocation: estimate ~40 non-zeros per row for FEM with 3 variables
        # (each node connects to ~12 neighbors, times 3 variables = ~36)
        self.mat.setPreallocationNNZ(40)
        # Allow new entries beyond preallocation (performance warning, but prevents errors)
        self.mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.mat.setUp()

        # Create vectors
        self.vec_rhs = self.mat.createVecLeft()
        self.vec_sol = self.mat.createVecRight()

        # Create KSP solver with MUMPS direct solver
        # MUMPS supports pivoting, required for matrices with zero diagonal entries
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOperators(self.mat)
        self.ksp.setType('preonly')
        pc = self.ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')

        # MUMPS handles scaling internally via ICNTL(8), default=77 (automatic)
        # Can be tuned via -mat_mumps_icntl_8 command line option if needed
        self.ksp.setFromOptions()

    def assemble(self, M_local: NDArray, R_local: NDArray):
        """
        Assemble local system into distributed PETSc objects.

        Uses point-interleaved ordering: for point p, residual r, the global row is
        p * nb_res + r. This ensures each rank owns all equations for its inner points.

        Parameters
        ----------
        M_local : NDArray
            Local tangent matrix, shape (res_size, var_size).
        R_local : NDArray
            Local residual vector, shape (res_size,).
        """
        s = self.solver
        nb_inner = s.nb_inner_pts
        nb_res = len(s.residuals)
        nb_var = len(s.variables)

        # Zero out previous values
        self.mat.zeroEntries()
        self.vec_rhs.zeroEntries()

        # Assemble matrix with point-interleaved ordering
        # Global row for (point p, residual r) = p * nb_res + r
        # Global col for (point p, variable v) = p * nb_var + v
        for local_pt in range(nb_inner):
            global_pt = int(self.l2g[local_pt])

            for res_idx, res_name in enumerate(s.residuals):
                global_row = global_pt * nb_res + res_idx
                row_slice = s._res_slice(res_name)
                local_row_in_block = local_pt
                local_row_idx = row_slice.start + local_row_in_block

                # Collect all columns for this row
                cols = []
                vals = []

                for var_idx, var_name in enumerate(s.variables):
                    col_slice = s._var_slice(var_name)

                    # Get row data from local matrix
                    row_data = M_local[local_row_idx, col_slice]
                    local_cols = np.nonzero(row_data)[0]

                    if len(local_cols) > 0:
                        # Convert local column indices to global with interleaved ordering
                        global_cols = self.l2g[local_cols] * nb_var + var_idx
                        cols.extend(global_cols.astype(np.int32))
                        vals.extend(row_data[local_cols])

                if len(cols) > 0:
                    self.mat.setValues(
                        [global_row], cols, vals,
                        addv=PETSc.InsertMode.ADD_VALUES
                    )

        # Assemble vector with point-interleaved ordering (negate R for Newton)
        for local_pt in range(nb_inner):
            global_pt = int(self.l2g[local_pt])

            for res_idx, res_name in enumerate(s.residuals):
                global_row = global_pt * nb_res + res_idx
                row_slice = s._res_slice(res_name)
                local_row_in_block = local_pt
                local_row_idx = row_slice.start + local_row_in_block

                self.vec_rhs.setValue(
                    global_row, -R_local[local_row_idx],
                    addv=PETSc.InsertMode.INSERT_VALUES
                )

        # Finalize assembly
        self.mat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL)
        self.mat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL)
        self.vec_rhs.assemblyBegin()
        self.vec_rhs.assemblyEnd()

    def solve(self) -> NDArray:
        """
        Solve the linear system and return local solution.

        Solves: mat @ vec_sol = vec_rhs

        Returns
        -------
        NDArray
            Solution vector for this process's inner points,
            shape (res_size,).
        """
        # Solve the system
        self.ksp.solve(self.vec_rhs, self.vec_sol)

        # Extract local solution from distributed vector
        # With point-interleaved ordering, vec_sol layout is:
        # [pt0_var0, pt0_var1, pt0_var2, pt1_var0, pt1_var1, pt1_var2, ...]
        # Each rank owns entries for its inner points (contiguous in this layout)
        s = self.solver
        nb_inner = s.nb_inner_pts
        nb_var = len(s.variables)

        # Get local portion of solution vector
        sol_array = self.vec_sol.getArray()

        # Reorder from interleaved PETSc layout to block layout
        # PETSc: [pt0_var0, pt0_var1, ..., pt1_var0, pt1_var1, ...]
        # Local: [var0_pt0, var0_pt1, ..., var1_pt0, var1_pt1, ...]
        x_local = np.zeros(s.res_size)

        for var_idx, var_name in enumerate(s.variables):
            sol_slice = s._sol_slice(var_name)
            # Extract values for this variable from interleaved array
            # In interleaved: value for (pt, var) is at index pt * nb_var + var
            var_values = sol_array[var_idx::nb_var]  # stride by nb_var, start at var_idx
            x_local[sol_slice] = var_values

        return x_local

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
