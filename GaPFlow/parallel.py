#
# Copyright 2025 Hannes Holey
#           2025 Christoph Huber
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

from mpi4py import MPI
import numpy as np
import numpy.typing as npt

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .problem import Problem

from muGrid import (
    CartesianDecomposition,
    GlobalFieldCollection,
    Communicator,
)


class DomainDecomposition:
    """
    Manages domain decomposition for MPI-parallel simulations.

    Parameters
    ----------
    grid : dict
        Grid configuration containing:
        - Nx, Ny: number of grid points
        - bc_xE, bc_xW, bc_yS, bc_yN: boundary conditions per component
        - nb_ghost: ghost cell width (default: 1)
    comm : MPI.Comm, optional
        MPI communicator (default: MPI.COMM_WORLD)
    """

    def __init__(self, grid: dict):

        self.grid = grid
        self._mpi_comm = MPI.COMM_WORLD
        self._comm = Communicator(self._mpi_comm)

        # Grid dimensions
        self._Nx = grid['Nx']
        self._Ny = grid['Ny']
        self._nb_domain_grid_pts = (self._Nx, self._Ny)

        # Ghost cells
        nb_ghost = 1
        self._nb_ghost_left = (nb_ghost, nb_ghost)
        self._nb_ghost_right = (nb_ghost, nb_ghost)

        # Subdivision
        self._nb_subdivisions = (1, self._comm.size)

        # Create CartesianDecomposition
        self._decomp = CartesianDecomposition(
            self._comm,
            list(self._nb_domain_grid_pts),
            list(self._nb_subdivisions),
            list(self._nb_ghost_left),
            list(self._nb_ghost_right),
        )

    def get_fc(self) -> GlobalFieldCollection:
        """
        Create a GlobalFieldCollection on the domain decomposition.

        Returns
        -------
        muGrid.GlobalFieldCollection
            The global field collection associated with the domain decomposition.
        """
        return self._decomp.collection

    # ---------------------------
    # MPI properties
    # ---------------------------

    @property
    def rank(self) -> int:
        """MPI rank of this process."""
        return self._comm.rank

    @property
    def size(self) -> int:
        """Total number of MPI processes."""
        return self._comm.size

    @property
    def nb_domain_grid_pts(self) -> tuple:
        """Global domain size (Nx, Ny)."""
        return self._nb_domain_grid_pts

    @property
    def nb_subdomain_grid_pts(self) -> tuple:
        """Local subdomain size including ghosts."""
        return tuple(self._decomp.nb_subdomain_grid_pts)

    @property
    def subdomain_locations(self) -> tuple:
        """Start position of this subdomain (including ghost offset)."""
        return tuple(self._decomp.subdomain_locations)

    # ---------------------------
    # Boundary ownership detection
    # ---------------------------

    @property
    def is_at_xW(self) -> bool:
        """True if this rank owns the West (left, x=0) boundary."""
        return self.subdomain_locations[0] == 0

    @property
    def is_at_xE(self) -> bool:
        """True if this rank owns the East (right, x=Lx) boundary."""
        loc_x = self.subdomain_locations[0]
        local_interior_x = self.nb_subdomain_grid_pts[0] - 2  # subtract ghosts
        return (loc_x + local_interior_x) >= self._Nx

    @property
    def is_at_yS(self) -> bool:
        """True if this rank owns the South (bottom, y=0) boundary."""
        return self.subdomain_locations[1] == 0

    @property
    def is_at_yN(self) -> bool:
        """True if this rank owns the North (top, y=Ly) boundary."""
        loc_y = self.subdomain_locations[1]
        local_interior_y = self.nb_subdomain_grid_pts[1] - 2  # subtract ghosts
        return (loc_y + local_interior_y) >= self._Ny

    # ---------------------------
    # Ghost cell handling
    # ---------------------------

    def communicate_ghost_buffers(self, problem: "Problem") -> None:
        """
        Communicate ghost buffers between MPI ranks and apply boundary conditions.

        This method:
        1. Uses muGrid's ghost exchange for inter-rank communication (MPI parallel only)
        2. Applies physical boundary conditions at domain boundaries

        Parameters
        ----------
        problem : Problem
            The problem instance containing fields and boundary condition info.
        """
        # Step 1: MPI ghost exchange (applies periodic wrap even in serial)
        self._decomp.communicate_ghosts(problem.fc.get_real_field('solution'))
        if problem.bEnergy:
            self._decomp.communicate_ghosts(problem.fc.get_real_field('total_energy'))

        # Step 2: Apply physical BCs at domain boundaries
        self._apply_solution_bcs(problem)
        if problem.bEnergy:
            self._apply_energy_bcs(problem)

    def _apply_solution_bcs(self, problem: "Problem") -> None:
        """
        Apply boundary conditions to solution field ghost cells.

        Only applies BCs if this rank owns the corresponding boundary.
        """
        grid = self.grid
        field = problem.q  # This returns the .pg array

        # West boundary (x=0, ghost at index 0)
        if self.is_at_xW:
            if all(grid["bc_xW_P"]):
                field[:, 0, :] = field[:, -2, :].copy()
            else:
                if any(grid["bc_xW_D"]):
                    field[grid["bc_xW_D"], :1, :] = self._get_ghost_cell_values(
                        field, grid, "D", axis=0, direction=-1)
                if any(grid["bc_xW_N"]):
                    field[grid["bc_xW_N"], :1, :] = self._get_ghost_cell_values(
                        field, grid, "N", axis=0, direction=-1)

        # East boundary (x=Lx, ghost at index -1)
        if self.is_at_xE:
            if all(grid["bc_xE_P"]):
                field[:, -1, :] = field[:, 1, :].copy()
            else:
                if any(grid["bc_xE_D"]):
                    field[grid["bc_xE_D"], -1:, :] = self._get_ghost_cell_values(
                        field, grid, "D", axis=0, direction=1)
                if any(grid["bc_xE_N"]):
                    field[grid["bc_xE_N"], -1:, :] = self._get_ghost_cell_values(
                        field, grid, "N", axis=0, direction=1)

        # South boundary (y=0, ghost at index 0)
        if self.is_at_yS:
            if all(grid["bc_yS_P"]):
                field[:, :, 0] = field[:, :, -2].copy()
            else:
                if any(grid["bc_yS_D"]):
                    field[grid["bc_yS_D"], :, :1] = self._get_ghost_cell_values(
                        field, grid, "D", axis=1, direction=-1)
                if any(grid["bc_yS_N"]):
                    field[grid["bc_yS_N"], :, :1] = self._get_ghost_cell_values(
                        field, grid, "N", axis=1, direction=-1)

        # North boundary (y=Ly, ghost at index -1)
        if self.is_at_yN:
            if all(grid["bc_yN_P"]):
                field[:, :, -1] = field[:, :, 1].copy()
            else:
                if any(grid["bc_yN_D"]):
                    field[grid["bc_yN_D"], :, -1:] = self._get_ghost_cell_values(
                        field, grid, "D", axis=1, direction=1)
                if any(grid["bc_yN_N"]):
                    field[grid["bc_yN_N"], :, -1:] = self._get_ghost_cell_values(
                        field, grid, "N", axis=1, direction=1)

    def _apply_energy_bcs(self, problem: "Problem") -> None:
        """
        Apply boundary conditions to energy field ghost cells.

        Only applies BCs if this rank owns the corresponding boundary.
        """
        energy = problem.energy
        rho = problem.q[0]
        jx = problem.q[1]
        jy = problem.q[2]

        # West boundary (ghost cell at index 0)
        if self.is_at_xW:
            if energy.bc_xW == 'P':
                energy.energy[0, :] = energy.energy[-2, :].copy()
            elif energy.bc_xW == 'D':
                ux = jx[0, :] / rho[0, :]
                uy = jy[0, :] / rho[0, :]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[0, :] = rho[0, :] * (energy.cv * energy.T_bc_xW + kinetic)
            elif energy.bc_xW == 'N':
                energy.energy[0, :] = energy.energy[1, :].copy()

        # East boundary (ghost cell at index -1)
        if self.is_at_xE:
            if energy.bc_xE == 'P':
                energy.energy[-1, :] = energy.energy[1, :].copy()
            elif energy.bc_xE == 'D':
                ux = jx[-1, :] / rho[-1, :]
                uy = jy[-1, :] / rho[-1, :]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[-1, :] = rho[-1, :] * (energy.cv * energy.T_bc_xE + kinetic)
            elif energy.bc_xE == 'N':
                energy.energy[-1, :] = energy.energy[-2, :].copy()

    @staticmethod
    def _get_ghost_cell_values(field: npt.NDArray[np.floating],
                               grid: dict,
                               bc_type: str,
                               axis: int,
                               direction: int,
                               num_ghost: int = 1) -> npt.NDArray[np.floating]:
        """
        Computes ghost cell values for Dirichlet ('D') or Neumann ('N') boundary
        conditions.

        Parameters
        ----------
        field : ndarray
            The field array (with ghost cells).
        grid : dict
            Grid configuration with boundary condition info.
        bc_type : str
            'D' for Dirichlet or 'N' for Neumann.
        axis : int
            0 for x-axis, 1 for y-axis.
        direction : int
            Upstream (<0) or downstream (>0) direction.
        num_ghost : int
            Number of ghost cells (<= 2 supported).

        Returns
        -------
        ndarray
            Ghost cell values extracted/computed for the selected mask.
        """
        assert bc_type in ["D", "N"]

        if axis == 0:  # x-axis
            if direction > 0:  # East boundary (downstream)
                mask = grid[f"bc_xE_{bc_type}"]
                q_target = grid["bc_xE_D_val"]
                q_adj = field[mask, -(num_ghost + num_ghost): -num_ghost, :]
            else:  # West boundary (upstream)
                mask = grid[f"bc_xW_{bc_type}"]
                q_target = grid["bc_xW_D_val"]
                q_adj = field[mask, num_ghost: num_ghost + num_ghost, :]

        elif axis == 1:  # y-axis
            if direction > 0:  # North boundary (downstream)
                mask = grid[f"bc_yN_{bc_type}"]
                q_target = grid["bc_yN_D_val"]
                q_adj = field[mask, :, -(num_ghost + num_ghost): -num_ghost]
            else:  # South boundary (upstream)
                mask = grid[f"bc_yS_{bc_type}"]
                q_target = grid["bc_yS_D_val"]
                q_adj = field[mask, :, num_ghost: num_ghost + num_ghost]
        else:
            raise RuntimeError("axis must be either 0 (x) or 1 (y)")

        a1 = 0.5
        a2 = 0.0
        q1 = q_adj
        q2 = 0.0

        if bc_type == "D":
            Q = (q_target - a1 * q1 + a2 * q2) / (a1 - a2)
        else:
            Q = ((1.0 - a1) * q1 + a2 * q2) / (a1 - a2)

        return Q
