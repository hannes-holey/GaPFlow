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
        Get the GlobalFieldCollection on the domain decomposition.

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
        """Local subdomain size (inner points, without ghosts)."""
        return tuple(self._decomp.nb_subdomain_grid_pts)

    @property
    def subdomain_locations(self) -> tuple:
        """Start position of this subdomain (including ghost offset)."""
        return tuple(self._decomp.subdomain_locations)

    @property
    def nb_ghost_pts(self) -> int:
        """Total number of ghost points around the subdomain without corners."""
        return 2 * self.nb_subdomain_grid_pts[0] + 2 * self.nb_subdomain_grid_pts[1]

    @property
    def icoordsg(self):
        """Global coordinate indices for each subdomain point."""
        return self._decomp.icoordsg

    # ---------------------------
    # Local shape utilities
    # ---------------------------

    @property
    def local_shape_inner(self) -> tuple:
        """Local subdomain shape without ghosts: (Nx_local, Ny_local)."""
        return self.nb_subdomain_grid_pts

    @property
    def local_shape_padded(self) -> tuple:
        """Local subdomain shape with ghosts: (Nx_local+2, Ny_local+2)."""
        inner = self.nb_subdomain_grid_pts
        return (inner[0] + 2, inner[1] + 2)

    def local_zeros(self, components=None):
        """Create zero array with local padded shape."""
        shape = self.local_shape_padded
        if components is not None:
            shape = (components,) + shape
        return np.zeros(shape)

    def local_ones(self, components=None):
        """Create ones array with local padded shape."""
        shape = self.local_shape_padded
        if components is not None:
            shape = (components,) + shape
        return np.ones(shape)

    def local_coordinates_midpoint(self, dx, dy):
        """Compute cell-center coordinates (xx, yy) for local subdomain."""
        icoords = self.icoordsg
        xx = icoords[0] * dx + dx / 2.0
        yy = icoords[1] * dy + dy / 2.0
        return xx, yy

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
        # nb_subdomain_grid_pts already returns inner points (without ghosts)
        local_interior_x = self.nb_subdomain_grid_pts[0]
        return (loc_x + local_interior_x) >= self._Nx

    @property
    def is_at_yS(self) -> bool:
        """True if this rank owns the South (bottom, y=0) boundary."""
        return self.subdomain_locations[1] == 0

    @property
    def is_at_yN(self) -> bool:
        """True if this rank owns the North (top, y=Ly) boundary."""
        loc_y = self.subdomain_locations[1]
        # nb_subdomain_grid_pts already returns inner points (without ghosts)
        local_interior_y = self.nb_subdomain_grid_pts[1]
        return (loc_y + local_interior_y) >= self._Ny

    @property
    def periodic_x(self) -> bool:
        """True if the x-boundaries are periodic."""
        grid = self.grid
        return all(grid["bc_xW_P"]) and all(grid["bc_xE_P"])

    @property
    def periodic_y(self) -> bool:
        """True if the y-boundaries are periodic."""
        grid = self.grid
        return all(grid["bc_yS_P"]) and all(grid["bc_yN_P"])

    @property
    def has_full_x(self) -> bool:
        """True if the full x-boundary is owned by this rank."""
        return self.is_at_xW and self.is_at_xE
    
    @property
    def has_full_y(self) -> bool:
        """True if the full y-boundary is owned by this rank."""
        return self.is_at_yS and self.is_at_yN

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
        field = problem.q  # returns the .pg array

        # Check if FEM 2D solver is active (nodal discretization)
        is_fem_2d = (problem.numerics.get('solver') == 'fem' and
                     grid.get('dim', 1) == 2)

        # West boundary (x=0, ghost at index 0)
        if self.is_at_xW:
            if not all(grid["bc_xW_P"]):
                if any(grid["bc_xW_D"]):
                    field[grid["bc_xW_D"], :1, :] = self._get_ghost_cell_values(
                        field, grid, "D", axis=0, direction=-1, nodal=is_fem_2d)
                if any(grid["bc_xW_N"]):
                    field[grid["bc_xW_N"], :1, :] = self._get_ghost_cell_values(
                        field, grid, "N", axis=0, direction=-1, nodal=is_fem_2d)

        # East boundary (x=Lx, ghost at index -1)
        if self.is_at_xE:
            if not all(grid["bc_xE_P"]):
                if any(grid["bc_xE_D"]):
                    field[grid["bc_xE_D"], -1:, :] = self._get_ghost_cell_values(
                        field, grid, "D", axis=0, direction=1, nodal=is_fem_2d)
                if any(grid["bc_xE_N"]):
                    field[grid["bc_xE_N"], -1:, :] = self._get_ghost_cell_values(
                        field, grid, "N", axis=0, direction=1, nodal=is_fem_2d)

        # South boundary (y=0, ghost at index 0)
        if self.is_at_yS:
            if not all(grid["bc_yS_P"]):
                if any(grid["bc_yS_D"]):
                    field[grid["bc_yS_D"], :, :1] = self._get_ghost_cell_values(
                        field, grid, "D", axis=1, direction=-1, nodal=is_fem_2d)
                if any(grid["bc_yS_N"]):
                    field[grid["bc_yS_N"], :, :1] = self._get_ghost_cell_values(
                        field, grid, "N", axis=1, direction=-1, nodal=is_fem_2d)

        # North boundary (y=Ly, ghost at index -1)
        if self.is_at_yN:
            if not all(grid["bc_yN_P"]):
                if any(grid["bc_yN_D"]):
                    field[grid["bc_yN_D"], :, -1:] = self._get_ghost_cell_values(
                        field, grid, "D", axis=1, direction=1, nodal=is_fem_2d)
                if any(grid["bc_yN_N"]):
                    field[grid["bc_yN_N"], :, -1:] = self._get_ghost_cell_values(
                        field, grid, "N", axis=1, direction=1, nodal=is_fem_2d)

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
            if energy.bc_xW == 'D':
                ux = jx[0, :] / rho[0, :]
                uy = jy[0, :] / rho[0, :]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[0, :] = rho[0, :] * (energy.cv * energy.T_bc_xW + kinetic)
            elif energy.bc_xW == 'N':
                energy.energy[0, :] = energy.energy[1, :].copy()

        # East boundary (ghost cell at index -1)
        if self.is_at_xE:
            if energy.bc_xE == 'D':
                ux = jx[-1, :] / rho[-1, :]
                uy = jy[-1, :] / rho[-1, :]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[-1, :] = rho[-1, :] * (energy.cv * energy.T_bc_xE + kinetic)
            elif energy.bc_xE == 'N':
                energy.energy[-1, :] = energy.energy[-2, :].copy()
        
        # South boundary (ghost cell at index 0)
        if self.is_at_yS:
            if energy.bc_yS == 'D':
                ux = jx[:, 0] / rho[:, 0]
                uy = jy[:, 0] / rho[:, 0]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[:, 0] = rho[:, 0] * (energy.cv * energy.T_bc_yS + kinetic)
            elif energy.bc_yS == 'N':
                energy.energy[:, 0] = energy.energy[:, 1].copy()
        
        # North boundary (ghost cell at index -1)
        if self.is_at_yN:
            if energy.bc_yN == 'D':
                ux = jx[:, -1] / rho[:, -1]
                uy = jy[:, -1] / rho[:, -1]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[:, -1] = rho[:, -1] * (energy.cv * energy.T_bc_yN + kinetic)
            elif energy.bc_yN == 'N':
                energy.energy[:, -1] = energy.energy[:, -2].copy()

    @staticmethod
    def _get_ghost_cell_values(field: npt.NDArray[np.floating],
                               grid: dict,
                               bc_type: str,
                               axis: int,
                               direction: int,
                               num_ghost: int = 1,
                               nodal: bool = False) -> npt.NDArray[np.floating]:
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
        nodal : bool
            If True, use nodal (FEM 2D) BC treatment where ghost nodes are set
            directly to the boundary value. If False, use cell-centered
            treatment with interpolation.

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

        if nodal:
            # FEM 2D nodal discretization: ghost node values are set directly
            # - Dirichlet: ghost = target value (fixed boundary value)
            # - Neumann: ghost = adjacent inner (zero normal gradient)
            if bc_type == "D":
                Q = np.full_like(q_adj, q_target)
            else:
                Q = q_adj.copy()
        else:
            # Cell-centered discretization: interpolate so cell boundary has target value
            a1 = 0.5
            a2 = 0.0
            q1 = q_adj
            q2 = 0.0

            if bc_type == "D":
                Q = (q_target - a1 * q1 + a2 * q2) / (a1 - a2)
            else:
                Q = ((1.0 - a1) * q1 + a2 * q2) / (a1 - a2)

        return Q
