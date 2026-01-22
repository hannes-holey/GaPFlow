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

# flake8: noqa: W503

from mpi4py import MPI
import numpy as np
import numpy.typing as npt

from functools import cached_property

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .problem import Problem

from muGrid import (
    CartesianDecomposition,
    GlobalFieldCollection,
    Communicator,
)
try:
    from muGrid import FFTEngine
    HAS_FFT_ENGINE = True
except ImportError:
    FFTEngine = None
    HAS_FFT_ENGINE = False

NDArray = npt.NDArray[np.floating]


class DomainDecomposition:
    """
    Manages domain decomposition for MPI-parallel simulations.

    Parameters
    ----------
    grid : dict
        Grid configuration containing:
        - Nx, Ny: number of grid points
        - bc_xE, bc_xW, bc_yS, bc_yN: boundary conditions per component
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
    def subdomain_info(self) -> str:
        """Subdomain info string for MPI runs, empty for single rank."""
        if self.size > 1:
            sub_x, sub_y = self.nb_subdomain_grid_pts
            return f" (subdomain: {sub_x}x{sub_y} on {self.size} ranks)"
        return ""

    @property
    def nb_ghost_pts(self) -> int:
        """Total number of ghost points around the subdomain without corners."""
        return 2 * self.nb_subdomain_grid_pts[0] + 2 * self.nb_subdomain_grid_pts[1]

    @property
    def icoordsg(self):
        """Global coordinate indices for each subdomain point (2, Nx_local+2, Ny_local+2)."""
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

    def local_coordinates_midpoint(self, dx, dy):
        """Compute cell-center coordinates (xx, yy) for local subdomain.

        Note: muGrid's icoordsg uses wrapped/periodic indices for ghost cells,
        which gives incorrect coordinates for non-periodic topography profiles.
        This method corrects ghost cell coordinates to proper extrapolated values.
        """
        icoords = self.icoordsg
        xx = icoords[0] * dx + dx / 2.0
        yy = icoords[1] * dy + dy / 2.0

        # Fix wrapped ghost cell coordinates
        Lx = self._Nx * dx
        Ly = self._Ny * dy

        if self.is_at_xW and not self.periodic_x:
            xx[0, :] = -dx / 2.0
        if self.is_at_xE and not self.periodic_x:
            xx[-1, :] = Lx + dx / 2.0
        if self.is_at_yS and not self.periodic_y:
            yy[:, 0] = -dy / 2.0
        if self.is_at_yN and not self.periodic_y:
            yy[:, -1] = Ly + dy / 2.0

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
        return all(b == 'P' for b in grid["bc_xW"]) and all(b == 'P' for b in grid["bc_xE"])

    @property
    def periodic_y(self) -> bool:
        """True if the y-boundaries are periodic."""
        grid = self.grid
        return all(b == 'P' for b in grid["bc_yS"]) and all(b == 'P' for b in grid["bc_yN"])

    @property
    def has_full_x(self) -> bool:
        """True if the full x-boundary is owned by this rank."""
        return self.is_at_xW and self.is_at_xE

    @property
    def has_full_y(self) -> bool:
        """True if the full y-boundary is owned by this rank."""
        return self.is_at_yS and self.is_at_yN

    @cached_property
    def index_mask_padded_global(self) -> NDArray:
        """GET global index mask for local padded subdomain shape."""
        mask = np.zeros(self.local_shape_padded, dtype=int)
        for x in range(self.local_shape_padded[0]):
            for y in range(self.local_shape_padded[1]):
                x_global = self.icoordsg[0][x, y]
                y_global = self.icoordsg[1][x, y]
                mask[x, y] = x_global + y_global * self._Nx
        return mask

    # ---------------------------
    # Global field gathering
    # ---------------------------

    def gather_global(self, local_field: NDArray) -> NDArray:
        """Gather local field to global array on rank 0.

        Parameters
        ----------
        local_field : NDArray
            Local 2D field. If shape matches local_shape_padded, ghost cells
            are excluded. If shape matches local_shape_inner, used directly.

        Returns
        -------
        NDArray or None
            Global field with shape nb_domain_grid_pts on rank 0, None otherwise.
        """
        comm = self._mpi_comm

        # extract inner part if field includes ghosts
        if local_field.shape == self.local_shape_padded:
            local_inner = local_field[1:-1, 1:-1]
        elif local_field.shape == self.local_shape_inner:
            local_inner = local_field
        else:
            raise ValueError(f"Field shape {local_field.shape} doesn't match "
                             f"inner {self.local_shape_inner} or padded {self.local_shape_padded}")

        # gather local fields and their positions
        all_fields = comm.gather(local_inner, root=0)
        all_locs = comm.gather(self.subdomain_locations, root=0)
        all_sizes = comm.gather(self.nb_subdomain_grid_pts, root=0)

        if self.rank == 0:
            global_field = np.zeros(self.nb_domain_grid_pts, dtype=local_field.dtype)
            for field, loc, sz in zip(all_fields, all_locs, all_sizes):
                global_field[loc[0]:loc[0] + sz[0], loc[1]:loc[1] + sz[1]] = field
            return global_field
        return None

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

    def _owns_boundary(self, bnd: str) -> bool:
        """Check if this rank owns the specified boundary."""
        return {'W': self.is_at_xW, 'E': self.is_at_xE,
                'S': self.is_at_yS, 'N': self.is_at_yN}[bnd]

    def _get_bc_slices(self, bnd: str):
        """Return (ghost_slice, interior_slice) for the specified boundary."""
        slices = {
            'W': ((slice(0, 1), slice(None)), (slice(1, 2), slice(None))),
            'E': ((slice(-1, None), slice(None)), (slice(-2, -1), slice(None))),
            'S': ((slice(None), slice(0, 1)), (slice(None), slice(1, 2))),
            'N': ((slice(None), slice(-1, None)), (slice(None), slice(-2, -1))),
        }
        return slices[bnd]

    def _apply_solution_bcs(self, problem: "Problem") -> None:
        """
        Apply boundary conditions to solution field ghost cells.

        Only applies BCs if this rank owns the corresponding boundary.
        Iterates through boundaries and variables to apply Dirichlet/Neumann BCs.
        """
        grid = self.grid
        field = problem.q  # shape: (num_vars, Nx_padded, Ny_padded)

        # Check if FEM 2D solver is active (nodal discretization)
        is_nodal = (problem.numerics.get('solver') == 'fem' and grid['dim'] == 2)

        # Map boundary name to grid key prefix
        bnd_to_key = {'W': 'xW', 'E': 'xE', 'S': 'yS', 'N': 'yN'}

        for bnd in ['W', 'E', 'S', 'N']:
            if not self._owns_boundary(bnd):
                continue

            key = bnd_to_key[bnd]
            bc_types = grid[f'bc_{key}']  # e.g., ['P', 'D', 'N']

            # Skip if all periodic (handled by ghost communication)
            if all(b == 'P' for b in bc_types):
                continue

            ghost, interior = self._get_bc_slices(bnd)

            for var_idx, bc_type in enumerate(bc_types):
                if bc_type == 'P':
                    continue

                elif bc_type == 'D':
                    bc_vals = grid.get(f'bc_{key}_D_val')
                    target = bc_vals[var_idx] if isinstance(bc_vals, list) else bc_vals

                    if is_nodal:
                        field[(var_idx,) + ghost] = target
                    else:
                        # Cell-centered: ghost = 2*target - interior
                        field[(var_idx,) + ghost] = (
                            2.0 * target - field[(var_idx,) + interior]
                        )

                elif bc_type == 'N':
                    # Neumann: zero normal gradient (ghost = interior)
                    field[(var_idx,) + ghost] = field[(var_idx,) + interior]

    def _apply_energy_bcs(self, problem: "Problem") -> None:
        """
        Apply boundary conditions to energy field ghost cells.

        Only applies BCs if this rank owns the corresponding boundary.
        If the grid uses periodic BCs (all components), the energy field
        uses the periodic ghost values from communicate_ghosts() instead.
        """
        grid = self.grid
        energy = problem.energy
        rho = problem.q[0]
        jx = problem.q[1]
        jy = problem.q[2]

        # Map boundary name to grid key prefix
        bnd_to_key = {'W': 'xW', 'E': 'xE', 'S': 'yS', 'N': 'yN'}

        for bnd in ['W', 'E', 'S', 'N']:
            if not self._owns_boundary(bnd):
                continue

            key = bnd_to_key[bnd]
            bc_types = grid[f'bc_{key}']

            # Skip if all periodic
            if all(b == 'P' for b in bc_types):
                continue

            ghost, interior = self._get_bc_slices(bnd)
            bc_type = getattr(energy, f'bc_{key}')
            T_bc = getattr(energy, f'T_bc_{key}')

            if bc_type == 'D':
                ux = jx[ghost] / rho[ghost]
                uy = jy[ghost] / rho[ghost]
                kinetic = 0.5 * (ux**2 + uy**2)
                energy.energy[ghost] = rho[ghost] * (energy.cv * T_bc + kinetic)
            elif bc_type == 'N':
                energy.energy[ghost] = energy.energy[interior].copy()


class FFTDomainTranslation:
    """
    Manages data transfer between GaPFlow's DomainDecomposition and
    the FFT domain used for elastic deformation computation.

    Handles:
    - FFTEngine instantiation with correct grid size for boundary conditions
    - MPI redistribution between different domain decompositions
    - Zero-padding for non-periodic boundaries

    Parameters
    ----------
    decomp : DomainDecomposition
        GaPFlow's domain decomposition instance.
        Provides: grid, periodic_x, periodic_y, nb_domain_grid_pts,
                  subdomain_locations, nb_subdomain_grid_pts, communicator
    """

    def __init__(self, decomp: DomainDecomposition):
        if not HAS_FFT_ENGINE:
            raise ImportError(
                "FFTDomainTranslation requires muGrid built with FFT support. "
                "Install FFTW (libfftw3-dev) and rebuild muGrid."
            )

        self.decomp = decomp
        self.periodic_x = decomp.periodic_x
        self.periodic_y = decomp.periodic_y
        self.Nx, self.Ny = decomp.nb_domain_grid_pts

        # Compute FFT grid size based on periodicity
        self._compute_fft_grid_size()

        # Create FFTEngine with computed size
        self.fft_engine = FFTEngine(
            [self.Nx_fft, self.Ny_fft],
            decomp._comm  # muGrid Communicator
        )

        # Build exchange plan for MPI redistribution
        self._build_exchange_plan()

    def _compute_fft_grid_size(self):
        """Compute FFT grid size based on boundary conditions.

        Periodic: N
        Semi-periodic (one direction free): 2*N - 1 in free direction
        Non-periodic (both directions free): 2*N in both directions
        """
        self._needs_redistribution = True

        if self.periodic_x and self.periodic_y:
            # Fully periodic
            self.Nx_fft = self.Nx
            self.Ny_fft = self.Ny
            self._needs_redistribution = False
        elif self.periodic_x and not self.periodic_y:
            # Semi-periodic: x periodic, y free
            self.Nx_fft = self.Nx
            self.Ny_fft = 2 * self.Ny - 1
        elif self.periodic_y and not self.periodic_x:
            # Semi-periodic: x free, y periodic
            self.Nx_fft = 2 * self.Nx - 1
            self.Ny_fft = self.Ny
        else:
            # Fully non-periodic
            self.Nx_fft = 2 * self.Nx
            self.Ny_fft = 2 * self.Ny

    def _build_exchange_plan(self):
        """Precompute send/recv maps for MPI redistribution."""
        comm = self.decomp._mpi_comm

        # GaPFlow Y-range for this rank
        src_y_start = self.decomp.subdomain_locations[1]
        src_y_size = self.decomp.nb_subdomain_grid_pts[1]
        src_y_end = src_y_start + src_y_size

        # FFT engine Y-range for this rank
        dst_y_start = self.fft_engine.subdomain_locations[1]
        dst_y_size = self.fft_engine.nb_subdomain_grid_pts[1]
        dst_y_end = dst_y_start + dst_y_size

        # Gather all ranks' Y-ranges
        src_ranges = comm.allgather((src_y_start, src_y_end))
        dst_ranges = comm.allgather((dst_y_start, dst_y_end))

        # Compute intersections
        self.send_map = {}
        self.recv_map = {}

        for other_rank in range(comm.size):
            # What I send: intersection of my src with their dst
            other_dst_start, other_dst_end = dst_ranges[other_rank]
            send_start = max(src_y_start, other_dst_start)
            send_end = min(src_y_end, other_dst_end)

            if send_start < send_end:
                self.send_map[other_rank] = {
                    'local_slice': slice(send_start - src_y_start,
                                         send_end - src_y_start),
                    'size': (self.Nx, send_end - send_start)
                }

            # What I receive: intersection of their src with my dst
            other_src_start, other_src_end = src_ranges[other_rank]
            recv_start = max(dst_y_start, other_src_start)
            recv_end = min(dst_y_end, other_src_end)

            if recv_start < recv_end:
                self.recv_map[other_rank] = {
                    'local_slice': slice(recv_start - dst_y_start,
                                         recv_end - dst_y_start),
                    'size': (self.Nx, recv_end - recv_start)
                }

    def embed(self, src: np.ndarray, dst: np.ndarray):
        """
        Transfer data from GaPFlow domain to FFT domain.

        Parameters
        ----------
        src : np.ndarray
            Source array from GaPFlow (local subdomain, shape Nx × Ny_local)
        dst : np.ndarray
            Destination array on FFT engine (will be zeroed first for padding)
        """
        if not self._needs_redistribution:
            # Periodic: direct copy
            dst[:] = src
            return

        comm = self.decomp._mpi_comm

        # Zero destination (handles padding)
        dst[:] = 0.0

        # Non-blocking sends
        send_reqs = []
        for dest_rank, info in self.send_map.items():
            data = np.ascontiguousarray(src[:, info['local_slice']])
            req = comm.Isend(data, dest=dest_rank, tag=100)
            send_reqs.append(req)

        # Non-blocking receives
        recv_reqs = []
        recv_buffers = []
        for src_rank, info in self.recv_map.items():
            buf = np.empty(info['size'], dtype=src.dtype)
            req = comm.Irecv(buf, source=src_rank, tag=100)
            recv_reqs.append(req)
            recv_buffers.append((buf, info['local_slice']))

        # Wait and copy (only to first Nx rows, rest is zero padding)
        MPI.Request.Waitall(recv_reqs)
        for buf, local_slice in recv_buffers:
            dst[:self.Nx, local_slice] = buf

        MPI.Request.Waitall(send_reqs)

    def extract(self, src: np.ndarray, dst: np.ndarray):
        """
        Transfer data from FFT domain back to GaPFlow domain.

        Parameters
        ----------
        src : np.ndarray
            Source array on FFT engine
        dst : np.ndarray
            Destination array on GaPFlow (local subdomain)
        """
        if not self._needs_redistribution:
            # Periodic: direct copy
            dst[:] = src
            return

        comm = self.decomp._mpi_comm

        # Reverse of embed: send_map and recv_map roles swap
        # We send from FFT positions (our recv_map locations) to GaPFlow positions

        # Non-blocking sends (using recv_map since we're going in reverse)
        # Only read from first Nx rows (discard padding)
        send_reqs = []
        for dest_rank, info in self.recv_map.items():
            data = np.ascontiguousarray(src[:self.Nx, info['local_slice']])
            req = comm.Isend(data, dest=dest_rank, tag=200)
            send_reqs.append(req)

        # Non-blocking receives (using send_map since we're going in reverse)
        recv_reqs = []
        recv_buffers = []
        for src_rank, info in self.send_map.items():
            buf = np.empty(info['size'], dtype=src.dtype)
            req = comm.Irecv(buf, source=src_rank, tag=200)
            recv_reqs.append(req)
            recv_buffers.append((buf, info['local_slice']))

        # Wait and copy
        MPI.Request.Waitall(recv_reqs)
        for buf, local_slice in recv_buffers:
            dst[:, local_slice] = buf

        MPI.Request.Waitall(send_reqs)
