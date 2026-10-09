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

import os
import numpy as np
import numpy.typing as npt
from scipy.ndimage import zoom

try:
    from mpi4py import MPI
    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False

    # Environment variables set by common MPI launchers (mpirun/mpiexec/srun).
    # If one of these indicates more than one rank, running without mpi4py
    # would silently produce N independent, duplicated serial runs instead of
    # erroring, so we fail loudly instead.
    _MPI_LAUNCHER_SIZE_VARS = (
        "OMPI_COMM_WORLD_SIZE",  # Open MPI
        "PMI_SIZE",              # MPICH / Intel MPI
        "SLURM_NTASKS",         # Slurm
    )
    for _var in _MPI_LAUNCHER_SIZE_VARS:
        if int(os.environ.get(_var, "1")) > 1:
            raise ImportError(
                "Detected an MPI launcher (mpirun/mpiexec/srun) requesting more "
                "than one rank, but mpi4py is not installed. Install GaPFlow "
                "with the 'parallel' extra, e.g. `pip install GaPFlow[parallel]`, "
                "to run MPI-parallel simulations."
            )
    del _var

    class _SerialRequest:
        """Stand-in for mpi4py.MPI.Request: our sends/receives complete
        synchronously, so waiting on them is always a no-op."""

        def Wait(self):
            pass

    class _SerialRequestClass:
        @staticmethod
        def Waitall(requests):
            pass

    class _SerialComm:
        """Minimal stand-in for an mpi4py communicator, used when mpi4py is
        not installed. Only valid for single-process (serial) execution."""

        rank = 0
        size = 1

        def __init__(self):
            self._pending = {}

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def barrier(self):
            pass

        Barrier = barrier

        def allreduce(self, value, op=None):
            return value

        def gather(self, value, root=0):
            return [value]

        def allgather(self, value):
            return [value]

        def bcast(self, value, root=0):
            return value

        def Isend(self, data, dest=0, tag=0):
            self._pending[tag] = np.array(data, copy=True)
            return _SerialRequest()

        def Irecv(self, buf, source=0, tag=0):
            buf[...] = self._pending.pop(tag)
            return _SerialRequest()

    class _SerialMPI:
        """Stand-in for the mpi4py.MPI module. Only implements what GaPFlow
        needs to run single-process without MPI installed."""

        SUM = 'SUM'
        MAX = 'MAX'
        MIN = 'MIN'
        Comm = _SerialComm
        Request = _SerialRequestClass
        COMM_WORLD = _SerialComm()
        COMM_SELF = _SerialComm()

    MPI = _SerialMPI()

from dataclasses import dataclass
from functools import cached_property

from typing import TYPE_CHECKING, Optional, Tuple
if TYPE_CHECKING:
    from .problem import Problem

from muGrid import (
    CartesianDecomposition,
    GlobalFieldCollection,
    Communicator
)
try:
    from muGrid import FFTEngine
    HAS_FFT_ENGINE = True
except ImportError:
    FFTEngine = None
    HAS_FFT_ENGINE = False

NDArray = npt.NDArray[np.floating]

@dataclass
class BCContext:
    """Context object passed to boundary condition callbacks.

    Attributes
    ----------
    problem : Problem
        The GaPFlow Problem instance.
    required_shape : tuple
        Required shape of the returned array.
    ghost_slice : tuple
        Slice for accessing ghost cells.
    interior_slice : tuple
        Slice for accessing adjacent interior cells.
    x_norm : NDArray
        Normalized x-coordinates at the boundary (0 to 1).
    y_norm : NDArray
        Normalized y-coordinates at the boundary (0 to 1).
    """
    problem: 'Problem'
    required_shape: tuple
    ghost_slice: tuple
    interior_slice: tuple
    x_norm: NDArray
    y_norm: NDArray


class DomainDecomposition:
    """
    Manages domain decomposition for MPI-parallel simulations.

    Parameters
    ----------
    grid : dict
        Grid configuration containing numer of grid points and
        boundary conditions.
    numerics : dict, optional
        Numerics configuration. When using FEM solver, the
        P2 mass flux grid (ghost depth 2) is also initialised.
    """

    _BND_TO_KEY = {'W': 'xW', 'E': 'xE', 'S': 'yS', 'N': 'yN'}

    def __init__(self, grid: dict, numerics: dict = None):

        self.grid = grid
        self._mpi_comm = MPI.COMM_WORLD
        self._comm = Communicator(self._mpi_comm)

        self._Nx = grid['Nx']
        self._Ny = grid['Ny']

        self._nb_subdivisions = self.split_domain()

        # Density grid (P1)
        self.init_decomposition()

        # Mass flux grid (P2) for FEM solver
        if numerics is not None and numerics['solver'] == 'fem':
            self.init_decomposition_P2()

    def split_domain(self) -> Tuple[int, int]:
        """Determine the number of splits in x and y
        """
        nx_splits = int(np.floor(np.sqrt(self._comm.size)))
        ny_splits = self._comm.size // nx_splits
        return (nx_splits, ny_splits)

    def init_decomposition(self) -> None:
        """Standard Decomposition for the main grid (ghost depth 1)
        """
        self._nb_domain_grid_pts = (self._Nx, self._Ny)
        self._decomp = CartesianDecomposition(
            self._comm,
            list(self._nb_domain_grid_pts),
            list(self._nb_subdivisions),
            [1, 1],
            [1, 1],
        )

    def init_decomposition_P2(self) -> None:
        """Decomposition for the P2 mass flux grid used in Taylor-Hood FEM solver with ghost depth 2.
        """

        nx_splits, ny_splits = self._nb_subdivisions

        if self._Nx % nx_splits != 0 or self._Ny % ny_splits != 0:
            raise ValueError(
                f"Grid dimensions ({self._Nx}, {self._Ny}) must be divisible by "
                f"the number of MPI splits ({nx_splits}, {ny_splits}). "
                f"Choose Nx divisible by {nx_splits} and Ny divisible by {ny_splits}."
            )

        Nx_P2 = self._Nx * 2 if self.periodic_x else self._Nx * 2 - 1
        Ny_P2 = self._Ny * 2 if self.periodic_y else self._Ny * 2 - 1
        self._nb_domain_grid_pts_P2 = (Nx_P2, Ny_P2)

        self._decomp_P2 = CartesianDecomposition(
            self._comm,
            list(self._nb_domain_grid_pts_P2),
            list(self._nb_subdivisions),
            [2, 2],
            [2, 2],
        )

    @property
    def fc(self) -> GlobalFieldCollection:
        """GlobalFieldCollection on the standard decomposition."""
        return self._decomp.collection

    @property
    def fc_P2(self) -> GlobalFieldCollection:
        """GlobalFieldCollection on the P2 decomposition."""
        return self._decomp_P2.collection

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
    def nb_domain_grid_pts_P2(self) -> tuple:
        """Global mass flux grid size (2*Nx-1, 2*Ny-1)."""
        return self._nb_domain_grid_pts_P2

    @property
    def nb_subdomain_grid_pts(self) -> tuple:
        """Local subdomain size (inner points)."""
        return tuple(self._decomp.nb_subdomain_grid_pts)

    @property
    def nb_subdomain_grid_pts_P2(self) -> tuple:
        """Local mass flux subdomain size (inner points)."""
        return tuple(self._decomp_P2.nb_subdomain_grid_pts)

    @property
    def subdomain_locations(self) -> tuple:
        """Start position of this subdomain (without ghost offset)."""
        return tuple(self._decomp.subdomain_locations)

    @property
    def subdomain_locations_P2(self) -> tuple:
        """Start position of mass flux subdomain (without ghost offset)."""
        return tuple(self._decomp_P2.subdomain_locations)

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

    @property
    def icoordsg_P2(self):
        """Global coordinate indices for mass flux subdomain (2, Nx_v_local+4, Ny_v_local+4)."""
        return self._decomp_P2.icoordsg

    # ---------------------------
    # Local shape utilities
    # ---------------------------

    @property
    def local_shape_padded(self) -> tuple:
        """Local subdomain shape with ghosts: (Nx_local+2, Ny_local+2)."""
        inner = self.nb_subdomain_grid_pts
        return (inner[0] + 2, inner[1] + 2)

    @property
    def local_shape_padded_P2(self) -> tuple:
        """Local mass flux subdomain shape with ghosts: (Nx_v_local+4, Ny_v_local+4)."""
        inner = self.nb_subdomain_grid_pts_P2
        return (inner[0] + 4, inner[1] + 4)

    # ---------------------------
    # Coordinate arrays
    # ---------------------------

    @cached_property
    def xx(self) -> NDArray:
        """Cell-center x-coordinates in physical units [m] (2D, with ghosts)."""
        grid = self.grid
        dx, Lx = grid['dx'], grid['Lx']
        xx = self.icoordsg[0] * dx + dx / 2.0
        if self.bc_at_W:
            xx[0, :] = -dx / 2.0
        if self.bc_at_E:
            xx[-1, :] = Lx + dx / 2.0
        return xx

    @cached_property
    def yy(self) -> NDArray:
        """Cell-center y-coordinates in physical units [m] (2D, with ghosts)."""
        grid = self.grid
        dy, Ly = grid['dy'], grid['Ly']
        yy = self.icoordsg[1] * dy + dy / 2.0
        if self.bc_at_S:
            yy[:, 0] = -dy / 2.0
        if self.bc_at_N:
            yy[:, -1] = Ly + dy / 2.0
        return yy

    @cached_property
    def xx_norm(self) -> NDArray:
        """Normalized x-coordinates [0, 1] (2D, with ghosts)."""
        return self.xx / self.grid['Lx']

    @cached_property
    def yy_norm(self) -> NDArray:
        """Normalized y-coordinates [0, 1] (2D, with ghosts)."""
        return self.yy / self.grid['Ly']

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

    @property
    def bc_at_W(self) -> bool:
        """True if this rank has a physical (non-periodic) West boundary."""
        return self.is_at_xW and not self.periodic_x

    @property
    def bc_at_E(self) -> bool:
        """True if this rank has a physical (non-periodic) East boundary."""
        return self.is_at_xE and not self.periodic_x

    @property
    def bc_at_S(self) -> bool:
        """True if this rank has a physical (non-periodic) South boundary."""
        return self.is_at_yS and not self.periodic_y

    @property
    def bc_at_N(self) -> bool:
        """True if this rank has a physical (non-periodic) North boundary."""
        return self.is_at_yN and not self.periodic_y

    @cached_property
    def index_mask_padded_global(self) -> NDArray:
        """Get global index mask for local padded subdomain shape."""
        return self.icoordsg[0] + self.icoordsg[1] * self._Nx

    @cached_property
    def index_mask_padded_global_P2(self) -> NDArray:
        """Global index mask for local mass flux padded subdomain shape."""
        Nx_P2 = self._nb_domain_grid_pts_P2[0]
        return self.icoordsg_P2[0] + self.icoordsg_P2[1] * Nx_P2

    # ---------------------------
    # Global field gathering
    # ---------------------------

    def gather_global(self, local_field: NDArray, P2: bool = False) -> NDArray:
        """Gather local field to global array on rank 0.

        Parameters
        ----------
        local_field : NDArray
            Local 2D field. If shape matches local_shape_padded, ghost cells
            are excluded. If shape matches nb_subdomain_grid_pts, used directly.
        P2 : bool, optional
            If True, use the P2 mass flux decomposition (e.g. for jx/jy) instead
            of the standard P1 decomposition, by default False.

        Returns
        -------
        NDArray or None
            Global field with shape nb_domain_grid_pts on rank 0, None otherwise.
        """
        if P2:
            nb_domain_grid_pts = self.nb_domain_grid_pts_P2
            subdomain_locations = self.subdomain_locations_P2
            nb_subdomain_grid_pts = self.nb_subdomain_grid_pts_P2
            local_shape_padded = self.local_shape_padded_P2
        else:
            nb_domain_grid_pts = self.nb_domain_grid_pts
            subdomain_locations = self.subdomain_locations
            nb_subdomain_grid_pts = self.nb_subdomain_grid_pts
            local_shape_padded = self.local_shape_padded

        comm = self._mpi_comm

        # extract inner part if field includes ghosts
        if local_field.shape == local_shape_padded:
            n_ghost = (local_shape_padded[0] - nb_subdomain_grid_pts[0]) // 2
            local_inner = local_field[n_ghost:-n_ghost, n_ghost:-n_ghost]
        elif local_field.shape == nb_subdomain_grid_pts:
            local_inner = local_field
        else:
            raise ValueError(f"Field shape {local_field.shape} doesn't match "
                             f"inner {nb_subdomain_grid_pts} or padded {local_shape_padded}")

        # gather local fields and their positions
        all_fields = comm.gather(local_inner, root=0)
        all_locs = comm.gather(subdomain_locations, root=0)
        all_sizes = comm.gather(nb_subdomain_grid_pts, root=0)

        if self.rank == 0:
            global_field = np.zeros(nb_domain_grid_pts, dtype=local_field.dtype)
            for field, loc, sz in zip(all_fields, all_locs, all_sizes):
                global_field[loc[0]:loc[0] + sz[0], loc[1]:loc[1] + sz[1]] = field
            return global_field
        return None

    def scatter_global(self, global_field: NDArray, P2: bool = False) -> NDArray:
        """Scatter global field from rank 0 to local arrays.

        Parameters
        ----------
        global_field : NDArray
            Global field with shape nb_domain_grid_pts. Only needs to be
            valid on rank 0; other ranks can pass None or empty array.
        P2 : bool, optional
            If True, use the P2 mass flux decomposition (e.g. for jx/jy) instead
            of the standard P1 decomposition, by default False.

        Returns
        -------
        NDArray
            Local field with shape nb_subdomain_grid_pts (without ghosts).
        """
        if P2:
            subdomain_locations = self.subdomain_locations_P2
            nb_subdomain_grid_pts = self.nb_subdomain_grid_pts_P2
        else:
            subdomain_locations = self.subdomain_locations
            nb_subdomain_grid_pts = self.nb_subdomain_grid_pts

        comm = self._mpi_comm

        # Gather all locations and sizes
        all_locs = comm.allgather(subdomain_locations)
        all_sizes = comm.allgather(nb_subdomain_grid_pts)

        # Prepare receive buffer
        local_inner = np.empty(nb_subdomain_grid_pts, dtype=np.float64)

        if self.rank == 0:
            # Send each rank's portion
            for dest, (loc, sz) in enumerate(zip(all_locs, all_sizes)):
                chunk = np.ascontiguousarray(
                    global_field[loc[0]:loc[0] + sz[0], loc[1]:loc[1] + sz[1]])
                if dest == 0:
                    local_inner[:] = chunk
                else:
                    comm.Send(chunk, dest=dest, tag=0)
        else:
            comm.Recv(local_inner, source=0, tag=0)

        return local_inner

    def gather_global_padded(self, local_field: NDArray, P2: bool = False):
        """Gather local field to global array on rank 0, including ghost cells."""
        if P2:
            nb_domain_grid_pts = self.nb_domain_grid_pts_P2
            subdomain_locations = self.subdomain_locations_P2
            n_ghost = 2
        else:
            nb_domain_grid_pts = self.nb_domain_grid_pts
            subdomain_locations = self.subdomain_locations
            n_ghost = 1

        comm = self._mpi_comm

        all_fields = comm.gather(local_field, root=0)
        all_locs = comm.gather(subdomain_locations, root=0)

        if self.rank == 0:
            global_shape = (nb_domain_grid_pts[0] + 2 * n_ghost,
                            nb_domain_grid_pts[1] + 2 * n_ghost)
            global_padded = np.zeros(global_shape, dtype=local_field.dtype)
            for field, loc in zip(all_fields, all_locs):
                px, py = field.shape
                global_padded[loc[0]:loc[0] + px, loc[1]:loc[1] + py] = field
            return global_padded
        return None

    def scatter_global_padded(self, global_padded: Optional[NDArray], P2: bool = False) -> NDArray:
        """Scatter global padded field from rank 0 to local arrays."""
        if P2:
            subdomain_locations = self.subdomain_locations_P2
            local_shape_padded = self.local_shape_padded_P2
        else:
            subdomain_locations = self.subdomain_locations
            local_shape_padded = self.local_shape_padded

        comm = self._mpi_comm
        all_locs = comm.allgather(subdomain_locations)
        all_shapes = comm.allgather(local_shape_padded)

        local = np.empty(local_shape_padded, dtype=np.float64)

        if self.rank == 0:
            for dest, (loc, shape) in enumerate(zip(all_locs, all_shapes)):
                chunk = np.ascontiguousarray(
                    global_padded[loc[0]:loc[0] + shape[0], loc[1]:loc[1] + shape[1]])
                if dest == 0:
                    local[:] = chunk
                else:
                    comm.Send(chunk, dest=dest, tag=0)
        else:
            comm.Recv(local, source=0, tag=0)

        return local

    # ---------------------------
    # Ghost cell handling
    # ---------------------------

    def exchange_ghosts(self, field) -> None:
        """MPI ghost exchange for a single P1 field."""
        self._decomp.communicate_ghosts(field)

    def exchange_ghosts_P2(self, field) -> None:
        """MPI ghost exchange for a P2 field (ghost depth 2)."""
        self._decomp_P2.communicate_ghosts(field)

    def owns_boundary(self, bnd: str) -> bool:
        """Check if this rank owns the specified boundary."""
        return {'W': self.is_at_xW, 'E': self.is_at_xE,
                'S': self.is_at_yS, 'N': self.is_at_yN}[bnd]


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
        if self.periodic_x and self.periodic_y:
            self.Nx_fft = self.Nx
            self.Ny_fft = self.Ny
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
        """Precompute send/recv maps for MPI redistribution.

        GaPFlow uses 2D block decomposition; FFTEngine uses Y-stripe decomposition
        (full X dimension on every rank). The exchange plan accounts for both X and Y
        extents of each GaPFlow rank so that block-decomposed data is correctly
        gathered into Y-stripes and scattered back.
        """
        comm = self.decomp._mpi_comm

        # GaPFlow X/Y range for this rank
        src_x_start = self.decomp.subdomain_locations[0]
        src_x_size  = self.decomp.nb_subdomain_grid_pts[0]
        src_x_end   = src_x_start + src_x_size

        src_y_start = self.decomp.subdomain_locations[1]
        src_y_size  = self.decomp.nb_subdomain_grid_pts[1]
        src_y_end   = src_y_start + src_y_size

        # FFT engine Y-range for this rank (X is always full: [0, Nx_fft))
        dst_y_start = self.fft_engine.subdomain_locations[1]
        dst_y_size  = self.fft_engine.nb_subdomain_grid_pts[1]
        dst_y_end   = dst_y_start + dst_y_size

        # Gather all ranks' ranges
        src_ranges = comm.allgather((src_x_start, src_x_end, src_y_start, src_y_end))
        dst_ranges = comm.allgather((dst_y_start, dst_y_end))

        self.send_map = {}
        self.recv_map = {}

        for other_rank in range(comm.size):
            # What I send: my Y-range intersected with their FFT Y-range
            other_dst_y_start, other_dst_y_end = dst_ranges[other_rank]
            send_y_start = max(src_y_start, other_dst_y_start)
            send_y_end   = min(src_y_end,   other_dst_y_end)

            if send_y_start < send_y_end:
                self.send_map[other_rank] = {
                    'y_slice': slice(send_y_start - src_y_start, send_y_end - src_y_start),
                    'size': (src_x_size, send_y_end - send_y_start),
                }

            # What I receive: their GaPFlow Y-range intersected with my FFT Y-range
            other_src_x_start, other_src_x_end, other_src_y_start, other_src_y_end = src_ranges[other_rank]
            recv_y_start = max(dst_y_start, other_src_y_start)
            recv_y_end   = min(dst_y_end,   other_src_y_end)

            if recv_y_start < recv_y_end:
                self.recv_map[other_rank] = {
                    'x_start': other_src_x_start,
                    'x_size':  other_src_x_end - other_src_x_start,
                    'y_slice': slice(recv_y_start - dst_y_start, recv_y_end - dst_y_start),
                    'size': (other_src_x_end - other_src_x_start, recv_y_end - recv_y_start),
                }

    def embed(self, src: np.ndarray, dst: np.ndarray):
        """Transfer data from GaPFlow block domain to FFT Y-stripe domain.

        Parameters
        ----------
        src : np.ndarray
            GaPFlow local block, shape (Nx_local, Ny_local).
        dst : np.ndarray
            FFT engine local buffer, shape (Nx_fft, Ny_fft_local). Will be zeroed
            first; rows beyond Nx remain zero (padding for non-periodic BCs).
        """
        comm = self.decomp._mpi_comm

        dst[:] = 0.0

        # send_bufs keeps each Isend buffer alive until Waitall (required by MPI).
        send_reqs = []
        send_bufs = []
        for dest_rank, info in self.send_map.items():
            data = np.ascontiguousarray(src[:, info['y_slice']])
            req = comm.Isend(data, dest=dest_rank, tag=100)
            send_reqs.append(req)
            send_bufs.append(data)

        recv_reqs = []
        recv_buffers = []
        for src_rank, info in self.recv_map.items():
            buf = np.empty(info['size'], dtype=src.dtype)
            req = comm.Irecv(buf, source=src_rank, tag=100)
            recv_reqs.append(req)
            recv_buffers.append((buf, info))

        MPI.Request.Waitall(recv_reqs)
        for buf, info in recv_buffers:
            x0, xs = info['x_start'], info['x_size']
            dst[x0:x0 + xs, info['y_slice']] = buf

        MPI.Request.Waitall(send_reqs)

    def extract(self, src: np.ndarray, dst: np.ndarray):
        """Transfer data from FFT Y-stripe domain back to GaPFlow block domain.

        Parameters
        ----------
        src : np.ndarray
            FFT engine local buffer, shape (Nx_fft, Ny_fft_local).
        dst : np.ndarray
            GaPFlow local block, shape (Nx_local, Ny_local).
        """
        comm = self.decomp._mpi_comm

        # Reverse of embed: recv_map entries describe what to send back,
        # send_map entries describe what to receive back.
        send_reqs = []
        send_bufs = []
        for dest_rank, info in self.recv_map.items():
            x0, xs = info['x_start'], info['x_size']
            data = np.ascontiguousarray(src[x0:x0 + xs, info['y_slice']])
            req = comm.Isend(data, dest=dest_rank, tag=200)
            send_reqs.append(req)
            send_bufs.append(data)

        recv_reqs = []
        recv_buffers = []
        for src_rank, info in self.send_map.items():
            buf = np.empty(info['size'], dtype=src.dtype)
            req = comm.Irecv(buf, source=src_rank, tag=200)
            recv_reqs.append(req)
            recv_buffers.append((buf, info['y_slice']))

        MPI.Request.Waitall(recv_reqs)
        for buf, y_slice in recv_buffers:
            dst[:, y_slice] = buf

        MPI.Request.Waitall(send_reqs)
