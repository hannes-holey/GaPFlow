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
import numpy as np
import copy
from muGrid import Field

import numpy.typing as npt
from typing import Tuple, Any

import warnings

from .parallel import DomainDecomposition, FFTDomainTranslation

from ContactMechanics.FFTElasticHalfSpace import (
    PeriodicFFTElasticHalfSpace,
    FreeFFTElasticHalfSpace,
    SemiPeriodicFFTElasticHalfSpace,
)

NDArray = npt.NDArray[np.floating]


def journal_bearing(xx, grid, geo):

    Lx = grid['Lx']
    freq = 2. * np.pi / Lx

    if 'CR' in geo and 'eps' in geo:
        shift = geo['CR'] / freq
        amp = geo['eps'] * shift

    elif 'hmin' in geo and 'hmax' in geo:
        amp = (geo['hmax'] - geo['hmin']) / 2.
        shift = (geo['hmax'] + geo['hmin']) / 2.

    h = shift + amp * np.cos(freq * xx)
    dh_dx = -amp * freq * np.sin(freq * xx)
    dh_dy = np.zeros_like(h)

    return h, dh_dx, dh_dy


def inclined_slider(xx, grid, geo):

    Lx = grid['Lx']
    h0 = geo['hmax']
    h1 = geo['hmin']
    slope = (h1 - h0) / Lx

    h = h0 + slope * xx
    dh_dx = np.ones_like(h) * slope
    dh_dy = np.zeros_like(h)

    return h, dh_dx, dh_dy


def parabolic_slider(xx, grid, geo):

    Lx = grid['Lx']
    h0 = geo['hmin']
    h1 = geo['hmax']
    # slope = (h1 - h0) / Lx

    prefac = 4. / Lx**2 * (h1 - h0)

    h = prefac * (xx - Lx / 2.)**2 + h0
    dh_dx = 2 * prefac * (xx - Lx / 2.)
    dh_dy = np.zeros_like(h)

    return h, dh_dx, dh_dy


def cdc(xx, grid, geo):
    Lx = grid['Lx']
    h0 = geo['hmin']
    h1 = geo['hmax']
    b = geo['b']

    slope = (h1 - h0) / (Lx / 2 - 2 * b)

    conv = np.logical_and(xx >= b, xx < Lx / 2 - b)
    center = np.logical_and(xx >= Lx / 2 - b, xx < Lx / 2 + b)
    div = np.logical_and(xx >= Lx / 2 + b, xx < Lx - b)

    h = np.ones_like(xx) * h1
    h[conv] = h1 - slope * (xx[conv] - b)
    h[center] = h0
    h[div] = h0 + slope * (xx[div] - (Lx / 2 + b))

    dh_dx = np.zeros_like(h)
    dh_dy = np.zeros_like(h)

    dh_dx[conv] = -slope
    dh_dx[div] = slope

    return h, dh_dx, dh_dy


def asperity(xx, yy, grid, geo):
    h0 = geo['hmin']
    h1 = geo['hmax']
    num = geo['num']  # per side

    Lx = grid['Lx']
    Ly = grid['Ly']

    if num == 1:
        hmins = np.array([h0])
    else:
        # Gaussian 99% between hmin and hmax
        std = (h1 - h0) / 2. / 2.57
        hmins = np.random.normal(loc=h0 + (h1 - h0) / 2., scale=std, size=num**2)

    xid = (xx // (Lx / num)).astype(int)
    yid = (yy // (Ly / num)).astype(int)

    masks = []
    for i in range(num):
        for j in range(num):
            masks.append(np.logical_and(xid == i, yid == j))

    bx = np.pi / (Lx / num)
    by = np.pi / (Ly / num)

    h = np.ones_like(xx) * h1
    dh_dx = np.zeros_like(h)
    dh_dy = np.zeros_like(h)

    for m, h0 in zip(masks, hmins):
        cx = np.mean(xx[m])
        cy = np.mean(yy[m])
        h[m] -= (h1 - h0) * (np.cos(bx * (xx[m] - cx)) * np.cos(by * (yy[m] - cy)))
        dh_dx[m] += bx * (h1 - h0) * (np.sin(bx * (xx[m] - cx)) * np.cos(by * (yy[m] - cy)))
        dh_dy[m] += by * (h1 - h0) * (np.cos(bx * (xx[m] - cx)) * np.sin(by * (yy[m] - cy)))

    return h, dh_dx, dh_dy


def parabolic_2d(xx, yy, grid, geo):
    """2D parabolic topography with minimum height at center.

    Creates a paraboloid surface profile where the gap height varies
    quadratically with radial distance from the domain center.

    Parameters
    ----------
    xx : NDArray
        X-coordinate grid.
    yy : NDArray
        Y-coordinate grid.
    grid : dict
        Grid parameters with keys 'Lx', 'Ly'.
    geo : dict
        Geometry parameters with keys 'hmin', 'hmax'.

    Returns
    -------
    h : NDArray
        Height field [m].
    dh_dx : NDArray
        Height gradient in x-direction [1].
    dh_dy : NDArray
        Height gradient in y-direction [1].
    """
    Lx = grid['Lx']
    Ly = grid['Ly']
    hmin = geo['hmin']
    hmax = geo['hmax']

    x_c = Lx / 2
    y_c = Ly / 2
    r_max_sq = (Lx / 2)**2 + (Ly / 2)**2

    r_sq = (xx - x_c)**2 + (yy - y_c)**2

    h = hmin + (hmax - hmin) * r_sq / r_max_sq
    dh_dx = 2 * (hmax - hmin) * (xx - x_c) / r_max_sq
    dh_dy = 2 * (hmax - hmin) * (yy - y_c) / r_max_sq

    return h, dh_dx, dh_dy


class Topography:
    """Topography container.

    Holds the rigid gap topography, and, in case of an elastic substrate, its deformation
    due to the fluid pressure field.
    """

    def __init__(self,
                 fc: Any,
                 grid: dict,
                 geo: dict,
                 prop: dict,
                 decomp: DomainDecomposition = None) -> None:
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            The field collection object.
        grid : dict
            Parameters controlling spatial discretization.
        geo : dict
            Geometry settings.
        prop : dict
            Material properties.
        decomp : DomainDecomposition
            Domain decomposition for MPI-parallel coordinate creation.
        """
        self._decomp = decomp
        self._topo_field = fc.get_real_field('topography')
        self.__field = Field(self._topo_field)

        self.dx = grid['dx']
        self.dy = grid['dy']

        xx, yy = decomp.xx, decomp.yy

        # 1D profiles
        if geo['type'] == 'journal':
            h, dh_dx, dh_dy = journal_bearing(xx, grid, geo)
        elif geo['type'] == 'inclined':
            h, dh_dx, dh_dy = inclined_slider(xx, grid, geo)
        elif geo['type'] == 'parabolic':
            h, dh_dx, dh_dy = parabolic_slider(xx, grid, geo)
        elif geo['type'] == 'cdc':
            h, dh_dx, dh_dy = cdc(xx, grid, geo)

        # 2D profiles
        elif geo['type'] == 'asperity':
            h, dh_dx, dh_dy = asperity(xx, yy, grid, geo)
        elif geo['type'] == 'parabolic_2d':
            h, dh_dx, dh_dy = parabolic_2d(xx, yy, grid, geo)

        ix = 1
        iy = 2
        if geo['flip']:
            h = h.T
            dh_dx = dh_dx.T
            dh_dy = dh_dy.T
            ix = 2
            iy = 1

        # elastic deformation
        if prop['elastic']['enabled']:
            self.elastic = True
            self.h_undeformed = h.copy()
            self.__pressure = Field(fc.get_real_field('pressure'))

            self.ElasticDeformation = ElasticDeformation(
                E=prop['elastic']['E'],
                v=prop['elastic']['v'],
                alpha_underrelax=prop['elastic']['alpha_underrelax'],
                grid=grid,
                n_images=prop['elastic']['n_images'],
                decomp=decomp
            )
        else:
            self.elastic = False

        # Parse reference point for elastic deformation
        if self.elastic:
            self._ref_point = self._parse_reference_point(
                prop['elastic']['reference_point'], grid)

        self.__field.pg[0] = h
        self.__field.pg[ix] = dh_dx
        self.__field.pg[iy] = dh_dy
        self.__field.pg[3] = np.zeros_like(h)  # inital deformation set to zero

    def update(self) -> None:
        """Updates the topography field in case of enabled elastic deformation.
        """
        if self.elastic:
            if self.ElasticDeformation.periodicity in ['half', 'none']:
                p_ref = self.get_reference_pressure()
                p = self.__pressure.pg - p_ref
                deformation = self._calc_deformation(p)
                d_ref = self.get_reference_displacement(deformation)
                deformation = deformation - d_ref
            else:
                p = self.__pressure.pg
                deformation = self._calc_deformation(p)
            self.deformation = deformation
            self.h = self.h_undeformed + deformation

    def _calc_deformation(self, p):
        """Calculate elastic deformation from pressure field."""
        return self.ElasticDeformation.get_deformation_underrelax(p)

    def set_mapped_height(self, h_arr: NDArray) -> None:
        """Set height from local array computed via coordinate mapping.

        Use this when computing height from a function of coordinates, e.g.:
            h_new = f(problem.topo.xx, problem.topo.yy, t)
            problem.topo.set_mapped_height(h_new)

        Parameters
        ----------
        h_arr : NDArray
            Height array with shape `local_shape` (includes ghost cells).
        """
        if self.elastic:
            self.h_undeformed = h_arr.copy()
            self.h = self.h_undeformed + self.deformation
        else:
            self.h = h_arr

    def set_global_height(self, h_global: NDArray) -> None:
        """Set height from global array (distributed to processes in parallel).

        Use this when loading height from a dataset, e.g.:
            h_data = np.loadtxt('topography.txt')
            problem.topo.set_global_height(h_data)

        Parameters
        ----------
        h_global : NDArray
            Height array with shape `global_shape` (Nx, Ny), without ghost cells.
            Only needs to be valid on rank 0; other ranks can pass None.
        """
        h_inner = self._decomp.scatter_global(h_global)

        # Pad with ghost cells
        h_local = np.zeros(self.local_shape)
        h_local[1:-1, 1:-1] = h_inner

        self.set_mapped_height(h_local)

    def _parse_reference_point(self, ref_cfg, grid: dict) -> tuple:
        """Convert reference_point config to (owns, local_i, local_j).

        Parameters
        ----------
        ref_cfg : str or list
            "corner", "center", or [i, j] explicit inner grid indices
        grid : dict
            Grid configuration with Nx, Ny

        Returns
        -------
        tuple
            (owns, local_i, local_j) where owns indicates if this rank owns
            the reference point, and local indices include +1 ghost offset
        """
        Nx, Ny = grid['Nx'], grid['Ny']

        if ref_cfg == 'corner':
            gi, gj = 0, 0
        elif ref_cfg == 'center':
            gi, gj = Nx // 2, Ny // 2
        elif isinstance(ref_cfg, (list, tuple)):
            gi, gj = int(ref_cfg[0]), int(ref_cfg[1])
        else:
            raise ValueError(f"Invalid reference_point: {ref_cfg}")

        # Handle MPI decomposition (only for FEM 2D parallel runs)
        d = self._decomp
        if d is not None and d.size > 1:
            y_start = d.subdomain_locations[1]
            y_end = y_start + d.nb_subdomain_grid_pts[1]
            owns = (gj >= y_start and gj < y_end)
            local_j = (gj - y_start) + 1
        else:
            # Single rank: always owns the point
            owns = True
            local_j = gj + 1

        local_i = gi + 1  # +1 ghost offset (X not decomposed)
        return (owns, local_i, local_j)

    def get_reference_pressure(self) -> float:
        """Get reference pressure at configured point. MPI-safe."""
        owns, i, j = self._ref_point
        d = self._decomp

        if d is not None and d.size > 1:
            val = self.__pressure.pg[i, j] if owns else None
            comm = d._mpi_comm
            root = comm.allgather(owns).index(True)
            return float(comm.bcast(val, root=root))
        else:
            return float(self.__pressure.pg[i, j])

    def get_reference_displacement(self, deformation) -> float:
        """Get reference displacement at configured point. MPI-safe."""
        owns, i, j = self._ref_point
        d = self._decomp

        if d is not None and d.size > 1:
            val = deformation[i, j] if owns else None
            comm = d._mpi_comm
            root = comm.allgather(owns).index(True)
            return float(comm.bcast(val, root=root))
        else:
            return float(deformation[i, j])

    def _update_gradients(self) -> None:
        """Update gradients with proper MPI ghost cell handling.

        Steps:
        1. Sync h ghost cells from MPI neighbors
        2. At domain boundaries (non-periodic): linear extrapolation of h
        3. Compute gradients on inner points using central differences
        4. Sync gradient ghost cells from MPI neighbors
        5. At domain boundaries: copy gradient from first inner line
        """
        d = self._decomp

        # 1. Sync h ghost cells from MPI neighbors
        d.communicate_ghosts(self._topo_field)

        # 2. At domain boundaries: linear extrapolation of h (overrides periodic wrap)
        if d.is_at_xW and not d.periodic_x:
            self.h[0, :] = 2 * self.h[1, :] - self.h[2, :]
        if d.is_at_xE and not d.periodic_x:
            self.h[-1, :] = 2 * self.h[-2, :] - self.h[-3, :]
        if d.is_at_yS and not d.periodic_y:
            self.h[:, 0] = 2 * self.h[:, 1] - self.h[:, 2]
        if d.is_at_yN and not d.periodic_y:
            self.h[:, -1] = 2 * self.h[:, -2] - self.h[:, -3]

        # 3. Compute gradients on inner points (ghost h is now correct)
        self.dh_dx[1:-1, 1:-1] = (self.h[2:, 1:-1] - self.h[:-2, 1:-1]) / (2 * self.dx)
        self.dh_dy[1:-1, 1:-1] = (self.h[1:-1, 2:] - self.h[1:-1, :-2]) / (2 * self.dy)

        # 4. Sync gradient ghost cells from MPI neighbors
        d.communicate_ghosts(self._topo_field)

        # 5. At domain boundaries: copy gradient from first inner line (overrides periodic wrap)
        if d.is_at_xW and not d.periodic_x:
            self.dh_dx[0, :] = self.dh_dx[1, :]
            self.dh_dy[0, :] = self.dh_dy[1, :]
        if d.is_at_xE and not d.periodic_x:
            self.dh_dx[-1, :] = self.dh_dx[-2, :]
            self.dh_dy[-1, :] = self.dh_dy[-2, :]
        if d.is_at_yS and not d.periodic_y:
            self.dh_dx[:, 0] = self.dh_dx[:, 1]
            self.dh_dy[:, 0] = self.dh_dy[:, 1]
        if d.is_at_yN and not d.periodic_y:
            self.dh_dx[:, -1] = self.dh_dx[:, -2]
            self.dh_dy[:, -1] = self.dh_dy[:, -2]

    @property
    def full(self) -> NDArray:
        """Return the full topography array (height, slopes, and displacement)"""
        return self.__field.pg

    @property
    def h(self) -> NDArray:
        """Height field."""
        return self.__field.pg[0]

    @h.setter
    def h(self, value: NDArray) -> None:
        self.__field.pg[0] = value
        self._update_gradients()

    @property
    def deformation(self) -> NDArray:
        """Displacement field."""
        return self.__field.pg[3]

    @deformation.setter
    def deformation(self, value: NDArray) -> None:
        self.__field.pg[3] = value

    @property
    def dh_dx(self) -> NDArray:
        """Height gradient field (∂h/∂x)"""
        return self.__field.pg[1]

    @property
    def dh_dy(self) -> NDArray:
        """Height gradient field (∂h/∂y)"""
        return self.__field.pg[2]

    # ---------------------------
    # Coordinate and shape properties (forwarded from decomp)
    # ---------------------------

    @property
    def xx(self) -> NDArray:
        """Cell-center x-coordinates in physical units [m] (2D, with ghosts)."""
        return self._decomp.xx

    @property
    def yy(self) -> NDArray:
        """Cell-center y-coordinates in physical units [m] (2D, with ghosts)."""
        return self._decomp.yy

    @property
    def xx_norm(self) -> NDArray:
        """Normalized x-coordinates [0, 1] (2D, with ghosts)."""
        return self._decomp.xx_norm

    @property
    def yy_norm(self) -> NDArray:
        """Normalized y-coordinates [0, 1] (2D, with ghosts)."""
        return self._decomp.yy_norm

    @property
    def local_shape(self) -> tuple:
        """Local array shape (with ghosts)."""
        return self._decomp.local_shape_padded

    @property
    def global_shape(self) -> tuple:
        """Global array shape (Nx, Ny)."""
        return self._decomp.nb_domain_grid_pts


class ElasticDeformation:
    """Thin wrapper around the FFTElasticHalfSpace classes from ContactMechanics.

    Selects the appropriate Half-Space class based on the periodicity of the problem.
    Uses FFTDomainTranslation for MPI redistribution between GaPFlow and FFT domains.
    """

    def __init__(self,
                 E: float,
                 v: float,
                 alpha_underrelax: float,
                 grid: dict,
                 n_images: int,
                 decomp: DomainDecomposition
                 ) -> None:
        """Constructor

        Parameters
        ----------
        E : float
            Young's modulus.
        v : float
            Poisson's ratio.
        alpha_underrelax : float
            Underrelaxation factor.
        grid : dict
            Parameters controlling spatial discretization.
        n_images : int
            Number of periodic images for semi-periodic grids.
        decomp : DomainDecomposition
            Domain decomposition instance (used in both serial and MPI-parallel modes).
        """

        self.area_per_cell = grid['dx'] * grid['dy']
        self.alpha_underrelax = alpha_underrelax
        self.decomp = decomp

        # Use inner grid size (without ghost cells) for FFT computation
        Nx, Ny = grid['Nx'], grid['Ny']
        self._Nx = Nx
        self._Ny = Ny

        # For underrelaxation, store previous displacement (with ghost cells)
        local_shape = decomp.local_shape_padded  # (Nx+2, Ny_local+2)
        self.u_prev = np.zeros(local_shape)

        perX = grid['bc_xE'][0] == 'P'
        perY = grid['bc_yS'][0] == 'P'

        young_effective = E / (1 - v**2)

        # check for semi-periodic 1D cases where direction with N=1 is marked as periodic
        if (perX != perY) and ((perY and grid['Ny'] == 1) or (perX and grid['Nx'] == 1)):
            warnings.warn(
                "You specified a semi-periodic 1D problem.\n"
                "For the calculation of elastic deformation, we assume a line contact with "
                "non-periodic boundary conditions in both directions.\n"
                "For the calculation of the effective force F=p*A per cell, "
                "we assume a unit length of {} = 1."
                .format("Ly" if perY else "Lx"))
            grid = copy.deepcopy(grid)  # do not modify original grid
            if perY:
                grid['Ly'] = 1.0
            else:
                grid['Lx'] = 1.0
            n_images = 0  # make it effectively non-periodic

        # Create FFTDomainTranslation for MPI redistribution
        self.fft_translation = FFTDomainTranslation(decomp)
        fftengine = self.fft_translation.fft_engine

        if perX and perY:
            self.periodicity = 'full'
            self.ElDef = PeriodicFFTElasticHalfSpace(
                nb_grid_pts=(Nx, Ny),
                young=young_effective,
                physical_sizes=(grid['Lx'], grid['Ly']),
                stiffness_q0=0.0,
                fftengine=fftengine
            )
        elif (perX != perY):
            self.periodicity = 'half'
            self.ElDef = SemiPeriodicFFTElasticHalfSpace(
                nb_grid_pts=(Nx, Ny),
                young=young_effective,
                physical_sizes=(grid['Lx'], grid['Ly']),
                periodicity=(perX, perY),
                n_images=n_images,
                fftengine=fftengine
            )
        else:
            self.periodicity = 'none'
            self.ElDef = FreeFFTElasticHalfSpace(
                nb_grid_pts=(Nx, Ny),
                young=young_effective,
                physical_sizes=(grid['Lx'], grid['Ly']),
                fftengine=fftengine
            )

    def get_deformation(self, p: NDArray) -> NDArray:
        """Calculation of the elastic deformation due to given pressure field p.
        Convention: positive displacement for positive pressure.

        Parameters
        ----------
        p : ndarray
            Pressure array with ghost cells [Pa]. Expected shape includes ghost cells: (Nx+2, Ny+2)
            in serial, or (Nx+2, Ny_local+2) in MPI parallel mode.

        Returns
        -------
        disp : ndarray
            Array of resulting displacements [m]. Same shape as input. Includes ghost cells
            with zero displacement.
        """
        # Extract inner cells (exclude ghost cells)
        p_inner = p[1:-1, 1:-1]

        # Allocate FFT domain buffer
        fft_shape = tuple(self.fft_translation.fft_engine.nb_subdomain_grid_pts)
        p_fft = np.zeros(fft_shape, dtype=p.dtype)

        # Embed into FFT domain (handles MPI redistribution + zero-padding)
        self.fft_translation.embed(p_inner, p_fft)

        # Compute displacement via ContactMechanics
        forces_fft = p_fft * self.area_per_cell
        disp_fft = -self.ElDef.evaluate_disp(forces_fft)

        # Extract back to GaPFlow domain
        disp_inner = np.zeros_like(p_inner)
        self.fft_translation.extract(disp_fft, disp_inner)

        # Pad result back to include ghost cells
        disp = np.zeros_like(p)
        disp[1:-1, 1:-1] = disp_inner

        return disp

    def get_deformation_underrelax(self, p: NDArray) -> NDArray:
        """Updates elastic deformation using underrelaxation

        Parameters
        ----------
        p : ndarray of shape (M, N)
            Pressure field.

        Returns
        -------
        u_relaxed : ndarray of shape (M, N)
            Updated, underrelaxed deformation field.
        """
        u_computed = self.get_deformation(p)
        u_relaxed = (1 - self.alpha_underrelax) * self.u_prev + self.alpha_underrelax * u_computed
        self.u_prev = u_relaxed.copy()

        return u_relaxed

    def get_G_real(self) -> NDArray:
        """For analysis and illustration purposes.
        Returns the 'ordered' G_real numpy array with centered zero frequency component.

        Returns
        -------
        G_real_ordered : ndarray
            Green's function array in real space.
        """
        return self.ElDef.get_G_real()

    def get_G_real_slices(self) -> Tuple[NDArray, NDArray]:
        """For analysis and illustration purposes.
        Returns two middle slices of the G_real array, in x- and y-direction

        Returns
        -------
        x_slice : ndarray of shape (M,)
            middle slice of G_real in x-direction
        y_slice : ndarray of shape (N,)
            middle slice of G_real in y-direction
        """
        return self.ElDef.get_G_real_slices()
