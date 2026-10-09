#
# Copyright 2025-2026 Christoph Huber
#           2025 Hannes Holey
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
import copy
import os
import warnings
from typing import Tuple, Any
import numpy as np
import numpy.typing as npt
from muGrid import Field

from ContactMechanics.FFTElasticHalfSpace import (
    PeriodicFFTElasticHalfSpace,
    FreeFFTElasticHalfSpace,
    SemiPeriodicFFTElasticHalfSpace,
)

from .parallel import DomainDecomposition, FFTDomainTranslation, MPI

_MPI_MIN = MPI.MIN


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
    cx = geo['x_contact_center']
    # slope = (h1 - h0) / Lx

    prefac = 4. / Lx**2 * (h1 - h0)

    dx = grid['dx']
    xx_ = (xx - 1.5 * dx) * Lx / (Lx - 3 * dx)
    xx_ = np.clip(xx_, 0.0, Lx)

    h = prefac * (xx_ - cx)**2 + h0
    dh_dx = 2 * prefac * (xx_ - cx)
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


def circular_1d(xx, grid, geo):
    """Line-contact (1D, Ny=1) cylinder topography, parameterized by curvature
    radius Rx instead of an hmin/hmax pair.

    Parameters
    ----------
    xx : NDArray
        X-coordinate grid.
    grid : dict
        Grid parameters with keys 'Lx', 'Ly'.
    geo : dict
        Geometry parameters with keys 'Rx', 'hmin', 'x_contact_center'
        (contact apex location).
    """

    cx = geo['x_contact_center']
    Rx = geo['Rx']
    hmin = geo['hmin']

    h = (xx - cx)**2 / (2 * Rx) + hmin
    dh_dx = (xx - cx) / Rx
    dh_dy = np.zeros_like(h)

    return h, dh_dx, dh_dy


def circular_contact(xx, yy, grid, geo):
    """Circular contact topography.

    Creates a circular contact area with specified radius and height.

    Parameters
    ----------
    xx : NDArray
        X-coordinate grid.
    yy : NDArray
        Y-coordinate grid.
    grid : dict
        Grid parameters with keys 'Lx', 'Ly'.
    geo : dict
        Geometry parameters with keys 'Rx', 'Ry', 'h_min', 'x_contact_center',
        'y_contact_center' (contact apex location).
    """

    cx = geo['x_contact_center']
    cy = geo['y_contact_center']

    Rx = geo['Rx']
    Ry = geo['Ry']
    hmin = geo['hmin']

    h = (xx - cx)**2 / (2 * Rx) + (yy - cy)**2 / (2 * Ry) + hmin
    dh_dx = (xx - cx) / Rx
    dh_dy = (yy - cy) / Ry
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
                 decomp: DomainDecomposition = None,
                 force_balance: dict = None,
                 problem: Any = None) -> None:
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
        force_balance : dict or None
            Force balance / load control settings.
        problem : Problem or None
            Owning Problem instance, used to read FEM Newton solver guard state.
        """
        self._decomp = decomp
        self._problem = problem
        self._topo_field = fc.get_real_field('topography')
        self.__field = Field(self._topo_field)

        self.dx = grid['dx']
        self.dy = grid['dy']
        self.h0 = 0.

        self.elastic = prop['elastic']['enabled']
        self.force_balance = force_balance is not None and force_balance['rigid_height_variation']['enabled']
        self._suspended = False

        xx, yy = decomp.xx, decomp.yy

        idc = force_balance is not None and force_balance['init_dry_contact']
        if idc and idc['enabled'] and idc['use_deformed_height']:
            h, u = self.height_and_defo_from_file(geo)
            self.h0 = idc['h_min_init'] - (h + u).min()
            self.set_global_height(h)
            self.set_global_deformation(u)

        elif geo['type'] == 'from_file':
            h = self.height_from_file(geo)
            self.set_global_height(h)

        else:
            h, dh_dx, dh_dy = self.compute_topography(xx, grid, geo, yy)
            self.set_local_topography(h, dh_dx, dh_dy)

        if self.elastic:
            self.init_elastic(prop, grid, decomp, fc)

        if self.force_balance:
            self.init_force_balance(force_balance, grid, decomp)

        self.check_flip(geo)

    def init_force_balance(self, force_balance, grid, decomp):
        """Initialise rigid-height-variation force balance controller."""
        self._fb_controller = ForceBalance(force_balance, grid, decomp, problem=self._problem)
        self.rhv_history = []

    def set_local_topography(self, h, dh_dx, dh_dy):
        """Sets local topography field.
        """
        self.__field.pg[0] = h
        self.dh_dx = dh_dx
        self.dh_dy = dh_dy
        self.h_undeformed = h

    def check_flip(self, geo):

        if geo['flip']:
            h = np.copy(self.h)
            dh_dx = np.copy(self.dh_dx)
            dh_dy = np.copy(self.dh_dy)
            h = h.T
            dh_dx = dh_dx.T
            dh_dy = dh_dy.T
            ix = 2
            iy = 1

            self.__field.pg[0] = h
            self.__field.pg[ix] = dh_dx
            self.__field.pg[iy] = dh_dy

    def init_elastic(self, prop, grid, decomp, fc):
        """Initializes elastic deformation object and reference point.
        """

        self.__pressure = Field(fc.get_real_field('pressure'))

        self.ElasticDeformation = ElasticDeformation(
            E=prop['elastic']['E'],
            v=prop['elastic']['v'],
            alpha_underrelax=prop['elastic']['alpha_underrelax'],
            grid=grid,
            n_images=prop['elastic']['n_images'],
            decomp=decomp,
            thickness=prop['elastic']['thickness'],
            u_init=self.deformation,
        )
        self._ref_point = self._parse_reference_point(
            prop['elastic']['reference_point'], grid)
        self.turn_off_reference = prop['elastic']['turn_off_reference']

    @staticmethod
    def compute_topography(xx, grid, geo, yy=None):
        """Predefined topography functions. Also externally used by dry_contact.

        Parameters
        ----------
        xx : NDArray
            coordinate grid in x
        grid : dict
            Grid parameters with keys 'Lx', 'Ly'.
        geo : dict
            Geometry parameters with keys 'hmin', 'hmax'.
        yy : NDArray, optional
            coordinate grid in y, by default None

        Returns
        -------
        h : NDArray
            Height field [m].
        dh_dx : NDArray
            Height gradient in x-direction.
        dh_dy : NDArray
            Height gradient in y-direction.
        """

        if geo['x_contact_center'] is None:
            geo['x_contact_center'] = grid['Lx'] / 2
        if geo['y_contact_center'] is None:
            geo['y_contact_center'] = grid['Ly'] / 2

        # 1D profiles
        if geo['type'] == 'journal':
            h, dh_dx, dh_dy = journal_bearing(xx, grid, geo)
        elif geo['type'] == 'inclined':
            h, dh_dx, dh_dy = inclined_slider(xx, grid, geo)
        elif geo['type'] == 'parabolic':
            h, dh_dx, dh_dy = parabolic_slider(xx, grid, geo)
        elif geo['type'] == 'cdc':
            h, dh_dx, dh_dy = cdc(xx, grid, geo)
        elif geo['type'] == 'circular_1d':
            h, dh_dx, dh_dy = circular_1d(xx, grid, geo)

        # 2D profiles
        elif geo['type'] == 'asperity':
            h, dh_dx, dh_dy = asperity(xx, yy, grid, geo)
        elif geo['type'] == 'parabolic_2d':
            h, dh_dx, dh_dy = parabolic_2d(xx, yy, grid, geo)
        elif geo['type'] == 'circular_contact':
            h, dh_dx, dh_dy = circular_contact(xx, yy, grid, geo)

        return h, dh_dx, dh_dy

    @staticmethod
    def height_from_file(geo):
        base_path = geo['basepath']
        file_path = geo['height_filepath']
        h = np.load(os.path.join(base_path, file_path))
        return h

    @staticmethod
    def height_and_defo_from_file(geo):
        base_path = geo['basepath']
        h = np.load(os.path.join(base_path, geo['height_filepath']))
        u = np.load(os.path.join(base_path, geo['deformation_filepath']))
        return h, u

    def update(self, it: int = 0, suspend_fb: bool = False, suspend_def: bool = False) -> None:
        """Updates the topography field in case of enabled elastic deformation
        and/or rigid-height-variation force balance.
        For full periodicity, no reference needed (displacement sum is zero).
        For half/no periodicity, displacement at reference point is kept to zero.

        `it` is the current Newton inner-iteration index. Elastic deformation is
        recomputed every iteration, but the force-balance bisection step on `h0` only
        fires on the first iteration (`it == 0`) of each outer timestep, so the
        remaining inner iterations stabilize against a fixed `h0`.
        """
        defo_disc = 0.0
        if self.elastic and not self._suspended and not suspend_def:
            if self.ElasticDeformation.periodicity in ['half', 'none'] and not self.turn_off_reference:
                p_ref = self.get_reference_pressure()
                p = self.__pressure.pg - p_ref
                deformation, defo_disc = self._calc_deformation(p)
                d_ref = self.get_reference_displacement(deformation)
                deformation = deformation - d_ref
            else:
                p = self.__pressure.pg
                deformation, defo_disc = self._calc_deformation(p)
            defo_disc = self._decomp._mpi_comm.allreduce(defo_disc, op=MPI.MAX)
            self.deformation = deformation

        if self.force_balance and it == 0 and not self._suspended and not suspend_fb:
            pid_hold_tol = self._fb_controller._pid_hold_tol
            defo_hold = pid_hold_tol > 0. and defo_disc > pid_hold_tol
            if defo_hold:
                print(f"  [ForceBalance] PID on hold: defo_disc={defo_disc:.3e} > tol={pid_hold_tol:.3e}")
                self.rhv_history.append(self.h0)
            else:
                self.h0 = self._fb_controller.update(self)
                self.rhv_history.append(self.h0)

        if self.elastic or self.force_balance:
            self.h = self.h_undeformed + self.deformation + self.h0

    def _calc_deformation(self, p):
        """Calculate elastic deformation from pressure field."""
        return self.ElasticDeformation.get_deformation_underrelax(p, h=self.h)

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
        if self.elastic or self.force_balance:
            self.h_undeformed = h_arr.copy()
            self.h = self.h_undeformed + self.deformation + self.h0
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

    def set_global_deformation(self, u_global: NDArray) -> None:
        """Set deformation from global array (distributed to processes in parallel).

        Seeds the deformation field (and recomputes h) from a precomputed
        displacement field, e.g. a dry-contact warm start, without altering
        h_undeformed. Does not seed the ElasticDeformation underrelaxation
        history (u_prev); that must be seeded separately at construction.

        Parameters
        ----------
        u_global : NDArray
            Deformation array with shape `global_shape` (Nx, Ny), without ghost cells.
            Only needs to be valid on rank 0; other ranks can pass None.
        """
        u_inner = self._decomp.scatter_global(u_global)

        # Pad with ghost cells
        u_local = np.zeros(self.local_shape)
        u_local[1:-1, 1:-1] = u_inner

        self.deformation = u_local
        self.h = self.h_undeformed + self.deformation + self.h0

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
            x_start = d.subdomain_locations[0]
            x_end = x_start + d.nb_subdomain_grid_pts[0]
            y_start = d.subdomain_locations[1]
            y_end = y_start + d.nb_subdomain_grid_pts[1]
            owns = (gi >= x_start and gi < x_end) and (gj >= y_start and gj < y_end)
            local_i = (gi - x_start) + 1
            local_j = (gj - y_start) + 1
        else:
            # Single rank: always owns the point
            owns = True
            local_i = gi + 1
            local_j = gj + 1

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
        d.exchange_ghosts(self._topo_field)

        # 2. At domain boundaries: linear extrapolation of h
        if d.is_at_xW and not d.periodic_x:
            # self.h[0, :] = 2 * self.h[1, :] - self.h[2, :]
            self.h[0, :] = self.h[1, :]
        if d.is_at_xE and not d.periodic_x:
            # self.h[-1, :] = 2 * self.h[-2, :] - self.h[-3, :]
            self.h[-1, :] = self.h[-2, :]
        if d.is_at_yS and not d.periodic_y:
            # self.h[:, 0] = 2 * self.h[:, 1] - self.h[:, 2]
            self.h[:, 0] = self.h[:, 1]
        if d.is_at_yN and not d.periodic_y:
            # self.h[:, -1] = 2 * self.h[:, -2] - self.h[:, -3]
            self.h[:, -1] = self.h[:, -2]

        # 3. Compute gradients on inner points (ghost h is now correct)
        self.dh_dx[1:-1, 1:-1] = (self.h[2:, 1:-1] - self.h[:-2, 1:-1]) / (2 * self.dx)
        self.dh_dy[1:-1, 1:-1] = (self.h[1:-1, 2:] - self.h[1:-1, :-2]) / (2 * self.dy)

        # 4. Sync gradient ghost cells from MPI neighbors
        d.exchange_ghosts(self._topo_field)

        # 5. At domain boundaries: override periodic wrap again: linear extrapolation of h
        # and copy gradient from first inner line
        if d.is_at_xW and not d.periodic_x:
            # self.h[0, :] = 2 * self.h[1, :] - self.h[2, :]
            self.h[0, :] = self.h[1, :]
        if d.is_at_xE and not d.periodic_x:
            # self.h[-1, :] = 2 * self.h[-2, :] - self.h[-3, :]
            self.h[-1, :] = self.h[-2, :]
        if d.is_at_yS and not d.periodic_y:
            # self.h[:, 0] = 2 * self.h[:, 1] - self.h[:, 2]
            self.h[:, 0] = self.h[:, 1]
        if d.is_at_yN and not d.periodic_y:
            # self.h[:, -1] = 2 * self.h[:, -2] - self.h[:, -3]
            self.h[:, -1] = self.h[:, -2]

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
        """Height field with ghost cells."""
        return self.__field.pg[0]

    @h.setter
    def h(self, value: NDArray) -> None:
        self.__field.pg[0] = value
        self._update_gradients()

    @property
    def deformation(self) -> NDArray:
        """Displacement field with ghost cells."""
        return self.__field.pg[3]

    @deformation.setter
    def deformation(self, value: NDArray) -> None:
        self.__field.pg[3] = value

    @property
    def dh_dx(self) -> NDArray:
        """Height gradient field (∂h/∂x) with ghost cells."""
        return self.__field.pg[1]

    @dh_dx.setter
    def dh_dx(self, value: NDArray) -> None:
        self.__field.pg[1] = value

    @property
    def dh_dy(self) -> NDArray:
        """Height gradient field (∂h/∂y) with ghost cells."""
        return self.__field.pg[2]

    @dh_dy.setter
    def dh_dy(self, value: NDArray) -> None:
        self.__field.pg[2] = value

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


class ForceBalance:
    """Bisection-controlled rigid-height-variation for load/force balance.

    Each timestep (after pressure is updated) computes the total film force,
    compares it to the imposed load, and adjusts the rigid body height offset
    h0 via a two-phase bisection controller.

    The update is always performed on rank 0 (which holds the global pressure)
    and the resulting h0 is broadcast to all ranks.
    """

    def __init__(self, fb_dict: dict, grid: dict, decomp, problem: Any = None) -> None:
        self._fb_dict = fb_dict
        self._decomp = decomp
        self._comm = decomp._mpi_comm
        self._problem = problem

        self._dA = grid['dx'] * grid['dy']
        self._p_ambient = fb_dict['rigid_height_variation']['ambient_pressure']
        self._force_imposed = self._get_force_imposed(grid)
        self._h0_prev = 0.
        self._pid_hold_tol = float(fb_dict.get('pid_hold_tol', 0.0))

        self._method = fb_dict['rigid_height_variation']['method']
        if self._method != 'bisection':
            raise IOError(f"Unknown rigid_height_variation method: '{self._method}'")

        if decomp.rank == 0:
            rhv = fb_dict['rigid_height_variation']
            self.controller = BisectionForceController(
                rhv['force_tol'], rhv['h_min_step_divisor'], problem.solver)
        else:
            self.controller = None

    def _get_force_imposed(self, grid):
        fb = self._fb_dict
        if 'force' in fb:
            force = float(fb['force'])
        elif 'pressure' in fb:
            force = float(fb['pressure']) * grid['Lx'] * grid['Ly']
        else:
            raise IOError("Need 'force' or 'pressure' in force_balance.")
        assert force > 0., "force_balance: imposed force must be positive."
        return force

    def update(self, topo) -> float:
        """Compute new rigid height offset h0.

        Gathers the global pressure field to rank 0, evaluates the total film
        force, computes the normalised residual, runs the bisection step, and
        broadcasts h0 to all ranks.

        Parameters
        ----------
        topo : Topography
            Current Topography instance (provides h).

        Returns
        -------
        float
            Updated h0 [m].
        """
        # Pressure lives on the padded P1 grid; gather inner part to rank 0
        p_local = np.maximum(self._problem.pressure.pressure, 0.)
        p_global = self._decomp.gather_global(p_local)

        qf_h = self._problem.solver.quad_mgr.qf('h')[..., :-1, :-1]
        qf_h_min = float(np.min(qf_h)) if np.any(qf_h) else float(np.min(topo.h[1:-1, 1:-1]))
        h_min = self._comm.allreduce(qf_h_min, op=_MPI_MIN)

        if self._decomp.rank == 0:
            p_reduced = p_global - self._p_ambient
            force_measured = float(np.sum(p_reduced * self._dA))
            p_max = float(np.max(p_global))
            residual = force_measured / self._force_imposed - 1.
            h0 = self.controller.update(residual, self._h0_prev, h_min)
            print(f"\033[36m  [FB] h_min={h_min:.3e}  p_max={p_max:.4g}  "
                  f"F_film={force_measured:.4g}  F_imposed={self._force_imposed:.4g} h0={h0:.4e} "
                  f"\033[0m")
        else:
            h0 = None

        self._h0_prev = self._comm.bcast(h0, root=0)
        return self._h0_prev


class BisectionForceController:
    """Two-phase bisection controller for rigid-height-variation force balance.

    Phase 1 ("descent"): starting from the initial h0 (film force too low),
    decrease h0 in h_min/h_min_step_divisor-sized steps (same limiter as
    `ForceBalance._solution_guard`) until the film force first exceeds the
    imposed force. That overshoot ends phase 1.

    Phase 2 ("bisect"): step h0 by a fixed magnitude `step` (initialised to
    half the last phase-1 step) in whichever direction reduces the residual.
    `step` is only halved when the residual's sign flips relative to the
    previous applied step (i.e. on overshoot), not on every iteration.

    Each proposed h0 change is only trusted/applied once at least 5
    effective-relaxation-equivalent Newton iterations (`sum(alpha)`) have
    accumulated since the last change, via `accumulate()` registered as a
    callback on `FEMSolver.cb_list`. This approximates "the Newton solve
    has settled enough to trust the force reading" for this
    pseudo-timestepping / quasi-steady-state setup.
    """

    SETTLE_THRESHOLD = 5.0

    def __init__(self, force_tol: float, h_min_step_divisor: float, solver) -> None:
        self.force_tol = force_tol
        self.h_min_step_divisor = h_min_step_divisor
        self.phase = 'descent'
        self.step = None
        self.last_dh0 = None
        self.last_residual_sign = None
        self.sum_alpha = 0.0
        self.converged = False

        solver.cb_list.append(self.accumulate)

    def accumulate(self, solver) -> None:
        self.sum_alpha += solver.alpha

    def reset(self) -> None:
        self.sum_alpha = 0.0

    def is_trustworthy(self) -> bool:
        return self.sum_alpha > self.SETTLE_THRESHOLD

    def restart_for_new_target(self) -> None:
        """Reset all controller state to start bisecting toward a new imposed force.

        Use after changing `ForceBalance._force_imposed` mid-run (e.g. a force
        sweep): the previous target's converged phase/step/sign history is not
        valid for a different target force.
        """
        self.phase = 'descent'
        self.step = None
        self.last_dh0 = None
        self.last_residual_sign = None
        self.sum_alpha = 0.0
        self.converged = False

    def update(self, residual: float, h0_prev: float, h_min: float) -> float:
        if self.phase == 'descent':
            return self._update_descent(residual, h0_prev, h_min)
        return self._update_bisect(residual, h0_prev)

    def _update_descent(self, residual: float, h0_prev: float, h_min: float) -> float:
        if residual > 0.:
            # First overshoot: end phase 1, seed phase 2's step from the last
            # phase-1 step. Do not move h0 further this call.
            self.phase = 'bisect'
            self.step = (abs(self.last_dh0) / 2. if self.last_dh0 is not None
                         else h_min / self.h_min_step_divisor / 2.)
            self.last_residual_sign = np.sign(residual)
            print(f"  [FB-bisect] overshoot detected, switching to bisect phase "
                  f"(step={self.step:.3e})")
            return h0_prev

        dh0 = -h_min / self.h_min_step_divisor
        self.last_dh0 = dh0
        self.reset()
        print(f"  [FB-bisect] descent step dh0={dh0:.3e}")
        return h0_prev + dh0

    def _update_bisect(self, residual: float, h0_prev: float) -> float:
        if not self.is_trustworthy():
            print(f"  [FB-bisect] held: not settled yet (sum_alpha={self.sum_alpha:.3e} "
                  f"<= {self.SETTLE_THRESHOLD:.1f})")
            return h0_prev

        if abs(residual) <= self.force_tol:
            if not self.converged:
                print(f"  [FB-bisect] converged: |residual|={abs(residual):.3e} <= tol={self.force_tol:.3e}")
            self.converged = True
            return h0_prev

        residual_sign = np.sign(residual)
        if self.last_residual_sign is not None and residual_sign != self.last_residual_sign:
            self.step /= 2.
            print(f"  [FB-bisect] sign flip, halving step to {self.step:.3e}")
        self.last_residual_sign = residual_sign

        # residual > 0 -> force too high -> increase h0 (larger gap)
        dh0 = self.step if residual > 0. else -self.step
        self.reset()
        print(f"  [FB-bisect] bisect step dh0={dh0:.3e}")
        return h0_prev + dh0


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
                 decomp: DomainDecomposition,
                 thickness: float = None,
                 u_init: NDArray = None
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
        thickness : float, optional
            Elastic layer thickness [m]. If given, uses finite-thickness Green's function
            for the periodic case, giving a finite q=0 stiffness under uniform pressure.
        """

        self.area_per_cell = grid['dx'] * grid['dy']
        self.alpha_underrelax = alpha_underrelax
        self.decomp = decomp

        # Use inner grid size (without ghost cells) for FFT computation
        Nx, Ny = grid['Nx'], grid['Ny']
        self._Nx = Nx
        self._Ny = Ny

        # For underrelaxation, store previous displacement (with ghost cells)
        self.u_prev = u_init

        perX = grid['bc_xE'][0] == 'P'
        perY = grid['bc_yS'][0] == 'P'

        young_effective = E / (1 - v**2)

        # check for semi-periodic 1D cases where direction with N=1 is marked as periodic
        if (perX != perY) and ((perY and grid['Ny'] == 1) or (perX and grid['Nx'] == 1)):
            warnings.warn(
                "You specified a semi-periodic 1D problem.\n"
                "For the calculation of elastic deformation, we assume a line contact, "
                "approximated via periodic images (n_images={}) along the {} == 1 "
                "direction; internally this solver's own copy of {} is set to 1 "
                "(force per cell F=p*A is computed from the grid's own physical "
                "{}, unaffected by this internal override)."
                .format(n_images, "Ny" if perY else "Nx",
                        "Ly" if perY else "Lx", "Ly" if perY else "Lx"))
            grid = copy.deepcopy(grid)  # do not modify original grid
            if perY:
                grid['Ly'] = 1.0
            else:
                grid['Lx'] = 1.0

        # Create FFTDomainTranslation for MPI redistribution
        self.fft_translation = FFTDomainTranslation(decomp)
        fftengine = self.fft_translation.fft_engine

        if perX and perY:
            self.periodicity = 'full'
            if thickness is not None:
                self.ElDef = PeriodicFFTElasticHalfSpace(
                    nb_grid_pts=(Nx, Ny),
                    young=young_effective,
                    physical_sizes=(grid['Lx'], grid['Ly']),
                    thickness=thickness,
                    poisson=v,
                    fftengine=fftengine
                )
            else:
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

    def get_deformation_underrelax(self, p: NDArray, h: NDArray | None = None) -> tuple:
        """Updates elastic deformation using underrelaxation

        Parameters
        ----------
        p : ndarray of shape (M, N)
            Pressure field.
        h : ndarray of shape (M, N) or None
            Gap height field (with ghost cells). If provided, the deformation
            discrepancy metric max(|u_relaxed - u_computed| / h) is computed
            over inner cells and returned as the second element of the tuple.

        Returns
        -------
        u_relaxed : ndarray of shape (M, N)
            Updated, underrelaxed deformation field.
        disc : float
            Maximum deformation discrepancy relative to gap height
            (max over inner cells of |u_relaxed - u_computed| / h).
            Zero if h is None.
        """
        u_computed = self.get_deformation(p)
        alpha = self.alpha_underrelax
        u_relaxed = (1 - alpha) * self.u_prev + alpha * u_computed
        self.u_prev = u_relaxed.copy()

        if h is not None:
            disc = float(np.max(
                np.abs(u_relaxed[1:-1, 1:-1] - u_computed[1:-1, 1:-1]) / h[1:-1, 1:-1]
            ))
        else:
            disc = 0.0

        return u_relaxed, disc

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
