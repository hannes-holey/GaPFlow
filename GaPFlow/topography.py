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
import ContactMechanics as CM
import numpy as np
import copy

import numpy.typing as npt
from typing import Tuple, Any
from muGrid.Field import wrap_field

import warnings

NDArray = npt.NDArray[np.floating]


def create_midpoint_grid(disc):
    Lx = disc['Lx']
    Ly = disc['Ly']
    Nx = disc['Nx']
    Ny = disc['Ny']

    dx = Lx / Nx
    ix = np.arange(-1, Nx + 1)
    x = ix / Nx * Lx + dx / 2.

    dy = Ly / Ny
    iy = np.arange(-1, Ny + 1)
    y = iy / Ny * Ly + dy / 2.

    xx, yy = np.meshgrid(x, y, indexing='ij')

    return xx, yy


def journal_bearing(xx, grid, geo):

    Lx = grid['Lx']
    freq = 2. * np.pi / Lx

    if 'CR' and 'eps' in geo.keys():
        shift = geo['CR'] / freq
        amp = geo['eps'] * shift

    elif 'hmin' and 'hmax' in geo.keys():
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


class Topography:
    """Topography container.

    Holds the rigid gap topography, and, in case of an elastic substrate, its deformation
    due to the fluid pressure field.
    """

    def __init__(self,
                 fc: Any,
                 grid: dict,
                 geo: dict,
                 prop: dict) -> None:
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
        """

        xx, yy = create_midpoint_grid(grid)

        self.__field = wrap_field(fc.get_real_field('topography'))
        self._x = wrap_field(fc.get_real_field('x'))
        self._y = wrap_field(fc.get_real_field('y'))

        self._x.p[...] = xx
        self._y.p[...] = yy

        self.dx = grid['dx']
        self.dy = grid['dy']

        # TODO: height profiles via callable passed to topography (next to implemented ones)

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
            self.__pressure = wrap_field(fc.get_real_field('pressure'))

            self.ElasticDeformation = ElasticDeformation(
                E=prop['elastic']['E'],
                v=prop['elastic']['v'],
                alpha_underrelax=prop['elastic']['alpha_underrelax'],
                grid=grid,
                n_images=prop['elastic']['n_images']
            )
        else:
            self.elastic = False

        self.__field.p[0] = h
        self.__field.p[ix] = dh_dx
        self.__field.p[iy] = dh_dy
        self.__field.p[3] = np.zeros_like(h)  # inital deformation set to zero

    def update(self) -> None:
        """Updates the topography field in case of enabled elastic deformation.
        """
        if self.elastic:
            if self.ElasticDeformation.periodicity in ['half', 'none']:
                p = self.__pressure.p - self.__pressure.p[0, 0]  # reference pressure
                deformation = self.ElasticDeformation.get_deformation_underrelax(p)
                deformation = deformation - deformation[0, 0]  # reference deformation
            else:
                p = self.__pressure.p
                deformation = self.ElasticDeformation.get_deformation_underrelax(p)
            self.deformation = deformation
            self.h = self.h_undeformed + deformation
        else:
            pass

    def update_gradients(self) -> None:
        """Updates gradient arrays using second-order central differences.
        """
        h = self.__field.p[0]
        dh_dx = np.gradient(h, axis=0) / self.dx
        dh_dy = np.gradient(h, axis=1) / self.dy
        self.__field.p[1] = dh_dx
        self.__field.p[2] = dh_dy

    @property
    def full(self) -> NDArray:
        """Return the full topography array (height, slopes, and displacement)"""
        return self.__field.p

    @property
    def h(self) -> NDArray:
        """Height field."""
        return self.__field.p[0]

    @h.setter
    def h(self, value: NDArray) -> None:
        self.__field.p[0] = value
        self.update_gradients()

    @property
    def deformation(self) -> NDArray:
        """Displacement field."""
        return self.__field.p[3]

    @deformation.setter
    def deformation(self, value: NDArray) -> None:
        self.__field.p[3] = value

    @property
    def dh_dx(self) -> NDArray:
        """Height gradient field (∂h/∂x)"""
        return self.__field.p[1]

    @property
    def dh_dy(self) -> NDArray:
        """Height gradient field (∂h/∂y)"""
        return self.__field.p[2]

    @property
    def x(self):
        """Cell center x coordinates"""
        return self._x.p

    @property
    def y(self):
        """Cell center y coordinates"""
        return self._y.p


class ElasticDeformation:
    """Thin wrapper around the FFTElasticHalfSpace classes from ContactMechanics.

    Selects the appropriate Half-Space class based on the periodicity of the problem.
    """

    def __init__(self,
                 E: float,
                 v: float,
                 alpha_underrelax: float,
                 grid: dict,
                 n_images: int
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
        """

        self.area_per_cell = grid['dx'] * grid['dy']
        Nx, Ny = grid['Nx'] + 2, grid['Ny'] + 2
        self.u_prev = np.zeros((Nx, Ny))
        self.alpha_underrelax = alpha_underrelax
        n_images = n_images

        perX = grid['bc_xE_P'][0]
        perY = grid['bc_yS_P'][0]

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

        if perX and perY:
            self.periodicity = 'full'
            self.ElDef = CM.PeriodicFFTElasticHalfSpace(nb_grid_pts=(Nx, Ny),
                                                        young=young_effective,
                                                        physical_sizes=(grid['Lx'], grid['Ly']),
                                                        stiffness_q0=0.0
                                                        )
        elif (perX != perY):
            self.periodicity = 'half'
            self.ElDef = CM.SemiPeriodicFFTElasticHalfSpace(nb_grid_pts=(Nx, Ny),
                                                            young=young_effective,
                                                            physical_sizes=(grid['Lx'], grid['Ly']),
                                                            periodicity=(perX, perY),
                                                            n_images=n_images
                                                            )
        else:
            self.periodicity = 'none'
            self.ElDef = CM.FreeFFTElasticHalfSpace(nb_grid_pts=(Nx, Ny),
                                                    young=young_effective,
                                                    physical_sizes=(grid['Lx'], grid['Ly'])
                                                    )

    def get_deformation(self, p: NDArray) -> NDArray:
        """Calculation of the elastic deformation due to given pressure field p.
        Convention: positive displacement for positive pressure.

        Parameters
        ----------
        p : ndarray
            Pressure array [Pa]

        Returns
        -------
        disp : ndarray
            Array of resulting displacements [m].
        """
        forces = p * self.area_per_cell
        disp = -self.ElDef.evaluate_disp(forces)

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
