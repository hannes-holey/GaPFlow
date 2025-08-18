#
# Copyright 2020-2022, 2024-2025 Hannes Holey
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
from mpi4py import MPI
from scipy import signal

from hans.field import ScalarField, VectorField
from hans.tools import abort


class GapHeight(VectorField):

    def __init__(self, disc, geometry, roughness=None):

        super().__init__(disc)

        self.geometry = geometry
        self.disc = disc
        self.roughness = roughness

        self.set_profile()

        if self.roughness is not None:
            if "file" in self.roughness.keys():
                self.add_roughness_from_file()
            else:
                self.add_roughness()

        self.set_gradients()

    @property
    def max_height(self):
        local_max = np.amax(self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_max, recvbuf, op=MPI.MAX)

        return recvbuf[0]

    @property
    def min_height(self):
        local_max = np.amin(self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_max, recvbuf, op=MPI.MIN)

        return recvbuf[0]

    def set_profile(self):
        """Generation of the gap height field (the height profile).
        For possible options refer to "Input options".
        """
        Lx = self.disc["Lx"]
        Ly = self.disc["Ly"]
        Nx = self.disc["Nx"]
        Ny = self.disc["Ny"]
        dx = self.disc["dx"]
        dy = self.disc["dy"]

        idxx, idyy = self.id_grid

        ng = self.disc["nghost"]

        # using xx, yy or combination of both arrays to encode some x- or y-dependency
        xx = idxx * (Lx + 2 * ng * dx) / (Nx + 2 * ng) + dx / 2
        yy = idyy * (Ly + 2 * ng * dy) / (Ny + 2 * ng) + dy / 2

        if self.geometry["type"] in ["journal", "journal_x"]:
            CR = self.geometry['CR']
            eps = self.geometry['eps']

            Rb = Lx / (2 * np.pi)
            c = CR * Rb
            e = eps * c
            self.field[0] = c + e * np.cos(xx / Rb)

        elif self.geometry["type"] == "journal_y":
            CR = self.geometry['CR']
            eps = self.geometry['eps']

            Rb = Ly / (2 * np.pi)
            c = CR * Rb
            e = eps * c
            self.field[0] = c + e * np.cos(yy / Rb)

        elif self.geometry["type"] == "parabolic":
            hmin = self.geometry['hmin']
            hmax = self.geometry['hmax']
            self.field[0] = 4 * (hmax - hmin) / Lx**2 * (xx - Lx / 2)**2 + hmin

        elif self.geometry["type"] == "twin_parabolic":
            hmin = self.geometry['hmin']
            hmax = self.geometry['hmax']

            right = np.greater(xx, Lx / 2)
            self.field[0] = 16 * (hmax - hmin) / Lx**2 * (xx - Lx / 4)**2 + hmin
            self.field[0][right] = 16 * (hmax - hmin) / Lx**2 * (xx[right] - 3 * Lx / 4)**2 + hmin

        elif self.geometry["type"] in ["inclined", "inclined_x"]:
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            self.field[0] = h1 + (h2 - h1) / (Lx) * xx

        elif self.geometry["type"] == "inclined_y":
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            self.field[0] = h1 + (h2 - h1) / (Ly) * yy

        elif self.geometry["type"] == "inclined_pocket":
            h1 = self.geometry['h1']
            h2 = self.geometry['h2']
            hp = self.geometry['hp']
            cp = self.geometry['c']
            lp = self.geometry['l']
            wp = self.geometry['w']
            tw = self.geometry['t']

            self.field[0] = h1 + (h2 - h1) / (Lx) * xx

            xmask = np.logical_and(xx > cp, xx <= cp + lp)
            ymask = np.logical_and(yy > (Ly - wp) / 2., yy <= (Ly + wp) / 2.)
            xymask = np.logical_and(xmask, ymask)

            xt1_mask = np.logical_and(xx > cp - tw, xx <= cp)
            xt2_mask = np.logical_and(xx > cp + lp, xx <= cp + lp + tw)

            self.field[0, xt1_mask] += (xx[xt1_mask] - cp + tw) * hp / tw
            self.field[0, xt2_mask] += -(xx[xt2_mask] - cp - lp - tw) * hp / tw

            self.field[0, xymask] += hp

        elif self.geometry["type"] == "half_sine":
            h0 = self.geometry['h0']
            amp = self.geometry['amp']
            num = self.geometry['num']

            self.field[0] = h0 - amp * np.sin(- 4 * np.pi * (xx - Lx / 2) * num / Lx)
            mask = np.greater(xx, Lx / 2)
            self.field[0][mask] = h0

        elif self.geometry["type"] == "half_sine_squared":
            h0 = self.geometry['h0']
            amp = self.geometry['amp']
            num = self.geometry['num']

            self.field[0] = h0 + amp * np.sin(- 4 * np.pi * (xx - Lx / 2) * num / Lx)**2
            mask = np.greater(xx, Lx / 2)
            self.field[0][mask] = h0

        elif self.geometry["type"] == "cdc":
            h0 = self.geometry['h0']
            h1 = self.geometry['h1']
            b = self.geometry['b']

            slope = (h1 - h0) / (Lx / 2 - 2 * b)

            conv = np.logical_and(xx >= b, xx < Lx / 2 - b)
            center = np.logical_and(xx >= Lx / 2 - b, xx < Lx / 2 + b)
            div = np.logical_and(xx >= Lx / 2 + b, xx < Lx - b)

            h = np.ones_like(xx) * h1
            h[conv] = h1 - slope * (xx[conv] - b)
            h[center] = h0
            h[div] = h0 + slope * (xx[div] - (Lx / 2 + b))

            self.field[0] = h

        elif self.geometry["type"] == "asperity":
            h0 = self.geometry['hmin']
            h1 = self.geometry['hmax']

            bumps = self.geometry['nperside']  # per side

            if bumps == 1:
                hmins = np.array([h0])
            else:
                # Gaussian 99% between hmin and hmax
                std = (h1 - h0) / 2. / 2.57
                hmins = np.random.normal(loc=h0 + (h1 - h0) / 2., scale=std, size=bumps**2)

            xid = (xx // (Lx / bumps)).astype(int)
            yid = (yy // (Ly / bumps)).astype(int)

            masks = []
            for i in range(bumps):
                for j in range(bumps):
                    masks.append(np.logical_and(xid == i, yid == j))

            bx = np.pi / (Lx / bumps)
            by = np.pi / (Ly / bumps)

            zz = np.ones_like(xx) * h1

            for m, h0 in zip(masks, hmins):
                cx = np.mean(xx[m])
                cy = np.mean(yy[m])
                zz[m] -= (h1 - h0) * (np.cos(bx * (xx[m] - cx)) * np.cos(by * (yy[m] - cy)))

            self.field[0] = zz

    def add_roughness_from_file(self):

        Nx = self.disc["Nx"]
        Ny = self.disc["Ny"]

        # periodic cartesian communicator
        pcomm = self.get_2d_cart_comm(MPI.COMM_WORLD, (Nx, Ny),
                                      periods=(True, True))

        topo = np.load(self.roughness["file"])

        if self.min_height < abs(np.amin(topo)):
            if pcomm.Get_rank() == 0:
                print("Rough topography not compatible with geometry. Abort!")
                abort()

        if topo.shape != (Nx, Ny):
            if pcomm.Get_rank() == 0:
                print(f"Shape mismatch. Expected ({Nx}, {Ny}), but imported topography has {topo.shape}. Abort!")
                abort()

        topo_periodic = self.roughness_pbc(topo, pcomm)

        self.field[0] += topo_periodic

    def add_roughness(self):
        """
        Add periodic, self-affine random roughness profile on top of analytical profiles.
        """

        Nx = self.disc["Nx"]
        Ny = self.disc["Ny"]
        Lx = self.disc["Lx"]
        Ly = self.disc["Ly"]

        # periodic cartesian communicator
        pcomm = self.get_2d_cart_comm(MPI.COMM_WORLD, (Nx, Ny),
                                      periods=(True, True))

        topo = np.ones((Nx, Ny), dtype=np.float64) * 1e8
        repeat = 0
        repeat_limit = 100

        while abs(np.amin(topo)) > self.min_height:
            if pcomm.Get_rank() == 0:
                if repeat == 0:
                    print("Generating random roughness ...")
                    seed = self.roughness["seed"]
                else:
                    seed = None

                np.random.seed(seed)

                hurst = self.roughness["Hurst"]
                rolloff = self.roughness["rolloff"]
                rms_height = self.roughness["rmsHeight"]
                rms_slope = self.roughness["rmsSlope"]
                short_cutoff = self.roughness["shortCutoff"]
                long_cutoff = self.roughness["longCutoff"]

                topo = fourier_synthesis((Nx, Ny), (Lx, Ly), hurst,
                                         rms_height=rms_height,
                                         rms_slope=rms_slope,
                                         short_cutoff=short_cutoff,
                                         long_cutoff=long_cutoff,
                                         rolloff=rolloff)
                repeat += 1

                if repeat == repeat_limit:
                    print("Could not generate valid topography. Abort!")
                    abort()

            pcomm.Bcast(topo)

        else:
            if pcomm.Get_rank() == 0:
                if repeat == 1:
                    print("Roughness generation successful!")
                else:
                    print(f"Roughness generation successful (tried {repeat} seeds)!")
                print(60 * "-")

        topo_periodic = self.roughness_pbc(topo, pcomm)

        # add roughness
        self.field[0] += topo_periodic

    def roughness_pbc(self, topo, pcomm):

        ng = self.disc["nghost"]
        ngt = 2 * ng
        topo_periodic = np.empty_like(self.field[0])
        topo_periodic[ng:-ng, ng:-ng] = topo[self.without_ghost]

        ls, ld = pcomm.Shift(0, -1)
        rs, rd = pcomm.Shift(0, 1)
        bs, bd = pcomm.Shift(1, -1)
        ts, td = pcomm.Shift(1, 1)

        # Send to left, receive from right
        recvbuf = np.ascontiguousarray(topo_periodic[-ng:, :])
        pcomm.Sendrecv(np.ascontiguousarray(topo_periodic[ng:ngt, :]), ld, recvbuf=recvbuf, source=ls)
        # fill ghost buffer
        topo_periodic[-ng:, :] = recvbuf

        # Send to right, receive from left
        recvbuf = np.ascontiguousarray(topo_periodic[:ng, :])
        pcomm.Sendrecv(np.ascontiguousarray(topo_periodic[-ngt:-ng, :]), rd, recvbuf=recvbuf, source=rs)
        topo_periodic[:ng, :] = recvbuf

        # Send to bottom, receive from top
        recvbuf = np.ascontiguousarray(topo_periodic[:, -ng:])
        pcomm.Sendrecv(np.ascontiguousarray(topo_periodic[:, ng:ngt]), bd, recvbuf=recvbuf, source=bs)
        # fill ghost buffer
        topo_periodic[:, -ng:] = recvbuf

        # Send to top, receive from bottom
        recvbuf = np.ascontiguousarray(topo_periodic[:, :ng])
        pcomm.Sendrecv(np.ascontiguousarray(topo_periodic[:, -ngt:-ng]), td, recvbuf=recvbuf, source=ts)
        # fill ghost buffer
        topo_periodic[:, :ng] = recvbuf

        return topo_periodic

    def set_gradients(self):
        "gradients for a scalar field (1st entry), stored in 2nd (dx) and 3rd (dy) entry of vectorField"

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        self.field[1:] = np.gradient(self.field[0], dx, dy, edge_order=2)


def fourier_synthesis(shape, size, Hurst, rms_height=None, rms_slope=None,
                      short_cutoff=None, long_cutoff=None, rolloff=1.0):
    """
    Create a self-affine, randomly rough surface using a Fourier filtering
    algorithm. Adapted from https://github.com/ContactEngineering/SurfaceTopography.
    The algorithm is described in:
    Ramisetti et al., J. Phys.: Condens. Matter 23, 215004 (2011);
    Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    Parameters
    ----------
    shape : tuple of ints
        Number of grid cells in x and y direction
    size : tuple of floats
        Physical size in x and y direction
    Hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope.
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.
    rolloff : float
        Value for the power-spectral density (PSD) below the long-wavelength
        cutoff. This multiplies the value at the cutoff, i.e. unit will give a
        PSD that is flat below the cutoff, zero will give a PSD that is
        vanishes below cutoff. (Default: 1.0)

    Returns
    -------
    np.array
        Rough topography.
    """

    Nx, Ny = shape
    Lx, Ly = size

    transpose = False

    if Nx == 1 and Ny > 1:
        grid_pts = (Ny,)
        physical_sizes = (Ly,)
    elif Ny == 1 and Nx > 1:
        grid_pts = (Nx,)
        physical_sizes = (Lx,)
        transpose = True
    else:
        grid_pts = (Nx, Ny)
        physical_sizes = (Lx, Ly)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(grid_pts) / np.asarray(physical_sizes))

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = None

    fac = self_affine_prefactor(grid_pts, physical_sizes, Hurst, rms_height=rms_height,
                                rms_slope=rms_slope, short_cutoff=short_cutoff,
                                long_cutoff=long_cutoff)

    if len(grid_pts) == 2:
        nx, ny = grid_pts
        sx, sy = physical_sizes
        kny = ny // 2 + 1
        kshape = (nx, kny)
    else:
        nx = 1
        ny, = grid_pts
        sx = 1
        sy, = physical_sizes
        kny = ny // 2 + 1
        kshape = (kny,)

    # Initialize arrays
    rarr = np.empty((nx, ny), dtype=np.float64)
    karr = np.empty(kshape, dtype=np.complex128)

    # Creating Fourier representation
    qy = 2 * np.pi * np.arange(kny) / sy
    for x in range(nx):
        if x > nx // 2:
            qx = 2 * np.pi * (nx - x) / sx
        else:
            qx = 2 * np.pi * x / sx
        q_sq = qx**2 + qy**2
        if x == 0:
            q_sq[0] = 1.
        phase = np.exp(2 * np.pi * np.random.rand(kny) * 1j)
        ran = fac * phase * np.random.normal(size=kny)  # amplitudes are normal distributed

        if len(kshape) == 2:
            karr[x, :] = ran * q_sq ** (-(1 + Hurst) / 2)
            karr[x, q_sq > q_max ** 2] = 0.
        else:
            karr[:] = ran * q_sq ** (-(0.5 + Hurst) / 2)
            karr[q_sq > q_max ** 2] = 0.

        if q_min is not None:
            mask = q_sq < q_min**2
            if len(kshape) == 2:
                karr[x, mask] = rolloff * ran[mask] * q_min ** (-(1 + Hurst))
            else:
                karr[mask] = rolloff * ran[mask] * q_min ** (-(0.5 + Hurst))

    # Inverse FFT
    if len(kshape) == 2:
        for iy in [0, -1] if ny % 2 == 0 else [0]:
            # Enforce symmetry
            if nx % 2 == 0:
                karr[0, iy] = np.real(karr[0, iy])
                karr[nx // 2, iy] = np.real(karr[nx // 2, iy])
                karr[1:nx // 2, iy] = karr[-1:nx // 2:-1, iy].conj()
            else:
                karr[0, iy] = np.real(karr[0, iy])
                karr[1:nx // 2 + 1, iy] = karr[-1:nx // 2:-1, iy].conj()
        _irfft2(karr, rarr)
    else:
        karr[0] = np.real(karr[0])
        rarr[:] = np.fft.irfft(karr, n=ny)

    # Shift to zero mean
    rarr -= np.mean(rarr)

    if transpose:
        return rarr.T
    else:
        return rarr


def self_affine_prefactor(grid_pts, physical_sizes, Hurst, rms_height=None,
                          rms_slope=None, short_cutoff=None, long_cutoff=None):
    r"""
    Compute prefactor :math:`C_0` for the power-spectrum density of an ideal
    self-affine topography given by

    .. math ::

        C(q) = C_0 q^{-2-2H}

    for two-dimensional topography maps and

    .. math ::

        C(q) = C_0 q^{-1-2H}

    for one-dimensional line scans. Here :math:`H` is the Hurst exponent.

    Note:
    In the 2D case:

    .. math ::

        h^2_{rms} = \frac{1}{2 \pi} \int_{0}^{\infty} q C^{iso}(q) dq

    whereas in the 1D case:

    .. math ::

        h^2_{rms} = \frac{1}{\pi} \int_{0}^{\infty} C^{1D}(q) dq

    See Equations (1) and (4) in [1].


    Parameters
    ----------
    grid_pts : array_like
        Resolution of the topography map or the line scan.
    physical_sizes : array_like
        Physical physical_sizes of the topography map or the line scan.
    Hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope of the topography map or the line scan.
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.

    Returns
    -------
    prefactor : float
        Prefactor :math:`\sqrt{C_0}`

    References
    -----------
    [1]: Jacobs, Junge, Pastewka,
    Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    """

    grid_pts = np.asarray(grid_pts)
    physical_sizes = np.asarray(physical_sizes)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(grid_pts / physical_sizes)

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = 2 * np.pi * np.max(1 / physical_sizes)

    area = np.prod(physical_sizes)

    if rms_height is not None:
        # Assuming no rolloff region
        fac = 2 * rms_height / np.sqrt(q_min**(-2 * Hurst) -
                                       q_max**(-2 * Hurst)) * np.sqrt(Hurst * np.pi)
    elif rms_slope is not None:
        fac = 2 * rms_slope / np.sqrt(q_max**(2 - 2 * Hurst) -
                                      q_min**(2 - 2 * Hurst)) * np.sqrt((1 - Hurst) * np.pi)
    else:
        # This is already caught in sanity checks
        raise ValueError('Neither rms height nor rms slope is defined!')

    return fac * np.prod(grid_pts) / np.sqrt(area)


def _irfft2(karr, rarr):
    """
    Inverse 2d real-to-complex FFT.

    Parameters
    ----------
    karr : array_like
        Fourier-space representation
    rarr : array_like
        Real-space representation
    """
    nrows, ncolumns = karr.shape
    for i in range(ncolumns):
        karr[:, i] = np.fft.ifft(karr[:, i])
    for i in range(nrows):
        if rarr.shape[1] % 2 == 0:
            rarr[i, :] = np.fft.irfft(karr[i, :])
        else:
            rarr[i, :] = np.fft.irfft(karr[i, :], n=rarr.shape[1])


class SlipLength(ScalarField):

    def __init__(self, disc, surface):

        super().__init__(disc)

        if surface is not None:

            self.surface = surface
            self.disc = disc

            self.fill()

    def fill(self):

        Lx = self.disc["Lx"]
        Ly = self.disc["Ly"]
        Nx = self.disc["Nx"]
        Ny = self.disc["Ny"]
        dx = self.disc["dx"]
        dy = self.disc["dy"]

        smooth = int(self.surface.get('smooth', 0))
        if smooth:
            window1d = np.abs(signal.windows.hann(smooth))  # was 20 hardcoded in prev version
            window1d /= window1d.sum()
            window2d = np.outer(window1d, window1d)

        idxx, idyy = self.id_grid

        ng = self.disc["nghost"]

        xx = idxx * (Lx + 2 * ng * dx) / (Nx + 2 * ng) + dx / 2
        yy = idyy * (Ly + 2 * ng * dy) / (Ny + 2 * ng) + dy / 2

        if self.surface["type"] in ["stripes", "stripes_x"]:
            num = self.surface["num"]
            sign = self.surface["sign"]
            sin = sign * np.sin(2 * np.pi * xx * num / Lx)
            mask = np.greater(sin, 0)

        elif self.surface["type"] in ["stripes_y"]:
            num = self.surface["num"]
            sign = self.surface["sign"]
            sin = sign * np.sin(2 * np.pi * yy * num / Ly)
            mask = np.greater(sin, 0)

        elif self.surface["type"] == "checkerboard":
            right_x = np.greater(xx, Lx / 2)
            center_y = np.logical_and(np.greater(yy, Ly / 4), np.less(yy, 3 * Ly / 4))
            mask_1 = np.logical_and(right_x, center_y)
            mask_2 = np.logical_and(np.logical_not(right_x), np.logical_not(center_y))
            mask = np.logical_or(mask_1, mask_2)

        elif self.surface["type"] == "circle":
            center = (3 * Lx / 4, Ly / 2)
            radius = Lx / 4
            mask = np.less((xx - center[0])**2 + (yy - center[1])**2, radius**2)

        elif self.surface["type"] == "circle2":
            center_1 = (3 * Lx / 4, Ly / 2)
            center_2 = (Lx / 4, Ly)
            center_3 = (Lx / 4, 0)
            radius = Lx / 4
            mask_1 = np.less((xx - center_1[0])**2 + (yy - center_1[1])**2, radius**2)
            mask_2 = np.less((xx - center_2[0])**2 + (yy - center_2[1])**2, radius**2)
            mask_3 = np.less((xx - center_3[0])**2 + (yy - center_3[1])**2, radius**2)

            mask = np.logical_or(np.logical_or(mask_1, mask_2), mask_3)

        elif self.surface["type"] == "square":
            right_x = np.greater(xx, Lx / 2)
            center_y = np.logical_and(np.greater(yy, Ly / 4), np.less(yy, 3 * Ly / 4))
            mask = np.logical_and(right_x, center_y)

        elif self.surface["type"] == "full":
            mask = None

        if smooth:
            mask = signal.convolve2d(mask, window2d,
                                     mode='same',
                                     boundary='wrap' if self.disc['pX'] and self.disc['pY'] else 'symm'
                                     )

        ls0 = self.surface.get("lslip0", 0.)
        ls1 = self.surface["lslip"]

        self.field[0] = ls0 + mask * (ls1 - ls0)
