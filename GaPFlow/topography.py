#
# Copyright 2025 Hannes Holey
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

    def __init__(self, fc, grid, geo):

        xx, yy = create_midpoint_grid(grid)

        self.__field = fc.real_field('topography', (3,))
        self._x = fc.real_field('x')
        self._y = fc.real_field('y')

        self._x.p = xx
        self._y.p = yy

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

        self.__field.p[0] = h
        self.__field.p[ix] = dh_dx
        self.__field.p[iy] = dh_dy

    @property
    def height_and_slopes(self):
        return self.__field.p

    @property
    def h(self):
        return self.__field.p[0]

    @property
    def dh_dx(self):
        return self.__field.p[1]

    @property
    def dh_dy(self):
        return self.__field.p[2]
