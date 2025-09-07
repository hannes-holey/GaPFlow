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


class Gap:

    def __init__(self, fc, grid, geo):

        self.field = fc.real_field('gap', (3,))

        xx, yy = create_midpoint_grid(grid)

        if geo['type'] == 'journal':
            h, dh_dx, dh_dy = journal_bearing(xx, grid, geo)

        self.field.p[0] = h
        self.field.p[1] = dh_dx
        self.field.p[2] = dh_dy

    @property
    def h(self):
        return self.field.p[0]

    @property
    def dh_dx(self):
        return self.field.p[1]

    @property
    def dh_dy(self):
        return self.field.p[2]
