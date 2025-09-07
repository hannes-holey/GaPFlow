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


def journal_bearing(xx, disc):

    CR = disc['CR']
    eps = disc['eps']
    Lx = disc['Lx']

    Rb = Lx / (2 * np.pi)
    c = CR * Rb
    e = eps * c

    h = c + e * np.cos(xx / Rb)
    dh_dx = -e / Rb * np.sin(xx / Rb)

    return h, dh_dx


class Gap:

    def __init__(self, fc, disc):

        self.field = fc.real_field('gap', (3,))

        xx, yy = create_midpoint_grid(disc)
        h, dh_dx = journal_bearing(xx, disc)

        self.field.p[0] = h
        self.field.p[1] = dh_dx

    @property
    def h(self):
        return self.field.p[0]

    @property
    def dh_dx(self):
        return self.field.p[1]

    @property
    def dh_dy(self):
        return self.field.p[2]
