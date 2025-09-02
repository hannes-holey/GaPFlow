import numpy as np


class GapHeight:

    def __init__(self, fc, disc):

        Lx = disc['Lx']
        Ly = disc['Ly']
        Nx = disc['Nx']
        Ny = disc['Ny']

        periodic_x, periodic_y = True, True

        dx = Lx / Nx
        ix = np.arange(-1, Nx + 1)
        x = ix / Nx * Lx + dx / 2.

        dy = Ly / Ny
        iy = np.arange(-1, Ny + 1)
        y = iy / Ny * Ly + dy / 2.

        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.field = fc.real_field('gap_height', (3,))

        CR = disc['CR']
        eps = disc['eps']

        Rb = Lx / (2 * np.pi)
        c = CR * Rb
        e = eps * c

        self.field.p[0] = c + e * np.cos(xx / Rb)
        self.field.p[1] = -e / Rb * np.sin(xx / Rb)

    @property
    def h(self):
        return self.field.p[0]

    @property
    def dh_dx(self):
        return self.field.p[1]

    @property
    def dh_dy(self):
        return self.field.p[2]
