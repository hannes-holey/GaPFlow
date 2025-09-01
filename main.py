import numpy as np
import matplotlib.pyplot as plt
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode
from copy import deepcopy

from integrate import predictor_corrector, source
from models import dowson_higginson_pressure, stress_bottom, stress_top, stress_avg


class GapHeight:

    def __init__(self, fc, disc):

        Lx = disc['Lx']
        Ly = disc['Ly']
        Nx = disc['Nx']
        Ny = disc['Ny']

        _x = np.linspace(0., Lx, Nx + 1)
        x = (_x[:-1] + _x[1:]) / 2.

        _y = np.linspace(0., Ly, Ny + 1)
        y = (_y[:-1] + _y[1:]) / 2.

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


class Solution:

    def __init__(self, fc, disc, prop):
        self.disc = disc
        self.__field = fc.real_field('solution', (3,))

        # Constant fields
        self.__gap_height = fc.get_real_field('gap_height')

        # Models
        self.wall_stress = WallStress(fc, prop)
        self.bulk_stress = BulkStress(fc, prop)
        self.pressure = Pressure(fc, prop)

        self.residual = np.inf

    @property
    def q(self):
        return self.__field.p

    @property
    def density(self):
        # centerline
        return self.__field.p[0, :, self.disc['Ny'] // 2]

    @property
    def flux_x(self):
        # centerline
        return self.__field.p[1, :, self.disc['Ny'] // 2]

    @property
    def flux_y(self):
        # centerline
        return self.__field.p[2, :, self.disc['Ny'] // 2]

    @property
    def kinetic_energy(self):
        return np.sum((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0] / 2.)

    def initialize(self, rho0, U, V):
        self.__field.p[0] = rho0
        self.__field.p[1] = rho0 * U / 2.
        self.__field.p[2] = rho0 * V / 2.

        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    def post_update(self):
        self.residual = abs(self.kinetic_energy - self.kinetic_energy_old) / self.kinetic_energy_old
        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    def update(self, switch):

        dx = self.disc["dx"]
        dy = self.disc["dy"]
        dt = self.disc["dt"]

        if switch == 0:
            directions = [-1, 1]
        elif switch == 1:
            directions = [1, -1]

        q0 = self.__field.p.copy()

        for d in directions:

            self.pressure.update()
            self.wall_stress.update()
            self.bulk_stress.update()

            fX, fY = predictor_corrector(self.__field.p,
                                         self.__gap_height.p,
                                         self.pressure.pressure,
                                         self.bulk_stress.stress,
                                         d)

            src = source(self.__field.p,
                         self.__gap_height.p,
                         self.pressure.pressure,
                         self.wall_stress.lower,
                         self.wall_stress.upper)

            self.__field.p = self.__field.p - dt * (fX / dx + fY / dy - src)

        self.__field.p = (self.__field.p + q0) / 2.

        self.post_update()


class WallStress:

    def __init__(self, fc, prop):
        self.__field = fc.real_field('wall_viscous_stress', (12,))
        self.__gap_height = fc.get_real_field('gap_height')
        self.__solution = fc.get_real_field('solution')

        self.prop = prop

    @property
    def upper(self):
        return self.__field.p[6:]

    @property
    def lower(self):
        return self.__field.p[:6]

    def update(self):

        U = 0.1
        V = 0.
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        self.__field.p[:6] = stress_bottom(self.__solution.p,
                                           self.__gap_height.p,
                                           U, V, eta, zeta, 0.)

        self.__field.p[6:] = stress_top(self.__solution.p,
                                        self.__gap_height.p,
                                        U, V, eta, zeta, 0.)


class BulkStress:

    def __init__(self, fc, prop):
        self.__field = fc.real_field('bulk_viscous_stress', (3,))
        self.__gap_height = fc.get_real_field('gap_height')
        self.__solution = fc.get_real_field('solution')

        self.prop = prop

    @property
    def stress(self):
        return self.__field.p

    def update(self):

        U = 0.1
        V = 0.
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        self.__field.p = stress_avg(self.__solution.p,
                                    self.__gap_height.p,
                                    U, V, eta, zeta, 0.)


class Pressure:

    def __init__(self, fc, prop):
        self.__field = fc.real_field('pressure')
        self.__solution = fc.get_real_field('solution')

        self.prop = prop

    @property
    def pressure(self):
        return self.__field.p

    def update(self):

        rho0 = self.prop['rho0']
        p0 = self.prop['P0']
        C1 = self.prop['C1']
        C2 = self.prop['C2']
        self.__field.p = dowson_higginson_pressure(self.__solution.p[0],
                                                   rho0, p0, C1, C2)


if __name__ == "__main__":

    prop = {'shear': 0.0794,
            'bulk': 0.,
            'P0': 101325,
            'rho0': 877.7007,
            'T0': 323.15,
            'C1': 3.5e10,
            'C2': 1.23}

    disc = {'Lx': 1.e-3,
            'Ly': 1.,
            'Nx': 100,
            'Ny': 1,
            'CR': 1.e-2,
            'eps': 0.7
            }

    disc['dx'] = disc['Lx'] / disc['Nx']
    disc['dy'] = disc['Ly'] / disc['Ny']
    disc['dt'] = 3e-10

    # Two dimensional grid
    nb_grid_pts = (disc['Nx'], disc['Ny'])
    fc = GlobalFieldCollection(nb_grid_pts)

    # intialize all fields
    geometry = GapHeight(fc, disc)
    solution = Solution(fc, disc, prop)

    solution.initialize(prop['rho0'], 0.1, 0.)

    file = FileIONetCDF('example.nc', OpenMode.Overwrite)
    file.register_field_collection(fc, field_names=['solution', 'pressure'])

    Nt = 2000

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for i in range(Nt):
        solution.update(0)

        if i % 100 == 0:

            print(f"{i:>5d} {solution.residual:.3e}")

            color_q = plt.cm.Blues(i / Nt)
            ax[0, 0].plot(solution.density, color=color_q)
            ax[0, 1].plot(solution.flux_x, color=color_q)
            ax[0, 2].plot(solution.flux_y, color=color_q)

            color_p = plt.cm.Greens(i / Nt)
            color_t = plt.cm.Oranges(i / Nt)
            ax[1, 0].plot(solution.pressure.pressure, color=color_p)
            ax[1, 1].plot(solution.wall_stress.lower[4, :, disc['Ny'] // 2], color=color_t)
            ax[1, 2].plot(solution.wall_stress.upper[4, :, disc['Ny'] // 2], color=color_t)

            file.append_frame().write()

    plt.show()
