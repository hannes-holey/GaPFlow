import numpy as np
from copy import deepcopy
from hans_mugrid.stress import WallStress, BulkStress, Pressure
from hans_mugrid.integrate import predictor_corrector, source


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
