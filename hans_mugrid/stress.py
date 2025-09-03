from hans_mugrid.models import dowson_higginson_pressure, stress_bottom, stress_top, stress_avg


class WallStress:

    def __init__(self, fc, prop):
        self.__field = fc.real_field('wall_stress', (12,))
        self.__gap_height = fc.get_real_field('gap_height')
        self.__solution = fc.get_real_field('solution')

        self.prop = prop

    @property
    def full(self):
        return self.__field.p

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
