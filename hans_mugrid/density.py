import numpy as np
from copy import deepcopy
from hans_mugrid.stress import WallStress, BulkStress, Pressure
from hans_mugrid.integrate import predictor_corrector, source


class Solution:

    def __init__(self, fc, disc, prop, bc):
        self.disc = disc

        for k, v in bc.items():
            if k in ['x0', 'x1', 'y0', 'y1']:
                bc[k] = np.array(v)
        self.bc = bc

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
        return self.__field.p[0, 1:-1, self.disc['Ny'] // 2]

    @property
    def flux_x(self):
        # centerline
        return self.__field.p[1, 1:-1, self.disc['Ny'] // 2]

    @property
    def flux_y(self):
        # centerline
        return self.__field.p[2, 1:-1, self.disc['Ny'] // 2]

    @property
    def kinetic_energy(self):
        return np.sum((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0] / 2.)

    def initialize(self, rho0, U, V):
        self.__field.p[0] = rho0
        self.__field.p[1] = rho0 * U / 2.
        self.__field.p[2] = rho0 * V / 2.

        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    def post_update(self):
        self._communicate_ghost_buffers()
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
                         self.bulk_stress.stress,
                         self.wall_stress.lower,
                         self.wall_stress.upper)

            self.__field.p = self.__field.p - dt * (fX / dx + fY / dy - src)

            self._communicate_ghost_buffers()

        self.__field.p = (self.__field.p + q0) / 2.

        self.post_update()

    def _communicate_ghost_buffers(self):

        # x0
        if np.all(self.bc['x0'] == 'P'):
            self.__field.p[:, 0, :] = self.__field.p[:, -2, :].copy()
        else:
            if np.any(self.bc['x0'] == 'D'):
                self.__field.p[self.bc['x0'] == 'D', :1, :] = self._get_ghost_cell_values('D', axis=0, direction=-1)
            if np.any(self.bc['x0'] == 'N'):
                self.__field.p[self.bc['x0'] == 'N', :1, :] = self._get_ghost_cell_values('N', axis=0, direction=-1)

        # x1
        if np.all(self.bc['x1'] == 'P'):
            self.__field.p[:, -1, :] = self.__field.p[:, 1, :].copy()
        else:
            if np.any(self.bc['x1'] == 'D'):
                self.__field.p[self.bc['x1'] == 'D', -1:, :] = self._get_ghost_cell_values('D', axis=0, direction=1)
            if np.any(self.bc['x1'] == 'N'):
                self.__field.p[self.bc['x1'] == 'N', -1:, :] = self._get_ghost_cell_values('N', axis=0, direction=1)

        # y0
        if np.all(self.bc['y0'] == 'P'):
            self.__field.p[:, :, 0] = self.__field.p[:, :, -2].copy()
        else:
            if np.any(self.bc['y0'] == 'D'):
                self.__field.p[self.bc['y0'] == 'D', :, :1] = self._get_ghost_cell_values('D', axis=1, direction=-1)
            if np.any(self.bc['y0'] == 'N'):
                self.__field.p[self.bc['y0'] == 'N', :, :1] = self._get_ghost_cell_values('N', axis=1, direction=-1)

        # y1
        if np.all(self.bc['y0'] == 'P'):
            self.__field.p[:, :, -1] = self.__field.p[:, :, 1].copy()
        else:
            if np.any(self.bc['y1'] == 'D'):
                self.__field.p[self.bc['y1'] == 'D', :, -1:] = self._get_ghost_cell_values('D', axis=1, direction=1)
            if np.any(self.bc['y0'] == 'N'):
                self.__field.p[self.bc['y1'] == 'N', :, -1:] = self._get_ghost_cell_values('N', axis=1, direction=1)

    def _get_ghost_cell_values(self, bc_type, axis, direction, num_ghost=1):
        """Computes the ghost cell values for boundary conditions.

        For Dirichlet BCs, the target value is reached at the interface between 
        the outermost cell within the physical domain and the first ghost cell. 

        Neumann BCs will always be with zero gradient.

        For both type of BCs, two different interpolation schemes are implemented 
        depending on the number of ghost cells (num_ghost<=2).


        Parameters
        ----------
        bc_type : str
            'D' for Dirichlet or 'N' for Neumann
        axis : int
            Axis, either 0 for x or 1 for y axis.
        direction : int
            Upstream (<0) or downstream (>1) direction.
        num_ghost : int
            Number of ghost cells.

        Returns
        -------
        np.ndarray
            Ghost cell values.
        """

        assert bc_type in ["D", "N"]

        if axis == 0:  # x
            if direction > 0:  # downstream
                q_target = self.bc["rhox1"]
                q_adj = self.__field.p[self.bc["x1"] == bc_type, -(num_ghost + num_ghost):-num_ghost, :]
            else:  # upstream
                q_target = self.bc["rhox0"]
                q_adj = self.__field.p[self.bc["x0"] == bc_type, num_ghost:num_ghost + num_ghost, :]

        elif axis == 1:  # y
            if direction > 0:  # downstream
                q_target = self.bc["rhoy1"]
                q_adj = self.__field.p[self.bc["y1"] == bc_type, :, -(num_ghost + num_ghost):-num_ghost]
            else:  # upstream
                q_target = self.bc["rhoy0"]
                q_adj = self.__field.p[self.bc["y0"] == bc_type, :, num_ghost:num_ghost + num_ghost]
        else:
            raise RuntimeError("axis must be either 0 (x) or (y)")

        a1 = 1. / 2.
        a2 = 0.
        q1 = q_adj
        q2 = 0.

        if bc_type == "D":
            Q = (q_target - a1 * q1 + a2 * q2) / (a1 - a2)
        else:
            Q = ((1. - a1) * q1 + a2 * q2) / (a1 - a2)

        return Q
