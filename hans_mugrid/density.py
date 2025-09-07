import numpy as np
from copy import deepcopy
from muGrid import FileIONetCDF, OpenMode
from hans_mugrid.stress import WallStress, BulkStress, Pressure
from hans_mugrid.integrate import predictor_corrector, source
import pandas as pd

import numpy.typing as npt
from typing import Union


class Solution:

    def __init__(self, fc, disc, prop, bc, data=None):
        self.disc = disc

        for k, v in bc.items():
            if k in ['x0', 'x1', 'y0', 'y1']:
                bc[k] = np.array(v)
        self.bc = bc

        self.__field = fc.real_field('solution', (3,))

        # Constant fields
        self.__gap_height = fc.get_real_field('gap')

        self._initialize(prop['rho0'], 0.1, 0.)

        # Models
        self.wall_stress = WallStress(fc, prop, data=data)
        self.bulk_stress = BulkStress(fc, prop, data=data)
        self.pressure = Pressure(fc, prop, data=data)

        # I/O
        self.file = FileIONetCDF('example.nc', OpenMode.Overwrite)
        self.file.register_field_collection(fc, field_names=['solution',
                                                             'pressure',
                                                             'pressure_var',
                                                             'wall_stress',
                                                             'wall_stress_var'])

        self.step = 0
        self.tol = 1e-7
        self.residual = np.inf

        self.history = {
            "step": [],
            "time": [],
            "ekin": [],
            "residual": []
        }

    # @classmethod
    # def from_yaml(cls):

    #     # read yaml file here
    #     return cls.__init__(...)

    @property
    def q(self) -> npt.NDArray[np.float64]:
        return self.__field.p

    @property
    def density(self) -> npt.NDArray[np.float64]:
        # centerline
        return self.__field.p[0, 1:-1, self.disc['Ny'] // 2]

    @property
    def flux_x(self) -> npt.NDArray[np.float64]:
        # centerline
        return self.__field.p[1, 1:-1, self.disc['Ny'] // 2]

    @property
    def flux_y(self) -> npt.NDArray[np.float64]:
        # centerline
        return self.__field.p[2, 1:-1, self.disc['Ny'] // 2]

    @property
    def kinetic_energy(self) -> float:
        return np.sum((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0] / 2.)

    @property
    def converged(self) -> bool:
        return self.residual < self.tol

    def write(self):
        print(f"{self.step:>5d} {self.residual:.3e}")

        self.history["step"].append(self.step)
        self.history["time"].append(self.step * self.disc["dt"])
        self.history["ekin"].append(self.kinetic_energy)
        self.history["residual"].append(self.residual)

        self.file.append_frame().write()

    def history_to_csv(self, fname):
        df = pd.DataFrame(data=self.history)
        df.to_csv(fname, index=False)

    def _initialize(self, rho0, U, V):
        self.__field.p[0] = rho0
        self.__field.p[1] = rho0 * U / 2.
        self.__field.p[2] = rho0 * V / 2.

        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    def post_update(self) -> None:
        self._communicate_ghost_buffers()
        self.residual = abs(self.kinetic_energy - self.kinetic_energy_old) / self.kinetic_energy_old
        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

        self.step += 1

    def update(self,
               switch: Union[None, bool] = None)-> None:

        if switch is None:
            switch = self.step % 2 == 0

        if switch == 0:
            directions = [-1, 1]
        elif switch == 1:
            directions = [1, -1]

        dx = self.disc["dx"]
        dy = self.disc["dy"]
        dt = self.disc["dt"]

        q0 = self.__field.p.copy()

        for i, d in enumerate(directions):

            self.pressure.update(gp=True, predictor=i == 0)
            self.wall_stress.update(gp=True, predictor=i == 0)
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

    def _communicate_ghost_buffers(self) -> None:

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
