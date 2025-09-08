import os
import numpy as np
import numpy.typing as npt
import pandas as pd
from copy import deepcopy
from datetime import datetime
from typing import Union
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode

from GaPFlow.io import read_yaml_input, write_yaml, create_output_directory
from GaPFlow.stress import WallStress, BulkStress, Pressure
from GaPFlow.integrate import predictor_corrector, source
from GaPFlow.gap import Gap
from GaPFlow.db import Database


class Problem:

    def __init__(self, input_dict):

        options = input_dict['options']
        grid = input_dict['grid']
        prop = input_dict['properties']
        geo = input_dict['geometry']
        numerics = input_dict['numerics']

        # Optional inputs
        if 'gp' in input_dict.keys():
            gp = input_dict['gp']
        else:
            gp = {'shear_gp': False, 'press_gp': False}

        # Intialize database
        if gp['shear_gp'] or gp['press_gp']:
            database = Database(minimum_size=gp['db_init_size'])
        else:
            database = None

        # TODO: check what is needed
        self.grid = grid
        self.numerics = numerics
        self.options = options

        # Initialize field collection
        nb_grid_pts = (grid['Nx'] + 2,
                       grid['Ny'] + 2)
        fc = GlobalFieldCollection(nb_grid_pts)

        # Solution field
        self.__field = fc.real_field('solution', (3,))
        self._initialize(rho0=prop['rho0'], U=geo['U'], V=geo['V'])

        # Intialize gap
        geometry = Gap(fc, grid, geo)
        self.__gap_height = fc.get_real_field('gap')

        # Dependent fields
        self.bulk_stress = BulkStress(fc, prop, geo, data=database)
        self.wall_stress = WallStress(fc, prop, geo,
                                      data=database,
                                      gp=gp['shear'] if gp['shear_gp'] else None)
        self.pressure = Pressure(fc, prop, geo,
                                 data=database,
                                 gp=gp['press'] if gp['press_gp'] else None)

        # I/O
        self.outdir = create_output_directory(options['output'], options['use_tstamp'])

        # Sanitized config file
        write_yaml(input_dict, os.path.join(self.outdir, 'config.yml'))

        # Write gap height and gradients once
        gapfile = FileIONetCDF(os.path.join(self.outdir, 'gap.nc'), OpenMode.Write)
        gapfile.register_field_collection(fc, field_names=['gap'])
        gapfile.append_frame().write()
        gapfile.close()

        # Solution fields
        self.file = FileIONetCDF(os.path.join(self.outdir, 'sol.nc'),
                                 OpenMode.Overwrite)

        field_names = ['solution', 'pressure', 'wall_stress']
        if gp['shear_gp']:
            field_names.append('wall_stress_var')
        if gp['press_gp']:
            field_names.append('pressure_var')
        self.file.register_field_collection(fc, field_names=field_names)

    @classmethod
    def from_yaml(cls, fname):

        input_dict = read_yaml_input(fname)

        # Optional
        # - gp:
        #   - press: tol, noise, freq, wait, max_steps
        #   - shear: ...
        # - db: dtool, location, template, remote, Ninit, QMC sampling
        # - md: ncpu, setup (lammps/moltemplate), temperature, velocity, sampling_time, dump_freq

        return cls(input_dict)

    @classmethod
    def from_problem(cls, config, outfile):
        # TODO: read a sanitized config file (yaml) and a NetCDF file
        # to initialize a new problem from the last frame of an existing one
        raise NotImplementedError

    def run(self):

        # TODO: numerics settings override (to continue a simulation)
        # Numerics
        self.step = 0
        self.residual = np.inf

        self.tol = self.numerics['tol']
        self.dt = self.numerics['dt']
        self.max_it = self.numerics['max_it']

        self.history = {
            "step": [],
            "time": [],
            "ekin": [],
            "residual": []
        }

        tic = datetime.now()
        self.write()

        # Run
        while not self.converged and self.step < self.max_it:
            self.update()

            if self.step % self.options['write_freq'] == 0:
                self.write()

        toc = datetime.now()

        walltime = toc - tic
        speed = self.step / walltime.total_seconds()

        print(33 * '=')
        print(f"Total walltime     : ", str(walltime).split('.')[0])
        print(f"({speed:.2f} steps/s)")
        if self.pressure.is_gp_model:
            print(f" - GP train (press): ", str(self.pressure.cumtime_train).split('.')[0])
            print(f" - GP infer (press): ", str(self.pressure.cumtime_infer).split('.')[0])
        if self.wall_stress.is_gp_model:
            print(f" - GP train (shear): ", str(self.wall_stress.cumtime_train).split('.')[0])
            print(f" - GP infer (shear): ", str(self.wall_stress.cumtime_infer).split('.')[0])
        print(33 * '=')

        self.history_to_csv(os.path.join(self.outdir, 'history.csv'))

    @property
    def q(self) -> npt.NDArray[np.float64]:
        return self.__field.p

    @property
    def density(self) -> npt.NDArray[np.float64]:
        # centerline
        return self.__field.p[0, 1:-1, self.grid['Ny'] // 2]

    @property
    def flux_x(self) -> npt.NDArray[np.float64]:
        # centerline
        return self.__field.p[1, 1:-1, self.grid['Ny'] // 2]

    @property
    def flux_y(self) -> npt.NDArray[np.float64]:
        # centerline
        return self.__field.p[2, 1:-1, self.grid['Ny'] // 2]

    @property
    def kinetic_energy(self) -> float:
        return np.sum((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0] / 2.)

    @property
    def converged(self) -> bool:
        return self.residual < self.tol

    def write(self):
        print(f"{self.step:>5d} {self.residual:.3e}")

        self.history["step"].append(self.step)
        self.history["time"].append(self.step * self.dt)
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

        dx = self.grid["dx"]
        dy = self.grid["dy"]
        dt = self.dt

        q0 = self.__field.p.copy()

        for i, d in enumerate(directions):

            self.pressure.update(predictor=i == 0)
            self.wall_stress.update(predictor=i == 0)
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
        if all(self.grid['bc_xE_P']):
            self.__field.p[:, 0, :] = self.__field.p[:, -2, :].copy()
        else:
            self.__field.p[self.grid['bc_xE_D'], :1, :] = self._get_ghost_cell_values('D', axis=0, direction=-1)
            self.__field.p[self.grid['bc_xE_N'], :1, :] = self._get_ghost_cell_values('N', axis=0, direction=-1)

        # x1
        if np.all(self.grid['bc_xW_P']):
            self.__field.p[:, -1, :] = self.__field.p[:, 1, :].copy()
        else:
            self.__field.p[self.grid['bc_xW_D'], -1:, :] = self._get_ghost_cell_values('D', axis=0, direction=1)
            self.__field.p[self.grid['bc_xW_N'], -1:, :] = self._get_ghost_cell_values('N', axis=0, direction=1)

        # y0
        if np.all(self.grid['bc_yS_P']):
            self.__field.p[:, :, 0] = self.__field.p[:, :, -2].copy()
        else:
            self.__field.p[self.bc['y0'] == 'D', :, :1] = self._get_ghost_cell_values('D', axis=1, direction=-1)
            self.__field.p[self.bc['y0'] == 'N', :, :1] = self._get_ghost_cell_values('N', axis=1, direction=-1)

        # y1
        if np.all(self.grid['bc_yN_P']):
            self.__field.p[:, :, -1] = self.__field.p[:, :, 1].copy()
        else:
            self.__field.p[self.grid['bc_yN_D'], :, -1:] = self._get_ghost_cell_values('D', axis=1, direction=1)
            self.__field.p[self.grid['bc_yN_N'], :, -1:] = self._get_ghost_cell_values('N', axis=1, direction=1)

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
                mask = self.grid[f"bc_xE_{bc_type}"]
                q_target = self.grid["bc_xE_D_val"]
                q_adj = self.__field.p[mask, -(num_ghost + num_ghost):-num_ghost, :]
            else:  # upstream
                mask = self.grid[f"bc_xW_{bc_type}"]
                q_target = self.grid["bc_xW_D_val"]
                q_adj = self.__field.p[mask, num_ghost:num_ghost + num_ghost, :]

        elif axis == 1:  # y
            if direction > 0:  # downstream
                mask = self.grid[f"bc_yS_{bc_type}"]
                q_target = self.grid["bc_yS_D_val"]
                q_adj = self.__field.p[mask, :, -(num_ghost + num_ghost):-num_ghost]
            else:  # upstream
                mask = self.grid[f"bc_yN_{bc_type}"]
                q_target = self.grid["bc_yN_D_val"]
                q_adj = self.__field.p[mask, :, num_ghost:num_ghost + num_ghost]
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
