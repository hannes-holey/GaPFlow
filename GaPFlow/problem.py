import os
import io
import signal
import numpy as np
import numpy.typing as npt
from copy import deepcopy
from datetime import datetime
from collections import deque
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode

from GaPFlow.io import read_yaml_input, write_yaml, create_output_directory, history_to_csv
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

        if gp['shear_gp'] or gp['press_gp']:
            db = input_dict['db']
        else:
            db = None

        # TODO: check what is needed
        self.grid = grid
        self.numerics = numerics
        self.options = options
        self.prop = prop

        if not self.options['silent']:
            self.outdir = create_output_directory(options['output'], options['use_tstamp'])
        else:
            self.outdir = None

        # Intialize database
        if db is not None:
            database = Database.from_dtool(db, outdir=self.outdir)
        else:
            database = None

        # Initialize field collection
        nb_grid_pts = (grid['Nx'] + 2,
                       grid['Ny'] + 2)
        fc = GlobalFieldCollection(nb_grid_pts)

        # Solution field
        self.__field = fc.real_field('solution', (3,))
        self._initialize(rho0=prop['rho0'], U=geo['U'], V=geo['V'])

        # Intialize gap
        self.gap = Gap(fc, grid, geo)
        self.__gap_height = fc.get_real_field('gap')

        # Dependent fields
        self.bulk_stress = BulkStress(fc, prop, geo, data=database)
        self.wall_stress = WallStress(fc, prop, geo,
                                      data=database,
                                      gp=gp if gp['shear_gp'] else None)
        self.pressure = Pressure(fc, prop, geo,
                                 data=database,
                                 gp=gp if gp['press_gp'] else None)

        # TODO: numerics settings override (to continue a simulation)
        # Numerics
        self.step = 0
        self.simtime = 0.
        self.residual = 1.
        self.residual_buffer = deque([self.residual, ], 5)

        if self.numerics["adaptive"]:
            self.dt = self.numerics["CFL"] * self.dt_crit
        else:
            self.dt = self.numerics['dt']

        self.tol = self.numerics['tol']
        self.max_it = self.numerics['max_it']

        # I/O
        if not self.options['silent']:
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
        print(f'Reading input file: {fname}')
        with open(fname, 'r') as ymlfile:
            input_dict = read_yaml_input(ymlfile)
        return cls(input_dict)

    @classmethod
    def from_string(cls, ymlstring):
        with io.StringIO(ymlstring) as ymlfile:
            input_dict = read_yaml_input(ymlfile)
        return cls(input_dict)

    @classmethod
    def from_problem(cls, config, outfile):
        # TODO: read a sanitized config file (yaml) and a NetCDF file
        # to initialize a new problem from the last frame of an existing one
        raise NotImplementedError

    def run(self):

        self._stop = False

        self.history = {
            "step": [],
            "time": [],
            "ekin": [],
            "residual": [],
            "vsound": []
        }

        if not self.options['silent']:
            print(61 * '-')
            print(f"{"Step":6s} {"Timestep":10s} {'Time':10s} {'CFL':10s} {"Residual":10s}")
            print(61 * '-')
            self.write(params=False)

        # Run
        self._tic = datetime.now()
        while not self.converged and self.step < self.max_it and not self._stop:
            self.update()

            if self.step % self.options['write_freq'] == 0 and not self.options['silent']:
                self.write()

            handle_signals(self.receive_signal)

        self.post_run()

    def receive_signal(self, signum, frame):
        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]
        if signum in signals:
            self._stop = True

    def post_run(self):

        walltime = datetime.now() - self._tic

        if self.step % self.options['write_freq'] != 0 and not self.options['silent']:
            self.write()

        speed = self.step / walltime.total_seconds()

        # Print runtime
        print(33 * '=')
        print("Total walltime     : ", str(walltime).split('.')[0])
        print(f"({speed:.2f} steps/s)")

        if self.pressure.is_gp_model:
            print(" - GP train (press): ", str(self.pressure.cumtime_train).split('.')[0])
            print(" - GP infer (press): ", str(self.pressure.cumtime_infer).split('.')[0])
        if self.wall_stress.is_gp_model:
            print(" - GP train (shear): ", str(self.wall_stress.cumtime_train).split('.')[0])
            print(" - GP infer (shear): ", str(self.wall_stress.cumtime_infer).split('.')[0])
        print(33 * '=')

        if not self.options['silent']:
            history_to_csv(os.path.join(self.outdir, 'history.csv'), self.history)

            if self.pressure.is_gp_model:
                history_to_csv(os.path.join(self.outdir, 'gp_press.csv'), self.pressure.history)
                with open(os.path.join(self.outdir, 'gp_press.txt'), 'w') as f:
                    print(self.pressure.gp, file=f)

            if self.wall_stress.is_gp_model:
                history_to_csv(os.path.join(self.outdir, 'gp_shear.csv'), self.wall_stress.history)
                with open(os.path.join(self.outdir, 'gp_shear.txt'), 'w') as f:
                    print(self.wall_stress.gp, file=f)

    # TODO: use these properties as accessors to fields without ghost cells
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
    def mass(self) -> float:
        return np.sum(self.__field.p[0] * self.__gap_height.p[0] * self.grid['dx'] * self.grid['dy'])

    @property
    def kinetic_energy(self) -> float:
        return np.sum((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0] / 2.)

    @property
    def v_max(self) -> float:
        return np.sqrt((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0]).max()

    @property
    def dt_crit(self) -> float:
        return min(self.grid["dx"], self.grid["dy"]) / (self.v_max + self.pressure.v_sound)

    @property
    def cfl(self) -> float:
        return self.dt / self.dt_crit

    @property
    def converged(self) -> bool:
        return np.all(np.array(self.residual_buffer) < self.tol)

    def write(self, scalars=True, fields=True, params=True):

        # write scalars
        if scalars:
            print(f"{self.step:<6d} {self.dt:.4e} {self.simtime:.4e} {self.cfl:.4e} {self.residual:.4e}")
            self.history["step"].append(self.step)
            self.history["time"].append(self.simtime)
            self.history["ekin"].append(self.kinetic_energy)
            self.history["residual"].append(self.residual)
            self.history["vsound"].append(self.pressure.v_sound)

        # write fields
        if fields:
            self.file.append_frame().write()

        # write hyperparameters
        if params:
            self.pressure.write()
            self.wall_stress.write()

    def _initialize(self, rho0, U, V):
        self.__field.p[0] = rho0
        self.__field.p[1] = rho0 * U / 2.
        self.__field.p[2] = rho0 * V / 2.

        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    def post_update(self) -> None:
        self._communicate_ghost_buffers()

        # last_cfl = np.copy(self.cfl)
        self.residual = abs(self.kinetic_energy - self.kinetic_energy_old) / self.kinetic_energy_old / self.cfl
        self.residual_buffer.append(self.residual)
        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

        self.step += 1
        self.simtime += self.dt

        # New time step
        if self.numerics["adaptive"]:
            self.dt = self.numerics["CFL"] * self.dt_crit

    def update(self) -> None:

        switch = (self.step % 2 == 0) * 2 - 1 if self.numerics['MC_order'] == 0 else self.numerics['MC_order']
        directions = [[-1, 1], [1, -1]][(switch + 1) // 2]

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
            self.__field.p[self.grid['bc_yS_D'] == 'D', :, :1] = self._get_ghost_cell_values('D', axis=1, direction=-1)
            self.__field.p[self.grid['bc_yS_D'] == 'N', :, :1] = self._get_ghost_cell_values('N', axis=1, direction=-1)

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


def handle_signals(func):
    for s in [signal.SIGHUP,
              signal.SIGINT,
              signal.SIGHUP,
              signal.SIGTERM,
              signal.SIGUSR1,
              signal.SIGUSR2]:
        signal.signal(s, func)
