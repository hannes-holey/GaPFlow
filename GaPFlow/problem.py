import os
import io
import signal
import numpy as np
import numpy.typing as npt
from typing import Self, Type
from copy import deepcopy
from datetime import datetime
from collections import deque
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode

from GaPFlow.io import read_yaml_input, write_yaml, create_output_directory, history_to_csv
from GaPFlow.stress import WallStress, BulkStress, Pressure
from GaPFlow.integrate import predictor_corrector, source
from GaPFlow.topography import Topography
from GaPFlow.db import Database
from GaPFlow.md import Mock, LennardJones, GoldAlkane


class Problem:
    """
    Problem driver for GaPFlow simulations.

    Sets up field collections, constitutive models (pressure, wall stress,
    bulk stress), optional Gaussian-process surrogate databases, time-stepping
    parameters, and I/O.

    Parameters
    ----------
    input_dict : dict
        Configuration dictionary parsed from YAML. Expected keys include
        ``options``, ``grid``, ``properties``, ``geometry``, ``numerics``,
        and optionally ``gp`` and ``db``.
    """

    def __init__(self,
                 outdir: str,
                 options: dict,
                 grid: dict,
                 numerics: dict,
                 prop: dict,
                 geo: dict,
                 gp: dict | None = None,
                 database: Database | None = None
                 ) -> None:

        self.outdir = outdir
        self.options = options
        self.grid = grid
        self.numerics = numerics

        # Initialize field collection
        nb_grid_pts = (self.grid['Nx'] + 2,
                       self.grid['Ny'] + 2)
        fc = GlobalFieldCollection(nb_grid_pts)

        # Solution field
        self.__field = fc.real_field('solution', (3,))
        self._initialize(rho0=prop['rho0'], U=geo['U'], V=geo['V'])

        # Intialize gap
        self.topo = Topography(fc, self.grid, geo)

        # Active GP models
        if gp is not None:
            if self.grid['dim'] == 1:
                gpz = gp if gp['press_gp'] else None
                gpx = gp if gp['shear_gp'] else None
                gpy = None
            elif self.grid['dim'] == 2:
                gpz = gp if gp['press_gp'] else None
                gpx = gp if gp['shear_gp'] else None
                gpy = gp if gp['shear_gp'] else None
        else:
            gpx, gpy, gpz = None, None, None

        # Stress fields
        self.pressure = Pressure(fc, prop, geo, data=database, gp=gpz)
        self.bulk_stress = BulkStress(fc, prop, geo, data=None, gp=None)
        self.wall_stress_xz = WallStress(fc, prop, geo, direction='x', data=database, gp=gpx)
        self.wall_stress_yz = WallStress(fc, prop, geo, direction='y', data=database, gp=gpy)

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

            # Write gap height and gradients once
            topofile = FileIONetCDF(os.path.join(self.outdir, 'topo.nc'), OpenMode.Write)
            topofile.register_field_collection(fc, field_names=['topography'])
            topofile.append_frame().write()
            topofile.close()

            # Solution fields
            self.file = FileIONetCDF(os.path.join(self.outdir, 'sol.nc'),
                                     OpenMode.Overwrite)

            field_names = ['solution', 'pressure', 'wall_stress_xz', 'wall_stress_yz']

            if gpx is not None:
                field_names.append('wall_stress_xz_var')

            if gpy is not None:
                field_names.append('wall_stress_yz_var')

            if gp['press_gp']:
                field_names.append('pressure_var')
            self.file.register_field_collection(fc, field_names=field_names)

    # ---------------------------
    # Constructors
    # ---------------------------

    @classmethod
    def from_yaml(cls: Type[Self], fname: str) -> Self:
        """
        Create a Problem instance from a YAML file.

        Parameters
        ----------
        fname : str
            Path to YAML configuration file.

        Returns
        -------
        Problem
            Instantiated `Problem` object.
        """
        print(f"Reading input file: {fname}")
        with open(fname, "r") as ymlfile:
            input_dict = read_yaml_input(ymlfile)
        return cls(*_split_input(input_dict))

    @classmethod
    def from_string(cls: Type[Self], ymlstring: str) -> Self:
        """
        Create a Problem instance from a YAML string.

        Parameters
        ----------
        ymlstring : str
            YAML content as a string.

        Returns
        -------
        Problem
            Instantiated `Problem` object.
        """
        with io.StringIO(ymlstring) as ymlfile:
            input_dict = read_yaml_input(ymlfile)
        return cls(*_split_input(input_dict))

    @classmethod
    def from_dict(cls: Type[Self], input_dict: dict) -> Self:
        """
        Create a Problem instance from a YAML string.

        Parameters
        ----------
        ymlstring : str
            YAML content as a string.

        Returns
        -------
        Problem
            Instantiated `Problem` object.
        """
        return cls(*_split_input(input_dict))

    # ---------------------------
    # Main run loop
    # ---------------------------

    def run(self) -> None:
        """
        Run the time-stepping loop until convergence, maximum iterations,
        or until a termination signal is received.
        """

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

            _handle_signals(self.receive_signal)

        self.post_run()

    def receive_signal(self, signum, frame) -> None:
        """
        Signal handler: set the `_stop` flag on termination signals.
        """
        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]
        if signum in signals:
            self._stop = True

    def post_run(self) -> None:
        """
        Finalize run: write history, print timing and GP timing info.
        """

        walltime = datetime.now() - self._tic

        if self.step % self.options['write_freq'] != 0 and not self.options['silent']:
            self.write()

        speed = self.step / walltime.total_seconds()

        # Print runtime
        print(33 * '=')
        print("Total walltime   : ", str(walltime).split('.')[0])
        print(f"({speed:.2f} steps/s)")

        if self.pressure.is_gp_model:
            print(" - GP train (zz) : ", str(self.pressure.cumtime_train).split('.')[0])
            print(" - GP infer (zz) : ", str(self.pressure.cumtime_infer).split('.')[0])
        if self.wall_stress_xz.is_gp_model:
            print(" - GP train (xz) : ", str(self.wall_stress_xz.cumtime_train).split('.')[0])
            print(" - GP infer (xz) : ", str(self.wall_stress_xz.cumtime_infer).split('.')[0])
        if self.wall_stress_yz.is_gp_model:
            print(" - GP train (yz) : ", str(self.wall_stress_yz.cumtime_train).split('.')[0])
            print(" - GP infer (yz) : ", str(self.wall_stress_yz.cumtime_infer).split('.')[0])

        print(33 * '=')

        if not self.options['silent']:
            history_to_csv(os.path.join(self.outdir, 'history.csv'), self.history)

            if self.pressure.is_gp_model:
                history_to_csv(os.path.join(self.outdir, 'gp_zz.csv'), self.pressure.history)
                with open(os.path.join(self.outdir, 'gp_zz.txt'), 'w') as f:
                    print(self.pressure.gp, file=f)

            if self.wall_stress_xz.is_gp_model:
                history_to_csv(os.path.join(self.outdir, 'gp_xz.csv'), self.wall_stress_xz.history)
                with open(os.path.join(self.outdir, 'gp_xz.txt'), 'w') as f:
                    print(self.wall_stress_xz.gp, file=f)

            if self.wall_stress_yz.is_gp_model:
                history_to_csv(os.path.join(self.outdir, 'gp_yz.csv'), self.wall_stress_yz.history)
                with open(os.path.join(self.outdir, 'gp_yz.txt'), 'w') as f:
                    print(self.wall_stress_yz.gp, file=f)

    # ---------------------------
    # Convenience properties (field accessors)
    # ---------------------------

    @property
    def q(self) -> npt.NDArray[np.floating]:
        """Full density field"""
        return self.__field.p

    @property
    def centerline_mass_density(self) -> npt.NDArray[np.floating]:
        """Centerline mass density w/o ghost buffers"""
        return self.__field.p[0, 1:-1, self.grid['Ny'] // 2]

    @property
    def mass(self) -> np.floating:
        """Total mass integrated over domain (scalar)."""
        return np.sum(self.__field.p[0] * self.topo.h * self.grid['dx'] * self.grid['dy'])

    @property
    def kinetic_energy(self) -> np.floating:
        """Total kinetic energy (scalar)."""
        return np.sum((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0] / 2.)

    @property
    def v_max(self) -> np.floating:
        """Maximum speed in the domain (scalar)."""
        return np.sqrt((self.__field.p[1]**2 + self.__field.p[2]**2) / self.__field.p[0]).max()

    @property
    def dt_crit(self) -> np.floating:
        """Critical timestep determined by grid spacing and sound speed."""
        return min(self.grid["dx"], self.grid["dy"]) / (self.v_max + self.pressure.v_sound)

    @property
    def cfl(self) -> np.floating:
        """Current CFL number."""
        return self.dt / self.dt_crit

    @property
    def converged(self) -> bool:
        """Return True if residuals in the buffer are below tolerance."""
        return np.all(np.array(self.residual_buffer) < self.tol)

    # ---------------------------
    # I/O and state writing
    # ---------------------------

    def write(self, scalars: bool = True, fields: bool = True, params: bool = True) -> None:
        """
        Write scalars, fields and hyperparameters to disk as configured.
        """
        if scalars:
            print(f"{self.step:<6d} {self.dt:.4e} {self.simtime:.4e} {self.cfl:.4e} {self.residual:.4e}")
            self.history["step"].append(self.step)
            self.history["time"].append(self.simtime)
            self.history["ekin"].append(self.kinetic_energy)
            self.history["residual"].append(self.residual)
            self.history["vsound"].append(self.pressure.v_sound)

        if fields:
            self.file.append_frame().write()

        if params:
            self.pressure.write()
            self.wall_stress_xz.write()
            self.wall_stress_yz.write()

    # ---------------------------
    # Initialization and update helpers
    # ---------------------------
    def _initialize(self, rho0: float, U: float, V: float) -> None:
        """
        Initialize solution field with given base density and mean velocities.
        """
        self.__field.p[0] = rho0
        self.__field.p[1] = rho0 * U / 2.0
        self.__field.p[2] = rho0 * V / 2.0

        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    def post_update(self) -> None:
        """
        Operations executed after each timestep: ghost cell comms, residual
        update, time advance, and adaptive dt update if enabled.
        """
        self._communicate_ghost_buffers()

        self.residual = abs(self.kinetic_energy - self.kinetic_energy_old) / self.kinetic_energy_old / self.cfl
        self.residual_buffer.append(self.residual)
        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

        self.step += 1
        self.simtime += self.dt

        if self.numerics["adaptive"]:
            self.dt = self.numerics["CFL"] * self.dt_crit

    def update(self) -> None:
        """
        Single update iteration performing predictor-corrector for each sweep
        direction and updating constitutive models (pressure, wall/bulk stress).
        """
        switch = (self.step % 2 == 0) * 2 - 1 if self.numerics["MC_order"] == 0 else self.numerics["MC_order"]
        directions = [[-1, 1], [1, -1]][(switch + 1) // 2]

        dx = self.grid["dx"]
        dy = self.grid["dy"]
        dt = self.dt

        q0 = self.__field.p.copy()

        for i, d in enumerate(directions):
            # update surrogates / constitutive models (predictor on first pass)
            self.pressure.update(predictor=i == 0)
            self.wall_stress_xz.update(predictor=i == 0)
            self.wall_stress_yz.update(predictor=i == 0)
            self.bulk_stress.update()

            # fluxes and source terms
            fX, fY = predictor_corrector(
                self.__field.p,
                self.pressure.pressure,
                self.bulk_stress.stress,
                d,
            )

            src = source(
                self.__field.p,
                self.topo.height_and_slopes,
                self.bulk_stress.stress,
                self.wall_stress_xz.lower + self.wall_stress_yz.lower,
                self.wall_stress_xz.upper + self.wall_stress_yz.upper,
            )

            self.__field.p = self.__field.p - dt * (fX / dx + fY / dy - src)

            self._communicate_ghost_buffers()

        # second-order temporal averaging (Crank-Nicolson-like)
        self.__field.p = (self.__field.p + q0) / 2.0

        self.post_update()

    # ---------------------------
    # Ghost cell handling
    # ---------------------------
    def _communicate_ghost_buffers(self) -> None:
        """
        Update ghost-cell values according to boundary conditions stored in
        `self.grid`. This mutates the solution field `self.__field.p`.
        """
        # x0 (left)
        if all(self.grid["bc_xE_P"]):
            self.__field.p[:, 0, :] = self.__field.p[:, -2, :].copy()
        else:
            self.__field.p[self.grid["bc_xE_D"], :1, :] = self._get_ghost_cell_values("D", axis=0, direction=-1)
            self.__field.p[self.grid["bc_xE_N"], :1, :] = self._get_ghost_cell_values("N", axis=0, direction=-1)

        # x1 (right)
        if np.all(self.grid["bc_xW_P"]):
            self.__field.p[:, -1, :] = self.__field.p[:, 1, :].copy()
        else:
            self.__field.p[self.grid["bc_xW_D"], -1:, :] = self._get_ghost_cell_values("D", axis=0, direction=1)
            self.__field.p[self.grid["bc_xW_N"], -1:, :] = self._get_ghost_cell_values("N", axis=0, direction=1)

        # y0 (bottom)
        if np.all(self.grid["bc_yS_P"]):
            self.__field.p[:, :, 0] = self.__field.p[:, :, -2].copy()
        else:
            self.__field.p[self.grid["bc_yS_D"], :, :1] = self._get_ghost_cell_values("D", axis=1, direction=-1)
            self.__field.p[self.grid["bc_yS_N"], :, :1] = self._get_ghost_cell_values("N", axis=1, direction=-1)

        # y1 (top)
        if np.all(self.grid["bc_yN_P"]):
            self.__field.p[:, :, -1] = self.__field.p[:, :, 1].copy()
        else:
            self.__field.p[self.grid["bc_yN_D"], :, -1:] = self._get_ghost_cell_values("D", axis=1, direction=1)
            self.__field.p[self.grid["bc_yN_N"], :, -1:] = self._get_ghost_cell_values("N", axis=1, direction=1)

    def _get_ghost_cell_values(self,
                               bc_type: str,
                               axis: int,
                               direction: int,
                               num_ghost: int = 1) -> npt.NDArray[np.floating]:
        """
        Computes ghost cell values for Dirichlet ('D') or Neumann ('N') boundary
        conditions.

        Parameters
        ----------
        bc_type : str
            'D' for Dirichlet or 'N' for Neumann.
        axis : int
            0 for x-axis, 1 for y-axis.
        direction : int
            Upstream (<0) or downstream (>0) direction.
        num_ghost : int
            Number of ghost cells (<= 2 supported).

        Returns
        -------
        Array
            Ghost cell values extracted/computed for the selected mask.
        """
        assert bc_type in ["D", "N"]

        if axis == 0:  # x-axis
            if direction > 0:  # downstream
                mask = self.grid[f"bc_xE_{bc_type}"]
                q_target = self.grid["bc_xE_D_val"]
                q_adj = self.__field.p[mask, -(num_ghost + num_ghost): -num_ghost, :]
            else:  # upstream
                mask = self.grid[f"bc_xW_{bc_type}"]
                q_target = self.grid["bc_xW_D_val"]
                q_adj = self.__field.p[mask, num_ghost: num_ghost + num_ghost, :]

        elif axis == 1:  # y-axis
            if direction > 0:  # downstream
                mask = self.grid[f"bc_yS_{bc_type}"]
                q_target = self.grid["bc_yS_D_val"]
                q_adj = self.__field.p[mask, :, -(num_ghost + num_ghost): -num_ghost]
            else:  # upstream
                mask = self.grid[f"bc_yN_{bc_type}"]
                q_target = self.grid["bc_yN_D_val"]
                q_adj = self.__field.p[mask, :, num_ghost: num_ghost + num_ghost]
        else:
            raise RuntimeError("axis must be either 0 (x) or 1 (y)")

        a1 = 0.5
        a2 = 0.0
        q1 = q_adj
        q2 = 0.0

        if bc_type == "D":
            Q = (q_target - a1 * q1 + a2 * q2) / (a1 - a2)
        else:
            Q = ((1.0 - a1) * q1 + a2 * q2) / (a1 - a2)

        return Q

# ---------------------------
# Helper functions
# ---------------------------


def _split_input(input_dict):
    # Mandatory inputs
    options = input_dict['options']
    grid = input_dict['grid']
    numerics = input_dict['numerics']
    prop = input_dict['properties']
    geo = input_dict['geometry']

    if not options['silent']:
        outdir = create_output_directory(options['output'], options['use_tstamp'])
        write_yaml(input_dict, os.path.join(outdir, 'config.yml'))
    else:
        outdir = "/tmp/"

    # Optional inputs
    gp = input_dict.get('gp', {'press_gp': False, 'shear_gp': False})
    md = input_dict.get('md', None)
    db = input_dict.get('db', None)

    # Intialize database
    if db is not None:
        if md is None:
            MD = Mock(prop, geo, gp)
        else:
            prop['shear'] = 0.
            prop['bulk'] = 0.

            if md['system'] == 'lj':
                MD = LennardJones(md)
            elif md['system'] == 'mol':
                MD = GoldAlkane(md)

        database = Database(outdir, MD, db)
    else:
        database = None

    return outdir, options, grid, numerics, prop, geo, gp, database


def _handle_signals(func) -> None:
    """
    Register a function as the handler for common termination signals.
    """
    for s in [
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGHUP,
        signal.SIGTERM,
        signal.SIGUSR1,
        signal.SIGUSR2,
    ]:
        signal.signal(s, func)
