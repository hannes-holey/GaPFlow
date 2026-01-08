#
# Copyright 2025 Hannes Holey
#           2025 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import io
import signal
import numpy as np
from copy import deepcopy
from collections import deque
from datetime import datetime
from muGrid import FileIONetCDF, OpenMode
from .parallel import DomainDecomposition

from typing import Type
import numpy.typing as npt
try:
    # Py>=3.11
    from typing import Self
except ImportError:
    # Py<=3.10
    from typing_extensions import Self

from . import __version__
from .db import Database
from .topography import Topography
from .io import read_yaml_input, write_yaml, create_output_directory, history_to_csv
from .models import WallStress, BulkStress, Pressure, Energy, Viscosity
from .md import Mock, LennardJones, GoldAlkane
from .viz.plotting import _plot_height_1d_from_field, _plot_height_2d_from_field
from .viz.plotting import _plot_sol_from_field_1d, _plot_sol_from_field_2d
from .viz.animations import animate_1d, animate_2d


class Problem:
    """
    Problem driver for GaPFlow simulations.

    Sets up field collections, constitutive models (pressure, wall stress,
    bulk stress), optional Gaussian-process surrogate databases, time-stepping
    parameters, and I/O.

    Notes
    -----
    Calling the constructor :meth:`__init__` directly expects properly formatted innput dictionaries.
    It is recommended to use the :meth:`from_yaml` or :meth:`from_string` class methods, which automatically
    sanitize the simulation input.

    Examples
    --------
        >>> from GaPFlow import Problem
        >>> myProblem = Problem.from_yaml('my_input_file.yaml')

    """

    def __init__(self,
                 options: dict,
                 grid: dict,
                 numerics: dict,
                 prop: dict,
                 geo: dict,
                 fem_solver: dict,
                 energy_spec: dict,
                 gp: dict | None = None,
                 database: Database | None = None,
                 extra_field: npt.NDArray | None = None
                 ) -> None:
        """Constructor.

        Parameters
        ----------
        options : dict
            general simulation options.
        grid : dict
            Parameters controlling spatial discretization.
        numerics : dict
            Time integration parameters.
        prop : dict
            Material properties.
        geo : dict
            Geometry settings.
        gp : dict or None
            Parameters controlling the GP surrogate models.
        database : GaPFlow.db.Database or None
            A database object, handling the GP training data with an attached MD runner.
        extra_field: numpy.ndarray or None
            An additional field, whose entries can be used as GP features
            (besides the solution itself and the topography).
        """

        if database is not None:
            if not database.has_mock_md:
                prop['shear'] = 0.
                prop['bulk'] = 0.

        self.options = options
        self.grid = grid
        self.numerics = numerics
        self.geo = geo
        self.prop = prop
        self.fem_solver = fem_solver
        self.energy_spec = energy_spec

        # Initialize solver
        if self.numerics['solver'] == 'explicit':
            from .solver_explicit import ExplicitSolver
            self.solver = ExplicitSolver(self)
        elif self.numerics['solver'] == 'fem':
            if self.grid['Ny'] == 1:
                from .solver_fem import FEMSolver1D
                self.solver = FEMSolver1D(self)
            else:
                from .solver_fem_2d import FEMSolver2D
                self.solver = FEMSolver2D(self.fem_solver, self)

        # Initialize domain decomposition and field collection
        self.decomp = DomainDecomposition(grid)
        self.fc = self.decomp.get_fc()

        # Solution field
        self.step = None
        self.__field = self.fc.real_field('solution', (3,))
        self._initialize(rho0=prop['rho0'], U=geo['U'], V=geo['V'])

        # Initialize extra field
        num_extra_features = 1 if database is None else database.num_features - 6
        extra = self.fc.real_field('extra', (num_extra_features,))
        if extra_field is not None:
            extra.pg[:] = extra_field

        # Forward declaration of cross-dependent fields
        self.fc.real_field('x')
        self.fc.real_field('y')
        self.fc.real_field('pressure')
        self.fc.real_field('topography', components=(4,))

        # Initialize stress and topography models
        gpx, gpy, gpz = self._select_gp_config(gp)
        self.pressure = Pressure(self.fc, prop, geo, data=database, gp=gpz)
        self.bulk_stress = BulkStress(self.fc, prop, geo, data=None, gp=None)
        self.wall_stress_xz = WallStress(self.fc, prop, geo, direction='x', data=database, gp=gpx)
        self.wall_stress_yz = WallStress(self.fc, prop, geo, direction='y', data=database, gp=gpy)
        self.viscosity = Viscosity(self.fc, prop)
        self.topo = Topography(self.fc, self.grid, geo, prop, decomp=self.decomp)

        self.bEnergy = (self.numerics['solver'] == 'fem' and self.fem_solver['equations']['energy'])
        if self.bEnergy:
            self.energy = Energy(self.fc, energy_spec, grid)
            self.energy.initialize()

        # I/O
        if not self.options['silent']:

            self.outdir = create_output_directory(options['output'], options['use_tstamp'])

            if database is not None:
                # Set training path inside output path
                # if database.overwrite_training_path:
                database.set_training_path(os.path.join(self.outdir, 'train'),
                                           check_temporary=True)

                database.output_path = self.outdir
                options['output'] = self.outdir

            # Reconstruct dict
            full_dict = {}
            full_dict.update(version=__version__)

            for k, v in zip(['options', 'grid', 'numerics', 'geo', 'prop'],
                            [options, grid, numerics, geo, prop]):
                full_dict[k] = v

            if database is not None:
                full_dict['gp'] = gp
                full_dict['db'] = database.config
                full_dict['md'] = database.md_config

            write_yaml(full_dict, os.path.join(self.outdir, 'config.yml'))

            # Write gap height and gradients
            # No elastic deformation - write once and close
            # Elastic deformation - write initial topo and keep open
            self.topofile = FileIONetCDF(os.path.join(self.outdir, 'topo.nc'), OpenMode.Overwrite)
            self.topofile.register_field_collection(self.fc, field_names=['topography'])
            self.topofile.append_frame().write()
            if not self.prop['elastic']['enabled']:
                self.topofile.close()

            # Solution fields
            self.file = FileIONetCDF(os.path.join(self.outdir, 'sol.nc'),
                                     OpenMode.Overwrite)

            field_names = ['solution', 'pressure', 'wall_stress_xz', 'wall_stress_yz']

            if gpx is not None:
                field_names.append('wall_stress_xz_var')

            if gpy is not None:
                field_names.append('wall_stress_yz_var')

            if gpz:
                field_names.append('pressure_var')

            self.file.register_field_collection(self.fc, field_names=field_names)

            # Energy and temperature fields
            if self.bEnergy:
                self.file.register_field_collection(self.fc, field_names=['total_energy', 'temperature'])

    # ---------------------------
    # Constructors
    # ---------------------------

    @staticmethod
    def _get_mandatory_input(input_dict):

        # Mandatory inputs
        options = input_dict['options']
        grid = input_dict['grid']
        numerics = input_dict['numerics']
        prop = input_dict['properties']
        geo = input_dict['geometry']
        fem_solver = input_dict['fem_solver']
        energy_spec = input_dict['energy_spec']

        return options, grid, numerics, prop, geo, fem_solver, energy_spec

    @staticmethod
    def _get_optional_input(input_dict):

        # Optional inputs
        gp = input_dict.get('gp', None)
        md = input_dict.get('md', None)
        db = input_dict.get('db', None)

        # Intialize database
        if db is not None:
            if md is None:
                prop = input_dict['properties']
                geo = input_dict['geometry']
                MD = Mock(prop, geo, gp)
            else:
                if md['system'] == 'lj':
                    MD = LennardJones(md)
                elif md['system'] == 'mol':
                    MD = GoldAlkane(md)

            database = Database(MD, db)
        else:
            database = None

        return {'gp': gp,
                'database': database,
                'extra_field': None}

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

        return cls._from_dict(input_dict)

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

        return cls._from_dict(input_dict)

    @classmethod
    def _from_dict(cls: Type[Self], input_dict: dict) -> Self:
        """
        Create a Problem instance from a sanitized input dictionary.

        Parameters
        ----------
        input_dict : dict
            Sanitized input dictionary

        Returns
        -------
        Problem
            Instantiated `Problem` object.
        """
        return cls(*cls._get_mandatory_input(input_dict),
                   **cls._get_optional_input(input_dict))

    # ---------------------------
    # Convenience properties (field accessors)
    # ---------------------------

    @property
    def q(self) -> npt.NDArray[np.floating]:
        """Full density field"""
        return self.__field.pg

    @q.setter
    def q(self, sol_field: npt.NDArray[np.floating]) -> None:
        self.__field.pg[:] = sol_field

    @property
    def q_has_nan(self) -> bool:
        """Check for NaNs in the solution field."""
        return np.any(np.isnan(self.q))

    @property
    def q_has_negative_density(self) -> bool:
        """Check for negative densities in the solution field."""
        return np.any(self.q[0] < 0.)

    @property
    def q_is_valid(self) -> bool:
        """Validity flag for the solution field."""
        return ~self.q_has_nan and ~self.q_has_negative_density

    @property
    def mass(self) -> np.floating:
        """Total mass integrated over domain (scalar)."""
        return np.sum(self.__field.pg[0] * self.topo.h * self.grid['dx'] * self.grid['dy'])

    @property
    def kinetic_energy(self) -> np.floating:
        """Total kinetic energy (scalar)."""
        return np.sum((self.__field.pg[1]**2 + self.__field.pg[2]**2) / self.__field.pg[0] / 2.)

    @property
    def v_max(self) -> np.floating:
        """Maximum speed in the domain (scalar)."""
        return np.sqrt((self.__field.pg[1]**2 + self.__field.pg[2]**2) / self.__field.pg[0]).max()

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
        return np.all(np.array(self.residual_buffer) < self.numerics['tol'])

    # ---------------------------
    # Simulation run utilities
    # ---------------------------

    def run(self,
            keep_open: bool = False) -> None:
        """
        Run the time-stepping loop until convergence, maximum iterations,
        or until a termination signal is received.

        Parameters
        ----------
        keep_open : bool, optional
            If True, keeps files open after run for following runs to be
            written in the same files, by default False.
        """
        if self.step is None:
            self._pre_run()

        self._stop = False

        self.history = {
            "step": [],
            "time": [],
            "ekin": [],
            "residual": [],
            "vsound": []
        }

        self.solver.print_status_header()

        # Run
        self._tic = datetime.now()
        while not self.converged and self.step < self.numerics['max_it'] and not self._stop:
            self.update()

            if self.step % self.options['write_freq'] == 0 and not self.options['silent']:
                self.write()

            _handle_signals(self._receive_signal)

        if not keep_open:
            self._post_run()

    def _pre_run(self, **kwargs) -> None:
        """Initialize time-stepping and GP models.

        Has to be called before the first call to :meth:`update`.
        """
        self.pressure.init_database(self.grid['dim'])
        self.wall_stress_xz.init_database(self.grid['dim'])
        self.wall_stress_yz.init_database(self.grid['dim'])

        self.pressure.init()
        self.wall_stress_xz.init()
        self.wall_stress_yz.init()

        if not self.options['silent']:
            self.pressure.write()
            self.wall_stress_xz.write()
            self.wall_stress_yz.write()

        self.step = 0
        self.dt = self.numerics['dt']
        self.simtime = 0.
        self.residual = 1.
        self.residual_buffer = deque([self.residual, ], 5)

        self.solver.pre_run(**kwargs)

    def _receive_signal(self, signum, frame) -> None:
        """
        Signal handler: set the `_stop` flag on termination signals.
        """
        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]
        if signum in signals:
            self._stop = True

    def _post_run(self) -> None:
        """
        Finalize run: write history, print timing and GP timing info.
        """

        walltime = datetime.now() - self._tic

        if self.step % self.options['write_freq'] != 0 and not self.options['silent']:
            self.write()

        if not self.options['silent']:
            self.file.close()  # need to be closed to be readable when animating from problem
            if self.prop['elastic']['enabled']:
                self.topofile.close()

        speed = self.step / walltime.total_seconds()

        # Print runtime (only on rank 0)
        if self.decomp.rank == 0:
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
    # Single time step (update)
    # ---------------------------

    def update(self) -> None:
        """
        Single update iteration delegated to the solver.
        """
        self.solver.update()

    def _post_update(self) -> None:
        """
        Operations executed after each timestep: ghost cell comms, residual
        update, time advance, and adaptive dt update if enabled.
        """
        self._communicate_ghost_buffers()

        E_kin_old = self.kinetic_energy_old
        self.residual = abs(self.kinetic_energy - E_kin_old) / (E_kin_old + 1e-12) / self.cfl
        self.residual_buffer.append(self.residual)
        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

        self.step += 1
        self.simtime += self.dt

        if self.numerics["adaptive"] and self.numerics["solver"] == "explicit":
            self.dt = self.numerics["CFL"] * self.dt_crit

    def _finalize(self, q0: npt.NDArray) -> None:
        """
        Reset the solution field to the one of the ols time step and update stresses.
        Sets the _stop flag to abort the simulation run.

        Parameters
        ----------
        q0 : np.ndarray
            Solution field
        """
        if self.q_has_nan:
            print('NaN detected.', end=' ')
        elif self.q_has_negative_density:
            print('Negative density detected.', end=' ')

        self.q = q0
        self.pressure.update(predictor=False, compute_var=True)
        self.wall_stress_xz.update(predictor=False, compute_var=True)
        self.wall_stress_yz.update(predictor=False, compute_var=True)
        self.bulk_stress.update()

        print('Writing previous step and aborting simulation.')
        self._stop = True

    # ---------------------------
    # I/O and state writing
    # ---------------------------

    def write(self, scalars: bool = True, fields: bool = True, params: bool = True) -> None:
        """
        Write scalars, fields and hyperparameters to disk as configured.
        """
        self.solver.print_status(scalars)

        if fields:
            self.file.append_frame().write()

        if params:
            self.pressure.write()
            self.wall_stress_xz.write()
            self.wall_stress_yz.write()

        if self.prop['elastic']['enabled']:
            self.topofile.append_frame().write()

    # ---------------------------
    # Initialization and update helpers
    # ---------------------------

    def _select_gp_config(self, gp):
        """
        Select active GP models
        """
        if gp is not None:
            if self.grid['dim'] == 1:
                gpz = gp.get('press')
                gpx = gp.get('shear')
                gpy = None
            elif self.grid['dim'] == 2:
                gpz = gp.get('press')
                gpx = gp.get('shear')
                gpy = gp.get('shear')

        else:
            gpx, gpy, gpz = None, None, None

        return gpx, gpy, gpz

    def _initialize(self, rho0: float, U: float, V: float) -> None:
        """
        Initialize solution field with given base density and mean velocities.
        """
        self.__field.pg[0] = rho0
        self.__field.pg[1] = rho0 * U / 2.0
        self.__field.pg[2] = rho0 * V / 2.0

        self.kinetic_energy_old = deepcopy(self.kinetic_energy)

    # ---------------------------
    # Ghost cell handling
    # ---------------------------

    def _communicate_ghost_buffers(self) -> None:
        """
        Update ghost-cell values via MPI communication and boundary conditions.

        Delegates to DomainDecomposition.communicate_ghost_buffers which handles:
        1. MPI ghost exchange between ranks
        2. Physical boundary condition application at domain boundaries
        """
        self.decomp.communicate_ghost_buffers(self)

    # ---------------------------
    # Plotting and animations
    # ---------------------------
    def plot(self, ax=None) -> None:
        """Plot a snaphsot of the solution and the current stress state.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis, optional
            An axis to plot into, if None or wrong shape a new axis is created
        """

        if self.grid['dim'] == 1:
            if ax is not None and ax.shape != (2, 3):
                ax = None

            _plot_sol_from_field_1d(self.q,
                                    self.pressure.pressure,
                                    self.wall_stress_xz.lower[4],
                                    self.wall_stress_xz.upper[4],
                                    var_press=self.pressure.variance
                                    if self.pressure.is_gp_model
                                    else None,
                                    var_shear=self.wall_stress_xz.variance
                                    if self.wall_stress_xz.is_gp_model
                                    else None,
                                    var_tol_press=self.pressure.variance_tol
                                    if self.pressure.is_gp_model and self.pressure.use_active_learning
                                    else None,
                                    var_tol_shear=self.wall_stress_xz.variance_tol
                                    if self.wall_stress_xz.is_gp_model and self.wall_stress_xz.use_active_learning
                                    else None,
                                    energy=[self.energy.energy, self.energy.temperature]
                                    if self.bEnergy
                                    else None,
                                    ax=ax)

        elif self.grid['dim'] == 2:
            if ax is not None and ax.shape != (3, 3):
                ax = None

            # TODO: plots for GPs
            _plot_sol_from_field_2d(self.q,
                                    self.pressure.pressure,
                                    self.wall_stress_xz.lower[4],
                                    self.wall_stress_xz.upper[4],
                                    self.wall_stress_yz.lower[3],
                                    self.wall_stress_yz.upper[3],
                                    var_press=None,
                                    var_shear_xz=None,
                                    var_shear_yz=None,
                                    ax=ax)

    def plot_topo(self,
                  show_defo=False,
                  show_pressure=False) -> None:
        """Plot the gap topography, optionally in deformed state and with pressure profile.

        Parameters
        ----------
        show_defo: bool
            Flag for showing deformation, default is False
        show_pressure: bool
            Flag for showing pressure, default is False
        """

        if self.grid['dim'] == 1:
            _plot_height_1d_from_field(self.topo.full,
                                       self.pressure.pressure,
                                       show_defo=show_defo,
                                       show_pressure=show_pressure)
        elif self.grid['dim'] == 2:
            # TODO: show defo in 2D
            _plot_height_2d_from_field(self.topo.full)

    def animate(self,
                save: bool = False,
                seconds: float = 10.0
                ) -> None:
        """Create an animation of the solution time series.

        Checks if simulation has run already and if output has been generated.
        For 1D elastic simulations, height and deformation are included.

        Parameters
        ----------
        save: bool, optional
            Whether to save the animation as an .mp4 file, by default False.
        seconds: float, optional
            Duration of the animation in seconds(if saved), by default 10.0
        """
        if not getattr(self, "step", 0) > 0:
            raise RuntimeError("Cannot animate before running the simulation.")

        if self.options['silent']:
            raise RuntimeError("Cannot animate in silent mode.")

        filename_sol = os.path.join(self.outdir, 'sol.nc')
        filename_topo = os.path.join(self.outdir, 'topo.nc')

        if self.grid['Ny'] == 1:
            return animate_1d(filename_sol,
                              filename_topo,
                              seconds=seconds,
                              save=save)

        else:
            return animate_2d(filename_sol,
                              seconds=seconds,
                              save=save)

# ---------------------------
# Helper functions
# ---------------------------


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
