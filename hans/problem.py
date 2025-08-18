#
# Copyright 2019, 2025 Hannes Holey
#           2019 Andrea Codrignani
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
import signal
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from importlib import metadata
import numpy as np
import shutil

from hans.material import Material
from hans.plottools import adaptiveLimits
from hans.integrate import ConservedField
from hans.tools import handle_signals


class Problem:

    def __init__(self, options, disc, bc, geometry, numerics, material, surface, ic, roughness, gp, md):
        """
        Collects all information about a single problem
        and contains the methods to run a simulation, based on the problem defintiion."

        Parameters
        ----------
        options : dict
            Contains IO options.
        disc : dict
            Contains discretization parameters.
        bc : dict
            Contains boundary condition parameters.
        geometry : dict
            Contains geometry parameters.
        numerics : dict
            Contains numerics parameters.
        material : dict
            Contains material parameters.
        surface : dict
            Contains surface parameters.
        ic : dict
            Contains initial conditions.
        roughness : dict
            Contains roughness parameters.
        """

        self.options = options
        self.disc = disc
        self.bc = bc
        self.geometry = geometry
        self.numerics = numerics
        self.material = material
        self.surface = surface
        self.ic = ic
        self.roughness = roughness
        self.gp = gp
        self.md = md

    def run(self, out_dir="data", out_name=None, plot=False):
        """
        Starts the simulation.

        Parameters
        ----------
        out_dir : str
            Output directory (default: data).
        out_name : str
            NetCDF output filename (default: None)
        plot : bool
            On-the-fly plotting flag (default: False).

        """

        # global write options
        writeInterval = self.options['writeInterval']

        # TODO: should be done in input via defaults
        if "maxT" not in self.numerics.keys():
            self.numerics['maxT'] = np.inf
        if "maxIt" not in self.numerics.keys():
            maxIt = np.inf
            self.numerics["maxIt"] = maxIt
        else:
            maxIt = self.numerics["maxIt"]

        # Initial conditions
        q_init, t_init = self.get_initial_conditions()

        # Initialize solution field
        self.q = ConservedField(self.disc,
                                self.bc,
                                self.geometry,
                                self.material,
                                self.numerics,
                                self.surface,
                                self.roughness,
                                self.gp,
                                self.md,
                                q_init=q_init,
                                t_init=t_init,
                                fallback=self.options['writeRestart'])

        rank = self.q.comm.Get_rank()

        # time stamp of simulation start time
        self.tStart = datetime.now()

        i = 0
        mode = 0
        self._stop = False

        # Header line for screen output
        if rank == 0:
            print(f"{'Step':10s}\t{'Timestep':12s}\t{'Time':12s}\t{'Epsilon':12s}", flush=True)
            self.write_to_stdout(i, mode=mode)

        if plot:
            # on-the-fly plotting
            self.plot(writeInterval)
        else:
            nc = self.init_netcdf(out_dir, out_name, rank)

            while mode <= 0 and i < maxIt:

                # Perform time update
                mode = self.q.update(i)

                # increase time step
                if mode >= 0:
                    i += 1

                # catch signals and execute signal handler
                handle_signals(self.receive_signal)

                if i == maxIt:
                    mode = 3

                if self._stop:
                    mode = 5
                    break

                if mode > 0:
                    break

                if i % writeInterval == 0:
                    self.write_to_netcdf(i, nc, mode=mode)
                    if rank == 0:
                        self.write_to_stdout(i, mode=mode)

            # write last step
            self.write_to_netcdf(i, nc, mode=mode)
            if rank == 0:
                self.write_to_stdout(i, mode=mode)

            # Delete GP objects (and thereby close files)
            if self.gp is not None:
                self.q.wall_stress.del_gp()
                self.q.eos.GP.__del__()

    def get_initial_conditions(self):
        """
        Return the initial field given by last frame of restart file
        or as defined through inputs.

        Returns
        -------
        np.array
            Inital field of conserved variables.
        tuple
            Inital time and timestep
        """

        if self.ic is None:
            # No ICs specified, fill with (rho0, 0, 0)
            q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
            q_init[0] += self.material["rho0"]
            q_init[1] += self.material["rho0"] * self.geometry['U'] / 2.
            q_init[2] += self.material["rho0"] * self.geometry['V'] / 2.
            t_init = (0., self.numerics["dt"])
        elif self.ic["type"] == "restart":
            # ICs read from last frame of restart file
            q_init, t_init = self.read_last_frame()
        else:
            # ICs as specified in config file
            t_init = (0., self.numerics["dt"])
            q_init = np.zeros((3, self.disc["Nx"], self.disc["Ny"]))
            q_init[0] = self.material["rho0"]

            if self.ic["type"] == "perturbation":
                q_init[0, self.disc["Nx"] // 2, self.disc["Ny"] // 2] *= self.ic["factor"]
            elif self.ic["type"] == "longitudinal_wave":
                x = np.linspace(0 + self.disc["dx"] / 2, self.disc["Lx"] - self.disc["dx"] / 2, self.disc["Nx"])
                y = np.linspace(0 + self.disc["dy"] / 2, self.disc["Ly"] - self.disc["dy"] / 2, self.disc["Ny"])
                xx, yy = np.meshgrid(x, y, indexing="ij")
                k = 2. * np.pi / self.disc["Lx"] * self.ic["nwave"]
                q_init[1] += self.ic["amp"] * np.sin(k * xx)
            elif self.ic["type"] == "shear_wave":
                x = np.linspace(0 + self.disc["dx"] / 2, self.disc["Lx"] - self.disc["dx"] / 2, self.disc["Nx"])
                y = np.linspace(0 + self.disc["dy"] / 2, self.disc["Ly"] - self.disc["dy"] / 2, self.disc["Ny"])
                xx, yy = np.meshgrid(x, y, indexing="ij")
                k = 2. * np.pi / self.disc["Lx"] * self.ic["nwave"]
                q_init[2] += self.ic["amp"] * np.sin(k * xx)
            elif self.ic["type"] == "random":
                q_init[0] += np.random.normal(0., self.ic["stdDens"], size=(self.disc["Nx"], self.disc["Ny"]))
                q_init[1] = np.random.normal(0., self.ic["stdFlux"], size=(self.disc["Nx"], self.disc["Ny"]))
                q_init[2] = np.random.normal(0., self.ic["stdFlux"], size=(self.disc["Nx"], self.disc["Ny"]))
            elif self.ic["type"] == "gauss1D":
                x = np.linspace(0 + self.disc["dx"] / 2, self.disc["Lx"] - self.disc["dx"] / 2, self.disc["Nx"])
                y = np.linspace(0 + self.disc["dy"] / 2, self.disc["Ly"] - self.disc["dy"] / 2, self.disc["Ny"])
                xx, yy = np.meshgrid(x, y, indexing="ij")

                # 1D centered Gaussian
                mean = self.ic['mean']
                stdev = self.ic['stdev']
                lc = self.ic['cutoff']

                # region inside cutoff
                inner = np.logical_and(xx > mean - lc, xx < mean + lc)

                p1D = np.exp(-(xx[inner] - mean)**2 / (2 * stdev**2)) / np.sqrt(2 * np.pi * stdev**2)
                p1D -= np.amin(p1D)  # shift
                p1D /= np.amax(p1D)  # rescale

                q_init[0, inner] += p1D

        return q_init, t_init

    def read_last_frame(self):
        """
        Read last frame from restart file and use as initial values for new run.

        Returns
        -------
        np.array
            Solution field at last frame, used as inital field.
        tuple
            Total time and timestep of last frame.
        """

        file = Dataset(self.ic["file"], "r")

        rho = np.array(file.variables['rho'])[-1]
        jx = np.array(file.variables['jx'])[-1]
        jy = np.array(file.variables['jy'])[-1]
        dt = float(file.variables["dt"][-1])
        t = float(file.variables["time"][-1])

        q0 = np.zeros([3] + list(rho.shape))

        q0[0] = rho
        q0[1] = jx
        q0[2] = jy

        return q0, (t, dt)

    def init_netcdf(self, out_dir, out_name, rank):
        """
        Initialize netCDF4 file, create dimensions, variables and metadata.

        Parameters
        ----------
        out_dir : str
            Output directoy.
        out_name : str
            Filename prefix.
        rank : int
            Rank of the MPI communicator
        Returns
        -------
        netCDF4.Dataset
            Initialized dataset.

        """

        if rank == 0:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        if self.ic is None or self.ic["type"] != "restart":
            if rank == 0:
                if out_name is None:
                    # default unique filename with timestamp
                    timestamp = datetime.now().replace(microsecond=0).strftime("%Y-%m-%d_%H%M%S")
                    name = self.options["name"]
                    outfile = f"{timestamp}_{name}.nc"
                else:
                    # custom filename with zero padded number
                    tag = str(len([1 for f in os.listdir(out_dir) if f.startswith(out_name)]) + 1).zfill(4)
                    outfile = f"{out_name}-{tag}.nc"

                self.outpath = os.path.join(out_dir, outfile)
            else:
                self.outpath = None

            self.outpath = self.q.comm.bcast(self.outpath, root=0)

            # initialize NetCDF file
            parallel = False
            if self.q.comm.Get_size() > 1:
                parallel = True

            nc = Dataset(self.outpath, 'w', parallel=parallel, format='NETCDF3_64BIT_OFFSET')
            nc.restarts = 0
            nc.createDimension('x', self.disc["Nx"])
            nc.createDimension('y', self.disc["Ny"])
            nc.createDimension('step', None)

            # create unknown variable buffer as timeseries of 2D fields
            var0 = nc.createVariable('rho', 'f8', ('step', 'x', 'y'))
            var1 = nc.createVariable('jx', 'f8', ('step', 'x', 'y'))
            var2 = nc.createVariable('jy', 'f8', ('step', 'x', 'y'))
            var0.set_collective(True)
            var1.set_collective(True)
            var2.set_collective(True)

            # create scalar variables
            nc.createVariable('time', 'f8', ('step'))
            nc.createVariable('mass', 'f8', ('step'))
            nc.createVariable('vmax', 'f8', ('step'))
            nc.createVariable('vSound', 'f8', ('step'))
            nc.createVariable('dt', 'f8', ('step'))
            nc.createVariable('eps', 'f8', ('step'))
            nc.createVariable('ekin', 'f8', ('step'))

            # write metadata
            nc.setncattr(f"tStart-{nc.restarts}", self.tStart.strftime("%d/%m/%Y %H:%M:%S"))
            nc.setncattr("version", metadata.version('hans'))

            disc = self.disc.copy()
            bc = self.bc.copy()

            categories = {"options": self.options,
                          "disc": disc,
                          "bc": bc,
                          "geometry": self.geometry,
                          "numerics": self.numerics,
                          "material": self.material}

            if self.surface is not None:
                categories["surface"] = self.surface

            if self.ic is not None:
                categories["ic"] = self.ic

            if self.gp is not None:
                categories["gp"] = self.gp

            if self.md is not None:
                categories['md'] = self.md

            if self.roughness is not None:
                roughness_noNone = {k: (v if v is not None else "None") for k, v in self.roughness.items()}
                categories["roughness"] = roughness_noNone

            # reset modified input dictionaries
            bc["x0"] = "".join(bc["x0"])
            bc["x1"] = "".join(bc["x1"])
            bc["y0"] = "".join(bc["y0"])
            bc["y1"] = "".join(bc["y1"])

            del disc["nghost"]
            del disc["pX"]
            del disc["pY"]

            for name, cat in categories.items():
                for key, value in cat.items():
                    nc.setncattr(f"{name}_{key}", value)

            self.write_to_netcdf(0, nc, mode=0)

        else:
            # append to existing netCDF file
            parallel = False
            if self.q.comm.Get_size() > 1:
                parallel = True
            nc = Dataset(self.ic["file"], 'a', parallel=parallel, format='NETCDF3_64BIT_OFFSET')
            self.outpath = os.path.relpath(self.ic["file"])

            backup_file = f"{os.path.splitext(self.ic['file'])[0]}-{nc.restarts}.nc"

            # create backup
            if rank == 0:
                shutil.copy(self.ic["file"], backup_file)

            # increase restart counter
            nc.restarts += 1

            # append modified attributes
            nc.setncattr(f"tStart-{nc.restarts}", self.tStart.strftime("%d/%m/%Y %H:%M:%S"))
            for key, value in self.numerics.items():
                name = f"numerics_{key}-{nc.restarts}"
                nc.setncattr(name, value)

            nc.setncattr(f"ic_type-{nc.restarts}", "restart")
            nc.setncattr(f"ic_file-{nc.restarts}", backup_file)

        return nc

    def write_to_stdout(self, i, mode):
        """
        Write information about the current time step to stdout.

        Parameters
        ----------
        i : int
            Current time step.
        mode : int
            Writing mode (0: normal, 1: converged, 2: max time, 3: execution stopped).

        """
        print(f"# {i:10d}\t{self.q.dt:.6e}\t{self.q.time:.6e}\t{self.q.eps:.6e}", flush=True)

        if mode == 1:
            print(f"\nSolution has converged after {i:d} steps.", flush=True)
        elif mode == 2:
            print(f"\nNo convergence within {i: d} steps. Stopping criterion: \
maximum time {self.numerics['maxT']: .1e} s reached.", flush=True)
        elif mode == 3:
            print(f"\nNo convergence within {i: d} steps. Stopping criterion: \
maximum number of iterations reached.", flush=True)
        elif mode == 4:
            print("Nan detetcted in solution. Execution stopped.", flush=True)
        elif mode == 5:
            print("Execution stopped.", flush=True)

        if mode > 0:
            walltime = datetime.now() - self.tStart
            print(f"Output written to: {self.outpath}", flush=True)
            print(f"Total wall clock time: {str(walltime).split('.')[0]}", end=" ", flush=True)
            print(f"(Performance: {i/walltime.total_seconds(): .2f} steps/s", end=" ", flush=True)
            print(f"on {self.q.comm.dims[0]} x {self.q.comm.dims[1]} MPI grid)", flush=True)

    def write_to_netcdf(self, i, nc, mode):
        """
        Append current solution field to netCDF file.

        Parameters
        ----------
        i : int
            Current time step.
        nc : netCDF4.Dataset
            NetCDF Dataset object.
        mode : int
            Writing mode (0: normal, 1: converged, 2: max time, 3: execution stopped).

        """

        step = nc.variables["rho"].shape[0]
        xrange, yrange = self.q.without_ghost

        nc.variables['rho'][step, xrange, yrange] = self.q.inner[0]
        nc.variables['jx'][step, xrange, yrange] = self.q.inner[1]
        nc.variables['jy'][step, xrange, yrange] = self.q.inner[2]

        nc.variables["time"][step] = self.q.time
        nc.variables["mass"][step] = self.q.mass
        nc.variables["vmax"][step] = self.q.vmax
        nc.variables["vSound"][step] = self.q.vSound
        nc.variables["dt"][step] = self.q.dt
        nc.variables["eps"][step] = self.q.eps
        nc.variables["ekin"][step] = self.q.ekin

        # TODO: scalar quantities from GP

        if bool(self.options['writeRestart']):
            if i % self.options['writeRestart'] == 0:
                with Dataset(f"{self.outpath.rstrip('.nc')}_restart.nc", "w") as dst:
                    # copy global attributes all at once via dictionary
                    dst.setncatts(nc.__dict__)
                    # copy dimensions
                    for name, dimension in nc.dimensions.items():
                        dst.createDimension(
                            name, (len(dimension) if not dimension.isunlimited() else None))
                    # copy all file data except for the excluded
                    for name, variable in nc.variables.items():
                        x = dst.createVariable(name, variable.datatype, variable.dimensions)
                        dst[name][:] = nc[name][:]
                        # copy variable attributes all at once via dictionary
                        dst[name].setncatts(nc[name].__dict__)

        if mode > 0:
            nc.setncattr(f"tEnd-{nc.restarts}", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            nc.close()

    def receive_signal(self, signum, frame):
        """
        Signal handler. Catches signals send to the process and sets write mode to 3 (abort).

        Parameters
        ----------
        signum :
            signal code
        frame :
            Description of parameter `frame`.

        """
        signals = [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1]
        self._stop = signum in signals

    def plot(self, writeInterval):
        """
        Initialize on-the-fly plotting.

        Parameters
        ----------
        writeInterval : int
            Write interval for stdout in plotting mode.

        """

        fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

        Nx = self.disc["Nx"]
        dx = self.disc["dx"]
        x = np.arange(Nx) * dx + dx / 2

        ax[0, 0].plot(x, self.q.centerline_x[1])
        ax[0, 1].plot(x, self.q.centerline_x[2])
        ax[1, 0].plot(x, self.q.centerline_x[0])
        ax[1, 1].plot(x, Material(self.material).eos_pressure(self.q.centerline_x[0]))

        ax[0, 0].set_title(r'$j_x$')
        ax[0, 1].set_title(r'$j_y$')
        ax[1, 0].set_title(r'$\rho$')
        ax[1, 1].set_title(r'$p$')

        ax[1, 0].set_xlabel('distance x (m)')
        ax[1, 1].set_xlabel('distance x (m)')

        def init():
            pass

        _ = animation.FuncAnimation(fig,
                                    self.animate1D,
                                    100000,
                                    fargs=(fig, ax, writeInterval),
                                    interval=1,
                                    init_func=init,
                                    repeat=False)

        plt.show()

    def animate1D(self, i, fig, ax, writeInterval):
        """
        Animator function. Update solution and plots.

        Parameters
        ----------
        i : type
            Current time step.
        fig : matplotlib.figure
            Figure object.
        ax :  np.array
            Array containing the axes of the figure.
        writeInterval : int
            Write interval for stdout in plotting mode.

        """

        self.q.update(i)
        fig.suptitle('time = {:.3g}'.format(self.q.time))

        ax[0, 0].lines[0].set_ydata(self.q.centerline_x[1])
        ax[0, 1].lines[0].set_ydata(self.q.centerline_x[2])
        ax[1, 0].lines[0].set_ydata(self.q.centerline_x[0])
        ax[1, 1].lines[0].set_ydata(Material(self.material).eos_pressure(self.q.centerline_x[0]))

        ax = adaptiveLimits(ax)

        if i % writeInterval == 0 and i > 0:
            print(f"{i:10d}\t{self.q.dt:.6e}\t{self.q.time:.6e}\t{self.q.eps:.6e}", flush=True)
