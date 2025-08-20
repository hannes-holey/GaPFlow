#
# Copyright 2020, 2025 Hannes Holey
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

import sys
import warnings
import numpy as np
from mpi4py import MPI
from copy import deepcopy
from unittest.mock import Mock
from collections import deque

from hans.field import VectorField
from hans.stress import SymStressField2D, WallStressField3D
from hans.geometry import GapHeight, SlipLength
from hans.material import Material
from hans.special.flux_limiter import TVD_MC_correction
from hans.multiscale.db import Database
from hans.elasticity import ElasticDeformation
from scipy.ndimage import gaussian_filter1d

class ConservedField(VectorField):

    def __init__(self, disc, bc, geometry, material, numerics, surface, roughness, gp, md,
                 q_init=None, t_init=None, options=None, fallback=0):
        """
        This class contains the field of conserved variable densities (rho, jx, jy),
        and the methods to update them. Derived from VectorField.

        Parameters
        ----------
        disc : dict
            Discretization parameters.
        bc : dict
            Boundary condition parameters.
        geometry : dict
            Geometry parameters.
        material : dict
            Material parameters.
        numerics : dict
            numerics parameters
        surface : dict
            surface parameters
        roughness : dict
            roughness parameters
        gp : dict
            Gaussian process parameters
        md : dict
            Molecular dynamics parameters
        q_init : np.array
            Inital field in case of a restart (the default is None)
        t_init : tuple
            Initial time and time step in case of a restart (the default is None).

        """

        super().__init__(disc)

        self.disc = disc
        self.bc = bc
        self.material = material
        self.numerics = numerics
        self.roughness = roughness
        self.surface = surface
        self.gp = gp
        self.geometry = geometry
        self.md = md
        self.options = options

        # self.fallback = fallback
        self.q_init = q_init
        self.num_resets = 0
        self.wait_p_after_reset = 0
        self.wait_tau_after_reset = 0

        self.initialize(q_init, t_init)

    def initialize(self, q_init, t_init, restart=False):

        # initialize gap height and slip length field
        self.height = GapHeight(self.disc, self.geometry, self.roughness)
        self.slip_length = SlipLength(self.disc, self.surface)
        # initialize the three height contributions
        self.height_base = np.copy(self.height.field[0])
        self.u_ext = np.zeros_like(self.height_base)
        self.dh_dt_force = 0.
        self.rigid_displacement = 0.
        self.int_dh_wallforce = 0.

        # intialize field and time
        if q_init is not None:
            self.inner = q_init[:, self.without_ghost[0], self.without_ghost[1]]
            self.fill_ghost_buffer()
            self.time = t_init[0]
            self.dt = t_init[1]
        else:
            self.field[0] = self.material['rho0']
            self.time = 0.
            self.dt = self.numerics["dt"]

        self.eps = np.inf
        self.eps_buffer = deque([self.eps, ], 5)
        self.Ekin_old = deepcopy(self.ekin)

        if not restart:

            # self.q_fallback = deepcopy(self.field)

            if self.gp is not None:
                if self.numerics['integrator'].startswith('MC'):
                    self.gp['_pCalls'] = 2
                    self.gp['_sCalls'] = 2
                elif self.numerics['integrator'] == 'LW':
                    self.gp['_pCalls'] = 4
                    self.gp['_sCalls'] = 2
                elif self.numerics['integrator'] == 'RK3':
                    self.gp['_pCalls'] = 6
                    self.gp['_sCalls'] = 3

            # Avg stress (xx, yy, xy) -- not used in GP runs, is small anyways
            self.viscous_stress = SymStressField2D(self.disc, self.geometry, self.material, self.surface, self.gp)

            # Wall stress (xx, yy, zz, yz, xz, xy)
            self.wall_stress = WallStressField3D(self.disc, self.geometry, self.material, self.surface, self.gp)

            # Equation of state
            self.eos = Material(self.material, self.gp)

            if self.gp is not None:
                # Initalize global training database
                db = Database(self.gp,
                              self.md,
                              {'kappa': self.slip_length.field,
                               'gap_height': self.height.field},
                              self.eos.eos_pressure,  # only w/o lammps
                              self.wall_stress.gp_wall_stress)  # only w/o lammps

                self.wall_stress.init_gp(self.field, db)
                self.eos.init_gp(self.field, db)
            else:
                self.wall_stress.GP_list = []
                self.wall_stress.gp = None

                self.eos.GP = Mock()
                self.eos.GP.reset = False
                self.eos.gp = None

        if bool(self.options['gradientAnalysis']):
            self.initialize_gradientAnalysis()

        if bool(self.material['elastic']):
            self.initialize_elasticDeform()

        if self.material['wallmode'] == 'force':
            self.initialize_wallforce()

        if bool(self.options['pressure']):
            self.initialize_pressure()

    def initialize_gradientAnalysis(self):
        """initialize variables for gradient analysis
        """
        # gradient analysis variables
        self.ga_rho_t = np.zeros_like(self.inner[0])
        self.ga_jx_x = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_h_t_rho = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_h_x_jx = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]

        self.ga_jx_t = np.zeros_like(self.inner[0])
        self.ga_p_x = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_uxjx_x = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_tauxx_x = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_h_x_uxjx = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_h_x_tauxx = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]

        self.ga_tauxz = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]
        self.ga_h_t_jx = [np.zeros_like(self.inner[0]), np.zeros_like(self.inner[0])]

        self.ga_mass = np.sum(self.inner[0] * self.height.inner[0] * self.disc['dx'] * self.disc['dy'])
        self.ga_ux = self.inner[1] / self.inner[0]
        self.ga_ux_at_hmin = np.zeros_like(self.inner[0])

    def initialize_elasticDeform(self):
        """initialize variables for elastic deformation
        """
        self.elasticDeform = ElasticDeformation(size=(self.disc['Nx'], self.disc['Ny']),
                                                HANS_slice=self.global_slice,
                                                dx=self.disc['dx'], dy=self.disc['dy'],
                                                E=self.material['E'], v=self.material['v'],
                                                perX=self.disc['pX'], perY=self.disc['pY'],
                                                bc=self.bc, comm=self.comm, p_ext=float(self.material['p_ext']))

        self.u = np.zeros_like(self.inner[0])
        self.g_u = np.zeros_like(self.inner[0])
        self.u_prev = np.zeros_like(self.field[0])
        self.du_dt = np.zeros_like(self.field[0])
        self.du_dt_prev = np.zeros_like(self.field[0])

    def initialize_wallforce(self):
        """initialize variables for wallforce
        """
        # TODO: self.material['P0'] not specified for all EoS
        self.delta_p_wallforce = float(self.material['p_ext']) - float(self.material['P0'])
        self.d2h_dt2_wallforce = 0.
        self.dh_dt_wallforce = 0.

    def initialize_pressure(self):
        """initialize variables for pressure
        """
        self.p = Material(self.material).eos_pressure(self.field[0])
        self.p = self.p[1:-1, 1:-1]
        self.p_prev = np.copy(self.p)
        self.p_last_step = np.copy(self.p)

    @property
    def mass(self):
        area = self.disc["dx"] * self.disc["dy"]
        local_mass = np.sum(self.inner[0] * self.height.inner[0] * area)
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_mass, recvbuf, op=MPI.SUM)

        return recvbuf[0]

    @property
    def vSound(self):
        local_vSound = self.eos.get_sound_speed(self.inner)
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_vSound, recvbuf, op=MPI.MAX)

        return recvbuf[0]

    @property
    def vmax(self):
        local_vmax = np.amax(np.sqrt(self.inner[1]**2 + self.inner[2]**2) / self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_vmax, recvbuf, op=MPI.MAX)

        return recvbuf[0]

    @property
    def vx_max(self):
        local_vx_max = np.amax(self.inner[1] / self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_vx_max, recvbuf, op=MPI.MAX)

        return recvbuf[0]

    @property
    def vy_max(self):
        local_vy_max = np.amax(self.inner[2] / self.inner[0])
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_vy_max, recvbuf, op=MPI.MAX)

        return recvbuf[0]

    @property
    def ekin(self):
        local_ekin = np.sum((self.inner[1]**2 + self.inner[2]**2) / self.inner[0]**2)
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_ekin, recvbuf, op=MPI.SUM)

        return recvbuf[0] * self.mass / 2.

    @property
    def isnan(self):
        local_nans = np.sum(np.isnan(self.inner))
        recvbuf = np.empty(1, dtype=int)
        self.comm.Allreduce(local_nans, recvbuf, op=MPI.SUM)

        return recvbuf[0]

    @property
    def dt_crit(self):
        return min(self.disc["dx"], self.disc["dy"]) / (self.vmax + self.vSound)

    @property
    def cfl(self):
        return self.dt / self.dt_crit

    @property
    def tv(self):
        grad_x = np.mean(np.abs(self.inner[:, 1:, :] - self.inner[:, :-1, :]))
        grad_y = np.mean(np.abs(self.inner[:, :, 1:] - self.inner[:, :, :-1]))

        local_tv = grad_x + grad_y
        recvbuf = np.empty(1, dtype=float)
        self.comm.Allreduce(local_tv, recvbuf, op=MPI.SUM)

        return recvbuf[0]

    def extend_inner(self, arr):

        # extend inner u by ghost cells
        arr__ = arr.reshape(-1)
        arr_ = np.concatenate(([arr__[0]], arr__, [arr__[-1]]))
        arr_ext = np.tile(arr_[:, np.newaxis], (1, 3))

        return arr_ext

    def update_wall(self, i):
        
        #self.geometry["U"] = min(0.1, 0.1 * i/100000)
        #if i%10000==0:
        #    print("current U: {}".format(self.geometry["U"]))

        if bool(self.material["elastic"]):

            u, du, u_p = self.elasticDeform.update_deformation_underrelax(self.p, self.u)
            self.u = u
            self.du_dt = du / self.dt
            self.g_u = u_p

            h_min = np.min(self.height.field[0])
            dh_wallforce, dp = self.elasticDeform.update_wallforce_pi(self.p, h_min, i)
            self.dh_dt_wallforce = dh_wallforce / self.dt
            self.int_dh_wallforce += dh_wallforce
            self.delta_p_wallforce = dp
            #dh_wallforce = 0.
            #du.fill(0.)

            dh = self.extend_inner(du) + dh_wallforce

            self.height.field[0] = self.height.field[0] + dh
            #self.height.field[3] = dh / self.dt
            self.height.field[1:3] = np.gradient(self.height.field[0], self.disc["dx"],
                                                 self.disc["dy"], edge_order=2)  # m/m

        return

    def compute_gradientAnalysis_terms(self, direction, iter):

        if (self.i + 1) % self.options['writeInterval'] != 0:
            return

        # ---------------- density --------------------
        # jx_x
        Fx = self.field[1]
        self.ga_jx_x[iter] = -(-direction * (np.roll(Fx, direction, axis=0) - Fx))[1:-1, 1:-1] / self.disc["dx"]
        # h_t_rho
        self.ga_h_t_rho[iter] = -((self.height.field[3] * self.field[0]) / self.height.field[0])[1:-1, 1:-1]
        # h_x_jx
        self.ga_h_x_jx[iter] = -((self.height.field[1] * self.field[1]) / self.height.field[0])[1:-1, 1:-1]

        # ---------------- flux --------------------
        # p_x
        Fx = self.eos.get_pressure(self.field)
        self.ga_p_x[iter] = -(-direction * (np.roll(Fx, direction, axis=0) - Fx))[1:-1, 1:-1] / self.disc["dx"]
        # uxjx_x
        Fx = self.field[1] * self.field[1] / self.field[0]
        self.ga_uxjx_x[iter] = -(-direction * (np.roll(Fx, direction, axis=0) - Fx))[1:-1, 1:-1] / self.disc["dx"]
        # tauxx_x
        Fx = self.viscous_stress.field[0]
        self.ga_tauxx_x[iter] = (-direction * (np.roll(Fx, direction, axis=0) - Fx))[1:-1, 1:-1] / self.disc["dx"]
        # h_x_uxjx
        self.ga_h_x_uxjx[iter] = -((self.height.field[1] * self.field[1]**2 / self.field[0]) / self.height.field[0])[1:-1, 1:-1]
        # h_x_tauxx
        self.ga_h_x_tauxx[iter] = -(self.height.field[1] * (self.viscous_stress.field[0] - self.wall_stress.upper[0]) / self.height.field[0])[1:-1, 1:-1]
        # tauxz
        self.ga_tauxz[iter] = ((self.wall_stress.upper[4] - self.wall_stress.lower[4]) / self.height.field[0])[1:-1, 1:-1]
        # h_t_jx
        self.ga_h_t_jx[iter] = -((self.height.field[3] * self.field[1]**2 / self.field[0]) / self.height.field[0])[1:-1, 1:-1]

    def update(self, i):
        warnings.warn(
            "update() is outdated — use update_fluid() instead",
            DeprecationWarning,
            stacklevel=2)
        return self.update_fluid(i)

    def update_fluid(self, i):
        """
        Wrapper function for the explicit time update of the solution field.

        Parameters
        ----------
        i : int
            time step

        """

        self.i = i
        # if self.fallback > 0:
        #     if i % self.fallback == 0:
        #         self.q_fallback = deepcopy(self.field)
        #         # self.num_resets = 0

        integrator = self.numerics["integrator"]

        # MacCormack forward backward
        if integrator == "MC" or integrator == "MC_fb":
            try:
                return self.mac_cormack(0)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Mac Cormack backward forward
        elif integrator == "MC_bf":
            try:
                return self.mac_cormack(1)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # MacCormack alternating
        elif integrator == "MC_alt":
            try:
                return self.mac_cormack(i % 2)
            except NotImplementedError:
                print("MacCormack scheme not implemented. Exit!")
                quit()

        # Richtmyer two-step Lax-Wendroff
        elif integrator == "LW":
            try:
                return self.richtmyer()
            except NotImplementedError:
                print("Lax-Wendroff scheme not implemented. Exit!")
                quit()

        # 3rd order Runge-Kutta
        elif integrator == "RK3":
            try:
                return self.runge_kutta3()
            except NotImplementedError:
                print("Runge-Kutta 3 scheme not implemented. Exit!")
                quit()

    def mac_cormack(self, switch):
        """
        Explicit time update using MacCormack's predictor corrector scheme with source term.
        Field is updated in place.

        Parameters
        ----------
        switch : int
            Determine if predictor step is forwards (0) or backwards (1).

        """

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        if switch == 0:
            directions = [-1, 1]
        elif switch == 1:
            directions = [1, -1]

        cfl = np.copy(self.cfl)
        q0 = self.field.copy()

        for i, d in enumerate(directions):

            fX, fY = self.predictor_corrector(self.field, self.height.field, d)
            if self.check_p_reset():
                return self.post_integrate(skip=True)
            src = self.get_source(self.field, self.height.field, self.slip_length.field)
            if self.check_tau_reset():
                return self.post_integrate(skip=True)

            if bool(self.options['gradientAnalysis']):  # at this position because field is not updated yet but stress is computed
                self.compute_gradientAnalysis_terms(direction=d, iter=i)

            self.field = self.field - self.dt * (fX / dx + fY / dy + src)
            self.fill_ghost_buffer()

        self.field = 0.5 * (self.field + q0)

        if bool(self.options['gradientAnalysis']):
            grad = self.field - q0
            self.ga_rho_t = grad[0][1:-1, 1:-1] / self.dt
            self.ga_jx_t = grad[1][1:-1, 1:-1] / self.dt
            self.ga_ux = self.inner[1] / self.inner[0]
            self.mass_sum = np.sum(self.inner[0] * self.height.inner[0] * dx * dy)  # kg

        if "fluxLim" in self.numerics.keys():
            self.field += self.numerics["fluxLim"] * TVD_MC_correction(q0, cfl)
            self.fill_ghost_buffer()

        return self.post_integrate()

    def richtmyer(self):
        """
        Explicit time update using Richtmyer's two-step scheme with source term.
        Field is updated in place.

        """

        q0 = self.field.copy()

        # Step 1
        self.lax_step()
        self.fill_ghost_buffer()

        # Step 2
        self.leap_frog(q0)
        self.fill_ghost_buffer()

        return self.post_integrate()

    def runge_kutta3(self):

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        q0 = self.field.copy()

        for stage in np.arange(1, 4):

            dX, dY = self.diffusiveCD()
            fX, fY = self.hyperbolicTVD()
            src = self.get_source(self.field, self.height.field, self.slip_length.field)

            tmp = self.field - self.dt * (fX / dx + fY / dy + dX / (2 * dx) + dY / (2 * dy) + src)

            if stage == 1:
                self.field = tmp
                self.fill_ghost_buffer()
            elif stage == 2:
                self.field = 3 / 4 * q0 + 1 / 4 * tmp
                self.fill_ghost_buffer()
            elif stage == 3:
                self.field = 1 / 3 * q0 + 2 / 3 * tmp
                self.fill_ghost_buffer()

        return self.post_integrate()

    def check_p_reset(self):
        if self.eos.GP.reset:
            if self.wait_p_after_reset == 0:
                if self.gp['wait'] < 0:
                    # Go back to start and pause GP update for some steps
                    self.initialize(q_init=self.q_init, t_init=(self.time, self.dt), restart=True)
                self.num_resets += 1
                self.wait_p_after_reset += 1
                self.eos.ncalls = 0
                return True
            elif self.wait_p_after_reset < 2 * abs(self.gp['wait']):
                # continue waiting
                self.wait_p_after_reset += 1
                return False
            else:
                # end waiting
                self.wait_p_after_reset = 0
                self.eos.GP.reset_reset()
                return False
        else:
            return False

    def check_tau_reset(self):
        # in 2D, this checks if any of the two GP's is set to 'reset'
        # if yes, both will be paused
        if self.wall_stress.reset:
            if self.wait_tau_after_reset == 0:
                if self.gp['wait'] < 0:
                    # Go back to start and pause GP update for some steps
                    self.initialize(q_init=self.q_init, t_init=(self.time, self.dt), restart=True)
                self.wait_tau_after_reset += 1
                self.num_resets += 1
                self.wall_stress.ncalls = 0
                return True
            elif self.wait_tau_after_reset < 2 * abs(self.gp['wait']):
                # continue waiting
                self.wait_tau_after_reset += 1
                return False
            else:
                # end waiting
                self.wait_tau_after_reset = 0
                self.wall_stress.reset_reset()
                return False
        else:
            return False

    def post_integrate(self, skip=False):
        """
        Compute relative change of solution after time update.
        Update time, and, if adaptive time-stepping is used, also the time step.

        Parameters
        ----------
        q0 : np.array
            Copy of the solution field from the previous step

        """

        if skip:
            print('>>>> Active learning seems to stall. ', end='')
            remaining_resets = self.gp["maxResets"] - self.num_resets
            if remaining_resets < 0:
                print('Stop immediately.')
                return 3
            else:
                prefix = 'Continue anyways' if self.gp['wait'] > 0 else 'Restart'
                print(f'{prefix}... ({remaining_resets} resets remaining, pause for {abs(self.gp["wait"])} steps).')
                return -1

        self.time += self.dt
        self.wall_stress.increment()
        self.eos.GP.increment()

        last_cfl = np.copy(self.cfl)

        if bool(self.numerics["adaptive"]):
            self.dt = self.numerics["C"] * self.dt_crit

        self.eps = abs(self.ekin - self.Ekin_old) / self.Ekin_old / last_cfl
        self.eps_buffer.append(self.eps)
        self.Ekin_old = deepcopy(self.ekin)

        # convergence
        if np.all(np.array(self.eps_buffer) < self.numerics['tol']):
            return 1
        # maximum time reached
        elif round(self.time, 15) >= self.numerics['maxT']:
            return 2
        # NaN detected
        elif self.isnan > 0:
            return 4
        else:
            return 0

    def fill_ghost_buffer(self):
        """
        Communicate results from adjacent domains to ghost buffers.
        Fill ghost cells at the boundary according to the applied BCs.

        """

        x0 = self.bc["x0"]
        x1 = self.bc["x1"]
        y0 = self.bc["y0"]
        y1 = self.bc["y1"]

        (ls, ld), (rs, rd), (bs, bd), (ts, td) = self.get_neighbors()

        ng = self.disc["nghost"]
        ngt = ng + 1

        # Send to left, receive from right
        recvbuf = np.ascontiguousarray(self.field[:, -ng:, :])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, ng:ngt, :]), ld, recvbuf=recvbuf, source=ls)
        # fill ghost buffer, adjust for non-periodic BCs
        if ls >= 0:
            self.field[:, -ng:, :] = recvbuf
        else:
            if np.any(x1 == "D"):
                self.field[x1 == "D", -ng:,
                           :] = self.dirichlet_bc_down(self.bc["rhox1"], self.field[x1 == "D", -ngt:-ng, :], ax=1)
            if np.any(x1 == "N"):

                self.field[x1 == "N", -ng:, :] = self.neumann_bc_down(self.field[x1 == "N", -ngt:-ng, :], ax=1)

        # Send to right, receive from left
        recvbuf = np.ascontiguousarray(self.field[:, :ng, :])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, -ngt:-ng, :]), rd, recvbuf=recvbuf, source=rs)
        # fill ghost buffer, adjust for non-periodic BCs
        if rs >= 0:
            self.field[:, :ng, :] = recvbuf
        else:
            if np.any(x0 == "D"):
                self.field[x0 == "D", :ng, :] = self.dirichlet_bc_up(
                    self.bc["rhox0"], self.field[x0 == "D", ng:ngt, :], ax=1)
            if np.any(x0 == "N"):
                self.field[x0 == "N", :ng, :] = self.neumann_bc_up(self.field[x0 == "N", ng:ngt, :], ax=1)

        # Send to bottom, receive from top
        recvbuf = np.ascontiguousarray(self.field[:, :, -ng:])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, :, ng:ngt]), bd, recvbuf=recvbuf, source=bs)
        # fill ghost buffer, adjust for non-periodic BCs
        if bs >= 0:
            self.field[:, :, -ng:] = recvbuf
        else:
            if np.any(y1 == "D"):
                self.field[y1 == "D", :, -
                           ng:] = self.dirichlet_bc_down(self.bc["rhoy1"], self.field[y1 == "D", :, -ngt:-ng], ax=2)
            if np.any(y1 == "N"):
                self.field[y1 == "N", :, -ng:] = self.neumann_bc_up(self.field[y1 == "N", :, -ngt:-ng], ax=2)

        # Send to top, receive from bottom
        recvbuf = np.ascontiguousarray(self.field[:, :, :ng])
        self.comm.Sendrecv(np.ascontiguousarray(self.field[:, :, -ngt:-ng]), td, recvbuf=recvbuf, source=ts)
        # fill ghost buffer, adjust for non-periodic BCs
        if ts >= 0:
            self.field[:, :, :ng] = recvbuf
        else:
            if np.any(y0 == "D"):
                self.field[y0 == "D", :, :ng] = self.dirichlet_bc_up(
                    self.bc["rhoy0"], self.field[y0 == "D", :, ng:ngt], ax=2)
            if np.any(y0 == "N"):
                # self.field[y0 == "N", :, ng:ngt]
                self.field[y0 == "N", :, :ng] = self.neumann_bc_up(self.field[y0 == "N", :, ng:ngt], ax=2)

    def get_source(self, q, h, Ls):
        """
        Compute the source term from the unknown and gap height.

        Parameters
        ----------
        q : np.array
            Solution field.
        h : np.array
            Gap height (and gradients).
        Ls : np.array
            Slip length (lower surface)

        Returns
        -------
        np.array
            Source term

        """

        self.viscous_stress.set(q, h, Ls)

        # self.upper_stress.set(q, h, Ls, "top")
        # self.lower_stress.set(q, h, Ls, "bottom")

        self.wall_stress.set(q, h, Ls)

        stress = self.viscous_stress.field

        out = np.zeros_like(q)

        # origin bottom, U_top = 0, U_bottom = U
        out[0] = (q[1] * h[1] + q[2] * h[2]) / h[0]

        if not bool(self.options['ignoreStresses']):
            out[1] = -((stress[0] - self.wall_stress.upper[0]) * h[1] +
                       (stress[2] - self.wall_stress.upper[5]) * h[2] +
                       self.wall_stress.upper[4] - self.wall_stress.lower[4]) / h[0]

            out[2] = -((stress[2] - self.wall_stress.upper[5]) * h[1] +
                       (stress[1] - self.wall_stress.upper[1]) * h[2] +
                       self.wall_stress.upper[3] - self.wall_stress.lower[3]) / h[0]

        if not bool(self.numerics["stokes"]):
            out[1] += (q[1] * q[1] * h[1] + q[1] * q[2] * h[2]) / h[0] / q[0]
            out[2] += (q[2] * q[1] * h[1] + q[2] * q[2] * h[2]) / h[0] / q[0]

        # source terms due to height change with time
        cor0 = (h[3] * q[0]) / h[0]  # m/s * kg/m^3 / m = kg/(m^3 s)
        cor1 = (h[3] * q[1]) / h[0]
        cor2 = (h[3] * q[2]) / h[0]

        #out[0] = out[0] - cor0
        #out[1] = out[1] - cor1
        #out[2] = out[2] - cor2

        return out

    def predictor_corrector(self, q, h, direction):
        """
        Compute predictor/corrector fluxes fro MacCormack's method.

        Parameters
        ----------
        q : np.array
            Solution field.
        h : np.array
            Gap height (and gradients).
        direction : int
            Direction of the substep (-1: forwards / 1: backwards differencing)

        Returns
        -------
        tuple
            Total numerical flux in x- and y-directions respectively.

        """

        # GP: 1 pressure call
        FxH, FyH = self.hyperbolicFlux(q)
        FxD, FyD = self.diffusiveFlux(q, h, self.slip_length.field)

        Fx = FxH - FxD
        Fy = FyH - FyD

        flux_x = -direction * (np.roll(Fx, direction, axis=1) - Fx)
        flux_y = -direction * (np.roll(Fy, direction, axis=2) - Fy)

        return flux_x, flux_y

    def hyperbolicFlux(self, q):
        """Compute hyperbolic flux.

        Parameters
        ----------
        q : np.array
            Solution field.

        Returns
        -------
        np.array
            Hyperbolic flux field.

        """

        p = self.eos.get_pressure(q)
        p = gaussian_filter1d(p, sigma=2)

        if bool(self.options['pressure']):
            self.p = p[1:-1, 1:-1]

        inertialess = bool(self.numerics["stokes"])

        Fx = np.zeros_like(q)
        Fy = np.zeros_like(q)

        if inertialess:
            # x
            Fx[0] = q[1]
            Fx[1] = p
            # y
            Fy[0] = q[2]
            Fy[2] = p
        else:
            # x
            Fx[0] = q[1]
            Fx[1] = q[1] * q[1] / q[0] + p
            Fx[2] = q[2] * q[1] / q[0]
            # y
            Fy[0] = q[2]
            Fy[1] = q[1] * q[2] / q[0]
            Fy[2] = q[2] * q[2] / q[0] + p

        return Fx, Fy

    def diffusiveFlux(self, q, h, Ls):
        """Compute diffusive flux.

        Parameters
        ----------
        q : np.array
            Solution field.
        h : np.array
            Gap height (and gradients).
        Ls : np.array
            Slip length (lower surface)

        Returns
        -------
        np.array
            Diffusive flux field

        """

        self.viscous_stress.set(q, h, Ls)

        Dx = np.zeros_like(q)
        Dy = np.zeros_like(q)

        if not bool(self.options['ignoreStresses']):
            Dx[1] += self.viscous_stress.field[0]
            Dx[2] += self.viscous_stress.field[2]

            Dy[1] += self.viscous_stress.field[2]
            Dy[2] += self.viscous_stress.field[1]

        return Dx, Dy

    def lax_step(self):

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        # GP: 2 pressure calls
        FxHE, _ = self.hyperbolicFlux(self.edgeE)
        _, FyHN = self.hyperbolicFlux(self.edgeN)
        FxHW = np.roll(FxHE, 1, axis=1)
        FyHS = np.roll(FyHN, 1, axis=2)

        FxDE, _ = self.diffusiveFlux(self.edgeE, self.height.edgeE, self.slip_length.field)
        FxDW, _ = self.diffusiveFlux(self.edgeW, self.height.edgeW, self.slip_length.field)
        _, FyDN = self.diffusiveFlux(self.edgeN, self.height.edgeN, self.slip_length.field)
        _, FyDS = self.diffusiveFlux(self.edgeS, self.height.edgeS, self.slip_length.field)

        FE = FxHE - FxDE
        FW = FxHW - FxDW
        FN = FyHN - FyDN
        FS = FyHS - FyDS

        # GP: 1 wall_stress call
        src = self.get_source(self.field, self.height.field, self.slip_length.field)

        self.field = self.field - self.dt / 2. * ((FE - FW) / dx + (FN - FS) / dy + src)

    def leap_frog(self, q0):

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        # GP: 2 pressure calls
        FxHE, _ = self.hyperbolicFlux(self.edgeE)
        _, FyHN = self.hyperbolicFlux(self.edgeN)
        FxHW = np.roll(FxHE, 1, axis=1)
        FyHS = np.roll(FyHN, 1, axis=2)

        FxDE, _ = self.diffusiveFlux(self.edgeE, self.height.edgeE, self.slip_length.field)
        FxDW, _ = self.diffusiveFlux(self.edgeW, self.height.edgeW, self.slip_length.field)
        _, FyDN = self.diffusiveFlux(self.edgeN, self.height.edgeN, self.slip_length.field)
        _, FyDS = self.diffusiveFlux(self.edgeS, self.height.edgeS, self.slip_length.field)

        FE = FxHE - FxDE
        FW = FxHW - FxDW
        FN = FyHN - FyDN
        FS = FyHS - FyDS

        # GP: 1 wall_stree call
        src = self.get_source(self.field, self.height.field, self.slip_length.field)

        self.field = q0 - self.dt * ((FE - FW) / dx + (FN - FS) / dy + src)

    def cubicInterpolation(self, q, ax):

        # weights proposed by Bell et al., PRE 76 (2007)
        # a1 = (np.sqrt(7) + 1) / 4
        # a2 = (np.sqrt(7) - 1) / 4

        # weights from piecewise parabolic method (PPM)
        # Collela and Woodward, J. Comp. Phys. 54 (1984)
        a1 = 7 / 12
        a2 = 1 / 12

        Q = a1 * (q + np.roll(q, -1, axis=ax)) - a2 * (np.roll(q, 1, axis=ax) + np.roll(q, -2, axis=ax))

        return Q

    def dirichlet_bc_down(self, q0, qadj, ax):

        if qadj.shape[ax] == 1:
            a1 = 1. / 2.
            a2 = 0.
            q1 = qadj
            q2 = 0.
        elif qadj.shape[ax] == 2:
            # weights proposed by Bell et al., PRE 76 (2007)
            # a1 = (np.sqrt(7) + 1) / 4
            # a2 = (np.sqrt(7) - 1) / 4

            # weights from piecewise parabolic method (PPM)
            # Collela and Woodward, J. Comp. Phys. 54 (1984)
            a1 = 7 / 12
            a2 = 1 / 12

            if ax == 1:
                q1 = qadj[:, 1:, :]
                q2 = qadj[:, :1, :]
            if ax == 2:
                q1 = qadj[:, :, 1:]
                q2 = qadj[:, :, :1]

        Q = (q0 - a1 * q1 + a2 * q2) / (a1 - a2)

        return Q

    def dirichlet_bc_up(self, q0, qadj, ax):

        if qadj.shape[ax] == 1:
            # linear reconstruction (MC; LW)
            a1 = 1. / 2.
            a2 = 0.
            q1 = qadj
            q2 = 0.
        elif qadj.shape[ax] == 2:
            # PPM reconstruction (RK3)
            # weights proposed by Bell et al., PRE 76 (2007)
            # a1 = (np.sqrt(7) + 1) / 4
            # a2 = (np.sqrt(7) - 1) / 4

            # weights from piecewise parabolic method (PPM)
            # Collela and Woodward, J. Comp. Phys. 54 (1984)
            a1 = 7 / 12
            a2 = 1 / 12

            if ax == 1:
                q1 = qadj[:, :1, :]
                q2 = qadj[:, 1:, :]
            if ax == 2:
                q1 = qadj[:, :, :1]
                q2 = qadj[:, :, 1:]

        Q = (q0 - a1 * q1 + a2 * q2) / (a1 - a2)

        return Q

    def neumann_bc_down(self, qadj, ax):

        if qadj.shape[ax] == 1:
            # linear reconstruction (MC; LW)
            a1 = 1. / 2.
            a2 = 0.
            q1 = qadj
            q2 = 0.
        elif qadj.shape[ax] == 2:
            # PPM reconstruction (RK3)
            # weights proposed by Bell et al., PRE 76 (2007)
            # a1 = (np.sqrt(7) + 1) / 4
            # a2 = (np.sqrt(7) - 1) / 4

            # weights from piecewise parabolic method (PPM)
            # Collela and Woodward, J. Comp. Phys. 54 (1984)
            a1 = 7 / 12
            a2 = 1 / 12

            if ax == 1:
                q1 = qadj[:, 1:, :]
                q2 = qadj[:, :1, :]
            if ax == 2:
                q1 = qadj[:, :, 1:]
                q2 = qadj[:, :, :1]

        Q = ((1. - a1) * q1 + a2 * q2) / (a1 - a2)

        return Q

    def neumann_bc_up(self, qadj, ax):

        if qadj.shape[ax] == 1:
            # linear reconstruction (MC; LW)
            a1 = 1. / 2.
            a2 = 0.
            q1 = qadj
            q2 = 0.
        elif qadj.shape[ax] == 2:
            # PPM reconstruction (RK3)
            # weights proposed by Bell et al., PRE 76 (2007)
            # a1 = (np.sqrt(7) + 1) / 4
            # a2 = (np.sqrt(7) - 1) / 4

            # weights from piecewise parabolic method (PPM)
            # Collela and Woodward, J. Comp. Phys. 54 (1984)
            a1 = 7 / 12
            a2 = 1 / 12

            if ax == 1:
                q1 = qadj[:, :1, :]
                q2 = qadj[:, 1:, :]
            if ax == 2:
                q1 = qadj[:, :, :1]
                q2 = qadj[:, :, 1:]

        Q = ((1. - a1) * q1 + a2 * q2) / (a1 - a2)

        return Q

    def hyperbolicTVD(self):

        Qx = self.cubicInterpolation(self.field, 1)
        Qy = self.cubicInterpolation(self.field, 2)

        # GP: 2 pressure calls
        Fx, _ = self.hyperbolicFlux(Qx)
        _, Fy = self.hyperbolicFlux(Qy)

        flux_x = Fx - np.roll(Fx, 1, axis=1)
        flux_y = Fy - np.roll(Fy, 1, axis=2)

        return flux_x, flux_y

    def hyperbolicCD(self, q, p, dt, ax):

        dx = self.disc["dx"]
        dy = self.disc["dy"]

        # GP: 1 pressure call
        FxH, FyH = self.hyperbolicFlux(q.field)

        Fx = np.roll(FxH, -1, axis=1) - np.roll(FxH, 1, axis=1) * dt / (2 * dx)
        Fy = np.roll(FxH, -1, axis=2) - np.roll(FxH, 1, axis=2) * dt / (2 * dy)

        return Fx, Fy

    def diffusiveCD(self):

        Dx, Dy = self.diffusiveFlux(self.field, self.height.field, self.slip_length.field)

        flux_x = np.roll(Dx, -1, axis=1) - np.roll(Dx, 1, axis=1)
        flux_y = np.roll(Dy, -1, axis=2) - np.roll(Dy, 1, axis=2)

        return flux_x, flux_y
