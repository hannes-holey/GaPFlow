#
# Copyright 2020, 2022 Hannes Holey
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

import ContactMechanics as CM
from mpi4py import MPI
import numpy as np

class ElasticDeformation:
    """wrapper class around the FFTElasticHalfSpace classes from ContactMechanics. 
    Checks periodicity and instanciates the corresponding class.
    """

    def __init__(self, size, HANS_slice, dx, dy, E, v, perX, perY, bc={}, dev_options={}, comm=None, p_ext=1e05):
        """Read boundary conditions and select the propriate class

        Parameters
        ----------
        size : 2-tuple of integers
            number of cells in x- and y-direction
        dx : float
            cell length in x-direction [m]
        dy : float
            cell length in y-direction [m]
        E : float
            Young's modulus [Pa]
        v : float
            Poisson ratio [-]
        perX : bool
            periodicity in x-direction
        perY : bool
            periodicity in y-direction
        bc : dict, optional
            in the case of non-periodicity, the boundary pressures need to be specified. 
            This model assumes constant pressure of the given value outside the domain
            for non-periodicity in x-direction specify: 'px0', 'px1'
            for non-periodicity in y-direction specify: 'py0', 'py1'
        dev_options : dict, optional
            developer and analysis options for Green's function calculation
            'iter_periodization' : int : value that specifies how many periodic images are taken into account for periodization
            'full_num' : bool : if True, numerical approximation is used in the full periodicity case
            'verbose' : bool
        """
        self.comm = comm
        self.nb_grid_pts_HANS = size

        # TODO: user-defined in yaml
        self.factor_u_underrelax = 0.5e-04
        self.p_ext = p_ext

        # for a delta p of 10 bar = 1e06 Pa, it should take 10.000 steps to approach
        # note: h_min is additionally considered at each step
        self.p_factor = 1.e-13
        # integral part should be allowed to grow as strong as p only after 10.000 steps
        self.i_factor = 0. #1e-16
        self.int_diff = 0.

        Lx = dx * size[0]
        Ly = dy * size[1]
        self.area_per_cell = dx * dy

        # fully periodic
        if perX and perY:
            self.periodicity = 'full'
            if 'full_num' in dev_options and dev_options['full_num']:
                self.ElDef = CM.SemiPeriodicFFTElasticHalfSpace(size, E, (Lx, Ly), (perX, perY), fft='mpi')
            else:
                self.ElDef = CM.PeriodicFFTElasticHalfSpace(size, E, (Lx, Ly), stiffness_q0=0., poisson=v, fft='mpi')
        # partially periodic
        elif (perX != perY):
            self.periodicity = 'partial'
            self.ElDef = CM.SemiPeriodicFFTElasticHalfSpace(size, E, (Lx, Ly), (perX, perY), fft='mpi')
        # non-periodic
        else:
            self.periodicity = 'none'
            self.ElDef = CM.FreeFFTElasticHalfSpace(size, E, (Lx, Ly), fft='mpi', communicator=self.comm)

        # check parallel
        if self.comm is not None and self.comm.Get_size() > 1:
            self._parallel_preparation(HANS_slice)
            self.parallel = True
        else:
            self.parallel = False

    def _parallel_preparation(self, HANS_slice):
        """Initializes data required for parallel mode for
        translation of domain decomposition between HANS and ContactMechanics (muFFT).
        HANS subdomain slice information has to be provided, muFFT slice information 
        will be gathered.

        Parameters
        ----------
        HANS_slice : tuple(slice, slice)
            slice information for the HANS subdomain (worker-specific)
        """
        self.rank = self.comm.Get_rank()
        self._gather_muFFT_slice_list()
        self._gather_HANS_slice_list(HANS_slice)
        self.nb_grid_pts_muFFT = self.ElDef.nb_grid_pts

    def _gather_muFFT_slice_list(self) -> None:
        """Gathers information about the muFFT subdomain decomposition
        """
        loc = self.ElDef.fftengine.subdomain_locations
        shape = self.ElDef.fftengine.nb_subdomain_grid_pts
        slice_ = tuple(slice(loc[d], loc[d] + shape[d]) for d in range(len(loc)))
        self.muFFT_slice_list = self.comm.gather(slice_, root=0)

    def _gather_HANS_slice_list(self, HANS_slice) -> None:
        """Gathers information about the HANS subdomain decomposition
        """
        self.HANS_slice_list = self.comm.gather(HANS_slice, root=0)

    def _redistribute_toCM(self, HANS_field):
        """Redistributes the pressure chunks to the ContactMechanics padded field using the muFFT slice information
        """
        # Gather on the root using HANS slices
        local_gather = self.comm.gather(HANS_field, root=0)

        if self.rank == 0:
            padded_global = np.zeros(self.nb_grid_pts_muFFT)
            for i, local in enumerate(local_gather):
                padded_global[self.HANS_slice_list[i]] = local
            # Redistribute to workers using muFFT slices
            chunks = [padded_global[slice_] for slice_ in self.muFFT_slice_list]
        else:
            chunks = None
        return self.comm.scatter(chunks, root=0)

    def _redistribute_toHANS(self, muFFT_field):
        """Redistributes the ContactMechanics field to the HANS subdomain using the muFFT slice information
        """
        # Gather on the root using muFFT slices
        local_gather = self.comm.gather(muFFT_field, root=0)

        if self.rank == 0:
            global_field = np.zeros(self.nb_grid_pts_muFFT) # in case the return is still padded
            for i, local in enumerate(local_gather):
                global_field[self.muFFT_slice_list[i]] = local
            # Redistribute to workers using HANS slices
            chunks = [global_field[slice_] for slice_ in self.HANS_slice_list]
        else:
            chunks = None
        return self.comm.scatter(chunks, root=0)

    def get_deformation(self, p):
        """Main function for the calculation of the elastic deformation due to given pressure field p.

        Parameters
        ----------
        p : 2D numpy array
            pressure field [Pa]

        Returns
        -------
        u : 2D numpy array
            elastic deformation [m]
        """

        # TODO: use boundary condition p
        p = p - 1e05
        forces = p * self.area_per_cell

        if self.parallel:
            forces = self._redistribute_toCM(forces)

        disp = -self.ElDef.evaluate_disp(forces)

        if self.parallel:
            disp = self._redistribute_toHANS(disp)
        
        return disp

    def get_G_real(self):
        """only for analysis and development purposes
        Returns the 'ordered' G_real numpy array
        Inverting the custom frequency ordering in preparation for FFT

        Returns
        -------
        G_real_ordered : 2D numpy array
            ordered G_real
        """

        #nx = int(np.ceil(self.G_real.shape[0]/2))
        #ny = int(np.ceil(self.G_real.shape[1]/2))
 
        #m = np.concatenate((np.arange(nx), np.arange(-(nx-1), 0)))
        #n = np.concatenate((np.arange(ny), np.arange(-(ny-1), 0)))

        return self.ElDef.get_G_real()
    

    def get_G_real_slices(self):
        """only for analysis and development purposes
        Returns two middle slices of the G_real array, in x- and y-direction
        """

        return self.ElDef.get_G_real_slices()

    def update_deformation_underrelax(self, p, u_prev):
        """Updates elastic deformation using underrelaxation

        Parameters
        ----------
        p : 2D np.array
            pressure field
        u_prev : 2D np.array
            deformation from last timestep

        Returns
        -------
        u_new : 2D np.array
            updated, underrelaxed deformation
        du : 2D np.array
            deformation delta to last timestep
        u_p : 2D np.array
            computed deformation of current pressure
        """

        u_p = self.get_deformation(p)

        u_ret = (1-self.factor_u_underrelax)*u_prev + self.factor_u_underrelax*u_p
        du = u_ret-u_prev

        return u_ret, du, u_p
    
    def update_wallforce_pi(self, p, h_min, i):
        """Update rigid height change due to wallforce
        using PI-control

        Parameters
        ----------
        p : 2D np.array
            pressure field
        h_min : float [m]
            minimum height in the domain

        Returns
        -------
        dh : float [m]
            rigid height change to be applied
        """

        if i>200000:
            return 0., 0.

        local_sum = np.sum(p)
        local_n = p.size

        # allreduce to get global mean
        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        global_n = self.comm.allreduce(local_n, op=MPI.SUM)
        global_mean = global_sum / global_n
        global_hmin = self.comm.allreduce(h_min, op=MPI.MIN)

        pressure_diff = global_mean - self.p_ext
        self.int_diff += pressure_diff

        P = pressure_diff * self.p_factor
        I = self.int_diff * self.i_factor

        dh = (P + I) * global_hmin

        return dh, -pressure_diff