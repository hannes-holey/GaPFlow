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


import numpy as np

from hans.multiscale.gp import GP_pressure, GP_pressure2D

import hans.models.pressure as pressure_model
import hans.models.density as density_model
import hans.models.sound_speed as sound_model
import hans.models.viscosity as viscosity_model


class Material:

    def __init__(self, material, gp=None):
        self.material = material

        self.gp = gp
        self.ncalls = 0

        self._eos_config()
        self._viscosity_config()

    def init_gp(self, q, db):

        if q.shape[-1] > 3:
            # 2D
            self.GP = GP_pressure2D(db, self.gp)
        else:
            # 1D
            self.GP = GP_pressure(db, self.gp)

        # Initialize
        self.GP.setup(q)

    def get_pressure(self, q):

        if self.gp is not None:
            # In contrast to viscous stress, pressure is not stored internally but just returned.
            # There are two calls to the EOS within one time step (MC).
            # We want to perform an active learning step only once.
            p, cov = self.GP.predict(q, self.ncalls % self.gp['_pCalls'] == 0)
            p = p[0]
        else:
            p = self.eos_pressure(q[0])

        self.ncalls += 1

        return p

    def get_sound_speed(self, q):

        if self.gp is not None:
            dp_dq, _ = self.GP.predict_gradient()
            c = np.sqrt(np.amax(np.abs(dp_dq[1])))
        else:
            c = self.eos_sound_speed(q[0])

        return c

    def eos_pressure(self, rho):
        pressure = self._eos_pressure_func(rho, *self._eos_args)

        if 'Pcav' in self.material.keys():
            return np.maximum(pressure, self.material['Pcav'])
        else:
            return pressure

    def eos_density(self, p):

        density = self._eos_density_func(p, *self._eos_args)

        return density

    def eos_sound_speed(self, rho):

        speed = self._eos_sound_func(rho, *self._eos_args)

        return np.amax(speed)

    def viscosity(self, U, V, rho, height):

        # density-dependence
        mu0 = self._viscosity_dens_func(self._viscosity_dens_transform(rho),
                                        *self._viscosity_dens_args)

        # shearrate-dependence
        # TODO: not only Couette shear rate
        shear_rate = np.sqrt(U**2 + V**2) / height

        mu = self._viscosity_shear_func(shear_rate,
                                        mu0,
                                        *self._viscosity_shear_args)

        return mu

    def _eos_config(self):
        # Dowson-Higginson (with cavitation)
        if self.material['EOS'] == "DH":
            self._eos_args = [self.material['rho0'],
                              self.material['P0'],
                              self.material['C1'],
                              self.material['C2']]
            self._eos_pressure_func = pressure_model.dowson_higginson
            self._eos_density_func = density_model.dowson_higginson
            self._eos_sound_func = sound_model.dowson_higginson

        # Power law, (alpha = 0: ideal gas)
        elif self.material['EOS'] == "PL":
            self._eos_args = [self.material['rho0'],
                              self.material['P0'],
                              self.material['alpha']]

            self._eos_pressure_func = pressure_model.power_law
            self._eos_density_func = density_model.power_law
            self._eos_sound_func = sound_model.power_law

        # Van der Waals equation
        elif self.material['EOS'] == "vdW":

            self._eos_args = [self.material['M'],
                              self.material['T0'],
                              self.material['a'],
                              self.material['b']]

            self._eos_pressure_func = pressure_model.van_der_waals
            # self._eos_density_func = density_model.van_der_waals
            self._eos_sound_func = sound_model.van_der_waals

        # Murnaghan-Tait equation (modified Tait eq.)
        elif self.material['EOS'] == "Tait":
            self._eos_args = [self.material['rho0'],
                              self.material['P0'],
                              self.material['K'],
                              self.material['n']]

            self._eos_pressure_func = pressure_model.murnaghan_tait
            self._eos_density_func = density_model.murnaghan_tait
            self._eos_sound_func = sound_model.murnaghan_tait

        # Cubic polynomial
        elif self.material['EOS'] == "cubic":
            self._eos_args = [self.material['a'],
                              self.material['b'],
                              self.material['c'],
                              self.material['d']]

            self._eos_pressure_func = pressure_model.cubic
            # self._eos_density_func = density_model.cubic
            self._eos_sound_func = sound_model.cubic

        elif self.material['EOS'] == "BWR":
            self._eos_args = [self.material['T']]
            self._eos_pressure_func = pressure_model.bwr
            # self._eos_density_func = density_model.bwr
            self._eos_sound_func = sound_model.bwr

        elif self.material['EOS'].startswith("Bayada"):
            self._eos_args = [self.material["rhol"],
                              self.material["rhov"],
                              self.material["cl"],
                              self.material["cv"]]

            self._eos_pressure_func = pressure_model.bayada_chupin
            self._eos_density_func = density_model.bayada_chupin
            self._eos_sound_func = sound_model.bayada_chupin

    def _viscosity_config(self):

        self._viscosity_dens_transform = lambda x: x

        # Dukler viscosity in mixture (default)
        if self.material["EOS"] == "Bayada" or self.material["EOS"] == "Bayada_D":

            self._viscosity_dens_args = [self.material["rhol"],
                                         self.material["rhov"],
                                         self.material["shear"],
                                         self.material["shearv"]]
            self._viscosity_dens_func = viscosity_model.dukler_mixture

        # McAdams viscosity in mixture
        elif self.material["EOS"] == "Bayada_MA":
            self._viscosity_dens_args = [self.material["rhol"],
                                         self.material["rhov"],
                                         self.material["shear"],
                                         self.material["shearv"]]
            self._viscosity_dens_func = viscosity_model.mc_adams_mixture

        else:
            if 'piezo' in self.material.keys():
                if self.material["piezo"] == "Vogel":
                    self._viscosity_dens_args = [self.material['rho0'],
                                                 self.material["g"],
                                                 self.material["mu_inf"],
                                                 self.material["phi_inf"],
                                                 self.material["BF"]]
                    self._viscosity_dens_func = viscosity_model.vogel_piezo

                elif self.material["piezo"] == "Barus":
                    self._viscosity_dens_args = [self.material["shear"],
                                                 self.material["aB"]]
                    self._viscosity_dens_func = viscosity_model.barus_piezo
                    self._viscosity_dens_transform = self.eos_pressure
            else:
                self._viscosity_dens_args = []
                self._viscosity_dens_func = lambda x: self.material['shear']

        if "thinning" in self.material.keys():

            if self.material["thinning"] == "Eyring":
                self._viscosity_shear_args = [self.material["tau0"]]
                self._viscosity_shear_func = viscosity_model.eyring_shear

            elif self.material["thinning"] == "Carreau":
                self._viscosity_shear_args = [self.material["shearinf"] if "shearinf" in self.material.keys() else 0.,
                                              self.material["relax"],
                                              self.material["a"],
                                              self.material["N"]]
                self._viscosity_shear_func = viscosity_model.carreau_shear

            elif self.material["thinning"] == "PL":
                self._viscosity_shear_args = [self.material["n"]]
                self._viscosity_shear_func = viscosity_model.power_law_shear

        else:
            self._viscosity_shear_args = []
            self._viscosity_shear_func = lambda x, mu0: mu0
