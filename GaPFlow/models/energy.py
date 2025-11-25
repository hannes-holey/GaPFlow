#
# Copyright 2025 Christoph Huber
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

import numpy.typing as npt
from typing import Callable, Any

NDArray = npt.NDArray[np.floating]


class Energy():

    def __init__(self,
                 fc: Any,
                 energy_spec: dict
                 ) -> None:

        self.__field = fc.register_real_field('total_energy')
        self.k = energy_spec['k']
        self.cv = energy_spec['cv']
        self.T_wall = energy_spec['T_wall']
        self.alpha_wall = energy_spec['alpha_wall']

    @property
    def energy(self) -> NDArray:
        """Total energy field"""
        return self.__field.p

    def init_quad(self, fc_fem, quad_list: list[int], create_fun: Callable) -> None:
        """Initialize quadrature point fields"""
        self.quad_list = quad_list
        self.field_list = ['E', 'T', 'dT_drho', 'dT_djx', 'dT_dE', 'S', 'dS_drho', 'dS_djx', 'dS_dE']
        create_fun(self, fc_fem, self.field_list, self.quad_list)

    def build_grad(self) -> None:
        """Build gradients of energy-related fields at quadrature points"""
        pass

    def update_quad(self,
                    quad_fun: Callable[[NDArray, int], NDArray],
                    inner_fun: Callable[[NDArray], NDArray],
                    get_quad_field: Callable[[str, int], NDArray],
                    *args) -> None:
        """Update pressure and gradients at nodal and quadrature points"""

        for nb_quad in self.quad_list:  # update p and dp_drho quadrature fields
            args = (get_quad_field('rho', nb_quad),
                    get_quad_field('jx', nb_quad),
                    get_quad_field('E', nb_quad))

            getattr(self, f'_T_quad_{nb_quad}').p = self.T_func(*args)
            getattr(self, f'_dT_drho_quad_{nb_quad}').p = self.T_grad_rho(*args)
            getattr(self, f'_dT_djx_quad_{nb_quad}').p = self.T_grad_jx(*args)
            getattr(self, f'_dT_dE_quad_{nb_quad}').p = self.T_grad_E(*args)
            getattr(self, f'_S_quad_{nb_quad}').p = self.S_wall(*args, dh_dx=None)
            getattr(self, f'_dS_drho_quad_{nb_quad}').p = self.S_grad_rho(*args, dh_dx=None)
            getattr(self, f'_dS_djx_quad_{nb_quad}').p = self.S_grad_jx(*args, dh_dx=None)
            getattr(self, f'_dS_dE_quad_{nb_quad}').p = self.S_grad_E(*args, dh_dx=None)

            E_quad = quad_fun(inner_fun(self.energy), nb_quad)
            getattr(self, f'_E_quad_{nb_quad}').p = E_quad.reshape(-1, nb_quad).T

    def T_func(self, rho, jx, E):
        return ((E / rho) - 0.5 * (jx / rho) ** 2) / self.cv

    def T_grad_rho(self, rho, jx, E):
        return (-(E / rho**2) + (jx**2) / (rho**3)) / self.cv

    def T_grad_jx(self, rho, jx, E):
        return (-jx / rho**2) / self.cv

    def T_grad_E(self, rho, jx, E):
        return (1 / (rho * self.cv))

    def k_func(self):
        return self.k

    def S_wall(self, rho, jx, E, dh_dx):
        return - (self.T_func(rho, jx, E) - self.T_wall) * self.alpha_wall

    def S_grad_rho(self, rho, jx, E, dh_dx):
        return - self.T_grad_rho(rho, jx, E) * self.alpha_wall

    def S_grad_jx(self, rho, jx, E, dh_dx):
        return - self.T_grad_jx(rho, jx, E) * self.alpha_wall

    def S_grad_E(self, rho, jx, E, dh_dx):
        return - self.T_grad_E(rho, jx, E) * self.alpha_wall

    def E_quad(self, nb_quad: int) -> NDArray:
        """Return energy quadrature field"""
        return getattr(self, f'_E_quad_{nb_quad}').p

    def T_quad(self, nb_quad: int) -> NDArray:
        """Return temperature quadrature field"""
        return getattr(self, f'_T_quad_{nb_quad}').p

    def dT_drho_quad(self, nb_quad: int) -> NDArray:
        """Return temperature gradient w.r.t. density quadrature field"""
        return getattr(self, f'_dT_drho_quad_{nb_quad}').p

    def dT_djx_quad(self, nb_quad: int) -> NDArray:
        """Return temperature gradient w.r.t. momentum quadrature field"""
        return getattr(self, f'_dT_djx_quad_{nb_quad}').p

    def dT_dE_quad(self, nb_quad: int) -> NDArray:
        """Return temperature gradient w.r.t. energy quadrature field"""
        return getattr(self, f'_dT_dE_quad_{nb_quad}').p

    def S_quad(self, nb_quad: int) -> NDArray:
        """Return entropy quadrature field"""
        return getattr(self, f'_S_quad_{nb_quad}').p

    def dS_drho_quad(self, nb_quad: int) -> NDArray:
        """Return entropy gradient w.r.t. density quadrature field"""
        return getattr(self, f'_dS_drho_quad_{nb_quad}').p

    def dS_djx_quad(self, nb_quad: int) -> NDArray:
        """Return entropy gradient w.r.t. momentum quadrature field"""
        return getattr(self, f'_dS_djx_quad_{nb_quad}').p

    def dS_dE_quad(self, nb_quad: int) -> NDArray:
        """Return entropy gradient w.r.t. energy quadrature field"""
        return getattr(self, f'_dS_dE_quad_{nb_quad}').p
