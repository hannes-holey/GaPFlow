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
from jax import vmap, grad, jit

from .heatflux import heatflux_bot, heatflux_top

import numpy.typing as npt
from typing import Any

NDArray = npt.NDArray[np.floating]


def vvmap(fn, map_list):
    return jit(
        vmap(
            vmap(fn, in_axes=map_list),
            in_axes=map_list
        )
    )


class Energy():
    """Energy model for gap-averaged energy equation.

    Handles total energy, temperature, and wall heat flux computations.
    Includes bulk wall temperatures (Tb_top, Tb_bot) for thermal boundary conditions.

    Parameters
    ----------
    fc : muGrid.GlobalFieldCollection
        Field collection for storing fields.
    energy_spec : dict
        Energy specification containing:
        - k: thermal conductivity [W/(m·K)]
        - cv: specific heat capacity [J/(kg·K)]
        - wall_flux_model: 'Tz_Robin' or 'simple'
        - T_wall: wall temperature (for simple model)
        - h_Robin: effective heat transfer coefficient
        - alpha_wall: wall heat transfer coefficient (for simple model)
    """

    def __init__(self,
                 fc: Any,
                 energy_spec: dict
                 ) -> None:

        self.__field = fc.register_real_field('total_energy')
        self.__temperature = fc.register_real_field('temperature')
        self.__q_wall = fc.register_real_field('q_wall', (2,))

        # Bulk wall temperatures (merged from WallTemperature class)
        self.__Tb_top = fc.register_real_field('Tb_top')
        self.__Tb_bot = fc.register_real_field('Tb_bot')

        self.k = energy_spec['k']
        self.cv = energy_spec['cv']

        self.wall_flux_model = energy_spec['wall_flux_model']
        self.T_wall = energy_spec.get('T_wall', 300.0)
        self.h_Robin = energy_spec['h_Robin']
        self.alpha_wall = energy_spec.get('alpha_wall', 0.0)

        self.__solution = fc.get_real_field('solution')

        # Initialize bulk temperatures with T_wall as default
        self.__Tb_top.p[:] = self.T_wall
        self.__Tb_bot.p[:] = self.T_wall

    @property
    def energy(self) -> NDArray:
        """Total energy field."""
        return self.__field.p

    @energy.setter
    def energy(self, value: NDArray) -> None:
        """Set total energy field."""
        self.__field.p = value

    @property
    def temperature(self) -> NDArray:
        """Temperature field."""
        return self.__temperature.p

    @temperature.setter
    def temperature(self, value: NDArray) -> None:
        """Set temperature field."""
        self.__temperature.p = value

    @property
    def Tb_top(self) -> NDArray:
        """Top wall bulk temperature field."""
        return self.__Tb_top.p

    @Tb_top.setter
    def Tb_top(self, value: NDArray) -> None:
        """Set top wall bulk temperature field."""
        self.__Tb_top.p = value

    @property
    def Tb_bot(self) -> NDArray:
        """Bottom wall bulk temperature field."""
        return self.__Tb_bot.p

    @Tb_bot.setter
    def Tb_bot(self, value: NDArray) -> None:
        """Set bottom wall bulk temperature field."""
        self.__Tb_bot.p = value

    @property
    def solution(self):
        """Return full solution field."""
        return self.__solution.p

    def update_temperature(self) -> None:
        """Update temperature field from current solution."""
        self.temperature = self.T_func(
            self.solution[0], self.solution[1], self.solution[2], self.energy
        )

    def build_grad(self) -> None:
        """Build JIT-compiled gradient functions for energy-related quantities.

        Creates q_wall functions and their gradients with respect to
        rho, jx, jy, and E.
        """
        if self.wall_flux_model == 'Tz_Robin':
            # h, h_Robin, k, cv are constants (None in map_list)
            # eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A are arrays (0 in map_list)
            map_list = (0, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)

            def q_wall_sum(h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A):
                # heatflux_top/bot return W/m^2, divide by h to get W/m^3
                q_top = heatflux_top(h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A)
                q_bot = heatflux_bot(h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A)
                return -(q_top + q_bot) / h

            # Wall heat fluxes (W/m^2)
            self.q_wall_top = vvmap(heatflux_top, map_list)
            self.q_wall_bot = vvmap(heatflux_bot, map_list)
            # Sum converted to W/m^3 for volume source term
            self.q_wall_sum = vvmap(q_wall_sum, map_list)

            # Gradients w.r.t. solution variables
            # argnums: 5=rho, 6=E, 7=jx, 8=jy
            self.q_wall_grad_rho = vvmap(grad(q_wall_sum, argnums=5), map_list)
            self.q_wall_grad_E = vvmap(grad(q_wall_sum, argnums=6), map_list)
            self.q_wall_grad_jx = vvmap(grad(q_wall_sum, argnums=7), map_list)
            self.q_wall_grad_jy = vvmap(grad(q_wall_sum, argnums=8), map_list)

        elif self.wall_flux_model == 'simple':
            self.q_wall_sum = self.q_wall_simple
            # For simple model, gradients would need to be defined separately
            # (placeholder for now)
            self.q_wall_grad_rho = lambda *args: np.zeros_like(args[5])
            self.q_wall_grad_E = lambda *args: np.zeros_like(args[6])
            self.q_wall_grad_jx = lambda *args: np.zeros_like(args[7])
            self.q_wall_grad_jy = lambda *args: np.zeros_like(args[8])

    def T_func(self, rho, jx, jy, E):
        """Compute temperature from solution variables.

        T = (E/rho - 0.5*(ux^2 + uy^2)) / cv
        where ux = jx/rho, uy = jy/rho
        """
        return ((E / rho) - 0.5 * (((jx / rho) ** 2) + ((jy / rho) ** 2))) / self.cv

    def T_grad_rho(self, rho, jx, jy, E):
        """Gradient of temperature w.r.t. density."""
        return (-(E / rho**2) + (jx**2) / (rho**3) + (jy**2) / (rho**3)) / self.cv

    def T_grad_jx(self, rho, jx, jy, E):
        """Gradient of temperature w.r.t. x-momentum."""
        return (-jx / rho**2) / self.cv

    def T_grad_jy(self, rho, jx, jy, E):
        """Gradient of temperature w.r.t. y-momentum."""
        return (-jy / rho**2) / self.cv

    def T_grad_E(self, rho, jx, jy, E):
        """Gradient of temperature w.r.t. total energy."""
        return (1 / (rho * self.cv))

    def k_func(self):
        """Return thermal conductivity."""
        return self.k

    def q_wall_simple(self, h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A):
        """Simple wall heat flux model (Newton cooling)."""
        T = self.T_func(rho, jx, 0.0, E)  # jy=0 for simple 1D
        return - (T - self.T_wall) * self.alpha_wall
