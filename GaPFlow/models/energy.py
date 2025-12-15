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
        - T_0: tuple, initial temperature profile specification
        - E_0: float or None, direct energy specification (alternative to T_0)
    """

    def __init__(self,
                 fc: Any,
                 energy_spec: dict,
                 grid: dict
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
        self.T_wall = energy_spec['T_wall']
        self.h_Robin = energy_spec['h_Robin']
        self.alpha_wall = energy_spec['alpha_wall']

        # Initial condition specification
        self.T_0_spec = energy_spec['T0']

        # Boundary conditions (x-direction)
        self.bc_xW = energy_spec.get('bc_xW', 'P')
        self.bc_xE = energy_spec.get('bc_xE', 'P')
        self.T_bc_xW = energy_spec.get('T_bc_xW', self.T_wall)
        self.T_bc_xE = energy_spec.get('T_bc_xE', self.T_wall)

        self.__solution = fc.get_real_field('solution')
        self.__x = fc.get_real_field('x')
        self.dim = grid['dim']
        self.Lx = grid['Lx']
        self.dx = grid['dx']
        self._flow_periodic_x = all(grid['bc_xW_P']) and all(grid['bc_xE_P'])

        # Initialize bulk temperatures with T_wall as default
        self.__Tb_top.pg[:] = self.T_wall
        self.__Tb_bot.pg[:] = self.T_wall

    @property
    def energy(self) -> NDArray:
        """Total energy field."""
        return self.__field.pg

    @energy.setter
    def energy(self, value: NDArray) -> None:
        """Set total energy field."""
        self.__field.pg = value

    @property
    def temperature(self) -> NDArray:
        """Temperature field."""
        return self.__temperature.pg

    @temperature.setter
    def temperature(self, value: NDArray) -> None:
        """Set temperature field."""
        self.__temperature.pg = value

    @property
    def Tb_top(self) -> NDArray:
        """Top wall bulk temperature field."""
        return self.__Tb_top.pg

    @Tb_top.setter
    def Tb_top(self, value: NDArray) -> None:
        """Set top wall bulk temperature field."""
        self.__Tb_top.pg = value

    @property
    def Tb_bot(self) -> NDArray:
        """Bottom wall bulk temperature field."""
        return self.__Tb_bot.pg

    @Tb_bot.setter
    def Tb_bot(self, value: NDArray) -> None:
        """Set bottom wall bulk temperature field."""
        self.__Tb_bot.pg = value

    @property
    def solution(self):
        """Return full solution field."""
        return self.__solution.pg

    def initialize(self) -> None:
        """
        Initialize energy field from T_0 specification.
        """
        # Check dimensionality for non-uniform profiles
        if self.T_0_spec[0] in ('half_sine', 'block') and self.dim != 1:
            raise ValueError(
                f"Temperature profile '{self.T_0_spec[0]}' is only supported for 1D problems. "
                f"Got dim={self.dim}. Use 'uniform' profile for 2D problems."
            )

        # Check: flow periodic but energy BC non-periodic
        if self._flow_periodic_x and (self.bc_xW != 'P' or self.bc_xE != 'P'):
            raise ValueError(
                "Flow is periodic in x but energy BC are non-periodic."
            )

        # Compute initial energy from initial temperature profile
        ux = self.solution[1] / self.solution[0]
        uy = self.solution[2] / self.solution[0]
        kinetic_energy = 0.5 * (ux**2 + uy**2)
        T_profile = self._get_T_profile(self.__x.pg)
        self.energy[:] = self.solution[0] * (self.cv * T_profile + kinetic_energy)

        # Update temperature field for consistency
        self.update_temperature()

    def _get_T_profile(self, x: NDArray) -> NDArray:
        """
        Compute initial temperature profile based on specification.
        """
        if self.T_0_spec[0] == 'uniform':
            T = self.T_0_spec[1]
            return np.full_like(x, T)

        elif self.T_0_spec[0] == 'half_sine':
            T_min, T_max = self.T_0_spec[1], self.T_0_spec[2]
            # Shift so first interior cell (x=dx/2) maps to x_norm=0
            x_norm = (x - self.dx / 2) / (self.Lx - self.dx)
            return T_min + (T_max - T_min) * np.sin(np.pi * x_norm)

        elif self.T_0_spec[0] == 'block':
            T_min, T_max = self.T_0_spec[1], self.T_0_spec[2]
            # Shift so first interior cell (x=dx/2) maps to x_norm=0
            x_norm = (x - self.dx / 2) / (self.Lx - self.dx)
            return np.where(
                (x_norm >= 0.4) & (x_norm <= 0.6),
                T_max,
                T_min
            )

        else:
            raise ValueError(f"Unknown temperature profile type: {self.T_0_spec[0]}")

    def update_temperature(self) -> None:
        """Update temperature field from current solution."""
        self.temperature = self.T_func(
            self.solution[0], self.solution[1], self.solution[2], self.energy
        )

    def build_grad(self) -> None:
        """Build JIT-compiled gradient functions for energy-related quantities.
        Creates q_wall functions and their gradients with respect to rho, jx, jy, and E.
        """
        if self.wall_flux_model == 'Tz_Robin':
            # h_Robin, k, cv, A are constants (None in map_list)
            # h, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot are arrays (0 in map_list)
            map_list = (0, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)

            def q_wall_sum(h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A):
                q_top = heatflux_top(h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A)
                q_bot = heatflux_bot(h, h_Robin, k, cv, eta, rho, E, jx, jy, U, V, Tb_top, Tb_bot, A)
                return -(q_top + q_bot) / h

            # Wall heat fluxes
            self.q_wall_top = vvmap(heatflux_top, map_list)  # W/m^2
            self.q_wall_bot = vvmap(heatflux_bot, map_list)  # W/m^2
            self.q_wall_sum = vvmap(q_wall_sum, map_list)    # W/m^3

            # Gradients w.r.t. solution variables
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
