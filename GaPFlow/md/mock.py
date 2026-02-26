#
# Copyright 2026 Hannes Holey
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

from .base import MolecularDynamics
from ..models.viscous import stress_bottom, stress_top
from ..models.pressure import eos_pressure

import jax.numpy as jnp
import jax.random as jr


class Mock(MolecularDynamics):
    """Mock implementation of an MD runner.

    Instances of this class mimick the behavior of an MD simulations.
    Instead of running an MD simulations, data is generated from implemented
    constitutive laws with added Gaussian noise. During an active learning simulation
    noisy look-up tables are generated, which are used to train a surrogate model.
    """

    name = 'mock'

    _ascii_art: str = r"""
  __  __  ___   ____ _  __
 |  \/  |/ _ \ / ___| |/ /
 | |\/| | | | | |   | ' /
 | |  | | |_| | |___| . \
 |_|  |_|\___/ \____|_|\_\

"""

    def __init__(self, prop, geo, gp):
        """Constructor.

        Parameters
        ----------
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        gp : dict or None, optional
            GP configuration dictionary.
        """

        self.is_mock = True

        self.noise = (gp['press']['obs_stddev'] if gp['press_gp'] else 0.,
                      gp['shear']['obs_stddev'] if gp['shear_gp'] else 0.)

        self.num_worker = 0
        self.geo = geo
        self.prop = prop

        self.params = {}
        self.params.update(prop)

    def build_input_files(self, dataset, location, X):
        self.X = X

    def read_output(self):
        key = jr.key(123)
        key, subkey = jr.split(key)
        noise_p = jr.normal(subkey) * self.noise[0]
        key, subkey = jr.split(subkey)
        noise_s0 = jr.normal(key) * self.noise[1]
        key, subkey = jr.split(subkey)
        noise_s1 = jr.normal(key) * self.noise[1]

        U, V = self.geo["U"], self.geo["V"]
        eta, zeta = self.prop["shear"], self.prop["bulk"]

        X = self.X
        tau_bot = stress_bottom(X[:3], X[3:6], U, V, eta, zeta, X[6]) + noise_s0
        tau_top = stress_top(X[:3], X[3:6], U, V, eta, zeta, X[6]) + noise_s1
        press = eos_pressure(X[0:1], self.prop) + noise_p

        Y = jnp.hstack([press, tau_bot, tau_top]).T
        Ye = jnp.array([
            self.noise[0],  # p
            0., 0., 0.,  # xx, yy, zz
            self.noise[1], self.noise[1], 0.,  # yz, xz, xy
            0., 0., 0.,  # xx, yy, zz
            self.noise[1], self.noise[1], 0.  # yz, xz, xy
        ])

        return Y, Ye
