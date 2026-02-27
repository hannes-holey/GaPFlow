#
# Copyright 2025 Hannes Holey
#           2026 Christoph Huber
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

# flake8: noqa: W503

import numpy as np
import jax.numpy as jnp
from .viscosity import piezoviscosity, shear_thinning_factor, shear_rate_avg


def stress_bottom(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
                  dqx=None, dqy=None):
    """Viscous stress tensor at the bottom wall.

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd height gradients in x and y direction, respectively.
    U_bot : float
        Bottom wall velocity in x direction.
    V_bot : float
        Bottom wall velocity in y direction.
    U_top : float
        Top wall velocity in x direction.
    V_top : float
        Top wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Ls_bot : float
        Slip length at bottom wall
    Ls_top : float
        Slip length at top wall
    dqx : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    dqy : numpy.ndarray, optional
        Gradient (y) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)

    Returns
    -------
    numpy.ndarray
        Viscous stress tensor components in Voigt ordering
    """
    if dqx is None:
        dqx = np.zeros_like(q)
    if dqy is None:
        dqy = np.zeros_like(q)

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    tau = np.zeros((6, *q.shape[1:]))

    D = 4 * Ls_bot * h[0] + 4 * Ls_top * h[0] + h[0] ** 2 + 12 * Ls_bot * Ls_top

    # tau_xz_bot
    tau[4] = -(2 * eta * (
                6 * Ls_top * U_bot * q[0]
                - 3 * h[0] * q[1]
                - 6 * Ls_top * q[1]
                + 2 * U_bot * h[0] * q[0]
                + U_top * h[0] * q[0]
            )) / (q[0] * D)
    # tau_yz_bot
    tau[3] = -(2 * eta * (
                6 * Ls_top * V_bot * q[0]
                - 3 * h[0] * q[2]
                - 6 * Ls_top * q[2]
                + 2 * V_bot * h[0] * q[0]
                + V_top * h[0] * q[0]
            )) / (q[0] * D)

    # tau_xx_bot
    tau[0] = (2 * Ls_bot * (
                6 * dqx[1] * eta * h[0] ** 3 * q[0]
                - 3 * dqx[0] * h[0] ** 3 * q[1] * v2
                - 3 * dqy[0] * h[0] ** 3 * q[2] * v2
                - 6 * dqx[0] * eta * h[0] ** 3 * q[1]
                + 3 * dqx[1] * h[0] ** 3 * v2 * q[0]
                + 3 * dqy[2] * h[0] ** 3 * v2 * q[0]
                + 24 * Ls_top ** 2 * U_bot * h[1] * v2 * q[0] ** 2
                + 24 * Ls_top ** 2 * V_bot * h[2] * v2 * q[0] ** 2
                + 4 * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 2 * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 2 * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 2 * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                - 144 * Ls_bot * Ls_top ** 2 * dqx[0] * eta * q[1]
                - 72 * Ls_bot * Ls_top ** 2 * dqx[0] * q[1] * v2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[0] * q[2] * v2
                + 144 * Ls_bot * Ls_top ** 2 * dqx[1] * eta * q[0]
                + 72 * Ls_bot * Ls_top ** 2 * dqx[1] * v2 * q[0]
                + 72 * Ls_bot * Ls_top ** 2 * dqy[2] * v2 * q[0]
                - 24 * Ls_bot * dqx[0] * eta * h[0] ** 2 * q[1]
                - 36 * Ls_top * dqx[0] * eta * h[0] ** 2 * q[1]
                - 48 * Ls_top ** 2 * dqx[0] * eta * h[0] * q[1]
                - 12 * Ls_bot * dqx[0] * h[0] ** 2 * q[1] * v2
                - 18 * Ls_top * dqx[0] * h[0] ** 2 * q[1] * v2
                - 24 * Ls_top ** 2 * dqx[0] * h[0] * q[1] * v2
                - 12 * Ls_bot * dqy[0] * h[0] ** 2 * q[2] * v2
                - 18 * Ls_top * dqy[0] * h[0] ** 2 * q[2] * v2
                - 24 * Ls_top ** 2 * dqy[0] * h[0] * q[2] * v2
                + 24 * Ls_bot * dqx[1] * eta * h[0] ** 2 * q[0]
                + 36 * Ls_top * dqx[1] * eta * h[0] ** 2 * q[0]
                + 48 * Ls_top ** 2 * dqx[1] * eta * h[0] * q[0]
                - 48 * Ls_top ** 2 * h[1] * eta * q[1] * q[0]
                + 12 * Ls_bot * dqx[1] * h[0] ** 2 * v2 * q[0]
                + 12 * Ls_bot * dqy[2] * h[0] ** 2 * v2 * q[0]
                + 18 * Ls_top * dqx[1] * h[0] ** 2 * v2 * q[0]
                + 24 * Ls_top ** 2 * dqx[1] * h[0] * v2 * q[0]
                + 18 * Ls_top * dqy[2] * h[0] ** 2 * v2 * q[0]
                + 24 * Ls_top ** 2 * dqy[2] * h[0] * v2 * q[0]
                - 24 * Ls_top ** 2 * h[1] * q[1] * v2 * q[0]
                - 24 * Ls_top ** 2 * h[2] * q[2] * v2 * q[0]
                - 6 * h[1] * eta * h[0] ** 2 * q[1] * q[0]
                - 3 * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 3 * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                + 48 * Ls_top ** 2 * U_bot * h[1] * eta * q[0] ** 2
                + 24 * Ls_top * U_bot * h[1] * eta * h[0] * q[0] ** 2
                + 12 * Ls_top * U_bot * h[1] * h[0] * v2 * q[0] ** 2
                + 12 * Ls_top * V_bot * h[2] * h[0] * v2 * q[0] ** 2
                - 120 * Ls_bot * Ls_top * dqx[0] * eta * h[0] * q[1]
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] * q[1] * v2
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] * q[2] * v2
                + 120 * Ls_bot * Ls_top * dqx[1] * eta * h[0] * q[0]
                + 24 * Ls_bot * Ls_top * h[1] * eta * q[1] * q[0]
                + 60 * Ls_bot * Ls_top * dqx[1] * h[0] * v2 * q[0]
                + 60 * Ls_bot * Ls_top * dqy[2] * h[0] * v2 * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * q[1] * v2 * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * q[2] * v2 * q[0]
                - 24 * Ls_top * h[1] * eta * h[0] * q[1] * q[0]
                - 12 * Ls_top * h[1] * h[0] * q[1] * v2 * q[0]
                - 12 * Ls_top * h[2] * h[0] * q[2] * v2 * q[0]
                - 24 * Ls_bot * Ls_top * U_top * h[1] * eta * q[0] ** 2
                - 12 * Ls_bot * Ls_top * U_top * h[1] * v2 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_top * h[2] * v2 * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    # tau_yy_bot
    tau[1] = (2 * Ls_bot * (
                6 * dqy[2] * eta * h[0] ** 3 * q[0]
                - 3 * dqx[0] * h[0] ** 3 * q[1] * v2
                - 3 * dqy[0] * h[0] ** 3 * q[2] * v2
                - 6 * dqy[0] * eta * h[0] ** 3 * q[2]
                + 3 * dqx[1] * h[0] ** 3 * v2 * q[0]
                + 3 * dqy[2] * h[0] ** 3 * v2 * q[0]
                + 24 * Ls_top ** 2 * U_bot * h[1] * v2 * q[0] ** 2
                + 24 * Ls_top ** 2 * V_bot * h[2] * v2 * q[0] ** 2
                + 4 * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 2 * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 2 * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 2 * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                - 144 * Ls_bot * Ls_top ** 2 * dqy[0] * eta * q[2]
                - 72 * Ls_bot * Ls_top ** 2 * dqx[0] * q[1] * v2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[0] * q[2] * v2
                + 144 * Ls_bot * Ls_top ** 2 * dqy[2] * eta * q[0]
                + 72 * Ls_bot * Ls_top ** 2 * dqx[1] * v2 * q[0]
                + 72 * Ls_bot * Ls_top ** 2 * dqy[2] * v2 * q[0]
                - 24 * Ls_bot * dqy[0] * eta * h[0] ** 2 * q[2]
                - 36 * Ls_top * dqy[0] * eta * h[0] ** 2 * q[2]
                - 48 * Ls_top ** 2 * dqy[0] * eta * h[0] * q[2]
                - 12 * Ls_bot * dqx[0] * h[0] ** 2 * q[1] * v2
                - 18 * Ls_top * dqx[0] * h[0] ** 2 * q[1] * v2
                - 24 * Ls_top ** 2 * dqx[0] * h[0] * q[1] * v2
                - 12 * Ls_bot * dqy[0] * h[0] ** 2 * q[2] * v2
                - 18 * Ls_top * dqy[0] * h[0] ** 2 * q[2] * v2
                - 24 * Ls_top ** 2 * dqy[0] * h[0] * q[2] * v2
                + 24 * Ls_bot * dqy[2] * eta * h[0] ** 2 * q[0]
                + 36 * Ls_top * dqy[2] * eta * h[0] ** 2 * q[0]
                + 48 * Ls_top ** 2 * dqy[2] * eta * h[0] * q[0]
                - 48 * Ls_top ** 2 * h[2] * eta * q[2] * q[0]
                + 12 * Ls_bot * dqx[1] * h[0] ** 2 * v2 * q[0]
                + 12 * Ls_bot * dqy[2] * h[0] ** 2 * v2 * q[0]
                + 18 * Ls_top * dqx[1] * h[0] ** 2 * v2 * q[0]
                + 24 * Ls_top ** 2 * dqx[1] * h[0] * v2 * q[0]
                + 18 * Ls_top * dqy[2] * h[0] ** 2 * v2 * q[0]
                + 24 * Ls_top ** 2 * dqy[2] * h[0] * v2 * q[0]
                - 24 * Ls_top ** 2 * h[1] * q[1] * v2 * q[0]
                - 24 * Ls_top ** 2 * h[2] * q[2] * v2 * q[0]
                - 6 * h[2] * eta * h[0] ** 2 * q[2] * q[0]
                - 3 * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 3 * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                + 48 * Ls_top ** 2 * V_bot * h[2] * eta * q[0] ** 2
                + 24 * Ls_top * V_bot * h[2] * eta * h[0] * q[0] ** 2
                + 12 * Ls_top * U_bot * h[1] * h[0] * v2 * q[0] ** 2
                + 12 * Ls_top * V_bot * h[2] * h[0] * v2 * q[0] ** 2
                - 120 * Ls_bot * Ls_top * dqy[0] * eta * h[0] * q[2]
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] * q[1] * v2
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] * q[2] * v2
                + 120 * Ls_bot * Ls_top * dqy[2] * eta * h[0] * q[0]
                + 24 * Ls_bot * Ls_top * h[2] * eta * q[2] * q[0]
                + 60 * Ls_bot * Ls_top * dqx[1] * h[0] * v2 * q[0]
                + 60 * Ls_bot * Ls_top * dqy[2] * h[0] * v2 * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * q[1] * v2 * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * q[2] * v2 * q[0]
                - 24 * Ls_top * h[2] * eta * h[0] * q[2] * q[0]
                - 12 * Ls_top * h[1] * h[0] * q[1] * v2 * q[0]
                - 12 * Ls_top * h[2] * h[0] * q[2] * v2 * q[0]
                - 24 * Ls_bot * Ls_top * V_top * h[2] * eta * q[0] ** 2
                - 12 * Ls_bot * Ls_top * U_top * h[1] * v2 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_top * h[2] * v2 * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    # tau_zz_bot
    tau[2] = (2 * Ls_bot * v2 * (
                3 * dqx[1] * h[0] ** 3 * q[0]
                - 3 * dqy[0] * h[0] ** 3 * q[2]
                - 3 * dqx[0] * h[0] ** 3 * q[1]
                + 3 * dqy[2] * h[0] ** 3 * q[0]
                - 72 * Ls_bot * Ls_top ** 2 * dqx[0] * q[1]
                - 72 * Ls_bot * Ls_top ** 2 * dqy[0] * q[2]
                + 72 * Ls_bot * Ls_top ** 2 * dqx[1] * q[0]
                + 72 * Ls_bot * Ls_top ** 2 * dqy[2] * q[0]
                - 12 * Ls_bot * dqx[0] * h[0] ** 2 * q[1]
                - 18 * Ls_top * dqx[0] * h[0] ** 2 * q[1]
                - 24 * Ls_top ** 2 * dqx[0] * h[0] * q[1]
                - 12 * Ls_bot * dqy[0] * h[0] ** 2 * q[2]
                - 18 * Ls_top * dqy[0] * h[0] ** 2 * q[2]
                - 24 * Ls_top ** 2 * dqy[0] * h[0] * q[2]
                + 12 * Ls_bot * dqx[1] * h[0] ** 2 * q[0]
                + 12 * Ls_bot * dqy[2] * h[0] ** 2 * q[0]
                + 18 * Ls_top * dqx[1] * h[0] ** 2 * q[0]
                + 24 * Ls_top ** 2 * dqx[1] * h[0] * q[0]
                + 18 * Ls_top * dqy[2] * h[0] ** 2 * q[0]
                + 24 * Ls_top ** 2 * dqy[2] * h[0] * q[0]
                - 24 * Ls_top ** 2 * h[1] * q[1] * q[0]
                - 24 * Ls_top ** 2 * h[2] * q[2] * q[0]
                - 3 * h[1] * h[0] ** 2 * q[1] * q[0]
                - 3 * h[2] * h[0] ** 2 * q[2] * q[0]
                + 24 * Ls_top ** 2 * U_bot * h[1] * q[0] ** 2
                + 24 * Ls_top ** 2 * V_bot * h[2] * q[0] ** 2
                + 2 * U_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + U_top * h[1] * h[0] ** 2 * q[0] ** 2
                + 2 * V_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + V_top * h[2] * h[0] ** 2 * q[0] ** 2
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] * q[1]
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] * q[2]
                + 60 * Ls_bot * Ls_top * dqx[1] * h[0] * q[0]
                + 60 * Ls_bot * Ls_top * dqy[2] * h[0] * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * q[1] * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * q[2] * q[0]
                - 12 * Ls_top * h[1] * h[0] * q[1] * q[0]
                - 12 * Ls_top * h[2] * h[0] * q[2] * q[0]
                - 12 * Ls_bot * Ls_top * U_top * h[1] * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_top * h[2] * q[0] ** 2
                + 12 * Ls_top * U_bot * h[1] * h[0] * q[0] ** 2
                + 12 * Ls_top * V_bot * h[2] * h[0] * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    # tau_xy_bot
    tau[5] = (2 * Ls_bot * eta * (
                3 * dqy[1] * h[0] ** 3 * q[0]
                - 3 * dqx[0] * h[0] ** 3 * q[2]
                - 3 * dqy[0] * h[0] ** 3 * q[1]
                + 3 * dqx[2] * h[0] ** 3 * q[0]
                - 72 * Ls_bot * Ls_top ** 2 * dqy[0] * q[1]
                - 72 * Ls_bot * Ls_top ** 2 * dqx[0] * q[2]
                + 72 * Ls_bot * Ls_top ** 2 * dqy[1] * q[0]
                + 72 * Ls_bot * Ls_top ** 2 * dqx[2] * q[0]
                - 12 * Ls_bot * dqy[0] * h[0] ** 2 * q[1]
                - 18 * Ls_top * dqy[0] * h[0] ** 2 * q[1]
                - 24 * Ls_top ** 2 * dqy[0] * h[0] * q[1]
                - 12 * Ls_bot * dqx[0] * h[0] ** 2 * q[2]
                - 18 * Ls_top * dqx[0] * h[0] ** 2 * q[2]
                - 24 * Ls_top ** 2 * dqx[0] * h[0] * q[2]
                + 12 * Ls_bot * dqy[1] * h[0] ** 2 * q[0]
                + 12 * Ls_bot * dqx[2] * h[0] ** 2 * q[0]
                + 18 * Ls_top * dqy[1] * h[0] ** 2 * q[0]
                + 24 * Ls_top ** 2 * dqy[1] * h[0] * q[0]
                + 18 * Ls_top * dqx[2] * h[0] ** 2 * q[0]
                + 24 * Ls_top ** 2 * dqx[2] * h[0] * q[0]
                - 24 * Ls_top ** 2 * h[2] * q[1] * q[0]
                - 24 * Ls_top ** 2 * h[1] * q[2] * q[0]
                - 3 * h[2] * h[0] ** 2 * q[1] * q[0]
                - 3 * h[1] * h[0] ** 2 * q[2] * q[0]
                + 24 * Ls_top ** 2 * U_bot * h[2] * q[0] ** 2
                + 24 * Ls_top ** 2 * V_bot * h[1] * q[0] ** 2
                + 2 * U_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + U_top * h[2] * h[0] ** 2 * q[0] ** 2
                + 2 * V_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + V_top * h[1] * h[0] ** 2 * q[0] ** 2
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] * q[1]
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] * q[2]
                + 60 * Ls_bot * Ls_top * dqy[1] * h[0] * q[0]
                + 60 * Ls_bot * Ls_top * dqx[2] * h[0] * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * q[1] * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * q[2] * q[0]
                - 12 * Ls_top * h[2] * h[0] * q[1] * q[0]
                - 12 * Ls_top * h[1] * h[0] * q[2] * q[0]
                - 12 * Ls_bot * Ls_top * U_top * h[2] * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_top * h[1] * q[0] ** 2
                + 12 * Ls_top * U_bot * h[2] * h[0] * q[0] ** 2
                + 12 * Ls_top * V_bot * h[1] * h[0] * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    return tau


def stress_top(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
               dqx=None, dqy=None):
    """Viscous stress tensor at the top wall.

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd height gradients in x and y direction, respectively.
    U_bot : float
        Bottom wall velocity in x direction.
    V_bot : float
        Bottom wall velocity in y direction.
    U_top : float
        Top wall velocity in x direction.
    V_top : float
        Top wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Ls_bot : float
        Slip length at bottom wall
    Ls_top : float
        Slip length at top wall
    dqx : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    dqy : numpy.ndarray, optional
        Gradient (y) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)

    Returns
    -------
    numpy.ndarray
        Viscous stress tensor components in Voigt ordering
    """
    if dqx is None:
        dqx = np.zeros_like(q)
    if dqy is None:
        dqy = np.zeros_like(q)

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    tau = np.zeros((6, *q.shape[1:]))

    D = 4 * Ls_bot * h[0] + 4 * Ls_top * h[0] + h[0] ** 2 + 12 * Ls_bot * Ls_top

    # tau_xz_top
    tau[4] = -eta * ((2 * (
                    6 * Ls_top * U_bot * q[0]
                    - 3 * h[0] * q[1]
                    - 6 * Ls_top * q[1]
                    + 2 * U_bot * h[0] * q[0]
                    + U_top * h[0] * q[0]
                )) / (q[0] * D) - (6 * (
                    2 * Ls_top * U_bot * q[0]
                    - 2 * Ls_top * q[1]
                    - 2 * h[0] * q[1]
                    - 2 * Ls_bot * q[1]
                    + 2 * Ls_bot * U_top * q[0]
                    + U_bot * h[0] * q[0]
                    + U_top * h[0] * q[0]
                )) / (q[0] * D))
    # tau_yz_top
    tau[3] = -eta * ((2 * (
                    6 * Ls_top * V_bot * q[0]
                    - 3 * h[0] * q[2]
                    - 6 * Ls_top * q[2]
                    + 2 * V_bot * h[0] * q[0]
                    + V_top * h[0] * q[0]
                )) / (q[0] * D) - (6 * (
                    2 * Ls_top * V_bot * q[0]
                    - 2 * Ls_top * q[2]
                    - 2 * h[0] * q[2]
                    - 2 * Ls_bot * q[2]
                    + 2 * Ls_bot * V_top * q[0]
                    + V_bot * h[0] * q[0]
                    + V_top * h[0] * q[0]
                )) / (q[0] * D))

    # tau_xx_top
    tau[0] = -(2 * (
                24 * Ls_top ** 2 * dqx[0] * eta * h[0] ** 2 * q[1]
                + 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2
                - 24 * Ls_top ** 2 * dqx[1] * eta * h[0] ** 2 * q[0]
                - 12 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0]
                - 12 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0]
                + 2 * U_bot * h[1] * eta * h[0] ** 3 * q[0] ** 2
                + 4 * U_top * h[1] * eta * h[0] ** 3 * q[0] ** 2
                + U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2
                + 2 * U_top * h[1] * h[0] ** 3 * v2 * q[0] ** 2
                + V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2
                + 2 * V_top * h[2] * h[0] ** 3 * v2 * q[0] ** 2
                + 6 * Ls_top * dqx[0] * eta * h[0] ** 3 * q[1]
                + 3 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2
                + 3 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2
                - 6 * Ls_top * dqx[1] * eta * h[0] ** 3 * q[0]
                - 3 * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0]
                - 3 * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0]
                - 6 * h[1] * eta * h[0] ** 3 * q[1] * q[0]
                - 3 * h[1] * h[0] ** 3 * q[1] * v2 * q[0]
                - 3 * h[2] * h[0] ** 3 * q[2] * v2 * q[0]
                + 144 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * eta * q[1]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * q[1] * v2
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * q[2] * v2
                - 144 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * eta * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * v2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * v2 * q[0]
                - 18 * Ls_bot * Ls_top * dqx[1] * h[0] ** 2 * v2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] * v2 * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] * v2 * q[0]
                - 18 * Ls_bot * Ls_top * dqy[2] * h[0] ** 2 * v2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] * v2 * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] * v2 * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[1] * q[1] * v2 * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[1] * q[1] * v2 * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[2] * q[2] * v2 * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[2] * q[2] * v2 * q[0]
                - 36 * Ls_bot * h[1] * eta * h[0] ** 2 * q[1] * q[0]
                - 48 * Ls_bot ** 2 * h[1] * eta * h[0] * q[1] * q[0]
                - 18 * Ls_top * h[1] * eta * h[0] ** 2 * q[1] * q[0]
                - 18 * Ls_bot * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[1] * v2 * q[0]
                - 9 * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 18 * Ls_bot * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[2] * v2 * q[0]
                - 9 * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * eta * q[0] ** 2
                + 96 * Ls_bot ** 2 * Ls_top * U_top * h[1] * eta * q[0] ** 2
                + 12 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * v2 * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * U_top * h[1] * v2 * q[0] ** 2
                + 12 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * v2 * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * V_top * h[2] * v2 * q[0] ** 2
                + 8 * Ls_bot * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 6 * Ls_top * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 28 * Ls_bot * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 48 * Ls_bot ** 2 * U_top * h[1] * eta * h[0] * q[0] ** 2
                + 12 * Ls_top * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 4 * Ls_bot * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 3 * Ls_top * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 14 * Ls_bot * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * U_top * h[1] * h[0] * v2 * q[0] ** 2
                + 6 * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 4 * Ls_bot * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 3 * Ls_top * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 14 * Ls_bot * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * V_top * h[2] * h[0] * v2 * q[0] ** 2
                + 6 * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top * dqx[0] * eta * h[0] ** 2 * q[1]
                + 120 * Ls_bot * Ls_top ** 2 * dqx[0] * eta * h[0] * q[1]
                + 48 * Ls_bot ** 2 * Ls_top * dqx[0] * eta * h[0] * q[1]
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[1] * v2
                + 60 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[1] * v2
                + 24 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[1] * v2
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[2] * v2
                + 60 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[2] * v2
                + 24 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[2] * v2
                - 36 * Ls_bot * Ls_top * dqx[1] * eta * h[0] ** 2 * q[0]
                - 120 * Ls_bot * Ls_top ** 2 * dqx[1] * eta * h[0] * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * dqx[1] * eta * h[0] * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * h[1] * eta * q[1] * q[0]
                - 96 * Ls_bot ** 2 * Ls_top * h[1] * eta * q[1] * q[0]
                - 96 * Ls_bot * Ls_top * h[1] * eta * h[0] * q[1] * q[0]
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[1] * v2 * q[0]
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[2] * v2 * q[0]
                + 24 * Ls_bot * Ls_top * U_bot * h[1] * eta * h[0] * q[0] ** 2
                + 72 * Ls_bot * Ls_top * U_top * h[1] * eta * h[0] * q[0] ** 2
                + 12 * Ls_bot * Ls_top * U_bot * h[1] * h[0] * v2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top * U_top * h[1] * h[0] * v2 * q[0] ** 2
                + 12 * Ls_bot * Ls_top * V_bot * h[2] * h[0] * v2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top * V_top * h[2] * h[0] * v2 * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    # tau_yy_top
    tau[1] = -(2 * (
                24 * Ls_top ** 2 * dqy[0] * eta * h[0] ** 2 * q[2]
                + 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2
                - 24 * Ls_top ** 2 * dqy[2] * eta * h[0] ** 2 * q[0]
                - 12 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0]
                - 12 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0]
                + 2 * V_bot * h[2] * eta * h[0] ** 3 * q[0] ** 2
                + 4 * V_top * h[2] * eta * h[0] ** 3 * q[0] ** 2
                + U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2
                + 2 * U_top * h[1] * h[0] ** 3 * v2 * q[0] ** 2
                + V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2
                + 2 * V_top * h[2] * h[0] ** 3 * v2 * q[0] ** 2
                + 6 * Ls_top * dqy[0] * eta * h[0] ** 3 * q[2]
                + 3 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2
                + 3 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2
                - 6 * Ls_top * dqy[2] * eta * h[0] ** 3 * q[0]
                - 3 * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0]
                - 3 * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0]
                - 6 * h[2] * eta * h[0] ** 3 * q[2] * q[0]
                - 3 * h[1] * h[0] ** 3 * q[1] * v2 * q[0]
                - 3 * h[2] * h[0] ** 3 * q[2] * v2 * q[0]
                + 144 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * eta * q[2]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * q[1] * v2
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * q[2] * v2
                - 144 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * eta * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * v2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * v2 * q[0]
                - 18 * Ls_bot * Ls_top * dqx[1] * h[0] ** 2 * v2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] * v2 * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] * v2 * q[0]
                - 18 * Ls_bot * Ls_top * dqy[2] * h[0] ** 2 * v2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] * v2 * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] * v2 * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[1] * q[1] * v2 * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[1] * q[1] * v2 * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[2] * q[2] * v2 * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[2] * q[2] * v2 * q[0]
                - 36 * Ls_bot * h[2] * eta * h[0] ** 2 * q[2] * q[0]
                - 48 * Ls_bot ** 2 * h[2] * eta * h[0] * q[2] * q[0]
                - 18 * Ls_top * h[2] * eta * h[0] ** 2 * q[2] * q[0]
                - 18 * Ls_bot * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[1] * v2 * q[0]
                - 9 * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 18 * Ls_bot * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[2] * v2 * q[0]
                - 9 * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * eta * q[0] ** 2
                + 96 * Ls_bot ** 2 * Ls_top * V_top * h[2] * eta * q[0] ** 2
                + 12 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * v2 * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * U_top * h[1] * v2 * q[0] ** 2
                + 12 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * v2 * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * V_top * h[2] * v2 * q[0] ** 2
                + 8 * Ls_bot * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 6 * Ls_top * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 28 * Ls_bot * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 48 * Ls_bot ** 2 * V_top * h[2] * eta * h[0] * q[0] ** 2
                + 12 * Ls_top * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 4 * Ls_bot * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 3 * Ls_top * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 14 * Ls_bot * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * U_top * h[1] * h[0] * v2 * q[0] ** 2
                + 6 * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 4 * Ls_bot * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 3 * Ls_top * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 14 * Ls_bot * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * V_top * h[2] * h[0] * v2 * q[0] ** 2
                + 6 * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top * dqy[0] * eta * h[0] ** 2 * q[2]
                + 120 * Ls_bot * Ls_top ** 2 * dqy[0] * eta * h[0] * q[2]
                + 48 * Ls_bot ** 2 * Ls_top * dqy[0] * eta * h[0] * q[2]
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[1] * v2
                + 60 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[1] * v2
                + 24 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[1] * v2
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[2] * v2
                + 60 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[2] * v2
                + 24 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[2] * v2
                - 36 * Ls_bot * Ls_top * dqy[2] * eta * h[0] ** 2 * q[0]
                - 120 * Ls_bot * Ls_top ** 2 * dqy[2] * eta * h[0] * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * dqy[2] * eta * h[0] * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * h[2] * eta * q[2] * q[0]
                - 96 * Ls_bot ** 2 * Ls_top * h[2] * eta * q[2] * q[0]
                - 96 * Ls_bot * Ls_top * h[2] * eta * h[0] * q[2] * q[0]
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[1] * v2 * q[0]
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[2] * v2 * q[0]
                + 24 * Ls_bot * Ls_top * V_bot * h[2] * eta * h[0] * q[0] ** 2
                + 72 * Ls_bot * Ls_top * V_top * h[2] * eta * h[0] * q[0] ** 2
                + 12 * Ls_bot * Ls_top * U_bot * h[1] * h[0] * v2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top * U_top * h[1] * h[0] * v2 * q[0] ** 2
                + 12 * Ls_bot * Ls_top * V_bot * h[2] * h[0] * v2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top * V_top * h[2] * h[0] * v2 * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    # tau_zz_top
    tau[2] = -(2 * v2 * (
                3 * Ls_top * dqx[0] * h[0] ** 3 * q[1]
                + 3 * Ls_top * dqy[0] * h[0] ** 3 * q[2]
                - 3 * Ls_top * dqx[1] * h[0] ** 3 * q[0]
                - 3 * Ls_top * dqy[2] * h[0] ** 3 * q[0]
                - 3 * h[1] * h[0] ** 3 * q[1] * q[0]
                - 3 * h[2] * h[0] ** 3 * q[2] * q[0]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * q[1]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * q[2]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * q[0]
                + 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1]
                + 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2]
                - 12 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * q[0]
                - 12 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * q[0]
                + U_bot * h[1] * h[0] ** 3 * q[0] ** 2
                + 2 * U_top * h[1] * h[0] ** 3 * q[0] ** 2
                + V_bot * h[2] * h[0] ** 3 * q[0] ** 2
                + 2 * V_top * h[2] * h[0] ** 3 * q[0] ** 2
                + 14 * Ls_bot * V_top * h[2] * h[0] ** 2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * V_top * h[2] * h[0] * q[0] ** 2
                + 6 * Ls_top * V_top * h[2] * h[0] ** 2 * q[0] ** 2
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[1]
                + 60 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[1]
                + 24 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[1]
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[2]
                + 60 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[2]
                + 24 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[2]
                - 18 * Ls_bot * Ls_top * dqx[1] * h[0] ** 2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] * q[0]
                - 18 * Ls_bot * Ls_top * dqy[2] * h[0] ** 2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[1] * q[1] * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[1] * q[1] * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[2] * q[2] * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[2] * q[2] * q[0]
                - 18 * Ls_bot * h[1] * h[0] ** 2 * q[1] * q[0]
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[1] * q[0]
                - 9 * Ls_top * h[1] * h[0] ** 2 * q[1] * q[0]
                - 18 * Ls_bot * h[2] * h[0] ** 2 * q[2] * q[0]
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[2] * q[0]
                - 9 * Ls_top * h[2] * h[0] ** 2 * q[2] * q[0]
                + 12 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * U_top * h[1] * q[0] ** 2
                + 12 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * V_top * h[2] * q[0] ** 2
                + 4 * Ls_bot * U_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + 3 * Ls_top * U_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + 14 * Ls_bot * U_top * h[1] * h[0] ** 2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * U_top * h[1] * h[0] * q[0] ** 2
                + 6 * Ls_top * U_top * h[1] * h[0] ** 2 * q[0] ** 2
                + 4 * Ls_bot * V_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + 3 * Ls_top * V_bot * h[2] * h[0] ** 2 * q[0] ** 2
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[1] * q[0]
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[2] * q[0]
                + 12 * Ls_bot * Ls_top * U_bot * h[1] * h[0] * q[0] ** 2
                + 36 * Ls_bot * Ls_top * U_top * h[1] * h[0] * q[0] ** 2
                + 12 * Ls_bot * Ls_top * V_bot * h[2] * h[0] * q[0] ** 2
                + 36 * Ls_bot * Ls_top * V_top * h[2] * h[0] * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    # tau_xy_top
    tau[5] = -(2 * eta * (
                3 * Ls_top * dqy[0] * h[0] ** 3 * q[1]
                + 3 * Ls_top * dqx[0] * h[0] ** 3 * q[2]
                - 3 * Ls_top * dqy[1] * h[0] ** 3 * q[0]
                - 3 * Ls_top * dqx[2] * h[0] ** 3 * q[0]
                - 3 * h[2] * h[0] ** 3 * q[1] * q[0]
                - 3 * h[1] * h[0] ** 3 * q[2] * q[0]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * q[1]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * q[2]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[1] * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[2] * q[0]
                + 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[1]
                + 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[2]
                - 12 * Ls_top ** 2 * dqy[1] * h[0] ** 2 * q[0]
                - 12 * Ls_top ** 2 * dqx[2] * h[0] ** 2 * q[0]
                + U_bot * h[2] * h[0] ** 3 * q[0] ** 2
                + 2 * U_top * h[2] * h[0] ** 3 * q[0] ** 2
                + V_bot * h[1] * h[0] ** 3 * q[0] ** 2
                + 2 * V_top * h[1] * h[0] ** 3 * q[0] ** 2
                + 14 * Ls_bot * V_top * h[1] * h[0] ** 2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * V_top * h[1] * h[0] * q[0] ** 2
                + 6 * Ls_top * V_top * h[1] * h[0] ** 2 * q[0] ** 2
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[1]
                + 60 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[1]
                + 24 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[1]
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[2]
                + 60 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[2]
                + 24 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[2]
                - 18 * Ls_bot * Ls_top * dqy[1] * h[0] ** 2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqy[1] * h[0] * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqy[1] * h[0] * q[0]
                - 18 * Ls_bot * Ls_top * dqx[2] * h[0] ** 2 * q[0]
                - 60 * Ls_bot * Ls_top ** 2 * dqx[2] * h[0] * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * dqx[2] * h[0] * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[2] * q[1] * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[2] * q[1] * q[0]
                - 12 * Ls_bot * Ls_top ** 2 * h[1] * q[2] * q[0]
                - 48 * Ls_bot ** 2 * Ls_top * h[1] * q[2] * q[0]
                - 18 * Ls_bot * h[2] * h[0] ** 2 * q[1] * q[0]
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[1] * q[0]
                - 9 * Ls_top * h[2] * h[0] ** 2 * q[1] * q[0]
                - 18 * Ls_bot * h[1] * h[0] ** 2 * q[2] * q[0]
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[2] * q[0]
                - 9 * Ls_top * h[1] * h[0] ** 2 * q[2] * q[0]
                + 12 * Ls_bot * Ls_top ** 2 * U_bot * h[2] * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * U_top * h[2] * q[0] ** 2
                + 12 * Ls_bot * Ls_top ** 2 * V_bot * h[1] * q[0] ** 2
                + 48 * Ls_bot ** 2 * Ls_top * V_top * h[1] * q[0] ** 2
                + 4 * Ls_bot * U_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + 3 * Ls_top * U_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + 14 * Ls_bot * U_top * h[2] * h[0] ** 2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * U_top * h[2] * h[0] * q[0] ** 2
                + 6 * Ls_top * U_top * h[2] * h[0] ** 2 * q[0] ** 2
                + 4 * Ls_bot * V_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + 3 * Ls_top * V_bot * h[1] * h[0] ** 2 * q[0] ** 2
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[1] * q[0]
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[2] * q[0]
                + 12 * Ls_bot * Ls_top * U_bot * h[2] * h[0] * q[0] ** 2
                + 36 * Ls_bot * Ls_top * U_top * h[2] * h[0] * q[0] ** 2
                + 12 * Ls_bot * Ls_top * V_bot * h[1] * h[0] * q[0] ** 2
                + 36 * Ls_bot * Ls_top * V_top * h[1] * h[0] * q[0] ** 2
            ))/(q[0] ** 2 * D ** 2)

    return tau


def stress_avg(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
               dqx=None, dqy=None):
    """Gap-averaged viscous stress tensor (normal and in-plane shear components).

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd height gradients in x and y direction, respectively.
    U_bot : float
        Bottom wall velocity in x direction.
    V_bot : float
        Bottom wall velocity in y direction.
    U_top : float
        Top wall velocity in x direction.
    V_top : float
        Top wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Ls_bot : float
        Slip length at bottom wall
    Ls_top : float
        Slip length at top wall
    dqx : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    dqy : numpy.ndarray, optional
        Gradient (y) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)

    Returns
    -------
    numpy.ndarray
        Gap-averaged viscous stress tensor components
    """
    if dqx is None:
        dqx = np.zeros_like(q)
    if dqy is None:
        dqy = np.zeros_like(q)

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    tau = np.zeros((3, *q.shape[1:]))

    D = 4 * Ls_bot * h[0] + 4 * Ls_top * h[0] + h[0] ** 2 + 12 * Ls_bot * Ls_top

    # tau_xx_avg
    tau[0] = -((h[0] * (
                    8 * Ls_bot * U_top * h[1] * eta
                    - 4 * Ls_top * U_bot * h[1] * eta
                    - 2 * Ls_top * U_bot * h[1] * v2
                    + 4 * Ls_bot * U_top * h[1] * v2
                    - 2 * Ls_top * V_bot * h[2] * v2
                    + 4 * Ls_bot * V_top * h[2] * v2
                    + 2 * U_top * h[1] * eta * h[0]
                    + U_top * h[1] * h[0] * v2
                    + V_top * h[2] * h[0] * v2
                ))/D + (h[0] * (2 * dqx[0] * eta * q[1] + dqx[0] * q[1] * v2 + dqy[0] * q[2] * v2) * D - h[0] * q[0] * (
                    2 * dqx[1] * eta * h[0] ** 2
                    + dqx[1] * h[0] ** 2 * v2
                    + dqy[2] * h[0] ** 2 * v2
                    + h[1] * h[0] * q[1] * v2
                    + h[2] * h[0] * q[2] * v2
                    + 24 * Ls_bot * Ls_top * dqx[1] * eta
                    + 12 * Ls_bot * Ls_top * dqx[1] * v2
                    + 12 * Ls_bot * Ls_top * dqy[2] * v2
                    + 8 * Ls_bot * dqx[1] * eta * h[0]
                    + 8 * Ls_top * dqx[1] * eta * h[0]
                    + 8 * Ls_bot * h[1] * eta * q[1]
                    - 4 * Ls_top * h[1] * eta * q[1]
                    + 4 * Ls_bot * dqx[1] * h[0] * v2
                    + 4 * Ls_bot * dqy[2] * h[0] * v2
                    + 4 * Ls_top * dqx[1] * h[0] * v2
                    + 4 * Ls_top * dqy[2] * h[0] * v2
                    + 4 * Ls_bot * h[1] * q[1] * v2
                    - 2 * Ls_top * h[1] * q[1] * v2
                    + 4 * Ls_bot * h[2] * q[2] * v2
                    - 2 * Ls_top * h[2] * q[2] * v2
                    + 2 * h[1] * eta * h[0] * q[1]
                ))/(q[0] ** 2 * D))/h[0]

    # tau_yy_avg
    tau[1] = -((h[0] * (
                    8 * Ls_bot * V_top * h[2] * eta
                    - 4 * Ls_top * V_bot * h[2] * eta
                    - 2 * Ls_top * U_bot * h[1] * v2
                    + 4 * Ls_bot * U_top * h[1] * v2
                    - 2 * Ls_top * V_bot * h[2] * v2
                    + 4 * Ls_bot * V_top * h[2] * v2
                    + 2 * V_top * h[2] * eta * h[0]
                    + U_top * h[1] * h[0] * v2
                    + V_top * h[2] * h[0] * v2
                ))/D + (h[0] * (2 * dqy[0] * eta * q[2] + dqx[0] * q[1] * v2 + dqy[0] * q[2] * v2) * D - h[0] * q[0] * (
                    2 * dqy[2] * eta * h[0] ** 2
                    + dqx[1] * h[0] ** 2 * v2
                    + dqy[2] * h[0] ** 2 * v2
                    + h[1] * h[0] * q[1] * v2
                    + h[2] * h[0] * q[2] * v2
                    + 24 * Ls_bot * Ls_top * dqy[2] * eta
                    + 12 * Ls_bot * Ls_top * dqx[1] * v2
                    + 12 * Ls_bot * Ls_top * dqy[2] * v2
                    + 8 * Ls_bot * dqy[2] * eta * h[0]
                    + 8 * Ls_top * dqy[2] * eta * h[0]
                    + 8 * Ls_bot * h[2] * eta * q[2]
                    - 4 * Ls_top * h[2] * eta * q[2]
                    + 4 * Ls_bot * dqx[1] * h[0] * v2
                    + 4 * Ls_bot * dqy[2] * h[0] * v2
                    + 4 * Ls_top * dqx[1] * h[0] * v2
                    + 4 * Ls_top * dqy[2] * h[0] * v2
                    + 4 * Ls_bot * h[1] * q[1] * v2
                    - 2 * Ls_top * h[1] * q[1] * v2
                    + 4 * Ls_bot * h[2] * q[2] * v2
                    - 2 * Ls_top * h[2] * q[2] * v2
                    + 2 * h[2] * eta * h[0] * q[2]
                ))/(q[0] ** 2 * D))/h[0]

    # tau_xy_avg
    tau[2] = -((eta * h[0] * (dqy[0] * q[1] + dqx[0] * q[2]) * D - eta * h[0] * q[0] * (
                    dqy[1] * h[0] ** 2
                    + dqx[2] * h[0] ** 2
                    + 12 * Ls_bot * Ls_top * dqy[1]
                    + 12 * Ls_bot * Ls_top * dqx[2]
                    + 4 * Ls_bot * dqy[1] * h[0]
                    + 4 * Ls_bot * dqx[2] * h[0]
                    + 4 * Ls_top * dqy[1] * h[0]
                    + 4 * Ls_top * dqx[2] * h[0]
                    + 4 * Ls_bot * h[2] * q[1]
                    - 2 * Ls_top * h[2] * q[1]
                    + 4 * Ls_bot * h[1] * q[2]
                    - 2 * Ls_top * h[1] * q[2]
                    + h[2] * h[0] * q[1]
                    + h[1] * h[0] * q[2]
                ))/(q[0] ** 2 * D) + (eta * h[0] * (
                    4 * Ls_bot * U_top * h[2]
                    - 2 * Ls_top * U_bot * h[2]
                    - 2 * Ls_top * V_bot * h[1]
                    + 4 * Ls_bot * V_top * h[1]
                    + U_top * h[2] * h[0]
                    + V_top * h[1] * h[0]
                ))/D)/h[0]

    return tau


def stress_bottom_xz(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
                     dqx=None, dqy=None):
    """Bottom wall shear stress tau_xz (dual-wall velocity, JAX version)."""
    if dqx is None:
        dqx = jnp.zeros_like(q[0])
    if dqy is None:
        dqy = jnp.zeros_like(q[0])

    tau_xz = -(2 * eta * (
                6 * Ls_top * U_bot * q[0]
                - 3 * h[0] * q[1]
                - 6 * Ls_top * q[1]
                + 2 * U_bot * h[0] * q[0]
                + U_top * h[0] * q[0]
            )) / (q[0] * (
                4 * Ls_bot * h[0]
                + 4 * Ls_top * h[0]
                + h[0] ** 2
                + 12 * Ls_bot * Ls_top
            ))

    return tau_xz


def stress_top_xz(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
                  dqx=None, dqy=None):
    """Top wall shear stress tau_xz (dual-wall velocity, JAX version)."""
    if dqx is None:
        dqx = jnp.zeros_like(q[0])
    if dqy is None:
        dqy = jnp.zeros_like(q[0])

    tau_xz = -eta * ((2 * (
                    6 * Ls_top * U_bot * q[0]
                    - 3 * h[0] * q[1]
                    - 6 * Ls_top * q[1]
                    + 2 * U_bot * h[0] * q[0]
                    + U_top * h[0] * q[0]
                )) / (q[0] * (
                    4 * Ls_bot * h[0]
                    + 4 * Ls_top * h[0]
                    + h[0] ** 2
                    + 12 * Ls_bot * Ls_top
                )) - (6 * (
                    2 * Ls_top * U_bot * q[0]
                    - 2 * Ls_top * q[1]
                    - 2 * h[0] * q[1]
                    - 2 * Ls_bot * q[1]
                    + 2 * Ls_bot * U_top * q[0]
                    + U_bot * h[0] * q[0]
                    + U_top * h[0] * q[0]
                )) / (q[0] * (
                    4 * Ls_bot * h[0]
                    + 4 * Ls_top * h[0]
                    + h[0] ** 2
                    + 12 * Ls_bot * Ls_top
                )))

    return tau_xz


def stress_bottom_yz(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
                     dqx=None, dqy=None):
    """Bottom wall shear stress tau_yz (dual-wall velocity, JAX version)."""
    if dqx is None:
        dqx = jnp.zeros_like(q[0])
    if dqy is None:
        dqy = jnp.zeros_like(q[0])

    tau_yz = -(2 * eta * (
                6 * Ls_top * V_bot * q[0]
                - 3 * h[0] * q[2]
                - 6 * Ls_top * q[2]
                + 2 * V_bot * h[0] * q[0]
                + V_top * h[0] * q[0]
            )) / (q[0] * (
                4 * Ls_bot * h[0]
                + 4 * Ls_top * h[0]
                + h[0] ** 2
                + 12 * Ls_bot * Ls_top
            ))

    return tau_yz


def stress_top_yz(q, h, U_bot, V_bot, U_top, V_top, eta, zeta, Ls_bot, Ls_top,
                  dqx=None, dqy=None):
    """Top wall shear stress tau_yz (dual-wall velocity, JAX version)."""
    if dqx is None:
        dqx = jnp.zeros_like(q[0])
    if dqy is None:
        dqy = jnp.zeros_like(q[0])

    tau_yz = -eta * ((2 * (
                    6 * Ls_top * V_bot * q[0]
                    - 3 * h[0] * q[2]
                    - 6 * Ls_top * q[2]
                    + 2 * V_bot * h[0] * q[0]
                    + V_top * h[0] * q[0]
                )) / (q[0] * (
                    4 * Ls_bot * h[0]
                    + 4 * Ls_top * h[0]
                    + h[0] ** 2
                    + 12 * Ls_bot * Ls_top
                )) - (6 * (
                    2 * Ls_top * V_bot * q[0]
                    - 2 * Ls_top * q[2]
                    - 2 * h[0] * q[2]
                    - 2 * Ls_bot * q[2]
                    + 2 * Ls_bot * V_top * q[0]
                    + V_bot * h[0] * q[0]
                    + V_top * h[0] * q[0]
                )) / (q[0] * (
                    4 * Ls_bot * h[0]
                    + 4 * Ls_top * h[0]
                    + h[0] ** 2
                    + 12 * Ls_bot * Ls_top
                )))

    return tau_yz


def get_shear_viscosity(stress_object):
    s = stress_object

    # piezoviscosity
    if 'piezo' in s.prop.keys():
        mu0 = piezoviscosity(s.pressure if not s.prop['EOS'] == 'Bayada' else s.solution[0],
                                s.prop['shear'],
                                s.prop['piezo'])
    else:
        mu0 = s.prop['shear']
    # shear-thinning
    if 'thinning' in s.prop.keys():
        shear_rate = shear_rate_avg(s.dp_dx,
                                    s.dp_dy,
                                    s.height,
                                    np.hypot(s.geo['U_bot'], s.geo['V_bot']),
                                    np.hypot(s.geo['U_top'], s.geo['V_top']),
                                    mu0)

        shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                        s.prop['thinning'])
    else:
        shear_viscosity = mu0
    return shear_viscosity

