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


def get_velocity_profiles(z, q, U_bot=0.0, V_bot=0.0, U_top=0.0, V_top=0.0,
                          Ls_bot=0.0, Ls_top=0.0):
    """Velocity profiles for a given flow rate and wall velocity

    Parameters
    ----------
    z : array-like
        Gap coordinate (z)
    q : array-like
        Height-averaged solution, (rho, jx, jy) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    U_bot : float, optional
        Bottom wall velocity in x direction (the default is 0.0)
    V_bot : float, optional
        Bottom wall velocity in y direction (the default is 0.0)
    U_top : float, optional
        Top wall velocity in x direction (the default is 0.0)
    V_top : float, optional
        Top wall velocity in y direction (the default is 0.0)
    Ls_bot : float, optional
        Slip length at bottom wall (the default is 0.0, which means no-slip)
    Ls_top : float, optional
        Slip length at top wall (the default is 0.0, which means no-slip)

    Returns
    -------
    array-like, array-like
        Discretized profiles u(z) and v(z)
    """

    h = z[-1]

    u = (
        12 * Ls_bot * Ls_top * h * q[1]
        + 4 * Ls_top * U_bot * h ** 2 * q[0]
        - 12 * Ls_top * U_bot * h * q[0] * z
        + 6 * Ls_top * U_bot * q[0] * z ** 2
        + 6 * Ls_bot * h ** 2 * q[1]
        + 12 * Ls_top * h * q[1] * z
        - 6 * Ls_top * q[1] * z ** 2
        - 2 * Ls_bot * U_top * h ** 2 * q[0]
        + 6 * Ls_bot * U_top * q[0] * z ** 2
        - 6 * Ls_bot * q[1] * z ** 2
        + U_bot * h ** 3 * q[0]
        - 4 * U_bot * h ** 2 * q[0] * z
        + 3 * U_bot * h * q[0] * z ** 2
        - 2 * U_top * h ** 2 * q[0] * z
        + 3 * U_top * h * q[0] * z ** 2
        + 6 * h ** 2 * q[1] * z
        - 6 * h * q[1] * z ** 2
    ) / (h * q[0] * (4 * Ls_bot * h + 4 * Ls_top * h + h ** 2 + 12 * Ls_bot * Ls_top))
    v = (
        12 * Ls_bot * Ls_top * h * q[2]
        + 4 * Ls_top * V_bot * h ** 2 * q[0]
        - 12 * Ls_top * V_bot * h * q[0] * z
        + 6 * Ls_top * V_bot * q[0] * z ** 2
        + 6 * Ls_bot * h ** 2 * q[2]
        + 12 * Ls_top * h * q[2] * z
        - 6 * Ls_top * q[2] * z ** 2
        - 2 * Ls_bot * V_top * h ** 2 * q[0]
        + 6 * Ls_bot * V_top * q[0] * z ** 2
        - 6 * Ls_bot * q[2] * z ** 2
        + V_bot * h ** 3 * q[0]
        - 4 * V_bot * h ** 2 * q[0] * z
        + 3 * V_bot * h * q[0] * z ** 2
        - 2 * V_top * h ** 2 * q[0] * z
        + 3 * V_top * h * q[0] * z ** 2
        + 6 * h ** 2 * q[2] * z
        - 6 * h * q[2] * z ** 2
    ) / (h * q[0] * (4 * Ls_bot * h + 4 * Ls_top * h + h ** 2 + 12 * Ls_bot * Ls_top))

    return u, v


def get_stress_profiles(z, h, q, dqx, dqy,
                        U_bot=0.0, V_bot=0.0, U_top=0.0, V_top=0.0,
                        eta=1.0, zeta=1.0, Ls_bot=0.0, Ls_top=0.0):
    """Viscous stress profiles for a given flow rate and wall velocity

    Parameters
    ----------
    z : array-like
        Gap coordinate (z)
    h : array-like
        Gap height and gradients, (h, ∂h/∂x, ∂h/∂y) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    q : array-like
        Gap-averaged solution, (ρ, jx, jy) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    dqx : array-like
        Gap-averaged solution gradients, (∂ρ/∂x, ∂jx/∂x, ∂jy/∂x) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    dqy : array-like
        Gap-averaged solution gradients, (∂ρ/∂y, ∂jx/∂y, ∂jy/∂y) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    U_bot : float, optional
        Bottom wall velocity in x direction (the default is 0.0)
    V_bot : float, optional
        Bottom wall velocity in y direction (the default is 0.0)
    U_top : float, optional
        Top wall velocity in x direction (the default is 0.0)
    V_top : float, optional
        Top wall velocity in y direction (the default is 0.0)
    eta : float, optional
        Dynamic shear viscosity (the default is 1.0)
    zeta : float, optional
        Dynamic bulk viscosity (the default is 1.0)
    Ls_bot : float, optional
        Slip length at bottom wall (the default is 0.0, which means no-slip)
    Ls_top : float, optional
        Slip length at top wall (the default is 0.0, which means no-slip)

    Returns
    -------
    Tuple[array-like, ...]
        Discretized stress profiles τ_xx(z), τ_yy(z), τ_zz(z), τ_yz(z), τ_xz(z), τ_xy(z)
    """

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    D = 4 * Ls_bot * h[0] + 4 * Ls_top * h[0] + h[0] ** 2 + 12 * Ls_bot * Ls_top

    # tau_xx(z)
    tau_xx = -(2 * (
                24 * Ls_bot ** 2 * dqx[0] * eta * h[0] ** 4 * q[1]
                + 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 4 * q[1] * v2
                + 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 4 * q[2] * v2
                - 24 * Ls_bot ** 2 * dqx[1] * eta * h[0] ** 4 * q[0]
                - 12 * Ls_bot ** 2 * dqx[1] * h[0] ** 4 * v2 * q[0]
                - 12 * Ls_bot ** 2 * dqy[2] * h[0] ** 4 * v2 * q[0]
                - 6 * dqx[0] * eta * h[0] ** 4 * q[1] * z ** 2
                - 3 * dqx[0] * h[0] ** 4 * q[1] * v2 * z ** 2
                - 3 * dqy[0] * h[0] ** 4 * q[2] * v2 * z ** 2
                + 6 * dqx[1] * eta * h[0] ** 4 * q[0] * z ** 2
                + 3 * dqx[1] * h[0] ** 4 * v2 * q[0] * z ** 2
                + 3 * dqy[2] * h[0] ** 4 * v2 * q[0] * z ** 2
                + 6 * Ls_bot * dqx[0] * eta * h[0] ** 5 * q[1]
                + 3 * Ls_bot * dqx[0] * h[0] ** 5 * q[1] * v2
                + 3 * Ls_bot * dqy[0] * h[0] ** 5 * q[2] * v2
                - 6 * Ls_bot * dqx[1] * eta * h[0] ** 5 * q[0]
                - 3 * Ls_bot * dqx[1] * h[0] ** 5 * v2 * q[0]
                - 3 * Ls_bot * dqy[2] * h[0] ** 5 * v2 * q[0]
                + 6 * dqx[0] * eta * h[0] ** 5 * q[1] * z
                + 3 * dqx[0] * h[0] ** 5 * q[1] * v2 * z
                + 3 * dqy[0] * h[0] ** 5 * q[2] * v2 * z
                - 6 * dqx[1] * eta * h[0] ** 5 * q[0] * z
                - 3 * dqx[1] * h[0] ** 5 * v2 * q[0] * z
                - 3 * dqy[2] * h[0] ** 5 * v2 * q[0] * z
                - 18 * Ls_bot * Ls_top * dqx[1] * h[0] ** 4 * v2 * q[0]
                - 18 * Ls_bot * Ls_top * dqy[2] * h[0] ** 4 * v2 * q[0]
                + 6 * Ls_bot * h[1] * eta * h[0] ** 4 * q[1] * q[0]
                + 3 * Ls_bot * h[1] * h[0] ** 4 * q[1] * v2 * q[0]
                + 3 * Ls_bot * h[2] * h[0] ** 4 * q[2] * v2 * q[0]
                + 24 * Ls_bot * dqx[0] * eta * h[0] ** 4 * q[1] * z
                + 36 * Ls_top * dqx[0] * eta * h[0] ** 4 * q[1] * z
                + 12 * Ls_bot * dqx[0] * h[0] ** 4 * q[1] * v2 * z
                + 18 * Ls_top * dqx[0] * h[0] ** 4 * q[1] * v2 * z
                + 12 * Ls_bot * dqy[0] * h[0] ** 4 * q[2] * v2 * z
                + 18 * Ls_top * dqy[0] * h[0] ** 4 * q[2] * v2 * z
                - 24 * Ls_bot * dqx[1] * eta * h[0] ** 4 * q[0] * z
                - 36 * Ls_top * dqx[1] * eta * h[0] ** 4 * q[0] * z
                - 12 * Ls_bot * dqx[1] * h[0] ** 4 * v2 * q[0] * z
                - 12 * Ls_bot * dqy[2] * h[0] ** 4 * v2 * q[0] * z
                - 18 * Ls_top * dqx[1] * h[0] ** 4 * v2 * q[0] * z
                - 18 * Ls_top * dqy[2] * h[0] ** 4 * v2 * q[0] * z
                + 6 * h[1] * eta * h[0] ** 4 * q[1] * q[0] * z
                + 3 * h[1] * h[0] ** 4 * q[1] * v2 * q[0] * z
                + 3 * h[2] * h[0] ** 4 * q[2] * v2 * q[0] * z
                + 48 * Ls_bot * Ls_top ** 2 * dqx[0] * eta * h[0] ** 3 * q[1]
                + 120 * Ls_bot ** 2 * Ls_top * dqx[0] * eta * h[0] ** 3 * q[1]
                + 24 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[1] * v2
                + 60 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2
                + 24 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[2] * v2
                + 60 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2
                - 48 * Ls_bot * Ls_top ** 2 * dqx[1] * eta * h[0] ** 3 * q[0]
                - 120 * Ls_bot ** 2 * Ls_top * dqx[1] * eta * h[0] ** 3 * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] ** 3 * v2 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] ** 3 * v2 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0]
                - 4 * Ls_bot * U_bot * h[1] * eta * h[0] ** 4 * q[0] ** 2
                - 2 * Ls_bot * U_top * h[1] * eta * h[0] ** 4 * q[0] ** 2
                - 2 * Ls_bot * U_bot * h[1] * h[0] ** 4 * v2 * q[0] ** 2
                - Ls_bot * U_top * h[1] * h[0] ** 4 * v2 * q[0] ** 2
                - 2 * Ls_bot * V_bot * h[2] * h[0] ** 4 * v2 * q[0] ** 2
                - Ls_bot * V_top * h[2] * h[0] ** 4 * v2 * q[0] ** 2
                - 30 * Ls_bot * dqx[0] * eta * h[0] ** 3 * q[1] * z ** 2
                - 30 * Ls_top * dqx[0] * eta * h[0] ** 3 * q[1] * z ** 2
                + 48 * Ls_top ** 2 * dqx[0] * eta * h[0] ** 3 * q[1] * z
                - 15 * Ls_bot * dqx[0] * h[0] ** 3 * q[1] * v2 * z ** 2
                - 15 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2 * z ** 2
                + 24 * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                - 15 * Ls_bot * dqy[0] * h[0] ** 3 * q[2] * v2 * z ** 2
                - 15 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2 * z ** 2
                + 24 * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                + 30 * Ls_bot * dqx[1] * eta * h[0] ** 3 * q[0] * z ** 2
                + 30 * Ls_top * dqx[1] * eta * h[0] ** 3 * q[0] * z ** 2
                - 48 * Ls_top ** 2 * dqx[1] * eta * h[0] ** 3 * q[0] * z
                + 15 * Ls_bot * dqx[1] * h[0] ** 3 * v2 * q[0] * z ** 2
                + 15 * Ls_bot * dqy[2] * h[0] ** 3 * v2 * q[0] * z ** 2
                + 15 * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqx[1] * h[0] ** 3 * v2 * q[0] * z
                + 15 * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqy[2] * h[0] ** 3 * v2 * q[0] * z
                - 4 * U_bot * h[1] * eta * h[0] ** 4 * q[0] ** 2 * z
                - 2 * U_top * h[1] * eta * h[0] ** 4 * q[0] ** 2 * z
                - 2 * U_bot * h[1] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - U_top * h[1] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - 2 * V_bot * h[2] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - V_top * h[2] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - 12 * h[1] * eta * h[0] ** 3 * q[1] * q[0] * z ** 2
                - 6 * h[1] * h[0] ** 3 * q[1] * v2 * q[0] * z ** 2
                - 6 * h[2] * h[0] ** 3 * q[2] * v2 * q[0] * z ** 2
                + 144 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * eta * h[0] ** 2 * q[1]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2
                - 144 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * eta * h[0] ** 2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0]
                - 24 * Ls_bot ** 2 * dqx[0] * eta * h[0] ** 2 * q[1] * z ** 2
                - 24 * Ls_top ** 2 * dqx[0] * eta * h[0] ** 2 * q[1] * z ** 2
                - 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z ** 2
                - 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z ** 2
                - 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z ** 2
                - 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z ** 2
                + 24 * Ls_bot ** 2 * dqx[1] * eta * h[0] ** 2 * q[0] * z ** 2
                + 24 * Ls_top ** 2 * dqx[1] * eta * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_bot ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 12 * Ls_bot ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 6 * U_bot * h[1] * eta * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 6 * U_top * h[1] * eta * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top * dqx[0] * eta * h[0] ** 4 * q[1]
                + 3 * U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 3 * U_top * h[1] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 3 * V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 3 * V_top * h[2] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 4 * q[1] * v2
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 4 * q[2] * v2
                - 36 * Ls_bot * Ls_top * dqx[1] * eta * h[0] ** 4 * q[0]
                - 48 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * Ls_top * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2
                - 24 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                - 24 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 72 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * eta * q[0] ** 2 * z ** 2
                + 72 * Ls_bot ** 2 * Ls_top * U_top * h[1] * eta * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * v2 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * U_top * h[1] * v2 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * v2 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * V_top * h[2] * v2 * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 30 * Ls_top * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 48 * Ls_top ** 2 * U_bot * h[1] * eta * h[0] * q[0] ** 2 * z ** 2
                - 48 * Ls_top ** 2 * U_bot * h[1] * eta * h[0] ** 2 * q[0] ** 2 * z
                + 30 * Ls_bot * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 48 * Ls_bot ** 2 * U_top * h[1] * eta * h[0] * q[0] ** 2 * z ** 2
                + 12 * Ls_top * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * U_bot * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z
                + 15 * Ls_bot * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * U_top * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 6 * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * V_bot * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z
                + 15 * Ls_bot * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * V_top * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 6 * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * h[1] * eta * h[0] ** 3 * q[1] * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * h[0] ** 3 * q[1] * v2 * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * h[0] ** 3 * q[2] * v2 * q[0]
                + 120 * Ls_bot * Ls_top * dqx[0] * eta * h[0] ** 3 * q[1] * z
                + 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                + 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                - 120 * Ls_bot * Ls_top * dqx[1] * eta * h[0] ** 3 * q[0] * z
                - 60 * Ls_bot * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0] * z
                - 60 * Ls_bot * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0] * z
                + 24 * Ls_top * h[1] * eta * h[0] ** 3 * q[1] * q[0] * z
                + 12 * Ls_top * h[1] * h[0] ** 3 * q[1] * v2 * q[0] * z
                + 12 * Ls_top * h[2] * h[0] ** 3 * q[2] * v2 * q[0] * z
                - 24 * Ls_bot * Ls_top * U_bot * h[1] * eta * h[0] ** 3 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2
                + 48 * Ls_bot * Ls_top ** 2 * h[1] * eta * h[0] ** 2 * q[1] * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * h[1] * eta * h[0] ** 2 * q[1] * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                - 120 * Ls_bot * Ls_top * dqx[0] * eta * h[0] ** 2 * q[1] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqx[0] * eta * h[0] * q[1] * z ** 2
                + 144 * Ls_bot * Ls_top ** 2 * dqx[0] * eta * h[0] ** 2 * q[1] * z
                - 72 * Ls_bot ** 2 * Ls_top * dqx[0] * eta * h[0] * q[1] * z ** 2
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[1] * v2 * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[1] * v2 * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z
                - 36 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[1] * v2 * z ** 2
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[2] * v2 * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[2] * v2 * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z
                - 36 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[2] * v2 * z ** 2
                + 120 * Ls_bot * Ls_top * dqx[1] * eta * h[0] ** 2 * q[0] * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqx[1] * eta * h[0] * q[0] * z ** 2
                - 144 * Ls_bot * Ls_top ** 2 * dqx[1] * eta * h[0] ** 2 * q[0] * z
                + 72 * Ls_bot ** 2 * Ls_top * dqx[1] * eta * h[0] * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * h[1] * eta * q[1] * q[0] * z ** 2
                - 72 * Ls_bot ** 2 * Ls_top * h[1] * eta * q[1] * q[0] * z ** 2
                + 60 * Ls_bot * Ls_top * dqx[1] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] * v2 * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] * v2 * q[0] * z ** 2
                + 60 * Ls_bot * Ls_top * dqy[2] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] * v2 * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] * v2 * q[0] * z ** 2
                - 24 * Ls_top * U_bot * h[1] * eta * h[0] ** 3 * q[0] ** 2 * z
                - 36 * Ls_bot * Ls_top ** 2 * h[1] * q[1] * v2 * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[1] * q[1] * v2 * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[2] * q[2] * v2 * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[2] * q[2] * v2 * q[0] * z ** 2
                - 12 * Ls_top * U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2 * z
                - 12 * Ls_top * V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2 * z
                - 42 * Ls_bot * h[1] * eta * h[0] ** 2 * q[1] * q[0] * z ** 2
                - 48 * Ls_bot ** 2 * h[1] * eta * h[0] * q[1] * q[0] * z ** 2
                - 42 * Ls_top * h[1] * eta * h[0] ** 2 * q[1] * q[0] * z ** 2
                - 48 * Ls_top ** 2 * h[1] * eta * h[0] * q[1] * q[0] * z ** 2
                + 48 * Ls_top ** 2 * h[1] * eta * h[0] ** 2 * q[1] * q[0] * z
                - 21 * Ls_bot * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[1] * v2 * q[0] * z ** 2
                - 21 * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[1] * h[0] * q[1] * v2 * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z
                - 21 * Ls_bot * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[2] * v2 * q[0] * z ** 2
                - 21 * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[2] * h[0] * q[2] * v2 * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z
                + 48 * Ls_bot * Ls_top * U_bot * h[1] * eta * h[0] * q[0] ** 2 * z ** 2
                + 48 * Ls_bot * Ls_top * U_top * h[1] * eta * h[0] * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * U_top * h[1] * eta * h[0] ** 2 * q[0] ** 2 * z
                + 24 * Ls_bot * Ls_top * U_bot * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * U_top * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z
                + 24 * Ls_bot * Ls_top * V_bot * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * V_top * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z
                - 96 * Ls_bot * Ls_top * h[1] * eta * h[0] * q[1] * q[0] * z ** 2
                - 24 * Ls_bot * Ls_top * h[1] * eta * h[0] ** 2 * q[1] * q[0] * z
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[1] * v2 * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[2] * v2 * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z
            ))/(h[0] ** 2 * q[0] ** 2 * D ** 2)

    # tau_yy(z)
    tau_yy = -(2 * (
                24 * Ls_bot ** 2 * dqy[0] * eta * h[0] ** 4 * q[2]
                + 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 4 * q[1] * v2
                + 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 4 * q[2] * v2
                - 24 * Ls_bot ** 2 * dqy[2] * eta * h[0] ** 4 * q[0]
                - 12 * Ls_bot ** 2 * dqx[1] * h[0] ** 4 * v2 * q[0]
                - 12 * Ls_bot ** 2 * dqy[2] * h[0] ** 4 * v2 * q[0]
                - 6 * dqy[0] * eta * h[0] ** 4 * q[2] * z ** 2
                - 3 * dqx[0] * h[0] ** 4 * q[1] * v2 * z ** 2
                - 3 * dqy[0] * h[0] ** 4 * q[2] * v2 * z ** 2
                + 6 * dqy[2] * eta * h[0] ** 4 * q[0] * z ** 2
                + 3 * dqx[1] * h[0] ** 4 * v2 * q[0] * z ** 2
                + 3 * dqy[2] * h[0] ** 4 * v2 * q[0] * z ** 2
                + 6 * Ls_bot * dqy[0] * eta * h[0] ** 5 * q[2]
                + 3 * Ls_bot * dqx[0] * h[0] ** 5 * q[1] * v2
                + 3 * Ls_bot * dqy[0] * h[0] ** 5 * q[2] * v2
                - 6 * Ls_bot * dqy[2] * eta * h[0] ** 5 * q[0]
                - 3 * Ls_bot * dqx[1] * h[0] ** 5 * v2 * q[0]
                - 3 * Ls_bot * dqy[2] * h[0] ** 5 * v2 * q[0]
                + 6 * dqy[0] * eta * h[0] ** 5 * q[2] * z
                + 3 * dqx[0] * h[0] ** 5 * q[1] * v2 * z
                + 3 * dqy[0] * h[0] ** 5 * q[2] * v2 * z
                - 6 * dqy[2] * eta * h[0] ** 5 * q[0] * z
                - 3 * dqx[1] * h[0] ** 5 * v2 * q[0] * z
                - 3 * dqy[2] * h[0] ** 5 * v2 * q[0] * z
                - 18 * Ls_bot * Ls_top * dqx[1] * h[0] ** 4 * v2 * q[0]
                - 18 * Ls_bot * Ls_top * dqy[2] * h[0] ** 4 * v2 * q[0]
                + 6 * Ls_bot * h[2] * eta * h[0] ** 4 * q[2] * q[0]
                + 3 * Ls_bot * h[1] * h[0] ** 4 * q[1] * v2 * q[0]
                + 3 * Ls_bot * h[2] * h[0] ** 4 * q[2] * v2 * q[0]
                + 24 * Ls_bot * dqy[0] * eta * h[0] ** 4 * q[2] * z
                + 36 * Ls_top * dqy[0] * eta * h[0] ** 4 * q[2] * z
                + 12 * Ls_bot * dqx[0] * h[0] ** 4 * q[1] * v2 * z
                + 18 * Ls_top * dqx[0] * h[0] ** 4 * q[1] * v2 * z
                + 12 * Ls_bot * dqy[0] * h[0] ** 4 * q[2] * v2 * z
                + 18 * Ls_top * dqy[0] * h[0] ** 4 * q[2] * v2 * z
                - 24 * Ls_bot * dqy[2] * eta * h[0] ** 4 * q[0] * z
                - 36 * Ls_top * dqy[2] * eta * h[0] ** 4 * q[0] * z
                - 12 * Ls_bot * dqx[1] * h[0] ** 4 * v2 * q[0] * z
                - 12 * Ls_bot * dqy[2] * h[0] ** 4 * v2 * q[0] * z
                - 18 * Ls_top * dqx[1] * h[0] ** 4 * v2 * q[0] * z
                - 18 * Ls_top * dqy[2] * h[0] ** 4 * v2 * q[0] * z
                + 6 * h[2] * eta * h[0] ** 4 * q[2] * q[0] * z
                + 3 * h[1] * h[0] ** 4 * q[1] * v2 * q[0] * z
                + 3 * h[2] * h[0] ** 4 * q[2] * v2 * q[0] * z
                + 48 * Ls_bot * Ls_top ** 2 * dqy[0] * eta * h[0] ** 3 * q[2]
                + 120 * Ls_bot ** 2 * Ls_top * dqy[0] * eta * h[0] ** 3 * q[2]
                + 24 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[1] * v2
                + 60 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2
                + 24 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[2] * v2
                + 60 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2
                - 48 * Ls_bot * Ls_top ** 2 * dqy[2] * eta * h[0] ** 3 * q[0]
                - 120 * Ls_bot ** 2 * Ls_top * dqy[2] * eta * h[0] ** 3 * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] ** 3 * v2 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] ** 3 * v2 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0]
                - 4 * Ls_bot * V_bot * h[2] * eta * h[0] ** 4 * q[0] ** 2
                - 2 * Ls_bot * V_top * h[2] * eta * h[0] ** 4 * q[0] ** 2
                - 2 * Ls_bot * U_bot * h[1] * h[0] ** 4 * v2 * q[0] ** 2
                - Ls_bot * U_top * h[1] * h[0] ** 4 * v2 * q[0] ** 2
                - 2 * Ls_bot * V_bot * h[2] * h[0] ** 4 * v2 * q[0] ** 2
                - Ls_bot * V_top * h[2] * h[0] ** 4 * v2 * q[0] ** 2
                - 30 * Ls_bot * dqy[0] * eta * h[0] ** 3 * q[2] * z ** 2
                - 30 * Ls_top * dqy[0] * eta * h[0] ** 3 * q[2] * z ** 2
                + 48 * Ls_top ** 2 * dqy[0] * eta * h[0] ** 3 * q[2] * z
                - 15 * Ls_bot * dqx[0] * h[0] ** 3 * q[1] * v2 * z ** 2
                - 15 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2 * z ** 2
                + 24 * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                - 15 * Ls_bot * dqy[0] * h[0] ** 3 * q[2] * v2 * z ** 2
                - 15 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2 * z ** 2
                + 24 * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                + 30 * Ls_bot * dqy[2] * eta * h[0] ** 3 * q[0] * z ** 2
                + 30 * Ls_top * dqy[2] * eta * h[0] ** 3 * q[0] * z ** 2
                - 48 * Ls_top ** 2 * dqy[2] * eta * h[0] ** 3 * q[0] * z
                + 15 * Ls_bot * dqx[1] * h[0] ** 3 * v2 * q[0] * z ** 2
                + 15 * Ls_bot * dqy[2] * h[0] ** 3 * v2 * q[0] * z ** 2
                + 15 * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqx[1] * h[0] ** 3 * v2 * q[0] * z
                + 15 * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqy[2] * h[0] ** 3 * v2 * q[0] * z
                - 4 * V_bot * h[2] * eta * h[0] ** 4 * q[0] ** 2 * z
                - 2 * V_top * h[2] * eta * h[0] ** 4 * q[0] ** 2 * z
                - 2 * U_bot * h[1] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - U_top * h[1] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - 2 * V_bot * h[2] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - V_top * h[2] * h[0] ** 4 * v2 * q[0] ** 2 * z
                - 12 * h[2] * eta * h[0] ** 3 * q[2] * q[0] * z ** 2
                - 6 * h[1] * h[0] ** 3 * q[1] * v2 * q[0] * z ** 2
                - 6 * h[2] * h[0] ** 3 * q[2] * v2 * q[0] * z ** 2
                + 144 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * eta * h[0] ** 2 * q[2]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2
                - 144 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * eta * h[0] ** 2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0]
                - 24 * Ls_bot ** 2 * dqy[0] * eta * h[0] ** 2 * q[2] * z ** 2
                - 24 * Ls_top ** 2 * dqy[0] * eta * h[0] ** 2 * q[2] * z ** 2
                - 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z ** 2
                - 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z ** 2
                - 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z ** 2
                - 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z ** 2
                + 24 * Ls_bot ** 2 * dqy[2] * eta * h[0] ** 2 * q[0] * z ** 2
                + 24 * Ls_top ** 2 * dqy[2] * eta * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_bot ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 12 * Ls_bot ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 6 * V_bot * h[2] * eta * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 6 * V_top * h[2] * eta * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top * dqy[0] * eta * h[0] ** 4 * q[2]
                + 3 * U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 3 * U_top * h[1] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 3 * V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 3 * V_top * h[2] * h[0] ** 3 * v2 * q[0] ** 2 * z ** 2
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 4 * q[1] * v2
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 4 * q[2] * v2
                - 36 * Ls_bot * Ls_top * dqy[2] * eta * h[0] ** 4 * q[0]
                - 48 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2
                + 24 * Ls_bot ** 2 * Ls_top * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2
                - 24 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2
                - 24 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2
                + 72 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * eta * q[0] ** 2 * z ** 2
                + 72 * Ls_bot ** 2 * Ls_top * V_top * h[2] * eta * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * v2 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * U_top * h[1] * v2 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * v2 * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * V_top * h[2] * v2 * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 30 * Ls_top * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 48 * Ls_top ** 2 * V_bot * h[2] * eta * h[0] * q[0] ** 2 * z ** 2
                - 48 * Ls_top ** 2 * V_bot * h[2] * eta * h[0] ** 2 * q[0] ** 2 * z
                + 30 * Ls_bot * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 48 * Ls_bot ** 2 * V_top * h[2] * eta * h[0] * q[0] ** 2 * z ** 2
                + 12 * Ls_top * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * U_bot * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * U_bot * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z
                + 15 * Ls_bot * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * U_top * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 6 * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * V_bot * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * V_bot * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z
                + 15 * Ls_bot * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * V_top * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 6 * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * h[2] * eta * h[0] ** 3 * q[2] * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * h[0] ** 3 * q[1] * v2 * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * h[0] ** 3 * q[2] * v2 * q[0]
                + 120 * Ls_bot * Ls_top * dqy[0] * eta * h[0] ** 3 * q[2] * z
                + 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                + 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                - 120 * Ls_bot * Ls_top * dqy[2] * eta * h[0] ** 3 * q[0] * z
                - 60 * Ls_bot * Ls_top * dqx[1] * h[0] ** 3 * v2 * q[0] * z
                - 60 * Ls_bot * Ls_top * dqy[2] * h[0] ** 3 * v2 * q[0] * z
                + 24 * Ls_top * h[2] * eta * h[0] ** 3 * q[2] * q[0] * z
                + 12 * Ls_top * h[1] * h[0] ** 3 * q[1] * v2 * q[0] * z
                + 12 * Ls_top * h[2] * h[0] ** 3 * q[2] * v2 * q[0] * z
                - 24 * Ls_bot * Ls_top * V_bot * h[2] * eta * h[0] ** 3 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2
                + 48 * Ls_bot * Ls_top ** 2 * h[2] * eta * h[0] ** 2 * q[2] * q[0]
                - 24 * Ls_bot ** 2 * Ls_top * h[2] * eta * h[0] ** 2 * q[2] * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0]
                - 120 * Ls_bot * Ls_top * dqy[0] * eta * h[0] ** 2 * q[2] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[0] * eta * h[0] * q[2] * z ** 2
                + 144 * Ls_bot * Ls_top ** 2 * dqy[0] * eta * h[0] ** 2 * q[2] * z
                - 72 * Ls_bot ** 2 * Ls_top * dqy[0] * eta * h[0] * q[2] * z ** 2
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[1] * v2 * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[1] * v2 * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z
                - 36 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[1] * v2 * z ** 2
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[2] * v2 * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[2] * v2 * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z
                - 36 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[2] * v2 * z ** 2
                + 120 * Ls_bot * Ls_top * dqy[2] * eta * h[0] ** 2 * q[0] * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqy[2] * eta * h[0] * q[0] * z ** 2
                - 144 * Ls_bot * Ls_top ** 2 * dqy[2] * eta * h[0] ** 2 * q[0] * z
                + 72 * Ls_bot ** 2 * Ls_top * dqy[2] * eta * h[0] * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * h[2] * eta * q[2] * q[0] * z ** 2
                - 72 * Ls_bot ** 2 * Ls_top * h[2] * eta * q[2] * q[0] * z ** 2
                + 60 * Ls_bot * Ls_top * dqx[1] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] * v2 * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] ** 2 * v2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] * v2 * q[0] * z ** 2
                + 60 * Ls_bot * Ls_top * dqy[2] * h[0] ** 2 * v2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] * v2 * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] ** 2 * v2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] * v2 * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[1] * q[1] * v2 * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[1] * q[1] * v2 * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[2] * q[2] * v2 * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[2] * q[2] * v2 * q[0] * z ** 2
                - 24 * Ls_top * V_bot * h[2] * eta * h[0] ** 3 * q[0] ** 2 * z
                - 12 * Ls_top * U_bot * h[1] * h[0] ** 3 * v2 * q[0] ** 2 * z
                - 12 * Ls_top * V_bot * h[2] * h[0] ** 3 * v2 * q[0] ** 2 * z
                - 42 * Ls_bot * h[2] * eta * h[0] ** 2 * q[2] * q[0] * z ** 2
                - 48 * Ls_bot ** 2 * h[2] * eta * h[0] * q[2] * q[0] * z ** 2
                - 42 * Ls_top * h[2] * eta * h[0] ** 2 * q[2] * q[0] * z ** 2
                - 48 * Ls_top ** 2 * h[2] * eta * h[0] * q[2] * q[0] * z ** 2
                + 48 * Ls_top ** 2 * h[2] * eta * h[0] ** 2 * q[2] * q[0] * z
                - 21 * Ls_bot * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[1] * v2 * q[0] * z ** 2
                - 21 * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[1] * h[0] * q[1] * v2 * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z
                - 21 * Ls_bot * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[2] * v2 * q[0] * z ** 2
                - 21 * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[2] * h[0] * q[2] * v2 * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z
                + 48 * Ls_bot * Ls_top * V_bot * h[2] * eta * h[0] * q[0] ** 2 * z ** 2
                + 48 * Ls_bot * Ls_top * V_top * h[2] * eta * h[0] * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * V_top * h[2] * eta * h[0] ** 2 * q[0] ** 2 * z
                + 24 * Ls_bot * Ls_top * U_bot * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * U_top * h[1] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * U_top * h[1] * h[0] ** 2 * v2 * q[0] ** 2 * z
                + 24 * Ls_bot * Ls_top * V_bot * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * V_top * h[2] * h[0] * v2 * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * V_top * h[2] * h[0] ** 2 * v2 * q[0] ** 2 * z
                - 96 * Ls_bot * Ls_top * h[2] * eta * h[0] * q[2] * q[0] * z ** 2
                - 24 * Ls_bot * Ls_top * h[2] * eta * h[0] ** 2 * q[2] * q[0] * z
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[1] * v2 * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[1] * h[0] ** 2 * q[1] * v2 * q[0] * z
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[2] * v2 * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[2] * h[0] ** 2 * q[2] * v2 * q[0] * z
            ))/(h[0] ** 2 * q[0] ** 2 * D ** 2)

    # tau_zz(z)
    tau_zz = -(2 * v2 * (
                3 * Ls_bot * dqx[0] * h[0] ** 5 * q[1]
                + 3 * Ls_bot * dqy[0] * h[0] ** 5 * q[2]
                - 3 * Ls_bot * dqx[1] * h[0] ** 5 * q[0]
                - 3 * Ls_bot * dqy[2] * h[0] ** 5 * q[0]
                + 3 * dqx[0] * h[0] ** 5 * q[1] * z
                + 3 * dqy[0] * h[0] ** 5 * q[2] * z
                - 3 * dqx[1] * h[0] ** 5 * q[0] * z
                - 3 * dqy[2] * h[0] ** 5 * q[0] * z
                + 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 4 * q[1]
                + 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 4 * q[2]
                - 12 * Ls_bot ** 2 * dqx[1] * h[0] ** 4 * q[0]
                - 12 * Ls_bot ** 2 * dqy[2] * h[0] ** 4 * q[0]
                - 3 * dqx[0] * h[0] ** 4 * q[1] * z ** 2
                - 3 * dqy[0] * h[0] ** 4 * q[2] * z ** 2
                + 3 * dqx[1] * h[0] ** 4 * q[0] * z ** 2
                + 3 * dqy[2] * h[0] ** 4 * q[0] * z ** 2
                - Ls_bot * V_top * h[2] * h[0] ** 4 * q[0] ** 2
                - 15 * Ls_bot * dqx[0] * h[0] ** 3 * q[1] * z ** 2
                - 15 * Ls_top * dqx[0] * h[0] ** 3 * q[1] * z ** 2
                + 24 * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[1] * z
                - 15 * Ls_bot * dqy[0] * h[0] ** 3 * q[2] * z ** 2
                - 15 * Ls_top * dqy[0] * h[0] ** 3 * q[2] * z ** 2
                + 24 * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[2] * z
                + 15 * Ls_bot * dqx[1] * h[0] ** 3 * q[0] * z ** 2
                + 15 * Ls_bot * dqy[2] * h[0] ** 3 * q[0] * z ** 2
                + 15 * Ls_top * dqx[1] * h[0] ** 3 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqx[1] * h[0] ** 3 * q[0] * z
                + 15 * Ls_top * dqy[2] * h[0] ** 3 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqy[2] * h[0] ** 3 * q[0] * z
                - 2 * U_bot * h[1] * h[0] ** 4 * q[0] ** 2 * z
                - U_top * h[1] * h[0] ** 4 * q[0] ** 2 * z
                - 2 * V_bot * h[2] * h[0] ** 4 * q[0] ** 2 * z
                - V_top * h[2] * h[0] ** 4 * q[0] ** 2 * z
                - 6 * h[1] * h[0] ** 3 * q[1] * q[0] * z ** 2
                - 6 * h[2] * h[0] ** 3 * q[2] * q[0] * z ** 2
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * q[0]
                - 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 2 * q[1] * z ** 2
                - 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * z ** 2
                - 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 2 * q[2] * z ** 2
                - 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * z ** 2
                + 12 * Ls_bot ** 2 * dqx[1] * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_bot ** 2 * dqy[2] * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqx[1] * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqy[2] * h[0] ** 2 * q[0] * z ** 2
                + 3 * U_bot * h[1] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 3 * U_top * h[1] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 3 * V_bot * h[2] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 3 * V_top * h[2] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 4 * q[1]
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 4 * q[2]
                - 18 * Ls_bot * Ls_top * dqx[1] * h[0] ** 4 * q[0]
                - 18 * Ls_bot * Ls_top * dqy[2] * h[0] ** 4 * q[0]
                + 3 * Ls_bot * h[1] * h[0] ** 4 * q[1] * q[0]
                + 3 * Ls_bot * h[2] * h[0] ** 4 * q[2] * q[0]
                + 12 * Ls_bot * dqx[0] * h[0] ** 4 * q[1] * z
                + 18 * Ls_top * dqx[0] * h[0] ** 4 * q[1] * z
                + 12 * Ls_bot * dqy[0] * h[0] ** 4 * q[2] * z
                + 18 * Ls_top * dqy[0] * h[0] ** 4 * q[2] * z
                - 12 * Ls_bot * dqx[1] * h[0] ** 4 * q[0] * z
                - 12 * Ls_bot * dqy[2] * h[0] ** 4 * q[0] * z
                - 18 * Ls_top * dqx[1] * h[0] ** 4 * q[0] * z
                - 18 * Ls_top * dqy[2] * h[0] ** 4 * q[0] * z
                + 3 * h[1] * h[0] ** 4 * q[1] * q[0] * z
                + 3 * h[2] * h[0] ** 4 * q[2] * q[0] * z
                + 24 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[1]
                + 60 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] ** 3 * q[1]
                + 24 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[2]
                + 60 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] ** 3 * q[2]
                - 24 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] ** 3 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] ** 3 * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] ** 3 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] ** 3 * q[0]
                - 2 * Ls_bot * U_bot * h[1] * h[0] ** 4 * q[0] ** 2
                - Ls_bot * U_top * h[1] * h[0] ** 4 * q[0] ** 2
                - 2 * Ls_bot * V_bot * h[2] * h[0] ** 4 * q[0] ** 2
                + 12 * Ls_bot * Ls_top * h[1] * h[0] ** 3 * q[1] * q[0]
                + 12 * Ls_bot * Ls_top * h[2] * h[0] ** 3 * q[2] * q[0]
                + 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 3 * q[1] * z
                + 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 3 * q[2] * z
                - 60 * Ls_bot * Ls_top * dqx[1] * h[0] ** 3 * q[0] * z
                - 60 * Ls_bot * Ls_top * dqy[2] * h[0] ** 3 * q[0] * z
                + 12 * Ls_top * h[1] * h[0] ** 3 * q[1] * q[0] * z
                + 12 * Ls_top * h[2] * h[0] ** 3 * q[2] * q[0] * z
                - 12 * Ls_bot * Ls_top * U_bot * h[1] * h[0] ** 3 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_bot * h[2] * h[0] ** 3 * q[0] ** 2
                + 24 * Ls_bot * Ls_top ** 2 * h[1] * h[0] ** 2 * q[1] * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[1] * h[0] ** 2 * q[1] * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * h[2] * h[0] ** 2 * q[2] * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[2] * h[0] ** 2 * q[2] * q[0]
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[1] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[1] * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[1] * z
                - 36 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[1] * z ** 2
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[2] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[2] * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[2] * z
                - 36 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[2] * z ** 2
                + 60 * Ls_bot * Ls_top * dqx[1] * h[0] ** 2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqx[1] * h[0] ** 2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqx[1] * h[0] * q[0] * z ** 2
                + 60 * Ls_bot * Ls_top * dqy[2] * h[0] ** 2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[2] * h[0] ** 2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqy[2] * h[0] * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[1] * q[1] * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[1] * q[1] * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[2] * q[2] * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[2] * q[2] * q[0] * z ** 2
                - 12 * Ls_top * U_bot * h[1] * h[0] ** 3 * q[0] ** 2 * z
                - 12 * Ls_top * V_bot * h[2] * h[0] ** 3 * q[0] ** 2 * z
                - 21 * Ls_bot * h[1] * h[0] ** 2 * q[1] * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[1] * q[0] * z ** 2
                - 21 * Ls_top * h[1] * h[0] ** 2 * q[1] * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[1] * h[0] * q[1] * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[1] * h[0] ** 2 * q[1] * q[0] * z
                - 21 * Ls_bot * h[2] * h[0] ** 2 * q[2] * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[2] * q[0] * z ** 2
                - 21 * Ls_top * h[2] * h[0] ** 2 * q[2] * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[2] * h[0] * q[2] * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[2] * h[0] ** 2 * q[2] * q[0] * z
                - 24 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * U_top * h[1] * h[0] ** 2 * q[0] ** 2
                - 24 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * V_top * h[2] * h[0] ** 2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top ** 2 * U_bot * h[1] * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * U_top * h[1] * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * V_bot * h[2] * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * V_top * h[2] * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * U_bot * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * U_bot * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * U_bot * h[1] * h[0] * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * U_bot * h[1] * h[0] ** 2 * q[0] ** 2 * z
                + 15 * Ls_bot * U_top * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * U_top * h[1] * h[0] * q[0] ** 2 * z ** 2
                + 6 * Ls_top * U_top * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * V_bot * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * V_bot * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * V_bot * h[2] * h[0] * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * V_bot * h[2] * h[0] ** 2 * q[0] ** 2 * z
                + 15 * Ls_bot * V_top * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * V_top * h[2] * h[0] * q[0] ** 2 * z ** 2
                + 6 * Ls_top * V_top * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[1] * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[1] * h[0] ** 2 * q[1] * q[0] * z
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[2] * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[2] * h[0] ** 2 * q[2] * q[0] * z
                + 24 * Ls_bot * Ls_top * U_bot * h[1] * h[0] * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * U_top * h[1] * h[0] * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * U_top * h[1] * h[0] ** 2 * q[0] ** 2 * z
                + 24 * Ls_bot * Ls_top * V_bot * h[2] * h[0] * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * V_top * h[2] * h[0] * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * V_top * h[2] * h[0] ** 2 * q[0] ** 2 * z
            ))/(h[0] ** 2 * q[0] ** 2 * D ** 2)

    # tau_yz(z)
    tau_yz = (2 * eta * (
                3 * h[0] ** 2 * q[2]
                + 6 * Ls_top * h[0] * q[2]
                - 6 * Ls_bot * q[2] * z
                - 6 * Ls_top * q[2] * z
                - 6 * h[0] * q[2] * z
                - 2 * V_bot * h[0] ** 2 * q[0]
                - V_top * h[0] ** 2 * q[0]
                + 3 * V_bot * h[0] * q[0] * z
                + 3 * V_top * h[0] * q[0] * z
                - 6 * Ls_top * V_bot * h[0] * q[0]
                + 6 * Ls_top * V_bot * q[0] * z
                + 6 * Ls_bot * V_top * q[0] * z
            ))/(h[0] * q[0] * D)

    # tau_xz(z)
    tau_xz = (2 * eta * (
                3 * h[0] ** 2 * q[1]
                + 6 * Ls_top * h[0] * q[1]
                - 6 * Ls_bot * q[1] * z
                - 6 * Ls_top * q[1] * z
                - 6 * h[0] * q[1] * z
                - 2 * U_bot * h[0] ** 2 * q[0]
                - U_top * h[0] ** 2 * q[0]
                + 3 * U_bot * h[0] * q[0] * z
                + 3 * U_top * h[0] * q[0] * z
                - 6 * Ls_top * U_bot * h[0] * q[0]
                + 6 * Ls_top * U_bot * q[0] * z
                + 6 * Ls_bot * U_top * q[0] * z
            ))/(h[0] * q[0] * D)

    # tau_xy(z)
    tau_xy = -(2 * eta * (
                3 * Ls_bot * dqy[0] * h[0] ** 5 * q[1]
                + 3 * Ls_bot * dqx[0] * h[0] ** 5 * q[2]
                - 3 * Ls_bot * dqy[1] * h[0] ** 5 * q[0]
                - 3 * Ls_bot * dqx[2] * h[0] ** 5 * q[0]
                + 3 * dqy[0] * h[0] ** 5 * q[1] * z
                + 3 * dqx[0] * h[0] ** 5 * q[2] * z
                - 3 * dqy[1] * h[0] ** 5 * q[0] * z
                - 3 * dqx[2] * h[0] ** 5 * q[0] * z
                + 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 4 * q[1]
                + 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 4 * q[2]
                - 12 * Ls_bot ** 2 * dqy[1] * h[0] ** 4 * q[0]
                - 12 * Ls_bot ** 2 * dqx[2] * h[0] ** 4 * q[0]
                - 3 * dqy[0] * h[0] ** 4 * q[1] * z ** 2
                - 3 * dqx[0] * h[0] ** 4 * q[2] * z ** 2
                + 3 * dqy[1] * h[0] ** 4 * q[0] * z ** 2
                + 3 * dqx[2] * h[0] ** 4 * q[0] * z ** 2
                - Ls_bot * V_top * h[1] * h[0] ** 4 * q[0] ** 2
                - 15 * Ls_bot * dqy[0] * h[0] ** 3 * q[1] * z ** 2
                - 15 * Ls_top * dqy[0] * h[0] ** 3 * q[1] * z ** 2
                + 24 * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[1] * z
                - 15 * Ls_bot * dqx[0] * h[0] ** 3 * q[2] * z ** 2
                - 15 * Ls_top * dqx[0] * h[0] ** 3 * q[2] * z ** 2
                + 24 * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[2] * z
                + 15 * Ls_bot * dqy[1] * h[0] ** 3 * q[0] * z ** 2
                + 15 * Ls_bot * dqx[2] * h[0] ** 3 * q[0] * z ** 2
                + 15 * Ls_top * dqy[1] * h[0] ** 3 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqy[1] * h[0] ** 3 * q[0] * z
                + 15 * Ls_top * dqx[2] * h[0] ** 3 * q[0] * z ** 2
                - 24 * Ls_top ** 2 * dqx[2] * h[0] ** 3 * q[0] * z
                - 2 * U_bot * h[2] * h[0] ** 4 * q[0] ** 2 * z
                - U_top * h[2] * h[0] ** 4 * q[0] ** 2 * z
                - 2 * V_bot * h[1] * h[0] ** 4 * q[0] ** 2 * z
                - V_top * h[1] * h[0] ** 4 * q[0] ** 2 * z
                - 6 * h[2] * h[0] ** 3 * q[1] * q[0] * z ** 2
                - 6 * h[1] * h[0] ** 3 * q[2] * q[0] * z ** 2
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[1]
                + 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[2]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqy[1] * h[0] ** 2 * q[0]
                - 72 * Ls_bot ** 2 * Ls_top ** 2 * dqx[2] * h[0] ** 2 * q[0]
                - 12 * Ls_bot ** 2 * dqy[0] * h[0] ** 2 * q[1] * z ** 2
                - 12 * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[1] * z ** 2
                - 12 * Ls_bot ** 2 * dqx[0] * h[0] ** 2 * q[2] * z ** 2
                - 12 * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[2] * z ** 2
                + 12 * Ls_bot ** 2 * dqy[1] * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_bot ** 2 * dqx[2] * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqy[1] * h[0] ** 2 * q[0] * z ** 2
                + 12 * Ls_top ** 2 * dqx[2] * h[0] ** 2 * q[0] * z ** 2
                + 3 * U_bot * h[2] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 3 * U_top * h[2] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 3 * V_bot * h[1] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 3 * V_top * h[1] * h[0] ** 3 * q[0] ** 2 * z ** 2
                + 18 * Ls_bot * Ls_top * dqy[0] * h[0] ** 4 * q[1]
                + 18 * Ls_bot * Ls_top * dqx[0] * h[0] ** 4 * q[2]
                - 18 * Ls_bot * Ls_top * dqy[1] * h[0] ** 4 * q[0]
                - 18 * Ls_bot * Ls_top * dqx[2] * h[0] ** 4 * q[0]
                + 3 * Ls_bot * h[2] * h[0] ** 4 * q[1] * q[0]
                + 3 * Ls_bot * h[1] * h[0] ** 4 * q[2] * q[0]
                + 12 * Ls_bot * dqy[0] * h[0] ** 4 * q[1] * z
                + 18 * Ls_top * dqy[0] * h[0] ** 4 * q[1] * z
                + 12 * Ls_bot * dqx[0] * h[0] ** 4 * q[2] * z
                + 18 * Ls_top * dqx[0] * h[0] ** 4 * q[2] * z
                - 12 * Ls_bot * dqy[1] * h[0] ** 4 * q[0] * z
                - 12 * Ls_bot * dqx[2] * h[0] ** 4 * q[0] * z
                - 18 * Ls_top * dqy[1] * h[0] ** 4 * q[0] * z
                - 18 * Ls_top * dqx[2] * h[0] ** 4 * q[0] * z
                + 3 * h[2] * h[0] ** 4 * q[1] * q[0] * z
                + 3 * h[1] * h[0] ** 4 * q[2] * q[0] * z
                + 24 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 3 * q[1]
                + 60 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] ** 3 * q[1]
                + 24 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 3 * q[2]
                + 60 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] ** 3 * q[2]
                - 24 * Ls_bot * Ls_top ** 2 * dqy[1] * h[0] ** 3 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqy[1] * h[0] ** 3 * q[0]
                - 24 * Ls_bot * Ls_top ** 2 * dqx[2] * h[0] ** 3 * q[0]
                - 60 * Ls_bot ** 2 * Ls_top * dqx[2] * h[0] ** 3 * q[0]
                - 2 * Ls_bot * U_bot * h[2] * h[0] ** 4 * q[0] ** 2
                - Ls_bot * U_top * h[2] * h[0] ** 4 * q[0] ** 2
                - 2 * Ls_bot * V_bot * h[1] * h[0] ** 4 * q[0] ** 2
                + 12 * Ls_bot * Ls_top * h[2] * h[0] ** 3 * q[1] * q[0]
                + 12 * Ls_bot * Ls_top * h[1] * h[0] ** 3 * q[2] * q[0]
                + 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 3 * q[1] * z
                + 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 3 * q[2] * z
                - 60 * Ls_bot * Ls_top * dqy[1] * h[0] ** 3 * q[0] * z
                - 60 * Ls_bot * Ls_top * dqx[2] * h[0] ** 3 * q[0] * z
                + 12 * Ls_top * h[2] * h[0] ** 3 * q[1] * q[0] * z
                + 12 * Ls_top * h[1] * h[0] ** 3 * q[2] * q[0] * z
                - 12 * Ls_bot * Ls_top * U_bot * h[2] * h[0] ** 3 * q[0] ** 2
                - 12 * Ls_bot * Ls_top * V_bot * h[1] * h[0] ** 3 * q[0] ** 2
                + 24 * Ls_bot * Ls_top ** 2 * h[2] * h[0] ** 2 * q[1] * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[2] * h[0] ** 2 * q[1] * q[0]
                + 24 * Ls_bot * Ls_top ** 2 * h[1] * h[0] ** 2 * q[2] * q[0]
                - 12 * Ls_bot ** 2 * Ls_top * h[1] * h[0] ** 2 * q[2] * q[0]
                - 60 * Ls_bot * Ls_top * dqy[0] * h[0] ** 2 * q[1] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] * q[1] * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqy[0] * h[0] ** 2 * q[1] * z
                - 36 * Ls_bot ** 2 * Ls_top * dqy[0] * h[0] * q[1] * z ** 2
                - 60 * Ls_bot * Ls_top * dqx[0] * h[0] ** 2 * q[2] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] * q[2] * z ** 2
                + 72 * Ls_bot * Ls_top ** 2 * dqx[0] * h[0] ** 2 * q[2] * z
                - 36 * Ls_bot ** 2 * Ls_top * dqx[0] * h[0] * q[2] * z ** 2
                + 60 * Ls_bot * Ls_top * dqy[1] * h[0] ** 2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqy[1] * h[0] * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqy[1] * h[0] ** 2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqy[1] * h[0] * q[0] * z ** 2
                + 60 * Ls_bot * Ls_top * dqx[2] * h[0] ** 2 * q[0] * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * dqx[2] * h[0] * q[0] * z ** 2
                - 72 * Ls_bot * Ls_top ** 2 * dqx[2] * h[0] ** 2 * q[0] * z
                + 36 * Ls_bot ** 2 * Ls_top * dqx[2] * h[0] * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[2] * q[1] * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[2] * q[1] * q[0] * z ** 2
                - 36 * Ls_bot * Ls_top ** 2 * h[1] * q[2] * q[0] * z ** 2
                - 36 * Ls_bot ** 2 * Ls_top * h[1] * q[2] * q[0] * z ** 2
                - 12 * Ls_top * U_bot * h[2] * h[0] ** 3 * q[0] ** 2 * z
                - 12 * Ls_top * V_bot * h[1] * h[0] ** 3 * q[0] ** 2 * z
                - 21 * Ls_bot * h[2] * h[0] ** 2 * q[1] * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[2] * h[0] * q[1] * q[0] * z ** 2
                - 21 * Ls_top * h[2] * h[0] ** 2 * q[1] * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[2] * h[0] * q[1] * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[2] * h[0] ** 2 * q[1] * q[0] * z
                - 21 * Ls_bot * h[1] * h[0] ** 2 * q[2] * q[0] * z ** 2
                - 24 * Ls_bot ** 2 * h[1] * h[0] * q[2] * q[0] * z ** 2
                - 21 * Ls_top * h[1] * h[0] ** 2 * q[2] * q[0] * z ** 2
                - 24 * Ls_top ** 2 * h[1] * h[0] * q[2] * q[0] * z ** 2
                + 24 * Ls_top ** 2 * h[1] * h[0] ** 2 * q[2] * q[0] * z
                - 24 * Ls_bot * Ls_top ** 2 * U_bot * h[2] * h[0] ** 2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * U_top * h[2] * h[0] ** 2 * q[0] ** 2
                - 24 * Ls_bot * Ls_top ** 2 * V_bot * h[1] * h[0] ** 2 * q[0] ** 2
                + 12 * Ls_bot ** 2 * Ls_top * V_top * h[1] * h[0] ** 2 * q[0] ** 2
                + 36 * Ls_bot * Ls_top ** 2 * U_bot * h[2] * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * U_top * h[2] * q[0] ** 2 * z ** 2
                + 36 * Ls_bot * Ls_top ** 2 * V_bot * h[1] * q[0] ** 2 * z ** 2
                + 36 * Ls_bot ** 2 * Ls_top * V_top * h[1] * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * U_bot * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * U_bot * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * U_bot * h[2] * h[0] * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * U_bot * h[2] * h[0] ** 2 * q[0] ** 2 * z
                + 15 * Ls_bot * U_top * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * U_top * h[2] * h[0] * q[0] ** 2 * z ** 2
                + 6 * Ls_top * U_top * h[2] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 6 * Ls_bot * V_bot * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 15 * Ls_top * V_bot * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_top ** 2 * V_bot * h[1] * h[0] * q[0] ** 2 * z ** 2
                - 24 * Ls_top ** 2 * V_bot * h[1] * h[0] ** 2 * q[0] ** 2 * z
                + 15 * Ls_bot * V_top * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                + 24 * Ls_bot ** 2 * V_top * h[1] * h[0] * q[0] ** 2 * z ** 2
                + 6 * Ls_top * V_top * h[1] * h[0] ** 2 * q[0] ** 2 * z ** 2
                - 48 * Ls_bot * Ls_top * h[2] * h[0] * q[1] * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[2] * h[0] ** 2 * q[1] * q[0] * z
                - 48 * Ls_bot * Ls_top * h[1] * h[0] * q[2] * q[0] * z ** 2
                - 12 * Ls_bot * Ls_top * h[1] * h[0] ** 2 * q[2] * q[0] * z
                + 24 * Ls_bot * Ls_top * U_bot * h[2] * h[0] * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * U_top * h[2] * h[0] * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * U_top * h[2] * h[0] ** 2 * q[0] ** 2 * z
                + 24 * Ls_bot * Ls_top * V_bot * h[1] * h[0] * q[0] ** 2 * z ** 2
                + 24 * Ls_bot * Ls_top * V_top * h[1] * h[0] * q[0] ** 2 * z ** 2
                + 12 * Ls_bot * Ls_top * V_top * h[1] * h[0] ** 2 * q[0] ** 2 * z
            ))/(h[0] ** 2 * q[0] ** 2 * D ** 2)

    return tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy

