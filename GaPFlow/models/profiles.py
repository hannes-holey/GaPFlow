#
# Copyright 2025 Hannes Holey
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

def get_velocity_profiles(z, q, Ls=0.0, U=1.0, V=0.0, slip="both"):
    """Velocity profiles for a given flow rate and wall velcoty

    Parameters
    ----------
    z : array-like
        Gap coordinate (z)
    q : array-like
        Height-averaged solution, (rho, jx, jy) for a single point (shape=(3,))
        or a field (shape=(3, nx, ny))
    Ls : float, optional
        Slip length (the default is 0.0, which means no-slip)
    U : float, optional
        Lower wall velocity in x direction (the default is 1.0)
    V : float, optional
        Lower wall velocity in y direction (the default is 1.0)
    slip : str, optional
        Type of slip boundary conditions ("both", "top", "bottom", or "none")
        (the default is "both", which means Ls applies to both the upper and lower wall)

    Returns
    -------
    array-like, array-like
        Discretized profiles u(z) and v(z)
    """

    h = z[-1]

    if slip == "both":
        u = (
            12 * Ls**2 * h * q[1]
            + 4 * Ls * U * h ** 2 * q[0]
            - 12 * Ls * U * h * q[0] * z
            + 6 * Ls * U * q[0] * z**2
            + 6 * Ls * h ** 2 * q[1]
            + 12 * Ls * h * q[1] * z
            - 12 * Ls * q[1] * z**2
            + U * h ** 3 * q[0]
            - 4 * U * h ** 2 * q[0] * z
            + 3 * U * h * q[0] * z**2
            + 6 * h ** 2 * q[1] * z
            - 6 * h * q[1] * z**2
        ) / (h * q[0] * (12 * Ls**2 + 8 * Ls * h + h ** 2))
        v = (
            12 * Ls**2 * h * q[2]
            + 4 * Ls * V * h ** 2 * q[0]
            - 12 * Ls * V * h * q[0] * z
            + 6 * Ls * V * q[0] * z**2
            + 6 * Ls * h ** 2 * q[2]
            + 12 * Ls * h * q[2] * z
            - 12 * Ls * q[2] * z**2
            + V * h ** 3 * q[0]
            - 4 * V * h ** 2 * q[0] * z
            + 3 * V * h * q[0] * z**2
            + 6 * h ** 2 * q[2] * z
            - 6 * h * q[2] * z**2
        ) / (h * q[0] * (12 * Ls**2 + 8 * Ls * h + h ** 2))
    elif slip == "top":
        u = (
            4 * Ls * U * h ** 2 * q[0]
            - 12 * Ls * U * h * q[0] * z
            + 6 * Ls * U * q[0] * z**2
            + 12 * Ls * h * q[1] * z
            - 6 * Ls * q[1] * z**2
            + U * h ** 3 * q[0]
            - 4 * U * h ** 2 * q[0] * z
            + 3 * U * h * q[0] * z**2
            + 6 * h ** 2 * q[1] * z
            - 6 * h * q[1] * z**2
        ) / (h ** 2 * q[0] * (4 * Ls + h))
        v = (
            4 * Ls * V * h ** 2 * q[0]
            - 12 * Ls * V * h * q[0] * z
            + 6 * Ls * V * q[0] * z**2
            + 12 * Ls * h * q[2] * z
            - 6 * Ls * q[2] * z**2
            + V * h ** 3 * q[0]
            - 4 * V * h ** 2 * q[0] * z
            + 3 * V * h * q[0] * z**2
            + 6 * h ** 2 * q[2] * z
            - 6 * h * q[2] * z**2
        ) / (h ** 2 * q[0] * (4 * Ls + h))
    elif slip == "bottom":
        u = (
            6 * Ls * h ** 2 * q[1]
            - 6 * Ls * q[1] * z**2
            + U * h ** 3 * q[0]
            - 4 * U * h ** 2 * q[0] * z
            + 3 * U * h * q[0] * z**2
            + 6 * h ** 2 * q[1] * z
            - 6 * h * q[1] * z**2
        ) / (h ** 2 * q[0] * (4 * Ls + h))
        v = (
            6 * Ls * h ** 2 * q[2]
            - 6 * Ls * q[2] * z**2
            + V * h ** 3 * q[0]
            - 4 * V * h ** 2 * q[0] * z
            + 3 * V * h * q[0] * z**2
            + 6 * h ** 2 * q[2] * z
            - 6 * h * q[2] * z**2
        ) / (h ** 2 * q[0] * (4 * Ls + h))
    elif slip == "none":
        u = (U * h ** 2 * q[0] - U * h * q[0] * z - 3 * z * (h - z) * (U * q[0] - 2 * q[1])) / (h ** 2 * q[0])
        v = (V * h ** 2 * q[0] - V * h * q[0] * z - 3 * z * (h - z) * (V * q[0] - 2 * q[2])) / (h ** 2 * q[0])

    return u, v


def get_stress_profiles(z, h, q, dqx, dqy, U=1.0, V=0.0, eta=1.0, zeta=1.0, Ls=0, mode="both"):
    """Viscous shear stress profiles for a given flow rate and wall velcoty

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
    Ls : float, optional
        Slip length (the default is 0.0, which means no-slip)
    U : float, optional
        Lower wall velocity in x direction (the default is 1.0)
    V : float, optional
        Lower wall velocity in y direction (the default is 1.0)
    slip : str, optional
        Type of slip boundary conditions ("both", "top", "bottom", or "none")
        (the default is "both", which means Ls applies to both the upper and lower wall)

    Returns
    -------
    Tuple[array-like, ...]
        Discretized shear stress profiles  τ_xx(z), τ_yy, τ_zz(z), τ_yz(z), τ_xz, τ_xy(z)
    """

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    if mode == "both":
        tau_xx = (
            2
            * (
                -72 * Ls**4 * dqx[0] * h[0] ** 2 * q[1] * v1
                + 72 * Ls**4 * dqx[1] * h[0] ** 2 * q[0] * v1
                - 72 * Ls**4 * dqy[0] * h[0] ** 2 * q[2] * v2
                + 72 * Ls**4 * dqy[2] * h[0] ** 2 * q[0] * v2
                + 24 * Ls**3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                - 36 * Ls**3 * U * h[1] * q[0] ** 2 * v1 * z**2
                + 24 * Ls**3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                - 36 * Ls**3 * V * h[2] * q[0] ** 2 * v2 * z**2
                - 84 * Ls**3 * dqx[0] * h[0] ** 3 * q[1] * v1
                - 72 * Ls**3 * dqx[0] * h[0] ** 2 * q[1] * v1 * z
                + 72 * Ls**3 * dqx[0] * h[0] * q[1] * v1 * z**2
                + 84 * Ls**3 * dqx[1] * h[0] ** 3 * q[0] * v1
                + 72 * Ls**3 * dqx[1] * h[0] ** 2 * q[0] * v1 * z
                - 72 * Ls**3 * dqx[1] * h[0] * q[0] * v1 * z**2
                - 84 * Ls**3 * dqy[0] * h[0] ** 3 * q[2] * v2
                - 72 * Ls**3 * dqy[0] * h[0] ** 2 * q[2] * v2 * z
                + 72 * Ls**3 * dqy[0] * h[0] * q[2] * v2 * z**2
                + 84 * Ls**3 * dqy[2] * h[0] ** 3 * q[0] * v2
                + 72 * Ls**3 * dqy[2] * h[0] ** 2 * q[0] * v2 * z
                - 72 * Ls**3 * dqy[2] * h[0] * q[0] * v2 * z**2
                - 12 * Ls**3 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                - 12 * Ls**3 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
                + 72 * Ls**3 * h[1] * q[0] * q[1] * v1 * z**2
                + 72 * Ls**3 * h[2] * q[0] * q[2] * v2 * z**2
                + 12 * Ls**2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1
                + 24 * Ls**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z
                - 48 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * v1 * z**2
                + 12 * Ls**2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2
                + 24 * Ls**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z
                - 48 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * v2 * z**2
                - 30 * Ls**2 * dqx[0] * h[0] ** 4 * q[1] * v1
                - 84 * Ls**2 * dqx[0] * h[0] ** 3 * q[1] * v1 * z
                + 84 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * v1 * z**2
                + 30 * Ls**2 * dqx[1] * h[0] ** 4 * q[0] * v1
                + 84 * Ls**2 * dqx[1] * h[0] ** 3 * q[0] * v1 * z
                - 84 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * v1 * z**2
                - 30 * Ls**2 * dqy[0] * h[0] ** 4 * q[2] * v2
                - 84 * Ls**2 * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                + 84 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * v2 * z**2
                + 30 * Ls**2 * dqy[2] * h[0] ** 4 * q[0] * v2
                + 84 * Ls**2 * dqy[2] * h[0] ** 3 * q[0] * v2 * z
                - 84 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * v2 * z**2
                - 12 * Ls**2 * h[0] ** 3 * h[1] * q[0] * q[1] * v1
                - 12 * Ls**2 * h[0] ** 3 * h[2] * q[0] * q[2] * v2
                - 12 * Ls**2 * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z
                - 12 * Ls**2 * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z
                + 96 * Ls**2 * h[0] * h[1] * q[0] * q[1] * v1 * z**2
                + 96 * Ls**2 * h[0] * h[2] * q[0] * q[2] * v2 * z**2
                + 2 * Ls * U * h[0] ** 4 * h[1] * q[0] ** 2 * v1
                + 12 * Ls * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1 * z
                - 21 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z**2
                + 2 * Ls * V * h[0] ** 4 * h[2] * q[0] ** 2 * v2
                + 12 * Ls * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2 * z
                - 21 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 5 * q[1] * v1
                - 30 * Ls * dqx[0] * h[0] ** 4 * q[1] * v1 * z
                + 30 * Ls * dqx[0] * h[0] ** 3 * q[1] * v1 * z**2
                + 3 * Ls * dqx[1] * h[0] ** 5 * q[0] * v1
                + 30 * Ls * dqx[1] * h[0] ** 4 * q[0] * v1 * z
                - 30 * Ls * dqx[1] * h[0] ** 3 * q[0] * v1 * z**2
                - 3 * Ls * dqy[0] * h[0] ** 5 * q[2] * v2
                - 30 * Ls * dqy[0] * h[0] ** 4 * q[2] * v2 * z
                + 30 * Ls * dqy[0] * h[0] ** 3 * q[2] * v2 * z**2
                + 3 * Ls * dqy[2] * h[0] ** 5 * q[0] * v2
                + 30 * Ls * dqy[2] * h[0] ** 4 * q[0] * v2 * z
                - 30 * Ls * dqy[2] * h[0] ** 3 * q[0] * v2 * z**2
                - 3 * Ls * h[0] ** 4 * h[1] * q[0] * q[1] * v1
                - 3 * Ls * h[0] ** 4 * h[2] * q[0] * q[2] * v2
                - 12 * Ls * h[0] ** 3 * h[1] * q[0] * q[1] * v1 * z
                - 12 * Ls * h[0] ** 3 * h[2] * q[0] * q[2] * v2 * z
                + 42 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z**2
                + 42 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z**2
                + 2 * U * h[0] ** 4 * h[1] * q[0] ** 2 * v1 * z
                - 3 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1 * z**2
                + 2 * V * h[0] ** 4 * h[2] * q[0] ** 2 * v2 * z
                - 3 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2 * z**2
                - 3 * dqx[0] * h[0] ** 5 * q[1] * v1 * z
                + 3 * dqx[0] * h[0] ** 4 * q[1] * v1 * z**2
                + 3 * dqx[1] * h[0] ** 5 * q[0] * v1 * z
                - 3 * dqx[1] * h[0] ** 4 * q[0] * v1 * z**2
                - 3 * dqy[0] * h[0] ** 5 * q[2] * v2 * z
                + 3 * dqy[0] * h[0] ** 4 * q[2] * v2 * z**2
                + 3 * dqy[2] * h[0] ** 5 * q[0] * v2 * z
                - 3 * dqy[2] * h[0] ** 4 * q[0] * v2 * z**2
                - 3 * h[0] ** 4 * h[1] * q[0] * q[1] * v1 * z
                - 3 * h[0] ** 4 * h[2] * q[0] * q[2] * v2 * z
                + 6 * h[0] ** 3 * h[1] * q[0] * q[1] * v1 * z**2
                + 6 * h[0] ** 3 * h[2] * q[0] * q[2] * v2 * z**2
            )
            / (
                h[0] ** 2
                * q[0] ** 2
                * (
                    144 * Ls**4
                    + 192 * Ls**3 * h[0]
                    + 88 * Ls**2 * h[0] ** 2
                    + 16 * Ls * h[0] ** 3
                    + h[0] ** 4
                )
            )
        )
        tau_yy = (
            2
            * (
                -72 * Ls**4 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 72 * Ls**4 * dqx[1] * h[0] ** 2 * q[0] * v2
                - 72 * Ls**4 * dqy[0] * h[0] ** 2 * q[2] * v1
                + 72 * Ls**4 * dqy[2] * h[0] ** 2 * q[0] * v1
                + 24 * Ls**3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                - 36 * Ls**3 * U * h[1] * q[0] ** 2 * v2 * z**2
                + 24 * Ls**3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                - 36 * Ls**3 * V * h[2] * q[0] ** 2 * v1 * z**2
                - 84 * Ls**3 * dqx[0] * h[0] ** 3 * q[1] * v2
                - 72 * Ls**3 * dqx[0] * h[0] ** 2 * q[1] * v2 * z
                + 72 * Ls**3 * dqx[0] * h[0] * q[1] * v2 * z**2
                + 84 * Ls**3 * dqx[1] * h[0] ** 3 * q[0] * v2
                + 72 * Ls**3 * dqx[1] * h[0] ** 2 * q[0] * v2 * z
                - 72 * Ls**3 * dqx[1] * h[0] * q[0] * v2 * z**2
                - 84 * Ls**3 * dqy[0] * h[0] ** 3 * q[2] * v1
                - 72 * Ls**3 * dqy[0] * h[0] ** 2 * q[2] * v1 * z
                + 72 * Ls**3 * dqy[0] * h[0] * q[2] * v1 * z**2
                + 84 * Ls**3 * dqy[2] * h[0] ** 3 * q[0] * v1
                + 72 * Ls**3 * dqy[2] * h[0] ** 2 * q[0] * v1 * z
                - 72 * Ls**3 * dqy[2] * h[0] * q[0] * v1 * z**2
                - 12 * Ls**3 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                - 12 * Ls**3 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
                + 72 * Ls**3 * h[1] * q[0] * q[1] * v2 * z**2
                + 72 * Ls**3 * h[2] * q[0] * q[2] * v1 * z**2
                + 12 * Ls**2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2
                + 24 * Ls**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z
                - 48 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * v2 * z**2
                + 12 * Ls**2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1
                + 24 * Ls**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z
                - 48 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * v1 * z**2
                - 30 * Ls**2 * dqx[0] * h[0] ** 4 * q[1] * v2
                - 84 * Ls**2 * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                + 84 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * v2 * z**2
                + 30 * Ls**2 * dqx[1] * h[0] ** 4 * q[0] * v2
                + 84 * Ls**2 * dqx[1] * h[0] ** 3 * q[0] * v2 * z
                - 84 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * v2 * z**2
                - 30 * Ls**2 * dqy[0] * h[0] ** 4 * q[2] * v1
                - 84 * Ls**2 * dqy[0] * h[0] ** 3 * q[2] * v1 * z
                + 84 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * v1 * z**2
                + 30 * Ls**2 * dqy[2] * h[0] ** 4 * q[0] * v1
                + 84 * Ls**2 * dqy[2] * h[0] ** 3 * q[0] * v1 * z
                - 84 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * v1 * z**2
                - 12 * Ls**2 * h[0] ** 3 * h[1] * q[0] * q[1] * v2
                - 12 * Ls**2 * h[0] ** 3 * h[2] * q[0] * q[2] * v1
                - 12 * Ls**2 * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z
                - 12 * Ls**2 * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z
                + 96 * Ls**2 * h[0] * h[1] * q[0] * q[1] * v2 * z**2
                + 96 * Ls**2 * h[0] * h[2] * q[0] * q[2] * v1 * z**2
                + 2 * Ls * U * h[0] ** 4 * h[1] * q[0] ** 2 * v2
                + 12 * Ls * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2 * z
                - 21 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z**2
                + 2 * Ls * V * h[0] ** 4 * h[2] * q[0] ** 2 * v1
                + 12 * Ls * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1 * z
                - 21 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 5 * q[1] * v2
                - 30 * Ls * dqx[0] * h[0] ** 4 * q[1] * v2 * z
                + 30 * Ls * dqx[0] * h[0] ** 3 * q[1] * v2 * z**2
                + 3 * Ls * dqx[1] * h[0] ** 5 * q[0] * v2
                + 30 * Ls * dqx[1] * h[0] ** 4 * q[0] * v2 * z
                - 30 * Ls * dqx[1] * h[0] ** 3 * q[0] * v2 * z**2
                - 3 * Ls * dqy[0] * h[0] ** 5 * q[2] * v1
                - 30 * Ls * dqy[0] * h[0] ** 4 * q[2] * v1 * z
                + 30 * Ls * dqy[0] * h[0] ** 3 * q[2] * v1 * z**2
                + 3 * Ls * dqy[2] * h[0] ** 5 * q[0] * v1
                + 30 * Ls * dqy[2] * h[0] ** 4 * q[0] * v1 * z
                - 30 * Ls * dqy[2] * h[0] ** 3 * q[0] * v1 * z**2
                - 3 * Ls * h[0] ** 4 * h[1] * q[0] * q[1] * v2
                - 3 * Ls * h[0] ** 4 * h[2] * q[0] * q[2] * v1
                - 12 * Ls * h[0] ** 3 * h[1] * q[0] * q[1] * v2 * z
                - 12 * Ls * h[0] ** 3 * h[2] * q[0] * q[2] * v1 * z
                + 42 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z**2
                + 42 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z**2
                + 2 * U * h[0] ** 4 * h[1] * q[0] ** 2 * v2 * z
                - 3 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2 * z**2
                + 2 * V * h[0] ** 4 * h[2] * q[0] ** 2 * v1 * z
                - 3 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1 * z**2
                - 3 * dqx[0] * h[0] ** 5 * q[1] * v2 * z
                + 3 * dqx[0] * h[0] ** 4 * q[1] * v2 * z**2
                + 3 * dqx[1] * h[0] ** 5 * q[0] * v2 * z
                - 3 * dqx[1] * h[0] ** 4 * q[0] * v2 * z**2
                - 3 * dqy[0] * h[0] ** 5 * q[2] * v1 * z
                + 3 * dqy[0] * h[0] ** 4 * q[2] * v1 * z**2
                + 3 * dqy[2] * h[0] ** 5 * q[0] * v1 * z
                - 3 * dqy[2] * h[0] ** 4 * q[0] * v1 * z**2
                - 3 * h[0] ** 4 * h[1] * q[0] * q[1] * v2 * z
                - 3 * h[0] ** 4 * h[2] * q[0] * q[2] * v1 * z
                + 6 * h[0] ** 3 * h[1] * q[0] * q[1] * v2 * z**2
                + 6 * h[0] ** 3 * h[2] * q[0] * q[2] * v1 * z**2
            )
            / (
                h[0] ** 2
                * q[0] ** 2
                * (
                    144 * Ls**4
                    + 192 * Ls**3 * h[0]
                    + 88 * Ls**2 * h[0] ** 2
                    + 16 * Ls * h[0] ** 3
                    + h[0] ** 4
                )
            )
        )
        tau_zz = (
            2
            * v2
            * (
                -72 * Ls**4 * dqx[0] * h[0] ** 2 * q[1]
                + 72 * Ls**4 * dqx[1] * h[0] ** 2 * q[0]
                - 72 * Ls**4 * dqy[0] * h[0] ** 2 * q[2]
                + 72 * Ls**4 * dqy[2] * h[0] ** 2 * q[0]
                + 24 * Ls**3 * U * h[0] ** 2 * h[1] * q[0] ** 2
                - 36 * Ls**3 * U * h[1] * q[0] ** 2 * z**2
                + 24 * Ls**3 * V * h[0] ** 2 * h[2] * q[0] ** 2
                - 36 * Ls**3 * V * h[2] * q[0] ** 2 * z**2
                - 84 * Ls**3 * dqx[0] * h[0] ** 3 * q[1]
                - 72 * Ls**3 * dqx[0] * h[0] ** 2 * q[1] * z
                + 72 * Ls**3 * dqx[0] * h[0] * q[1] * z**2
                + 84 * Ls**3 * dqx[1] * h[0] ** 3 * q[0]
                + 72 * Ls**3 * dqx[1] * h[0] ** 2 * q[0] * z
                - 72 * Ls**3 * dqx[1] * h[0] * q[0] * z**2
                - 84 * Ls**3 * dqy[0] * h[0] ** 3 * q[2]
                - 72 * Ls**3 * dqy[0] * h[0] ** 2 * q[2] * z
                + 72 * Ls**3 * dqy[0] * h[0] * q[2] * z**2
                + 84 * Ls**3 * dqy[2] * h[0] ** 3 * q[0]
                + 72 * Ls**3 * dqy[2] * h[0] ** 2 * q[0] * z
                - 72 * Ls**3 * dqy[2] * h[0] * q[0] * z**2
                - 12 * Ls**3 * h[0] ** 2 * h[1] * q[0] * q[1]
                - 12 * Ls**3 * h[0] ** 2 * h[2] * q[0] * q[2]
                + 72 * Ls**3 * h[1] * q[0] * q[1] * z**2
                + 72 * Ls**3 * h[2] * q[0] * q[2] * z**2
                + 12 * Ls**2 * U * h[0] ** 3 * h[1] * q[0] ** 2
                + 24 * Ls**2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * z
                - 48 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * z**2
                + 12 * Ls**2 * V * h[0] ** 3 * h[2] * q[0] ** 2
                + 24 * Ls**2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * z
                - 48 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * z**2
                - 30 * Ls**2 * dqx[0] * h[0] ** 4 * q[1]
                - 84 * Ls**2 * dqx[0] * h[0] ** 3 * q[1] * z
                + 84 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * z**2
                + 30 * Ls**2 * dqx[1] * h[0] ** 4 * q[0]
                + 84 * Ls**2 * dqx[1] * h[0] ** 3 * q[0] * z
                - 84 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * z**2
                - 30 * Ls**2 * dqy[0] * h[0] ** 4 * q[2]
                - 84 * Ls**2 * dqy[0] * h[0] ** 3 * q[2] * z
                + 84 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * z**2
                + 30 * Ls**2 * dqy[2] * h[0] ** 4 * q[0]
                + 84 * Ls**2 * dqy[2] * h[0] ** 3 * q[0] * z
                - 84 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * z**2
                - 12 * Ls**2 * h[0] ** 3 * h[1] * q[0] * q[1]
                - 12 * Ls**2 * h[0] ** 3 * h[2] * q[0] * q[2]
                - 12 * Ls**2 * h[0] ** 2 * h[1] * q[0] * q[1] * z
                - 12 * Ls**2 * h[0] ** 2 * h[2] * q[0] * q[2] * z
                + 96 * Ls**2 * h[0] * h[1] * q[0] * q[1] * z**2
                + 96 * Ls**2 * h[0] * h[2] * q[0] * q[2] * z**2
                + 2 * Ls * U * h[0] ** 4 * h[1] * q[0] ** 2
                + 12 * Ls * U * h[0] ** 3 * h[1] * q[0] ** 2 * z
                - 21 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
                + 2 * Ls * V * h[0] ** 4 * h[2] * q[0] ** 2
                + 12 * Ls * V * h[0] ** 3 * h[2] * q[0] ** 2 * z
                - 21 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 5 * q[1]
                - 30 * Ls * dqx[0] * h[0] ** 4 * q[1] * z
                + 30 * Ls * dqx[0] * h[0] ** 3 * q[1] * z**2
                + 3 * Ls * dqx[1] * h[0] ** 5 * q[0]
                + 30 * Ls * dqx[1] * h[0] ** 4 * q[0] * z
                - 30 * Ls * dqx[1] * h[0] ** 3 * q[0] * z**2
                - 3 * Ls * dqy[0] * h[0] ** 5 * q[2]
                - 30 * Ls * dqy[0] * h[0] ** 4 * q[2] * z
                + 30 * Ls * dqy[0] * h[0] ** 3 * q[2] * z**2
                + 3 * Ls * dqy[2] * h[0] ** 5 * q[0]
                + 30 * Ls * dqy[2] * h[0] ** 4 * q[0] * z
                - 30 * Ls * dqy[2] * h[0] ** 3 * q[0] * z**2
                - 3 * Ls * h[0] ** 4 * h[1] * q[0] * q[1]
                - 3 * Ls * h[0] ** 4 * h[2] * q[0] * q[2]
                - 12 * Ls * h[0] ** 3 * h[1] * q[0] * q[1] * z
                - 12 * Ls * h[0] ** 3 * h[2] * q[0] * q[2] * z
                + 42 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * z**2
                + 42 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * z**2
                + 2 * U * h[0] ** 4 * h[1] * q[0] ** 2 * z
                - 3 * U * h[0] ** 3 * h[1] * q[0] ** 2 * z**2
                + 2 * V * h[0] ** 4 * h[2] * q[0] ** 2 * z
                - 3 * V * h[0] ** 3 * h[2] * q[0] ** 2 * z**2
                - 3 * dqx[0] * h[0] ** 5 * q[1] * z
                + 3 * dqx[0] * h[0] ** 4 * q[1] * z**2
                + 3 * dqx[1] * h[0] ** 5 * q[0] * z
                - 3 * dqx[1] * h[0] ** 4 * q[0] * z**2
                - 3 * dqy[0] * h[0] ** 5 * q[2] * z
                + 3 * dqy[0] * h[0] ** 4 * q[2] * z**2
                + 3 * dqy[2] * h[0] ** 5 * q[0] * z
                - 3 * dqy[2] * h[0] ** 4 * q[0] * z**2
                - 3 * h[0] ** 4 * h[1] * q[0] * q[1] * z
                - 3 * h[0] ** 4 * h[2] * q[0] * q[2] * z
                + 6 * h[0] ** 3 * h[1] * q[0] * q[1] * z**2
                + 6 * h[0] ** 3 * h[2] * q[0] * q[2] * z**2
            )
            / (
                h[0] ** 2
                * q[0] ** 2
                * (
                    144 * Ls**4
                    + 192 * Ls**3 * h[0]
                    + 88 * Ls**2 * h[0] ** 2
                    + 16 * Ls * h[0] ** 3
                    + h[0] ** 4
                )
            )
        )
        tau_xy = (
            2
            * eta
            * (
                -72 * Ls**4 * dqx[0] * h[0] ** 2 * q[2]
                + 72 * Ls**4 * dqx[2] * h[0] ** 2 * q[0]
                - 72 * Ls**4 * dqy[0] * h[0] ** 2 * q[1]
                + 72 * Ls**4 * dqy[1] * h[0] ** 2 * q[0]
                + 24 * Ls**3 * U * h[0] ** 2 * h[2] * q[0] ** 2
                - 36 * Ls**3 * U * h[2] * q[0] ** 2 * z**2
                + 24 * Ls**3 * V * h[0] ** 2 * h[1] * q[0] ** 2
                - 36 * Ls**3 * V * h[1] * q[0] ** 2 * z**2
                - 84 * Ls**3 * dqx[0] * h[0] ** 3 * q[2]
                - 72 * Ls**3 * dqx[0] * h[0] ** 2 * q[2] * z
                + 72 * Ls**3 * dqx[0] * h[0] * q[2] * z**2
                + 84 * Ls**3 * dqx[2] * h[0] ** 3 * q[0]
                + 72 * Ls**3 * dqx[2] * h[0] ** 2 * q[0] * z
                - 72 * Ls**3 * dqx[2] * h[0] * q[0] * z**2
                - 84 * Ls**3 * dqy[0] * h[0] ** 3 * q[1]
                - 72 * Ls**3 * dqy[0] * h[0] ** 2 * q[1] * z
                + 72 * Ls**3 * dqy[0] * h[0] * q[1] * z**2
                + 84 * Ls**3 * dqy[1] * h[0] ** 3 * q[0]
                + 72 * Ls**3 * dqy[1] * h[0] ** 2 * q[0] * z
                - 72 * Ls**3 * dqy[1] * h[0] * q[0] * z**2
                - 12 * Ls**3 * h[0] ** 2 * h[1] * q[0] * q[2]
                - 12 * Ls**3 * h[0] ** 2 * h[2] * q[0] * q[1]
                + 72 * Ls**3 * h[1] * q[0] * q[2] * z**2
                + 72 * Ls**3 * h[2] * q[0] * q[1] * z**2
                + 12 * Ls**2 * U * h[0] ** 3 * h[2] * q[0] ** 2
                + 24 * Ls**2 * U * h[0] ** 2 * h[2] * q[0] ** 2 * z
                - 48 * Ls**2 * U * h[0] * h[2] * q[0] ** 2 * z**2
                + 12 * Ls**2 * V * h[0] ** 3 * h[1] * q[0] ** 2
                + 24 * Ls**2 * V * h[0] ** 2 * h[1] * q[0] ** 2 * z
                - 48 * Ls**2 * V * h[0] * h[1] * q[0] ** 2 * z**2
                - 30 * Ls**2 * dqx[0] * h[0] ** 4 * q[2]
                - 84 * Ls**2 * dqx[0] * h[0] ** 3 * q[2] * z
                + 84 * Ls**2 * dqx[0] * h[0] ** 2 * q[2] * z**2
                + 30 * Ls**2 * dqx[2] * h[0] ** 4 * q[0]
                + 84 * Ls**2 * dqx[2] * h[0] ** 3 * q[0] * z
                - 84 * Ls**2 * dqx[2] * h[0] ** 2 * q[0] * z**2
                - 30 * Ls**2 * dqy[0] * h[0] ** 4 * q[1]
                - 84 * Ls**2 * dqy[0] * h[0] ** 3 * q[1] * z
                + 84 * Ls**2 * dqy[0] * h[0] ** 2 * q[1] * z**2
                + 30 * Ls**2 * dqy[1] * h[0] ** 4 * q[0]
                + 84 * Ls**2 * dqy[1] * h[0] ** 3 * q[0] * z
                - 84 * Ls**2 * dqy[1] * h[0] ** 2 * q[0] * z**2
                - 12 * Ls**2 * h[0] ** 3 * h[1] * q[0] * q[2]
                - 12 * Ls**2 * h[0] ** 3 * h[2] * q[0] * q[1]
                - 12 * Ls**2 * h[0] ** 2 * h[1] * q[0] * q[2] * z
                - 12 * Ls**2 * h[0] ** 2 * h[2] * q[0] * q[1] * z
                + 96 * Ls**2 * h[0] * h[1] * q[0] * q[2] * z**2
                + 96 * Ls**2 * h[0] * h[2] * q[0] * q[1] * z**2
                + 2 * Ls * U * h[0] ** 4 * h[2] * q[0] ** 2
                + 12 * Ls * U * h[0] ** 3 * h[2] * q[0] ** 2 * z
                - 21 * Ls * U * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
                + 2 * Ls * V * h[0] ** 4 * h[1] * q[0] ** 2
                + 12 * Ls * V * h[0] ** 3 * h[1] * q[0] ** 2 * z
                - 21 * Ls * V * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 5 * q[2]
                - 30 * Ls * dqx[0] * h[0] ** 4 * q[2] * z
                + 30 * Ls * dqx[0] * h[0] ** 3 * q[2] * z**2
                + 3 * Ls * dqx[2] * h[0] ** 5 * q[0]
                + 30 * Ls * dqx[2] * h[0] ** 4 * q[0] * z
                - 30 * Ls * dqx[2] * h[0] ** 3 * q[0] * z**2
                - 3 * Ls * dqy[0] * h[0] ** 5 * q[1]
                - 30 * Ls * dqy[0] * h[0] ** 4 * q[1] * z
                + 30 * Ls * dqy[0] * h[0] ** 3 * q[1] * z**2
                + 3 * Ls * dqy[1] * h[0] ** 5 * q[0]
                + 30 * Ls * dqy[1] * h[0] ** 4 * q[0] * z
                - 30 * Ls * dqy[1] * h[0] ** 3 * q[0] * z**2
                - 3 * Ls * h[0] ** 4 * h[1] * q[0] * q[2]
                - 3 * Ls * h[0] ** 4 * h[2] * q[0] * q[1]
                - 12 * Ls * h[0] ** 3 * h[1] * q[0] * q[2] * z
                - 12 * Ls * h[0] ** 3 * h[2] * q[0] * q[1] * z
                + 42 * Ls * h[0] ** 2 * h[1] * q[0] * q[2] * z**2
                + 42 * Ls * h[0] ** 2 * h[2] * q[0] * q[1] * z**2
                + 2 * U * h[0] ** 4 * h[2] * q[0] ** 2 * z
                - 3 * U * h[0] ** 3 * h[2] * q[0] ** 2 * z**2
                + 2 * V * h[0] ** 4 * h[1] * q[0] ** 2 * z
                - 3 * V * h[0] ** 3 * h[1] * q[0] ** 2 * z**2
                - 3 * dqx[0] * h[0] ** 5 * q[2] * z
                + 3 * dqx[0] * h[0] ** 4 * q[2] * z**2
                + 3 * dqx[2] * h[0] ** 5 * q[0] * z
                - 3 * dqx[2] * h[0] ** 4 * q[0] * z**2
                - 3 * dqy[0] * h[0] ** 5 * q[1] * z
                + 3 * dqy[0] * h[0] ** 4 * q[1] * z**2
                + 3 * dqy[1] * h[0] ** 5 * q[0] * z
                - 3 * dqy[1] * h[0] ** 4 * q[0] * z**2
                - 3 * h[0] ** 4 * h[1] * q[0] * q[2] * z
                - 3 * h[0] ** 4 * h[2] * q[0] * q[1] * z
                + 6 * h[0] ** 3 * h[1] * q[0] * q[2] * z**2
                + 6 * h[0] ** 3 * h[2] * q[0] * q[1] * z**2
            )
            / (
                h[0] ** 2
                * q[0] ** 2
                * (
                    144 * Ls**4
                    + 192 * Ls**3 * h[0]
                    + 88 * Ls**2 * h[0] ** 2
                    + 16 * Ls * h[0] ** 3
                    + h[0] ** 4
                )
            )
        )
        tau_xz = (
            2
            * eta
            * (
                -6 * Ls * U * h[0] * q[0]
                + 6 * Ls * U * q[0] * z
                + 6 * Ls * h[0] * q[1]
                - 12 * Ls * q[1] * z
                - 2 * U * h[0] ** 2 * q[0]
                + 3 * U * h[0] * q[0] * z
                + 3 * h[0] ** 2 * q[1]
                - 6 * h[0] * q[1] * z
            )
            / (h[0] * q[0] * (12 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
        tau_yz = (
            2
            * eta
            * (
                -6 * Ls * V * h[0] * q[0]
                + 6 * Ls * V * q[0] * z
                + 6 * Ls * h[0] * q[2]
                - 12 * Ls * q[2] * z
                - 2 * V * h[0] ** 2 * q[0]
                + 3 * V * h[0] * q[0] * z
                + 3 * h[0] ** 2 * q[2]
                - 6 * h[0] * q[2] * z
            )
            / (h[0] * q[0] * (12 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
    elif mode == "bottom":
        tau_xx = (
            2
            * (
                -12 * Ls**2 * dqx[0] * h[0] ** 3 * q[1] * v1
                + 12 * Ls**2 * dqx[0] * h[0] * q[1] * v1 * z**2
                + 12 * Ls**2 * dqx[1] * h[0] ** 3 * q[0] * v1
                - 12 * Ls**2 * dqx[1] * h[0] * q[0] * v1 * z**2
                - 12 * Ls**2 * dqy[0] * h[0] ** 3 * q[2] * v2
                + 12 * Ls**2 * dqy[0] * h[0] * q[2] * v2 * z**2
                + 12 * Ls**2 * dqy[2] * h[0] ** 3 * q[0] * v2
                - 12 * Ls**2 * dqy[2] * h[0] * q[0] * v2 * z**2
                + 24 * Ls**2 * h[1] * q[0] * q[1] * v1 * z**2
                + 24 * Ls**2 * h[2] * q[0] * q[2] * v2 * z**2
                + 2 * Ls * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1
                - 6 * Ls * U * h[0] * h[1] * q[0] ** 2 * v1 * z**2
                + 2 * Ls * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2
                - 6 * Ls * V * h[0] * h[2] * q[0] ** 2 * v2 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 4 * q[1] * v1
                - 12 * Ls * dqx[0] * h[0] ** 3 * q[1] * v1 * z
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[1] * v1 * z**2
                + 3 * Ls * dqx[1] * h[0] ** 4 * q[0] * v1
                + 12 * Ls * dqx[1] * h[0] ** 3 * q[0] * v1 * z
                - 15 * Ls * dqx[1] * h[0] ** 2 * q[0] * v1 * z**2
                - 3 * Ls * dqy[0] * h[0] ** 4 * q[2] * v2
                - 12 * Ls * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[2] * v2 * z**2
                + 3 * Ls * dqy[2] * h[0] ** 4 * q[0] * v2
                + 12 * Ls * dqy[2] * h[0] ** 3 * q[0] * v2 * z
                - 15 * Ls * dqy[2] * h[0] ** 2 * q[0] * v2 * z**2
                - 3 * Ls * h[0] ** 3 * h[1] * q[0] * q[1] * v1
                - 3 * Ls * h[0] ** 3 * h[2] * q[0] * q[2] * v2
                + 21 * Ls * h[0] * h[1] * q[0] * q[1] * v1 * z**2
                + 21 * Ls * h[0] * h[2] * q[0] * q[2] * v2 * z**2
                + 2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1 * z
                - 3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z**2
                + 2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2 * z
                - 3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z**2
                - 3 * dqx[0] * h[0] ** 4 * q[1] * v1 * z
                + 3 * dqx[0] * h[0] ** 3 * q[1] * v1 * z**2
                + 3 * dqx[1] * h[0] ** 4 * q[0] * v1 * z
                - 3 * dqx[1] * h[0] ** 3 * q[0] * v1 * z**2
                - 3 * dqy[0] * h[0] ** 4 * q[2] * v2 * z
                + 3 * dqy[0] * h[0] ** 3 * q[2] * v2 * z**2
                + 3 * dqy[2] * h[0] ** 4 * q[0] * v2 * z
                - 3 * dqy[2] * h[0] ** 3 * q[0] * v2 * z**2
                - 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v1 * z
                - 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v2 * z
                + 6 * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z**2
                + 6 * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z**2
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_yy = (
            2
            * (
                -12 * Ls**2 * dqx[0] * h[0] ** 3 * q[1] * v2
                + 12 * Ls**2 * dqx[0] * h[0] * q[1] * v2 * z**2
                + 12 * Ls**2 * dqx[1] * h[0] ** 3 * q[0] * v2
                - 12 * Ls**2 * dqx[1] * h[0] * q[0] * v2 * z**2
                - 12 * Ls**2 * dqy[0] * h[0] ** 3 * q[2] * v1
                + 12 * Ls**2 * dqy[0] * h[0] * q[2] * v1 * z**2
                + 12 * Ls**2 * dqy[2] * h[0] ** 3 * q[0] * v1
                - 12 * Ls**2 * dqy[2] * h[0] * q[0] * v1 * z**2
                + 24 * Ls**2 * h[1] * q[0] * q[1] * v2 * z**2
                + 24 * Ls**2 * h[2] * q[0] * q[2] * v1 * z**2
                + 2 * Ls * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2
                - 6 * Ls * U * h[0] * h[1] * q[0] ** 2 * v2 * z**2
                + 2 * Ls * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1
                - 6 * Ls * V * h[0] * h[2] * q[0] ** 2 * v1 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 4 * q[1] * v2
                - 12 * Ls * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[1] * v2 * z**2
                + 3 * Ls * dqx[1] * h[0] ** 4 * q[0] * v2
                + 12 * Ls * dqx[1] * h[0] ** 3 * q[0] * v2 * z
                - 15 * Ls * dqx[1] * h[0] ** 2 * q[0] * v2 * z**2
                - 3 * Ls * dqy[0] * h[0] ** 4 * q[2] * v1
                - 12 * Ls * dqy[0] * h[0] ** 3 * q[2] * v1 * z
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[2] * v1 * z**2
                + 3 * Ls * dqy[2] * h[0] ** 4 * q[0] * v1
                + 12 * Ls * dqy[2] * h[0] ** 3 * q[0] * v1 * z
                - 15 * Ls * dqy[2] * h[0] ** 2 * q[0] * v1 * z**2
                - 3 * Ls * h[0] ** 3 * h[1] * q[0] * q[1] * v2
                - 3 * Ls * h[0] ** 3 * h[2] * q[0] * q[2] * v1
                + 21 * Ls * h[0] * h[1] * q[0] * q[1] * v2 * z**2
                + 21 * Ls * h[0] * h[2] * q[0] * q[2] * v1 * z**2
                + 2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2 * z
                - 3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z**2
                + 2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1 * z
                - 3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z**2
                - 3 * dqx[0] * h[0] ** 4 * q[1] * v2 * z
                + 3 * dqx[0] * h[0] ** 3 * q[1] * v2 * z**2
                + 3 * dqx[1] * h[0] ** 4 * q[0] * v2 * z
                - 3 * dqx[1] * h[0] ** 3 * q[0] * v2 * z**2
                - 3 * dqy[0] * h[0] ** 4 * q[2] * v1 * z
                + 3 * dqy[0] * h[0] ** 3 * q[2] * v1 * z**2
                + 3 * dqy[2] * h[0] ** 4 * q[0] * v1 * z
                - 3 * dqy[2] * h[0] ** 3 * q[0] * v1 * z**2
                - 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v2 * z
                - 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v1 * z
                + 6 * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z**2
                + 6 * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z**2
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_zz = (
            2
            * v2
            * (
                -12 * Ls**2 * dqx[0] * h[0] ** 3 * q[1]
                + 12 * Ls**2 * dqx[0] * h[0] * q[1] * z**2
                + 12 * Ls**2 * dqx[1] * h[0] ** 3 * q[0]
                - 12 * Ls**2 * dqx[1] * h[0] * q[0] * z**2
                - 12 * Ls**2 * dqy[0] * h[0] ** 3 * q[2]
                + 12 * Ls**2 * dqy[0] * h[0] * q[2] * z**2
                + 12 * Ls**2 * dqy[2] * h[0] ** 3 * q[0]
                - 12 * Ls**2 * dqy[2] * h[0] * q[0] * z**2
                + 24 * Ls**2 * h[1] * q[0] * q[1] * z**2
                + 24 * Ls**2 * h[2] * q[0] * q[2] * z**2
                + 2 * Ls * U * h[0] ** 3 * h[1] * q[0] ** 2
                - 6 * Ls * U * h[0] * h[1] * q[0] ** 2 * z**2
                + 2 * Ls * V * h[0] ** 3 * h[2] * q[0] ** 2
                - 6 * Ls * V * h[0] * h[2] * q[0] ** 2 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 4 * q[1]
                - 12 * Ls * dqx[0] * h[0] ** 3 * q[1] * z
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[1] * z**2
                + 3 * Ls * dqx[1] * h[0] ** 4 * q[0]
                + 12 * Ls * dqx[1] * h[0] ** 3 * q[0] * z
                - 15 * Ls * dqx[1] * h[0] ** 2 * q[0] * z**2
                - 3 * Ls * dqy[0] * h[0] ** 4 * q[2]
                - 12 * Ls * dqy[0] * h[0] ** 3 * q[2] * z
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[2] * z**2
                + 3 * Ls * dqy[2] * h[0] ** 4 * q[0]
                + 12 * Ls * dqy[2] * h[0] ** 3 * q[0] * z
                - 15 * Ls * dqy[2] * h[0] ** 2 * q[0] * z**2
                - 3 * Ls * h[0] ** 3 * h[1] * q[0] * q[1]
                - 3 * Ls * h[0] ** 3 * h[2] * q[0] * q[2]
                + 21 * Ls * h[0] * h[1] * q[0] * q[1] * z**2
                + 21 * Ls * h[0] * h[2] * q[0] * q[2] * z**2
                + 2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * z
                - 3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
                + 2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * z
                - 3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
                - 3 * dqx[0] * h[0] ** 4 * q[1] * z
                + 3 * dqx[0] * h[0] ** 3 * q[1] * z**2
                + 3 * dqx[1] * h[0] ** 4 * q[0] * z
                - 3 * dqx[1] * h[0] ** 3 * q[0] * z**2
                - 3 * dqy[0] * h[0] ** 4 * q[2] * z
                + 3 * dqy[0] * h[0] ** 3 * q[2] * z**2
                + 3 * dqy[2] * h[0] ** 4 * q[0] * z
                - 3 * dqy[2] * h[0] ** 3 * q[0] * z**2
                - 3 * h[0] ** 3 * h[1] * q[0] * q[1] * z
                - 3 * h[0] ** 3 * h[2] * q[0] * q[2] * z
                + 6 * h[0] ** 2 * h[1] * q[0] * q[1] * z**2
                + 6 * h[0] ** 2 * h[2] * q[0] * q[2] * z**2
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_xy = (
            2
            * eta
            * (
                -12 * Ls**2 * dqx[0] * h[0] ** 3 * q[2]
                + 12 * Ls**2 * dqx[0] * h[0] * q[2] * z**2
                + 12 * Ls**2 * dqx[2] * h[0] ** 3 * q[0]
                - 12 * Ls**2 * dqx[2] * h[0] * q[0] * z**2
                - 12 * Ls**2 * dqy[0] * h[0] ** 3 * q[1]
                + 12 * Ls**2 * dqy[0] * h[0] * q[1] * z**2
                + 12 * Ls**2 * dqy[1] * h[0] ** 3 * q[0]
                - 12 * Ls**2 * dqy[1] * h[0] * q[0] * z**2
                + 24 * Ls**2 * h[1] * q[0] * q[2] * z**2
                + 24 * Ls**2 * h[2] * q[0] * q[1] * z**2
                + 2 * Ls * U * h[0] ** 3 * h[2] * q[0] ** 2
                - 6 * Ls * U * h[0] * h[2] * q[0] ** 2 * z**2
                + 2 * Ls * V * h[0] ** 3 * h[1] * q[0] ** 2
                - 6 * Ls * V * h[0] * h[1] * q[0] ** 2 * z**2
                - 3 * Ls * dqx[0] * h[0] ** 4 * q[2]
                - 12 * Ls * dqx[0] * h[0] ** 3 * q[2] * z
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[2] * z**2
                + 3 * Ls * dqx[2] * h[0] ** 4 * q[0]
                + 12 * Ls * dqx[2] * h[0] ** 3 * q[0] * z
                - 15 * Ls * dqx[2] * h[0] ** 2 * q[0] * z**2
                - 3 * Ls * dqy[0] * h[0] ** 4 * q[1]
                - 12 * Ls * dqy[0] * h[0] ** 3 * q[1] * z
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[1] * z**2
                + 3 * Ls * dqy[1] * h[0] ** 4 * q[0]
                + 12 * Ls * dqy[1] * h[0] ** 3 * q[0] * z
                - 15 * Ls * dqy[1] * h[0] ** 2 * q[0] * z**2
                - 3 * Ls * h[0] ** 3 * h[1] * q[0] * q[2]
                - 3 * Ls * h[0] ** 3 * h[2] * q[0] * q[1]
                + 21 * Ls * h[0] * h[1] * q[0] * q[2] * z**2
                + 21 * Ls * h[0] * h[2] * q[0] * q[1] * z**2
                + 2 * U * h[0] ** 3 * h[2] * q[0] ** 2 * z
                - 3 * U * h[0] ** 2 * h[2] * q[0] ** 2 * z**2
                + 2 * V * h[0] ** 3 * h[1] * q[0] ** 2 * z
                - 3 * V * h[0] ** 2 * h[1] * q[0] ** 2 * z**2
                - 3 * dqx[0] * h[0] ** 4 * q[2] * z
                + 3 * dqx[0] * h[0] ** 3 * q[2] * z**2
                + 3 * dqx[2] * h[0] ** 4 * q[0] * z
                - 3 * dqx[2] * h[0] ** 3 * q[0] * z**2
                - 3 * dqy[0] * h[0] ** 4 * q[1] * z
                + 3 * dqy[0] * h[0] ** 3 * q[1] * z**2
                + 3 * dqy[1] * h[0] ** 4 * q[0] * z
                - 3 * dqy[1] * h[0] ** 3 * q[0] * z**2
                - 3 * h[0] ** 3 * h[1] * q[0] * q[2] * z
                - 3 * h[0] ** 3 * h[2] * q[0] * q[1] * z
                + 6 * h[0] ** 2 * h[1] * q[0] * q[2] * z**2
                + 6 * h[0] ** 2 * h[2] * q[0] * q[1] * z**2
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_xz = (
            2
            * eta
            * (
                -6 * Ls * q[1] * z
                - 2 * U * h[0] ** 2 * q[0]
                + 3 * U * h[0] * q[0] * z
                + 3 * h[0] ** 2 * q[1]
                - 6 * h[0] * q[1] * z
            )
            / (h[0] ** 2 * q[0] * (4 * Ls + h[0]))
        )
        tau_yz = (
            2
            * eta
            * (
                -6 * Ls * q[2] * z
                - 2 * V * h[0] ** 2 * q[0]
                + 3 * V * h[0] * q[0] * z
                + 3 * h[0] ** 2 * q[2]
                - 6 * h[0] * q[2] * z
            )
            / (h[0] ** 2 * q[0] * (4 * Ls + h[0]))
        )

    elif mode == "top":
        tau_xx = (
            2
            * z
            * (
                24 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * v1
                - 24 * Ls**2 * U * h[1] * q[0] ** 2 * v1 * z
                + 24 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * v2
                - 24 * Ls**2 * V * h[2] * q[0] ** 2 * v2 * z
                - 24 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * v1
                + 12 * Ls**2 * dqx[0] * h[0] * q[1] * v1 * z
                + 24 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * v1
                - 12 * Ls**2 * dqx[1] * h[0] * q[0] * v1 * z
                - 24 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * v2
                + 12 * Ls**2 * dqy[0] * h[0] * q[2] * v2 * z
                + 24 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * v2
                - 12 * Ls**2 * dqy[2] * h[0] * q[0] * v2 * z
                - 24 * Ls**2 * h[0] * h[1] * q[0] * q[1] * v1
                - 24 * Ls**2 * h[0] * h[2] * q[0] * q[2] * v2
                + 24 * Ls**2 * h[1] * q[0] * q[1] * v1 * z
                + 24 * Ls**2 * h[2] * q[0] * q[2] * v2 * z
                + 12 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                - 15 * Ls * U * h[0] * h[1] * q[0] ** 2 * v1 * z
                + 12 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                - 15 * Ls * V * h[0] * h[2] * q[0] ** 2 * v2 * z
                - 18 * Ls * dqx[0] * h[0] ** 3 * q[1] * v1
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[1] * v1 * z
                + 18 * Ls * dqx[1] * h[0] ** 3 * q[0] * v1
                - 15 * Ls * dqx[1] * h[0] ** 2 * q[0] * v1 * z
                - 18 * Ls * dqy[0] * h[0] ** 3 * q[2] * v2
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[2] * v2 * z
                + 18 * Ls * dqy[2] * h[0] ** 3 * q[0] * v2
                - 15 * Ls * dqy[2] * h[0] ** 2 * q[0] * v2 * z
                - 12 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                - 12 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * v2
                + 21 * Ls * h[0] * h[1] * q[0] * q[1] * v1 * z
                + 21 * Ls * h[0] * h[2] * q[0] * q[2] * v2 * z
                + 2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v1
                - 3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1 * z
                + 2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v2
                - 3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2 * z
                - 3 * dqx[0] * h[0] ** 4 * q[1] * v1
                + 3 * dqx[0] * h[0] ** 3 * q[1] * v1 * z
                + 3 * dqx[1] * h[0] ** 4 * q[0] * v1
                - 3 * dqx[1] * h[0] ** 3 * q[0] * v1 * z
                - 3 * dqy[0] * h[0] ** 4 * q[2] * v2
                + 3 * dqy[0] * h[0] ** 3 * q[2] * v2 * z
                + 3 * dqy[2] * h[0] ** 4 * q[0] * v2
                - 3 * dqy[2] * h[0] ** 3 * q[0] * v2 * z
                - 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v1
                - 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v2
                + 6 * h[0] ** 2 * h[1] * q[0] * q[1] * v1 * z
                + 6 * h[0] ** 2 * h[2] * q[0] * q[2] * v2 * z
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_yy = (
            2
            * z
            * (
                24 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * v2
                - 24 * Ls**2 * U * h[1] * q[0] ** 2 * v2 * z
                + 24 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * v1
                - 24 * Ls**2 * V * h[2] * q[0] ** 2 * v1 * z
                - 24 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 12 * Ls**2 * dqx[0] * h[0] * q[1] * v2 * z
                + 24 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * v2
                - 12 * Ls**2 * dqx[1] * h[0] * q[0] * v2 * z
                - 24 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * v1
                + 12 * Ls**2 * dqy[0] * h[0] * q[2] * v1 * z
                + 24 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * v1
                - 12 * Ls**2 * dqy[2] * h[0] * q[0] * v1 * z
                - 24 * Ls**2 * h[0] * h[1] * q[0] * q[1] * v2
                - 24 * Ls**2 * h[0] * h[2] * q[0] * q[2] * v1
                + 24 * Ls**2 * h[1] * q[0] * q[1] * v2 * z
                + 24 * Ls**2 * h[2] * q[0] * q[2] * v1 * z
                + 12 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                - 15 * Ls * U * h[0] * h[1] * q[0] ** 2 * v2 * z
                + 12 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                - 15 * Ls * V * h[0] * h[2] * q[0] ** 2 * v1 * z
                - 18 * Ls * dqx[0] * h[0] ** 3 * q[1] * v2
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[1] * v2 * z
                + 18 * Ls * dqx[1] * h[0] ** 3 * q[0] * v2
                - 15 * Ls * dqx[1] * h[0] ** 2 * q[0] * v2 * z
                - 18 * Ls * dqy[0] * h[0] ** 3 * q[2] * v1
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[2] * v1 * z
                + 18 * Ls * dqy[2] * h[0] ** 3 * q[0] * v1
                - 15 * Ls * dqy[2] * h[0] ** 2 * q[0] * v1 * z
                - 12 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                - 12 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * v1
                + 21 * Ls * h[0] * h[1] * q[0] * q[1] * v2 * z
                + 21 * Ls * h[0] * h[2] * q[0] * q[2] * v1 * z
                + 2 * U * h[0] ** 3 * h[1] * q[0] ** 2 * v2
                - 3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2 * z
                + 2 * V * h[0] ** 3 * h[2] * q[0] ** 2 * v1
                - 3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1 * z
                - 3 * dqx[0] * h[0] ** 4 * q[1] * v2
                + 3 * dqx[0] * h[0] ** 3 * q[1] * v2 * z
                + 3 * dqx[1] * h[0] ** 4 * q[0] * v2
                - 3 * dqx[1] * h[0] ** 3 * q[0] * v2 * z
                - 3 * dqy[0] * h[0] ** 4 * q[2] * v1
                + 3 * dqy[0] * h[0] ** 3 * q[2] * v1 * z
                + 3 * dqy[2] * h[0] ** 4 * q[0] * v1
                - 3 * dqy[2] * h[0] ** 3 * q[0] * v1 * z
                - 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v2
                - 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v1
                + 6 * h[0] ** 2 * h[1] * q[0] * q[1] * v2 * z
                + 6 * h[0] ** 2 * h[2] * q[0] * q[2] * v1 * z
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_zz = (
            2
            * v2
            * z
            * (
                24 * Ls**2 * U * h[0] * h[1] * q[0] ** 2
                - 24 * Ls**2 * U * h[1] * q[0] ** 2 * z
                + 24 * Ls**2 * V * h[0] * h[2] * q[0] ** 2
                - 24 * Ls**2 * V * h[2] * q[0] ** 2 * z
                - 24 * Ls**2 * dqx[0] * h[0] ** 2 * q[1]
                + 12 * Ls**2 * dqx[0] * h[0] * q[1] * z
                + 24 * Ls**2 * dqx[1] * h[0] ** 2 * q[0]
                - 12 * Ls**2 * dqx[1] * h[0] * q[0] * z
                - 24 * Ls**2 * dqy[0] * h[0] ** 2 * q[2]
                + 12 * Ls**2 * dqy[0] * h[0] * q[2] * z
                + 24 * Ls**2 * dqy[2] * h[0] ** 2 * q[0]
                - 12 * Ls**2 * dqy[2] * h[0] * q[0] * z
                - 24 * Ls**2 * h[0] * h[1] * q[0] * q[1]
                - 24 * Ls**2 * h[0] * h[2] * q[0] * q[2]
                + 24 * Ls**2 * h[1] * q[0] * q[1] * z
                + 24 * Ls**2 * h[2] * q[0] * q[2] * z
                + 12 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2
                - 15 * Ls * U * h[0] * h[1] * q[0] ** 2 * z
                + 12 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2
                - 15 * Ls * V * h[0] * h[2] * q[0] ** 2 * z
                - 18 * Ls * dqx[0] * h[0] ** 3 * q[1]
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[1] * z
                + 18 * Ls * dqx[1] * h[0] ** 3 * q[0]
                - 15 * Ls * dqx[1] * h[0] ** 2 * q[0] * z
                - 18 * Ls * dqy[0] * h[0] ** 3 * q[2]
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[2] * z
                + 18 * Ls * dqy[2] * h[0] ** 3 * q[0]
                - 15 * Ls * dqy[2] * h[0] ** 2 * q[0] * z
                - 12 * Ls * h[0] ** 2 * h[1] * q[0] * q[1]
                - 12 * Ls * h[0] ** 2 * h[2] * q[0] * q[2]
                + 21 * Ls * h[0] * h[1] * q[0] * q[1] * z
                + 21 * Ls * h[0] * h[2] * q[0] * q[2] * z
                + 2 * U * h[0] ** 3 * h[1] * q[0] ** 2
                - 3 * U * h[0] ** 2 * h[1] * q[0] ** 2 * z
                + 2 * V * h[0] ** 3 * h[2] * q[0] ** 2
                - 3 * V * h[0] ** 2 * h[2] * q[0] ** 2 * z
                - 3 * dqx[0] * h[0] ** 4 * q[1]
                + 3 * dqx[0] * h[0] ** 3 * q[1] * z
                + 3 * dqx[1] * h[0] ** 4 * q[0]
                - 3 * dqx[1] * h[0] ** 3 * q[0] * z
                - 3 * dqy[0] * h[0] ** 4 * q[2]
                + 3 * dqy[0] * h[0] ** 3 * q[2] * z
                + 3 * dqy[2] * h[0] ** 4 * q[0]
                - 3 * dqy[2] * h[0] ** 3 * q[0] * z
                - 3 * h[0] ** 3 * h[1] * q[0] * q[1]
                - 3 * h[0] ** 3 * h[2] * q[0] * q[2]
                + 6 * h[0] ** 2 * h[1] * q[0] * q[1] * z
                + 6 * h[0] ** 2 * h[2] * q[0] * q[2] * z
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_xy = (
            2
            * eta
            * z
            * (
                24 * Ls**2 * U * h[0] * h[2] * q[0] ** 2
                - 24 * Ls**2 * U * h[2] * q[0] ** 2 * z
                + 24 * Ls**2 * V * h[0] * h[1] * q[0] ** 2
                - 24 * Ls**2 * V * h[1] * q[0] ** 2 * z
                - 24 * Ls**2 * dqx[0] * h[0] ** 2 * q[2]
                + 12 * Ls**2 * dqx[0] * h[0] * q[2] * z
                + 24 * Ls**2 * dqx[2] * h[0] ** 2 * q[0]
                - 12 * Ls**2 * dqx[2] * h[0] * q[0] * z
                - 24 * Ls**2 * dqy[0] * h[0] ** 2 * q[1]
                + 12 * Ls**2 * dqy[0] * h[0] * q[1] * z
                + 24 * Ls**2 * dqy[1] * h[0] ** 2 * q[0]
                - 12 * Ls**2 * dqy[1] * h[0] * q[0] * z
                - 24 * Ls**2 * h[0] * h[1] * q[0] * q[2]
                - 24 * Ls**2 * h[0] * h[2] * q[0] * q[1]
                + 24 * Ls**2 * h[1] * q[0] * q[2] * z
                + 24 * Ls**2 * h[2] * q[0] * q[1] * z
                + 12 * Ls * U * h[0] ** 2 * h[2] * q[0] ** 2
                - 15 * Ls * U * h[0] * h[2] * q[0] ** 2 * z
                + 12 * Ls * V * h[0] ** 2 * h[1] * q[0] ** 2
                - 15 * Ls * V * h[0] * h[1] * q[0] ** 2 * z
                - 18 * Ls * dqx[0] * h[0] ** 3 * q[2]
                + 15 * Ls * dqx[0] * h[0] ** 2 * q[2] * z
                + 18 * Ls * dqx[2] * h[0] ** 3 * q[0]
                - 15 * Ls * dqx[2] * h[0] ** 2 * q[0] * z
                - 18 * Ls * dqy[0] * h[0] ** 3 * q[1]
                + 15 * Ls * dqy[0] * h[0] ** 2 * q[1] * z
                + 18 * Ls * dqy[1] * h[0] ** 3 * q[0]
                - 15 * Ls * dqy[1] * h[0] ** 2 * q[0] * z
                - 12 * Ls * h[0] ** 2 * h[1] * q[0] * q[2]
                - 12 * Ls * h[0] ** 2 * h[2] * q[0] * q[1]
                + 21 * Ls * h[0] * h[1] * q[0] * q[2] * z
                + 21 * Ls * h[0] * h[2] * q[0] * q[1] * z
                + 2 * U * h[0] ** 3 * h[2] * q[0] ** 2
                - 3 * U * h[0] ** 2 * h[2] * q[0] ** 2 * z
                + 2 * V * h[0] ** 3 * h[1] * q[0] ** 2
                - 3 * V * h[0] ** 2 * h[1] * q[0] ** 2 * z
                - 3 * dqx[0] * h[0] ** 4 * q[2]
                + 3 * dqx[0] * h[0] ** 3 * q[2] * z
                + 3 * dqx[2] * h[0] ** 4 * q[0]
                - 3 * dqx[2] * h[0] ** 3 * q[0] * z
                - 3 * dqy[0] * h[0] ** 4 * q[1]
                + 3 * dqy[0] * h[0] ** 3 * q[1] * z
                + 3 * dqy[1] * h[0] ** 4 * q[0]
                - 3 * dqy[1] * h[0] ** 3 * q[0] * z
                - 3 * h[0] ** 3 * h[1] * q[0] * q[2]
                - 3 * h[0] ** 3 * h[2] * q[0] * q[1]
                + 6 * h[0] ** 2 * h[1] * q[0] * q[2] * z
                + 6 * h[0] ** 2 * h[2] * q[0] * q[1] * z
            )
            / (
                h[0] ** 3
                * q[0] ** 2
                * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2)
            )
        )
        tau_xz = (
            2
            * eta
            * (
                -6 * Ls * U * h[0] * q[0]
                + 6 * Ls * U * q[0] * z
                + 6 * Ls * h[0] * q[1]
                - 6 * Ls * q[1] * z
                - 2 * U * h[0] ** 2 * q[0]
                + 3 * U * h[0] * q[0] * z
                + 3 * h[0] ** 2 * q[1]
                - 6 * h[0] * q[1] * z
            )
            / (h[0] ** 2 * q[0] * (4 * Ls + h[0]))
        )
        tau_yz = (
            2
            * eta
            * (
                -6 * Ls * V * h[0] * q[0]
                + 6 * Ls * V * q[0] * z
                + 6 * Ls * h[0] * q[2]
                - 6 * Ls * q[2] * z
                - 2 * V * h[0] ** 2 * q[0]
                + 3 * V * h[0] * q[0] * z
                + 3 * h[0] ** 2 * q[2]
                - 6 * h[0] * q[2] * z
            )
            / (h[0] ** 2 * q[0] * (4 * Ls + h[0]))
        )
    elif mode == "none":
        tau_xx = (
            z
            * (
                v1
                * (
                    U * h[0] * h[1] * q[0] ** 2
                    + 3 * dqx[0] * h[0] * (h[0] - z) * (U * q[0] - 2 * q[1])
                    + 12 * h[1] * q[0] * (h[0] - z) * (U * q[0] - 2 * q[1])
                    - 3
                    * q[0]
                    * (
                        h[0] * h[1] * (U * q[0] - 2 * q[1])
                        + (h[0] - z)
                        * (
                            U * dqx[0] * h[0]
                            + 2 * U * h[1] * q[0]
                            - 2 * dqx[1] * h[0]
                            - 4 * h[1] * q[1]
                        )
                    )
                )
                + v2
                * (
                    V * h[0] * h[2] * q[0] ** 2
                    + 3 * dqy[0] * h[0] * (h[0] - z) * (V * q[0] - 2 * q[2])
                    + 12 * h[2] * q[0] * (h[0] - z) * (V * q[0] - 2 * q[2])
                    - 3
                    * q[0]
                    * (
                        h[0] * h[2] * (V * q[0] - 2 * q[2])
                        + (h[0] - z)
                        * (
                            V * dqy[0] * h[0]
                            + 2 * V * h[2] * q[0]
                            - 2 * dqy[2] * h[0]
                            - 4 * h[2] * q[2]
                        )
                    )
                )
            )
            / (h[0] ** 3 * q[0] ** 2)
        )
        tau_yy = (
            z
            * (
                v1
                * (
                    V * h[0] * h[2] * q[0] ** 2
                    + 3 * dqy[0] * h[0] * (h[0] - z) * (V * q[0] - 2 * q[2])
                    + 12 * h[2] * q[0] * (h[0] - z) * (V * q[0] - 2 * q[2])
                    - 3
                    * q[0]
                    * (
                        h[0] * h[2] * (V * q[0] - 2 * q[2])
                        + (h[0] - z)
                        * (
                            V * dqy[0] * h[0]
                            + 2 * V * h[2] * q[0]
                            - 2 * dqy[2] * h[0]
                            - 4 * h[2] * q[2]
                        )
                    )
                )
                + v2
                * (
                    U * h[0] * h[1] * q[0] ** 2
                    + 3 * dqx[0] * h[0] * (h[0] - z) * (U * q[0] - 2 * q[1])
                    + 12 * h[1] * q[0] * (h[0] - z) * (U * q[0] - 2 * q[1])
                    - 3
                    * q[0]
                    * (
                        h[0] * h[1] * (U * q[0] - 2 * q[1])
                        + (h[0] - z)
                        * (
                            U * dqx[0] * h[0]
                            + 2 * U * h[1] * q[0]
                            - 2 * dqx[1] * h[0]
                            - 4 * h[1] * q[1]
                        )
                    )
                )
            )
            / (h[0] ** 3 * q[0] ** 2)
        )
        tau_zz = (
            v2
            * z
            * (
                h[0] * q[0] ** 2 * (U * h[1] + V * h[2])
                + 3
                * h[0]
                * (h[0] - z)
                * (
                    dqx[0] * (U * q[0] - 2 * q[1])
                    + dqy[0] * (V * q[0] - 2 * q[2])
                )
                + 12
                * q[0]
                * (h[0] - z)
                * (h[1] * (U * q[0] - 2 * q[1]) + h[2] * (V * q[0] - 2 * q[2]))
                - 3
                * q[0]
                * (
                    h[0] * h[1] * (U * q[0] - 2 * q[1])
                    + h[0] * h[2] * (V * q[0] - 2 * q[2])
                    + (h[0] - z)
                    * (
                        U * dqx[0] * h[0]
                        + 2 * U * h[1] * q[0]
                        - 2 * dqx[1] * h[0]
                        - 4 * h[1] * q[1]
                    )
                    + (h[0] - z)
                    * (
                        V * dqy[0] * h[0]
                        + 2 * V * h[2] * q[0]
                        - 2 * dqy[2] * h[0]
                        - 4 * h[2] * q[2]
                    )
                )
            )
            / (h[0] ** 3 * q[0] ** 2)
        )
        tau_xy = (
            eta
            * z
            * (
                h[0] * q[0] ** 2 * (U * h[2] + V * h[1])
                + 3
                * h[0]
                * (h[0] - z)
                * (
                    dqx[0] * (V * q[0] - 2 * q[2])
                    + dqy[0] * (U * q[0] - 2 * q[1])
                )
                + 12
                * q[0]
                * (h[0] - z)
                * (h[1] * (V * q[0] - 2 * q[2]) + h[2] * (U * q[0] - 2 * q[1]))
                - 3
                * q[0]
                * (
                    h[0] * h[1] * (V * q[0] - 2 * q[2])
                    + h[0] * h[2] * (U * q[0] - 2 * q[1])
                    + (h[0] - z)
                    * (
                        U * dqy[0] * h[0]
                        + 2 * U * h[2] * q[0]
                        - 2 * dqy[1] * h[0]
                        - 4 * h[2] * q[1]
                    )
                    + (h[0] - z)
                    * (
                        V * dqx[0] * h[0]
                        + 2 * V * h[1] * q[0]
                        - 2 * dqx[2] * h[0]
                        - 4 * h[1] * q[2]
                    )
                )
            )
            / (h[0] ** 3 * q[0] ** 2)
        )
        tau_xz = (
            -eta
            * (U * h[0] * q[0] + 3 * (h[0] - 2 * z) * (U * q[0] - 2 * q[1]))
            / (h[0] ** 2 * q[0])
        )
        tau_yz = (
            -eta
            * (V * h[0] * q[0] + 3 * (h[0] - 2 * z) * (V * q[0] - 2 * q[2]))
            / (h[0] ** 2 * q[0])
        )

    return tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy
