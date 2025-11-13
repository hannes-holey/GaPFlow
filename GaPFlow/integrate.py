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
from typing import Tuple
import numpy as np
import numpy.typing as npt


def predictor_corrector(
    q: npt.NDArray[np.floating],
    p: npt.NDArray[np.floating],
    tau: npt.NDArray[np.floating],
    direction: int
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute predictor–corrector fluxes for a 2D conservation law system.

    Combines hyperbolic (advective) and diffusive flux contributions, applying
    a directional shift to evaluate flux gradients.

    Parameters
    ----------
    q : ndarray
        Density array of shape (3, nx, ny).
    p : ndarray
        Pressure field of shape (nx, ny).
    tau : ndarray
        Gap-averaged viscous stress tensor components of shape (3, nx, ny).
    direction : int
        Direction of finite difference shift: +1 (upwind) or -1 (downwind).

    Returns
    -------
    flux_x : ndarray
        Flux contribution along the x-direction.
    flux_y : ndarray
        Flux contribution along the y-direction.
    """
    FxH, FyH = hyperbolicFlux(q, p)
    FxD, FyD = diffusiveFlux(q, tau)

    Fx = FxH + FxD
    Fy = FyH + FyD

    flux_x = -direction * (np.roll(Fx, direction, axis=1) - Fx)
    flux_y = -direction * (np.roll(Fy, direction, axis=2) - Fy)

    return flux_x, flux_y


def source(
    q: npt.NDArray[np.floating],
    h: npt.NDArray[np.floating],
    stress: npt.NDArray[np.floating],
    stress_lower: npt.NDArray[np.floating],
    stress_upper: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Compute the source term for the momentum equations.

    Accounts for stress differences across upper and lower
    boundaries.

    Parameters
    ----------
    q : ndarray
        Density array of shape (3, nx, ny).
    h : ndarray
        Geometry array of the same shape as `q`. Contains gap height and gradients.
    stress : ndarray
        Gap-averaged viscous stress tensor components of shape (3, nx, ny).
    stress_lower : ndarray
        Viscous stress tensor components at the lower wall of shape (6, nx, ny).
    stress_upper : ndarray
        Viscous stress tensor components at the upper wall of shape (6, nx, ny).

    Returns
    -------
    out : ndarray
        Computed source term of the same shape as `q`.
    """
    out = np.zeros_like(q)

    # Origin bottom, U_top = 0, U_bottom = U
    out[0] = (-q[1] * h[1] - q[2] * h[2]) / h[0]

    out[1] = ((stress[0] - stress_upper[0]) * h[1] +  # noqa: W504
              (stress[2] - stress_upper[5]) * h[2] +  # noqa: W504
              stress_upper[4] - stress_lower[4]) / h[0]

    out[2] = ((stress[2] - stress_upper[5]) * h[1] +  # noqa: W504
              (stress[1] - stress_upper[1]) * h[2] +  # noqa: W504
              stress_upper[3] - stress_lower[3]) / h[0]

    return out


def hyperbolicFlux(
    q: npt.NDArray[np.floating],
    p: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute hyperbolic (advective) fluxes for the conservation equations.

    Parameters
    ----------
    q : ndarray
        Density array of shape (3, nx, ny).
    p : ndarray
        Pressure field of shape (nx, ny).

    Returns
    -------
    Fx : ndarray
        Flux components along the x-direction.
    Fy : ndarray
        Flux components along the y-direction.
    """
    Fx = np.zeros_like(q)
    Fy = np.zeros_like(q)

    # x-direction fluxes
    Fx[0] = q[1]
    Fx[1] = p

    # y-direction fluxes
    Fy[0] = q[2]
    Fy[2] = p

    return Fx, Fy


def diffusiveFlux(
    q: npt.NDArray[np.floating],
    tau: npt.NDArray[np.floating]
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute diffusive (viscous) flux components.

    Parameters
    ----------
    q : ndarray
        Density array of shape (3, nx, ny).
    tau : ndarray
        Gap-averaged viscous stress tensor components of shape (3, nx, ny).

    Returns
    -------
    Dx : ndarray
        Diffusive flux along x-direction.
    Dy : ndarray
        Diffusive flux along y-direction.
    """
    Dx = np.zeros_like(q)
    Dy = np.zeros_like(q)

    Dx[1] = tau[0]
    Dx[2] = tau[2]

    Dy[1] = tau[2]
    Dy[2] = tau[1]

    return Dx, Dy
