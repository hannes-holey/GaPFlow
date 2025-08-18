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


import numpy as np


def stress_bottom(q, h, U, V, eta, zeta, Ls, dqx=None, dqy=None, slip="top"):
    """Viscous stress tensor at the bottom wall.

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd heihgt gradients in x and y direction, respectively.
    U : float
        Lower wall velocity in x direction.
    V : float
        Upper wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Ls : float
        Slip length
    dqx : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    dqy : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    slip : str, optional
        Slipping walls keyword, either "both", "top", or "bottom".
        (the default is 'top', which means no slip bottom wall)

    Returns
    -------
    numpy.ndarray
        Viscous stress tensor components
    """

    if dqx is None:
        dqx = np.zeros_like(q)

    if dqy is None:
        dqy = np.zeros_like(q)

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    # Voigt ordering: xx, yy, zz, yz, xz, xy
    tau = np.zeros((6, *q.shape[1:]))

    if slip == "top":
        # slip top
        tau[0] = 0
        tau[1] = 0
        tau[2] = 0
        tau[3] = 2 * eta * (-6 * Ls * V * q[0] + 6 * Ls * q[2] - 2 * V * h[0] * q[0] + 3 * h[0] * q[2]) / (h[0] * q[0] * (4 * Ls + h[0]))
        tau[4] = 2 * eta * (-6 * Ls * U * q[0] + 6 * Ls * q[1] - 2 * U * h[0] * q[0] + 3 * h[0] * q[1]) / (h[0] * q[0] * (4 * Ls + h[0]))
        tau[5] = 0
    else:
        # slip both
        tau[0] = (
            2
            * Ls
            * (
                -72 * Ls**3 * dqx[0] * q[1] * v1
                + 72 * Ls**3 * dqx[1] * q[0] * v1
                - 72 * Ls**3 * dqy[0] * q[2] * v2
                + 72 * Ls**3 * dqy[2] * q[0] * v2
                + 24 * Ls**2 * U * h[1] * q[0] ** 2 * v1
                + 24 * Ls**2 * V * h[2] * q[0] ** 2 * v2
                - 84 * Ls**2 * dqx[0] * h[0] * q[1] * v1
                + 84 * Ls**2 * dqx[1] * h[0] * q[0] * v1
                - 84 * Ls**2 * dqy[0] * h[0] * q[2] * v2
                + 84 * Ls**2 * dqy[2] * h[0] * q[0] * v2
                - 12 * Ls**2 * h[1] * q[0] * q[1] * v1
                - 12 * Ls**2 * h[2] * q[0] * q[2] * v2
                + 12 * Ls * U * h[0] * h[1] * q[0] ** 2 * v1
                + 12 * Ls * V * h[0] * h[2] * q[0] ** 2 * v2
                - 30 * Ls * dqx[0] * h[0] ** 2 * q[1] * v1
                + 30 * Ls * dqx[1] * h[0] ** 2 * q[0] * v1
                - 30 * Ls * dqy[0] * h[0] ** 2 * q[2] * v2
                + 30 * Ls * dqy[2] * h[0] ** 2 * q[0] * v2
                - 12 * Ls * h[0] * h[1] * q[0] * q[1] * v1
                - 12 * Ls * h[0] * h[2] * q[0] * q[2] * v2
                + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                - 3 * dqx[0] * h[0] ** 3 * q[1] * v1
                + 3 * dqx[1] * h[0] ** 3 * q[0] * v1
                - 3 * dqy[0] * h[0] ** 3 * q[2] * v2
                + 3 * dqy[2] * h[0] ** 3 * q[0] * v2
                - 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                - 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )
        tau[1] = (
            2
            * Ls
            * (
                -72 * Ls**3 * dqx[0] * q[1] * v2
                + 72 * Ls**3 * dqx[1] * q[0] * v2
                - 72 * Ls**3 * dqy[0] * q[2] * v1
                + 72 * Ls**3 * dqy[2] * q[0] * v1
                + 24 * Ls**2 * U * h[1] * q[0] ** 2 * v2
                + 24 * Ls**2 * V * h[2] * q[0] ** 2 * v1
                - 84 * Ls**2 * dqx[0] * h[0] * q[1] * v2
                + 84 * Ls**2 * dqx[1] * h[0] * q[0] * v2
                - 84 * Ls**2 * dqy[0] * h[0] * q[2] * v1
                + 84 * Ls**2 * dqy[2] * h[0] * q[0] * v1
                - 12 * Ls**2 * h[1] * q[0] * q[1] * v2
                - 12 * Ls**2 * h[2] * q[0] * q[2] * v1
                + 12 * Ls * U * h[0] * h[1] * q[0] ** 2 * v2
                + 12 * Ls * V * h[0] * h[2] * q[0] ** 2 * v1
                - 30 * Ls * dqx[0] * h[0] ** 2 * q[1] * v2
                + 30 * Ls * dqx[1] * h[0] ** 2 * q[0] * v2
                - 30 * Ls * dqy[0] * h[0] ** 2 * q[2] * v1
                + 30 * Ls * dqy[2] * h[0] ** 2 * q[0] * v1
                - 12 * Ls * h[0] * h[1] * q[0] * q[1] * v2
                - 12 * Ls * h[0] * h[2] * q[0] * q[2] * v1
                + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                - 3 * dqx[0] * h[0] ** 3 * q[1] * v2
                + 3 * dqx[1] * h[0] ** 3 * q[0] * v2
                - 3 * dqy[0] * h[0] ** 3 * q[2] * v1
                + 3 * dqy[2] * h[0] ** 3 * q[0] * v1
                - 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                - 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )
        tau[2] = (
            2
            * Ls
            * v2
            * (
                -72 * Ls**3 * dqx[0] * q[1]
                + 72 * Ls**3 * dqx[1] * q[0]
                - 72 * Ls**3 * dqy[0] * q[2]
                + 72 * Ls**3 * dqy[2] * q[0]
                + 24 * Ls**2 * U * h[1] * q[0] ** 2
                + 24 * Ls**2 * V * h[2] * q[0] ** 2
                - 84 * Ls**2 * dqx[0] * h[0] * q[1]
                + 84 * Ls**2 * dqx[1] * h[0] * q[0]
                - 84 * Ls**2 * dqy[0] * h[0] * q[2]
                + 84 * Ls**2 * dqy[2] * h[0] * q[0]
                - 12 * Ls**2 * h[1] * q[0] * q[1]
                - 12 * Ls**2 * h[2] * q[0] * q[2]
                + 12 * Ls * U * h[0] * h[1] * q[0] ** 2
                + 12 * Ls * V * h[0] * h[2] * q[0] ** 2
                - 30 * Ls * dqx[0] * h[0] ** 2 * q[1]
                + 30 * Ls * dqx[1] * h[0] ** 2 * q[0]
                - 30 * Ls * dqy[0] * h[0] ** 2 * q[2]
                + 30 * Ls * dqy[2] * h[0] ** 2 * q[0]
                - 12 * Ls * h[0] * h[1] * q[0] * q[1]
                - 12 * Ls * h[0] * h[2] * q[0] * q[2]
                + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2
                + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2
                - 3 * dqx[0] * h[0] ** 3 * q[1]
                + 3 * dqx[1] * h[0] ** 3 * q[0]
                - 3 * dqy[0] * h[0] ** 3 * q[2]
                + 3 * dqy[2] * h[0] ** 3 * q[0]
                - 3 * h[0] ** 2 * h[1] * q[0] * q[1]
                - 3 * h[0] ** 2 * h[2] * q[0] * q[2]
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )
        tau[3] = (
            2
            * eta
            * (-6 * Ls * V * q[0] + 6 * Ls * q[2] - 2 * V * h[0] * q[0] + 3 * h[0] * q[2])
            / (q[0] * (12 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
        tau[4] = (
            2
            * eta
            * (-6 * Ls * U * q[0] + 6 * Ls * q[1] - 2 * U * h[0] * q[0] + 3 * h[0] * q[1])
            / (q[0] * (12 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
        tau[5] = (
            2
            * Ls
            * eta
            * (
                -72 * Ls**3 * dqx[0] * q[2]
                + 72 * Ls**3 * dqx[2] * q[0]
                - 72 * Ls**3 * dqy[0] * q[1]
                + 72 * Ls**3 * dqy[1] * q[0]
                + 24 * Ls**2 * U * h[2] * q[0] ** 2
                + 24 * Ls**2 * V * h[1] * q[0] ** 2
                - 84 * Ls**2 * dqx[0] * h[0] * q[2]
                + 84 * Ls**2 * dqx[2] * h[0] * q[0]
                - 84 * Ls**2 * dqy[0] * h[0] * q[1]
                + 84 * Ls**2 * dqy[1] * h[0] * q[0]
                - 12 * Ls**2 * h[1] * q[0] * q[2]
                - 12 * Ls**2 * h[2] * q[0] * q[1]
                + 12 * Ls * U * h[0] * h[2] * q[0] ** 2
                + 12 * Ls * V * h[0] * h[1] * q[0] ** 2
                - 30 * Ls * dqx[0] * h[0] ** 2 * q[2]
                + 30 * Ls * dqx[2] * h[0] ** 2 * q[0]
                - 30 * Ls * dqy[0] * h[0] ** 2 * q[1]
                + 30 * Ls * dqy[1] * h[0] ** 2 * q[0]
                - 12 * Ls * h[0] * h[1] * q[0] * q[2]
                - 12 * Ls * h[0] * h[2] * q[0] * q[1]
                + 2 * U * h[0] ** 2 * h[2] * q[0] ** 2
                + 2 * V * h[0] ** 2 * h[1] * q[0] ** 2
                - 3 * dqx[0] * h[0] ** 3 * q[2]
                + 3 * dqx[2] * h[0] ** 3 * q[0]
                - 3 * dqy[0] * h[0] ** 3 * q[1]
                + 3 * dqy[1] * h[0] ** 3 * q[0]
                - 3 * h[0] ** 2 * h[1] * q[0] * q[2]
                - 3 * h[0] ** 2 * h[2] * q[0] * q[1]
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )

    return tau


def stress_top(q, h, U, V, eta, zeta, Ls, dqx=None, dqy=None, slip="top"):
    """Viscous stress tensor at the top wall.

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd heihgt gradients in x and y direction, respectively.
    U : float
        Lower wall velocity in x direction.
    V : float
        Upper wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Ls : float
        Slip length
    dqx : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    dqy : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    slip : str, optional
        Slipping walls keyword, either "both", "top", or "bottom".
        (the default is 'top', which means no slip bottom wall)

    Returns
    -------
    numpy.ndarray
        Viscous stress tensor components
    """

    if dqx is None:
        dqx = np.zeros_like(q)

    if dqy is None:
        dqy = np.zeros_like(q)

    v1 = zeta + 4 / 3 * eta
    v2 = zeta - 2 / 3 * eta

    # Voigt ordering: xx, yy, zz, yz, xz, xy
    tau = np.zeros((6, *q.shape[1:]))

    # slip top
    if slip == "top":
        tau[0] = (
            2
            * (
                -12 * Ls**2 * dqx[0] * q[1] * v1
                + 12 * Ls**2 * dqx[1] * q[0] * v1
                - 12 * Ls**2 * dqy[0] * q[2] * v2
                + 12 * Ls**2 * dqy[2] * q[0] * v2
                - 3 * Ls * U * h[1] * q[0] ** 2 * v1
                - 3 * Ls * V * h[2] * q[0] ** 2 * v2
                - 3 * Ls * dqx[0] * h[0] * q[1] * v1
                + 3 * Ls * dqx[1] * h[0] * q[0] * v1
                - 3 * Ls * dqy[0] * h[0] * q[2] * v2
                + 3 * Ls * dqy[2] * h[0] * q[0] * v2
                + 9 * Ls * h[1] * q[0] * q[1] * v1
                + 9 * Ls * h[2] * q[0] * q[2] * v2
                - U * h[0] * h[1] * q[0] ** 2 * v1
                - V * h[0] * h[2] * q[0] ** 2 * v2
                + 3 * h[0] * h[1] * q[0] * q[1] * v1
                + 3 * h[0] * h[2] * q[0] * q[2] * v2
            )
            / (q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
        tau[1] = (
            2
            * (
                -12 * Ls**2 * dqx[0] * q[1] * v2
                + 12 * Ls**2 * dqx[1] * q[0] * v2
                - 12 * Ls**2 * dqy[0] * q[2] * v1
                + 12 * Ls**2 * dqy[2] * q[0] * v1
                - 3 * Ls * U * h[1] * q[0] ** 2 * v2
                - 3 * Ls * V * h[2] * q[0] ** 2 * v1
                - 3 * Ls * dqx[0] * h[0] * q[1] * v2
                + 3 * Ls * dqx[1] * h[0] * q[0] * v2
                - 3 * Ls * dqy[0] * h[0] * q[2] * v1
                + 3 * Ls * dqy[2] * h[0] * q[0] * v1
                + 9 * Ls * h[1] * q[0] * q[1] * v2
                + 9 * Ls * h[2] * q[0] * q[2] * v1
                - U * h[0] * h[1] * q[0] ** 2 * v2
                - V * h[0] * h[2] * q[0] ** 2 * v1
                + 3 * h[0] * h[1] * q[0] * q[1] * v2
                + 3 * h[0] * h[2] * q[0] * q[2] * v1
            )
            / (q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
        tau[2] = (
            2
            * v2
            * (
                -12 * Ls**2 * dqx[0] * q[1]
                + 12 * Ls**2 * dqx[1] * q[0]
                - 12 * Ls**2 * dqy[0] * q[2]
                + 12 * Ls**2 * dqy[2] * q[0]
                - 3 * Ls * U * h[1] * q[0] ** 2
                - 3 * Ls * V * h[2] * q[0] ** 2
                - 3 * Ls * dqx[0] * h[0] * q[1]
                + 3 * Ls * dqx[1] * h[0] * q[0]
                - 3 * Ls * dqy[0] * h[0] * q[2]
                + 3 * Ls * dqy[2] * h[0] * q[0]
                + 9 * Ls * h[1] * q[0] * q[1]
                + 9 * Ls * h[2] * q[0] * q[2]
                - U * h[0] * h[1] * q[0] ** 2
                - V * h[0] * h[2] * q[0] ** 2
                + 3 * h[0] * h[1] * q[0] * q[1]
                + 3 * h[0] * h[2] * q[0] * q[2]
            )
            / (q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )
        tau[3] = 2 * eta * (V * q[0] - 3 * q[2]) / (q[0] * (4 * Ls + h[0]))
        tau[4] = 2 * eta * (U * q[0] - 3 * q[1]) / (q[0] * (4 * Ls + h[0]))
        tau[5] = (
            2
            * eta
            * (
                -12 * Ls**2 * dqx[0] * q[2]
                + 12 * Ls**2 * dqx[2] * q[0]
                - 12 * Ls**2 * dqy[0] * q[1]
                + 12 * Ls**2 * dqy[1] * q[0]
                - 3 * Ls * U * h[2] * q[0] ** 2
                - 3 * Ls * V * h[1] * q[0] ** 2
                - 3 * Ls * dqx[0] * h[0] * q[2]
                + 3 * Ls * dqx[2] * h[0] * q[0]
                - 3 * Ls * dqy[0] * h[0] * q[1]
                + 3 * Ls * dqy[1] * h[0] * q[0]
                + 9 * Ls * h[1] * q[0] * q[2]
                + 9 * Ls * h[2] * q[0] * q[1]
                - U * h[0] * h[2] * q[0] ** 2
                - V * h[0] * h[1] * q[0] ** 2
                + 3 * h[0] * h[1] * q[0] * q[2]
                + 3 * h[0] * h[2] * q[0] * q[1]
            )
            / (q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )

    else:
        # slip both
        tau[0] = (
            2
            * (
                -72 * Ls**4 * dqx[0] * q[1] * v1
                + 72 * Ls**4 * dqx[1] * q[0] * v1
                - 72 * Ls**4 * dqy[0] * q[2] * v2
                + 72 * Ls**4 * dqy[2] * q[0] * v2
                - 12 * Ls**3 * U * h[1] * q[0] ** 2 * v1
                - 12 * Ls**3 * V * h[2] * q[0] ** 2 * v2
                - 84 * Ls**3 * dqx[0] * h[0] * q[1] * v1
                + 84 * Ls**3 * dqx[1] * h[0] * q[0] * v1
                - 84 * Ls**3 * dqy[0] * h[0] * q[2] * v2
                + 84 * Ls**3 * dqy[2] * h[0] * q[0] * v2
                + 60 * Ls**3 * h[1] * q[0] * q[1] * v1
                + 60 * Ls**3 * h[2] * q[0] * q[2] * v2
                - 12 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * v1
                - 12 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * v2
                - 30 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * v1
                + 30 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * v1
                - 30 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * v2
                + 30 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * v2
                + 72 * Ls**2 * h[0] * h[1] * q[0] * q[1] * v1
                + 72 * Ls**2 * h[0] * h[2] * q[0] * q[2] * v2
                - 7 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                - 7 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                - 3 * Ls * dqx[0] * h[0] ** 3 * q[1] * v1
                + 3 * Ls * dqx[1] * h[0] ** 3 * q[0] * v1
                - 3 * Ls * dqy[0] * h[0] ** 3 * q[2] * v2
                + 3 * Ls * dqy[2] * h[0] ** 3 * q[0] * v2
                + 27 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                + 27 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * v2
                - U * h[0] ** 3 * h[1] * q[0] ** 2 * v1
                - V * h[0] ** 3 * h[2] * q[0] ** 2 * v2
                + 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v1
                + 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v2
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )
        tau[1] = (
            2
            * (
                -72 * Ls**4 * dqx[0] * q[1] * v2
                + 72 * Ls**4 * dqx[1] * q[0] * v2
                - 72 * Ls**4 * dqy[0] * q[2] * v1
                + 72 * Ls**4 * dqy[2] * q[0] * v1
                - 12 * Ls**3 * U * h[1] * q[0] ** 2 * v2
                - 12 * Ls**3 * V * h[2] * q[0] ** 2 * v1
                - 84 * Ls**3 * dqx[0] * h[0] * q[1] * v2
                + 84 * Ls**3 * dqx[1] * h[0] * q[0] * v2
                - 84 * Ls**3 * dqy[0] * h[0] * q[2] * v1
                + 84 * Ls**3 * dqy[2] * h[0] * q[0] * v1
                + 60 * Ls**3 * h[1] * q[0] * q[1] * v2
                + 60 * Ls**3 * h[2] * q[0] * q[2] * v1
                - 12 * Ls**2 * U * h[0] * h[1] * q[0] ** 2 * v2
                - 12 * Ls**2 * V * h[0] * h[2] * q[0] ** 2 * v1
                - 30 * Ls**2 * dqx[0] * h[0] ** 2 * q[1] * v2
                + 30 * Ls**2 * dqx[1] * h[0] ** 2 * q[0] * v2
                - 30 * Ls**2 * dqy[0] * h[0] ** 2 * q[2] * v1
                + 30 * Ls**2 * dqy[2] * h[0] ** 2 * q[0] * v1
                + 72 * Ls**2 * h[0] * h[1] * q[0] * q[1] * v2
                + 72 * Ls**2 * h[0] * h[2] * q[0] * q[2] * v1
                - 7 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                - 7 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                - 3 * Ls * dqx[0] * h[0] ** 3 * q[1] * v2
                + 3 * Ls * dqx[1] * h[0] ** 3 * q[0] * v2
                - 3 * Ls * dqy[0] * h[0] ** 3 * q[2] * v1
                + 3 * Ls * dqy[2] * h[0] ** 3 * q[0] * v1
                + 27 * Ls * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                + 27 * Ls * h[0] ** 2 * h[2] * q[0] * q[2] * v1
                - U * h[0] ** 3 * h[1] * q[0] ** 2 * v2
                - V * h[0] ** 3 * h[2] * q[0] ** 2 * v1
                + 3 * h[0] ** 3 * h[1] * q[0] * q[1] * v2
                + 3 * h[0] ** 3 * h[2] * q[0] * q[2] * v1
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )
        tau[2] = (
            2
            * v2
            * (
                -72 * Ls**4 * dqx[0] * q[1]
                + 72 * Ls**4 * dqx[1] * q[0]
                - 72 * Ls**4 * dqy[0] * q[2]
                + 72 * Ls**4 * dqy[2] * q[0]
                - 12 * Ls**3 * U * h[1] * q[0] ** 2
                - 12 * Ls**3 * V * h[2] * q[0] ** 2
                - 84 * Ls**3 * dqx[0] * h[0] * q[1]
                + 84 * Ls**3 * dqx[1] * h[0] * q[0]
                - 84 * Ls**3 * dqy[0] * h[0] * q[2]
                + 84 * Ls**3 * dqy[2] * h[0] * q[0]
                + 60 * Ls**3 * h[1] * q[0] * q[1]
                + 60 * Ls**3 * h[2] * q[0] * q[2]
                - 12 * Ls**2 * U * h[0] * h[1] * q[0] ** 2
                - 12 * Ls**2 * V * h[0] * h[2] * q[0] ** 2
                - 30 * Ls**2 * dqx[0] * h[0] ** 2 * q[1]
                + 30 * Ls**2 * dqx[1] * h[0] ** 2 * q[0]
                - 30 * Ls**2 * dqy[0] * h[0] ** 2 * q[2]
                + 30 * Ls**2 * dqy[2] * h[0] ** 2 * q[0]
                + 72 * Ls**2 * h[0] * h[1] * q[0] * q[1]
                + 72 * Ls**2 * h[0] * h[2] * q[0] * q[2]
                - 7 * Ls * U * h[0] ** 2 * h[1] * q[0] ** 2
                - 7 * Ls * V * h[0] ** 2 * h[2] * q[0] ** 2
                - 3 * Ls * dqx[0] * h[0] ** 3 * q[1]
                + 3 * Ls * dqx[1] * h[0] ** 3 * q[0]
                - 3 * Ls * dqy[0] * h[0] ** 3 * q[2]
                + 3 * Ls * dqy[2] * h[0] ** 3 * q[0]
                + 27 * Ls * h[0] ** 2 * h[1] * q[0] * q[1]
                + 27 * Ls * h[0] ** 2 * h[2] * q[0] * q[2]
                - U * h[0] ** 3 * h[1] * q[0] ** 2
                - V * h[0] ** 3 * h[2] * q[0] ** 2
                + 3 * h[0] ** 3 * h[1] * q[0] * q[1]
                + 3 * h[0] ** 3 * h[2] * q[0] * q[2]
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )
        tau[3] = 2 * eta * (-6 * Ls * q[2] + V * h[0] * q[0] - 3 * h[0] * q[2]) / (q[0] * (12 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        tau[4] = 2 * eta * (-6 * Ls * q[1] + U * h[0] * q[0] - 3 * h[0] * q[1]) / (q[0] * (12 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        tau[5] = (
            2
            * eta
            * (
                -72 * Ls**4 * dqx[0] * q[2]
                + 72 * Ls**4 * dqx[2] * q[0]
                - 72 * Ls**4 * dqy[0] * q[1]
                + 72 * Ls**4 * dqy[1] * q[0]
                - 12 * Ls**3 * U * h[2] * q[0] ** 2
                - 12 * Ls**3 * V * h[1] * q[0] ** 2
                - 84 * Ls**3 * dqx[0] * h[0] * q[2]
                + 84 * Ls**3 * dqx[2] * h[0] * q[0]
                - 84 * Ls**3 * dqy[0] * h[0] * q[1]
                + 84 * Ls**3 * dqy[1] * h[0] * q[0]
                + 60 * Ls**3 * h[1] * q[0] * q[2]
                + 60 * Ls**3 * h[2] * q[0] * q[1]
                - 12 * Ls**2 * U * h[0] * h[2] * q[0] ** 2
                - 12 * Ls**2 * V * h[0] * h[1] * q[0] ** 2
                - 30 * Ls**2 * dqx[0] * h[0] ** 2 * q[2]
                + 30 * Ls**2 * dqx[2] * h[0] ** 2 * q[0]
                - 30 * Ls**2 * dqy[0] * h[0] ** 2 * q[1]
                + 30 * Ls**2 * dqy[1] * h[0] ** 2 * q[0]
                + 72 * Ls**2 * h[0] * h[1] * q[0] * q[2]
                + 72 * Ls**2 * h[0] * h[2] * q[0] * q[1]
                - 7 * Ls * U * h[0] ** 2 * h[2] * q[0] ** 2
                - 7 * Ls * V * h[0] ** 2 * h[1] * q[0] ** 2
                - 3 * Ls * dqx[0] * h[0] ** 3 * q[2]
                + 3 * Ls * dqx[2] * h[0] ** 3 * q[0]
                - 3 * Ls * dqy[0] * h[0] ** 3 * q[1]
                + 3 * Ls * dqy[1] * h[0] ** 3 * q[0]
                + 27 * Ls * h[0] ** 2 * h[1] * q[0] * q[2]
                + 27 * Ls * h[0] ** 2 * h[2] * q[0] * q[1]
                - U * h[0] ** 3 * h[2] * q[0] ** 2
                - V * h[0] ** 3 * h[1] * q[0] ** 2
                + 3 * h[0] ** 3 * h[1] * q[0] * q[2]
                + 3 * h[0] ** 3 * h[2] * q[0] * q[1]
            )
            / (q[0] ** 2 * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4))
        )

    return tau


def stress_avg(q, h, U, V, eta, zeta, Ls, dqx=None, dqy=None, slip="top"):
    """Gap-averaged viscous stress tensor (normal and in-plane shear components).

    Parameters
    ----------
    q : numpy.ndarray
        Height-averaged variables field. First index is mass density, 2nd
        and 3rd mass flux in x and y direction, respectively.
    h : numpy.ndarray
        Gap height field. First index is actual height, 2nd
        and 3rd heihgt gradients in x and y direction, respectively.
    U : float
        Lower wall velocity in x direction.
    V : float
        Upper wall velocity in y direction.
    eta : float
        Dynamic shear viscosity
    zeta : float
        Dynamic bulk viscosity
    Ls : float
        Slip length
    dqx : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    dqy : numpy.ndarray, optional
        Gradient (x) field of the height-averaged variables. First index is mass density,
        2nd and 3rd mass flux in x and y direction, respectively.
        (the default is None, which falls back to zero)
    slip : str, optional
        Slipping walls keyword, either "both", "top", or "bottom".
        (the default is 'top', which means no slip bottom wall)

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

    # Ordering: xx, yy, xy
    tau = np.zeros((3, *q.shape[1:]))

    if slip == "top":
        # slip top
        tau[0] = (
            -2
            * h[0]
            * (
                8 * Ls**2 * U * h[1] * q[0] ** 2 * v1
                + 8 * Ls**2 * V * h[2] * q[0] ** 2 * v2
                - 4 * Ls**2 * dqx[0] * h[0] * q[1] * v1
                + 4 * Ls**2 * dqx[1] * h[0] * q[0] * v1
                - 4 * Ls**2 * dqy[0] * h[0] * q[2] * v2
                + 4 * Ls**2 * dqy[2] * h[0] * q[0] * v2
                - 8 * Ls**2 * h[1] * q[0] * q[1] * v1
                - 8 * Ls**2 * h[2] * q[0] * q[2] * v2
                + 5 * Ls * U * h[0] * h[1] * q[0] ** 2 * v1
                + 5 * Ls * V * h[0] * h[2] * q[0] ** 2 * v2
                - 5 * Ls * dqx[0] * h[0] ** 2 * q[1] * v1
                + 5 * Ls * dqx[1] * h[0] ** 2 * q[0] * v1
                - 5 * Ls * dqy[0] * h[0] ** 2 * q[2] * v2
                + 5 * Ls * dqy[2] * h[0] ** 2 * q[0] * v2
                - 7 * Ls * h[0] * h[1] * q[0] * q[1] * v1
                - 7 * Ls * h[0] * h[2] * q[0] * q[2] * v2
                + U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                + V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                - dqx[0] * h[0] ** 3 * q[1] * v1
                + dqx[1] * h[0] ** 3 * q[0] * v1
                - dqy[0] * h[0] ** 3 * q[2] * v2
                + dqy[2] * h[0] ** 3 * q[0] * v2
                - 2 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                - 2 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            )
            + h[0]
            * (
                24 * Ls**2 * U * h[1] * q[0] ** 2 * v1
                + 24 * Ls**2 * V * h[2] * q[0] ** 2 * v2
                - 24 * Ls**2 * dqx[0] * h[0] * q[1] * v1
                + 24 * Ls**2 * dqx[1] * h[0] * q[0] * v1
                - 24 * Ls**2 * dqy[0] * h[0] * q[2] * v2
                + 24 * Ls**2 * dqy[2] * h[0] * q[0] * v2
                - 24 * Ls**2 * h[1] * q[0] * q[1] * v1
                - 24 * Ls**2 * h[2] * q[0] * q[2] * v2
                + 12 * Ls * U * h[0] * h[1] * q[0] ** 2 * v1
                + 12 * Ls * V * h[0] * h[2] * q[0] ** 2 * v2
                - 18 * Ls * dqx[0] * h[0] ** 2 * q[1] * v1
                + 18 * Ls * dqx[1] * h[0] ** 2 * q[0] * v1
                - 18 * Ls * dqy[0] * h[0] ** 2 * q[2] * v2
                + 18 * Ls * dqy[2] * h[0] ** 2 * q[0] * v2
                - 12 * Ls * h[0] * h[1] * q[0] * q[1] * v1
                - 12 * Ls * h[0] * h[2] * q[0] * q[2] * v2
                + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                - 3 * dqx[0] * h[0] ** 3 * q[1] * v1
                + 3 * dqx[1] * h[0] ** 3 * q[0] * v1
                - 3 * dqy[0] * h[0] ** 3 * q[2] * v2
                + 3 * dqy[2] * h[0] ** 3 * q[0] * v2
                - 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                - 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            )
        ) / (h[0] ** 2 * q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        tau[1] = (
            -2
            * h[0]
            * (
                8 * Ls**2 * U * h[1] * q[0] ** 2 * v2
                + 8 * Ls**2 * V * h[2] * q[0] ** 2 * v1
                - 4 * Ls**2 * dqx[0] * h[0] * q[1] * v2
                + 4 * Ls**2 * dqx[1] * h[0] * q[0] * v2
                - 4 * Ls**2 * dqy[0] * h[0] * q[2] * v1
                + 4 * Ls**2 * dqy[2] * h[0] * q[0] * v1
                - 8 * Ls**2 * h[1] * q[0] * q[1] * v2
                - 8 * Ls**2 * h[2] * q[0] * q[2] * v1
                + 5 * Ls * U * h[0] * h[1] * q[0] ** 2 * v2
                + 5 * Ls * V * h[0] * h[2] * q[0] ** 2 * v1
                - 5 * Ls * dqx[0] * h[0] ** 2 * q[1] * v2
                + 5 * Ls * dqx[1] * h[0] ** 2 * q[0] * v2
                - 5 * Ls * dqy[0] * h[0] ** 2 * q[2] * v1
                + 5 * Ls * dqy[2] * h[0] ** 2 * q[0] * v1
                - 7 * Ls * h[0] * h[1] * q[0] * q[1] * v2
                - 7 * Ls * h[0] * h[2] * q[0] * q[2] * v1
                + U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                + V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                - dqx[0] * h[0] ** 3 * q[1] * v2
                + dqx[1] * h[0] ** 3 * q[0] * v2
                - dqy[0] * h[0] ** 3 * q[2] * v1
                + dqy[2] * h[0] ** 3 * q[0] * v1
                - 2 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                - 2 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            )
            + h[0]
            * (
                24 * Ls**2 * U * h[1] * q[0] ** 2 * v2
                + 24 * Ls**2 * V * h[2] * q[0] ** 2 * v1
                - 24 * Ls**2 * dqx[0] * h[0] * q[1] * v2
                + 24 * Ls**2 * dqx[1] * h[0] * q[0] * v2
                - 24 * Ls**2 * dqy[0] * h[0] * q[2] * v1
                + 24 * Ls**2 * dqy[2] * h[0] * q[0] * v1
                - 24 * Ls**2 * h[1] * q[0] * q[1] * v2
                - 24 * Ls**2 * h[2] * q[0] * q[2] * v1
                + 12 * Ls * U * h[0] * h[1] * q[0] ** 2 * v2
                + 12 * Ls * V * h[0] * h[2] * q[0] ** 2 * v1
                - 18 * Ls * dqx[0] * h[0] ** 2 * q[1] * v2
                + 18 * Ls * dqx[1] * h[0] ** 2 * q[0] * v2
                - 18 * Ls * dqy[0] * h[0] ** 2 * q[2] * v1
                + 18 * Ls * dqy[2] * h[0] ** 2 * q[0] * v1
                - 12 * Ls * h[0] * h[1] * q[0] * q[1] * v2
                - 12 * Ls * h[0] * h[2] * q[0] * q[2] * v1
                + 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                + 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                - 3 * dqx[0] * h[0] ** 3 * q[1] * v2
                + 3 * dqx[1] * h[0] ** 3 * q[0] * v2
                - 3 * dqy[0] * h[0] ** 3 * q[2] * v1
                + 3 * dqy[2] * h[0] ** 3 * q[0] * v1
                - 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                - 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            )
        ) / (h[0] ** 2 * q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        tau[2] = (
            eta
            * (
                -2
                * h[0]
                * (
                    8 * Ls**2 * U * h[2] * q[0] ** 2
                    + 8 * Ls**2 * V * h[1] * q[0] ** 2
                    - 4 * Ls**2 * dqx[0] * h[0] * q[2]
                    + 4 * Ls**2 * dqx[2] * h[0] * q[0]
                    - 4 * Ls**2 * dqy[0] * h[0] * q[1]
                    + 4 * Ls**2 * dqy[1] * h[0] * q[0]
                    - 8 * Ls**2 * h[1] * q[0] * q[2]
                    - 8 * Ls**2 * h[2] * q[0] * q[1]
                    + 5 * Ls * U * h[0] * h[2] * q[0] ** 2
                    + 5 * Ls * V * h[0] * h[1] * q[0] ** 2
                    - 5 * Ls * dqx[0] * h[0] ** 2 * q[2]
                    + 5 * Ls * dqx[2] * h[0] ** 2 * q[0]
                    - 5 * Ls * dqy[0] * h[0] ** 2 * q[1]
                    + 5 * Ls * dqy[1] * h[0] ** 2 * q[0]
                    - 7 * Ls * h[0] * h[1] * q[0] * q[2]
                    - 7 * Ls * h[0] * h[2] * q[0] * q[1]
                    + U * h[0] ** 2 * h[2] * q[0] ** 2
                    + V * h[0] ** 2 * h[1] * q[0] ** 2
                    - dqx[0] * h[0] ** 3 * q[2]
                    + dqx[2] * h[0] ** 3 * q[0]
                    - dqy[0] * h[0] ** 3 * q[1]
                    + dqy[1] * h[0] ** 3 * q[0]
                    - 2 * h[0] ** 2 * h[1] * q[0] * q[2]
                    - 2 * h[0] ** 2 * h[2] * q[0] * q[1]
                )
                + h[0]
                * (
                    24 * Ls**2 * U * h[2] * q[0] ** 2
                    + 24 * Ls**2 * V * h[1] * q[0] ** 2
                    - 24 * Ls**2 * dqx[0] * h[0] * q[2]
                    + 24 * Ls**2 * dqx[2] * h[0] * q[0]
                    - 24 * Ls**2 * dqy[0] * h[0] * q[1]
                    + 24 * Ls**2 * dqy[1] * h[0] * q[0]
                    - 24 * Ls**2 * h[1] * q[0] * q[2]
                    - 24 * Ls**2 * h[2] * q[0] * q[1]
                    + 12 * Ls * U * h[0] * h[2] * q[0] ** 2
                    + 12 * Ls * V * h[0] * h[1] * q[0] ** 2
                    - 18 * Ls * dqx[0] * h[0] ** 2 * q[2]
                    + 18 * Ls * dqx[2] * h[0] ** 2 * q[0]
                    - 18 * Ls * dqy[0] * h[0] ** 2 * q[1]
                    + 18 * Ls * dqy[1] * h[0] ** 2 * q[0]
                    - 12 * Ls * h[0] * h[1] * q[0] * q[2]
                    - 12 * Ls * h[0] * h[2] * q[0] * q[1]
                    + 2 * U * h[0] ** 2 * h[2] * q[0] ** 2
                    + 2 * V * h[0] ** 2 * h[1] * q[0] ** 2
                    - 3 * dqx[0] * h[0] ** 3 * q[2]
                    + 3 * dqx[2] * h[0] ** 3 * q[0]
                    - 3 * dqy[0] * h[0] ** 3 * q[1]
                    + 3 * dqy[1] * h[0] ** 3 * q[0]
                    - 3 * h[0] ** 2 * h[1] * q[0] * q[2]
                    - 3 * h[0] ** 2 * h[2] * q[0] * q[1]
                )
            )
            / (h[0] ** 2 * q[0] ** 2 * (16 * Ls**2 + 8 * Ls * h[0] + h[0] ** 2))
        )

    else:
        # slip both
        tau[0] = (
            -h[0] ** 2
            * (2 * Ls + h[0])
            * (36 * Ls**2 + 12 * Ls * h[0] + h[0] ** 2)
            * (
                72 * Ls**3 * dqx[0] * q[1] * v1
                - 72 * Ls**3 * dqx[1] * q[0] * v1
                + 72 * Ls**3 * dqy[0] * q[2] * v2
                - 72 * Ls**3 * dqy[2] * q[0] * v2
                - 24 * Ls**2 * U * h[1] * q[0] ** 2 * v1
                - 24 * Ls**2 * V * h[2] * q[0] ** 2 * v2
                + 84 * Ls**2 * dqx[0] * h[0] * q[1] * v1
                - 84 * Ls**2 * dqx[1] * h[0] * q[0] * v1
                + 84 * Ls**2 * dqy[0] * h[0] * q[2] * v2
                - 84 * Ls**2 * dqy[2] * h[0] * q[0] * v2
                + 12 * Ls**2 * h[1] * q[0] * q[1] * v1
                + 12 * Ls**2 * h[2] * q[0] * q[2] * v2
                - 12 * Ls * U * h[0] * h[1] * q[0] ** 2 * v1
                - 12 * Ls * V * h[0] * h[2] * q[0] ** 2 * v2
                + 30 * Ls * dqx[0] * h[0] ** 2 * q[1] * v1
                - 30 * Ls * dqx[1] * h[0] ** 2 * q[0] * v1
                + 30 * Ls * dqy[0] * h[0] ** 2 * q[2] * v2
                - 30 * Ls * dqy[2] * h[0] ** 2 * q[0] * v2
                + 12 * Ls * h[0] * h[1] * q[0] * q[1] * v1
                + 12 * Ls * h[0] * h[2] * q[0] * q[2] * v2
                - 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v1
                - 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v2
                + 3 * dqx[0] * h[0] ** 3 * q[1] * v1
                - 3 * dqx[1] * h[0] ** 3 * q[0] * v1
                + 3 * dqy[0] * h[0] ** 3 * q[2] * v2
                - 3 * dqy[2] * h[0] ** 3 * q[0] * v2
                + 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v1
                + 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v2
            )
            - 2
            * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4)
            * (
                3 * Ls * U * h[1] * q[0] ** 2 * v1
                + 3 * Ls * V * h[2] * q[0] ** 2 * v2
                - 6 * Ls * dqx[0] * h[0] * q[1] * v1
                + 6 * Ls * dqx[1] * h[0] * q[0] * v1
                - 6 * Ls * dqy[0] * h[0] * q[2] * v2
                + 6 * Ls * dqy[2] * h[0] * q[0] * v2
                - 6 * Ls * h[1] * q[0] * q[1] * v1
                - 6 * Ls * h[2] * q[0] * q[2] * v2
                + U * h[0] * h[1] * q[0] ** 2 * v1
                + V * h[0] * h[2] * q[0] ** 2 * v2
                - dqx[0] * h[0] ** 2 * q[1] * v1
                + dqx[1] * h[0] ** 2 * q[0] * v1
                - dqy[0] * h[0] ** 2 * q[2] * v2
                + dqy[2] * h[0] ** 2 * q[0] * v2
                - 2 * h[0] * h[1] * q[0] * q[1] * v1
                - 2 * h[0] * h[2] * q[0] * q[2] * v2
            )
            * h[0] ** 2
        ) / (
            h[0] ** 2
            * q[0] ** 2
            * (36 * Ls**2 + 12 * Ls * h[0] + h[0] ** 2)
            * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4)
        )
        tau[1] = (
            -h[0] ** 2
            * (2 * Ls + h[0])
            * (36 * Ls**2 + 12 * Ls * h[0] + h[0] ** 2)
            * (
                72 * Ls**3 * dqx[0] * q[1] * v2
                - 72 * Ls**3 * dqx[1] * q[0] * v2
                + 72 * Ls**3 * dqy[0] * q[2] * v1
                - 72 * Ls**3 * dqy[2] * q[0] * v1
                - 24 * Ls**2 * U * h[1] * q[0] ** 2 * v2
                - 24 * Ls**2 * V * h[2] * q[0] ** 2 * v1
                + 84 * Ls**2 * dqx[0] * h[0] * q[1] * v2
                - 84 * Ls**2 * dqx[1] * h[0] * q[0] * v2
                + 84 * Ls**2 * dqy[0] * h[0] * q[2] * v1
                - 84 * Ls**2 * dqy[2] * h[0] * q[0] * v1
                + 12 * Ls**2 * h[1] * q[0] * q[1] * v2
                + 12 * Ls**2 * h[2] * q[0] * q[2] * v1
                - 12 * Ls * U * h[0] * h[1] * q[0] ** 2 * v2
                - 12 * Ls * V * h[0] * h[2] * q[0] ** 2 * v1
                + 30 * Ls * dqx[0] * h[0] ** 2 * q[1] * v2
                - 30 * Ls * dqx[1] * h[0] ** 2 * q[0] * v2
                + 30 * Ls * dqy[0] * h[0] ** 2 * q[2] * v1
                - 30 * Ls * dqy[2] * h[0] ** 2 * q[0] * v1
                + 12 * Ls * h[0] * h[1] * q[0] * q[1] * v2
                + 12 * Ls * h[0] * h[2] * q[0] * q[2] * v1
                - 2 * U * h[0] ** 2 * h[1] * q[0] ** 2 * v2
                - 2 * V * h[0] ** 2 * h[2] * q[0] ** 2 * v1
                + 3 * dqx[0] * h[0] ** 3 * q[1] * v2
                - 3 * dqx[1] * h[0] ** 3 * q[0] * v2
                + 3 * dqy[0] * h[0] ** 3 * q[2] * v1
                - 3 * dqy[2] * h[0] ** 3 * q[0] * v1
                + 3 * h[0] ** 2 * h[1] * q[0] * q[1] * v2
                + 3 * h[0] ** 2 * h[2] * q[0] * q[2] * v1
            )
            - 2
            * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4)
            * (
                3 * Ls * U * h[1] * q[0] ** 2 * v2
                + 3 * Ls * V * h[2] * q[0] ** 2 * v1
                - 6 * Ls * dqx[0] * h[0] * q[1] * v2
                + 6 * Ls * dqx[1] * h[0] * q[0] * v2
                - 6 * Ls * dqy[0] * h[0] * q[2] * v1
                + 6 * Ls * dqy[2] * h[0] * q[0] * v1
                - 6 * Ls * h[1] * q[0] * q[1] * v2
                - 6 * Ls * h[2] * q[0] * q[2] * v1
                + U * h[0] * h[1] * q[0] ** 2 * v2
                + V * h[0] * h[2] * q[0] ** 2 * v1
                - dqx[0] * h[0] ** 2 * q[1] * v2
                + dqx[1] * h[0] ** 2 * q[0] * v2
                - dqy[0] * h[0] ** 2 * q[2] * v1
                + dqy[2] * h[0] ** 2 * q[0] * v1
                - 2 * h[0] * h[1] * q[0] * q[1] * v2
                - 2 * h[0] * h[2] * q[0] * q[2] * v1
            )
            * h[0] ** 2
        ) / (
            h[0] ** 2
            * q[0] ** 2
            * (36 * Ls**2 + 12 * Ls * h[0] + h[0] ** 2)
            * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4)
        )
        tau[2] = (
            eta
            * (
                -h[0] ** 2
                * (2 * Ls + h[0])
                * (36 * Ls**2 + 12 * Ls * h[0] + h[0] ** 2)
                * (
                    72 * Ls**3 * dqx[0] * q[2]
                    - 72 * Ls**3 * dqx[2] * q[0]
                    + 72 * Ls**3 * dqy[0] * q[1]
                    - 72 * Ls**3 * dqy[1] * q[0]
                    - 24 * Ls**2 * U * h[2] * q[0] ** 2
                    - 24 * Ls**2 * V * h[1] * q[0] ** 2
                    + 84 * Ls**2 * dqx[0] * h[0] * q[2]
                    - 84 * Ls**2 * dqx[2] * h[0] * q[0]
                    + 84 * Ls**2 * dqy[0] * h[0] * q[1]
                    - 84 * Ls**2 * dqy[1] * h[0] * q[0]
                    + 12 * Ls**2 * h[1] * q[0] * q[2]
                    + 12 * Ls**2 * h[2] * q[0] * q[1]
                    - 12 * Ls * U * h[0] * h[2] * q[0] ** 2
                    - 12 * Ls * V * h[0] * h[1] * q[0] ** 2
                    + 30 * Ls * dqx[0] * h[0] ** 2 * q[2]
                    - 30 * Ls * dqx[2] * h[0] ** 2 * q[0]
                    + 30 * Ls * dqy[0] * h[0] ** 2 * q[1]
                    - 30 * Ls * dqy[1] * h[0] ** 2 * q[0]
                    + 12 * Ls * h[0] * h[1] * q[0] * q[2]
                    + 12 * Ls * h[0] * h[2] * q[0] * q[1]
                    - 2 * U * h[0] ** 2 * h[2] * q[0] ** 2
                    - 2 * V * h[0] ** 2 * h[1] * q[0] ** 2
                    + 3 * dqx[0] * h[0] ** 3 * q[2]
                    - 3 * dqx[2] * h[0] ** 3 * q[0]
                    + 3 * dqy[0] * h[0] ** 3 * q[1]
                    - 3 * dqy[1] * h[0] ** 3 * q[0]
                    + 3 * h[0] ** 2 * h[1] * q[0] * q[2]
                    + 3 * h[0] ** 2 * h[2] * q[0] * q[1]
                )
                - 2
                * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4)
                * (
                    3 * Ls * U * h[2] * q[0] ** 2
                    + 3 * Ls * V * h[1] * q[0] ** 2
                    - 6 * Ls * dqx[0] * h[0] * q[2]
                    + 6 * Ls * dqx[2] * h[0] * q[0]
                    - 6 * Ls * dqy[0] * h[0] * q[1]
                    + 6 * Ls * dqy[1] * h[0] * q[0]
                    - 6 * Ls * h[1] * q[0] * q[2]
                    - 6 * Ls * h[2] * q[0] * q[1]
                    + U * h[0] * h[2] * q[0] ** 2
                    + V * h[0] * h[1] * q[0] ** 2
                    - dqx[0] * h[0] ** 2 * q[2]
                    + dqx[2] * h[0] ** 2 * q[0]
                    - dqy[0] * h[0] ** 2 * q[1]
                    + dqy[1] * h[0] ** 2 * q[0]
                    - 2 * h[0] * h[1] * q[0] * q[2]
                    - 2 * h[0] * h[2] * q[0] * q[1]
                )
                * h[0] ** 2
            )
            / (
                h[0] ** 2
                * q[0] ** 2
                * (36 * Ls**2 + 12 * Ls * h[0] + h[0] ** 2)
                * (144 * Ls**4 + 192 * Ls**3 * h[0] + 88 * Ls**2 * h[0] ** 2 + 16 * Ls * h[0] ** 3 + h[0] ** 4)
            )
        )

    return tau
