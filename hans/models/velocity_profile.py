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
