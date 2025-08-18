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
from scipy.optimize import fsolve
from hans.tools import power


def stress_powerlaw_bottom(q, h, U, V, eta, n, method="exact"):

    u_mean = q[1] / q[0]
    v_mean = q[2] / q[0]

    if method == "approx":
        zmaxu, zmaxv = approximate_zmax(U, V, u_mean, v_mean, n)
    else:
        zmaxu, zmaxv = solve_zmax(U, V, u_mean, v_mean, n)

    zmaxu *= h[0]
    zmaxv *= h[0]

    U_crit = u_mean * (1 + 2 * n) / (1 + n)
    V_crit = v_mean * (1 + 2 * n) / (1 + n)

    # case 1: U <= U_crit; V <= V_crit
    maskU1 = U <= U_crit
    maskV1 = V <= V_crit

    # case 2: U > U_crit; V > V_crit
    maskU2 = U > U_crit
    maskV2 = V > V_crit

    # case 1:
    s3_1 = -(
        ((1 + 2 * n) * power(zmaxv, 1 / n) * (q[0] * V * zmaxv - q[2] * h[0]))
        / (n * q[0] * (power(zmaxv, 2 + 1 / n) + power(-zmaxv + h[0], 2 + 1 / n)))
    )

    s4_1 = -(
        ((1 + 2 * n) * power(zmaxu, 1 / n) * (q[0] * U * zmaxu - q[1] * h[0]))
        / (n * q[0] * (power(zmaxu, 2 + 1 / n) + power(-zmaxu + h[0], 2 + 1 / n)))
    )

    # case 2:
    s3_2 = (power(zmaxv, 1 / n) * (1 + 1 / n) * (1 + 2 * n) * (q[2] - V * q[0]) * h[0]) / (
        q[0] * (n * power(zmaxv - h[0], 2 + 1 / n) + power(zmaxv, 1 + 1 / n) * (-zmaxv * n + (1 + 2 * n) * h[0]))
    )

    s4_2 = (power(zmaxu, 1 / n) * (1 + 1 / n) * (1 + 2 * n) * (q[1] - U * q[0]) * h[0]) / (
        q[0] * (n * power(zmaxu - h[0], 2 + 1 / n) + power(zmaxu, 1 + 1 / n) * (-zmaxu * n + (1 + 2 * n) * h[0]))
    )
    
    tau = np.zeros((6, q.shape[1:]))

    tau[3, maskV1] = eta * power(s3_1[maskV1], n)
    tau[3, maskV2] = eta * power(s3_2[maskV2], n)

    tau[4, maskU1] = eta * power(s4_1[maskU1], n)
    tau[4, maskU2] = eta * power(s4_2[maskU2], n)

    return tau


def stress_powerlaw_top(q, h, U, V, eta, n, method="exact"):

    u_mean = q[1] / q[0]
    v_mean = q[2] / q[0]

    if method == "approx":
        zmaxu, zmaxv = approximate_zmax(U, V, u_mean, v_mean, n)
    else:
        zmaxu, zmaxv = solve_zmax(U, V, u_mean, v_mean, n)

    zmaxu *= h[0]
    zmaxv *= h[0]

    U_crit = u_mean * (1 + 2 * n) / (1 + n)
    V_crit = v_mean * (1 + 2 * n) / (1 + n)

    # case 1: U <= U_crit; V <= V_crit
    maskU1 = U <= U_crit
    maskV1 = V <= V_crit

    # case 2: U > U_crit; V > V_crit
    maskU2 = U > U_crit
    maskV2 = V > V_crit

    # case 1
    s0_1 = (
        -2
        * ((1 + 2 * n) * power(-zmaxu + h[0], 1 / n) * (q[0] * U * zmaxu - q[1] * h[0]) * h[1])
        / (n * q[0] * (power(zmaxu, 2 + 1 / n) + power(-zmaxu + h[0], 2 + 1 / n)))
    )

    s1_1 = (
        -2
        * ((1 + 2 * n) * power(-zmaxv + h[0], 1 / n) * (q[0] * V * zmaxv - q[2] * h[0]) * h[2])
        / (n * q[0] * (power(zmaxv, 2 + 1 / n) + power(-zmaxv + h[0], 2 + 1 / n)))
    )

    s3_1 = ((1 + 2 * n) * power(-zmaxv + h[0], 1 / n) * (q[0] * V * zmaxv - q[2] * h[0])) / (
        n * q[0] * (power(zmaxv, 2 + 1 / n) + power(-zmaxv + h[0], 2 + 1 / n))
    )

    s4_1 = ((1 + 2 * n) * power(-zmaxu + h[0], 1 / n) * (q[0] * U * zmaxu - q[1] * h[0])) / (
        n * q[0] * (power(zmaxu, 2 + 1 / n) + power(-zmaxu + h[0], 2 + 1 / n))
    )

    s5_u1 = -s4_1 * h[2]
    s5_v1 = -s3_1 * h[1]

    # case 2
    s0_2 = (
        -2
        * (
            (1 + 2 * n)
            * (q[1] - U * q[0])
            * (power(zmaxu, 1 + 1 / n) - power(zmaxu - h[0], 1 + 1 / n))
            * (power(zmaxu, 2 + 1 / n) * n - power(zmaxu - h[0], 1 + 1 / n) * (zmaxu * n + (1 + n) * h[0]))
            * h[1]
        )
        / (q[0] * (n * power(zmaxu - h[0], 2 + 1 / n) + power(zmaxu, 1 + 1 / n) * (-zmaxu * n + (1 + 2 * n) * h[0])) ** 2)
    )

    s1_2 = (
        -2
        * (
            (1 + 2 * n)
            * (q[2] - V * q[0])
            * (power(zmaxv, 1 + 1 / n) - power(zmaxv - h[0], 1 + 1 / n))
            * (power(zmaxv, 2 + 1 / n) * n - power(zmaxv - h[0], 1 + 1 / n) * (zmaxv * n + (1 + n) * h[0]))
            * h[2]
        )
        / (q[0] * (n * power(zmaxv - h[0], 2 + 1 / n) + power(zmaxv, 1 + 1 / n) * (-zmaxv * n + (1 + 2 * n) * h[0])) ** 2)
    )

    s3_2 = ((1 + 1 / n) * (1 + 2 * n) * (q[2] - V * q[0]) * power(zmaxv - h[0], 1 / n) * h[0]) / (
        q[0] * (n * power(zmaxv - h[0], 2 + 1 / n) + power(zmaxv, 1 + 1 / n) * (-zmaxv * n + (1 + 2 * n) * h[0]))
    )

    s4_2 = ((1 + 1 / n) * (1 + 2 * n) * (q[1] - U * q[0]) * power(zmaxu - h[0], 1 / n) * h[0]) / (
        q[0] * (n * power(zmaxu - h[0], 2 + 1 / n) + power(zmaxu, 1 + 1 / n) * (-zmaxu * n + (1 + 2 * n) * h[0]))
    )

    s5_u2 = (
        (1 + 2 * n)
        / q[0]
        * (
            -(
                ((1 + 2 * n) * (q[1] - U * q[0]) * (power(zmaxu, 1 + 1 / n) - power(zmaxu - h[0], 1 + 1 / n)) ** 2 * h[0] * h[2])
                / (n * power(zmaxu - h[0], 2 + 1 / n) + power(zmaxu, 1 + 1 / n) * (-zmaxu * n + (1 + 2 * n) * h[0])) ** 2
            )
            + ((q[1] - U * q[0]) * (power(zmaxu, 1 + 1 / n) - power(zmaxu - h[0], 1 + 1 / n)) * h[2])
            / (n * power(zmaxu - h[0], 2 + 1 / n) + power(zmaxu, 1 + 1 / n) * (-zmaxu * n + (1 + 2 * n) * h[0]))
        )
    )

    s5_v2 = (
        (1 + 2 * n)
        / q[0]
        * (
            -(
                ((1 + 2 * n) * (q[2] - V * q[0]) * (power(zmaxv, 1 + 1 / n) - power(zmaxv - h[0], 1 + 1 / n)) ** 2 * h[0] * h[1])
                / (n * power(zmaxv - h[0], 2 + 1 / n) + power(zmaxv, 1 + 1 / n) * (-zmaxv * n + (1 + 2 * n) * h[0])) ** 2
            )
            + ((q[2] - V * q[0]) * (power(zmaxv, 1 + 1 / n) - power(zmaxv - h[0], 1 + 1 / n)) * h[1])
            / (n * power(zmaxv - h[0], 2 + 1 / n) + power(zmaxv, 1 + 1 / n) * (-zmaxv * n + (1 + 2 * n) * h[0]))
        )
    )

    tau = np.zeros((6, q.shape[1:]))

    tau[0, maskU1] = eta * power(s0_1[maskU1], n)
    tau[0, maskU2] = eta * power(s0_2[maskU2], n)

    tau[1, maskV1] = eta * power(s1_1[maskV1], n)
    tau[1, maskV2] = eta * power(s1_2[maskV2], n)

    tau[3, maskV1] = eta * power(s3_1[maskV1], n)
    tau[3, maskV2] = eta * power(s3_2[maskV2], n)

    tau[4, maskU1] = eta * power(s4_1[maskU1], n)
    tau[4, maskU2] = eta * power(s4_2[maskU2], n)

    tau[5, maskU1] = eta * power(s5_u1[maskU1], n)
    tau[5, maskU2] = eta * power(s5_u2[maskU2], n)

    tau[5, maskV1] += eta * power(s5_v1[maskV1], n)
    tau[5, maskV2] += eta * power(s5_v2[maskV2], n)

    return tau


def solve_zmax(U, V, um, vm, n):
    """Calculate the location of maximum velocity in z-direction to compute the power law stresses.

    Parameters
    ----------
    U : float
        Sliding velocity of the lower surface in x-direction.
    V : float
        Sliding velocity of the lower surface in y-direction.
    Um : numpy.ndarray
        Array of mean velocities in x-direction (jx/rho).
    Vm : numpy.ndarray
        Array of mean velocities in y-direction (jy/rho).
    n : float
        Exponent

    Returns
    -------
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in x-direction
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in y-direction
    """

    Nx, Ny = vm.shape

    if V == 0:
        zmaxv = np.ones_like(vm) * 0.5
    else:
        vn = vm / V
        zmaxv = np.ones_like(vm)
        init = 0.5 - 1 / (12 * n * (vn - 0.5))

        for i in range(Nx):
            for j in range(Ny):
                if vn[i, j] >= (1 + n) / (1 + 2 * n):
                    zmaxv[i, j] = fsolve(zmax_nleq_case1, init[i, j], args=(n, vn[i, j]))[0]
                else:
                    zmaxv[i, j] = fsolve(zmax_nleq_case2, init[i, j], args=(n, vn[i, j]))[0]

    if U == 0:
        zmaxu = np.ones_like(um) * 0.5
    else:
        un = um / U
        zmaxu = np.ones_like(um)
        init = 0.5 - 1 / (12 * n * (un - 0.5))

        for i in range(Nx):
            for j in range(Ny):
                if un[i, j] >= (1 + n) / (1 + 2 * n):
                    zmaxu[i, j] = fsolve(zmax_nleq_case1, init[i, j], args=(n, un[i, j]))[0]
                else:
                    zmaxu[i, j] = fsolve(zmax_nleq_case2, init[i, j], args=(n, un[i, j]))[0]

    # upper and lower bound for asymptotic behavior
    zmaxu = np.minimum(zmaxu, 1e5)
    zmaxu = np.maximum(zmaxu, -1e5)
    zmaxv = np.minimum(zmaxv, 1e5)
    zmaxv = np.maximum(zmaxv, -1e5)

    return zmaxu, zmaxv


def approximate_zmax(U, V, Um, Vm, n):
    """Approximate the location of maximum velocity in z-direction to compute the power law stresses.
    Exact values have been found numerically for both cases, but a good approximation
    is given by a hyperbola with vertical asymptote at jx/rho = U/2 (Couette profile).

    Parameters
    ----------
    U : float
        Sliding velocity of the lower surface in x-direction.
    V : float
        Sliding velocity of the lower surface in y-direction.
    Um : numpy.ndarray
        Array of mean velocities in x-direction (jx/rho).
    Vm : numpy.ndarray
        Array of mean velocities in y-direction (jy/rho).

    Returns
    -------
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in x-direction
    numpy.ndarray
        Nondimensional z-coordinate of velocity maximum in y-direction
    """

    a = 0.5
    bu = -U / (12 * n)
    cu = U / 2

    bv = -V / (12 * n)
    cv = V / 2

    if V == 0:
        zmaxv = np.ones_like(Vm) * 0.5
    else:
        zmaxv = a + bv / (Vm - cv)

    if U == 0:
        zmaxu = np.ones_like(Um) * 0.5
    else:
        zmaxu = a + bu / (Um - cu)

    # upper and lower bound for asymptotic behavior
    zmaxu = np.minimum(zmaxu, 1e5)
    zmaxu = np.maximum(zmaxu, -1e5)
    zmaxv = np.minimum(zmaxv, 1e5)
    zmaxv = np.maximum(zmaxv, -1e5)

    return zmaxu, zmaxv


def zmax_nleq_case1(zmax, n, un):
    """Definition of nonlinear equation (Case 1) to be solved for zmax with scipy.optimize.fsolve.

    Parameters
    ----------
    zmax : float
        z-location of velocity maximum
    n : float
        Power-law exponent
    un : float
        non-dimensional height-averaged velocity.

    Returns
    -------
    float
        Function value f(zmax) = 0 for root-finding.
    """

    # Case1
    return power((1 + n) / (n * (-power(zmax, 1 + 1 / n) + power(1.0 - zmax, 1 + 1 / n))), n) - power(
        ((1 + 2 * n) * (un - zmax)) / (n * (power(zmax, 2 + 1 / n) + power(1.0 - zmax, 2 + 1 / n))), n
    )


def zmax_nleq_case2(zmax, n, un):
    """Definition of nonlinear equation (Case 2) to be solved for zmax with scipy.optimize.fsolve.

    Parameters
    ----------
    zmax : float
        z-location of velocity maximum
    n : float
        Power-law exponent
    un : float
        non-dimensional height-averaged velocity.

    Returns
    -------
    float
        Function value f(zmax) = 0 for root-finding.
    """

    return 1.0 - (un * (1 + 2 * n) * (power(zmax, 1 + 1 / n) - power(zmax - 1, 1 + 1 / n))) / (
        power(zmax, 2 + 1 / n) * n - power(zmax - 1, 1 + 1 / n) * (zmax * n + (1 + n))
    )
