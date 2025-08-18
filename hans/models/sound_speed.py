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


def dowson_higginson(dens, rho0, P0, C1, C2):
    """
    Computes the isothermal speed of sound using the Dowson-Higginson equation of state.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}} = \\sqrt{\\frac{C_1 \\rho_0 (C_2 - 1)}{\\rho^2 (C_2 \\rho_0 / \\rho - 1)^2}}

    Parameters
    ----------
    dens : float or np.ndarray
        Current density.
    rho0 : float
        Reference density.
    P0 : float
        Reference pressure.
    C1 : float
        Empirical constant.
    C2 : float
        Empirical constant.

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """
    c_squared = C1 * rho0 * (C2 - 1.0) * (1 / dens) ** 2 / ((C2 * rho0 / dens - 1.0) ** 2)

    return np.sqrt(c_squared)


def power_law(dens, rho0, P0, alpha):
    """
    Computes the isothermal speed of sound using a power-law equation of state.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}} = \\sqrt{\\frac{-2 P_0}{(\\alpha - 2) \\rho}
        \\left(\\frac{\\rho}{\\rho_0}\\right)^{-2 / (\\alpha - 2)}}

    Parameters
    ----------
    dens : float or np.ndarray
        Density.
    rho0 : float
        Reference density.
    P0 : float
        Reference pressure.
    alpha : float
        Power-law exponent.

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """
    c_squared = -2.0 * P0 * (dens / rho0) ** (-2.0 / (alpha - 2.0)) / ((alpha - 2) * dens)
    return np.sqrt(c_squared)


def van_der_waals(dens, M, T, a, b):
    """
    Computes the speed of sound using the Van der Waals equation of state.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}} = \\sqrt{\\frac{RTM}{(M - b\\rho)^2} - \\frac{2a\\rho}{M^2}}

    Parameters
    ----------
    dens : float or np.ndarray
        Density.
    M : float
        Molar mass.
    T : float
        Temperature (K).
    a : float
        Attraction parameter.
    b : float
        Repulsion parameter.

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """

    R = 8.314462618
    c_squared = R * T * M / (M - b * dens) ** 2 - 2 * a * dens / M**2

    return np.sqrt(c_squared)


def murnaghan_tait(dens, rho0, P0, K, n):
    """
    Computes the speed of sound from the Murnaghan-Tait equation of state.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}} = \\sqrt{\\frac{K}{\\rho_0^n} \\rho^{n - 1}}

    Parameters
    ----------
    dens : float or np.ndarray
        Current density.
    rho0 : float
        Reference density.
    P0 : float
        Reference pressure.
    K : float
        Bulk modulus.
    n : float
        Murnaghan exponent.

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """

    c_squared = K / rho0**n * dens ** (n - 1)

    return np.sqrt(c_squared)


def cubic(dens, a, b, c, d):
    """
    Computes the speed of sound from a cubic polynomial pressure law.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}} = \\sqrt{3a \\rho^2 + 2b \\rho + c}

    Parameters
    ----------
    dens : float or np.ndarray
        Density.
    a, b, c, d : float
        Polynomial coefficients.

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """
    c_squared = 3 * a * dens**2 + 2 * b * dens + c

    return np.sqrt(c_squared)


def bwr(rho, T, gamma=3.0):
    """
    Computes the speed of sound using the Benedict–Webb–Rubin (BWR) equation of state.

    Parameters
    ----------
    rho : float or np.ndarray
        Density.
    T : float
        Temperature.
    gamma : float, optional
        Exponential decay parameter (default is 3.0).

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """

    params = """
            0.8623085097507421
            2.976218765822098
            -8.402230115796038
            0.1054136629203555
            -0.8564583828174598
            1.582759470107601
            0.7639421948305453
            1.753173414312048
            2.798291772190376e+03
            -4.8394220260857657e-02
            0.9963265197721935
            -3.698000291272493e+01
            2.084012299434647e+01
            8.305402124717285e+01
            -9.574799715203068e+02
            -1.477746229234994e+02
            6.398607852471505e+01
            1.603993673294834e+01
            6.805916615864377e+01
            -2.791293578795945e+03
            -6.245128304568454
            -8.116836104958410e+03
            1.488735559561229e+01
            -1.059346754655084e+04
            -1.131607632802822e+02
            -8.867771540418822e+03
            -3.986982844450543e+01
            -4.689270299917261e+03
            2.593535277438717e+02
            -2.694523589434903e+03
            -7.218487631550215e+02
            1.721802063863269e+02
            """

    x = [float(val) for val in params.split()]

    exp_prefac = (
        rho**3 * (x[19] / T**2 + x[20] / T**3)
        + rho**5 * (x[21] / T**2 + x[22] / T**4)
        + rho**7 * (x[23] / T**2 + x[24] / T**3)
        + rho**9 * (x[25] / T**2 + x[26] / T**4)
        + rho**11 * (x[27] / T**2 + x[28] / T**3)
        + rho**13 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4)
    )

    D_exp_prefac = (
        3.0 * rho**2 * (x[19] / T**2 + x[20] / T**3)
        + 5.0 * rho**4 * (x[21] / T**2 + x[22] / T**4)
        + 7.0 * rho**6 * (x[23] / T**2 + x[24] / T**3)
        + 9.0 * rho**8 * (x[25] / T**2 + x[26] / T**4)
        + 11.0 * rho**10 * (x[27] / T**2 + x[28] / T**3)
        + 13.0 * rho**12 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4)
    )

    c_squared = (
        T
        + 2.0 * rho * (x[0] * T + x[1] * np.sqrt(T) + x[2] + x[3] / T + x[4] / T**2)
        + 3.0 * rho**2 * (x[5] * T + x[6] + x[7] / T + x[8] / T**2)
        + 4.0 * rho**3 * (x[9] * T + x[10] + x[11] / T)
        + 5.0 * rho**4 * x[12]
        + 6.0 * rho**5 * (x[13] / T + x[14] / T**2)
        + 7.0 * rho**6 * (x[15] / T)
        + 8.0 * rho**7 * (x[16] / T + x[17] / T**2)
        + 9.0 * rho**8 * (x[18] / T**2)
        + np.exp(-gamma * rho**2) * D_exp_prefac
        - 2.0 * rho * gamma * np.exp(-gamma * rho**2) * exp_prefac
    )

    return np.sqrt(c_squared)


def bayada_chupin(rho, rho_l, rho_v, c_l, c_v):
    """
    Computes the isothermal speed of sound using the Bayada-Chupin cavitation model.


    Parameters
    ----------
    rho : float or np.ndarray
        Density.
    rho_l : float
        Liquid density.
    rho_v : float
        Vapor density.
    c_l : float
        Speed of sound in liquid.
    c_v : float
        Speed of sound in vapor.

    Returns
    -------
    float or np.ndarray
        Speed of sound.
    """

    alpha = (rho - rho_l) / (rho_v - rho_l)

    if np.isscalar(rho):
        if alpha < 0:
            c_squared = c_l**2
        elif alpha >= 0 and alpha <= 1:
            c_squared = rho_v * rho_l * (c_v * c_l) ** 2 / (alpha * rho_l * c_l**2 + (1 - alpha) * rho_v * c_v**2) / rho
        else:
            c_squared = c_v**2

    else:
        mix = np.logical_and(alpha <= 1, alpha >= 0)
        c_squared = np.ones_like(rho) * c_v**2
        c_squared[alpha < 0] = c_l**2
        c_squared[mix] = rho_v * rho_l * (c_v * c_l) ** 2 / (alpha[mix] * rho_l * c_l**2 + (1 - alpha[mix]) * rho_v * c_v**2) / rho[mix]

    return np.sqrt(c_squared)
