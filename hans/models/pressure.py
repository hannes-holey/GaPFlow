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
    Computes pressure using the Dowson-Higginson isothermal equation of state.

    .. math::
        P(\\rho) = P_0 + \\frac{C_1 (\\rho/\\rho_0 - 1)}{C_2 - \\rho/\\rho_0}

    This equation is used to describe lubricant behavior under high-pressure conditions.
    Reference: Dowson, D., & Higginson, G. R. (1977). *Elastohydrodynamic Lubrication*.

    Parameters
    ----------
    dens : float or np.ndarray
        Current fluid density.
    rho0 : float
        Reference density.
    P0 : float
        Pressure at reference density.
    C1 : float
        Empirical constant.
    C2 : float
        Empirical constant limiting maximum density ratio.

    Returns
    -------
    float or np.ndarray
        Computed pressure.

    """
    rho = np.minimum(dens, 0.99 * C2 * rho0)
    return P0 + (C1 * (rho / rho0 - 1.)) / (C2 - rho / rho0)


def power_law(dens, rho0, P0, alpha):
    """
    Computes pressure using a power-law equation of state.

    .. math::
        P(\\rho) = P_0 \\left(\\frac{\\rho}{\\rho_0}\\right)^{1 / (1 - \\frac{\\alpha}{2})}

    A generalization that includes ideal gas as a special case when alpha=0.

    Parameters
    ----------
    dens : float or np.ndarray
        Current density.
    rho0 : float
        Reference density.
    P0 : float
        Reference pressure.
    alpha : float
        Power-law exponent parameter.

    Returns
    -------
    float or np.ndarray
        Computed pressure.
    """
    return P0 * (dens / rho0)**(1. / (1. - 0.5 * alpha))


def van_der_waals(dens, M, T, a, b):
    """
    Computes pressure using the Van der Waals equation of state.

    .. math::
        P = \\frac{RT \\rho}{M - b \\rho} - a \\frac{\\rho^2}{M^2}

    Includes molecular interaction (a) and finite size (b) corrections to ideal gas law.

    Parameters
    ----------
    dens : float or np.ndarray
        Molar density (mol/m³).
    M : float
        Molar mass (g/mol).
    T : float
        Temperature (K).
    a : float
        Attraction parameter.
    b : float
        Repulsion parameter.

    Returns
    -------
    float or np.ndarray
        Computed pressure.
    """
    R = 8.314462618
    return R * T * dens / (M - b * dens) - a * dens**2 / M**2


def murnaghan_tait(dens, rho0, P0, K, n):
    """
    Computes pressure using the Murnaghan-Tait equation of state.

    .. math::
        P(\\rho) = \\frac{K}{n} \\left(\\left(\\frac{\\rho}{\\rho_0}\\right)^n - 1\\right) + P_0

    Commonly used in compressible fluid and shock wave studies.

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
        Computed pressure.

    """
    return K / n * ((dens / rho0)**n - 1) + P0


def cubic(dens, a, b, c, d):
    """
    Computes pressure using a general cubic polynomial fit.

    .. math::
        P(\\rho) = a \\rho^3 + b \\rho^2 + c \\rho + d

    Useful for empirical models where data fits a polynomial relationship.

    Parameters
    ----------
    dens : float or np.ndarray
        Density.
    a, b, c, d : float
        Polynomial coefficients.

    Returns
    -------
    float or np.ndarray
        Computed pressure.

    """
    return a * dens**3 + b * dens**2 + c * dens + d


def bwr(rho, T, gamma=3.):
    """
    Computes pressure using the Benedict–Webb–Rubin (BWR) equation of state.

    This complex EoS models real fluid behavior accurately over wide conditions.

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
        Computed pressure.
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

    p = rho * T +\
        rho**2 * (x[0] * T + x[1] * np.sqrt(T) + x[2] + x[3] / T + x[4] / T**2) +\
        rho**3 * (x[5] * T + x[6] + x[7] / T + x[8] / T**2) +\
        rho**4 * (x[9] * T + x[10] + x[11] / T) +\
        rho**5 * x[12] +\
        rho**6 * (x[13] / T + x[14] / T**2) +\
        rho**7 * (x[15] / T) +\
        rho**8 * (x[16] / T + x[17] / T**2) +\
        rho**9 * (x[18] / T**2) +\
        np.exp(-gamma * rho**2) * (rho**3 * (x[19] / T**2 + x[20] / T**3) +
                                   rho**5 * (x[21] / T**2 + x[22] / T**4) +
                                   rho**7 * (x[23] / T**2 + x[24] / T**3) +
                                   rho**9 * (x[25] / T**2 + x[26] / T**4) +
                                   rho**11 * (x[27] / T**2 + x[28] / T**3) +
                                   rho**13 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4))

    return p


def bayada_chupin(rho, rho_l, rho_v, c_l, c_v):
    """
    Computes pressure using the Bayada-Chupin cavitation model.

    Models lubricated film pressure in the presence of phase change.
    Reference: Bayada, G., & Chupin, L. (2013). *Journal of Tribology, 135(4), 041703*.

    Parameters
    ----------
    rho : float or np.ndarray
        Current density.
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
        Computed pressure.

    """
    N = rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l) / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    Pcav = rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))
    alpha = (rho - rho_l) / (rho_v - rho_l)

    if np.isscalar(rho):
        if alpha < 0:
            p = Pcav + (rho - rho_l) * c_l**2
        elif alpha >= 0 and alpha <= 1:
            p = Pcav + N * np.log(rho_v * c_v**2 * rho / (rho_l * (rho_v * c_v **
                                                                   2 * (1 - alpha) + rho_l * c_l**2 * alpha)))
        else:
            p = c_v**2 * rho

    else:
        rho_mix = rho[np.logical_and(alpha <= 1, alpha >= 0)]
        alpha_mix = alpha[np.logical_and(alpha <= 1, alpha >= 0)]

        p = c_v**2 * rho
        p[alpha < 0] = Pcav + (rho[alpha < 0] - rho_l) * c_l**2
        p[np.logical_and(alpha <= 1, alpha >= 0)] = Pcav + \
            N * np.log(rho_v * c_v**2 * rho_mix / (rho_l * (rho_v * c_v **
                                                            2 * (1 - alpha_mix) + rho_l * c_l**2 * alpha_mix)))

    return p
