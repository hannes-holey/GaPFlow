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
import os
import numpy as np
from scipy.constants import gas_constant


def eos_pressure(density, prop):
    """Wrapper around all implemented equation of state models.

    Parameters
    ----------
    density : np.ndarray
        The mass density field
    prop : dict
        Material properties

    Returns
    -------
    np.ndarray
        Pressure field for the corresponding density field
    """

    if prop['EOS'] == 'DH':
        func = dowson_higginson
        args = ['rho0', 'P0', 'C1', 'C2']
    elif prop['EOS'] == 'PL':
        func = power_law
        args = ['rho0', 'P0', 'alpha']
    elif prop['EOS'] == 'vdW':
        func = van_der_waals
        args = ['M', 'T', 'a', 'b']
    elif prop['EOS'] == "MT":
        func = murnaghan_tait
        args = ['rho0', 'P0', 'K', 'n']
    elif prop['EOS'] == "cubic":
        func = cubic
        args = ['a', 'b', 'c', 'd']
    elif prop['EOS'] == "BWR":
        func = bwr
        args = ['T', 'gamma']
    elif prop['EOS'] == 'Bayada':
        func = bayada_chupin
        args = ['rho_l', 'rho_v', 'c_l', 'c_v']

    # TODO: split EOS and stress arguments already in input
    kwargs = {k: v for k, v in prop.items() if k in args}

    return func(density, **kwargs)


def dowson_higginson(dens, rho0=877.7007, P0=101325., C1=3.5e8, C2=1.23):
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


def power_law(dens, rho0=1.1853, P0=101325., alpha=0.):
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


def van_der_waals(dens, M=39.948, T=100., a=1.355, b=0.03201):
    """
    Computes pressure using the Van der Waals equation of state.

    .. math::
        P = \\frac{RT \\rho}{M - b \\rho} - a \\frac{\\rho^2}{M^2}

    Includes molecular interaction (a) and finite size (b) corrections to ideal gas law.

    Parameters
    ----------
    dens : float or np.ndarray
        Mass density (kg/m³).
    M : float
        Molar mass (g/mol).
    T : float
        Temperature (K).
    a : float
        Attraction parameter (L^2 bar/mol^2).
    b : float
        Repulsion parameter (L/mol).

    Returns
    -------
    float or np.ndarray
        Computed pressure.
    """

    R = gas_constant
    mol_dens = dens / M * 1000.
    a /= 10.  # to m^6 Pa / mol^2
    b /= 1000.  # to m^3  / mol

    return R * T * mol_dens / (1. - b * mol_dens) - a * mol_dens**2


def murnaghan_tait(dens, rho0=700, P0=0.101e6, K=0.557e9, n=7.33):
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


def cubic(dens, a=15.2, b=-9.6, c=3.35, d=-0.07):
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


def bwr(dens, T, gamma=3.):
    """
    Computes pressure using the Benedict–Webb–Rubin (BWR) equation of state.

    This complex EoS models real fluid behavior accurately over wide conditions.

    Parameters
    ----------
    dens : float or np.ndarray
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

    config = os.path.join(os.path.dirname(__file__), "bwr_coeffs.txt")
    x = np.loadtxt(config)

    p = dens * T +\
        dens**2 * (x[0] * T + x[1] * np.sqrt(T) + x[2] + x[3] / T + x[4] / T**2) +\
        dens**3 * (x[5] * T + x[6] + x[7] / T + x[8] / T**2) +\
        dens**4 * (x[9] * T + x[10] + x[11] / T) +\
        dens**5 * x[12] +\
        dens**6 * (x[13] / T + x[14] / T**2) +\
        dens**7 * (x[15] / T) +\
        dens**8 * (x[16] / T + x[17] / T**2) + \
        dens**9 * (x[18] / T**2) +\
        np.exp(-gamma * dens**2) * (dens**3 * (x[19] / T**2 + x[20] / T**3) +  # noqa: W504
                                    dens**5 * (x[21] / T**2 + x[22] / T**4) +  # noqa: W504
                                    dens**7 * (x[23] / T**2 + x[24] / T**3) +  # noqa: W504
                                    dens**9 * (x[25] / T**2 + x[26] / T**4) +  # noqa: W504
                                    dens**11 * (x[27] / T**2 + x[28] / T**3) +  # noqa: W504
                                    dens**13 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4))

    return p


def bayada_chupin(dens, rho_l, rho_v, c_l, c_v):
    """
    Computes pressure using the Bayada-Chupin cavitation model.

    Models lubricated film pressure in the presence of phase change.
    Reference: Bayada, G., & Chupin, L. (2013). *Journal of Tribology, 135(4), 041703*.

    Parameters
    ----------
    dens : float or np.ndarray
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
    alpha = (dens - rho_l) / (rho_v - rho_l)

    if np.isscalar(dens):
        if alpha < 0:
            p = Pcav + (dens - rho_l) * c_l**2
        elif alpha >= 0 and alpha <= 1:
            denominator = rho_l * (rho_v * c_v**2 * (1 - alpha) + rho_l * c_l**2 * alpha)
            p = Pcav + N * np.log(rho_v * c_v**2 * dens / denominator)
        else:
            p = c_v**2 * dens

    else:
        dens_mix = dens[np.logical_and(alpha <= 1, alpha >= 0)]
        alpha_mix = alpha[np.logical_and(alpha <= 1, alpha >= 0)]

        p = c_v**2 * dens
        p[alpha < 0] = Pcav + (dens[alpha < 0] - rho_l) * c_l**2
        denominator = rho_l * (rho_v * c_v**2 * (1 - alpha_mix) + rho_l * c_l**2 * alpha_mix)
        p[np.logical_and(alpha <= 1, alpha >= 0)] = Pcav + N * np.log(rho_v * c_v**2 * dens_mix / denominator)

    return p
