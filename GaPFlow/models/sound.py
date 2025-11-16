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


def eos_sound_velocity(density, prop):
    """Wrapper around all implemented equation of state models.

    Computes the local speed of sound for a given density field.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}}

    Parameters
    ----------
    density : np.ndarray
        The mass density field
    prop : dict
        Material properties

    Returns
    -------
    np.ndarray
        Sound speed field for the corresponding density field
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
    dp_drho = C1 * rho0 * (C2 - 1.0) * (1 / dens) ** 2 / ((C2 * rho0 / dens - 1.0) ** 2)

    return np.sqrt(dp_drho)


def power_law(dens, rho0=1.1853, P0=101325., alpha=0.):
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
    dp_drho = -2.0 * P0 * (dens / rho0) ** (-2.0 / (alpha - 2.0)) / ((alpha - 2) * dens)
    return np.sqrt(dp_drho)


def van_der_waals(dens, M=39.948, T=100., a=1.355, b=0.03201):
    """
    Computes the speed of sound using the Van der Waals equation of state.

    .. math::
        c = \\sqrt{\\frac{dp}{d\\rho}} = \\sqrt{\\frac{RTM}{(M - b\\rho)^2} - \\frac{2a\\rho}{M^2}}

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
        Speed of sound.
    """

    R = gas_constant
    mol_dens = dens / M * 1000.
    a /= 10.  # to m^6 Pa / mol^2
    b /= 1000.  # to m^3  / mol

    dp_drho = R * T / (1. - b * mol_dens)**2 - 2. * a * mol_dens
    return np.sqrt(dp_drho)


def murnaghan_tait(dens, rho0=700, P0=0.101e6, K=0.557e9, n=7.33):
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

    dp_drho = K / rho0**n * dens ** (n - 1)

    return np.sqrt(dp_drho)


def cubic(dens, a=15.2, b=-9.6, c=3.35, d=-0.07):
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
    dp_drho = 3 * a * dens**2 + 2 * b * dens + c

    return np.sqrt(dp_drho)


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
    config = os.path.join(os.path.dirname(__file__), "bwr_coeffs.txt")
    x = np.loadtxt(config)

    exp_prefac = (
        rho**3 * (x[19] / T**2 + x[20] / T**3)
        + rho**5 * (x[21] / T**2 + x[22] / T**4)  # noqa: W503
        + rho**7 * (x[23] / T**2 + x[24] / T**3)  # noqa: W503
        + rho**9 * (x[25] / T**2 + x[26] / T**4)  # noqa: W503
        + rho**11 * (x[27] / T**2 + x[28] / T**3)  # noqa: W503
        + rho**13 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4)  # noqa: W503
    )

    D_exp_prefac = (
        3.0 * rho**2 * (x[19] / T**2 + x[20] / T**3)
        + 5.0 * rho**4 * (x[21] / T**2 + x[22] / T**4)  # noqa: W503
        + 7.0 * rho**6 * (x[23] / T**2 + x[24] / T**3)  # noqa: W503
        + 9.0 * rho**8 * (x[25] / T**2 + x[26] / T**4)  # noqa: W503
        + 11.0 * rho**10 * (x[27] / T**2 + x[28] / T**3)  # noqa: W503
        + 13.0 * rho**12 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4)  # noqa: W503
    )

    dp_drho = (
        T
        + 2.0 * rho * (x[0] * T + x[1] * np.sqrt(T) + x[2] + x[3] / T + x[4] / T**2)  # noqa: W503
        + 3.0 * rho**2 * (x[5] * T + x[6] + x[7] / T + x[8] / T**2)  # noqa: W503
        + 4.0 * rho**3 * (x[9] * T + x[10] + x[11] / T)  # noqa: W503
        + 5.0 * rho**4 * x[12]  # noqa: W503
        + 6.0 * rho**5 * (x[13] / T + x[14] / T**2)  # noqa: W503
        + 7.0 * rho**6 * (x[15] / T)  # noqa: W503
        + 8.0 * rho**7 * (x[16] / T + x[17] / T**2)  # noqa: W503
        + 9.0 * rho**8 * (x[18] / T**2)  # noqa: W503
        + np.exp(-gamma * rho**2) * D_exp_prefac  # noqa: W503
        - 2.0 * rho * gamma * np.exp(-gamma * rho**2) * exp_prefac  # noqa: W503
    )

    return np.sqrt(dp_drho)


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
        c_squared[mix] = rho_v * rho_l * (c_v * c_l) ** 2 / (alpha[mix] * rho_l * c_l **  # noqa: W504
                                                             2 + (1 - alpha[mix]) * rho_v * c_v**2) / rho[mix]

    return np.sqrt(c_squared)
