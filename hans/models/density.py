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


def dowson_higginson(p, rho0, P0, C1, C2):
    """
    Computes density using the inverse of the Dowson-Higginson isothermal equation of state.

    This model is often used to describe lubricants under high pressure.

    .. math::
        \\rho(P) = \\rho_0 \\frac{C_1 + C_2 (P - P_0)}{C_1 + (P - P_0)}

    Parameters
    ----------
    p : float or np.ndarray
        Pressure.
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
        Computed density.
    """

    return rho0 * (C1 + C2 * (p - P0)) / (C1 + p - P0)


def power_law(p, rho0, P0, alpha):
    """
    Computes density from pressure using an inverse power-law equation of state.

    This general form includes the ideal gas law as a special case when :math:`\\alpha = 0`.

    .. math::
        \\rho(P) = \\rho_0 \\left(\\frac{P}{P_0}\\right)^{1 - \\alpha/2}

    Parameters
    ----------
    p : float or np.ndarray
        Pressure.
    rho0 : float
        Reference density.
    P0 : float
        Reference pressure.
    alpha : float
        Power-law parameter.

    Returns
    -------
    float or np.ndarray
        Computed density.
    """
    return rho0 * (p / P0)**(1. - alpha / 2.)


def bayada_chupin(p, rho_l, rho_v, c_l, c_v):
    """
    Computes density from pressure using the Bayada-Chupin cavitation model.

    This model accounts for phase transitions in lubricated thin films.
    Reference: Bayada & Chupin, *J. Tribology*, 135(4), 2013.

    The model defines density as a piecewise function depending on pressure:

    - For :math:`P > P_{cav}`:

    .. math::
        \\rho = \\rho_l + \\frac{P - P_{cav}}{c_l^2}

    - For :math:`P < c_v^2 \\rho_v`:

    .. math::
        \\rho = \\frac{P}{c_v^2}

    - Else (mixture phase):

    .. math::
        \\rho = \\frac{(c_l^2 \\rho_l^3 - c_v^2 \\rho_l \\rho_v^2) \\exp\\left(\\frac{P - P_{cav}}{N}\\right)}
        {(c_l^2 \\rho_l^2 - c_v^2 \\rho_l \\rho_v) \\exp\\left(\\frac{P - P_{cav}}{N}\\right) + c_v^2 \\rho_v (\\rho_l - \\rho_v)}

    Parameters
    ----------
    p : float or np.ndarray
        Pressure.
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
        Computed density.
    """

    N = rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l) / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    Pcav = rho_v * c_v**2 - N * np.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))

    if np.isscalar(p):
        if p > Pcav:
            rho = (p - Pcav) / c_l**2 + rho_l
        elif p < c_v**2 * rho_v:
            rho = p / c_v**2
        else:
            rho = (c_l**2 * rho_l**3 - c_v**2 * rho_l * rho_v**2) * np.exp((p - Pcav) / N) \
                / ((c_l**2 * rho_l**2 - c_v**2 * rho_l * rho_v) * np.exp((p - Pcav) / N) + c_v**2 * rho_v * (-rho_v + rho_l))
    else:
        p_mix = np.logical_and(p <= Pcav, p >= c_v**2 * rho_v)
        rho = p / c_v**2
        rho[p > Pcav] = (p[p > Pcav] - Pcav) / c_l**2 + rho_l
        rho[p_mix] = (c_l**2 * rho_l**3 - c_v**2 * rho_l * rho_v**2) * np.exp((p[p_mix] - Pcav) / N) \
            / ((c_l**2 * rho_l**2 - c_v**2 * rho_l * rho_v) * np.exp((p[p_mix] - Pcav) / N) + c_v**2 * rho_v * (-rho_v + rho_l))

    return rho


def murnaghan_tait(p, rho0, P0, K, n):
    """
    Computes density from pressure using the inverse of the Murnaghan-Tait equation of state.

    Commonly used in shockwave modeling and compressible fluid flow.

    .. math::
        \\rho(P) = \\rho_0 \\left(1 + \\frac{n}{K} (P - P_0) \\right)^{1/n}

    Parameters
    ----------
    p : float or np.ndarray
        Pressure.
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
        Computed density.
    """

    return rho0 * (1 + n / K * (p - P0))**(1 / n)
