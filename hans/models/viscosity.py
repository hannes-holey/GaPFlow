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


def dukler_mixture(rho, rho_l, rho_v, eta_l, eta_v):
    """
    Computes mixture viscosity using the linear Dukler model.

    .. math::
        \\eta = \\alpha \\eta_v + (1 - \\alpha) \\eta_l

    where :math:`\\alpha = \\frac{\\rho - \\rho_l}{\\rho_v - \\rho_l}` is the vapor mass fraction.

    Parameters
    ----------
    rho : float or np.ndarray
        Mixture density.
    rho_l : float
        Liquid density.
    rho_v : float
        Vapor density.
    eta_l : float
        Viscosity of the liquid phase.
    eta_v : float
        Viscosity of the vapor phase.

    Returns
    -------
    float or np.ndarray
        Mixture viscosity.
    """
    alpha = (rho - rho_l) / (rho_v - rho_l)
    return alpha * eta_v + (1 - alpha) * eta_l


def mc_adams_mixture(rho, rho_l, rho_v, eta_l, eta_v):
    """
    Computes mixture viscosity using the McAdams model.

    .. math::
        M = \\alpha \\frac{\\rho_v}{\\rho}, \\quad
        \\eta = \\frac{\\eta_v \\eta_l}{\\eta_l M + \\eta_v (1 - M)}

    Parameters
    ----------
    rho : float or np.ndarray
        Mixture density.
    rho_l : float
        Liquid density.
    rho_v : float
        Vapor density.
    eta_l : float
        Viscosity of the liquid phase.
    eta_v : float
        Viscosity of the vapor phase.

    Returns
    -------
    float or np.ndarray
        Mixture viscosity.
    """
    alpha = (rho - rho_l) / (rho_v - rho_l)
    M = alpha * rho_v / rho
    return eta_v * eta_l / (eta_l * M + eta_v * (1 - M))


def barus_piezo(p, mu0, aB):
    """
    Computes viscosity under pressure using the Barus equation.

    .. math::
        \\mu(p) = \\mu_0 e^{a_B p}

    Parameters
    ----------
    p : float or np.ndarray
        Pressure.
    mu0 : float
        Reference viscosity.
    aB : float
        Barus pressure-viscosity coefficient.

    Returns
    -------
    float or np.ndarray
        Pressure-dependent viscosity.
    """
    return mu0 * np.exp(aB * p)


def vogel_piezo(rho, rho0, mu_inf, phi_inf, BF, g):
    """
    Computes viscosity from density using the Vogel-Fulcher-like empirical law.

    .. math::
        \\phi = \\left(\\frac{\\rho_0}{\\rho}\\right)^g, \\quad
        \\mu(\\rho) = \\mu_\\infty e^{B_F \\phi_\\infty / (\\phi - \\phi_\\infty)}

    Parameters
    ----------
    rho : float or np.ndarray
        Density.
    rho0 : float
        Reference density.
    mu_inf : float
        Asymptotic high-density viscosity.
    phi_inf : float
        Vogel parameter.
    BF : float
        Vogel-Fulcher constant.
    g : float
        Density exponent.

    Returns
    -------
    float or np.ndarray
        Density-dependent viscosity.
    """
    phi = (rho0 / rho)**g
    return mu_inf * np.exp(BF * phi_inf / (phi - phi_inf))


def eyring_shear(shear_rate, mu0, tau0):
    """
    Computes shear-thinning viscosity using the Eyring model.

    .. math::
        \\mu(\\dot{\\gamma}) = \\frac{\\tau_0}{\\dot{\\gamma}} \\sinh^{-1}\\left(\\frac{\\mu_0 \\dot{\\gamma}}{\\tau_0}\\right)

    Parameters
    ----------
    shear_rate : float or np.ndarray
        Shear rate.
    mu0 : float
        Zero-shear viscosity.
    tau0 : float
        Characteristic shear stress.

    Returns
    -------
    float or np.ndarray
        Shear-rate-dependent viscosity.
    """
    return tau0 / shear_rate * np.arcsinh(mu0 * shear_rate / tau0)


def carreau_shear(shear_rate, mu0, mu_inf, lam, a, N):
    """
    Computes shear-thinning viscosity using the Carreau model.

    .. math::
        \\mu(\\dot{\\gamma}) = \\mu_\\infty + (\\mu_0 - \\mu_\\infty) \\left[1 + (\\lambda \\dot{\\gamma})^a \\right]^{(N - 1)/a}

    Parameters
    ----------
    shear_rate : float or np.ndarray
        Shear rate.
    mu0 : float
        Zero-shear viscosity.
    mu_inf : float
        Infinite-shear viscosity.
    lam : float
        Time constant (relaxation time).
    a : float
        Power-law exponent factor.
    N : float
        Flow behavior index.

    Returns
    -------
    float or np.ndarray
        Shear-rate-dependent viscosity.
    """
    return mu_inf + (mu0 - mu_inf) * (1 + (lam * shear_rate)**a)**((N - 1) / a)


def power_law_shear(shear_rate, mu0, N):
    """
    Computes viscosity using the simple power-law model.

    .. math::
        \\mu(\\dot{\\gamma}) = \\mu_0 \\dot{\\gamma}^{N - 1}

    Parameters
    ----------
    shear_rate : float or np.ndarray
        Shear rate.
    mu0 : float
        Consistency coefficient.
    N : float
        Flow behavior index.

    Returns
    -------
    float or np.ndarray
        Shear-rate-dependent viscosity.
    """
    return mu0 * shear_rate**(N - 1)
