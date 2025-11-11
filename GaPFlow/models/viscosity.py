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
import numpy.typing as npt


def piezoviscosity(p: float | npt.NDArray,
                   mu0: float | npt.NDArray,
                   piezo_dict: dict) -> float | npt.NDArray:
    """Wrapper around implemented piezoviscosity models.

    Parameters
    ----------
    p : float or Array
        Pressure (field) (or density for Bayada-Chupin EoS)
    mu0 : float
        Newtonian viscosity
    piezo_dict : dict
        Parameters

    Returns
    -------
    float or Array
        Pressure-dependent viscosity
    """

    if piezo_dict['name'] == 'Barus':
        func = barus_piezo
    elif piezo_dict['name'] == 'Roelands':
        func = roelands_piezo
    # for cavitation model of Bayada & Chupin
    elif piezo_dict['name'] == 'Dukler':
        func = dukler_mixture
    elif piezo_dict['name'] == 'McAdams':
        func = mc_adams_mixture
    else:
        func = lambda p, mu, **kwargs: np.ones_like(p) * mu

    return func(p, mu0, **piezo_dict)


def shear_thinning_factor(shear_rate: float | npt.NDArray,
                          mu0: float | npt.NDArray,
                          thinning_dict: dict) -> float | npt.NDArray:
    """Wrapper around implemented piezoviscosity models.

    Parameters
    ----------
    shear_rate : float or Array
        Shear rate (field)
    mu0 : float
        Newtonian viscosity
    thinning_dict : dict
        Parameters

    Returns
    -------
    float or Array
        Shear rate-dependent viscosity
    """

    if thinning_dict['name'] == 'Eyring':
        func = eyring_shear
    elif thinning_dict['name'] == 'Carreau':
        func = carreau_shear
    else:
        func = lambda gamma, mu, **kwargs: np.ones_like(gamma)

    return func(shear_rate, mu0, **thinning_dict)


def srate_wall_newton(dp_dx, h=1., u1=1., u2=0., mu=1.):
    """
    Shear rate of a Newtonian fluid at bottom and top walls.
    """

    duPois = h * dp_dx / (2 * mu)
    duCarr = (u2 - u1) / h

    return -duPois + duCarr, duPois + duCarr


def shear_rate_avg(dp_dx, dp_dy, h, u1, u2, mu):
    """Average shear rate.

    This assumes a Newtonian flow profile, i.e. a linear shear rate across the gap.

    Parameters
    ----------
    dp_dx : float or Array
        Pressure gradient x
    dp_dy : float or Array
        Pressure gradient y
    h : float or Array
        Gap height
    u1 : float
        Velocity lower wall
    u2 : float
        Velocity upper wall
    mu : float or Array
        Viscosity

    Returns
    -------
    float or Array
        Average shear rate
    """

    # instead of different viscosities in x and y direction
    grad_p = np.hypot(dp_dx, dp_dy)

    sr_bot, sr_top = srate_wall_newton(grad_p, h, u1, u2, mu)

    return (np.abs(sr_top) + np.abs(sr_bot)) / 2.


def barus_piezo(p, mu0, aB=2.e-8, name='Barus'):
    """
    Computes viscosity under pressure using the Barus equation.

    .. math::
        \\mu(p) = \\mu_0 e^{a_B p}

    Parameters
    ----------
    p : float or Array
        Pressure.
    mu0 : float
        Reference viscosity.
    aB : float
        Barus pressure-viscosity coefficient.

    Returns
    -------
    float or Array
        Pressure-dependent viscosity.
    """
    return mu0 * np.exp(aB * p)


def roelands_piezo(p, mu0, mu_inf=1.e-3, p_ref=1.96e8, z=0.68, name='Roelands'):
    """
    Computes the pressure-dependent viscosity using Roeland's empirical piezoviscosity equation.

    Roeland's equation models the increase of viscosity with pressure, commonly used
    in elastohydrodynamic lubrication and high-pressure fluid applications.

    .. math::
        \\mu(p) = \\mu_0 * \\exp( \\ln(\\mu_0/\\mu_{\\infty})(-1 + (1 + p/p_{ref})^z))

    Parameters
    ----------
    p : float or Array
        Pressure at which the viscosity is evaluated.
    mu0 : float
        Viscosity at ambient pressure.
    mu_inf : float
        Viscosity at very high pressure.
    pR : float
        Reference pressure, characteristic of the fluid.
    zR : float
        Pressure exponent, controlling the curvature of the viscosity increase.

    Returns
    -------
    float or Array
        Pressure-dependent viscosity, same shape as `p`.
    """

    return mu0 * np.exp(np.log(mu0 / mu_inf) * (-1 + (1 + p / p_ref)**z))


def dukler_mixture(rho, eta_l, eta_v=3.9e-5, rho_l=850., rho_v=0.019, name='Dukler'):
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


def mc_adams_mixture(rho, eta_l, eta_v=3.9e-5, rho_l=850., rho_v=0.019, name='McAdams'):
    """
    Computes mixture viscosity using the McAdams model.

    .. math::
        M = \\alpha \\frac{\\rho_v}{\\rho}, \\quad
        \\eta = \\frac{\\eta_v \\eta_l}{\\eta_l M + \\eta_v (1 - M)}

    Parameters
    ----------
    rho : float or np.ndarray
        Mixture density.
    eta_l : float
        Viscosity of the liquid phase.
    eta_v : float
        Viscosity of the vapor phase.
    rho_l : float
        Liquid density.
    rho_v : float
        Vapor density.

    Returns
    -------
    float or np.ndarray
        Mixture viscosity.
    """
    alpha = (rho - rho_l) / (rho_v - rho_l)
    M = alpha * rho_v / rho
    return eta_v * eta_l / (eta_l * M + eta_v * (1 - M))


def eyring_shear(shear_rate, mu0, tauE=5.e5, name='Eyring'):
    """
    Computes shear-thinning viscosity using the Eyring model.

    .. math::
        \\mu(\\dot{\\gamma}) = \\frac{\\tau_0}{\\dot{\\gamma}}
        \\sinh^{-1}\\left(\\frac{\\mu_0 \\dot{\\gamma}}{\\tau_0}\\right)

    Parameters
    ----------
    shear_rate : float or Array
        Shear rate.
    mu0 : float
        Zero-shear viscosity.
    tauE : float
        Eyring stress.

    Returns
    -------
    float or Array
        Shear-rate-dependent viscosity.
    """
    tau0 = mu0 * shear_rate
    return tauE / tau0 * np.arcsinh(tau0 / tauE)


def carreau_shear(shear_rate, mu0, mu_inf=1.e-3, lam=0.02, a=2, N=0.8, name='Carreau'):
    """
    Computes shear-thinning viscosity using the Carreau model.

    .. math::
        \\mu(\\dot{\\gamma}) = \\mu_\\infty +
        (\\mu_0 - \\mu_\\infty) \\left[1 + (\\lambda \\dot{\\gamma})^a \\right]^{(N - 1)/a}

    Parameters
    ----------
    shear_rate : float or Array
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
    float or Array
        Shear-rate-dependent viscosity.
    """
    mu = mu_inf + (mu0 - mu_inf) * (1 + (lam * shear_rate)**a)**((N - 1) / a)

    return mu / mu0
