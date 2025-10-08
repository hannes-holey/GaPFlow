import numpy as np


def piezoviscosity(p, mu0, piezo_dict):

    # name = piezo_dict.pop('type', None)

    if piezo_dict['name'] == 'Barus':
        func = barus_piezo
    elif piezo_dict['name'] == 'Roelands':
        func = roelands_piezo
    else:
        func = lambda p, mu, **kwargs: np.ones_like(p) * mu

    return func(p, mu0, **piezo_dict)


def shear_thinning_factor(shear_rate, mu0, thinning_dict):

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


def roelands_piezo(p, mu0, mu_inf=1.e-3, p_ref=1.96e8, z=0.68, name='Roelands'):
    """
    Computes the pressure-dependent viscosity using Roeland's empirical piezoviscosity equation.

    Roeland's equation models the increase of viscosity with pressure, commonly used 
    in elastohydrodynamic lubrication and high-pressure fluid applications.

    .. math::
        \\mu(p) = \\mu_0 * \\exp( \\ln(\\mu_0/\\mu_{\\infty})(-1 + (1 + p/p_R)^z_R))

    Parameters
    ----------
    p : float or np.ndarray
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
    float or np.ndarray
        Pressure-dependent viscosity, same shape as `p`.
    """

    return mu0 * np.exp(np.log(mu0 / mu_inf) * (-1 + (1 + p / p_ref)**z))


def eyring_shear(shear_rate, mu0, tauE=5.e5, name='Eyring'):
    """
    Computes shear-thinning viscosity using the Eyring model.

    .. math::
        \\mu(\\dot{\\gamma}) = \\frac{\\tau_0}{\\dot{\\gamma}} 
        \\sinh^{-1}\\left(\\frac{\\mu_0 \\dot{\\gamma}}{\\tau_0}\\right)

    Parameters
    ----------
    shear_rate : float or np.ndarray
        Shear rate.
    mu0 : float
        Zero-shear viscosity.
    tauE : float
        Eyring stress.

    Returns
    -------
    float or np.ndarray
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
    mu = mu_inf + (mu0 - mu_inf) * (1 + (lam * shear_rate)**a)**((N - 1) / a)

    return mu / mu0
