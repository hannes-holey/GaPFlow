import numpy as np


def srate_wall_newton(dp_dx, h=1., u1=1., u2=0., mu=1.):
    """
    Shear rate of a Newtonian fluid at bottom and top walls.
    """

    duPois = h * dp_dx / (2 * mu)
    duCarr = (u2 - u1) / h

    return -duPois + duCarr, duPois + duCarr


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
        \\mu(\\dot{\\gamma}) = \\frac{\\tau_0}{\\dot{\\gamma}} 
        \\sinh^{-1}\\left(\\frac{\\mu_0 \\dot{\\gamma}}{\\tau_0}\\right)

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
