import numpy as np


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
