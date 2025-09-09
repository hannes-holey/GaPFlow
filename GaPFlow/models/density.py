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


def power_law(p, rho0=1.1853, P0=101325., alpha=0.):
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
