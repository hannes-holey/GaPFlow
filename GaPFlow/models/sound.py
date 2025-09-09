import numpy as np


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
        args = ['rho', 'P0', 'C1', 'C2']

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
    c_squared = C1 * rho0 * (C2 - 1.0) * (1 / dens) ** 2 / ((C2 * rho0 / dens - 1.0) ** 2)

    return np.sqrt(c_squared)
