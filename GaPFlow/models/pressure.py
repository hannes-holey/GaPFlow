#
# Copyright 2025-2026 Hannes Holey
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

# flake8: noqa: W503

"""Equation of state (pressure).

Pressure-density relations for the implemented models.
"""

import os
import numpy as np
import jax.numpy as jnp
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
    elif prop['EOS'] == 'user':
        return prop['EOS_user'](density)

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
    rho = jnp.minimum(dens, 0.99 * C2 * rho0)
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
    Reference: Macdonald, J. R. (1966). *Reviews of Modern Physics, 38, 669*

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
    Reference: Benedict, M.; Webb, G. B.; Rubin, L. C. (1940), *Journal of Chemical Physics, 8, 334–345*

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
    x = jnp.array(np.loadtxt(config))

    p = dens * T +\
        dens**2 * (x[0] * T + x[1] * jnp.sqrt(T) + x[2] + x[3] / T + x[4] / T**2) +\
        dens**3 * (x[5] * T + x[6] + x[7] / T + x[8] / T**2) +\
        dens**4 * (x[9] * T + x[10] + x[11] / T) +\
        dens**5 * x[12] +\
        dens**6 * (x[13] / T + x[14] / T**2) +\
        dens**7 * (x[15] / T) +\
        dens**8 * (x[16] / T + x[17] / T**2) + \
        dens**9 * (x[18] / T**2) +\
        jnp.exp(-gamma * dens**2) * (dens**3 * (x[19] / T**2 + x[20] / T**3) +  # noqa: W504
                                    dens**5 * (x[21] / T**2 + x[22] / T**4) +  # noqa: W504
                                    dens**7 * (x[23] / T**2 + x[24] / T**3) +  # noqa: W504
                                    dens**9 * (x[25] / T**2 + x[26] / T**4) +  # noqa: W504
                                    dens**11 * (x[27] / T**2 + x[28] / T**3) +  # noqa: W504
                                    dens**13 * (x[29] / T**2 + x[30] / T**3 + x[31] / T**4))

    return p


def bayada_chupin(dens, rho_l, rho_v, c_l, c_v, smooth_width=0.0):
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
    N = (
        rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
        / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2)
    )

    Pcav = rho_v * c_v**2 - N * jnp.log(
        rho_v**2 * c_v**2 / (rho_l**2 * c_l**2)
    )

    alpha = (dens - rho_l) / (rho_v - rho_l)

    # --- Region 1: alpha < 0 (liquid)
    p_liq = Pcav + (dens - rho_l) * c_l**2

    # --- Region 2: 0 <= alpha <= 1 (mixture)
    denominator = rho_l * (
        rho_v * c_v**2 * (1 - alpha)
        + rho_l * c_l**2 * alpha
    )

    p_mix = Pcav + N * jnp.log(
        rho_v * c_v**2 * dens / denominator
    )

    # --- Region 3: alpha > 1 (vapor)
    p_vap = c_v**2 * dens

    # --- Combine using jnp.where (no dynamic indexing!)
    p = jnp.where(
        alpha < 0,
        p_liq,
        jnp.where(alpha <= 1, p_mix, p_vap),
    )

    return p


# ---------------------------------------------------------------------------
# Inverse EoS — closed-form inverses for the pressure-based solver.
#
# Only 'DH' and 'Bayada' are implemented for now; other EoS variants raise
# NotImplementedError in the eos_rho / eos_drho_dp dispatchers.
# ---------------------------------------------------------------------------


def rho_of_p_dh(pressure, rho0=877.7007, P0=101325., C1=3.5e8, C2=1.23):
    """Inverse of Dowson-Higginson EoS.

    Closed-form:
        rho(p) = rho_0 * [C_2 (p - P_0) + C_1] / [(p - P_0) + C_1]

    Inverts the forward form
        p(rho) = P_0 + C_1 (rho/rho_0 - 1) / (C_2 - rho/rho_0).
    """
    u = pressure - P0
    return rho0 * (C2 * u + C1) / (u + C1)


def rho_of_p_pl(pressure, rho0=1.1853, P0=101325., alpha=0.):
    """Inverse of power-law EoS.

    Closed-form:
        rho(p) = rho_0 * (p / P_0)^(1 - alpha/2)

    Inverts the forward form
        p(rho) = P_0 * (rho/rho_0)^(1 / (1 - alpha/2)).

    For alpha=0 this is a linear EoS; widely used as the canonical test
    case for FD Jacobian verification.
    """
    return rho0 * (pressure / P0) ** (1. - 0.5 * alpha)


def rho_of_p_bayada(pressure, rho_l, rho_v, c_l, c_v):
    """Inverse of Bayada-Chupin EoS (three-region piecewise).

    Liquid  (p >= Pcav):          rho = rho_l + (p - Pcav) / c_l^2
    Mixture (P_vt <= p <= Pcav):  rho = q*A / (1 - q*B) with
                                  q = exp((p - Pcav)/N) / (rho_v*c_v^2)
    Vapor   (p < P_vt):           rho = p / c_v^2

    P_vt = rho_v*c_v^2 is the mixture-vapor boundary pressure; Pcav is the
    cavitation (mixture-liquid) boundary pressure.

    jnp.clip on the mixture branch is mandatory: exp((p-Pcav)/N) overflows
    for liquid-region p, and jax.grad through jnp.where is poisoned by NaN
    intermediates on the inactive branch unless the input is clamped.
    """
    N = (rho_v * c_v**2 * rho_l * c_l**2 * (rho_v - rho_l)
         / (rho_v**2 * c_v**2 - rho_l**2 * c_l**2))
    Pcav = rho_v * c_v**2 - N * jnp.log(rho_v**2 * c_v**2 / (rho_l**2 * c_l**2))
    P_vt = rho_v * c_v**2
    A = rho_l * (rho_v**2 * c_v**2 - rho_l**2 * c_l**2) / (rho_v - rho_l)
    B = rho_l * (rho_l * c_l**2 - rho_v * c_v**2) / (rho_v - rho_l)

    p_mix_safe = jnp.clip(pressure, P_vt, Pcav)

    rho_liq = rho_l + (pressure - Pcav) / c_l**2
    rho_vap = pressure / c_v**2
    q = jnp.exp((p_mix_safe - Pcav) / N) / (rho_v * c_v**2)
    rho_mix = q * A / (1.0 - q * B)

    return jnp.where(pressure >= Pcav, rho_liq,
                     jnp.where(pressure >= P_vt, rho_mix, rho_vap))


def eos_rho(pressure, prop):
    """Inverse EoS dispatcher. Returns density for a given pressure field.

    Mirrors `eos_pressure` in structure. Supported: 'DH', 'PL', 'Bayada'.
    """
    if prop['EOS'] == 'DH':
        func = rho_of_p_dh
        args = ['rho0', 'P0', 'C1', 'C2']
    elif prop['EOS'] == 'PL':
        func = rho_of_p_pl
        args = ['rho0', 'P0', 'alpha']
    elif prop['EOS'] == 'Bayada':
        func = rho_of_p_bayada
        args = ['rho_l', 'rho_v', 'c_l', 'c_v']
    elif prop['EOS'] == 'user':
        return prop['EOS_user_inv'](pressure)
    else:
        raise NotImplementedError(
            f"Pressure-based solver requires an inverse EoS; "
            f"'{prop['EOS']}' not implemented. Supported: 'DH', 'PL', 'Bayada'.")

    kwargs = {k: v for k, v in prop.items() if k in args}
    return func(pressure, **kwargs)


def eos_drho_dp(pressure, prop):
    """Derivative drho/dp via jax.grad on the inverse EoS.

    Consistent with `eos_pressure` by construction. Supported: 'DH', 'PL',
    'Bayada'. Handles arbitrary input shapes (0D/1D/2D/nD).
    """
    from jax import grad, vmap

    if prop['EOS'] == 'DH':
        args = ['rho0', 'P0', 'C1', 'C2']
        kwargs = {k: prop[k] for k in args}
        scalar_fn = lambda p: rho_of_p_dh(p, **kwargs)
    elif prop['EOS'] == 'PL':
        args = ['rho0', 'P0', 'alpha']
        kwargs = {k: prop[k] for k in args}
        scalar_fn = lambda p: rho_of_p_pl(p, **kwargs)
    elif prop['EOS'] == 'Bayada':
        args = ['rho_l', 'rho_v', 'c_l', 'c_v']
        kwargs = {k: prop[k] for k in args}
        scalar_fn = lambda p: rho_of_p_bayada(p, **kwargs)
    else:
        raise NotImplementedError(
            f"eos_drho_dp not implemented for EOS '{prop['EOS']}'. "
            f"Supported: 'DH', 'PL', 'Bayada'.")

    pressure = jnp.asarray(pressure)
    shape = pressure.shape
    if shape == ():
        return grad(scalar_fn)(pressure)
    return vmap(grad(scalar_fn))(pressure.ravel()).reshape(shape)
