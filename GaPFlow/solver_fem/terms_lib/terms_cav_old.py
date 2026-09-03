#
# Copyright 2026 Christoph Huber
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# flake8: noqa: E501

import numpy as np

from ..terms import Term

__all__ = [
    'R11x_cav_old', 'R11y_cav_old', 'R11Sx_cav_old', 'R11Sy_cav_old',
    'R1T_cav_old', 'R1Th_cav_old', 'R_cav_old', 'R1STx_old', 'R1STy_old',
    'R24x_cav_old', 'R24y_cav_old',
    'CAV_MASS_TERMS_old', 'THETA_TERMS_AD_old', 'THETA_TERMS_WALL_STRESS_old',
]


def _fb_denom(a, b):
    """sqrt(a²+b²), bumped to machine epsilon at (0,0) to avoid singularity."""
    d = np.sqrt(a**2 + b**2)
    return np.where(d == 0.0, np.finfo(float).eps, d)


# -----------------------------------------------------------------------------
# mass conservation: fluxes with cavity fraction (theta)
# -----------------------------------------------------------------------------

R11x_cav_old = Term(
    name='R11x_cav_old',
    description='flux divergence x Elrod-Adams (IBP)',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=[],
    fun=lambda ctx: lambda jx, theta: - (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta: - (1 - theta),
        lambda ctx: lambda jx, theta:  jx,
    ],
    test_deriv='x')

R11y_cav_old = Term(
    name='R11y_cav_old',
    description='flux divergence y Elrod-Adams (IBP)',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=[],
    fun=lambda ctx: lambda jy, theta: - (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta: - (1 - theta),
        lambda ctx: lambda jy, theta:  jy,
    ],
    test_deriv='y')

R11Sx_cav_old = Term(
    name='R11Sx_cav_old',
    description='flux divergence height source x Elrod-Adams',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda jx, theta: -1 / ctx['h']() * ctx['dh_dx']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta: -1 / ctx['h']() * ctx['dh_dx']() * (1 - theta),
        lambda ctx: lambda jx, theta:  1 / ctx['h']() * ctx['dh_dx']() * jx,
    ],
    der_h=lambda ctx: lambda jx, theta: 1 / ctx['h']() ** 2 * ctx['dh_dx']() * (1 - theta) * jx)

R11Sy_cav_old = Term(
    name='R11Sy_cav_old',
    description='flux divergence height source y Elrod-Adams',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda jy, theta: - 1 / ctx['h']() * ctx['dh_dy']() * (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta: - 1 / ctx['h']() * ctx['dh_dy']() * (1 - theta),
        lambda ctx: lambda jy, theta:  1 / ctx['h']() * ctx['dh_dy']() * jy,
    ],
    der_h=lambda ctx: lambda jy, theta: 1 / ctx['h']() ** 2 * ctx['dh_dy']() * (1 - theta) * jy)

# -----------------------------------------------------------------------------
# mass conservation: time-dependent terms
# -----------------------------------------------------------------------------

R1T_cav_old = Term(
    name='R1T_cav_old',
    description='local pressure change',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['rho', 'rho_prev', 'theta_prev'],
    fun=lambda ctx: lambda p, theta: - (ctx['rho']() * (1 - theta) - ctx['rho_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[
        lambda ctx: lambda p, theta: - ctx['drho_dp']() * (1 - theta) / ctx['dt'](),
        lambda ctx: lambda p, theta: ctx['rho']() / ctx['dt'](),
    ])

R1Th_cav_old = Term(
    name='R1Th_cav_old',
    description='squeeze source term',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['drho_dp', 'rho', 'h_prev', 'dh_dt'],
    fun=lambda ctx: lambda p, theta: - ctx['rho']() * (1 - theta) / ctx['h_prev']() * ctx['dh_dt'](),
    der_funs=[
        lambda ctx: lambda p, theta: - ctx['drho_dp']() * (1 - theta) / ctx['h_prev']() * ctx['dh_dt'](),
        lambda ctx: lambda p, theta: ctx['rho']() / ctx['h_prev']() * ctx['dh_dt'](),
    ],
    der_h=lambda ctx: lambda p, theta: - ctx['rho']() * (1 - theta) / ctx['h_prev']() / ctx['dt']())

# -----------------------------------------------------------------------------
# Fischer-Burmeister complementarity condition
# -----------------------------------------------------------------------------

R_cav_old = Term(
    name='R_cav_old',
    description='Fischer-Burmeister complementarity condition',
    res='fb',
    dep_vars=['p', 'theta'],
    dep_vals=[],
    fun=lambda ctx: lambda p, theta: (
        lambda a_nd: np.sqrt(a_nd**2 + theta**2) - a_nd - theta
    )((p - ctx['p_cav']()) / ctx['fb_p_ref']()),
    der_funs=[
        lambda ctx: lambda p, theta: (
            lambda a_nd: (a_nd / _fb_denom(a_nd, theta) - 1.0) / ctx['fb_p_ref']()
        )((p - ctx['p_cav']()) / ctx['fb_p_ref']()),
        lambda ctx: lambda p, theta: (
            lambda a_nd: theta / _fb_denom(a_nd, theta) - 1.0
        )((p - ctx['p_cav']()) / ctx['fb_p_ref']()),
    ])


# -----------------------------------------------------------------------------
# Artificial diffusion stabilization (stabilization.ad: true)
# -----------------------------------------------------------------------------

R1STx_old = Term(
    name='R1STx_old',
    description='theta diffusion stabilization in mass equation x',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['jx', 'jy', 'd_dx_theta'],
    fun=lambda ctx: lambda theta: -(ctx['ad_alpha']()
                                    * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30) * ctx['d_dx_theta']()),
    der_funs=[lambda ctx: lambda theta: -(ctx['ad_alpha']()
                                          * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30))],
    trial_deriv='x',
    test_deriv='x')

R1STy_old = Term(
    name='R1STy_old',
    description='theta diffusion stabilization in mass equation y',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['jx', 'jy', 'd_dy_theta'],
    fun=lambda ctx: lambda theta: -(ctx['ad_alpha']()
                                    * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30) * ctx['d_dy_theta']()),
    der_funs=[lambda ctx: lambda theta: -(ctx['ad_alpha']()
                                          * np.sqrt(ctx['jx']()**2 + ctx['jy']()**2 + 1e-30))],
    trial_deriv='y',
    test_deriv='y')


# -----------------------------------------------------------------------------
# Wall stress with theta-dependent effective density (replaces R24x/y when cavitation: true)
# -----------------------------------------------------------------------------

R24x_cav_old = Term(
    name='R24x_cav_old',
    description='wall stress x with theta-dependent effective density',
    res='momentum_x',
    dep_vars=['p', 'jx', 'theta'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'dtau_xz_dtheta', 'dtau_xz_dh', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_dtheta'](),
    ],
    der_h=lambda ctx: lambda *args: -1 / ctx['h']() ** 2 * ctx['tau_xz']() + 1 / ctx['h']() * ctx['dtau_xz_dh']())

R24y_cav_old = Term(
    name='R24y_cav_old',
    description='wall stress y with theta-dependent effective density',
    res='momentum_y',
    dep_vars=['p', 'jy', 'theta'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'dtau_yz_dtheta', 'dtau_yz_dh', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy'](),
        lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_dtheta'](),
    ],
    der_h=lambda ctx: lambda *args: -1 / ctx['h']() ** 2 * ctx['tau_yz']() + 1 / ctx['h']() * ctx['dtau_yz_dh']())


CAV_MASS_TERMS_old = [
    R11x_cav_old, R11y_cav_old, R11Sx_cav_old, R11Sy_cav_old,
    R1T_cav_old
]

THETA_TERMS_AD_old = [R1STx_old, R1STy_old]

THETA_TERMS_WALL_STRESS_old = [R24x_cav_old, R24y_cav_old]
