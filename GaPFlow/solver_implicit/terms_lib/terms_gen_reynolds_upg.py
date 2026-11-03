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

"""Upwind Petrov-Galerkin (UPG) companions of the generalized Reynolds terms (x only).

Test function phi + tau_supg * dphi/dx, tau_supg = alpha * dx/2 * sign(U_m). Each term
holds -tau_supg * r_part of the strong residual (test_deriv='x' assembles -int f dphi/dx)

    r = -d/dt(m) + d/dx(epsilon dp/dx) - U_m d/dx(m),    m = rho_eff h (1-theta)

with d/dx(epsilon) = eps_p dp/dx + eps_h dh/dx (d2p/dx2 = 0 on Q1) and
d/dx(rho_eff) = rho_eff_p dp/dx + rho_eff_h dh/dx. Second derivatives of epsilon and
rho_eff are dropped in the Jacobian.
"""

from ..terms import Term

__all__ = [
    'GRey_R1T_upg', 'GRey_R11Sx_upg_a', 'GRey_R11Sx_upg_b', 'GRey_R11Sx_upg_c',
    'GRey_R11x_upg', 'GEN_REYNOLDS_UPG_TERMS',
]

GRey_R1T_upg = Term(
    name='GRey_R1T_upg',
    description='UPG storage: (rho_eff h (1-theta) - rho_eff_prev h_prev (1-theta_prev)) / dt',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['drho_eff_dp', 'rho_eff', 'rho_eff_prev', 'h', 'h_prev', 'dt', 'theta_prev', 'tau_supg'],
    fun=lambda ctx: lambda p, theta: ctx['tau_supg']() * (ctx['rho_eff']() * ctx['h']() * (1 - theta) - ctx['rho_eff_prev']() * ctx['h_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['h']() * ctx['drho_eff_dp']() * (1 - theta) / ctx['dt'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['rho_eff']() * ctx['h']() / ctx['dt']()],
    test_deriv='x')

GRey_R11Sx_upg_a = Term(
    name='GRey_R11Sx_upg_a',
    description='UPG Couette x, theta gradient: U_m rho_eff h d_dx_theta',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho_eff', 'drho_eff_dp', 'U_m', 'tau_supg', 'd_dx_theta'],
    fun=lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['rho_eff']() * ctx['h']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['drho_eff_dp']() * ctx['h']() * ctx['d_dx_theta'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['rho_eff']() * ctx['h']()],
    trial_deriv=[None, 'x'],
    test_deriv='x')

GRey_R11Sx_upg_b = Term(
    name='GRey_R11Sx_upg_b',
    description='UPG Couette x, rho_eff(p) gradient: -U_m h (1-theta) rho_eff_p d_dx_p',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'drho_eff_dp', 'U_m', 'tau_supg', 'd_dx_p'],
    fun=lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * ctx['h']() * (1 - theta) * ctx['drho_eff_dp']() * ctx['d_dx_p'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * ctx['h']() * (1 - theta) * ctx['drho_eff_dp'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['h']() * ctx['drho_eff_dp']() * ctx['d_dx_p']()],
    trial_deriv=['x', None],
    test_deriv='x')

GRey_R11Sx_upg_c = Term(
    name='GRey_R11Sx_upg_c',
    description='UPG Couette x, h gradient: -U_m (1-theta) (rho_eff + h rho_eff_h) dh_dx',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho_eff', 'drho_eff_dp', 'drho_eff_dh', 'U_m', 'tau_supg', 'dh_dx'],
    fun=lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * (1 - theta) * (ctx['rho_eff']() + ctx['h']() * ctx['drho_eff_dh']()) * ctx['dh_dx'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * (1 - theta) * ctx['drho_eff_dp']() * ctx['dh_dx'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * (ctx['rho_eff']() + ctx['h']() * ctx['drho_eff_dh']()) * ctx['dh_dx']()],
    test_deriv='x')

GRey_R11x_upg = Term(
    name='GRey_R11x_upg',
    description='UPG Poiseuille x, coefficient gradient: (eps_p d_dx_p + eps_h dh_dx) d_dx_p',
    res='mass',
    dep_vars=['p'],
    dep_vals=['depsilon_dp', 'depsilon_dh', 'dh_dx', 'd_dx_p', 'tau_supg'],
    fun=lambda ctx: lambda p: - ctx['tau_supg']() * (ctx['depsilon_dp']() * ctx['d_dx_p']() + ctx['depsilon_dh']() * ctx['dh_dx']()) * ctx['d_dx_p'](),
    der_funs=[lambda ctx: lambda p: - ctx['tau_supg']() * (2 * ctx['depsilon_dp']() * ctx['d_dx_p']() + ctx['depsilon_dh']() * ctx['dh_dx']())],
    trial_deriv='x',
    test_deriv='x')

GEN_REYNOLDS_UPG_TERMS = [
    GRey_R1T_upg, GRey_R11Sx_upg_a, GRey_R11Sx_upg_b, GRey_R11Sx_upg_c,
    GRey_R11x_upg,
]
