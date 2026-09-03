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
    'Rey_R1T_cav', 'Rey_R11x_cav', 'Rey_R11x_drho_cav', 'Rey_R11y_cav', 'Rey_R11y_drho_cav',
    'Rey_R11Sx_cav_a', 'Rey_R11Sx_cav_b', 'Rey_R11Sx_cav_c', 'Rey_R11Sy_cav',
    'REYNOLDS_CAV_TERMS',
]

Rey_R1T_cav = Term(
    name='Rey_R1T_cav',
    description='Full conservation time derivative',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['drho_dp', 'rho', 'rho_prev', 'h', 'h_prev', 'dt', 'theta_prev'],
    fun=lambda ctx: lambda p, theta: - (ctx['rho']() * ctx['h']() * (1 - theta) - ctx['rho_prev']() * ctx['h_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p, theta: - ctx['h']() * ctx['drho_dp']() * (1 - theta) / ctx['dt'](),
               lambda ctx: lambda p, theta: ctx['rho']() * ctx['h']() / ctx['dt']()])

Rey_R11x_cav = Term(
    name='Rey_R11x_cav',
    description='Poiseuille diffusion x: rho * h^3/(12 mu) * dp/dx (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'rho', 'd_dx_p'],
    fun=lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dx_p'](),
    der_funs=[lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']())],
    trial_deriv='x',
    test_deriv='x')

Rey_R11x_drho_cav = Term(
    name='Rey_R11x_drho_cav',
    description='Poiseuille diffusion x, d/dp chain rule through rho(p) (Jacobian-only, fun=0)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'drho_dp', 'd_dx_p'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: ctx['drho_dp']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dx_p']()],
    test_deriv='x')

Rey_R11y_cav = Term(
    name='Rey_R11y_cav',
    description='Poiseuille diffusion y: rho * h^3/(12 mu) * dp/dy (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'rho', 'd_dy_p'],
    fun=lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dy_p'](),
    der_funs=[lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']())],
    trial_deriv='y',
    test_deriv='y')

Rey_R11y_drho_cav = Term(
    name='Rey_R11y_drho_cav',
    description='Poiseuille diffusion y, d/dp chain rule through rho(p) (Jacobian-only, fun=0)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'drho_dp', 'd_dy_p'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: ctx['drho_dp']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dy_p']()],
    test_deriv='y')

Rey_R11Sx_cav_a = Term(
    name='Rey_R11Sx_cav_a',
    description='Wedge (Couette) term x, strong form part a: U_m * rho * h * d_dx_theta',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_m', 'd_dx_theta'],
    fun=lambda ctx: lambda p, theta: ctx['U_m']() * ctx['rho']() * ctx['h']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['U_m']() * ctx['drho_dp']() * ctx['h']() * ctx['d_dx_theta'](),
              lambda ctx: lambda p, theta: ctx['U_m']() * ctx['rho']() * ctx['h']()],
    trial_deriv=[None, 'x'])

Rey_R11Sx_cav_b = Term(
    name='Rey_R11Sx_cav_b',
    description='Wedge (Couette) term x, strong form part b: -U_m * h * (1-theta) * d_dx_rho',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_m', 'd_dx_p'],
    fun=lambda ctx: lambda p, theta: - ctx['U_m']() * ctx['h']() * (1 - theta) * ctx['d_dx_p']() * ctx['drho_dp'](),
    der_funs=[lambda ctx: lambda p, theta: - ctx['U_m']() * (ctx['h']() * (1 - theta)) * ctx['drho_dp'](),
              lambda ctx: lambda p, theta: ctx['U_m']() * ctx['h']() * ctx['d_dx_p']() * ctx['drho_dp']()],
    trial_deriv=['x', None])

Rey_R11Sx_cav_c = Term(
    name='Rey_R11Sx_cav_c',
    description='Wedge (Couette) term x, strong form part c: -U_m * rho * (1-theta) * d_dx_h',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_m', 'd_dx_h'],
    fun=lambda ctx: lambda p, theta: - ctx['U_m']() * ctx['rho']() * (1 - theta) * ctx['d_dx_h'](),
    der_funs=[lambda ctx: lambda p, theta: - ctx['U_m']() * ctx['drho_dp']() * (1 - theta) * ctx['d_dx_h'](),
              lambda ctx: lambda p, theta: ctx['U_m']() * ctx['rho']() * ctx['d_dx_h']()])

Rey_R11Sy_cav = Term(
    name='Rey_R11Sy_cav',
    description='Wedge (Couette) term y: -d/dy(rho * h * v_s/2), v_s = V_bot + V_top (IBP)',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'V_bot', 'V_top'],
    fun=lambda ctx: lambda p, theta: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['h']() * ctx['rho']() * (1 - theta),
    der_funs=[lambda ctx: lambda p, theta: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['h']() * ctx['drho_dp']() * (1 - theta),
              lambda ctx: lambda p, theta: 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['h']() * ctx['rho']()],
    test_deriv='y')

REYNOLDS_CAV_TERMS = [
    Rey_R1T_cav, Rey_R11x_cav, Rey_R11x_drho_cav, Rey_R11y_cav, Rey_R11y_drho_cav,
    Rey_R11Sx_cav_a, Rey_R11Sx_cav_b, Rey_R11Sx_cav_c, Rey_R11Sy_cav,
]
