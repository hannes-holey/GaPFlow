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
    'Rey_R1T_supg', 'Rey_R11Sx_supg_a', 'Rey_R11Sx_supg_b', 'Rey_R11Sx_supg_c',
    'REYNOLDS_CAV_SUPG_TERMS',
]

# time derivative

Rey_R1T_supg = Term(
    name='Rey_R1T_supg',
    description='Full conservation time derivative',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['drho_dp', 'rho', 'rho_prev', 'h', 'h_prev', 'dt', 'theta_prev', 'tau_supg', 'U_m'],
    fun=lambda ctx: lambda p, theta: ctx['tau_supg']() * (ctx['rho']() * ctx['h']() * (1 - theta) - ctx['rho_prev']() * ctx['h_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['h']() * ctx['drho_dp']() * (1 - theta) / ctx['dt'](),
               lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['rho']() * ctx['h']() / ctx['dt']()],
    test_deriv='x')

# wedge x

Rey_R11Sx_supg_a = Term(
    name='Rey_R11Sx_supg_a',
    description='rho h d_dx_theta',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_bot', 'U_top', 'U_m', 'tau_supg', 'd_dx_theta'],
    fun=lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['rho']() * ctx['h']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['drho_dp']() * ctx['h']() * ctx['d_dx_theta'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['rho']() * ctx['h']()],
    trial_deriv=[None, 'x'],
    test_deriv='x'
)

Rey_R11Sx_supg_b = Term(
    name='Rey_R11Sx_supg_b',
    description='h (1-theta) d_dx_rho',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_bot', 'U_top', 'U_m', 'tau_supg', 'd_dx_p'],
    fun=lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * ctx['h']() * (1 - theta) * ctx['d_dx_p']() * ctx['drho_dp'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * (ctx['h']() * (1 - theta)) * ctx['drho_dp'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['h']() * ctx['d_dx_p']() * ctx['drho_dp']()],
    trial_deriv=['x', None],
    test_deriv='x'
)

Rey_R11Sx_supg_c = Term(
    name='Rey_R11Sx_supg_c',
    description='rho (1-theta) d_dx_h',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_bot', 'U_top', 'U_m', 'tau_supg', 'd_dx_h'],
    fun=lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * ctx['rho']() * (1 - theta) * ctx['d_dx_h'](),
    der_funs=[lambda ctx: lambda p, theta: ctx['tau_supg']() * ctx['U_m']() * ctx['drho_dp']() * (1 - theta) * ctx['d_dx_h'](),
              lambda ctx: lambda p, theta: - ctx['tau_supg']() * ctx['U_m']() * ctx['rho']() * ctx['d_dx_h']()],
    test_deriv='x'
)

REYNOLDS_CAV_SUPG_TERMS = [
    Rey_R1T_supg, Rey_R11Sx_supg_a, Rey_R11Sx_supg_b, Rey_R11Sx_supg_c
]
