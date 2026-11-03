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

"""Generalized Reynolds equation with Elrod-Adams cavitation.

    d/dt(rho_eff h (1-theta)) = div(epsilon grad p) - div(rho_eff h (1-theta) U_m)

epsilon(p, h) and rho_eff(p, h) come from `models/reynolds_functions.py`
(default: rho h^3/(12 eta) and rho). rho_eff is the height-averaged density, used
consistently in storage and Couette transport. Poiseuille and Couette terms in IBP form.
"""

import numpy as np

from ..terms import Term

__all__ = [
    'GRey_R1T', 'GRey_R11x', 'GRey_R11x_deps', 'GRey_R11y', 'GRey_R11y_deps',
    'GRey_R11Sx', 'GRey_R11Sy', 'GEN_REYNOLDS_CAV_TERMS',
]

GRey_R1T = Term(
    name='GRey_R1T',
    description='Mass storage: -(rho_eff h (1-theta) - rho_eff_prev h_prev (1-theta_prev)) / dt',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['drho_eff_dp', 'drho_eff_dh', 'rho_eff', 'rho_eff_prev', 'h', 'h_prev', 'dt', 'theta_prev'],
    fun=lambda ctx: lambda p, theta: - (ctx['rho_eff']() * ctx['h']() * (1 - theta) - ctx['rho_eff_prev']() * ctx['h_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p, theta: - ctx['h']() * ctx['drho_eff_dp']() * (1 - theta) / ctx['dt'](),
              lambda ctx: lambda p, theta: ctx['rho_eff']() * ctx['h']() / ctx['dt']()],
    der_h=lambda ctx: lambda p, theta: - (1 - theta) * (ctx['rho_eff']() + ctx['h']() * ctx['drho_eff_dh']()) / ctx['dt']())

GRey_R11x = Term(
    name='GRey_R11x',
    description='Poiseuille x: epsilon * dp/dx (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['epsilon', 'depsilon_dh', 'd_dx_p'],
    fun=lambda ctx: lambda p: ctx['epsilon']() * ctx['d_dx_p'](),
    der_funs=[lambda ctx: lambda p: ctx['epsilon']()],
    der_h=lambda ctx: lambda p: ctx['depsilon_dh']() * ctx['d_dx_p'](),
    trial_deriv='x',
    test_deriv='x')

GRey_R11x_deps = Term(
    name='GRey_R11x_deps',
    description='Poiseuille x, d/dp through epsilon(p) (Jacobian-only, fun=0)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['depsilon_dp', 'd_dx_p'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: ctx['depsilon_dp']() * ctx['d_dx_p']()],
    test_deriv='x')

GRey_R11y = Term(
    name='GRey_R11y',
    description='Poiseuille y: epsilon * dp/dy (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['epsilon', 'depsilon_dh', 'd_dy_p'],
    fun=lambda ctx: lambda p: ctx['epsilon']() * ctx['d_dy_p'](),
    der_funs=[lambda ctx: lambda p: ctx['epsilon']()],
    der_h=lambda ctx: lambda p: ctx['depsilon_dh']() * ctx['d_dy_p'](),
    trial_deriv='y',
    test_deriv='y')

GRey_R11y_deps = Term(
    name='GRey_R11y_deps',
    description='Poiseuille y, d/dp through epsilon(p) (Jacobian-only, fun=0)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['depsilon_dp', 'd_dy_p'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: ctx['depsilon_dp']() * ctx['d_dy_p']()],
    test_deriv='y')

GRey_R11Sx = Term(
    name='GRey_R11Sx',
    description='Couette x: -rho_eff h (1-theta) U_m, U_m = (U_bot + U_top)/2 (IBP)',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho_eff', 'drho_eff_dp', 'drho_eff_dh', 'U_bot', 'U_top'],
    fun=lambda ctx: lambda p, theta: - 0.5 * (ctx['U_bot']() + ctx['U_top']()) * ctx['rho_eff']() * ctx['h']() * (1 - theta),
    der_funs=[lambda ctx: lambda p, theta: - 0.5 * (ctx['U_bot']() + ctx['U_top']()) * ctx['drho_eff_dp']() * ctx['h']() * (1 - theta),
              lambda ctx: lambda p, theta: 0.5 * (ctx['U_bot']() + ctx['U_top']()) * ctx['rho_eff']() * ctx['h']()],
    test_deriv='x',
    der_h=lambda ctx: lambda p, theta: - 0.5 * (ctx['U_bot']() + ctx['U_top']()) * (1 - theta) * (ctx['rho_eff']() + ctx['h']() * ctx['drho_eff_dh']()))

GRey_R11Sy = Term(
    name='GRey_R11Sy',
    description='Couette y: -rho_eff h (1-theta) V_m, V_m = (V_bot + V_top)/2 (IBP)',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['h', 'rho_eff', 'drho_eff_dp', 'drho_eff_dh', 'V_bot', 'V_top'],
    fun=lambda ctx: lambda p, theta: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['rho_eff']() * ctx['h']() * (1 - theta),
    der_funs=[lambda ctx: lambda p, theta: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['drho_eff_dp']() * ctx['h']() * (1 - theta),
              lambda ctx: lambda p, theta: 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['rho_eff']() * ctx['h']()],
    test_deriv='y',
    der_h=lambda ctx: lambda p, theta: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * (1 - theta) * (ctx['rho_eff']() + ctx['h']() * ctx['drho_eff_dh']()))

GEN_REYNOLDS_CAV_TERMS = [
    GRey_R1T, GRey_R11x, GRey_R11x_deps, GRey_R11y, GRey_R11y_deps,
    GRey_R11Sx, GRey_R11Sy,
]
