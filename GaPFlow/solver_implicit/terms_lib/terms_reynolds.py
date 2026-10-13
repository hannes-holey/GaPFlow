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

"""No-squeeze Reynolds equation terms."""

import numpy as np

from ..terms import Term

__all__ = [
    'Rey_R1T', 'Rey_R11x', 'Rey_R11x_drho', 'Rey_R11y', 'Rey_R11y_drho',
    'Rey_R11Sx', 'Rey_R11Sy', 'REYNOLDS_TERMS',
]

Rey_R1T = Term(
    name='Rey_R1T',
    description='Reynolds time derivative: -h/rho * drho/dt',
    res='mass',
    dep_vars=['p'],
    dep_vals=['drho_dp', 'rho', 'rho_prev', 'h', 'h_prev', 'dt'],
    fun=lambda ctx: lambda p: - ctx['h_prev']() * (ctx['rho']() - ctx['rho_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p: - ctx['h_prev']() * ctx['drho_dp']() / ctx['dt']()])

Rey_R11x = Term(
    name='Rey_R11x',
    description='Poiseuille diffusion x: rho * h^3/(12 mu) * dp/dx (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'rho', 'd_dx_p'],
    fun=lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dx_p'](),
    der_funs=[lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']())],
    trial_deriv='x',
    test_deriv='x')

Rey_R11x_drho = Term(
    name='Rey_R11x_drho',
    description='Poiseuille diffusion x, d/dp chain rule through rho(p) (Jacobian-only, fun=0)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'drho_dp', 'd_dx_p'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: ctx['drho_dp']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dx_p']()],
    test_deriv='x')

Rey_R11y = Term(
    name='Rey_R11y',
    description='Poiseuille diffusion y: rho * h^3/(12 mu) * dp/dy (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'rho', 'd_dy_p'],
    fun=lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dy_p'](),
    der_funs=[lambda ctx: lambda p: ctx['rho']() * ctx['h']() ** 3 / (12 * ctx['eta']())],
    trial_deriv='y',
    test_deriv='y')

Rey_R11y_drho = Term(
    name='Rey_R11y_drho',
    description='Poiseuille diffusion y, d/dp chain rule through rho(p) (Jacobian-only, fun=0)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'eta', 'drho_dp', 'd_dy_p'],
    fun=lambda ctx: lambda p: np.zeros_like(p),
    der_funs=[lambda ctx: lambda p: ctx['drho_dp']() * ctx['h']() ** 3 / (12 * ctx['eta']()) * ctx['d_dy_p']()],
    test_deriv='y')

Rey_R11Sx = Term(
    name='Rey_R11Sx',
    description='Wedge (Couette) term x: -d/dx(rho * h * u_s/2), u_s = U_bot + U_top (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'rho', 'drho_dp', 'U_bot', 'U_top'],
    fun=lambda ctx: lambda p: - 0.5 * (ctx['U_bot']() + ctx['U_top']()) * ctx['h']() * ctx['rho'](),
    der_funs=[lambda ctx: lambda p: - 0.5 * (ctx['U_bot']() + ctx['U_top']()) * ctx['h']() * ctx['drho_dp']()],
    test_deriv='x')

Rey_R11Sy = Term(
    name='Rey_R11Sy',
    description='Wedge (Couette) term y: -d/dy(rho * h * v_s/2), v_s = V_bot + V_top (IBP)',
    res='mass',
    dep_vars=['p'],
    dep_vals=['h', 'rho', 'drho_dp', 'V_bot', 'V_top'],
    fun=lambda ctx: lambda p: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['h']() * ctx['rho'](),
    der_funs=[lambda ctx: lambda p: - 0.5 * (ctx['V_bot']() + ctx['V_top']()) * ctx['h']() * ctx['drho_dp']()],
    test_deriv='y')

REYNOLDS_TERMS = [
    Rey_R1T, Rey_R11x, Rey_R11x_drho, Rey_R11y, Rey_R11y_drho,
    Rey_R11Sx, Rey_R11Sy,
]
