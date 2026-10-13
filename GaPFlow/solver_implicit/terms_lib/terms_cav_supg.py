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

from ..terms import Term

__all__ = [
    'R11x_supg_x1', 'R11x_supg_y1',
    'R11y_supg_y1', 'R11y_supg_x1',
    'R11Sx_supg_x', 'R11Sy_supg_x', 'R11Sx_supg_y', 'R11Sy_supg_y',
    'R1T_cav_supg_x', 'R1T_cav_supg_y', 'R1Th_cav_supg_x', 'R1Th_cav_supg_y',
    'SUPG_TERMS', 'SUPG_SQUEEZE_TERMS',
]

# -----------------------------------------------------------------------------
# R11x_cav SUPG companions.
#
# Naming: <base>_supg_<dir>, where <dir> is the test-function integration
# direction ('x' or 'y'). R11x_cav/R11y_cav carry no (1-theta) flux factor,
# so their strong form is plain d/d<dir>(j) -- a single SUPG term each,
# no product-rule split against theta needed.
# -----------------------------------------------------------------------------

R11x_supg_x1 = Term(
    name='R11x_supg_x1',
    description='SUPG x: flux divergence x, strong-form piece d_dx_jx',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['f_x', 'd_dx_jx'],
    fun=lambda ctx: lambda jx: ctx['f_x']() * ctx['d_dx_jx'](),
    der_funs=[
        lambda ctx: lambda jx: ctx['f_x'](),
    ],
    trial_deriv='x',
    test_deriv='x')

R11x_supg_y1 = Term(
    name='R11x_supg_y1',
    description='SUPG y companion of R11x_cav, strong-form piece d_dy_jx',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['f_y', 'd_dy_jx'],
    fun=lambda ctx: lambda jx: ctx['f_y']() * ctx['d_dy_jx'](),
    der_funs=[
        lambda ctx: lambda jx: ctx['f_y'](),
    ],
    trial_deriv='x',
    test_deriv='y')

# -----------------------------------------------------------------------------
# R11y_cav SUPG companions (mirror of R11x_cav with x<->y, jx<->jy)
# -----------------------------------------------------------------------------

R11y_supg_y1 = Term(
    name='R11y_supg_y1',
    description='SUPG y: flux divergence y, strong-form piece d_dy_jy',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['f_y', 'd_dy_jy'],
    fun=lambda ctx: lambda jy: ctx['f_y']() * ctx['d_dy_jy'](),
    der_funs=[
        lambda ctx: lambda jy: ctx['f_y'](),
    ],
    trial_deriv='y',
    test_deriv='x')

R11y_supg_x1 = Term(
    name='R11y_supg_x1',
    description='SUPG x companion of R11y_cav, strong-form piece d_dx_jy',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['f_x', 'd_dx_jy'],
    fun=lambda ctx: lambda jy: ctx['f_x']() * ctx['d_dx_jy'](),
    der_funs=[
        lambda ctx: lambda jy: ctx['f_x'](),
    ],
    trial_deriv='y',
    test_deriv='y')

# -----------------------------------------------------------------------------
# Direct SUPG duplicates (base terms already have test_deriv=None, so each
# companion is the same integrand times f_x or f_y, with test_deriv='x'/'y')
# -----------------------------------------------------------------------------

R11Sx_supg_x = Term(
    name='R11Sx_supg_x',
    description='SUPG x: flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['f_x', 'h', 'dh_dx'],
    fun=lambda ctx: lambda jx:  ctx['f_x']() / ctx['h']() * ctx['dh_dx']() * jx,
    der_funs=[
        lambda ctx: lambda jx:  ctx['f_x']() / ctx['h']() * ctx['dh_dx'](),
    ],
    test_deriv='x')

R11Sy_supg_x = Term(
    name='R11Sy_supg_x',
    description='SUPG x: flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['f_x', 'h', 'dh_dy'],
    fun=lambda ctx: lambda jy:  ctx['f_x']() / ctx['h']() * ctx['dh_dy']() * jy,
    der_funs=[
        lambda ctx: lambda jy:  ctx['f_x']() / ctx['h']() * ctx['dh_dy'](),
    ],
    test_deriv='x')

R11Sx_supg_y = Term(
    name='R11Sx_supg_y',
    description='SUPG y: flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['f_y', 'h', 'dh_dx'],
    fun=lambda ctx: lambda jx:  ctx['f_y']() / ctx['h']() * ctx['dh_dx']() * jx,
    der_funs=[
        lambda ctx: lambda jx:  ctx['f_y']() / ctx['h']() * ctx['dh_dx'](),
    ],
    test_deriv='y')

R11Sy_supg_y = Term(
    name='R11Sy_supg_y',
    description='SUPG y: flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['f_y', 'h', 'dh_dy'],
    fun=lambda ctx: lambda jy:  ctx['f_y']() / ctx['h']() * ctx['dh_dy']() * jy,
    der_funs=[
        lambda ctx: lambda jy:  ctx['f_y']() / ctx['h']() * ctx['dh_dy'](),
    ],
    test_deriv='y')

R1T_cav_supg_x = Term(
    name='R1T_cav_supg_x',
    description='SUPG x: local pressure change',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['f_x', 'drho_dp', 'rho', 'rho_prev', 'theta_prev'],
    fun=lambda ctx: lambda p, theta: ctx['f_x']() * (ctx['rho']() * (1 - theta) - ctx['rho_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[
        lambda ctx: lambda p, theta: ctx['f_x']() * ctx['drho_dp']() * (1 - theta) / ctx['dt'](),
        lambda ctx: lambda p, theta: - ctx['f_x']() * ctx['rho']() / ctx['dt'](),
    ],
    test_deriv='x')

R1T_cav_supg_y = Term(
    name='R1T_cav_supg_y',
    description='SUPG y: local pressure change',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['f_y', 'drho_dp', 'rho', 'rho_prev', 'theta_prev'],
    fun=lambda ctx: lambda p, theta: ctx['f_y']() * (ctx['rho']() * (1 - theta) - ctx['rho_prev']() * (1 - ctx['theta_prev']())) / ctx['dt'](),
    der_funs=[
        lambda ctx: lambda p, theta: ctx['f_y']() * ctx['drho_dp']() * (1 - theta) / ctx['dt'](),
        lambda ctx: lambda p, theta: - ctx['f_y']() * ctx['rho']() / ctx['dt'](),
    ],
    test_deriv='y')

R1Th_cav_supg_x = Term(
    name='R1Th_cav_supg_x',
    description='SUPG x: squeeze source term (elastic der_h not propagated to the SUPG companion)',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['f_x', 'drho_dp', 'rho', 'h_prev', 'dh_dt'],
    fun=lambda ctx: lambda p, theta: ctx['f_x']() * ctx['rho']() * (1 - theta) / ctx['h_prev']() * ctx['dh_dt'](),
    der_funs=[
        lambda ctx: lambda p, theta: ctx['f_x']() * ctx['drho_dp']() * (1 - theta) / ctx['h_prev']() * ctx['dh_dt'](),
        lambda ctx: lambda p, theta: - ctx['f_x']() * ctx['rho']() / ctx['h_prev']() * ctx['dh_dt'](),
    ],
    test_deriv='x')

R1Th_cav_supg_y = Term(
    name='R1Th_cav_supg_y',
    description='SUPG y: squeeze source term (elastic der_h not propagated to the SUPG companion)',
    res='mass',
    dep_vars=['p', 'theta'],
    dep_vals=['f_y', 'drho_dp', 'rho', 'h_prev', 'dh_dt'],
    fun=lambda ctx: lambda p, theta: ctx['f_y']() * ctx['rho']() * (1 - theta) / ctx['h_prev']() * ctx['dh_dt'](),
    der_funs=[
        lambda ctx: lambda p, theta: ctx['f_y']() * ctx['drho_dp']() * (1 - theta) / ctx['h_prev']() * ctx['dh_dt'](),
        lambda ctx: lambda p, theta: - ctx['f_y']() * ctx['rho']() / ctx['h_prev']() * ctx['dh_dt'](),
    ],
    test_deriv='y')


SUPG_TERMS = [
    R11x_supg_x1, R11x_supg_y1,
    R11y_supg_y1, R11y_supg_x1,
    R11Sx_supg_x, R11Sy_supg_x, R11Sx_supg_y, R11Sy_supg_y,
    R1T_cav_supg_x, R1T_cav_supg_y,
]

SUPG_SQUEEZE_TERMS = [R1Th_cav_supg_x, R1Th_cav_supg_y]
