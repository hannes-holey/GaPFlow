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
    'R11x_supg_x1_old', 'R11x_supg_x2_old', 'R11x_supg_y1_old', 'R11x_supg_y2_old',
    'R11y_supg_y1_old', 'R11y_supg_y2_old', 'R11y_supg_x1_old', 'R11y_supg_x2_old',
    'R11Sx_supg_x_old', 'R11Sy_supg_x_old', 'R11Sx_supg_y_old', 'R11Sy_supg_y_old',
    'R1T_cav_supg_x_old', 'R1T_cav_supg_y_old', 'R1Th_cav_supg_x_old', 'R1Th_cav_supg_y_old',
    'SUPG_TERMS_old', 'SUPG_SQUEEZE_TERMS_old',
]

# -----------------------------------------------------------------------------
# R11x_cav SUPG companions (split into two terms each -- see module docstring)
#
# Naming: <base>_supg_<dir><n>, where <dir> is the test-function integration
# direction ('x' or 'y') and <n> in {1, 2} distinguishes the two strong-form
# pieces of the same companion (product-rule split of d/d<dir>((1-theta)*j)).
# -----------------------------------------------------------------------------

R11x_supg_x1_old = Term(
    name='R11x_supg_x1_old',
    description='SUPG x: flux divergence x Elrod-Adams, strong-form piece (1-theta)*d_dx_jx',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['f_x', 'd_dx_jx'],
    fun=lambda ctx: lambda jx, theta: ctx['f_x']() * (1 - theta) * ctx['d_dx_jx'](),
    der_funs=[
        lambda ctx: lambda jx, theta: ctx['f_x']() * (1 - theta),
        lambda ctx: lambda jx, theta: -ctx['f_x']() * ctx['d_dx_jx'](),
    ],
    trial_deriv=['x', None],
    test_deriv='x')

R11x_supg_x2_old = Term(
    name='R11x_supg_x2_old',
    description='SUPG x: flux divergence x Elrod-Adams, strong-form piece -jx*d_dx_theta',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['f_x', 'd_dx_theta'],
    fun=lambda ctx: lambda jx, theta: -ctx['f_x']()* jx * ctx['d_dx_theta'](),
    der_funs=[
        lambda ctx: lambda jx, theta: -ctx['f_x']() * ctx['d_dx_theta'](),
        lambda ctx: lambda jx, theta: -ctx['f_x']() * jx,
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R11x_supg_y1_old = Term(
    name='R11x_supg_y1_old',
    description='SUPG y companion of R11x_cav, strong-form piece (1-theta)*d_dy_jx',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['f_y', 'd_dy_jx'],
    fun=lambda ctx: lambda jx, theta: ctx['f_y']() * (1 - theta) * ctx['d_dy_jx'](),
    der_funs=[
        lambda ctx: lambda jx, theta: ctx['f_y']() * (1 - theta),
        lambda ctx: lambda jx, theta: -ctx['f_y']() * ctx['d_dy_jx'](),
    ],
    trial_deriv=['y', None],
    test_deriv='y')

R11x_supg_y2_old = Term(
    name='R11x_supg_y2_old',
    description='SUPG y companion of R11x_cav, strong-form piece -jx*d_dy_theta',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['f_y', 'd_dy_theta'],
    fun=lambda ctx: lambda jx, theta: -ctx['f_y']() * jx * ctx['d_dy_theta'](),
    der_funs=[
        lambda ctx: lambda jx, theta: -ctx['f_y']() * ctx['d_dy_theta'](),
        lambda ctx: lambda jx, theta: -ctx['f_y']() * jx,
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

# -----------------------------------------------------------------------------
# R11y_cav SUPG companions (mirror of R11x_cav with x<->y, jx<->jy)
# -----------------------------------------------------------------------------

R11y_supg_y1_old = Term(
    name='R11y_supg_y1_old',
    description='SUPG y: flux divergence y Elrod-Adams, strong-form piece (1-theta)*d_dy_jy',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['f_y', 'd_dy_jy'],
    fun=lambda ctx: lambda jy, theta: ctx['f_y']() *(1 - theta) * ctx['d_dy_jy'](),
    der_funs=[
        lambda ctx: lambda jy, theta: ctx['f_y']() * (1 - theta),
        lambda ctx: lambda jy, theta: -ctx['f_y']() * ctx['d_dy_jy'](),
    ],
    trial_deriv=['y', None],
    test_deriv='y')

R11y_supg_y2_old = Term(
    name='R11y_supg_y2_old',
    description='SUPG y: flux divergence y Elrod-Adams, strong-form piece -jy*d_dy_theta',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['f_y', 'd_dy_theta'],
    fun=lambda ctx: lambda jy, theta: -ctx['f_y']() * jy * ctx['d_dy_theta'](),
    der_funs=[
        lambda ctx: lambda jy, theta: -ctx['f_y']() * ctx['d_dy_theta'](),
        lambda ctx: lambda jy, theta: -ctx['f_y']() * jy,
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R11y_supg_x1_old = Term(
    name='R11y_supg_x1_old',
    description='SUPG x companion of R11y_cav, strong-form piece (1-theta)*d_dx_jy',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['f_x', 'd_dx_jy'],
    fun=lambda ctx: lambda jy, theta: ctx['f_x']() * (1 - theta) * ctx['d_dx_jy'](),
    der_funs=[
        lambda ctx: lambda jy, theta: ctx['f_x']() * (1 - theta),
        lambda ctx: lambda jy, theta: -ctx['f_x']() * ctx['d_dx_jy'](),
    ],
    trial_deriv=['x', None],
    test_deriv='x')

R11y_supg_x2_old = Term(
    name='R11y_supg_x2_old',
    description='SUPG x companion of R11y_cav, strong-form piece -jy*d_dx_theta',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['f_x', 'd_dx_theta'],
    fun=lambda ctx: lambda jy, theta: -ctx['f_x']() * jy * ctx['d_dx_theta'](),
    der_funs=[
        lambda ctx: lambda jy, theta: -ctx['f_x']() * ctx['d_dx_theta'](),
        lambda ctx: lambda jy, theta: -ctx['f_x']() * jy,
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

# -----------------------------------------------------------------------------
# Direct SUPG duplicates (base terms already have test_deriv=None, so each
# companion is the same integrand times f_x or f_y, with test_deriv='x'/'y')
# -----------------------------------------------------------------------------

R11Sx_supg_x_old = Term(
    name='R11Sx_supg_x_old',
    description='SUPG x: flux divergence height source x Elrod-Adams',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['f_x', 'h', 'dh_dx'],
    fun=lambda ctx: lambda jx, theta:  ctx['f_x']() / ctx['h']() * ctx['dh_dx']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta:  ctx['f_x']() / ctx['h']() * ctx['dh_dx']() * (1 - theta),
        lambda ctx: lambda jx, theta: -ctx['f_x']() / ctx['h']() * ctx['dh_dx']() * jx,
    ],
    test_deriv='x')

R11Sy_supg_x_old = Term(
    name='R11Sy_supg_x_old',
    description='SUPG x: flux divergence height source y Elrod-Adams',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['f_x', 'h', 'dh_dy'],
    fun=lambda ctx: lambda jy, theta:  ctx['f_x']() / ctx['h']() * ctx['dh_dy']() * (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta:  ctx['f_x']() / ctx['h']() * ctx['dh_dy']() * (1 - theta),
        lambda ctx: lambda jy, theta: -ctx['f_x']() / ctx['h']() * ctx['dh_dy']() * jy,
    ],
    test_deriv='x')

R11Sx_supg_y_old = Term(
    name='R11Sx_supg_y_old',
    description='SUPG y: flux divergence height source x Elrod-Adams',
    res='mass',
    dep_vars=['jx', 'theta'],
    dep_vals=['f_y', 'h', 'dh_dx'],
    fun=lambda ctx: lambda jx, theta:  ctx['f_y']() / ctx['h']() * ctx['dh_dx']() * (1 - theta) * jx,
    der_funs=[
        lambda ctx: lambda jx, theta:  ctx['f_y']() / ctx['h']() * ctx['dh_dx']() * (1 - theta),
        lambda ctx: lambda jx, theta: -ctx['f_y']() / ctx['h']() * ctx['dh_dx']() * jx,
    ],
    test_deriv='y')

R11Sy_supg_y_old = Term(
    name='R11Sy_supg_y_old',
    description='SUPG y: flux divergence height source y Elrod-Adams',
    res='mass',
    dep_vars=['jy', 'theta'],
    dep_vals=['f_y', 'h', 'dh_dy'],
    fun=lambda ctx: lambda jy, theta:  ctx['f_y']() / ctx['h']() * ctx['dh_dy']() * (1 - theta) * jy,
    der_funs=[
        lambda ctx: lambda jy, theta:  ctx['f_y']() / ctx['h']() * ctx['dh_dy']() * (1 - theta),
        lambda ctx: lambda jy, theta: -ctx['f_y']() / ctx['h']() * ctx['dh_dy']() * jy,
    ],
    test_deriv='y')

R1T_cav_supg_x_old = Term(
    name='R1T_cav_supg_x_old',
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

R1T_cav_supg_y_old = Term(
    name='R1T_cav_supg_y_old',
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

R1Th_cav_supg_x_old = Term(
    name='R1Th_cav_supg_x_old',
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

R1Th_cav_supg_y_old = Term(
    name='R1Th_cav_supg_y_old',
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


SUPG_TERMS_old = [
    R11x_supg_x1_old, R11x_supg_x2_old, R11x_supg_y1_old, R11x_supg_y2_old,
    R11y_supg_y1_old, R11y_supg_y2_old, R11y_supg_x1_old, R11y_supg_x2_old,
    R11Sx_supg_x_old, R11Sy_supg_x_old, R11Sx_supg_y_old, R11Sy_supg_y_old,
    R1T_cav_supg_x_old, R1T_cav_supg_y_old,
]

SUPG_SQUEEZE_TERMS_old = [R1Th_cav_supg_x_old, R1Th_cav_supg_y_old]
