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
    'R11x', 'R11y', 'R11Sx', 'R11Sy', 'R1T', 'R1Th',
    'R21x', 'R21y', 'R22xx', 'R22Sxx', 'R22yx', 'R22Syx', 'R22xy', 'R22Sxy', 'R22yy', 'R22Syy',
    'R23xy', 'R23yx', 'R23xx', 'R23yy', 'R24x', 'R24y', 'R25x', 'R25y',
    'R2Tx', 'R2Ty', 'R2Thx', 'R2Thy',
]

# -----------------------------------------------------------------------------
# Mass equation terms (R1*)
# -----------------------------------------------------------------------------

R11x = Term(
    name='R11x',
    description='flux divergence x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['d_dx_jx'],
    fun=lambda ctx: lambda jx: -ctx['d_dx_jx'](),
    der_funs=[lambda ctx: lambda jx: np.full_like(jx, -1.0)],
    trial_deriv='x')

R11y = Term(
    name='R11y',
    description='flux divergence y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['d_dy_jy'],
    fun=lambda ctx: lambda jy: -ctx['d_dy_jy'](),
    der_funs=[lambda ctx: lambda jy: np.full_like(jy, -1.0)],
    trial_deriv='y')

R11Sx = Term(
    name='R11Sx',
    description='flux divergence height source x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda jx: -(1 / ctx['h']()) * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -(1 / ctx['h']()) * ctx['dh_dx']()],
    der_h=lambda ctx: lambda jx: (1 / ctx['h']() ** 2) * ctx['dh_dx']() * jx,
    der_h_dx=lambda ctx: lambda jx: -(1 / ctx['h']()) * jx)

R11Sy = Term(
    name='R11Sy',
    description='flux divergence height source y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda jy: -(1 / ctx['h']()) * ctx['dh_dy']() * jy,
    der_funs=[lambda ctx: lambda jy: -(1 / ctx['h']()) * ctx['dh_dy']()],
    der_h=lambda ctx: lambda jy: (1 / ctx['h']() ** 2) * ctx['dh_dy']() * jy,
    der_h_dy=lambda ctx: lambda jy: -(1 / ctx['h']()) * jy)

R1T = Term(
    name='R1T',
    description='time derivative — density space, no h factor, pairs with R1Th',
    res='mass',
    dep_vars=['p'],
    dep_vals=['drho_dp', 'rho', 'rho_prev'],
    fun=lambda ctx: lambda p: - (ctx['rho']() - ctx['rho_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda p: - ctx['drho_dp']() / ctx['dt']()])

R1Th = Term(
    name='R1Th',
    description='squeeze source — dp_drho * rho / h_prev * dh_dt, pairs with plain R1T',
    res='mass',
    dep_vars=['p'],
    dep_vals=['dp_drho', 'd2p_drho2', 'drho_dp', 'rho', 'h_prev', 'dh_dt'],
    fun=lambda ctx: lambda p: - ctx['rho']() / ctx['h_prev']() * ctx['dh_dt'](),
    der_funs=[lambda ctx: lambda p: - ctx['drho_dp']() / ctx['h_prev']() * ctx['dh_dt']()],
    der_h=lambda ctx: lambda p: - ctx['rho']() / ctx['h_prev']() / ctx['dt']())

# -----------------------------------------------------------------------------
# Momentum equation terms (R2*)
# -----------------------------------------------------------------------------

R21x = Term(
    name='R21x',
    description='pressure gradient x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    test_deriv='x')

R21y = Term(
    name='R21y',
    description='pressure gradient y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=[],
    fun=lambda ctx: lambda p: -p,
    der_funs=[lambda ctx: lambda p: np.full_like(p, -1.0)],
    test_deriv='y')

R22xx = Term(
    name='R22xx',
    description='convective momentum flux jx*jx in x (IBP)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx: -(jx * jx) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: (jx * jx) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: -2 * jx / ctx['rho']()
    ],
    test_deriv='x')

R22Sxx = Term(
    name='R22Sxx',
    description='convective momentum flux jx*jx height source',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'dh_dx', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: -1 / ctx['h']() * ctx['dh_dx']() * 2 * jx / ctx['rho']()
    ],
    der_h=lambda ctx: lambda p, jx: 1 / ctx['h']() ** 2 * ctx['dh_dx']() * (jx * jx) / ctx['rho']())

R22yx = Term(
    name='R22yx',
    description='convective momentum flux jx*jy in y (for momentum_x, IBP)',
    res='momentum_x',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -(jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -jx / ctx['rho']()
    ],
    test_deriv='y')

R22Syx = Term(
    name='R22Syx',
    description='convective momentum flux jx*jy height source (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['h', 'dh_dy', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jx / ctx['rho']()
    ],
    der_h=lambda ctx: lambda p, jx, jy: 1 / ctx['h']() ** 2 * ctx['dh_dy']() * (jx * jy) / ctx['rho']())

R22xy = Term(
    name='R22xy',
    description='convective momentum flux jx*jy in x (for momentum_y, IBP)',
    res='momentum_y',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -(jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -jx / ctx['rho']()
    ],
    test_deriv='x')

R22Sxy = Term(
    name='R22Sxy',
    description='convective momentum flux jx*jy height source (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jx', 'jy'],
    dep_vals=['h', 'dh_dx', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx, jy: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jy / ctx['rho'](),
        lambda ctx: lambda p, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jx / ctx['rho']()
    ],
    der_h=lambda ctx: lambda p, jx, jy: 1 / ctx['h']() ** 2 * ctx['dh_dx']() * (jx * jy) / ctx['rho']())

R22yy = Term(
    name='R22yy',
    description='convective momentum flux jy*jy in y (IBP)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jy: -(jy * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: (jy * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: -2 * jy / ctx['rho']()
    ],
    test_deriv='y')

R22Syy = Term(
    name='R22Syy',
    description='convective momentum flux jy*jy height source',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['h', 'dh_dy', 'rho', 'drho_dp'],
    fun=lambda ctx: lambda p, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / ctx['rho']() ** 2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: -1 / ctx['h']() * ctx['dh_dy']() * 2 * jy / ctx['rho']()
    ],
    der_h=lambda ctx: lambda p, jy: 1 / ctx['h']() ** 2 * ctx['dh_dy']() * (jy * jy) / ctx['rho']())

R23xy = Term(
    name='R23xy',
    description='shear viscous stress tau_xy in y (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dy_jx'],
    fun=lambda ctx: lambda p, jx: ctx['eta']() * ctx['d_dy_jx']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: -ctx['eta']() * jx / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R23yx = Term(
    name='R23yx',
    description='shear viscous stress tau_xy in x (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dx_jy'],
    fun=lambda ctx: lambda p, jy: ctx['eta']() * ctx['d_dx_jy']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: -ctx['eta']() * jy / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R23xx = Term(
    name='R23xx',
    description='shear viscous stress tau_xx in x (for momentum_x)',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dx_jx'],
    fun=lambda ctx: lambda p, jx: ctx['eta']() * ctx['d_dx_jx']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jx: -ctx['eta']() * jx / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jx: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'x'],
    test_deriv='x')

R23yy = Term(
    name='R23yy',
    description='shear viscous stress tau_yy in y (for momentum_y)',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['rho', 'drho_dp', 'eta', 'd_dy_jy'],
    fun=lambda ctx: lambda p, jy: ctx['eta']() * ctx['d_dy_jy']() / ctx['rho'](),
    der_funs=[
        lambda ctx: lambda p, jy: -ctx['eta']() * jy / ctx['rho']()**2 * ctx['drho_dp'](),
        lambda ctx: lambda p, jy: ctx['eta']() / ctx['rho']()
    ],
    trial_deriv=[None, 'y'],
    test_deriv='y')

R24x = Term(
    name='R24x',
    description='wall stress x',
    res='momentum_x',
    dep_vars=['p', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx', 'dtau_xz_dh', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()],
    der_h=lambda ctx: lambda *args: -1 / ctx['h']() ** 2 * ctx['tau_xz']() + 1 / ctx['h']() * ctx['dtau_xz_dh']())

R24y = Term(
    name='R24y',
    description='wall stress y',
    res='momentum_y',
    dep_vars=['p', 'jy'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy', 'dtau_yz_dh', 'drho_dp'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho']() * ctx['drho_dp'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()],
    der_h=lambda ctx: lambda *args: -1 / ctx['h']() ** 2 * ctx['tau_yz']() + 1 / ctx['h']() * ctx['dtau_yz_dh']())

R25x = Term(
    name='R25x',
    description='body force x',
    res='momentum_x',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_x'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_x'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_x']()],
    der_h=lambda ctx: lambda p: ctx['rho']() * ctx['force_x']())

R25y = Term(
    name='R25y',
    description='body force y',
    res='momentum_y',
    dep_vars=['p'],
    dep_vals=['rho', 'drho_dp', 'h', 'force_y'],
    fun=lambda ctx: lambda p: ctx['h']() * ctx['rho']() * ctx['force_y'](),
    der_funs=[lambda ctx: lambda p: ctx['h']() * ctx['drho_dp']() * ctx['force_y']()],
    der_h=lambda ctx: lambda p: ctx['rho']() * ctx['force_y']())

R2Tx = Term(
    name='R2Tx',
    description='time derivative momentum_x',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=['jx_prev'],
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']()])

R2Ty = Term(
    name='R2Ty',
    description='time derivative momentum_y',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=['jy_prev'],
    fun=lambda ctx: lambda jy: - (jy - ctx['jy_prev']()) / ctx['dt'](),
    der_funs=[lambda ctx: lambda jy: - np.full_like(jy, 1.0) / ctx['dt']()])

R2Thx = Term(
    name='R2Thx',
    description='squeeze source momentum_x (height rate of change, exact, matches R1Th)',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=['h_prev', 'dh_dt'],
    fun=lambda ctx: lambda jx: - jx / ctx['h_prev']() * ctx['dh_dt'](),
    der_funs=[lambda ctx: lambda jx: - ctx['dh_dt']() / ctx['h_prev']()],
    der_h=lambda ctx: lambda jx: - jx / ctx['h_prev']() / ctx['dt']())

R2Thy = Term(
    name='R2Thy',
    description='squeeze source momentum_y (height rate of change, exact, matches R1Th)',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=['h_prev', 'dh_dt'],
    fun=lambda ctx: lambda jy: - jy / ctx['h_prev']() * ctx['dh_dt'](),
    der_funs=[lambda ctx: lambda jy: - ctx['dh_dt']() / ctx['h_prev']()],
    der_h=lambda ctx: lambda jy: - jy / ctx['h_prev']() / ctx['dt']())
