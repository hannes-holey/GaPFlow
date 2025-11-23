#
# Copyright 2025 Christoph Huber
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
from .utils import NonLinearTerm
import numpy as np

R11 = NonLinearTerm(
    name='R11',
    res='mass',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: -jx,
    der_funs=[lambda ctx: lambda jx: np.full_like(jx, -1.0)],
    d_dx_resfun=True,
    d_dx_testfun=False,
    nb_quad_pts=2)

R12 = NonLinearTerm(
    name='R12',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda jx: -1 / ctx['h']() * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -1 / ctx['h']() * ctx['dh_dx']()],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=2)

R1T = NonLinearTerm(
    name='R1T',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: - (rho - ctx['rho_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda rho: - np.full_like(rho, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

R21 = NonLinearTerm(
    name='R21',
    res='momentum_x',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: -ctx['p'](rho),
    der_funs=[lambda ctx: lambda rho: -ctx['dp_drho'](rho)],
    d_dx_resfun=True,
    d_dx_testfun=False,
    nb_quad_pts=2)

R24 = NonLinearTerm(
    name='R24',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx'],
    fun=lambda ctx: lambda rho, jx: 1 / ctx['h']() * ctx['tau_xz'](rho, jx, ctx['h'](), ctx['dh_dx'](), ctx['U']()),
    der_funs=[lambda ctx: lambda rho, jx: 1 / ctx['h']() * ctx['dtau_xz_drho'](rho, jx, ctx['h'](), ctx['dh_dx'](), ctx['U']()),
              lambda ctx: lambda rho, jx: 1 / ctx['h']() * ctx['dtau_xz_djx'](rho, jx, ctx['h'](), ctx['dh_dx'](), ctx['U']())],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

R2T = NonLinearTerm(
    name='R2T',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

R31 = NonLinearTerm(
    name='R31',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E,
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E,
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jx, E: - (jx / rho)],
    d_dx_resfun=True,
    d_dx_testfun=False,
    nb_quad_pts=3)

R31S = NonLinearTerm(
    name='R31S',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

R32 = NonLinearTerm(
    name='R32',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, E: - ctx['p'](rho) * (jx / rho),
    der_funs=[lambda ctx: lambda rho, jx: - (ctx['dp_drho'](rho) * (jx / rho) + ctx['p'](rho) * (jx / rho**2)),
              lambda ctx: lambda rho, jx: - ctx['p'](rho) * (1 / rho)],
    d_dx_resfun=True,
    d_dx_testfun=False,
    nb_quad_pts=3)

R32S = NonLinearTerm(
    name='R32S',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: - ctx['p'](rho) * (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx: - ((ctx['dp_drho'](rho) * (jx / rho) - ctx['p'](rho) * (jx / rho**2)) * (1 / ctx['h']() * ctx['dh_dx']())),
              lambda ctx: lambda rho, jx: - ctx['p'](rho) * (1 / rho) * (1 / ctx['h']() * ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

# R33: shear stress work

R34 = NonLinearTerm(
    name='R34',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx', 'tau_xz_bot', 'U'],
    fun=lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['tau_xz_bot'](rho, jx, ctx['h'](), ctx['dh_dx'](), ctx['U']()) * ctx['U'](),
    der_funs=[lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['dtau_xz_bot_drho'](rho, jx, ctx['h'](), ctx['dh_dx'](), ctx['U']()) * ctx['U'](),
              lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['dtau_xz_bot_djx'](rho, jx, ctx['h'](), ctx['dh_dx'](), ctx['U']()) * ctx['U']()],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

R35 = NonLinearTerm(
    name='R35',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, E: - ctx['k']() * ctx['T'](rho, jx, E),
    der_funs=[lambda ctx: lambda rho, jx, E: - ctx['k']() * ctx['dT_drho'](rho, jx, E),
              lambda ctx: lambda rho, jx, E: - ctx['k']() * ctx['dT_djx'](rho, jx, E),
              lambda ctx: lambda rho, jx, E: - ctx['k']() * ctx['dT_dE'](rho, jx, E)],
    d_dx_resfun=True,
    d_dx_testfun=True,
    nb_quad_pts=3)

R36 = NonLinearTerm(
    name='R36',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: ctx['S'](rho, jx, E, ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx, E: ctx['dS_drho'](rho, jx, E, ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: ctx['dS_djx'](rho, jx, E, ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: ctx['dS_dE'](rho, jx, E, ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3
)

R3T = NonLinearTerm(
    name='R3T',
    res='energy',
    dep_vars=['E'],
    dep_vals=[],
    fun=lambda ctx: lambda E: - (E - ctx['E_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda E: - np.full_like(E, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dx_testfun=False,
    nb_quad_pts=3)

term_list = [R11, R12, R1T, R21, R24, R2T, R31, R31S, R32, R32S, R34, R35, R36, R3T]
