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

# flake8: noqa: E501

from .utils2d import NonLinearTerm
import numpy as np

R11x = NonLinearTerm(
    name='R11x',
    description='flux divergence x',
    res='mass',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: -jx,
    der_funs=[lambda ctx: lambda jx: np.full_like(jx, -1.0)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R11y = NonLinearTerm(
    name='R11y',
    description='flux divergence y',
    res='mass',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: -jy,
    der_funs=[lambda ctx: lambda jy: np.full_like(jy, -1.0)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R21x = NonLinearTerm(
    name='R21x',
    description='pressure gradient x',
    res='momentum_x',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda *args: -ctx['p'](),
    der_funs=[lambda ctx: lambda *args: -ctx['dp_drho']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R21y = NonLinearTerm(
    name='R21y',
    description='pressure gradient y',
    res='momentum_y',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda *args: -ctx['p'](),
    der_funs=[lambda ctx: lambda *args: -ctx['dp_drho']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R24x= NonLinearTerm(
    name='R24x',
    description='wall stress',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'tau_xz', 'dtau_xz_drho', 'dtau_xz_djx'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_xz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_drho'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_xz_djx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R24y= NonLinearTerm(
    name='R24y',
    description='wall stress',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'tau_yz', 'dtau_yz_drho', 'dtau_yz_djy'],
    fun=lambda ctx: lambda *args: 1 / ctx['h']() * ctx['tau_yz'](),
    der_funs=[lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_drho'](),
              lambda ctx: lambda *args: 1 / ctx['h']() * ctx['dtau_yz_djy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

term_list = [R11x, R11y,
             R21x, R21y,
             R24x, R24y]
