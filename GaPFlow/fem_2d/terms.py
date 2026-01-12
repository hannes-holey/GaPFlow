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

from typing import Callable
import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class NonLinearTerm():
    def __init__(self,
                 name: str,
                 description: str,
                 res: str,
                 dep_vars: list[str],
                 dep_vals: list[str],
                 fun: Callable,
                 der_funs: list[Callable],
                 d_dx_resfun: bool = False,
                 d_dy_resfun: bool = False,
                 der_testfun: bool = False):
        self.name = name
        self.description = description
        self.res = res
        self.dep_vars = dep_vars
        self.dep_vals = dep_vals
        self.fun_ = fun
        self.der_funs_ = der_funs
        self.d_dx_resfun = d_dx_resfun
        self.d_dy_resfun = d_dy_resfun
        self.der_testfun = der_testfun
        self.built = False

    def build(self, ctx: dict) -> None:
        self.fun = self.fun_(ctx)
        self.der_funs = [der_fun_(ctx) for der_fun_ in self.der_funs_]
        self.built = True

    def evaluate(self, *args) -> NDArray:
        if not self.built:
            raise Exception("Term not built")
        return self.fun(*args)

    def evaluate_deriv(self, dep_var: str, *args) -> NDArray:
        if not self.built:
            raise Exception("Term not built")
        i = self.dep_vars.index(dep_var)
        return self.der_funs[i](*args)


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

R11Sx = NonLinearTerm(
    name='R11Sx',
    description='flux divergence height source',
    res='mass',
    dep_vars=['jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda jx: -1 / ctx['h']() * ctx['dh_dx']() * jx,
    der_funs=[lambda ctx: lambda jx: -1 / ctx['h']() * ctx['dh_dx']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R11Sy = NonLinearTerm(
    name='R11Sy',
    description='flux divergence height source',
    res='mass',
    dep_vars=['jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda jy: -1 / ctx['h']() * ctx['dh_dy']() * jy,
    der_funs=[lambda ctx: lambda jy: -1 / ctx['h']() * ctx['dh_dy']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R1Stabx = NonLinearTerm(
    name='R1Stabx',
    description='pressure stabilization x',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: -ctx['pressure_stab']() * ctx['p'](),
    der_funs=[lambda ctx: lambda rho: -ctx['pressure_stab']() * ctx['dp_drho']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=True)

R1Staby = NonLinearTerm(
    name='R1Staby',
    description='pressure stabilization y',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: -ctx['pressure_stab']() * ctx['p'](),
    der_funs=[lambda ctx: lambda rho: -ctx['pressure_stab']() * ctx['dp_drho']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=True)

R1T = NonLinearTerm(
    name='R1T',
    description='time derivative',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: - (rho - ctx['rho_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda rho: - np.full_like(rho, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
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

R2Tx = NonLinearTerm(
    name='R2Tx',
    description='time derivative',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: - (jx - ctx['jx_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda jx: - np.full_like(jx, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R2Ty = NonLinearTerm(
    name='R2Ty',
    description='time derivative',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: - (jy - ctx['jy_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda jy: - np.full_like(jy, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)


term_list = [R11x, R11y, R11Sx, R11Sy, R1T, R1Stabx, R1Staby,
             R21x, R21y, R24x, R24y, R2Tx, R2Ty]


def get_active_terms(fem_solver: dict) -> list[NonLinearTerm]:
    term_list_res = []
    for term in term_list:
        if term.name in fem_solver['equations']['term_list']:
            term_list_res.append(term)
    return term_list_res
