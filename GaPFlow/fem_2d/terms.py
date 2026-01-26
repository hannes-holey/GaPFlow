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


# -----------------------------------------------------------------------------
# Mass equation terms (R1*)
# -----------------------------------------------------------------------------

# R11: Flux divergence
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

# R1T: Time derivative
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

# R1Stab: Pressure stabilization
R1Stabx = NonLinearTerm(
    name='R1Stabx',
    description='pressure stabilization x',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: -ctx['tau_mass']() * ctx['p'](),
    der_funs=[lambda ctx: lambda rho: -ctx['tau_mass']() * ctx['dp_drho']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=True)

R1Staby = NonLinearTerm(
    name='R1Staby',
    description='pressure stabilization y',
    res='mass',
    dep_vars=['rho'],
    dep_vals=[],
    fun=lambda ctx: lambda rho: -ctx['tau_mass']() * ctx['p'](),
    der_funs=[lambda ctx: lambda rho: -ctx['tau_mass']() * ctx['dp_drho']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=True)

# -----------------------------------------------------------------------------
# Momentum equation terms (R2*)
# -----------------------------------------------------------------------------

# R21: Pressure gradient
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

# R22: Convective momentum flux
R22xx = NonLinearTerm(
    name='R22xx',
    description='convective momentum flux jx*jx in x',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx: -(jx * jx) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: (jx * jx) / (rho ** 2),
        lambda ctx: lambda rho, jx: -2 * jx / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R22xxS = NonLinearTerm(
    name='R22xxS',
    description='convective momentum flux jx*jx height source',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jx) / (rho ** 2),
        lambda ctx: lambda rho, jx: -1 / ctx['h']() * ctx['dh_dx']() * 2 * jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R22yx = NonLinearTerm(
    name='R22yx',
    description='convective momentum flux jx*jy in y',
    res='momentum_x',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy: -(jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -jy / rho,
        lambda ctx: lambda rho, jx, jy: -jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R22yxS = NonLinearTerm(
    name='R22yxS',
    description='convective momentum flux jx*jy height source',
    res='momentum_x',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jy / rho,
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dy']() * jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R22xy = NonLinearTerm(
    name='R22xy',
    description='convective momentum flux jx*jy in x',
    res='momentum_y',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy: -(jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -jy / rho,
        lambda ctx: lambda rho, jx, jy: -jx / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R22xyS = NonLinearTerm(
    name='R22xyS',
    description='convective momentum flux jx*jy height source',
    res='momentum_y',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jx, jy: 1 / ctx['h']() * ctx['dh_dx']() * (jx * jy) / (rho ** 2),
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jy / rho,
        lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dh_dx']() * jx / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R22yy = NonLinearTerm(
    name='R22yy',
    description='convective momentum flux jy*jy in y',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy: -(jy * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: (jy * jy) / (rho ** 2),
        lambda ctx: lambda rho, jy: -2 * jy / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R22yyS = NonLinearTerm(
    name='R22yyS',
    description='convective momentum flux jy*jy height source',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy: -1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: 1 / ctx['h']() * ctx['dh_dy']() * (jy * jy) / (rho ** 2),
        lambda ctx: lambda rho, jy: -1 / ctx['h']() * ctx['dh_dy']() * 2 * jy / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R23: In-plane shear stress (viscous diffusion)
# Simplified diffusion form: exact for incompressible flow (div v = 0)
R23xy = NonLinearTerm(
    name='R23xy',
    description='shear viscous stress tau_xy in y (for momentum_x)',
    res='momentum_x',
    dep_vars=['rho', 'jx'],
    dep_vals=['eta'],
    fun=lambda ctx: lambda rho, jx: -ctx['eta']() * jx / rho,
    der_funs=[
        lambda ctx: lambda rho, jx: ctx['eta']() * jx / (rho ** 2),
        lambda ctx: lambda rho, jx: -ctx['eta']() / rho
    ],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=True)

R23yx = NonLinearTerm(
    name='R23yx',
    description='shear viscous stress tau_xy in x (for momentum_y)',
    res='momentum_y',
    dep_vars=['rho', 'jy'],
    dep_vals=['eta'],
    fun=lambda ctx: lambda rho, jy: -ctx['eta']() * jy / rho,
    der_funs=[
        lambda ctx: lambda rho, jy: ctx['eta']() * jy / (rho ** 2),
        lambda ctx: lambda rho, jy: -ctx['eta']() / rho
    ],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=True)

# R24: Wall stress
R24x = NonLinearTerm(
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

R24y = NonLinearTerm(
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

# R25: Body force (for driving flow with periodic BCs)
# Force per unit mass (like gravity), contributes h*rho*force to momentum
R25x = NonLinearTerm(
    name='R25x',
    description='body force x',
    res='momentum_x',
    dep_vars=['rho'],
    dep_vals=['h', 'force_x'],
    fun=lambda ctx: lambda rho: ctx['h']() * rho * ctx['force_x'](),
    der_funs=[lambda ctx: lambda rho: ctx['h']() * ctx['force_x']() * np.ones_like(rho)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R25y = NonLinearTerm(
    name='R25y',
    description='body force y',
    res='momentum_y',
    dep_vars=['rho'],
    dep_vals=['h', 'force_y'],
    fun=lambda ctx: lambda rho: ctx['h']() * rho * ctx['force_y'](),
    der_funs=[lambda ctx: lambda rho: ctx['h']() * ctx['force_y']() * np.ones_like(rho)],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R2T: Time derivative
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

# R2Stab: Momentum stabilization
R2Stabx = NonLinearTerm(
    name='R2Stabx',
    description='momentum stabilization x',
    res='momentum_x',
    dep_vars=['jx'],
    dep_vals=[],
    fun=lambda ctx: lambda jx: -ctx['tau_mom']() * jx,
    der_funs=[lambda ctx: lambda jx: -ctx['tau_mom']() * np.ones_like(jx)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=True)

R2Staby = NonLinearTerm(
    name='R2Staby',
    description='momentum stabilization y',
    res='momentum_y',
    dep_vars=['jy'],
    dep_vals=[],
    fun=lambda ctx: lambda jy: -ctx['tau_mom']() * jy,
    der_funs=[lambda ctx: lambda jy: -ctx['tau_mom']() * np.ones_like(jy)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=True)

# -----------------------------------------------------------------------------
# Energy equation terms (R3*)
# -----------------------------------------------------------------------------

# R31: Energy convection
R31x = NonLinearTerm(
    name='R31x',
    description='energy convection x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E,
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E,
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jx, E: - (jx / rho)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R31y = NonLinearTerm(
    name='R31y',
    description='energy convection y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E,
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E,
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E,
              lambda ctx: lambda rho, jy, E: - (jy / rho)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R31Sx = NonLinearTerm(
    name='R31Sx',
    description='energy convection height source x',
    res='energy',
    dep_vars=['rho', 'jx', 'E'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx, E: - (jx / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx, E: (jx / rho**2) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dx']()),
              lambda ctx: lambda rho, jx, E: - (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R31Sy = NonLinearTerm(
    name='R31Sy',
    description='energy convection height source y',
    res='energy',
    dep_vars=['rho', 'jy', 'E'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy, E: - (jy / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy, E: (jy / rho**2) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (1 / rho) * E * (1 / ctx['h']() * ctx['dh_dy']()),
              lambda ctx: lambda rho, jy, E: - (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R32: Pressure work
R32x = NonLinearTerm(
    name='R32x',
    description='pressure work x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho),
    der_funs=[lambda ctx: lambda rho, jx: - (ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=False)

R32y = NonLinearTerm(
    name='R32y',
    description='pressure work y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho),
    der_funs=[lambda ctx: lambda rho, jy: - (ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=False)

R32Sx = NonLinearTerm(
    name='R32Sx',
    description='pressure work height source x',
    res='energy',
    dep_vars=['rho', 'jx'],
    dep_vals=['h', 'dh_dx'],
    fun=lambda ctx: lambda rho, jx: - ctx['p']() * (jx / rho) * (1 / ctx['h']() * ctx['dh_dx']()),
    der_funs=[lambda ctx: lambda rho, jx: - ((ctx['dp_drho']() * (jx / rho) - ctx['p']() * (jx / rho**2)) * (1 / ctx['h']() * ctx['dh_dx']())),
              lambda ctx: lambda rho, jx: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dx']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

R32Sy = NonLinearTerm(
    name='R32Sy',
    description='pressure work height source y',
    res='energy',
    dep_vars=['rho', 'jy'],
    dep_vals=['h', 'dh_dy'],
    fun=lambda ctx: lambda rho, jy: - ctx['p']() * (jy / rho) * (1 / ctx['h']() * ctx['dh_dy']()),
    der_funs=[lambda ctx: lambda rho, jy: - ((ctx['dp_drho']() * (jy / rho) - ctx['p']() * (jy / rho**2)) * (1 / ctx['h']() * ctx['dh_dy']())),
              lambda ctx: lambda rho, jy: - ctx['p']() * (1 / rho) * (1 / ctx['h']() * ctx['dh_dy']())],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R34: Wall stress work
R34 = NonLinearTerm(
    name='R34',
    description='wall stress work',
    res='energy',
    dep_vars=['rho', 'jx', 'jy'],
    dep_vals=['h', 'tau_xz_bot', 'tau_yz_bot', 'dtau_xz_bot_drho', 'dtau_xz_bot_djx',
              'dtau_yz_bot_drho', 'dtau_yz_bot_djy', 'U', 'V'],
    fun=lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * (ctx['tau_xz_bot']() * ctx['U']() + ctx['tau_yz_bot']() * ctx['V']()),
    der_funs=[lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * (ctx['dtau_xz_bot_drho']() * ctx['U']() + ctx['dtau_yz_bot_drho']() * ctx['V']()),
              lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dtau_xz_bot_djx']() * ctx['U'](),
              lambda ctx: lambda rho, jx, jy: -1 / ctx['h']() * ctx['dtau_yz_bot_djy']() * ctx['V']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R35: Thermal diffusion
R35x = NonLinearTerm(
    name='R35x',
    description='thermal diffusion x',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['T'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_drho'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djx'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djy'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_dE']()],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=True)

R35y = NonLinearTerm(
    name='R35y',
    description='thermal diffusion y',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['T'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_drho'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djx'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_djy'](),
              lambda ctx: lambda rho, jx, jy, E: - ctx['k']() * ctx['dT_dE']()],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=True)

# R36: Wall heat balance (source term)
R36 = NonLinearTerm(
    name='R36',
    description='wall heat balance',
    res='energy',
    dep_vars=['rho', 'jx', 'jy', 'E'],
    dep_vals=[],
    fun=lambda ctx: lambda rho, jx, jy, E: ctx['S'](),
    der_funs=[lambda ctx: lambda rho, jx, jy, E: ctx['dS_drho'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_djx'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_djy'](),
              lambda ctx: lambda rho, jx, jy, E: ctx['dS_dE']()],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R3T: Time derivative
R3T = NonLinearTerm(
    name='R3T',
    description='energy time derivative',
    res='energy',
    dep_vars=['E'],
    dep_vals=[],
    fun=lambda ctx: lambda E: - (E - ctx['E_prev']()) / ctx['dt'],
    der_funs=[lambda ctx: lambda E: - np.full_like(E, 1.0) / ctx['dt']],
    d_dx_resfun=False,
    d_dy_resfun=False,
    der_testfun=False)

# R3Stab: Energy stabilization
R3Stabx = NonLinearTerm(
    name='R3Stabx',
    description='energy stabilization x',
    res='energy',
    dep_vars=['E'],
    dep_vals=[],
    fun=lambda ctx: lambda E: -ctx['tau_energy']() * E,
    der_funs=[lambda ctx: lambda E: -ctx['tau_energy']() * np.ones_like(E)],
    d_dx_resfun=True,
    d_dy_resfun=False,
    der_testfun=True)

R3Staby = NonLinearTerm(
    name='R3Staby',
    description='energy stabilization y',
    res='energy',
    dep_vars=['E'],
    dep_vals=[],
    fun=lambda ctx: lambda E: -ctx['tau_energy']() * E,
    der_funs=[lambda ctx: lambda E: -ctx['tau_energy']() * np.ones_like(E)],
    d_dx_resfun=False,
    d_dy_resfun=True,
    der_testfun=True)


# -----------------------------------------------------------------------------
# Term list and selection functions
# -----------------------------------------------------------------------------

term_list = [
    # Mass equation
    R11x, R11y, R11Sx, R11Sy, R1T, R1Stabx, R1Staby,
    # Momentum equation (numerical order: R21 -> R22 -> R23 -> R24 -> R25 -> R2T -> R2Stab)
    R21x, R21y,
    R22xx, R22xxS, R22yx, R22yxS, R22xy, R22xyS, R22yy, R22yyS,
    R23xy, R23yx,
    R24x, R24y,
    R25x, R25y,
    R2Tx, R2Ty,
    R2Stabx, R2Staby,
    # Energy equation
    R31x, R31y, R31Sx, R31Sy, R32x, R32y, R32Sx, R32Sy,
    R34, R35x, R35y, R36, R3T, R3Stabx, R3Staby,
]


def get_default_terms(fem_solver: dict) -> list[str]:
    """Determine default 2D term names based on config flags."""
    terms = ['R11x', 'R11y', 'R11Sx', 'R11Sy',   # Mass
             'R21x', 'R21y', 'R24x', 'R24y',     # Momentum
             'R1T', 'R2Tx', 'R2Ty']              # Time derivatives

    # Pressure stabilization
    if fem_solver['pressure_stab_alpha'] > 0:
        terms.extend(['R1Stabx', 'R1Staby'])

    # Momentum stabilization
    if fem_solver['momentum_stab_alpha'] > 0:
        terms.extend(['R2Stabx', 'R2Staby'])

    # Energy
    if fem_solver['equations']['energy']:
        terms.extend(['R31x', 'R31y', 'R31Sx', 'R31Sy',
                      'R32x', 'R32y', 'R32Sx', 'R32Sy',
                      'R34', 'R35x', 'R35y', 'R36', 'R3T'])

        # Energy stabilization: enabled when alpha > 0
        if fem_solver['energy_stab_alpha'] > 0:
            terms.extend(['R3Stabx', 'R3Staby'])

    return terms


def get_active_terms(fem_solver: dict) -> list[NonLinearTerm]:
    """Get active terms based on config.

    If user specified explicit term_list, use that.
    Otherwise, auto-select based on config flags.
    """
    user_terms = fem_solver['equations'].get('term_list')
    if user_terms is not None:
        requested = set(user_terms)
    else:
        requested = set(get_default_terms(fem_solver))

    return [t for t in term_list if t.name in requested]
