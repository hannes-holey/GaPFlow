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
"""Coefficient functions of the generalized Reynolds equation.

    d/dt(rho_eff h (1-theta)) = div(epsilon grad p) - div(rho_eff h (1-theta) U_m)

`ReynoldsEpsilon` provides the Poiseuille coefficient epsilon and `ReynoldsRhoEff`
the effective, height-averaged density rho_eff, each with its derivatives, as quad-point
functions for the FEM registry (see `fieldspec.QUAD_FIELD_REGISTRY`).
Both can be replaced by user functions of (p, h), see `Problem.set_custom_epsilon`
and `Problem.set_custom_rho_eff`.
"""

from typing import Callable, Optional

import jax.numpy as jnp
from jax import grad, jit, vmap

from .viscosity import piezoviscosity

vmap2 = lambda f: jit(vmap(vmap(f)))


class ReynoldsEpsilon:
    """Poiseuille coefficient epsilon(p, h) and its derivatives w.r.t. p and h.

    Default: epsilon = rho h^3 / (12 eta) with rho from the EoS and the (underrelaxed)
    nodal viscosity eta. deta/dp follows the piezoviscosity law and the
    `viscosity.freeze_gradient` / `underrelax_gradient_value` options; shear-thinning
    is frozen.

    Custom: closed-form, JAX-differentiable scalar function `epsilon(p, h)`.
    """

    def __init__(self, prop: dict) -> None:
        self.prop = prop
        self.custom_fn: Optional[Callable] = None

    @property
    def is_custom(self) -> bool:
        return self.custom_fn is not None

    def build_grad(self) -> None:
        if self.is_custom:
            fn = self.custom_fn
            self.epsilon = vmap2(fn)
            self.depsilon_dp = vmap2(grad(fn, argnums=0))
            self.depsilon_dh = vmap2(grad(fn, argnums=1))
            return

        deta_dp = self._build_deta_dp()

        def _eps(rho, h, eta, drho_dp, p):
            return rho * h ** 3 / (12.0 * eta)

        def _deps_dp(rho, h, eta, drho_dp, p):
            return h ** 3 / (12.0 * eta) * drho_dp - rho * h ** 3 / (12.0 * eta ** 2) * deta_dp(p)

        def _deps_dh(rho, h, eta, drho_dp, p):
            return rho * h ** 2 / (4.0 * eta)

        self.epsilon = vmap2(_eps)
        self.depsilon_dp = vmap2(_deps_dp)
        self.depsilon_dh = vmap2(_deps_dh)

    def _build_deta_dp(self) -> Callable:
        visc = self.prop['viscosity']
        if visc['freeze_gradient'] or 'piezo' not in self.prop:
            return lambda p: 0.0

        piezo = self.prop['piezo']
        mu0 = self.prop['shear']
        deta = grad(lambda p: piezoviscosity(p, mu0, piezo))
        if visc['underrelax_gradient_value']:
            alpha = visc['alpha_underrelax']
            return lambda p: alpha * deta(p)
        return deta


class ReynoldsRhoEff:
    """Effective, height-averaged density rho_eff(p, h) and its derivative w.r.t. p.

    Default: rho_eff = rho from the EoS (passed through from the `rho` / `drho_dp`
    quad fields). Custom: closed-form, JAX-differentiable scalar function `rho_eff(p, h)`.
    """

    def __init__(self) -> None:
        self.custom_fn: Optional[Callable] = None

    @property
    def is_custom(self) -> bool:
        return self.custom_fn is not None

    def build_grad(self) -> None:
        if self.is_custom:
            fn = self.custom_fn
            self.rho_eff = vmap2(fn)
            self.drho_eff_dp = vmap2(grad(fn, argnums=0))
            self.drho_eff_dh = vmap2(grad(fn, argnums=1))
            return

        self.rho_eff = lambda rho, drho_dp: jnp.asarray(rho)
        self.drho_eff_dp = lambda rho, drho_dp: jnp.asarray(drho_dp)
        self.drho_eff_dh = lambda rho, drho_dp: jnp.zeros_like(rho)
