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
    'R_OSS_advx', 'R_OSS_advy', 'R_OSS_proj',
    'R_OSS_mass_xx', 'R_OSS_mass_yy', 'R_OSS_mass_xy', 'R_OSS_mass_yx',
    'R_OSS_corrx', 'R_OSS_corry',
    'R_FC_x', 'R_FC_y',
    'OSS_TERMS', 'OSS_TERM_NAMES', 'FC_TERMS',
]


# ---------------------------------------------------------------------------
# R_oss residual — projection equation
# ---------------------------------------------------------------------------

R_OSS_advx = Term(
    name='R_OSS_advx',
    description='OSS projection eq x: +(η_h, a_vec_x · ∂θ/∂x)',
    res='R_oss',
    dep_vars=['theta'],
    dep_vals=['a_vec_x', 'd_dx_theta'],
    fun=lambda ctx: lambda theta: ctx['a_vec_x']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda theta: ctx['a_vec_x']()],
    trial_deriv='x')

R_OSS_advy = Term(
    name='R_OSS_advy',
    description='OSS projection eq y: +(η_h, a_vec_y · ∂θ/∂y)',
    res='R_oss',
    dep_vars=['theta'],
    dep_vals=['a_vec_y', 'd_dy_theta'],
    fun=lambda ctx: lambda theta: ctx['a_vec_y']() * ctx['d_dy_theta'](),
    der_funs=[lambda ctx: lambda theta: ctx['a_vec_y']()],
    trial_deriv='y')

R_OSS_proj = Term(
    name='R_OSS_proj',
    description='OSS projection eq: −(η_h, ξ_h)',
    res='R_oss',
    dep_vars=['xi'],
    dep_vals=[],
    fun=lambda ctx: lambda xi: -xi,
    der_funs=[lambda ctx: lambda xi: np.full_like(xi, -1.0)])

# ---------------------------------------------------------------------------
# mass conservation residual
# ---------------------------------------------------------------------------

R_OSS_mass_xx = Term(
    name='R_OSS_mass_xx',
    description='OSS mass Laplacian xx: +(∂q/∂x, tau_a_x · a_vec_x · ∂θ/∂x)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_x', 'a_vec_x', 'd_dx_theta'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_x']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_x']()],
    trial_deriv='x',
    test_deriv='x')

R_OSS_mass_yy = Term(
    name='R_OSS_mass_yy',
    description='OSS mass Laplacian yy: +(∂q/∂y, tau_a_y · a_vec_y · ∂θ/∂y)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_y', 'a_vec_y', 'd_dy_theta'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_y']() * ctx['d_dy_theta'](),
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_y']()],
    trial_deriv='y',
    test_deriv='y')

R_OSS_mass_xy = Term(
    name='R_OSS_mass_xy',
    description='OSS mass Laplacian xy: +(∂q/∂y, tau_a_y · a_vec_x · ∂θ/∂x)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_y', 'a_vec_x', 'd_dx_theta'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_x']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_y']() * ctx['a_vec_x']()],
    trial_deriv='x',
    test_deriv='y')

R_OSS_mass_yx = Term(
    name='R_OSS_mass_yx',
    description='OSS mass Laplacian yx: +(∂q/∂x, tau_a_x · a_vec_y · ∂θ/∂y)',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['tau_a_x', 'a_vec_y', 'd_dy_theta'],
    fun=lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_y']() * ctx['d_dy_theta'](),
    der_funs=[lambda ctx: lambda theta: -ctx['tau_a_x']() * ctx['a_vec_y']()],
    trial_deriv='y',
    test_deriv='x')

R_OSS_corrx = Term(
    name='R_OSS_corrx',
    description='OSS mass correction x: −(∂q/∂x, tau_a_x · ξ)',
    res='mass',
    dep_vars=['xi'],
    dep_vals=['tau_a_x'],
    fun=lambda ctx: lambda xi: ctx['tau_a_x']() * xi,
    der_funs=[lambda ctx: lambda xi: ctx['tau_a_x']()],
    test_deriv='x')

R_OSS_corry = Term(
    name='R_OSS_corry',
    description='OSS mass correction y: −(∂q/∂y, tau_a_y · ξ)',
    res='mass',
    dep_vars=['xi'],
    dep_vals=['tau_a_y'],
    fun=lambda ctx: lambda xi: ctx['tau_a_y']() * xi,
    der_funs=[lambda ctx: lambda xi: ctx['tau_a_y']()],
    test_deriv='y')


OSS_TERMS = [
    R_OSS_advx, R_OSS_advy, R_OSS_proj,
    R_OSS_mass_xx, R_OSS_mass_yy, R_OSS_mass_xy, R_OSS_mass_yx,
    R_OSS_corrx, R_OSS_corry,
]

OSS_TERM_NAMES = [t.name for t in OSS_TERMS]

# ---------------------------------------------------------------------------
# Flux-capturing terms
# ---------------------------------------------------------------------------

R_FC_x = Term(
    name='R_FC_x',
    description='flux-capturing theta diffusion x',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['fc_tau', 'd_dx_theta'],
    fun=lambda ctx: lambda theta: -ctx['fc_tau']() * ctx['d_dx_theta'](),
    der_funs=[lambda ctx: lambda theta: -ctx['fc_tau']()],
    trial_deriv='x',
    test_deriv='x')

R_FC_y = Term(
    name='R_FC_y',
    description='flux-capturing theta diffusion y',
    res='mass',
    dep_vars=['theta'],
    dep_vals=['fc_tau', 'd_dy_theta'],
    fun=lambda ctx: lambda theta: -ctx['fc_tau']() * ctx['d_dy_theta'](),
    der_funs=[lambda ctx: lambda theta: -ctx['fc_tau']()],
    trial_deriv='y',
    test_deriv='y')

FC_TERMS = [R_FC_x, R_FC_y]
