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

# flake8: noqa: E501, F401, F403
"""`Term` class plus the coordinator for the term library in `terms_lib/`.

- `terms_lib/terms_hydro.py`    — base (non-cavitation) mass/momentum terms (R1*, R2*)
- `terms_lib/terms_energy.py`   — energy equation terms (R3*)
- `terms_lib/terms_cav.py`      — Elrod-Adams cavitation mass/wall-stress terms,
                                   Fischer-Burmeister complementarity (R_cav), AD stabilization
- `terms_lib/terms_cav_oss.py`  — OSS + flux-capturing (FC) stabilization terms
- `terms_lib/terms_cav_supg.py` — SUPG companions for the cavitation mass equation

Each `terms_lib` submodule imports `Term` back from this module (`from ..terms
import Term`) and declares `__all__`; `get_active_terms()` below is the single
dispatcher assembling the active term list from `fem_solver` config flags.
"""

from typing import Callable, List

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class Term():
    def __init__(self,
                 name: str,
                 description: str,
                 res: str,
                 dep_vars: list[str],
                 dep_vals: list[str],
                 fun: Callable,
                 der_funs: list[Callable],
                 trial_deriv: 'str | list[str | None] | None' = None,
                 test_deriv: 'str | None' = None,
                 der_h: 'Callable | None' = None,
                 der_h_dx: 'Callable | None' = None,
                 der_h_dy: 'Callable | None' = None):
        self.name = name
        self.description = description
        self.res = res
        self.dep_vars = dep_vars
        self.dep_vals = dep_vals
        self.fun_ = fun
        self.der_funs_ = der_funs
        self.trial_deriv = trial_deriv
        self.test_deriv = test_deriv
        self.der_h_ = der_h
        self.der_h_dx_ = der_h_dx
        self.der_h_dy_ = der_h_dy
        self.built = False

    def build(self, ctx: dict) -> None:
        self.fun = self.fun_(ctx)
        self.der_funs = [der_fun_(ctx) for der_fun_ in self.der_funs_]
        self.der_h = self.der_h_(ctx) if self.der_h_ is not None else None
        self.der_h_dx = self.der_h_dx_(ctx) if self.der_h_dx_ is not None else None
        self.der_h_dy = self.der_h_dy_(ctx) if self.der_h_dy_ is not None else None
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

    def depvar_deriv_for(self, var: str) -> str:
        """Direction of spatial derivative acting on dep_var 'var': 'none', 'x', or 'y'."""
        i = self.dep_vars.index(var)
        d = self.trial_deriv[i] if isinstance(self.trial_deriv, list) else self.trial_deriv
        return d if d is not None else 'none'

    @property
    def depvar_deriv(self):
        """Direction of spatial derivative acting on dep_var: 'none', 'x', or 'y'.
        All dep_vars must share the same derivative direction (use depvar_deriv_for otherwise)."""
        td = self.trial_deriv
        if isinstance(td, list):
            directions = [d for d in td if d is not None]
            return directions[0] if directions else 'none'
        return td if td is not None else 'none'

    @property
    def deriv_key(self):
        """Return (depvar_deriv, test_deriv) tuple for template lookup.
        Only valid when all dep_vars share the same derivative direction."""
        return (self.depvar_deriv, self.test_deriv)


def collect_required_fields(terms: List[Term]):
    """Collect all quad field keys required by a list of active terms."""
    plain = set()
    der = set()
    for term in terms:
        for key in term.dep_vals:
            if key.startswith('d_d'):
                der.add(key)
            else:
                plain.add(key)
    return plain, der


from .terms_lib.terms_hydro import *
from .terms_lib.terms_energy import *
from .terms_lib.terms_cav import *
from .terms_lib.terms_cav_oss import *
from .terms_lib.terms_cav_supg import *
from .terms_lib.terms_reynolds import *


def get_active_terms(fem_solver: dict) -> List[Term]:
    """Return active Term instances based on fem_solver config.

    Base terms:
    - mass change, flux, height source (R1T, R11*, R11S*)
    - momentum changes, pressure gradients (R2T*, R21*)

    Physics flags (in fem_solver['physics']):
    - gap_shear:          Gap-averaged wall shear (R24x, R24y)
    - plane_shear:        In-plane viscous diffusion (R23xy, R23yx)
    - inertia:            Momentum convection (R22*)
    - body_force:         Body force (R25x, R25y)
    - squeeze:            Height rate-of-change source in mass and momentum (R1Th, R2Thx, R2Thy)
    - energy:              Energy equation master switch, subflags below default to True
    - energy_convection:  Energy advection (R31*)
    - pressure_work:      Pressure-volume work (R32*)
    - thermal_diffusion:  Heat conduction (R35x, R35y)
    - wall_heat_balance:  Wall heat flux BC (R36)
    - wall_shear_work:    Wall stress work / shear heating (R34)

    Reynolds mode (fem_solver['equations']['reynolds']):
    - No-squeeze scalar Reynolds equation, DOF `p` only, residual `mass` only
      (Rey_R1T, Rey_R11x/y, Rey_R11Sx/y). Mutually exclusive with all other
      equation/physics flags -- see sanitize_fem_solver in io.py.
    """
    physics = fem_solver['physics']
    stab = fem_solver['stabilization']
    cavitation = fem_solver['equations']['cavitation']

    if fem_solver['equations'].get('reynolds', False):
        return list(REYNOLDS_TERMS)

    if cavitation:
        terms = [*CAV_MASS_TERMS, R21x, R21y, R2Tx, R2Ty, R_cav]
    else:
        terms = [R11x, R11y, R11Sx, R11Sy, R1T,
                 R21x, R21y, R2Tx, R2Ty]

    if cavitation and stab['ad']:
        terms += THETA_TERMS_AD

    if cavitation and stab['oss']:
        terms += OSS_TERMS

    if cavitation and stab['fc']:
        terms += FC_TERMS

    if cavitation and stab['mass_supg']:
        terms += SUPG_TERMS
        if physics['squeeze']:
            terms += SUPG_SQUEEZE_TERMS

    if physics['gap_shear']:
        terms += THETA_TERMS_WALL_STRESS if cavitation else [R24x, R24y]

    if physics['plane_shear']:
        terms += [R23xy, R23yx, R23xx, R23yy]

    if physics['inertia']:
        terms += [R22xx, R22Sxx, R22yx, R22Syx, R22xy, R22Sxy, R22yy, R22Syy]

    if physics['body_force']:
        terms += [R25x, R25y]

    if physics['squeeze']:
        terms += [R2Thx, R2Thy]
        terms += [R1Th_cav] if cavitation else [R1Th]

    if physics['energy']:
        terms.append(R3T)

        if physics['wall_shear_work']:
            terms.append(R34)

        if physics['energy_convection']:
            terms += [R31x, R31y, R31Sx, R31Sy]

        if physics['pressure_work']:
            terms += [R32x, R32y, R32Sx, R32Sy]

        if physics['thermal_diffusion']:
            terms += [R35x, R35y]

        if physics['wall_heat_balance']:
            terms.append(R36)

    return terms
