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
import numpy as np
from numpy.polynomial.legendre import leggauss

from typing import Callable  # used in NonLinearTerm type hints
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
                 d_dx_testfun: bool = False,
                 nb_quad_pts: int = 3):
        self.name = name
        self.description = description
        self.res = res
        self.dep_vars = dep_vars
        self.dep_vals = dep_vals
        self.fun_ = fun
        self.der_funs_ = der_funs
        self.d_dx_resfun = d_dx_resfun
        self.d_dx_testfun = d_dx_testfun
        self.nb_quad_pts = nb_quad_pts
        self.built = False

    def build(self, ctx: dict) -> None:
        self.fun = self.fun_(ctx)
        self.der_funs = [der_fun_(ctx) for der_fun_ in self.der_funs_]
        self.built = True

    def evaluate(self, *args) -> NDArray:
        if not self.built:
            raise Exception("Term not built. Call build(ctx) before evaluate().")
        return self.fun(*args)

    def evaluate_deriv(self, dep_var: str, *args) -> NDArray:
        if not self.built:
            raise Exception("Term not built. Call build(ctx) before evaluate_deriv().")
        i = self.dep_vars.index(dep_var)
        return self.der_funs[i](*args)


def get_default_terms(fem_solver: dict) -> list[str]:
    """Determine default 1D term names based on config flags.

    For energy, includes the commonly used terms (R34, R35, R36) but not
    convection/pressure work (R31, R32) which may need additional setup.
    Use explicit term_list for full energy equation.
    """
    terms = ['R11', 'R11S', 'R21', 'R22', 'R22S', 'R24']  # Base mass + momentum

    if fem_solver['dynamic']:
        terms.extend(['R1T', 'R2T'])

    if fem_solver['equations']['energy']:
        # Include commonly used energy terms (wall stress work, diffusion, wall heat)
        # Convection (R31, R31S) and pressure work (R32, R32S) require explicit term_list
        terms.extend(['R34', 'R35', 'R36'])
        if fem_solver['dynamic']:
            terms.append('R3T')

    return terms


def get_active_terms(fem_solver: dict) -> list[NonLinearTerm]:
    """Get active terms based on config.

    If user specified explicit term_list, use that.
    Otherwise, auto-select based on config flags.
    """
    from .terms import term_list

    user_terms = fem_solver['equations'].get('term_list')
    if user_terms is not None:
        requested = set(user_terms)
    else:
        requested = set(get_default_terms(fem_solver))

    return [t for t in term_list if t.name in requested]


def get_norm_quad_pts(nb_quad_pts: int) -> NDArray:
    xi, _ = leggauss(nb_quad_pts)
    xi = 0.5 * (xi + 1)
    return xi


def get_norm_quad_wts(nb_quad_pts: int) -> NDArray:
    _, wi = leggauss(nb_quad_pts)
    wi = 0.5 * wi
    return wi


def print_matrix(mat):
    for row in mat:
        formatted = []
        for x in row:
            if x == 0:
                formatted.append(f'{0:>8}')  # width 8, right aligned
            else:
                formatted.append(f'{x:8.2f}')  # width 8, 2 decimals
        print('[', ' '.join(formatted), ']')
    print(np.linalg.matrix_rank(mat))
