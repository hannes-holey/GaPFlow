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
from muGrid import ConvolutionOperator

from typing import Callable
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


def get_N0_left_test_vals():
    val_1 = 2/3
    val_2 = 1/6
    val_3 = 1/6
    return np.array([val_1, val_2, val_3])

def get_N1_left_test_vals():
    val_1 = 1/6
    val_2 = 1/6
    val_3 = 2/3
    return np.array([val_1, val_2, val_3])

def get_N2_left_test_vals():
    val_1 = 1/6
    val_2 = 2/3
    val_3 = 1/6
    return np.array([val_1, val_2, val_3])

def get_N_left_test_vals():
    N0_vals = get_N0_left_test_vals()
    N1_vals = get_N1_left_test_vals()
    N2_vals = get_N2_left_test_vals()
    return np.array([N0_vals, N1_vals, N2_vals])

def get_N0_right_test_vals():
    val_1 = 2/3
    val_2 = 1/6
    val_3 = 1/6
    return np.array([val_1, val_2, val_3])

def get_N1_right_test_vals():
    # N_br at right triangle quad points (indices 3, 4, 5 in kernel)
    # From kernel: quad3: br=1/6, quad4: br=1/6, quad5: br=2/3
    val_1 = 1/6
    val_2 = 1/6
    val_3 = 2/3
    return np.array([val_1, val_2, val_3])

def get_N2_right_test_vals():
    # N_tl at right triangle quad points (indices 3, 4, 5 in kernel)
    # From kernel: quad3: tl=1/6, quad4: tl=2/3, quad5: tl=1/6
    val_1 = 1/6
    val_2 = 2/3
    val_3 = 1/6
    return np.array([val_1, val_2, val_3])

def get_N_right_test_vals():
    N0_vals = get_N0_right_test_vals()
    N1_vals = get_N1_right_test_vals()
    N2_vals = get_N2_right_test_vals()
    return np.array([N0_vals, N1_vals, N2_vals])

def get_quad_weights():
    w_1 = 1/6
    w_2 = 1/6
    w_3 = 1/6
    return np.array([w_1, w_2, w_3])

def get_triangle_3_operator():
    # a_bl, a_tl, a_br, a_tr
    kernel = np.array([
        [
            [[[2/3, 1/6], [1/6, 0]]],
            [[[1/6, 1/6], [2/3, 0]]],
            [[[1/6, 2/3], [1/6, 0]]],
            [[[0, 1/6], [1/6, 2/3]]],
            [[[0, 2/3], [1/6, 1/6]]],
            [[[0, 1/6], [2/3, 1/6]]],
        ]
    ])
    return ConvolutionOperator([0, 0], kernel)


def get_triangle_2_operator_dx(dx: float):
    """Derivative operator d/dx for linear triangular elements.

    Returns ConvolutionOperator producing 2 values per square (one per triangle).
    Derivative is constant within each linear element.

    muGrid kernel uses (X, Y) convention: kernel[x_offset, y_offset]
    Node layout in (X, Y): bl=(0,0), br=(1,0), tl=(0,1), tr=(1,1)
    Left triangle (bl, tl, br):  dF/dx = (F_br - F_bl) / dx
    Right triangle (tr, br, tl): dF/dx = (F_tr - F_tl) / dx
    """
    kernel = np.array([
        [
            [[[-1/dx, 0], [1/dx, 0]]],      # Left: -bl + br
            [[[0, -1/dx], [0, 1/dx]]],      # Right: -tl + tr
        ]
    ])
    return ConvolutionOperator([0, 0], kernel)


def get_triangle_2_operator_dy(dy: float):
    """Derivative operator d/dy for linear triangular elements.

    muGrid kernel uses (X, Y) convention: kernel[x_offset, y_offset]
    Node layout in (X, Y): bl=(0,0), br=(1,0), tl=(0,1), tr=(1,1)
    Left triangle (bl, tl, br):  dF/dy = (F_tl - F_bl) / dy
    Right triangle (tr, br, tl): dF/dy = (F_tr - F_br) / dy
    """
    kernel = np.array([
        [
            [[[-1/dy, 1/dy], [0, 0]]],      # Left: -bl + tl
            [[[0, 0], [-1/dy, 1/dy]]],      # Right: -br + tr
        ]
    ])
    return ConvolutionOperator([0, 0], kernel)

def get_N_left_test_vals_dx(dx: float):
    val_1 = np.repeat([-1/dx], 3)  # bl: decrease to right
    val_2 = np.repeat([0.0], 3)  # tl: constant in x
    val_3 = np.repeat([1/dx], 3)  # br: increase to right
    return np.array([val_1, val_2, val_3])

def get_N_right_test_vals_dx(dx: float):
    val_1 = np.repeat([1/dx], 3)  # tr: increase to right
    val_2 = np.repeat([0.0], 3)  # br: constant in x
    val_3 = np.repeat([-1/dx], 3)  # tl: decrease to right
    return np.array([val_1, val_2, val_3])

def get_N_left_test_vals_dy(dy: float):
    val_1 = np.repeat([ -1/dy], 3)  # bl: decrease upwards
    val_2 = np.repeat([ 1/dy], 3)  # tl: increase upwards
    val_3 = np.repeat([ 0.0], 3)  # br: constant in y
    return np.array([val_1, val_2, val_3])

def get_N_right_test_vals_dy(dy: float):
    val_1 = np.repeat([ 1/dy], 3)  # tr: increase upwards
    val_2 = np.repeat([-1/dy], 3)  # br: decrease upwards
    val_3 = np.repeat([ 0.0], 3)  # tl: constant in y
    return np.array([val_1, val_2, val_3])

def get_active_terms(fem_solver: dict) -> list[NonLinearTerm]:

    from .terms_2d import term_list
    term_list_res = []
    for term in term_list:
        if term.name in fem_solver['equations']['term_list']:
            term_list_res.append(term)
        else:
            pass
    return term_list_res
