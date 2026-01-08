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
from functools import cached_property

import numpy as np
from muGrid import GenericLinearOperator

from typing import Callable
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class TriangleQuadrature:
    """Quadrature data for linear triangular elements on a square cell.

    Single source of truth: 3-point Gauss quadrature with barycentric
    coordinates cycling through [2/3, 1/6, 1/6].

    Node ordering:
    - Left triangle:  [bl, tl, br] (bottom-left, top-left, bottom-right)
    - Right triangle: [tr, br, tl] (top-right, bottom-right, top-left)
    """

    # Barycentric coordinates for 3-point Gauss quadrature on triangle
    # Each row is one quad point, columns are barycentric coords for nodes 0,1,2
    BARY_COORDS = np.array([
        [2 / 3, 1 / 6, 1 / 6],
        [1 / 6, 2 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 3],
    ])

    # Quadrature weights (equal for 3-point rule)
    WEIGHTS = np.array([1 / 6, 1 / 6, 1 / 6])

    def __init__(self, dx: float = 1.0, dy: float = 1.0):
        self.dx = dx
        self.dy = dy

        # Shape function values: N[node, quad_pt]
        # Node ordering: [node0, node1, node2] maps to barycentric [0, 2, 1]
        # This matches the historical FEM node ordering convention
        self._N = self.BARY_COORDS.T[[0, 2, 1], :].copy()

        # Derivatives are constant within each triangle (linear elements)
        # Left triangle nodes: bl(0,0), tl(0,1), br(1,0)
        # dN/dx: bl→br is +x direction, tl is on same x as bl
        self._dN_left_dx = np.array([
            np.full(3, -1 / dx),  # bl: decreases toward br
            np.full(3, 0.0),      # tl: constant in x
            np.full(3, 1 / dx),   # br: increases from bl
        ])
        # dN/dy: bl→tl is +y direction, br is on same y as bl
        self._dN_left_dy = np.array([
            np.full(3, -1 / dy),  # bl: decreases toward tl
            np.full(3, 1 / dy),   # tl: increases from bl
            np.full(3, 0.0),      # br: constant in y
        ])

        # Right triangle nodes: tr(1,1), br(1,0), tl(0,1)
        # dN/dx: tl→tr is +x direction, br is on same x as tr
        self._dN_right_dx = np.array([
            np.full(3, 1 / dx),   # tr: increases from tl
            np.full(3, 0.0),      # br: constant in x
            np.full(3, -1 / dx),  # tl: decreases toward tr
        ])
        # dN/dy: br→tr is +y direction, tl is on same y as tr
        self._dN_right_dy = np.array([
            np.full(3, 1 / dy),   # tr: increases from br
            np.full(3, -1 / dy),  # br: decreases toward tr
            np.full(3, 0.0),      # tl: constant in y
        ])

    @property
    def weights(self) -> NDArray:
        """Quadrature weights, shape (3,)."""
        return self.WEIGHTS

    @property
    def N_left(self) -> NDArray:
        """Shape function values for left triangle, shape (3 nodes, 3 quad pts)."""
        return self._N

    @property
    def N_right(self) -> NDArray:
        """Shape function values for right triangle, shape (3 nodes, 3 quad pts)."""
        return self._N  # Same values, different node interpretation

    @property
    def dN_left_dx(self) -> NDArray:
        """dN/dx for left triangle, shape (3 nodes, 3 quad pts)."""
        return self._dN_left_dx

    @property
    def dN_right_dx(self) -> NDArray:
        """dN/dx for right triangle, shape (3 nodes, 3 quad pts)."""
        return self._dN_right_dx

    @property
    def dN_left_dy(self) -> NDArray:
        """dN/dy for left triangle, shape (3 nodes, 3 quad pts)."""
        return self._dN_left_dy

    @property
    def dN_right_dy(self) -> NDArray:
        """dN/dy for right triangle, shape (3 nodes, 3 quad pts)."""
        return self._dN_right_dy

    # =========================================================================
    # muGrid Operators (for field interpolation)
    # =========================================================================

    @cached_property
    def interpolation_operator(self) -> GenericLinearOperator:
        """Operator to interpolate nodal values to 6 quadrature points.

        Interpolates from 4 corner nodes (bl, tl, br, tr) to 6 quad points
        (3 per triangle). Left triangle uses bl, tl, br; right uses tr, br, tl.

        Kernel layout: kernel[quad_pt, 1, 1, x_offset, y_offset]
        Node positions: bl=(0,0), tl=(0,1), br=(1,0), tr=(1,1)
        """
        B = self.BARY_COORDS
        # Quad point order uses [0, 2, 1] permutation of BARY_COORDS
        # Left triangle: [bl, tl, br] -> barycentric [0, 1, 2]
        # Right triangle: [tr, br, tl] -> barycentric [0, 1, 2]
        # Kernel format: [[[bl, tl], [br, tr]]]
        kernel = np.array([
            [
                # Left triangle quad points (bl, tl, br active; tr=0)
                [[[B[0, 0], B[0, 1]], [B[0, 2], 0]]],  # quad0: BARY[0]
                [[[B[2, 0], B[2, 1]], [B[2, 2], 0]]],  # quad1: BARY[2]
                [[[B[1, 0], B[1, 1]], [B[1, 2], 0]]],  # quad2: BARY[1]
                # Right triangle quad points (tr, br, tl active; bl=0)
                [[[0, B[0, 2]], [B[0, 1], B[0, 0]]]],  # quad3: BARY[0]
                [[[0, B[2, 2]], [B[2, 1], B[2, 0]]]],  # quad4: BARY[2]
                [[[0, B[1, 2]], [B[1, 1], B[1, 0]]]],  # quad5: BARY[1]
            ]
        ])
        return GenericLinearOperator([0, 0], kernel)

    @cached_property
    def dx_operator(self) -> GenericLinearOperator:
        """Derivative operator d/dx for linear triangular elements.

        Returns 2 values per element (one per triangle).
        Left triangle: d/dx = (F_br - F_bl) / dx
        Right triangle: d/dx = (F_tr - F_tl) / dx
        """
        dx = self.dx
        kernel = np.array([
            [
                [[[-1 / dx, 0], [1 / dx, 0]]],      # Left: -bl + br
                [[[0, -1 / dx], [0, 1 / dx]]],      # Right: -tl + tr
            ]
        ])
        return GenericLinearOperator([0, 0], kernel)

    @cached_property
    def dy_operator(self) -> GenericLinearOperator:
        """Derivative operator d/dy for linear triangular elements.

        Returns 2 values per element (one per triangle).
        Left triangle: d/dy = (F_tl - F_bl) / dy
        Right triangle: d/dy = (F_tr - F_br) / dy
        """
        dy = self.dy
        kernel = np.array([
            [
                [[[-1 / dy, 1 / dy], [0, 0]]],      # Left: -bl + tl
                [[[0, 0], [-1 / dy, 1 / dy]]],      # Right: -br + tr
            ]
        ])
        return GenericLinearOperator([0, 0], kernel)


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


def get_active_terms(fem_solver: dict) -> list[NonLinearTerm]:

    from .terms_2d import term_list
    term_list_res = []
    for term in term_list:
        if term.name in fem_solver['equations']['term_list']:
            term_list_res.append(term)
        else:
            pass
    return term_list_res
