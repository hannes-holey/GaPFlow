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

from functools import cached_property
import numpy as np
from muGrid import GenericLinearOperator
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class TaylorHoodQ2Q1:

    """
    =================================
    Q1
    =================================

    square to node index mapping:
    2 ----- 3
    |       |
    |       |
    0 ----- 1

    =================================
    Q2
    =================================

    square to node index mapping:
    2 --- 7 --- 3
    |           |
    4     6     8
    |           |
    0 --- 5 --- 1
    """

    left_idx = 0
    right_idx = 1

    # ===============================================================
    # Stencils
    # ===============================================================

    # All u-u combinations. p-* combinations are derived by checking even points.
    # even/odd refer to the target node's indices (even-even coincides with P1 node).
    stencil_even_even = [
        (0, 0),
        (-2, 0), (-1, 0), (1, 0), (2, 0),  # straight
        (0, -2), (0, -1), (0, 1), (0, 2),
        (-2, -2), (-1, -1), (1, 1), (2, 2),  # diagonals
        (-2, 2), (-1, 1), (1, -1), (2, -2),
        (-2, 1), (-1, 2), (1, 2), (2, 1),  # knight moves
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    stencil_odd_odd = [  # middle Q2 point
        (0, 0),
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (1, 1),
        (-1, 1), (1, -1)
    ]
    stencil_even_odd = [  # on a vertical Q1 connection line
        (0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0),
        (0, -1), (-1, -1), (1, -1), (-2, -1), (2, -1),
        (0, 1), (-1, 1), (1, 1), (-2, 1), (2, 1)
    ]
    stencil_odd_even = [  # on a horizontal Q1 connection line
        (0, 0), (0, -1), (0, 1), (0, -2), (0, 2),
        (-1, 0), (-1, -1), (-1, 1), (-1, -2), (-1, 2),
        (1, 0), (1, -1), (1, 1), (1, -2), (1, 2)
    ]

    def __init__(self, dx: float, dy: float):
        self.dx = dx
        self.dy = dy
        self.sq_area = dx * dy
        self.Quadrature = Quadrature3x3Points()
        self.Q1 = self.Q1(self.Quadrature)
        self.Q2 = self.Q2(self.Quadrature)

    class Q1:

        """Bilinear interpolation on the full square. Used for nodal fields
        that are not Newton solution variables and which should not have a
        diagonal-orientation bias (e.g. h, dh, eta). Specified in fieldspec.py.

        square node index mapping:
        2 ----- 3
        |       |
        |       |
        0 ----- 1
        """

        nodes_per_element = 4

        square_node_offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

        _N_funcs = [None] * 4
        _N_funcs[0] = lambda x, y: (1 - x) * (1 - y)
        _N_funcs[1] = lambda x, y: x * (1 - y)
        _N_funcs[2] = lambda x, y: (1 - x) * y
        _N_funcs[3] = lambda x, y: x * y

        _N_x_funcs = [None] * 4
        _N_x_funcs[0] = lambda x, y: -(1 - y)
        _N_x_funcs[1] = lambda x, y: (1 - y)
        _N_x_funcs[2] = lambda x, y: -y
        _N_x_funcs[3] = lambda x, y: y

        _N_y_funcs = [None] * 4
        _N_y_funcs[0] = lambda x, y: -(1 - x)
        _N_y_funcs[1] = lambda x, y: -x
        _N_y_funcs[2] = lambda x, y: (1 - x)
        _N_y_funcs[3] = lambda x, y: x

        def __init__(self, quadrature):
            self.quadrature = quadrature

        def _eval_funcs(self, funcs) -> NDArray:
            """Returns shape (nb_quad, 4): shape functions evaluated directly
            at the square's own quadrature points."""
            return np.array([[f(x, y) for f in funcs]
                             for x, y in self.quadrature.coordinates])

        @cached_property
        def N(self) -> NDArray:
            return self._eval_funcs(self._N_funcs)

        @cached_property
        def dN_dx(self) -> NDArray:
            return self._eval_funcs(self._N_x_funcs)

        @cached_property
        def dN_dy(self) -> NDArray:
            return self._eval_funcs(self._N_y_funcs)

        def _make_operator(self, dN: NDArray) -> "QuadOperator":
            """Build a QuadOperator for a given shape function matrix dN
            (nb_quad, 4).
            Input shape:  (ny+1, nx+1)
            Output shape: (nb_quad, ny, nx)
            """
            offsets = self.square_node_offsets
            nb_quad, nb_nodes = dN.shape

            def numpy_fn(input_field, output_field):
                pg = input_field.pg
                nodal = pg[0] if pg.ndim == 3 else pg
                out_pg = output_field.pg
                _, nx_pad, ny_pad = out_pg.shape
                nx = nx_pad - 1
                ny = ny_pad - 1

                node_vals = np.empty((nb_nodes, nx, ny))
                for k in range(nb_nodes):
                    ox, oy = offsets[k]
                    node_vals[k] = nodal[ox:nx + ox, oy:ny + oy]

                result = np.einsum('qk, krc -> qrc', dN, node_vals)
                out_pg[:, :nx, :ny] = result.reshape(nb_quad, nx, ny)

            return QuadOperator(None, numpy_fn=numpy_fn, backend='numpy')

        @property
        def weights(self) -> NDArray:
            return self.quadrature.weights

        @cached_property
        def interpolation_operator(self) -> "QuadOperator":
            return self._make_operator(self.N)

        @cached_property
        def dx_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dx)

        @cached_property
        def dy_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dy)

    class Q2:

        """Biquadratic interpolation on the full square. Tensor-product
        counterpart to Q1, for use on the fine (Q2-spaced) grid.

        square node index mapping:
        2 --- 7 --- 3
        |           |
        4     6     8
        |           |
        0 --- 5 --- 1
        """

        nodes_per_element = 9

        square_node_offsets = np.array([[0, 0], [2, 0], [0, 2], [2, 2],
                                        [0, 1], [1, 0], [1, 1],
                                        [1, 2], [2, 1]])

        _N_funcs = [None] * 9
        _N_funcs[0] = lambda x, y: (1 - x) * (1 - 2 * x) * (1 - y) * (1 - 2 * y)
        _N_funcs[1] = lambda x, y: x * (2 * x - 1) * (1 - y) * (1 - 2 * y)
        _N_funcs[2] = lambda x, y: (1 - x) * (1 - 2 * x) * y * (2 * y - 1)
        _N_funcs[3] = lambda x, y: x * (2 * x - 1) * y * (2 * y - 1)
        _N_funcs[4] = lambda x, y: (1 - x) * (1 - 2 * x) * 4 * y * (1 - y)
        _N_funcs[5] = lambda x, y: 4 * x * (1 - x) * (1 - y) * (1 - 2 * y)
        _N_funcs[6] = lambda x, y: 4 * x * (1 - x) * 4 * y * (1 - y)
        _N_funcs[7] = lambda x, y: 4 * x * (1 - x) * y * (2 * y - 1)
        _N_funcs[8] = lambda x, y: x * (2 * x - 1) * 4 * y * (1 - y)

        _N_x_funcs = [None] * 9
        _N_x_funcs[0] = lambda x, y: (4 * x - 3) * (1 - y) * (1 - 2 * y)
        _N_x_funcs[1] = lambda x, y: (4 * x - 1) * (1 - y) * (1 - 2 * y)
        _N_x_funcs[2] = lambda x, y: (4 * x - 3) * y * (2 * y - 1)
        _N_x_funcs[3] = lambda x, y: (4 * x - 1) * y * (2 * y - 1)
        _N_x_funcs[4] = lambda x, y: (4 * x - 3) * 4 * y * (1 - y)
        _N_x_funcs[5] = lambda x, y: (4 - 8 * x) * (1 - y) * (1 - 2 * y)
        _N_x_funcs[6] = lambda x, y: (4 - 8 * x) * 4 * y * (1 - y)
        _N_x_funcs[7] = lambda x, y: (4 - 8 * x) * y * (2 * y - 1)
        _N_x_funcs[8] = lambda x, y: (4 * x - 1) * 4 * y * (1 - y)

        _N_y_funcs = [None] * 9
        _N_y_funcs[0] = lambda x, y: (1 - x) * (1 - 2 * x) * (4 * y - 3)
        _N_y_funcs[1] = lambda x, y: x * (2 * x - 1) * (4 * y - 3)
        _N_y_funcs[2] = lambda x, y: (1 - x) * (1 - 2 * x) * (4 * y - 1)
        _N_y_funcs[3] = lambda x, y: x * (2 * x - 1) * (4 * y - 1)
        _N_y_funcs[4] = lambda x, y: (1 - x) * (1 - 2 * x) * (4 - 8 * y)
        _N_y_funcs[5] = lambda x, y: 4 * x * (1 - x) * (4 * y - 3)
        _N_y_funcs[6] = lambda x, y: 4 * x * (1 - x) * (4 - 8 * y)
        _N_y_funcs[7] = lambda x, y: 4 * x * (1 - x) * (4 * y - 1)
        _N_y_funcs[8] = lambda x, y: x * (2 * x - 1) * (4 - 8 * y)

        def __init__(self, quadrature):
            self.quadrature = quadrature

        def _eval_funcs(self, funcs) -> NDArray:
            """Returns shape (nb_quad, 9): shape functions evaluated directly
            at the square's own quadrature points."""
            return np.array([[f(x, y) for f in funcs]
                             for x, y in self.quadrature.coordinates])

        @cached_property
        def N(self) -> NDArray:
            return self._eval_funcs(self._N_funcs)

        @cached_property
        def dN_dx(self) -> NDArray:
            return self._eval_funcs(self._N_x_funcs)

        @cached_property
        def dN_dy(self) -> NDArray:
            return self._eval_funcs(self._N_y_funcs)

        def _make_operator(self, dN: NDArray) -> "QuadOperator":
            """Build a QuadOperator for a given shape function matrix dN
            (nb_quad, 9).
            Input shape:  (ny_fine, nx_fine)  where ny_fine=2*ny+1, nx_fine=2*nx+1
            Output shape: (nb_quad, ny, nx)
            """
            offsets = self.square_node_offsets
            nb_quad, nb_nodes = dN.shape

            def numpy_fn(input_field, output_field):
                pg = input_field.pg
                nodal = pg[0] if pg.ndim == 3 else pg
                out_pg = output_field.pg
                _, nx_pad, ny_pad = out_pg.shape
                nx = nx_pad - 1
                ny = ny_pad - 1

                node_vals = np.empty((nb_nodes, nx, ny))
                for k in range(nb_nodes):
                    ox, oy = offsets[k]
                    node_vals[k] = nodal[ox:2 * nx + ox:2, oy:2 * ny + oy:2]

                result = np.einsum('qk, krc -> qrc', dN, node_vals)
                out_pg[:, :nx, :ny] = result.reshape(nb_quad, nx, ny)

            return QuadOperator(None, numpy_fn=numpy_fn, backend='numpy')

        @property
        def weights(self) -> NDArray:
            return self.quadrature.weights

        @cached_property
        def interpolation_operator(self) -> "QuadOperator":
            return self._make_operator(self.N)

        @cached_property
        def dx_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dx)

        @cached_property
        def dy_operator(self) -> "QuadOperator":
            return self._make_operator(self.dN_dy)


class Quadrature3x3Points:
    """Tensor-product 3x3 Gauss-Legendre rule on the reference square."""

    nb_points = 9
    _p0 = 0.5 - np.sqrt(3 / 5) / 2
    _p1 = 0.5
    _p2 = 0.5 + np.sqrt(3 / 5) / 2
    _w0 = 5 / 18
    _w1 = 8 / 18
    _w2 = 5 / 18
    coordinates = np.array([[_p0, _p0], [_p1, _p0], [_p2, _p0],
                            [_p0, _p1], [_p1, _p1], [_p2, _p1],
                            [_p0, _p2], [_p1, _p2], [_p2, _p2]])
    weights = np.array([_w0 * _w0, _w1 * _w0, _w2 * _w0,
                        _w0 * _w1, _w1 * _w1, _w2 * _w1,
                        _w0 * _w2, _w1 * _w2, _w2 * _w2])


class QuadOperator:
    """Wraps GenericLinearOperator with an optional numpy backend.
    Right now, numpy is faster than muGrid.
    """

    def __init__(self, mugrid_op: GenericLinearOperator, numpy_fn=None,
                 backend: str = 'mugrid'):
        self._mugrid_op = mugrid_op
        self._numpy_fn = numpy_fn
        self.backend = backend

    def apply(self, input_field, output_field) -> None:
        if self.backend == 'numpy' and self._numpy_fn is not None:
            self._numpy_fn(input_field, output_field)
        else:
            self._mugrid_op.apply(input_field, output_field)
