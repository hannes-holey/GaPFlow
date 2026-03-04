from functools import cached_property

import numpy as np
from muGrid import GenericLinearOperator
import numpy.typing as npt

NDArray = npt.NDArray[np.floating]


class TriangleQuadrature:
    """Quadrature data for linear triangular elements on a structured square grid.

    Each square cell has corners [bl=0, br=1, tl=2, tr=3] and is split
    into two triangles by the anti-diagonal (tl--br):

        tl --- tr        2 ---- 3
        |\\ t=1|         |\\ 1  |
        | \\   |    =    | \\   |
        |  \\  |         |  \\  |
        |t=0\\ |         | 0 \\ |
        bl --- br        0 ---- 1

    All quantities are derived from the five specifications marked
    "source of truth" below.  Triangle-indexed arrays use t=0 (left)
    and t=1 (right).
    """

    # ================================================================
    # Sources of truth
    # ================================================================

    TRI_PTS = np.array([
        [0, 1, 2],   # left:  bl, br, tl
        [3, 2, 1],   # right: tr, tl, br
    ])

    # Nodes that contribute to d/dx and d/dy for each triangle, and their signs.
    DERIV_NODES_DX = [0, 1]
    DERIV_NODES_DY = [0, 2]
    DERIV_SIGNS = np.array([+1, -1])

    # 3-point Gauss quadrature (barycentric coords on reference triangle).
    # Row q = weights for [node0, node1, node2] at quadrature point q.
    BARY_COORDS = np.array([
        [2 / 3, 1 / 6, 1 / 6],
        [1 / 6, 2 / 3, 1 / 6],
        [1 / 6, 1 / 6, 2 / 3],
    ])
    WEIGHTS = np.array([1 / 6, 1 / 6, 1 / 6])

    # 7-point stencil: self + 6 neighbors (cardinal + anti-diagonal).
    # Main-diagonal neighbors (-1,-1) and (1,1) are NOT connected.
    STENCIL_OFFSETS = [
        (0, 0),           # self
        (-1, 0), (1, 0),  # horizontal
        (0, -1), (0, 1),  # vertical
        (-1, 1), (1, -1),  # anti-diagonal
    ]

    # ================================================================

    def __init__(self, dx: float = 1.0, dy: float = 1.0):
        self.dx = dx
        self.dy = dy

    # ================================================================
    # Shape functions
    # ================================================================

    @cached_property
    def N(self) -> NDArray:
        """Shape functions N[node, quad_pt], shape (3, 3).

        Same for both triangles (barycentric coords are defined relative
        to each triangle's own node ordering).
        """
        return self.BARY_COORDS.T.copy()

    @cached_property
    def dN_dx(self) -> NDArray:
        """dN/dx[tri, node, quad_pt], shape (2, 3, 3).

        Constant per triangle (linear elements).  Only nodes 0, 1 nonzero.
        """
        dN = np.zeros((2, 3, 3))
        for t in range(2):
            s = self.DERIV_SIGNS[t]
            dN[t, 0] = -s / self.dx
            dN[t, 1] = +s / self.dx
        return dN

    @cached_property
    def dN_dy(self) -> NDArray:
        """dN/dy[tri, node, quad_pt], shape (2, 3, 3).

        Only nodes 0, 2 nonzero.
        """
        dN = np.zeros((2, 3, 3))
        for t in range(2):
            s = self.DERIV_SIGNS[t]
            dN[t, 0] = -s / self.dy
            dN[t, 2] = +s / self.dy
        return dN

    @property
    def weights(self) -> NDArray:
        """Quadrature weights, shape (3,)."""
        return self.WEIGHTS

    # ================================================================
    # Element tensors  (tangent matrix assembly)
    # ================================================================

    @cached_property
    def elem_tensor(self) -> NDArray:
        """N_i * N_j * w * A,  shape (2, 3, 3, 3) = [tri, i, j, q].

        Identical for both triangles.
        """
        A = self.dx * self.dy
        single = self.N[:, None, :] * self.N[None, :, :] * self.WEIGHTS * A
        return np.stack([single, single])

    @cached_property
    def elem_tensor_testfun_dx(self) -> NDArray:
        """(dN_i/dx) * N_j * w * A,  shape (2, 3, 3, 3)."""
        A = self.dx * self.dy
        return np.stack([
            self.dN_dx[t, :, None, :] * self.N[None, :, :] * self.WEIGHTS * A
            for t in range(2)
        ])

    @cached_property
    def elem_tensor_testfun_dy(self) -> NDArray:
        """(dN_i/dy) * N_j * w * A,  shape (2, 3, 3, 3)."""
        A = self.dx * self.dy
        return np.stack([
            self.dN_dy[t, :, None, :] * self.N[None, :, :] * self.WEIGHTS * A
            for t in range(2)
        ])

    # ================================================================
    # Test function integrals  (residual vector assembly)
    # ================================================================

    @cached_property
    def test_wA(self) -> NDArray:
        """N_i * w * A,  shape (2, 3, 3) = [tri, node, q].

        Identical for both triangles.
        """
        single = self.N * self.WEIGHTS * self.dx * self.dy
        return np.stack([single, single])

    @cached_property
    def test_wA_dx(self) -> NDArray:
        """(dN_i/dx) * w * A,  shape (2, 3, 3)."""
        wA = self.WEIGHTS * self.dx * self.dy
        return np.stack([self.dN_dx[t] * wA for t in range(2)])

    @cached_property
    def test_wA_dy(self) -> NDArray:
        """(dN_i/dy) * w * A,  shape (2, 3, 3)."""
        wA = self.WEIGHTS * self.dx * self.dy
        return np.stack([self.dN_dy[t] * wA for t in range(2)])

    # ================================================================
    # muGrid operators  (field interpolation)
    # ================================================================

    @cached_property
    def interpolation_operator(self) -> GenericLinearOperator:
        """Interpolate 4 corner values -> 6 quadrature point values.

        Quad points 0..2 = triangle 0,  3..5 = triangle 1.
        Kernel[1, 6, 1, 2, 2]: weight at grid offset (x, y) per quad point.
        """
        kernel = np.zeros((1, 6, 1, 2, 2))
        for t in range(2):
            for q in range(3):
                for k in range(3):
                    c = self.TRI_PTS[t, k]
                    kernel[0, t * 3 + q, 0, c % 2, c // 2] = self.N[k, q]
        return GenericLinearOperator([0, 0], kernel)

    @cached_property
    def dx_operator(self) -> GenericLinearOperator:
        """d/dx operator -> 2 values per cell (one per triangle).

        Kernel[1, 2, 1, 2, 2].
        """
        kernel = np.zeros((1, 2, 1, 2, 2))
        for t in range(2):
            s = self.DERIV_SIGNS[t]
            c0, c1 = self.TRI_PTS[t, self.DERIV_NODES_DX]
            kernel[0, t, 0, c0 % 2, c0 // 2] = -s / self.dx
            kernel[0, t, 0, c1 % 2, c1 // 2] = +s / self.dx
        return GenericLinearOperator([0, 0], kernel)

    @cached_property
    def dy_operator(self) -> GenericLinearOperator:
        """d/dy operator -> 2 values per cell (one per triangle).

        Kernel[1, 2, 1, 2, 2].
        """
        kernel = np.zeros((1, 2, 1, 2, 2))
        for t in range(2):
            s = self.DERIV_SIGNS[t]
            c0, c2 = self.TRI_PTS[t, self.DERIV_NODES_DY]
            kernel[0, t, 0, c0 % 2, c0 // 2] = -s / self.dy
            kernel[0, t, 0, c2 % 2, c2 // 2] = +s / self.dy
        return GenericLinearOperator([0, 0], kernel)
