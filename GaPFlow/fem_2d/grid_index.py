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
"""Grid index management for FEM assembly.

Handles index masks, element connectivity, and stencil patterns for
triangular FEM on structured grids with domain decomposition support.
"""
from functools import cached_property, lru_cache
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..domain_decomposition import DomainDecomposition

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]


class GridIndexManager:
    """Manages grid index masks and element connectivity for FEM assembly.

    Handles:
    - Index masks for inner (residual) and padded (contributor) grids
    - Square element coordinates and corner connectivity
    - 7-point stencil connectivity for sparse matrix pattern

    Parameters
    ----------
    Nx_inner, Ny_inner : int
        Number of inner grid points (excluding ghost cells).
    bc_at_W, bc_at_E, bc_at_S, bc_at_N : bool
        True if subdomain is at domain boundary AND not periodic.
    decomp : DomainDecomposition
        Domain decomposition object for periodic/MPI info.
    variables : list of str
        Variable names ['rho', 'jx', 'jy', ...] for Neumann BC lookup.
    bc_neumann : dict
        Neumann BC flags per boundary, e.g. {'xW': [F,T,T], 'xE': [F,T,T], ...}.
    """

    # 7-point stencil offsets (excludes main diagonal corners)
    STENCIL_OFFSETS = [
        (0, 0),            # self
        (-1, 0), (1, 0),   # horizontal
        (0, -1), (0, 1),   # vertical
        (-1, 1), (1, -1),  # anti-diagonal (connected via triangles)
    ]

    def __init__(self, Nx_inner: int, Ny_inner: int,
                 bc_at_W: bool, bc_at_E: bool, bc_at_S: bool, bc_at_N: bool,
                 decomp: "DomainDecomposition",
                 variables: List[str], bc_neumann: dict):
        self.Nx_inner = Nx_inner
        self.Ny_inner = Ny_inner
        self.Nx_padded = Nx_inner + 2
        self.Ny_padded = Ny_inner + 2

        self.bc_at_W = bc_at_W
        self.bc_at_E = bc_at_E
        self.bc_at_S = bc_at_S
        self.bc_at_N = bc_at_N

        self._decomp = decomp
        self._variables = variables
        self._bc_neumann = bc_neumann

        # Derived quantities
        self.nb_inner_pts = Nx_inner * Ny_inner
        self.sq_per_row = Nx_inner + 1
        self.sq_per_col = Ny_inner + 1
        self.nb_sq = self.sq_per_row * self.sq_per_col

        # Square coordinate arrays
        sq_idx = np.arange(self.nb_sq)
        self.sq_x_arr = sq_idx % self.sq_per_row
        self.sq_y_arr = sq_idx // self.sq_per_row

    def is_bc_point(self, x: int, y: int) -> bool:
        """Check if point at (x, y) is a Dirichlet boundary condition point."""
        if (x == 0 and self.bc_at_W) or (x == self.Nx_padded - 1 and self.bc_at_E):
            return True
        if (y == 0 and self.bc_at_S) or (y == self.Ny_padded - 1 and self.bc_at_N):
            return True
        return False

    @cached_property
    def index_mask_inner_local(self) -> IntArray:
        """Local indices for inner (residual/TO) points only.

        Returns mask of shape (Nx_padded, Ny_padded) where:
        - Inner points have sequential indices 0..nb_inner_pts-1 (column-major)
        - Ghost/boundary points have -1
        """
        mask = np.full((self.Nx_padded, self.Ny_padded), -1, dtype=np.int32)
        inner_shape = (self.Nx_inner, self.Ny_inner)
        mask[1:-1, 1:-1] = np.arange(self.nb_inner_pts).reshape(inner_shape, order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local(self, var: str = '') -> IntArray:
        """Local indices for all contributor (FROM) points with BC handling.

        Parameters
        ----------
        var : str, optional
            Variable name for Neumann BC forwarding. If empty, no Neumann handling.

        Returns mask of shape (Nx_padded, Ny_padded) where:
        - Inner points retain indices from index_mask_inner_local
        - Periodic ghost cells map to corresponding inner points
        - Non-periodic ghost cells get new sequential indices
        - Dirichlet BC points remain -1
        - Neumann BC points forward to interior neighbor
        """
        mask = self.index_mask_inner_local.copy()
        decomp = self._decomp

        # Periodic wrapping for ghost cells when full extent is owned
        if decomp.periodic_x and decomp.has_full_x:
            mask[0, :] = mask[self.Nx_padded - 2, :]
            mask[self.Nx_padded - 1, :] = mask[1, :]

        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_padded - 2]
            mask[:, self.Ny_padded - 1] = mask[:, 1]

        # Assign new indices to remaining valid ghost cells
        cur_val = self.nb_inner_pts
        for x in range(self.Nx_padded):
            for y in range(self.Ny_padded):
                if mask[x, y] == -1 and not self.is_bc_point(x, y):
                    mask[x, y] = cur_val
                    cur_val += 1

        # Neumann BC forwarding (ghost forwards to interior neighbor)
        if var:
            var_idx = self._variables.index(var)
            if self.bc_at_W and self._bc_neumann['xW'][var_idx]:
                mask[0, :] = mask[1, :]
            if self.bc_at_E and self._bc_neumann['xE'][var_idx]:
                mask[self.Nx_padded - 1, :] = mask[self.Nx_padded - 2, :]
            if self.bc_at_S and self._bc_neumann['yS'][var_idx]:
                mask[:, 0] = mask[:, 1]
            if self.bc_at_N and self._bc_neumann['yN'][var_idx]:
                mask[:, self.Ny_padded - 1] = mask[:, self.Ny_padded - 2]

        return mask

    @cached_property
    def nb_contributors(self) -> int:
        """Number of unique contributor indices (inner + valid ghost points).

        Accounts for periodic wrapping where ghost points reuse inner indices.
        """
        mask = self.index_mask_padded_local('')
        valid_indices = mask[mask >= 0]
        return int(np.max(valid_indices)) + 1 if len(valid_indices) > 0 else 0

    @cached_property
    def sq_TO_inner(self) -> IntArray:
        """Inner (residual) indices for all square corners.

        Returns shape (nb_sq, 4) for corners [bl, br, tl, tr].
        Value is -1 for corners outside inner domain.

        Note: Periodic BCs are handled implicitly via muGrid ghost cell
        communication, NOT through index wrapping here. Ghost cells at
        periodic boundaries receive values from the opposite domain side
        before field interpolation/derivatives are computed. We only
        assemble residuals for inner points; ghost points provide neighbor
        information for derivative computations, not residual equations.
        """
        m = self.index_mask_inner_local
        sx, sy = self.sq_x_arr, self.sq_y_arr
        return np.column_stack([
            m[sx, sy],         # bl
            m[sx + 1, sy],     # br
            m[sx, sy + 1],     # tl
            m[sx + 1, sy + 1]  # tr
        ])

    @lru_cache(maxsize=None)
    def sq_FROM_padded(self, var: str) -> IntArray:
        """Padded (contributor) indices for all square corners.

        Parameters
        ----------
        var : str
            Variable name for Neumann BC handling.

        Returns shape (nb_sq, 4) for corners [bl, br, tl, tr].
        """
        m = self.index_mask_padded_local(var)
        sx, sy = self.sq_x_arr, self.sq_y_arr
        return np.column_stack([
            m[sx, sy],         # bl
            m[sx + 1, sy],     # br
            m[sx, sy + 1],     # tl
            m[sx + 1, sy + 1]  # tr
        ])

    def get_stencil_connectivity(self) -> Tuple[IntArray, IntArray]:
        """Get FEM stencil connectivity as parallel arrays.

        Returns (inner_pts, contrib_pts) where inner_pts[i] connects to
        contrib_pts[i] in the sparse matrix pattern.

        Uses 7-point stencil (self + 6 neighbors). Main diagonal corners
        (-1,-1) and (1,1) are excluded since triangular elements don't
        connect them.
        """
        m_inner = self.index_mask_inner_local
        m_padded = self.index_mask_padded_local('')

        inner_list = []
        contrib_list = []

        inner_pts_2d = np.argwhere(m_inner >= 0)

        for dx, dy in self.STENCIL_OFFSETS:
            nx = inner_pts_2d[:, 0] + dx
            ny = inner_pts_2d[:, 1] + dy

            inner_idx = m_inner[inner_pts_2d[:, 0], inner_pts_2d[:, 1]]
            contrib_idx = m_padded[nx, ny]

            valid = contrib_idx >= 0
            inner_list.append(inner_idx[valid])
            contrib_list.append(contrib_idx[valid])

        return np.concatenate(inner_list), np.concatenate(contrib_list)
