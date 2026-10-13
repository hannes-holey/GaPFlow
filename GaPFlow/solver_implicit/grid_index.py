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

from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..parallel import DomainDecomposition
    from .solver_implicit import ImplicitSolver

NDArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.signedinteger]


class GridIndexManager:
    """Index masks and square connectivity for P2P1 assembly.

    Parameters
    ----------
    decomp : DomainDecomposition
        Must have been initialised with solver='fem'.
    variables : list of str
        Ordered variable names, e.g. ['rho', 'jx', 'jy']. Used to look up
        per-variable BC types.
    """

    def __init__(self,
                 decomp: "DomainDecomposition",
                 fem_solver: "ImplicitSolver"):

        self.decomp = decomp
        self.fem_solver = fem_solver

        # node counts
        Nx_P1, Ny_P1 = decomp.nb_subdomain_grid_pts
        self.Nx_P1_inner = Nx_P1
        self.Ny_P1_inner = Ny_P1
        self.Nx_P1_padded = Nx_P1 + 2
        self.Ny_P1_padded = Ny_P1 + 2

        Nx_P2, Ny_P2 = decomp.nb_subdomain_grid_pts_P2
        self.Nx_P2_inner = Nx_P2
        self.Ny_P2_inner = Ny_P2
        self.Nx_P2_padded = Nx_P2 + 4
        self.Ny_P2_padded = Ny_P2 + 4

        # square counts
        self.sq_per_row = self.Nx_P1_padded - 1
        self.sq_per_col = self.Ny_P1_padded - 1
        self.nb_sq = self.sq_per_row * self.sq_per_col
        sq_idx = np.arange(self.nb_sq)

        self.sq_x_arr_P1 = sq_idx % self.sq_per_row
        self.sq_y_arr_P1 = sq_idx // self.sq_per_row

        self.sq_x_arr_P2 = self.sq_x_arr_P1 * 2
        self.sq_y_arr_P2 = self.sq_y_arr_P1 * 2

    # ======================================================================
    # Helper
    # ======================================================================

    def is_neumann(self, side: str, var: str) -> bool:
        """Check if a variable has Neumann BC on a given side."""

        idx = self.fem_solver.variables.index(var)
        return self.fem_solver.bc_specs[idx].get_bc_type(side) == 'N'

    # ======================================================================
    # P1 index masks
    # ======================================================================

    @cached_property
    def index_mask_inner_local_P1(self) -> IntArray:
        """Local indices for inner density nodes, shape (Nx_P1_padded, Ny_P1_padded).

        Inner nodes: sequential 0..Nx_p*Ny_p-1 (row-major, x varies fastest).
        Ghost/boundary nodes: -1.
        """
        mask = np.full((self.Nx_P1_padded, self.Ny_P1_padded), -1, dtype=np.int32)
        nb_inner = self.Nx_P1_inner * self.Ny_P1_inner
        mask[1:-1, 1:-1] = np.arange(nb_inner).reshape(
            (self.Nx_P1_inner, self.Ny_P1_inner), order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local_P1(self, var: str = '') -> IntArray:
        """Local contributor indices for mass flux grid, shape (Nx_P2_padded, Ny_P2_padded).

        - serial + periodic: wrap around (both ghost layers)
        - inter-subdomain ghost nodes: new sequential indices
        - Neumann ghost nodes: forward both layers to adjacent inner node

        Effectively, only Dirichlet ghost nodes remain -1.

        Parameters
        ----------
        var : str
            Variable name for Neumann forwarding. If empty, no forwarding.
        """
        mask = self.index_mask_inner_local_P1.copy()
        decomp = self.decomp

        # periodic AND serial -> wrap around
        if decomp.periodic_x and decomp.has_full_x:
            mask[0, :] = mask[self.Nx_P1_padded - 2, :]
            mask[self.Nx_P1_padded - 1, :] = mask[1, :]
        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_P1_padded - 2]
            mask[:, self.Ny_P1_padded - 1] = mask[:, 1]

        # fill inter-subdomain ghost nodes with new indices (non-boundary)
        boundary_P1 = np.zeros((self.Nx_P1_padded, self.Ny_P1_padded), dtype=bool)
        if decomp.bc_at_W:
            boundary_P1[0, :] = True
        if decomp.bc_at_E:
            boundary_P1[-1, :] = True
        if decomp.bc_at_S:
            boundary_P1[:, 0] = True
        if decomp.bc_at_N:
            boundary_P1[:, -1] = True
        ghost_coords = np.argwhere((mask == -1) & ~boundary_P1)
        nb_inner_P1 = self.Nx_P1_inner * self.Ny_P1_inner
        mask[ghost_coords[:, 0], ghost_coords[:, 1]] = np.arange(
            nb_inner_P1, nb_inner_P1 + len(ghost_coords), dtype=np.int32)

        # Neumann forwarding
        if var:
            if decomp.bc_at_W and self.is_neumann('W', var):
                mask[0, :] = mask[1, :]
            if decomp.bc_at_E and self.is_neumann('E', var):
                mask[self.Nx_P1_padded - 1, :] = mask[self.Nx_P1_padded - 2, :]
            if decomp.bc_at_S and self.is_neumann('S', var):
                mask[:, 0] = mask[:, 1]
            if decomp.bc_at_N and self.is_neumann('N', var):
                mask[:, self.Ny_P1_padded - 1] = mask[:, self.Ny_P1_padded - 2]

        return mask

    @cached_property
    def nb_contributors_P1(self) -> int:
        """Number of unique contributor indices on the density grid."""
        mask = self.index_mask_padded_local_P1()
        valid = mask[mask >= 0]
        return int(np.max(valid)) + 1

    @cached_property
    def l2g_list_P1(self) -> IntArray:
        """Local-to-global mapping for density contributors, shape (nb_contributors_P1,)."""
        mask_local = self.index_mask_padded_local_P1()
        mask_global = self.decomp.index_mask_padded_global
        l2g = np.zeros(self.nb_contributors_P1, dtype=np.int32)
        valid = mask_local >= 0
        l2g[mask_local[valid]] = mask_global[valid]
        return l2g

    # ======================================================================
    # P2 index masks
    # ======================================================================

    @cached_property
    def index_mask_inner_local_P2(self) -> IntArray:
        """Local indices for inner mass flux nodes, shape (Nx_P2_padded, Ny_P2_padded).

        Inner nodes: sequential 0..Nx_P2*Ny_P2-1 (row-major, x varies fastest).
        Ghost/boundary nodes: -1.
        """
        mask = np.full((self.Nx_P2_padded, self.Ny_P2_padded), -1, dtype=np.int32)
        nb_inner = self.Nx_P2_inner * self.Ny_P2_inner
        mask[2:-2, 2:-2] = np.arange(nb_inner).reshape(
            (self.Nx_P2_inner, self.Ny_P2_inner), order='F')
        return mask

    @lru_cache(maxsize=None)
    def index_mask_padded_local_P2(self, var: str = '') -> IntArray:
        """Local contributor indices for mass flux grid, shape (Nx_P2_padded, Ny_P2_padded).

        - serial + periodic: wrap around (both ghost layers)
        - inter-subdomain ghost nodes: new sequential indices
        - Neumann ghost nodes: forward both layers to adjacent inner node

        Effectively, only Dirichlet ghost nodes remain -1.

        Parameters
        ----------
        var : str
            Variable name for Neumann forwarding. If empty, no forwarding.
        """
        mask = self.index_mask_inner_local_P2.copy()
        decomp = self.decomp

        # periodic AND serial -> wrap around
        if decomp.periodic_x and decomp.has_full_x:
            mask[0, :] = mask[self.Nx_P2_padded - 4, :]
            mask[1, :] = mask[self.Nx_P2_padded - 3, :]
            mask[self.Nx_P2_padded - 2, :] = mask[2, :]
            mask[self.Nx_P2_padded - 1, :] = mask[3, :]
        if decomp.periodic_y and decomp.has_full_y:
            mask[:, 0] = mask[:, self.Ny_P2_padded - 4]
            mask[:, 1] = mask[:, self.Ny_P2_padded - 3]
            mask[:, self.Ny_P2_padded - 2] = mask[:, 2]
            mask[:, self.Ny_P2_padded - 1] = mask[:, 3]

        # fill inter-subdomain ghost nodes with new indices (non-boundary)
        boundary_P2 = np.zeros((self.Nx_P2_padded, self.Ny_P2_padded), dtype=bool)
        if decomp.bc_at_W:
            boundary_P2[:2, :] = True
        if decomp.bc_at_E:
            boundary_P2[-2:, :] = True
        if decomp.bc_at_S:
            boundary_P2[:, :2] = True
        if decomp.bc_at_N:
            boundary_P2[:, -2:] = True
        ghost_coords = np.argwhere((mask == -1) & ~boundary_P2)
        nb_inner_P2 = self.Nx_P2_inner * self.Ny_P2_inner
        mask[ghost_coords[:, 0], ghost_coords[:, 1]] = np.arange(
            nb_inner_P2, nb_inner_P2 + len(ghost_coords), dtype=np.int32)

        # Neumann forwarding
        if var:
            if decomp.bc_at_W and self.is_neumann('W', var):
                mask[1, :] = mask[2, :]
                mask[0, :] = mask[2, :]
            if decomp.bc_at_E and self.is_neumann('E', var):
                mask[self.Nx_P2_padded - 2, :] = mask[self.Nx_P2_padded - 3, :]
                mask[self.Nx_P2_padded - 1, :] = mask[self.Nx_P2_padded - 3, :]
            if decomp.bc_at_S and self.is_neumann('S', var):
                mask[:, 1] = mask[:, 2]
                mask[:, 0] = mask[:, 2]
            if decomp.bc_at_N and self.is_neumann('N', var):
                mask[:, self.Ny_P2_padded - 2] = mask[:, self.Ny_P2_padded - 3]
                mask[:, self.Ny_P2_padded - 1] = mask[:, self.Ny_P2_padded - 3]

        return mask

    @cached_property
    def nb_contributors_P2(self) -> int:
        """Number of unique contributor indices on the mass flux grid."""
        mask = self.index_mask_padded_local_P2()
        valid = mask[mask >= 0]
        return int(np.max(valid)) + 1

    @cached_property
    def l2g_list_P2(self) -> IntArray:
        """Local-to-global mapping for mass flux contributors, shape (nb_contributors_P2,)."""
        mask_local = self.index_mask_padded_local_P2()
        mask_global = self.decomp.index_mask_padded_global_P2
        l2g = np.zeros(self.nb_contributors_P2, dtype=np.int32)
        valid = mask_local >= 0
        l2g[mask_local[valid]] = mask_global[valid]
        return l2g

    # ======================================================================
    # Square corner connectivity
    # ======================================================================

    @cached_property
    def sq_TO_inner_P1(self) -> IntArray:
        """Residual (inner) indices for P1 square corners, shape (nb_sq, 4).

        Corner order: [bl, br, tl, tr]
        """
        m = self.index_mask_inner_local_P1
        sx = self.sq_x_arr_P1
        sy = self.sq_y_arr_P1
        return np.column_stack([
            m[sx, sy],   # bl
            m[sx + 1, sy],   # br
            m[sx, sy + 1],   # tl
            m[sx + 1, sy + 1],   # tr
        ])

    @lru_cache(maxsize=None)
    def sq_FROM_padded_P1(self, var: str) -> IntArray:
        """Contributor indices for P1 square corners, shape (nb_sq, 4).

        Parameters
        ----------
        var : str
            Variable name for Neumann BC handling.
        """
        m = self.index_mask_padded_local_P1(var)
        sx = self.sq_x_arr_P1
        sy = self.sq_y_arr_P1
        return np.column_stack([
            m[sx, sy],
            m[sx + 1, sy],
            m[sx, sy + 1],
            m[sx + 1, sy + 1],
        ])

    @cached_property
    def sq_TO_inner_P2(self) -> IntArray:
        """Residual (inner) indices for P2 square nodes, shape (nb_sq, 9).

        Corner order: [bl, br, tl, tr, ml, bm, mm, tm, mr]
        """
        m = self.index_mask_inner_local_P2
        sx = self.sq_x_arr_P2
        sy = self.sq_y_arr_P2
        return np.column_stack([
            m[sx, sy],   # bl  (0,0)
            m[sx + 2, sy],   # br  (2,0)
            m[sx, sy + 2],   # tl  (0,2)
            m[sx + 2, sy + 2],   # tr  (2,2)
            m[sx, sy + 1],   # ml  (0,1)
            m[sx + 1, sy],   # bm  (1,0)
            m[sx + 1, sy + 1],   # mm  (1,1)
            m[sx + 1, sy + 2],   # tm  (1,2)
            m[sx + 2, sy + 1],   # mr  (2,1)
        ])

    @lru_cache(maxsize=None)
    def sq_FROM_padded_P2(self, var: str) -> IntArray:
        """Contributor indices for P2 square nodes, shape (nb_sq, 9).

        Parameters
        ----------
        var : str
            Variable name for Neumann BC handling.
        """
        m = self.index_mask_padded_local_P2(var)
        sx = self.sq_x_arr_P2
        sy = self.sq_y_arr_P2
        return np.column_stack([
            m[sx, sy],
            m[sx + 2, sy],
            m[sx, sy + 2],
            m[sx + 2, sy + 2],
            m[sx, sy + 1],
            m[sx + 1, sy],
            m[sx + 1, sy + 1],
            m[sx + 1, sy + 2],
            m[sx + 2, sy + 1],
        ])
