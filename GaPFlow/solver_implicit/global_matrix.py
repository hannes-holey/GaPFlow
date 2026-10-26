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

import numpy as np

from .grid_index import GridIndexManager


def field_to_global(field_idx,
                    spec,
                    grid_index: GridIndexManager,
                    p1_factor: int = 1,
                    p2_factor: int = 2,
                    ):
    """Computes the unique global node index.

    p1_factor : number of active P1-grid DOF slots per P1 node.
    p2_factor : number of active P2-grid DOF slots per P2 node (2 for jx+jy;
        0 when no P2 variable is active, e.g. the scalar Reynolds equation).
    """

    decomp = grid_index.decomp
    Nx_global_p = decomp.nb_domain_grid_pts[0]
    Nx_global_P2 = decomp.nb_domain_grid_pts_P2[0]

    if spec.grid == 'P1':
        y_p, x_p = np.divmod(field_idx, Nx_global_p)
        y_P2, x_P2 = 2 * y_p, 2 * x_p
    else:
        y_P2, x_P2 = np.divmod(field_idx, Nx_global_P2)

    nb_nodes_cur_row = _get_nb_nodes_cur_row(x_P2, y_P2, p1_factor, p2_factor)
    nb_nodes_block_below = _get_nb_nodes_block_below(y_P2, Nx_global_P2, Nx_global_p, p1_factor, p2_factor)

    return nb_nodes_block_below + nb_nodes_cur_row + spec.idx


def _get_nb_nodes_cur_row(x_P2, y_P2, p1_factor, p2_factor):
    """Number of nodes in the current row BEFORE x_P2.
    If y_P2 is even: x_P2//2 additional nodes in the current row
    """
    nb_p = np.where(y_P2 % 2 == 0, (x_P2 + 1) // 2, 0)
    nb_P2 = x_P2
    return p2_factor * nb_P2 + p1_factor * nb_p


def _get_nb_nodes_block_below(y_P2, Nx_global_P2, Nx_global_p, p1_factor, p2_factor):
    """Number of nodes in all rows BELOW y_P2.
    Each row has cols_P2 nodes, and each even row has cols_p//2 additional nodes.
     - if r is odd: no additional nodes in the current row
     - if r is even: c//2 additional nodes in the current row
    """
    nb_P2 = y_P2 * Nx_global_P2
    rows_p_below = (y_P2 + 1) // 2
    nb_p = rows_p_below * Nx_global_p
    return p2_factor * nb_P2 + p1_factor * nb_p
