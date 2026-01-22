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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver_fem_2d import FEMSolver2D


def apply_guards(q: np.ndarray, dq: np.ndarray, solver: "FEMSolver2D") -> np.ndarray:
    """Apply solution update with physical safeguards.

    Parameters
    ----------
    q : ndarray
        Current solution vector (nb_vars * nb_inner_pts,)
    dq : ndarray
        Newton increment (already scaled by alpha)
    solver : FEMSolver2D
        Solver instance for accessing problem parameters

    Returns
    -------
    q_new : ndarray
        Updated and guarded solution vector
    """
    q_new = q + dq

    nb = solver.nb_inner_pts
    rho_min = 1e-10

    # Clamp density to positive values
    rho = q_new[0:nb]
    rho[:] = np.maximum(rho, rho_min)

    return q_new
