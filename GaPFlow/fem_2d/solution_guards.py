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
from mpi4py import MPI
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solver_fem_2d import FEMSolver2D


# Guard parameters
MAX_DENSITY_CHANGE = 1.0  # Maximum relative density change per Newton step (10%)
RHO_MIN = 1e-10           # Minimum allowed density


def apply_guards(q: np.ndarray, dq: np.ndarray, solver: "FEMSolver2D") -> np.ndarray:
    """Apply solution update with physical safeguards.

    Limits density changes to MAX_DENSITY_CHANGE per Newton step by uniformly scaling dq.
    Uses MPI allreduce to ensure consistent scaling across all processes.
    """
    nb = solver.nb_inner_pts
    comm = solver.problem.decomp._mpi_comm

    # Max relative density change (only density part: first nb elements)
    rho = q[:nb]
    d_rho = dq[:nb]
    max_rel = np.max(np.abs(d_rho) / np.maximum(np.abs(rho), RHO_MIN))

    # Global max across MPI processes
    max_rel = comm.allreduce(max_rel, op=MPI.MAX)

    # Uniform scaling if density change exceeds limit
    if max_rel > MAX_DENSITY_CHANGE:
        scale = MAX_DENSITY_CHANGE / max_rel
        dq = dq * scale
        print(f"Applied density change guard: scaled dq by {scale:.3f}")

    q_new = q + dq
    q_new[:nb] = np.maximum(q_new[:nb], RHO_MIN)  # Clamp density only

    return q_new
