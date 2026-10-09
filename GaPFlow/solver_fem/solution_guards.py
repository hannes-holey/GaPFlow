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

# flake8: noqa: W503
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .solver_fem import FEMSolver

THETA_MAX = 0.90


def _clamp_pressure(q: np.ndarray, solver: "FEMSolver") -> np.ndarray:
    p = solver._sol_slices['p']
    p_cav = float(solver.problem.prop['p_cav'])
    q[p] = np.maximum(q[p], p_cav)
    return q


def _clamp_theta(q: np.ndarray, solver: "FEMSolver") -> np.ndarray:
    p = solver._sol_slices['p']
    theta = solver._sol_slices['theta']
    p_cav = float(solver.problem.prop['p_cav'])
    theta_min = np.finfo(float).eps

    th = np.clip(q[theta], 0.0, THETA_MAX)
    a = q[p] - p_cav
    q[theta] = np.where((np.abs(a) < theta_min) & (th < theta_min), theta_min, th)
    return q


def clamp_solution(q: np.ndarray, solver: "FEMSolver") -> np.ndarray:
    """Enforce p >= p_cav and 0 <= theta <= THETA_MAX, with a singularity
    guard at (p - p_cav, theta) = (0, 0)."""
    if solver.cavitation:
        q = _clamp_pressure(q, solver)
        q = _clamp_theta(q, solver)
    return q
