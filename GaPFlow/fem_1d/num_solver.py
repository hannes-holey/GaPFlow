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
from typing import Callable
import numpy as np


class SolutionDict:
    def __init__(self):
        self.R_norm_history = []
        self.iter = 0
        self.success = False

    def reset(self):
        self.R_norm_history = []
        self.iter = 0
        self.success = False


def newton_alpha_solver(fem_solver,
                        sol: SolutionDict,
                        get_MR: Callable,
                        **kwargs
                        ):

    sol.alpha = getattr(fem_solver, 'alpha', 1.0)
    sol.q = sol.q0.copy()

    while True:
        M, R = get_MR(sol.q)

        R_norm = np.linalg.norm(R)
        sol.R_norm_history.append(R_norm)
        if 'silent' not in kwargs or not kwargs['silent']:
            print(f"{sol.iter:<10d} {R_norm:<12.4e}")

        if sol.iter > fem_solver['max_iter']:
            sol.success = False
            break
        if R_norm < fem_solver['R_norm_tol']:
            sol.success = True
            break

        sol.delta_q = np.linalg.solve(M, -R)

        sol.q += sol.alpha * sol.delta_q
        sol.iter += 1


class Solver():
    def __init__(self, fem_solver: dict):
        self.get_MR_fun = None

        self.sol_dict = SolutionDict()
        self.fem_solver = fem_solver

        self.solve_fun = newton_alpha_solver

    def solve(self, **kwargs) -> SolutionDict:
        assert self.get_MR_fun is not None, "get_MR_fun not set"

        self.solve_fun(self.fem_solver,
                       self.sol_dict,
                       self.get_MR_fun,
                       **kwargs)
        return self.sol_dict
