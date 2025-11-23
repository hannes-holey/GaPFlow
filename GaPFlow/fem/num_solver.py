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
        pass

    def reset(self):
        pass


def newton_alpha_solver(config,
                        sol: SolutionDict,
                        get_MR: Callable,
                        check_delta: Callable
                        ):

    sol.alpha = getattr(config, 'alpha', 1.0)
    sol.a = config.a0.copy()
    sol.iter = 0

    while True:
        M, R = get_MR(sol)

        R_norm = np.linalg.norm(R)
        sol.R_norm_history.append(R_norm)
        print("Iteration", sol.iter, "Residual norm:", R_norm, "alpha:", sol.alpha)

        if sol.iter > config.max_iter:
            print("Did not converge")
            break
        if R_norm < config.R_norm_tol:
            print("Converged!")
            break

        sol.delta_a = np.linalg.solve(M, -R)
        check_delta(sol)

        sol.a += sol.alpha * sol.delta_a
        sol.iter += 1


class Solver():
    def __init__(self):
        self.get_MR_fun = None
        self.callback_fun = lambda sol: None

    def solve(self):
        assert self.get_MR_fun is not None, "get_MR_fun not set"
        self.solve_fun(self.config,
                       self.sol_dict,
                       self.get_MR_fun,
                       self.check_delta_fun)
