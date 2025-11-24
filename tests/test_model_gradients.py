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
from GaPFlow import Problem
from GaPFlow.models.pressure import eos_pressure

sim = """
options:
    output: data/inclined
    write_freq: 1000
    silent: False
grid:
    Lx: 0.1
    Ly: 1.
    Nx: 10
    Ny: 1
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 1.1853
    xW_D: 1.1853
geometry:
    type: inclined
    hmax: 6.6e-5
    hmin: 1e-5
    U: 50.
    V: 0.
numerics:
    solver: fem
    CFL: 0.4
    adaptive: True
    tol: 1e-6
    dt: 1e-8
    max_it: 20_000
properties:
    EOS: PL
    shear: 1.846e-5
    bulk: 0.
    P0: 101325
    rho0: 1.1853
    alpha: 0.
fem_solver:
    type: newton_alpha
"""


def dp_drho_fd(rho: float, problem: Problem, eps: float = 1e-8) -> np.ndarray:
    return (eos_pressure(rho + eps, problem.prop) - eos_pressure(rho - eps, problem.prop)) / (2 * eps)


def test_dp_drho():

    problem = Problem.from_string(sim)
    problem.pre_run()

    problem.solver.update_quad()
    p_grad = problem.pressure.dp_drho_quad(3)
    # p_quad = problem.pressure.p_quad(3)
    rho = problem.solver.get_quad_field('rho', 3)

    p_grad_fd = dp_drho_fd(rho[0, 0], problem)

    assert np.isclose(p_grad[0, 0], p_grad_fd, rtol=1e-6), f"Analytical: {p_grad[0, 0]}, FD: {p_grad_fd}"


test_dp_drho()
