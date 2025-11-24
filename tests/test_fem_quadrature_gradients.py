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
import pytest
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


def init_problem(quad_list) -> Problem:
    problem = Problem.from_string(sim)
    problem.solver.quad_list = quad_list
    problem.pre_run(enforced_quad_list=quad_list)
    return problem


@pytest.fixture(scope="module")
def problem():
    quad_list = [1, 2, 3, 4, 5]
    return init_problem(quad_list)


def dp_drho_fd(rho: float, problem: Problem, eps: float = 1e-8) -> np.ndarray:
    return (eos_pressure(rho + eps, problem.prop) - eos_pressure(rho - eps, problem.prop)) / (2 * eps)


def test_dp_drho(problem: Problem):

    problem.solver.update_quad()
    p_grad = problem.pressure.dp_drho_quad(3)
    rho = problem.solver.get_quad_field('rho', 3)

    p_grad_fd = dp_drho_fd(rho[0, 0], problem)

    assert np.isclose(p_grad[0, 0], p_grad_fd, rtol=1e-6), f"Analytical: {p_grad[0, 0]}, FD: {p_grad_fd}"


def dtau_xz_drho_fd(problem: Problem, eps: float = 1e-8) -> np.ndarray:

    val = problem.q[0, 0, 0]

    problem.q[0, :, :] = val + eps
    problem.wall_stress_xz.update()
    tau_xz_plus = problem.wall_stress_xz.upper[4] - problem.wall_stress_xz.lower[4]

    problem.q[0, :, :] = val - eps
    problem.wall_stress_xz.update()
    tau_xz_minus = problem.wall_stress_xz.upper[4] - problem.wall_stress_xz.lower[4]

    return ((tau_xz_plus - tau_xz_minus) / (2 * eps))[1:-1, 1:-1]


@pytest.mark.parametrize("nb_quad", [1, 2, 3, 4, 5])
def test_dtau_xz_drho(problem: Problem, nb_quad: int):
    nb_ele = 9

    problem.solver.update_quad()
    dtau_xz_drho = problem.wall_stress_xz.dtau_xz_drho_quad(nb_quad)

    grad_fd = dtau_xz_drho_fd(problem).ravel()

    # now we compare nodal values (grad_fd) to quad values (dtau_xz_drho)
    # check for each element that quad value is between nodal values
    for i in range(nb_ele):
        nodal_vals = grad_fd[i:i + 2]
        quad_val = dtau_xz_drho[:, i]
        min_nodal = np.min(nodal_vals)
        max_nodal = np.max(nodal_vals)
        for qv in quad_val:
            assert min_nodal <= qv <= max_nodal, f"Quad val {qv} not between nodal vals {min_nodal}, {max_nodal}"


@pytest.mark.parametrize("nb_quad", [1, 2, 3, 4, 5])
def test_tau_xz_quad_structure(problem: Problem, nb_quad: int):
    nb_ele = 9

    problem.solver.update_quad()
    quad_field = problem.wall_stress_xz.dtau_xz_drho_quad(nb_quad)
    assert quad_field.shape == (nb_quad, nb_ele), f"Expected shape ({nb_quad}, {nb_ele}), got {quad_field.shape}"

    # in this test, value should increase over elements and within each element (h gets smaller)
    last_val = -np.inf
    for i in range(nb_ele):
        for j in range(1, nb_quad):
            assert quad_field[j, i] > last_val, f"Value did not increase: {quad_field[j, i]} {i}_{j} <= {last_val}"
            last_val = quad_field[j, i]


@pytest.mark.parametrize("nb_quad", [1, 2, 3, 4, 5])
def test_height_quad_structure(problem: Problem, nb_quad: int):
    nb_ele = 9

    problem.solver.update_quad()
    quad_field = problem.topo.h_quad(nb_quad)
    assert quad_field.shape == (nb_quad, nb_ele), f"Expected shape ({nb_quad}, {nb_ele}), got {quad_field.shape}"

    # in this test, value should decrease over elements and within each element (h gets smaller)
    last_val = np.inf
    for i in range(nb_ele):
        for j in range(nb_quad):
            assert quad_field[j, i] < last_val, f"Value did not decrease: {quad_field[j, i]} {i}_{j} >= {last_val}"
            last_val = quad_field[j, i]


@pytest.mark.parametrize("nb_quad", [1, 2, 3, 4, 5])
def test_jx_quad_structure(problem: Problem, nb_quad: int):
    nb_ele = 9

    problem.solver.update_quad()
    quad_field = problem.solver.get_quad_field('jx', nb_quad)
    assert quad_field.shape == (nb_quad, nb_ele), f"Expected shape ({nb_quad}, {nb_ele}), got {quad_field.shape}"

    # in this test, value should increase over elements and within each element (jx gets larger)
    last_val = -np.inf
    for i in range(nb_ele):
        for j in range(nb_quad):
            assert quad_field[j, i] >= last_val, f"Value did not increase: {quad_field[j, i]} {i}_{j} <= {last_val}"
            last_val = quad_field[j, i]


@pytest.mark.parametrize("nb_quad", [1, 2, 3, 4, 5])
def test_p_quad_structure(problem: Problem, nb_quad: int):
    nb_ele = 9

    rho = problem.q[0, 0, 0]
    vals = np.linspace(rho - 0.1, rho + 0.1, nb_ele + 3)
    arr = np.tile(vals, (3, 1)).T

    problem.q[0, :, :] += arr
    problem.solver.update_quad()

    quad_field = problem.pressure.p_quad(nb_quad)
    assert quad_field.shape == (nb_quad, nb_ele), f"Expected shape ({nb_quad}, {nb_ele}), got {quad_field.shape}"

    # in this test, value should increase over elements and within each element (rho gets larger)
    last_val = -np.inf
    for i in range(nb_ele):
        for j in range(nb_quad):
            assert quad_field[j, i] >= last_val, f"Value did not increase: {quad_field[j, i]} {i}_{j} <= {last_val}"
            last_val = quad_field[j, i]

    # check rho similarly
    quad_rho = problem.solver.get_quad_field('rho', nb_quad)
    last_val = -np.inf
    for i in range(nb_ele):
        for j in range(nb_quad):
            assert quad_rho[j, i] >= last_val, f"Value did not increase: {quad_rho[j, i]} {i}_{j} <= {last_val}"
            last_val = quad_rho[j, i]
