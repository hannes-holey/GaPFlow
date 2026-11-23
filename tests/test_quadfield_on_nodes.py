#
# Copyright 2025 Hannes Holey
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
"""Tests for GaPFlow.viz.evaluate_quadfield_on_nodes and plot_quadfield_on_nodes."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.viz import evaluate_quadfield_on_nodes, plot_quadfield_on_nodes
from GaPFlow.solver_implicit.fieldspec import patch_registry_for_gp


# 2D implicit-solver config: 8x10 parabolic gap, V_bot=2 (yz Couette driver),
# equal Dirichlet x-BCs → no net x-pressure gradient, periodic y.
# Used for shape and inverse-h scaling tests that need spatially varying h.
CONFIG = """
options:
    output: {tmp_path}
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 8
    Ny: 10
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 877.7007
    xW_D: 877.7007

geometry:
    type: parabolic_2d
    hmax: 2e-5
    hmin: 1e-5
    U: 1.0
    V: 2.0

numerics:
    solver: implicit
    dt: 1e-8
    tol: 1e-6
    max_it: 5

implicit_solver:
    equations:
        energy: False
        cavitation: False
    physics:
        gap_shear: True
        plane_shear: False
        inertia: False
        body_force: False
        squeeze: False

properties:
    EOS: DH
    rho_ref: 877.7007
    P0: 101325
    C1: 3.5e7
    C2: 1.23
    shear: 0.0794
    bulk: 0.0
    p_cav: 0.0
"""

# Flat-plates variant: hmin == hmax → h is uniform, dp/dy = 0 everywhere,
# so tau_yz_bot = eta*(V_top - V_bot)/h holds to machine precision.
CONFIG_FLAT = CONFIG.replace('    hmax: 2e-5\n    hmin: 1e-5', '    hmax: 1e-5\n    hmin: 1e-5')


@pytest.fixture(scope='module')
def problem_2d(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('quadfield')
    p = Problem.from_string(CONFIG.format(tmp_path=str(tmp_path)))
    p.run()
    return p


@pytest.fixture(scope='module')
def problem_flat(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('quadfield_flat')
    p = Problem.from_string(CONFIG_FLAT.format(tmp_path=str(tmp_path)))
    p.run()
    return p


# ---------------------------------------------------------------------------
# 1. Import + shape
# ---------------------------------------------------------------------------

def test_import_from_gapflow_viz(problem_2d):
    """evaluate_quadfield_on_nodes and plot_quadfield_on_nodes are importable from GaPFlow.viz."""
    assert callable(evaluate_quadfield_on_nodes)
    assert callable(plot_quadfield_on_nodes)


def test_h_shape(problem_2d):
    """evaluate_quadfield_on_nodes(problem, 'h') returns shape (Nx, Ny) after problem.run()."""
    h = evaluate_quadfield_on_nodes(problem_2d, 'h')
    Nx = problem_2d.grid['Nx']
    Ny = problem_2d.grid['Ny']
    assert h.shape == (Nx, Ny), f"expected ({Nx}, {Ny}), got {h.shape}"


# ---------------------------------------------------------------------------
# 2. tau_yz_bot tracks 1/h and velocity ratio
# ---------------------------------------------------------------------------

def test_tau_yz_bot_shape_and_sign(problem_2d):
    """tau_yz_bot has shape (Nx, Ny) and is negative everywhere (V_top - V_bot < 0)."""
    tau = evaluate_quadfield_on_nodes(problem_2d, 'tau_yz_bot')
    Nx, Ny = problem_2d.grid['Nx'], problem_2d.grid['Ny']
    assert tau.shape == (Nx, Ny)
    # V_bot = 2.0, V_top = 0.0  →  V_top - V_bot = -2 < 0  →  tau_yz_bot < 0
    assert np.all(tau < 0), "tau_yz_bot should be uniformly negative for V_bot > V_top"


def test_tau_yz_bot_inverse_h_scaling(problem_2d):
    """|tau_yz_bot| is larger where h is smaller (tracks 1/h)."""
    tau = evaluate_quadfield_on_nodes(problem_2d, 'tau_yz_bot')
    h   = evaluate_quadfield_on_nodes(problem_2d, 'h')
    h_min_idx = np.unravel_index(h.argmin(), h.shape)
    h_max_idx = np.unravel_index(h.argmax(), h.shape)
    assert abs(tau[h_min_idx]) > abs(tau[h_max_idx]), (
        "expected |tau| to be larger at minimum h than at maximum h"
    )


def test_tau_yz_bot_uniform_h_gives_uniform_stress(problem_flat):
    """For uniform-h parallel plates, tau_yz_bot * h / eta is spatially constant,
    confirming that 1/h is the sole source of spatial variation in tau."""
    tau = evaluate_quadfield_on_nodes(problem_flat, 'tau_yz_bot')
    h   = evaluate_quadfield_on_nodes(problem_flat, 'h')
    eta = evaluate_quadfield_on_nodes(problem_flat, 'eta')
    ratio = tau * h / eta
    # All nodes must give exactly the same value (uniform h → uniform tau).
    np.testing.assert_allclose(ratio, ratio.mean(), rtol=1e-10)
    # Sign follows velocity ratio: V_top - V_bot = -2 < 0.
    assert ratio.mean() < 0


# ---------------------------------------------------------------------------
# 3. Custom wall-stress model: evaluator matches solver's cached field
# ---------------------------------------------------------------------------

def test_tau_yz_bot_custom_matches_internal_field(tmp_path):
    """After patch_registry_for_gp(custom_yz=True), evaluate_quadfield_on_nodes reproduces
    problem.wall_stress_yz._WallStress__field.pg[3][1:-1, 1:-1] to rtol=1e-10."""
    eta0 = 0.0794

    def custom_wall_stress(rho, theta, j, h, U_bot, U_top):
        # Analytic Couette formula; depends only on h and wall velocities so
        # the evaluator and solver internal call are guaranteed identical.
        tau_bot = eta0 * (U_top - U_bot) / h
        return tau_bot, -tau_bot

    p = Problem.from_string(CONFIG.format(tmp_path=str(tmp_path)))
    p.set_custom_wall_stress(custom_wall_stress)
    p.run()

    patch_registry_for_gp(custom_yz=True)
    try:
        tau_yz_bot = evaluate_quadfield_on_nodes(p, 'tau_yz_bot')
        ref = p.wall_stress_yz._WallStress__field.pg[3][1:-1, 1:-1]
        np.testing.assert_allclose(tau_yz_bot, ref, rtol=1e-10, atol=0)
    finally:
        patch_registry_for_gp()   # reset registry args to defaults


# ---------------------------------------------------------------------------
# 4. h_prev raises NotImplementedError
# ---------------------------------------------------------------------------

def test_h_prev_raises_not_implemented(problem_2d):
    """h_prev is an inline-computed squeeze field with no standalone source;
    evaluate_quadfield_on_nodes must raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        evaluate_quadfield_on_nodes(problem_2d, 'h_prev')