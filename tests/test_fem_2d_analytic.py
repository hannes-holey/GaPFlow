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
"""Analytical validation tests for 2D FEM solver."""

import os
import pytest
import numpy as np

from GaPFlow import Problem
from GaPFlow.models.pressure import eos_pressure

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')


def compute_l2_error(numerical, analytical):
    """Compute L2 relative error."""
    return np.sqrt(((numerical - analytical)**2).mean()) / analytical.max()


# =============================================================================
# Test: 2D Poiseuille Flow (body force, periodic x, no-slip y walls)
# =============================================================================

@pytest.fixture
def poiseuille_problem():
    """Create and run Poiseuille flow problem."""
    config_path = os.path.join(CONFIG_DIR, 'poiseuille_2d_body_force.yaml')
    problem = Problem.from_yaml(config_path)
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_poiseuille_2d(poiseuille_problem):
    """2D Poiseuille flow with body force.

    Tests in-plane viscous diffusion (R23xy, R23yx) driven by body force.
    BCs applied at ghost cell centers: y_lo=-dy/2, y_hi=Ly+dy/2.
    """
    problem = poiseuille_problem

    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = problem.grid['Ly'], problem.grid['Ny']
    dy = Ly / Ny
    mu = problem.prop['shear']
    f_x = problem.prop['force_x']
    rho_m = rho.mean()

    # Cell centers and analytical solution
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    y_lo, y_hi = -dy / 2, Ly + dy / 2
    u_ana = (rho_m * f_x) / (2 * mu) * (y - y_lo) * (y_hi - y)

    l2_err = compute_l2_error(vx, u_ana)

    assert l2_err < 0.01, f"L2 error = {l2_err:.2e} exceeds tolerance 0.01"


# =============================================================================
# Test: 2D Couette Flow (wall velocity, periodic x, no-slip y walls)
# =============================================================================

@pytest.fixture
def couette_problem():
    """Create and run Couette flow problem."""
    config_path = os.path.join(CONFIG_DIR, 'couette_2d_wall_velocity.yaml')
    problem = Problem.from_yaml(config_path)
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_couette_2d(couette_problem):
    """2D Couette flow with wall velocity.

    Tests in-plane viscous diffusion (R23xy, R23yx) driven by wall motion.
    BCs applied at ghost cell centers: y_lo=-dy/2, y_hi=Ly+dy/2.
    """
    problem = couette_problem

    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = problem.grid['Ly'], problem.grid['Ny']
    dy = Ly / Ny
    rho_m = rho.mean()

    # Wall velocity from BC (jx_wall / rho = U)
    U_wall = problem.grid['bc_yN_D_val'][1] / rho_m

    # Cell centers and analytical solution (linear profile)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    y_lo, y_hi = -dy / 2, Ly + dy / 2
    u_ana = U_wall * (y - y_lo) / (y_hi - y_lo)

    l2_err = compute_l2_error(vx, u_ana)

    assert l2_err < 0.01, f"L2 error = {l2_err:.2e} exceeds tolerance 0.01"


# =============================================================================
# Test: Bernoulli Venturi Flow
# =============================================================================

# Geometry parameters
H_INLET, H_THROAT = 1.0e-3, 0.5e-3
X_RAMP_START, X_RAMP_END = 0.03, 0.07
THROAT_HALF_WIDTH = 0.005
RHO0, V_INLET = 1000.0, 5.0

def _venturi_topography(xx):
    """Symmetric venturi with cosine transitions."""
    x_mid = (X_RAMP_START + X_RAMP_END) / 2
    ramp_len = x_mid - THROAT_HALF_WIDTH - X_RAMP_START
    
    # Distance from throat edge, clipped to [0, ramp_len]
    dist = np.clip(np.abs(xx - x_mid) - THROAT_HALF_WIDTH, 0, ramp_len)
    xi = dist / ramp_len
    
    return np.where(
        (xx >= X_RAMP_START) & (xx <= X_RAMP_END),
        (H_INLET + H_THROAT)/2 - (H_INLET - H_THROAT)/2 * np.cos(np.pi * xi),
        H_INLET
    )


@pytest.fixture
def bernoulli_problem():
    """Create and run Bernoulli venturi flow problem."""
    config_path = os.path.join(CONFIG_DIR, 'bernoulli_venturi.yaml')
    problem = Problem.from_yaml(config_path)
    problem.topo.set_mapped_height(_venturi_topography(problem.topo.xx))
    problem.run()
    return problem


def test_bernoulli_venturi(bernoulli_problem):
    """Bernoulli flow through symmetric venturi.

    Tests convective terms (R22xx, R22yx, ...) against Bernoulli equation.
    Validates: velocity scaling with area, pressure drop, mass conservation.
    """
    problem = bernoulli_problem

    # Extract centerline data
    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    h = problem.topo.h[1:-1, 1:-1]
    j_center = rho.shape[1] // 2

    dx = problem.grid['dx']
    x = np.arange(rho.shape[0]) * dx + dx / 2
    vx = jx[:, j_center] / rho[:, j_center]
    p = np.asarray(eos_pressure(rho[:, j_center], problem.prop))

    # Define regions
    inlet = x < X_RAMP_START
    throat = (x >= 0.045) & (x <= 0.055)
    outlet = x > X_RAMP_END

    # Measured values
    v_in, v_throat, v_out = vx[inlet].mean(), vx[throat].mean(), vx[outlet].mean()
    p_in, p_throat = p[inlet].mean(), p[throat].mean()

    # Bernoulli predictions
    v_throat_theory = V_INLET * H_INLET / H_THROAT
    dp_theory = 0.5 * RHO0 * (v_throat_theory**2 - V_INLET**2)
    dp_sim = p_in - p_throat

    # Mass flux conservation
    mass_flux = rho[:, j_center] * vx * h[:, j_center]
    mass_var = (mass_flux.max() - mass_flux.min()) / mass_flux.mean()

    # Assertions
    v_ratio_err = abs(v_throat / v_in - 2.0) / 2.0
    assert v_ratio_err < 0.02, f"Velocity ratio error = {v_ratio_err:.1%} exceeds 2%"

    dp_err = abs(dp_sim - dp_theory) / dp_theory
    assert dp_err < 0.05, f"Pressure drop error = {dp_err:.1%} exceeds 5%"

    assert mass_var < 0.01, f"Mass flux variation = {mass_var:.1%} exceeds 1%"

    # Venturi symmetry: outlet velocity should return to inlet velocity
    v_symmetry_err = abs(v_out - v_in) / v_in
    assert v_symmetry_err < 0.02, f"Venturi symmetry error = {v_symmetry_err:.1%} exceeds 2%"


# =============================================================================
# Test: 2D Temperature Diffusion
# =============================================================================

def test_temp_diffusion_2d():
    """2D temperature diffusion with Dirichlet T=0 boundaries.

    Tests thermal diffusion terms (R35x, R35y) against analytical solution.
    Compares at each timestep since both profiles decay to zero at steady state.
    """
    config_path = os.path.join(CONFIG_DIR, 'temp_diffusion_2d.yaml')
    problem = Problem.from_yaml(config_path)

    # Parameters for analytical solution
    Lx, Nx = problem.grid['Lx'], problem.grid['Nx']
    dx, dt = problem.grid['dx'], problem.numerics['dt']
    cv, k = problem.energy_spec['cv'], problem.energy_spec['k']
    rho = problem.prop['rho0']
    T_max = problem.energy_spec['T0'][2]
    L_eff = Lx + dx  # effective length (BCs at ghost cell centers)

    # Precompute spatial part of analytical solution
    x = np.arange(Nx + 2) * dx - dx / 2
    x_shifted = x + dx / 2
    sin_profile = np.sin(np.pi * x_shifted / L_eff)

    # Decay rate
    alpha = k / (cv * rho)
    lam_sq = alpha * (np.pi / L_eff)**2

    # Track max error across all timesteps
    max_l2_err = 0.0
    j_mid = problem.grid['Ny'] // 2 + 1

    def check_consistency():
        nonlocal max_l2_err
        t = problem.simtime
        T_ana = T_max * np.exp(-lam_sq * t) * sin_profile
        T_num = problem.energy.temperature[:, j_mid]
        l2_err = compute_l2_error(T_num, T_ana)
        max_l2_err = max(max_l2_err, l2_err)

    problem.add_callback(check_consistency)
    problem.run()

    assert max_l2_err < 0.02, f"Max L2 error = {max_l2_err:.2e} exceeds tolerance 0.02"
