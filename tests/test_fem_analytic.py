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
import pandas as pd
import pytest
from pathlib import Path

from GaPFlow.problem import Problem
from GaPFlow.models.pressure import eos_pressure

CONFIG_DIR = Path(__file__).parents[1] / "config_lib" / "validation"
DATA_DIR = Path(__file__).parent / "ref_data"


# =============================================================================
# Cavitation topography helpers
# =============================================================================

def _geo_grid_1d(L, N):
    d = L / N
    return np.linspace(d / 2, L - d / 2, N)


def _regen_conv_slider_pocket_1d(problem, Nx_geo=None):
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmin, hmax, h_pock = 1.0e-6, 1.1e-6, 0.4e-6
    x_p_start, x_p_end = 4.0e-3, 10.0e-3

    h_geo = hmax - (hmax - hmin) * x_geo / Lx
    h_geo[(x_geo >= x_p_start) & (x_geo < x_p_end)] += h_pock

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)
    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def _regen_twin_parabolic_slider(problem, Nx_geo=None):
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmax, hmin1, hmin2 = 6.0e-5, 3.0e-5, 3.5e-5
    x_s1_start, x_s1_end = 0.000, 0.036
    x_s2_start, x_s2_end = 0.046, 0.082

    h_geo = np.full(Nx_geo, hmax)
    mask1 = (x_geo >= x_s1_start) & (x_geo < x_s1_end)
    x1_mid = 0.5 * (x_s1_start + x_s1_end)
    x1_half = 0.5 * (x_s1_end - x_s1_start)
    h_geo[mask1] = hmin1 + (hmax - hmin1) * ((x_geo[mask1] - x1_mid) / x1_half) ** 2

    mask2 = (x_geo >= x_s2_start) & (x_geo < x_s2_end)
    x2_mid = 0.5 * (x_s2_start + x_s2_end)
    x2_half = 0.5 * (x_s2_end - x_s2_start)
    h_geo[mask2] = hmin2 + (hmax - hmin2) * ((x_geo[mask2] - x2_mid) / x2_half) ** 2

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)
    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


def _regen_twin_parabolic_slider_id(problem, Nx_geo=None):
    g = problem.grid
    Nx, Ny, Lx = g['Nx'], g['Ny'], g['Lx']
    if Nx_geo is None:
        Nx_geo = Nx

    x_geo = _geo_grid_1d(Lx, Nx_geo)
    hmax, hmin = 50.2e-6, 25.4e-6
    L_slider = Lx / 2.0

    h_geo = np.full(Nx_geo, hmax)
    mask1 = x_geo < L_slider
    x1_mid = L_slider / 2.0
    h_geo[mask1] = hmin + (hmax - hmin) * ((x_geo[mask1] - x1_mid) / (L_slider / 2.0)) ** 2

    mask2 = x_geo >= L_slider
    x2_mid = L_slider + L_slider / 2.0
    h_geo[mask2] = hmin + (hmax - hmin) * ((x_geo[mask2] - x2_mid) / (L_slider / 2.0)) ** 2

    x_fem = np.linspace(g['dx'] / 2, Lx - g['dx'] / 2, Nx)
    h1d = np.interp(x_fem, x_geo, h_geo)
    problem.topo.set_global_height(np.tile(h1d[:, np.newaxis], (1, Ny)))


# =============================================================================
# Helpers
# =============================================================================

def load_template(name: str) -> str:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return f.read()


def compute_l2_error(numerical, analytical):
    return np.sqrt(((numerical - analytical) ** 2).mean()) / analytical.max()


def compare_solutions(sol1, sol2, bc_type="periodic", rtol=0.05, field_name="field"):
    """Point-wise comparison; Dirichlet cases use inner-region clipping."""
    if bc_type == "periodic":
        np.testing.assert_allclose(sol1, sol2, rtol=rtol,
                                   err_msg=f"{field_name}: point-wise mismatch")
    else:
        N = len(sol1)
        inner = slice(int(0.05 * N), int(0.95 * N))
        np.testing.assert_allclose(
            sol1[inner], sol2[inner], rtol=rtol,
            err_msg=f"{field_name}: inner region (5%-95%) mismatch")
        int1, int2 = np.trapezoid(sol1), np.trapezoid(sol2)
        np.testing.assert_allclose(
            int1, int2, rtol=0.02,
            err_msg=f"{field_name}: integral mismatch ({int1:.4f} vs {int2:.4f})")
        np.testing.assert_allclose(
            sol1[inner].max(), sol2[inner].max(), rtol=rtol,
            err_msg=f"{field_name}: inner-region peak mismatch")


# =============================================================================
# Solver comparison: solver_fem vs saved explicit reference
# =============================================================================

FEM2D_DEFAULTS = {
    "Ny": 4,
    "solver": "fem",
    "dt": 1e-2,
    "max_it": 100,
    "adaptive": 0,
    "CFL": 0.5,
    "pressure_stab_alpha": 0,
    "momentum_stab_alpha": 0,
}

COMPARISON_CASES = {
    "inclined_slider": {
        "template": "inclined_slider",
        "bc_type": "dirichlet",
        "rho_bc": 1.0,
    },
    "journal_bearing": {
        "template": "journal_bearing",
        "bc_type": "periodic",
        "rho_bc": 1.0,
    },
    "parabolic_slider": {
        "template": "parabolic_slider",
        "bc_type": "dirichlet",
        "rho_bc": 850.0,
    },
}


def _run_fem(case_name: str):
    case = COMPARISON_CASES[case_name]
    template = load_template(case["template"])
    params = {**FEM2D_DEFAULTS, "rho_bc": case["rho_bc"]}
    problem = Problem.from_string(template.format(**params))
    problem.run()
    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()
    return rho, jx


def _load_explicit_reference(case_name: str):
    path = DATA_DIR / f"explicit_{case_name}.npz"
    if not path.exists():
        pytest.skip(f"Reference data not found: {path}. Run generate_explicit_reference.py.")
    data = np.load(path)
    return data["rho"], data["jx"]


@pytest.fixture(scope="module", params=list(COMPARISON_CASES.keys()))
def comparison(request):
    case_name = request.param
    rho_ref, jx_ref = _load_explicit_reference(case_name)
    rho_fem, jx_fem = _run_fem(case_name)
    bc_type = COMPARISON_CASES[case_name]["bc_type"]
    return dict(rho_ref=rho_ref, jx_ref=jx_ref, rho_fem=rho_fem, jx_fem=jx_fem,
                bc_type=bc_type, case=case_name)


def test_fem_vs_explicit_rho(comparison):
    r = comparison
    compare_solutions(r["rho_ref"], r["rho_fem"], bc_type=r["bc_type"],
                      field_name=f"rho ({r['case']})")


def test_fem_vs_explicit_jx(comparison):
    r = comparison
    compare_solutions(r["jx_ref"], r["jx_fem"], bc_type=r["bc_type"],
                      field_name=f"jx ({r['case']})")


# =============================================================================
# Analytic: 2D Poiseuille flow
# =============================================================================

@pytest.fixture(scope="module")
def poiseuille_problem():
    problem = Problem.from_yaml(str(CONFIG_DIR / "poiseuille_2d_body_force.yaml"))
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_poiseuille_velocity_profile(poiseuille_problem):
    """In-plane viscous diffusion (R23xy, R23yx) driven by body force.

    jx/jy Dirichlet BCs use the mirror formula: effective wall at the cell face
    (y=0, y=Ly), not at the ghost cell center.
    """
    p = poiseuille_problem
    rho = p.q[0][1:-1, 1:-1]
    jx = p.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = p.grid["Ly"], p.grid["Ny"]
    dy = Ly / Ny
    mu = p.prop["shear"]
    f_x = p.prop["force_x"]
    rho_m = rho.mean()

    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    u_ana = (rho_m * f_x) / (2 * mu) * y * (Ly - y)

    l2_err = compute_l2_error(vx, u_ana)
    assert l2_err < 0.01, f"Poiseuille L2 error = {l2_err:.2e} > 0.01"


# =============================================================================
# Analytic: 2D Couette flow
# =============================================================================

@pytest.fixture(scope="module")
def couette_problem():
    problem = Problem.from_yaml(str(CONFIG_DIR / "couette_2d_wall_velocity.yaml"))
    problem.q[0][:] = 1.0
    problem.q[1][:] = 0.0
    problem.q[2][:] = 0.0
    problem.run()
    return problem


def test_couette_velocity_profile(couette_problem):
    """In-plane viscous diffusion (R23xy, R23yx) driven by wall motion.

    jx/jy Dirichlet BCs use the mirror formula: effective wall at the cell face
    (y=0, y=Ly), not at the ghost cell center.
    """
    p = couette_problem
    rho = p.q[0][1:-1, 1:-1]
    jx = p.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Ly, Ny = p.grid["Ly"], p.grid["Ny"]
    dy = Ly / Ny
    rho_m = rho.mean()
    U_wall = p.grid["bc_yN_D_val"][1] / rho_m

    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    u_ana = U_wall * y / Ly

    l2_err = compute_l2_error(vx, u_ana)
    assert l2_err < 0.01, f"Couette L2 error = {l2_err:.2e} > 0.01"


# =============================================================================
# Analytic: Bernoulli venturi nozzle
# =============================================================================

_H_INLET = 1.0e-3
_H_THROAT = 0.5e-3
_X_RAMP_START = 0.03
_X_RAMP_END = 0.07
_THROAT_HALF_WIDTH = 0.005
_RHO0 = 1000.0
_V_INLET = 5.0


def _venturi_topography(xx):
    x_mid = (_X_RAMP_START + _X_RAMP_END) / 2
    ramp_len = x_mid - _THROAT_HALF_WIDTH - _X_RAMP_START
    dist = np.clip(np.abs(xx - x_mid) - _THROAT_HALF_WIDTH, 0, ramp_len)
    xi = dist / ramp_len
    return np.where(
        (xx >= _X_RAMP_START) & (xx <= _X_RAMP_END),
        (_H_INLET + _H_THROAT) / 2 - (_H_INLET - _H_THROAT) / 2 * np.cos(np.pi * xi),
        _H_INLET,
    )


@pytest.fixture(scope="module")
def bernoulli_problem():
    problem = Problem.from_yaml(str(CONFIG_DIR / "bernoulli_venturi.yaml"))
    problem.topo.set_mapped_height(_venturi_topography(problem.topo.xx))
    problem.q[1][:] = _RHO0 * _V_INLET
    problem.run()
    return problem


def test_bernoulli_velocity_ratio(bernoulli_problem):
    """Velocity doubles at throat (mass conservation, h_inlet / h_throat = 2)."""
    p = bernoulli_problem
    rho_b = p.q[0][1:-1, 1:-1]
    jx_b = p.q[1][1:-1, 1:-1]
    dx = p.grid["dx"]
    x_b = np.arange(rho_b.shape[0]) * dx + dx / 2
    j_center = rho_b.shape[1] // 2
    vx_b = jx_b[:, j_center] / rho_b[:, j_center]

    inlet = x_b < _X_RAMP_START
    throat = (x_b >= 0.045) & (x_b <= 0.055)
    v_ratio = vx_b[throat].mean() / vx_b[inlet].mean()

    assert abs(v_ratio - 2.0) < 0.01, \
        f"Bernoulli velocity ratio = {v_ratio:.4f}, expected 2.0"


def test_bernoulli_pressure_drop(bernoulli_problem):
    """Pressure drop at throat matches Bernoulli prediction within 1%."""
    p = bernoulli_problem
    rho_b = p.q[0][1:-1, 1:-1]
    dx = p.grid["dx"]
    x_b = np.arange(rho_b.shape[0]) * dx + dx / 2
    j_center = rho_b.shape[1] // 2
    p_b = np.asarray(eos_pressure(rho_b[:, j_center], p.prop))

    inlet = x_b < _X_RAMP_START
    throat = (x_b >= 0.045) & (x_b <= 0.055)
    dp_sim = p_b[inlet].mean() - p_b[throat].mean()

    v_throat_theory = _V_INLET * _H_INLET / _H_THROAT
    dp_theory = 0.5 * _RHO0 * (v_throat_theory ** 2 - _V_INLET ** 2)

    rel_err = abs(dp_sim - dp_theory) / dp_theory
    assert rel_err < 0.01, \
        f"Bernoulli Δp error = {rel_err * 100:.2f}%, sim={dp_sim:.1f} Pa, theory={dp_theory:.1f} Pa"


# =============================================================================
# Cavitation: pressure profiles vs. published reference data
# =============================================================================

def _load_ref_pressure(csv_path, ref_x_scale, ref_p_scale):
    """Load a reference CSV and return (x_metres, p_Pa) arrays."""
    ref = pd.read_csv(csv_path, skipinitialspace=True)
    return ref['x'].values * ref_x_scale, ref['y'].values * ref_p_scale


def _fem_pressure_1d(problem):
    """Return (x_metres, p_Pa) on cell-centre grid, averaged over y."""
    g = problem.grid
    x = np.linspace(g['dx'] / 2, g['Lx'] - g['dx'] / 2, g['Nx'])
    p = problem.pressure.pressure[1:-1, 1:-1].mean(axis=1)
    return x, p


CAVITATION_CASES = {
    "conv_slider_pocket_1D": {
        "yaml": "conv_slider_pocket_1D.yaml",
        "regen": lambda p: _regen_conv_slider_pocket_1d(p, Nx_geo=50),
        "ref_csv": "ref_conv_slider_pocket_1D.csv",
        "ref_x_scale": 0.020,
        "ref_p_scale": 1e6,
    },
    "parabolic_slider_cav": {
        "yaml": "parabolic_slider_cav.yaml",
        "regen": None,
        "ref_csv": "ref_parabolic_slider.csv",
        "ref_x_scale": 0.0762,
        "ref_p_scale": 1.0,
    },
    "twin_parabolic_slider": {
        "yaml": "twin_parabolic_slider.yaml",
        "regen": _regen_twin_parabolic_slider,
        "ref_csv": "ref_twin_parabolic_slider.csv",
        "ref_x_scale": 0.09,
        "ref_p_scale": 1e5,
    },
    "twin_parabolic_slider_id": {
        "yaml": "twin_parabolic_slider_id.yaml",
        "regen": _regen_twin_parabolic_slider_id,
        "ref_csv": "ref_twin_parabolic_slider_id.csv",
        "ref_x_scale": 0.0762,
        "ref_p_scale": 1e6,
    },
}


@pytest.fixture(scope="module", params=list(CAVITATION_CASES.keys()))
def cavitation_comparison(request):
    case_name = request.param
    case = CAVITATION_CASES[case_name]

    problem = Problem.from_yaml(str(CONFIG_DIR / case["yaml"]))
    if case["regen"] is not None:
        case["regen"](problem)
    problem.run()

    x_fem, p_fem = _fem_pressure_1d(problem)

    x_ref, p_ref = _load_ref_pressure(
        DATA_DIR / case["ref_csv"],
        case["ref_x_scale"],
        case["ref_p_scale"],
    )
    p_fem_at_ref = np.interp(x_ref, x_fem, p_fem)

    return dict(p_fem=p_fem_at_ref, p_ref=p_ref, case=case_name)


def test_cavitation_pressure_profile(cavitation_comparison):
    r = cavitation_comparison
    l2_err = compute_l2_error(r["p_fem"], r["p_ref"])
    assert l2_err < 0.05, \
        f"Cavitation pressure L2 error = {l2_err:.2e} > 0.05 ({r['case']})"
