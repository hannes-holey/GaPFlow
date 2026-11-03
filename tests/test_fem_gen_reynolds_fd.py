"""Finite-difference verification of the generalized Reynolds terms (Elrod-Adams
cavitation), term by term, see `test_fem_assembly_fd.py` for the infrastructure.

Two variants:
- 'default': built-in epsilon = rho h^3/(12 eta) and rho_eff = rho, no piezoviscosity
  (with piezoviscosity, the nodal eta is interpolated to the quad points while deta/dp
  is evaluated pointwise, so the Jacobian is exact only up to the interpolation error).
- 'custom': user epsilon(p, h) with DH EoS and Barus law, user rho_eff(p, h).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.solver_implicit.solver_implicit import ImplicitSolver
from GaPFlow.solver_implicit.terms import (
    GRey_R1T, GRey_R11x, GRey_R11x_deps, GRey_R11y, GRey_R11y_deps,
    GRey_R11Sx, GRey_R11Sy, R_cav,
)

from test_fem_assembly_fd import compute_fd_jacobian, rel_err, _set_and_sync


CONFIG = """
options:
    output: /tmp/fem_gen_reynolds_fd
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 3
    Ny: 3
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 877.7007
    xW_D: 877.7007
    yS_D: 877.7007
    yN_D: 877.7007

geometry:
    type: parabolic_2d
    hmax: 2e-4
    hmin: 1e-4
    U: 1.0
    V: 2.0

numerics:
    solver: implicit
    dt: 1e-8
    tol: 1e-6
    max_it: 100

implicit_solver:
    equations:
        reynolds: True
        cavitation: True
        generalized_reynolds: True

properties:
    EOS: DH
    rho0: 877.7007
    P0: 101325
    C1: 3.5e7
    C2: 1.23
    shear: 0.0794
    bulk: 0.0
    p_cav: 0.0
"""

P0, RHO0, C1, C2 = 101325., 877.7007, 3.5e7, 1.23
ETA0, AB, H_REF = 0.0794, 2e-8, 1e-4


def _rho_dh(p):
    u = p - P0
    return RHO0 * (C2 * u + C1) / (u + C1)


def _epsilon(p, h):
    return _rho_dh(p) * h ** 3 / (12.0 * ETA0 * jnp.exp(AB * p))


def _rho_eff(p, h):
    return _rho_dh(p) * (1.0 + 0.1 * h / H_REF)


TOL = 1e-6

# Jacobian-only companions (*_deps) are checked together with their main term
TERM_GROUPS = [
    (GRey_R1T,),
    (GRey_R11x, GRey_R11x_deps),
    (GRey_R11y, GRey_R11y_deps),
    (GRey_R11Sx,),
    (GRey_R11Sy,),
    (R_cav,),
]


def _make_solver(variant: str) -> ImplicitSolver:
    problem = Problem.from_string(CONFIG)
    if variant == 'custom':
        problem.set_custom_epsilon(_epsilon)
        problem.set_custom_rho_eff(_rho_eff)
    problem._pre_run()
    return problem.solver


def _init_state(solver: ImplicitSolver) -> None:
    """Half full-film (p ramp, theta at floor), half cavitated (p = p_cav, theta ramp)."""
    p_cav = float(solver.problem.prop['p_cav'])
    q = solver.get_q_nodal().copy()

    p_sl = solver._sol_slices['p']
    theta_sl = solver._sol_slices['theta']
    n_p = p_sl.stop - p_sl.start
    n_full_film = n_p // 2

    p_vals = np.full(n_p, p_cav)
    theta_vals = np.full(n_p, np.finfo(float).eps)
    p_vals[:n_full_film] = p_cav + np.linspace(1e4, 1e6, n_full_film)
    theta_vals[n_full_film:] = np.linspace(1e-3, 0.5, n_p - n_full_film)

    q[p_sl] = p_vals
    q[theta_sl] = theta_vals
    _set_and_sync(solver, q)

    solver.quad_mgr.qf('h_prev')[:] *= 0.9


@pytest.fixture(scope='module', params=['default', 'custom'])
def solver(request):
    s = _make_solver(request.param)
    _init_state(s)
    return s


@pytest.mark.parametrize('group', TERM_GROUPS, ids=['+'.join(t.name for t in g) for g in TERM_GROUPS])
def test_jacobian(solver, group):
    all_terms = solver.assembly.assembly_terms
    name = '+'.join(t.name for t in group)
    try:
        solver.assembly.assembly_terms = list(group)
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver)
        assert np.linalg.norm(J) > 1e-10, f"{name}: FD Jacobian is essentially zero"
        err = rel_err(M, J)
        assert err < TOL, f"{name}: rel_err={err:.2e} >= {TOL:.0e}"
    finally:
        solver.assembly.assembly_terms = all_terms
