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

# flake8: noqa: E501
"""Finite-difference verification of `assemble_matrix` against `assemble_rhs`,
term by term, on one DH + Elrod-Adams cavitation configuration.

Excluded (not tolerance-relaxed, excluded): R23xy/yx/xx/yy -- eta is a plain
nodal field with no deta/dp wired into any R23* term, so live piezoviscosity
breaks it; R1STx/y, OSS_TERMS, SUPG_TERMS, SUPG_SQUEEZE_TERMS, FC_TERMS --
stabilization coefficients depending on jx/jy/theta but frozen by design.

Elastic/force-balance are off because EINM terms use a cutoff, giving only
approximate gradients.

Energy-equation terms (R3*) are out of scope.
"""
import numpy as np
import pytest

from GaPFlow.problem import Problem
from GaPFlow.solver_fem.solver_fem import FEMSolver
from GaPFlow.solver_fem.terms import (
    R11x_cav, R11y_cav, R11Sx_cav, R11Sy_cav, R1T_cav, R1Th_cav, R_cav,
    R21x, R21y, R2Tx, R2Ty, R2Thx, R2Thy,
    R22xx, R22Sxx, R22yx, R22Syx, R22xy, R22Sxy, R22yy, R22Syy,
    R24x_cav, R24y_cav,
    R25x, R25y,
)

# =============================================================================
# Problem configuration
#
# Dowson-Higginson EOS + cavitation, Barus piezoviscosity, and
# every physics flag that keeps an exact Jacobian: inertia, gap-shear wall
# stress, body force, squeeze. No stabilization (ad/oss/fc/supg default to off).
# =============================================================================

CONFIG = """
options:
    output: /tmp/fem_assembly_fd
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: 2
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
    hmax: 2e-5
    hmin: 1e-5
    U: 1.0
    V: 2.0

numerics:
    solver: fem
    dt: 1e-8
    tol: 1e-6
    max_it: 100

fem_solver:
    type: newton_alpha
    equations:
        energy: False
        cavitation: True
    physics:
        gap_shear: True
        plane_shear: False
        inertia: True
        body_force: True
        squeeze: True

properties:
    EOS: DH
    rho0: 877.7007
    P0: 101325
    C1: 3.5e7
    C2: 1.23
    shear: 0.0794
    bulk: 0.0
    p_cav: 0.0
    force_x: 1e6
    force_y: -5e5
    piezo:
        name: Barus
        aB: 2e-8
    viscosity:
        freeze_gradient: False
        underrelax_gradient_value: False
"""

# =============================================================================
# Terms under test — each checked individually against the FD Jacobian.
# =============================================================================

TOL = 1e-6

TERMS = [
    # mass equation
    R11x_cav, R11y_cav, R11Sx_cav, R11Sy_cav, R1T_cav, R1Th_cav,
    # cavitation
    R_cav,
    # momentum equation
    R21x, R21y, R2Tx, R2Ty, R2Thx, R2Thy,
    R22xx, R22Sxx, R22yx, R22Syx, R22xy, R22Sxy, R22yy, R22Syy,
    R24x_cav, R24y_cav,
    R25x, R25y,
]


# =============================================================================
# Helpers
# =============================================================================

def _make_solver() -> FEMSolver:
    problem = Problem.from_string(CONFIG)
    problem._pre_run()
    return problem.solver


def _set_and_sync(solver: FEMSolver, q: np.ndarray) -> None:
    solver.update_q_nodal(q)
    solver.update_quad()


def _init_state(solver: FEMSolver) -> None:
    """Set up a solution state that exercises every term group at once."""
    p_cav = float(solver.problem.prop['p_cav'])
    theta_eps = np.finfo(float).eps
    q = solver.get_q_nodal().copy()

    p_sl = solver._sol_slices['p']
    theta_sl = solver._sol_slices['theta']
    n_p = p_sl.stop - p_sl.start
    n_full_film = n_p // 2

    p_vals = np.empty(n_p)
    theta_vals = np.empty(n_p)

    # full-film branch: p ramps from just above p_cav (near-transition) up to
    # a converged, cavitation-free pressure; theta pinned at its floor.
    p_vals[:n_full_film] = p_cav + np.linspace(1.0, 1e4, n_full_film)
    theta_vals[:n_full_film] = theta_eps

    # cavitated branch: pinned at p_cav; theta ramps up from its floor.
    p_vals[n_full_film:] = p_cav
    theta_vals[n_full_film:] = np.linspace(1e-3, 0.5, n_p - n_full_film)

    q[p_sl] = p_vals
    q[theta_sl] = theta_vals

    jx_sl = solver._sol_slices['jx']
    q[jx_sl] *= 1.0 + 0.3 * np.linspace(-1, 1, jx_sl.stop - jx_sl.start)

    jy_sl = solver._sol_slices['jy']
    q[jy_sl] *= 1.0 + 0.3 * np.linspace(-1, 1, jy_sl.stop - jy_sl.start)

    _set_and_sync(solver, q)

    h_prev = solver.quad_mgr.qf('h_prev')
    h_prev[:] *= 0.9


_REF_SCALE = {'p': 1e5, 'jx': 1e3, 'jy': 1e3, 'theta': 1.0}


def _ref_scale_per_dof(solver: FEMSolver) -> np.ndarray:
    ref = np.empty(solver.res_size)
    for var in solver.variables:
        ref[solver._sol_slices[var]] = _REF_SCALE[var]
    return ref


def compute_fd_jacobian(solver: FEMSolver, eps: float = 1e-4) -> np.ndarray:
    """Central finite difference Jacobian of get_R_ w.r.t. nodal DOFs."""
    q0 = solver.get_q_nodal().copy()
    n = len(q0)
    J_fd = np.zeros((n, n))
    ref = _ref_scale_per_dof(solver)

    for j in range(n):
        eps_j = eps * max(ref[j], abs(q0[j]))

        q_plus = q0.copy()
        q_plus[j] += eps_j
        _set_and_sync(solver, q_plus)
        R_plus = solver.get_R_().copy()

        q_minus = q0.copy()
        q_minus[j] -= eps_j
        _set_and_sync(solver, q_minus)
        R_minus = solver.get_R_().copy()

        J_fd[:, j] = (R_plus - R_minus) / (2.0 * eps_j)

    _set_and_sync(solver, q0)
    return J_fd


def rel_err(M_anal: np.ndarray, M_fd: np.ndarray) -> float:
    return np.linalg.norm(M_anal - M_fd) / (np.linalg.norm(M_fd) + 1e-15)


def _check_term(solver: FEMSolver, term) -> None:
    all_terms = solver.assembly.assembly_terms
    try:
        solver.assembly.assembly_terms = [term]
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver)
        assert np.linalg.norm(J) > 1e-10, f"{term.name}: FD Jacobian is essentially zero"
        err = rel_err(M, J)
        assert err < TOL, f"{term.name}: rel_err={err:.2e} >= {TOL:.0e}"
    finally:
        solver.assembly.assembly_terms = all_terms


# =============================================================================
# Fixture
# =============================================================================

@pytest.fixture(scope='module')
def solver():
    s = _make_solver()
    _init_state(s)
    return s


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.parametrize('term', TERMS, ids=[t.name for t in TERMS])
def test_jacobian(solver, term):
    _check_term(solver, term)


# =============================================================================
# Dev utility (not run by pytest)
# =============================================================================

def compare_matrices(solver: FEMSolver, term, block: str,
                     eps: float = 1e-4, fmt: str = '11.2e') -> None:
    """Print analytic and FD Jacobian side-by-side for one term, one dep-var block."""
    all_terms = solver.assembly.assembly_terms
    try:
        solver.assembly.assembly_terms = [term]
        M = solver.get_M_dense()
        J = compute_fd_jacobian(solver, eps=eps)
    finally:
        solver.assembly.assembly_terms = all_terms

    col_sl = solver._sol_slices.get(block)
    if col_sl is None:
        raise ValueError(f"block={block!r} not in {list(solver._sol_slices)}")

    res_sl = solver._res_slices[term.res]
    row_indices = list(range(res_sl.start, res_sl.stop))
    col_indices = list(range(col_sl.start, col_sl.stop))

    M_blk = M[np.ix_(row_indices, col_indices)]
    J_blk = J[np.ix_(row_indices, col_indices)]
    diff = M_blk - J_blk

    rel = np.linalg.norm(diff) / (np.linalg.norm(J_blk) + 1e-15)
    print(f"\nterm={term.name}  block=d/d{block}  eps={eps:.0e}  rel_err={rel:.3e}")

    entry_w = len(f'{0:{fmt}}')
    gap = '  '
    col_w = entry_w + len(gap)
    row_label_w = max(4, len(str(max(row_indices))))
    col_hdr = ' ' * (row_label_w + 2) + ''.join(f'{c:>{col_w}d}' for c in col_indices)
    sep_line = '-' * len(col_hdr)

    def _row_str(mat, r):
        return ''.join(
            gap + (' ' * entry_w if mat[r, c] == 0.0 else f'{mat[r, c]:{fmt}}')
            for c in range(mat.shape[1])
        )

    for title, mat in [('analytic (M)', M_blk), ('FD (J)', J_blk), ('diff (M-J)', diff)]:
        print(f"\n  {title}")
        print(col_hdr)
        print(sep_line)
        for r, global_r in enumerate(row_indices):
            print(f"  {global_r:<{row_label_w}}{_row_str(mat, r)}")
