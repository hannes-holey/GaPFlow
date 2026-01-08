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
"""
Finite difference verification tests for 2D FEM Jacobian assembly.

Tests the assembled Jacobian matrix against central finite differences
for various grid sizes and boundary condition configurations.
"""
import pytest
import numpy as np
from GaPFlow.problem import Problem

from GaPFlow.solver_fem_2d import FEMSolver2D  # type: ignore

# Base configuration template
CONFIG_TEMPLATE = """
options:
    output: /tmp/fem2d_jacobian_test
    write_freq: 1000
    silent: True

grid:
    Lx: 0.1
    Ly: 0.1
    Nx: {Nx}
    Ny: {Ny}
    xE: {xE}
    xW: {xW}
    yS: {yS}
    yN: {yN}
    xE_D: 1.1
    xW_D: 1.0
    yS_D: 1.05
    yN_D: 1.05

geometry:
    type: inclined
    hmax: 1e-5
    hmin: 1e-5
    U: 0.0
    V: 0.0

numerics:
    solver: fem
    dt: 1e-8
    tol: 1e-6
    max_it: 100

properties:
    EOS: PL
    rho0: 1.0
    shear: 1e-3
    bulk: 0.
    P0: 101325
    alpha: 0.

fem_solver:
    type: newton_alpha
    dynamic: False
    equations:
        term_list: ['R11x', 'R11y', 'R21x', 'R21y', 'R24x', 'R24y']
"""


def make_config(Nx: int, Ny: int, bc_config: dict) -> str:
    """Generate configuration string with given grid size and BCs."""
    return CONFIG_TEMPLATE.format(
        Nx=Nx,
        Ny=Ny,
        xE=bc_config['xE'],
        xW=bc_config['xW'],
        yS=bc_config['yS'],
        yN=bc_config['yN'],
    )


def compute_fd_jacobian(solver: FEMSolver2D, problem, eps: float = 1e-6) -> np.ndarray:
    """Compute Jacobian using central finite differences."""
    q0 = solver.get_q_nodal().copy()
    n_dof = len(q0)
    J_fd = np.zeros((n_dof, n_dof))

    for j in range(n_dof):
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[j] += eps
        q_minus[j] -= eps

        solver.set_q_nodal(q_plus)
        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        R_plus = solver.get_R().copy()

        solver.set_q_nodal(q_minus)
        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        R_minus = solver.get_R().copy()

        J_fd[:, j] = (R_plus - R_minus) / (2 * eps)

    # Restore original state
    solver.set_q_nodal(q0)
    problem.decomp.communicate_ghost_buffers(problem)
    solver.update_quad()

    return J_fd


# Boundary condition configurations to test
BC_CONFIGS = {
    'all_dirichlet_rho': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['D', 'N', 'N']",
        'yN': "['D', 'N', 'N']",
    },
    'periodic_y': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
    },
    'periodic_x': {
        'xE': "['P', 'P', 'P']",
        'xW': "['P', 'P', 'P']",
        'yS': "['D', 'N', 'N']",
        'yN': "['D', 'N', 'N']",
    },
}

# Grid sizes to test
GRID_SIZES = [
    (3, 2),   # Minimal grid
    (4, 3),   # Small grid
    (5, 4),   # Medium grid
    (6, 5),   # Larger grid
]


class TestJacobianFiniteDifference:
    """Test Jacobian assembly against finite differences."""

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    @pytest.mark.parametrize("bc_name", list(BC_CONFIGS.keys()))
    def test_jacobian_full(self, Nx: int, Ny: int, bc_name: str):
        """Test full Jacobian matrix against FD."""
        bc_config = BC_CONFIGS[bc_name]
        config = make_config(Nx, Ny, bc_config)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        # Assembled Jacobian (use dense version for testing)
        M = solver.get_M_dense()
        nb = solver.nb_inner_pts

        # Extract inner DOFs only (exclude Dirichlet ghost columns)
        n_vars = len(solver.variables)
        M_inner = M[:, :n_vars * nb]

        # Finite difference Jacobian
        J_fd = compute_fd_jacobian(solver, problem)
        J_fd_inner = J_fd[:, :n_vars * nb]

        # Compare
        diff_norm = np.linalg.norm(M_inner - J_fd_inner)
        fd_norm = np.linalg.norm(J_fd_inner)
        rel_err = diff_norm / (fd_norm + 1e-15)

        assert rel_err < 1e-8, (
            f"Jacobian mismatch for Nx={Nx}, Ny={Ny}, bc={bc_name}: "
            f"rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    @pytest.mark.parametrize("bc_name", list(BC_CONFIGS.keys()))
    def test_jacobian_blocks(self, Nx: int, Ny: int, bc_name: str):
        """Test individual Jacobian blocks against FD."""
        bc_config = BC_CONFIGS[bc_name]
        config = make_config(Nx, Ny, bc_config)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        # Assembled Jacobian (use dense version for testing)
        M = solver.get_M_dense()
        nb = solver.nb_inner_pts

        # Finite difference Jacobian
        J_fd = compute_fd_jacobian(solver, problem)

        # Check each block
        for i, res in enumerate(solver.residuals):
            for j, var in enumerate(solver.variables):
                M_block = M[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb]
                J_block = J_fd[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb]

                block_diff = np.linalg.norm(M_block - J_block)
                block_norm = np.linalg.norm(J_block)

                if block_norm > 1e-10:
                    rel_err = block_diff / block_norm
                    assert rel_err < 1e-8, (
                        f"Block M[{res},{var}] mismatch for Nx={Nx}, Ny={Ny}, "
                        f"bc={bc_name}: rel_err={rel_err:.2e}"
                    )
                else:
                    # If FD block is ~zero, assembled should be too
                    assert block_diff < 1e-10, (
                        f"Block M[{res},{var}] should be zero for Nx={Nx}, "
                        f"Ny={Ny}, bc={bc_name}: ||M||={block_diff:.2e}"
                    )


class TestShapeFunctionConsistency:
    """Test shape function values are consistent with interpolation kernel."""

    # Expected kernel values from get_triangle_3_operator
    # kernel layout: [[bl, tl], [br, tr]] for each quad point
    # quad 0-2: left triangle, quad 3-5: right triangle
    KERNEL_VALUES = np.array([
        # Left triangle quad points (0, 1, 2)
        [[[2 / 3, 1 / 6], [1 / 6, 0]]],   # quad0: bl=2/3, tl=1/6, br=1/6, tr=0
        [[[1 / 6, 1 / 6], [2 / 3, 0]]],   # quad1: bl=1/6, tl=1/6, br=2/3, tr=0
        [[[1 / 6, 2 / 3], [1 / 6, 0]]],   # quad2: bl=1/6, tl=2/3, br=1/6, tr=0
        # Right triangle quad points (3, 4, 5)
        [[[0, 1 / 6], [1 / 6, 2 / 3]]],   # quad3: bl=0, tl=1/6, br=1/6, tr=2/3
        [[[0, 2 / 3], [1 / 6, 1 / 6]]],   # quad4: bl=0, tl=2/3, br=1/6, tr=1/6
        [[[0, 1 / 6], [2 / 3, 1 / 6]]],   # quad5: bl=0, tl=1/6, br=2/3, tr=1/6
    ])

    def test_left_triangle_shape_functions(self):
        """Verify left triangle shape functions match kernel.

        Left triangle node order: [bl, tl, br]
        """
        from GaPFlow.fem.utils2d import get_N_left_test_vals

        N_left = get_N_left_test_vals()
        kernel = self.KERNEL_VALUES

        # Check at each quad point (0, 1, 2)
        for q in range(3):
            N_bl_expected = kernel[q, 0, 0, 0]  # bl position
            N_tl_expected = kernel[q, 0, 0, 1]  # tl position
            N_br_expected = kernel[q, 0, 1, 0]  # br position

            assert np.isclose(N_left[0, q], N_bl_expected), (
                f"N_bl mismatch at quad {q}: {N_left[0, q]} vs {N_bl_expected}"
            )
            assert np.isclose(N_left[1, q], N_tl_expected), (
                f"N_tl mismatch at quad {q}: {N_left[1, q]} vs {N_tl_expected}"
            )
            assert np.isclose(N_left[2, q], N_br_expected), (
                f"N_br mismatch at quad {q}: {N_left[2, q]} vs {N_br_expected}"
            )

    def test_right_triangle_shape_functions(self):
        """Verify right triangle shape functions match kernel.

        Right triangle node order: [tr, br, tl]
        """
        from GaPFlow.fem.utils2d import get_N_right_test_vals

        N_right = get_N_right_test_vals()
        kernel = self.KERNEL_VALUES

        # Check at each quad point (3, 4, 5) mapped to indices (0, 1, 2) in N_right
        for i, q in enumerate([3, 4, 5]):
            N_tr_expected = kernel[q, 0, 1, 1]  # tr position
            N_br_expected = kernel[q, 0, 1, 0]  # br position
            N_tl_expected = kernel[q, 0, 0, 1]  # tl position

            assert np.isclose(N_right[0, i], N_tr_expected), (
                f"N_tr mismatch at quad {q}: {N_right[0, i]} vs {N_tr_expected}"
            )
            assert np.isclose(N_right[1, i], N_br_expected), (
                f"N_br mismatch at quad {q}: {N_right[1, i]} vs {N_br_expected}"
            )
            assert np.isclose(N_right[2, i], N_tl_expected), (
                f"N_tl mismatch at quad {q}: {N_right[2, i]} vs {N_tl_expected}"
            )

    def test_shape_functions_partition_of_unity(self):
        """Verify shape functions sum to 1 at each quad point."""
        from GaPFlow.fem.utils2d import (
            get_N_left_test_vals,
            get_N_right_test_vals,
        )

        N_left = get_N_left_test_vals()
        N_right = get_N_right_test_vals()

        # Sum over nodes at each quad point should be 1
        for q in range(3):
            sum_left = N_left[0, q] + N_left[1, q] + N_left[2, q]
            sum_right = N_right[0, q] + N_right[1, q] + N_right[2, q]

            assert np.isclose(sum_left, 1.0), (
                f"Left triangle partition of unity failed at quad {q}: sum={sum_left}"
            )
            assert np.isclose(sum_right, 1.0), (
                f"Right triangle partition of unity failed at quad {q}: sum={sum_right}"
            )


class TestJacobianSymmetry:
    """Test expected symmetry properties of Jacobian blocks."""

    @pytest.mark.parametrize("Nx,Ny", [(4, 3), (5, 4)])
    def test_mass_jx_structure(self, Nx: int, Ny: int):
        """Test M[mass, jx] has expected sparsity pattern."""
        bc_config = BC_CONFIGS['all_dirichlet_rho']
        config = make_config(Nx, Ny, bc_config)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        M = solver.get_M_dense()
        nb = solver.nb_inner_pts

        # M[mass, jx] block
        j_jx = solver.variables.index('jx')
        i_mass = solver.residuals.index('mass')
        M_mass_jx = M[i_mass * nb:(i_mass + 1) * nb, j_jx * nb:(j_jx + 1) * nb]

        # Each row should have at most a few nonzeros (sparse structure)
        for i in range(nb):
            nnz = np.sum(np.abs(M_mass_jx[i, :]) > 1e-12)
            assert nnz <= 6, (
                f"Row {i} of M[mass,jx] has too many nonzeros: {nnz}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
