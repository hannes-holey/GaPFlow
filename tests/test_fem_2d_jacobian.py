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
for various grid sizes, boundary condition configurations, and with/without
energy equation.
"""
import pytest
import numpy as np
from mpi4py import MPI

from GaPFlow import HAS_PETSC
from GaPFlow.problem import Problem
from GaPFlow.solver_fem_2d import FEMSolver2D

# Skip entire module if running in parallel without PETSc
# (serial execution has SciPy fallback, but parallel requires PETSc)
_parallel_without_petsc = MPI.COMM_WORLD.size > 1 and not HAS_PETSC
pytestmark = pytest.mark.skipif(
    _parallel_without_petsc,
    reason="PETSc required for parallel execution"
)


# =============================================================================
# Configuration Templates
# =============================================================================

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
    equations:
        energy: {energy}
"""

CONFIG_ENERGY_SPEC = """
energy_spec:
    k: 0.1
    cv: 100.0
    wall_flux_model: Tz_Robin
    T_wall: 300.0
    h_Robin: 100.0
    alpha_wall: 0.1
    T0: [uniform, 300.0]
    bc_xW: {bc_E_xW}
    bc_xE: {bc_E_xE}
    bc_yS: {bc_E_yS}
    bc_yN: {bc_E_yN}
    T_bc_xW: 300.0
    T_bc_xE: 330.0
    T_bc_yS: 300.0
    T_bc_yN: 300.0
"""


# =============================================================================
# Helper Functions
# =============================================================================

def make_config(Nx: int, Ny: int, bc_config: dict, energy: bool = False) -> str:
    """Generate configuration string with given grid size and BCs."""
    config = CONFIG_TEMPLATE.format(
        Nx=Nx,
        Ny=Ny,
        xE=bc_config['xE'],
        xW=bc_config['xW'],
        yS=bc_config['yS'],
        yN=bc_config['yN'],
        energy=energy,
    )
    if energy:
        config += CONFIG_ENERGY_SPEC.format(
            bc_E_xE=bc_config.get('bc_E_xE', 'N'),
            bc_E_xW=bc_config.get('bc_E_xW', 'N'),
            bc_E_yS=bc_config.get('bc_E_yS', 'N'),
            bc_E_yN=bc_config.get('bc_E_yN', 'N'),
        )
    return config


def compute_fd_jacobian(solver: FEMSolver2D, problem, eps: float = 1e-6) -> np.ndarray:
    """Compute Jacobian using central finite differences.

    Uses relative perturbation for better accuracy across different variable scales
    (important when energy E has much larger values than rho, jx, jy).
    """
    q0 = solver.get_q_nodal().copy()
    n_dof = len(q0)
    J_fd = np.zeros((n_dof, n_dof))

    for j in range(n_dof):
        # Use relative perturbation for variables with large values
        eps_j = max(eps, eps * abs(q0[j]))

        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[j] += eps_j
        q_minus[j] -= eps_j

        solver.set_q_nodal(q_plus)
        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        R_plus = solver.get_R().copy()

        solver.set_q_nodal(q_minus)
        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        R_minus = solver.get_R().copy()

        J_fd[:, j] = (R_plus - R_minus) / (2 * eps_j)

    # Restore original state
    solver.set_q_nodal(q0)
    problem.decomp.communicate_ghost_buffers(problem)
    solver.update_quad()

    return J_fd


# =============================================================================
# Boundary Condition Configurations
# =============================================================================

BC_CONFIGS = {
    'fully_periodic': {
        'xE': "['P', 'P', 'P']",
        'xW': "['P', 'P', 'P']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
        'bc_E_xE': 'N',
        'bc_E_xW': 'N',
        'bc_E_yS': 'N',
        'bc_E_yN': 'N',
    },
    'all_dirichlet_rho': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['D', 'N', 'N']",
        'yN': "['D', 'N', 'N']",
        'bc_E_xE': 'N',
        'bc_E_xW': 'N',
        'bc_E_yS': 'N',
        'bc_E_yN': 'N',
    },
    'periodic_y': {
        'xE': "['D', 'N', 'N']",
        'xW': "['D', 'N', 'N']",
        'yS': "['P', 'P', 'P']",
        'yN': "['P', 'P', 'P']",
        'bc_E_xE': 'N',
        'bc_E_xW': 'N',
        'bc_E_yS': 'N',
        'bc_E_yN': 'N',
    },
    'periodic_x': {
        'xE': "['P', 'P', 'P']",
        'xW': "['P', 'P', 'P']",
        'yS': "['D', 'N', 'N']",
        'yN': "['D', 'N', 'N']",
        'bc_E_xE': 'N',
        'bc_E_xW': 'N',
        'bc_E_yS': 'N',
        'bc_E_yN': 'N',
    },
}

# Grid sizes to test (include non-square to catch indexing bugs)
GRID_SIZES = [
    (3, 2),   # Minimal grid
    (4, 3),   # Small non-square
    (4, 4),   # Small square
    (5, 4),   # Medium non-square
]


# =============================================================================
# Test Classes - Jacobian Finite Difference Verification
# =============================================================================

class TestJacobianFiniteDifference:
    """Test Jacobian assembly against finite differences."""

    @pytest.mark.parametrize("Nx,Ny", GRID_SIZES)
    @pytest.mark.parametrize("bc_name", ['fully_periodic', 'all_dirichlet_rho', 'periodic_y'])
    def test_jacobian_no_energy(self, Nx: int, Ny: int, bc_name: str):
        """Test Jacobian without energy equation."""
        bc_config = BC_CONFIGS[bc_name]
        config = make_config(Nx, Ny, bc_config, energy=False)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        solver.update_prev_quad()  # Initialize prev values for time derivative terms

        M = solver.get_M_dense()
        nb = solver.nb_inner_pts
        n_vars = len(solver.variables)
        M_inner = M[:, :n_vars * nb]

        J_fd = compute_fd_jacobian(solver, problem)
        J_fd_inner = J_fd[:, :n_vars * nb]

        diff_norm = np.linalg.norm(M_inner - J_fd_inner)
        fd_norm = np.linalg.norm(J_fd_inner)
        rel_err = diff_norm / (fd_norm + 1e-15)

        # Periodic BCs give cleaner results
        tol = 1e-7 if bc_name == 'fully_periodic' else 5e-2
        assert rel_err < tol, (
            f"Jacobian mismatch for Nx={Nx}, Ny={Ny}, bc={bc_name}: "
            f"rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (5, 4)])
    @pytest.mark.parametrize("bc_name", ['fully_periodic', 'all_dirichlet_rho'])
    def test_jacobian_with_energy(self, Nx: int, Ny: int, bc_name: str):
        """Test Jacobian with energy equation."""
        bc_config = BC_CONFIGS[bc_name]
        config = make_config(Nx, Ny, bc_config, energy=True)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        solver.update_prev_quad()

        M = solver.get_M_dense()
        nb = solver.nb_inner_pts
        n_vars = len(solver.variables)
        M_inner = M[:, :n_vars * nb]

        J_fd = compute_fd_jacobian(solver, problem)
        J_fd_inner = J_fd[:, :n_vars * nb]

        diff_norm = np.linalg.norm(M_inner - J_fd_inner)
        fd_norm = np.linalg.norm(J_fd_inner)
        rel_err = diff_norm / (fd_norm + 1e-15)

        tol = 1e-7 if bc_name == 'fully_periodic' else 5e-2
        assert rel_err < tol, (
            f"Energy Jacobian mismatch for Nx={Nx}, Ny={Ny}, bc={bc_name}: "
            f"rel_err={rel_err:.2e}"
        )

    @pytest.mark.parametrize("Nx,Ny", [(4, 3), (5, 4)])
    @pytest.mark.parametrize("bc_name", ['fully_periodic'])
    def test_jacobian_blocks(self, Nx: int, Ny: int, bc_name: str):
        """Test individual Jacobian blocks against FD."""
        bc_config = BC_CONFIGS[bc_name]
        config = make_config(Nx, Ny, bc_config, energy=False)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()
        solver.update_prev_quad()

        M = solver.get_M_dense()
        nb = solver.nb_inner_pts

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
                    assert rel_err < 1e-7, (
                        f"Block M[{res},{var}] mismatch for Nx={Nx}, Ny={Ny}, "
                        f"bc={bc_name}: rel_err={rel_err:.2e}"
                    )
                else:
                    assert block_diff < 1e-10, (
                        f"Block M[{res},{var}] should be zero for Nx={Nx}, "
                        f"Ny={Ny}, bc={bc_name}: ||M||={block_diff:.2e}"
                    )


# =============================================================================
# Test Classes - Shape Function Consistency
# =============================================================================

class TestShapeFunctionConsistency:
    """Test shape function values are consistent with interpolation kernel."""

    def test_left_triangle_shape_functions(self):
        """Verify left triangle shape functions match kernel."""
        from GaPFlow.fem_2d.elements import TriangleQuadrature

        quad = TriangleQuadrature()
        N_left = quad.N_left

        expected = np.array([
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
            [1 / 6, 2 / 3, 1 / 6],
        ])

        assert np.allclose(N_left, expected), (
            f"N_left mismatch:\n{N_left}\nvs expected:\n{expected}"
        )

    def test_right_triangle_shape_functions(self):
        """Verify right triangle shape functions match kernel."""
        from GaPFlow.fem_2d.elements import TriangleQuadrature

        quad = TriangleQuadrature()
        N_right = quad.N_right

        expected = np.array([
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
            [1 / 6, 2 / 3, 1 / 6],
        ])

        assert np.allclose(N_right, expected), (
            f"N_right mismatch:\n{N_right}\nvs expected:\n{expected}"
        )

    def test_shape_functions_partition_of_unity(self):
        """Verify shape functions sum to 1 at each quad point."""
        from GaPFlow.fem_2d.elements import TriangleQuadrature

        quad = TriangleQuadrature()

        for q in range(3):
            sum_left = quad.N_left[:, q].sum()
            sum_right = quad.N_right[:, q].sum()

            assert np.isclose(sum_left, 1.0), (
                f"Left triangle partition of unity failed at quad {q}: sum={sum_left}"
            )
            assert np.isclose(sum_right, 1.0), (
                f"Right triangle partition of unity failed at quad {q}: sum={sum_right}"
            )

    def test_quadrature_weights(self):
        """Verify quadrature weights sum to 1/2 (area of unit triangle)."""
        from GaPFlow.fem_2d.elements import TriangleQuadrature

        quad = TriangleQuadrature()
        weight_sum = quad.weights.sum()

        assert np.isclose(weight_sum, 0.5), (
            f"Quadrature weights should sum to 0.5, got {weight_sum}"
        )


# =============================================================================
# Test Classes - Block Structure
# =============================================================================

class TestJacobianBlockStructure:
    """Test expected structure and symmetry properties of Jacobian blocks."""

    @pytest.mark.parametrize("Nx,Ny", [(4, 3), (5, 4)])
    def test_mass_jx_sparsity(self, Nx: int, Ny: int):
        """Test M[mass, jx] has expected sparsity pattern."""
        bc_config = BC_CONFIGS['all_dirichlet_rho']
        config = make_config(Nx, Ny, bc_config, energy=False)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        M = solver.get_M_dense()
        nb = solver.nb_inner_pts

        j_jx = solver.variables.index('jx')
        i_mass = solver.residuals.index('mass')
        M_mass_jx = M[i_mass * nb:(i_mass + 1) * nb, j_jx * nb:(j_jx + 1) * nb]

        # Each row should have at most a few nonzeros (sparse structure)
        for i in range(nb):
            nnz = np.sum(np.abs(M_mass_jx[i, :]) > 1e-12)
            assert nnz <= 6, (
                f"Row {i} of M[mass,jx] has too many nonzeros: {nnz}"
            )

    @pytest.mark.parametrize("Nx,Ny", [(4, 4), (5, 5)])
    def test_energy_not_coupled_to_mass_momentum(self, Nx: int, Ny: int):
        """Verify M[mass, E] and M[momentum_*, E] are zero."""
        bc_config = BC_CONFIGS['fully_periodic']
        config = make_config(Nx, Ny, bc_config, energy=True)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        M = solver.get_M_dense()
        nb = solver.nb_inner_pts

        i_mass = solver.residuals.index('mass')
        i_mom_x = solver.residuals.index('momentum_x')
        i_mom_y = solver.residuals.index('momentum_y')
        j_E = solver.variables.index('E')

        # Check M[mass, E] is zero
        M_mass_E = M[i_mass * nb:(i_mass + 1) * nb, j_E * nb:(j_E + 1) * nb]
        assert np.linalg.norm(M_mass_E) < 1e-12, (
            f"M[mass, E] should be zero, got norm={np.linalg.norm(M_mass_E):.2e}"
        )

        # Check M[momentum_x, E] is zero
        M_momx_E = M[i_mom_x * nb:(i_mom_x + 1) * nb, j_E * nb:(j_E + 1) * nb]
        assert np.linalg.norm(M_momx_E) < 1e-12, (
            f"M[momentum_x, E] should be zero, got norm={np.linalg.norm(M_momx_E):.2e}"
        )

        # Check M[momentum_y, E] is zero
        M_momy_E = M[i_mom_y * nb:(i_mom_y + 1) * nb, j_E * nb:(j_E + 1) * nb]
        assert np.linalg.norm(M_momy_E) < 1e-12, (
            f"M[momentum_y, E] should be zero, got norm={np.linalg.norm(M_momy_E):.2e}"
        )


# =============================================================================
# Test Classes - Energy Quadrature Field Consistency
# =============================================================================

class TestEnergyQuadFieldConsistency:
    """Test that energy quadrature fields are computed correctly."""

    def test_temperature_field_computed(self):
        """Verify temperature field is computed at quad points."""
        bc_config = BC_CONFIGS['fully_periodic']
        config = make_config(4, 4, bc_config, energy=True)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        T = solver.quad_mgr.get('T')
        assert T.shape[0] == 6, "Should have 6 quad points per square"
        assert np.all(T > 0), "Temperature should be positive"
        assert np.all(T < 1000), "Temperature should be reasonable (<1000K)"

    def test_temperature_gradients_computed(self):
        """Verify temperature gradient fields are computed."""
        bc_config = BC_CONFIGS['fully_periodic']
        config = make_config(4, 4, bc_config, energy=True)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        for name in ['dT_drho', 'dT_djx', 'dT_djy', 'dT_dE']:
            field = solver.quad_mgr.get(name)
            assert field.shape[0] == 6, f"{name} should have 6 quad points"

        # dT_dE should be positive (T increases with E at fixed rho, j)
        dT_dE = solver.quad_mgr.get('dT_dE')
        assert np.all(dT_dE > 0), "dT/dE should be positive"

    def test_wall_heat_flux_computed(self):
        """Verify wall heat flux S and gradients are computed."""
        bc_config = BC_CONFIGS['fully_periodic']
        config = make_config(4, 4, bc_config, energy=True)

        problem = Problem.from_string(config)
        solver = problem.solver
        solver.pre_run()

        problem.decomp.communicate_ghost_buffers(problem)
        solver.update_quad()

        S = solver.quad_mgr.get('S')
        assert S.shape[0] == 6, "S should have 6 quad points"

        for name in ['dS_drho', 'dS_djx', 'dS_djy', 'dS_dE']:
            field = solver.quad_mgr.get(name)
            assert field.shape[0] == 6, f"{name} should have 6 quad points"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
