"""
Cross-solver comparison tests.

Compares solutions from different solvers (explicit, fem_1d, fem_2d) on the same
physical problem to verify consistency.

Problem config files are in /tests/configs/.

Test Cases
----------
1. Inclined Slider Bearing (PL EOS, air)
   - Dirichlet density BC in X, periodic in Y
   - Linear converging gap

2. Journal Bearing (DH EOS, oil)
   - Periodic BC in X and Y
   - Cosine gap profile (journal geometry)

3. Parabolic Slider Bearing (DH EOS, oil)
   - Dirichlet density BC in X, periodic in Y
   - Parabolic gap profile with converging/diverging sections

Notes
-----
- Solutions are compared using [:, 0] slice to match 1D/2D array shapes
- Dirichlet BC cases use inner-region comparison to avoid boundary artifacts
"""
import pytest
import numpy as np
from pathlib import Path

from GaPFlow.problem import Problem


# Path to config templates
CONFIG_DIR = Path(__file__).parent / "configs"

# =============================================================================
# Solver Defaults (use-case independent)
# =============================================================================

SOLVER_DEFAULTS = {
    'explicit': {
        'Ny': 1,
        'solver': 'explicit',
        'max_it': 50000,
    },
    'fem_1d': {
        'Ny': 1,
        'solver': 'fem',
        'dt': 0.1,
        'max_it': 100,
    },
    'fem_2d': {
        'Ny': 4,
        'solver': 'fem',
        'dt': 0.1,
        'max_it': 100,
        'pressure_stab_alpha': 1000,
        'momentum_stab_alpha': 10000,
    },
}

# =============================================================================
# Use-Case Configurations
# =============================================================================

USE_CASES = {
    'inclined_slider': {
        'template': 'inclined_slider',
        'comparison': 'dirichlet',
        'explicit': {'dt': 1e-6},
    },
    'journal_bearing': {
        'template': 'journal_bearing',
        'comparison': 'periodic',
        'explicit': {'dt': 1e-10, 'adaptive': 1, 'CFL': 0.1},
    },
    'parabolic_slider': {
        'template': 'parabolic_slider',
        'comparison': 'dirichlet',
        'rho_bc': 850.0,
        'explicit': {'dt': 1e-10, 'adaptive': 1, 'CFL': 0.45},
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def load_template(name: str) -> str:
    """Load a YAML template from the configs directory."""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, 'r') as f:
        return f.read()


def build_config(use_case: str, solver: str) -> str:
    """Build YAML config string by merging solver defaults with use-case overrides."""
    case = USE_CASES[use_case]
    template = load_template(case['template'])

    # Start with template defaults
    params = {
        'pressure_stab_alpha': 0,
        'momentum_stab_alpha': 0,
        'adaptive': 0,
        'CFL': 0.5,
        'rho_bc': case.get('rho_bc', 1.0),
    }

    # Apply solver defaults
    params.update(SOLVER_DEFAULTS[solver])

    # Apply use-case overrides for this solver
    params.update(case.get(solver, {}))

    return template.format(**params)


def extract_solution(problem) -> tuple[np.ndarray, np.ndarray]:
    """Extract rho and jx arrays from problem.

    Uses [:, 0] slice to get first Y-row, making 2D solutions
    comparable to 1D solutions.
    """
    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()
    return rho, jx


def run_solver(use_case: str, solver: str) -> tuple[np.ndarray, np.ndarray]:
    """Run a single solver configuration and return solution."""
    yaml_str = build_config(use_case, solver)
    problem = Problem.from_string(yaml_str)
    problem.run()
    return extract_solution(problem)


def compare_solutions(sol1: np.ndarray, sol2: np.ndarray, bc_type: str = 'periodic',
                      rtol: float = 0.05, field_name: str = 'field') -> None:
    """Compare two solution arrays with BC-aware tolerance.

    For periodic BC: standard point-wise comparison
    For dirichlet BC: multi-metric comparison (due to artifacts near boundaries)
      1. Inner region (5%-95%) point-wise: rtol
      2. Integral conservation: 2%
      3. Inner-region peak value: rtol
    """
    if bc_type == 'periodic':
        np.testing.assert_allclose(sol1, sol2, rtol=rtol,
                                   err_msg=f"{field_name}: point-wise mismatch")
    else:
        N = len(sol1)
        inner = slice(int(0.05 * N), int(0.95 * N))

        # Inner region point-wise comparison
        np.testing.assert_allclose(
            sol1[inner], sol2[inner], rtol=rtol,
            err_msg=f"{field_name}: inner region (5%-95%) mismatch")

        # Integral conservation
        int1, int2 = np.trapezoid(sol1), np.trapezoid(sol2)
        np.testing.assert_allclose(
            int1, int2, rtol=0.02,
            err_msg=f"{field_name}: integral mismatch ({int1:.4f} vs {int2:.4f})")

        # Inner-region peak value comparison (avoid boundary artifacts)
        np.testing.assert_allclose(
            sol1[inner].max(), sol2[inner].max(), rtol=rtol,
            err_msg=f"{field_name}: inner-region peak value mismatch")


# =============================================================================
# Results Cache (computed once per use case)
# =============================================================================

_RESULTS_CACHE: dict[str, dict] = {}


def get_results(use_case: str) -> dict:
    """Get results for a use case, computing if not cached."""
    if use_case not in _RESULTS_CACHE:
        results = {}
        for solver in ['explicit', 'fem_1d', 'fem_2d']:
            rho, jx = run_solver(use_case, solver)
            results[solver] = {'rho': rho, 'jx': jx}
        _RESULTS_CACHE[use_case] = results
    return _RESULTS_CACHE[use_case]


# =============================================================================
# Parametrized Test Class
# =============================================================================

@pytest.mark.parametrize("use_case", list(USE_CASES.keys()))
class TestSolverComparison:
    """Compare solvers on various bearing problems."""

    @pytest.fixture
    def results(self, use_case):
        """Get cached results for this use case."""
        return get_results(use_case)

    def test_explicit_vs_fem_1d_rho(self, results, use_case):
        """Density should match between explicit and fem_1d."""
        bc_type = USE_CASES[use_case]['comparison']
        compare_solutions(results['explicit']['rho'], results['fem_1d']['rho'],
                          bc_type=bc_type, field_name='rho (explicit vs fem_1d)')

    def test_explicit_vs_fem_1d_jx(self, results, use_case):
        """Mass flux should match between explicit and fem_1d."""
        bc_type = USE_CASES[use_case]['comparison']
        compare_solutions(results['explicit']['jx'], results['fem_1d']['jx'],
                          bc_type=bc_type, field_name='jx (explicit vs fem_1d)')

    def test_explicit_vs_fem_2d_rho(self, results, use_case):
        """Density should match between explicit and fem_2d."""
        bc_type = USE_CASES[use_case]['comparison']
        compare_solutions(results['explicit']['rho'], results['fem_2d']['rho'],
                          bc_type=bc_type, field_name='rho (explicit vs fem_2d)')

    def test_explicit_vs_fem_2d_jx(self, results, use_case):
        """Mass flux should match between explicit and fem_2d."""
        bc_type = USE_CASES[use_case]['comparison']
        compare_solutions(results['explicit']['jx'], results['fem_2d']['jx'],
                          bc_type=bc_type, field_name='jx (explicit vs fem_2d)')

    def test_solution_shape(self, results, use_case):
        """All solutions should have the same shape."""
        shapes = {name: res['rho'].shape for name, res in results.items()}
        assert len(set(shapes.values())) == 1, f"Shape mismatch: {shapes}"

    def test_density_positive(self, results, use_case):
        """Density should be positive."""
        for solver, res in results.items():
            assert np.all(res['rho'] > 0), f"{solver}: negative density found"
