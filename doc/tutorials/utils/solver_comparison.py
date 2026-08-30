import time
from pathlib import Path

import yaml

from GaPFlow.problem import Problem

CONFIG_DIR = Path('../../tests/configs')

SOLVER_DEFAULTS = {
    'explicit': {'Ny': 1, 'solver': 'explicit', 'max_it': 50000, 'output_plots': False},
    'solver_fem': {'Ny': 4, 'solver': 'fem', 'dt': 1e-02, 'max_it': 100, 'output_plots': True},
}


def load_template(name):
    with open(CONFIG_DIR / f'{name}.yaml', 'r') as f:
        return f.read()


def build_config(template, solver, **overrides):
    params = {'adaptive': 0, 'CFL': 0.5, 'rho_bc': 1.0}
    params.update(SOLVER_DEFAULTS[solver])
    params.update(overrides)
    return template.format(**params)


def run_solver(template, solver, **overrides):
    yaml_str = build_config(template, solver, **overrides)
    problem = Problem.from_string(yaml_str)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    rho = problem.q[0][1:-1, 0].copy()
    jx = problem.q[1][1:-1, 0].copy()
    return rho, jx, elapsed


def run_all_solvers(template, explicit_overrides=None, **common_overrides):
    explicit_overrides = explicit_overrides or {}
    results = {}
    for solver in ['explicit', 'solver_fem']:
        overrides = {**common_overrides, **(explicit_overrides if solver == 'explicit' else {})}
        rho, jx, elapsed = run_solver(template, solver, **overrides)
        results[solver] = {'rho': rho, 'jx': jx, 'time': elapsed}
    return results


def print_timing(results):
    print("Timing:")
    for solver in ['explicit', 'solver_fem']:
        print(f"  {solver:<10} {results[solver]['time']:>6.2f} s")


# =============================================================================
# FEM NS (full momentum) vs FEM Reynolds (scalar p-only) comparison
# =============================================================================

REYNOLDS_SOLVER_DEFAULTS = {
    'fem_ns': {'solver': 'fem'},
    'fem_reynolds': {'solver': 'fem'},
}


def build_reynolds_config(template, solver, **overrides):
    """Build a fem_solver config for `solver` in {'fem_ns', 'fem_reynolds'}.

    Fills the template's {solver}/{rho_bc}/... placeholders as usual, then
    parses the result and extends fem_solver.equations.reynolds in the parsed
    dict (rather than depending on a {reynolds} placeholder in the YAML text)
    so the shared config templates in tests/configs/ stay untouched.
    """
    params = {'rho_bc': 1.0}
    params.update(REYNOLDS_SOLVER_DEFAULTS[solver])
    params.update(overrides)
    yaml_str = template.format(**params)

    config = yaml.safe_load(yaml_str)
    config.setdefault('fem_solver', {}).setdefault('equations', {})['reynolds'] = (
        solver == 'fem_reynolds')
    return yaml.safe_dump(config)


def run_reynolds_solver(template, solver, **overrides):
    print(f"Running solver: {solver}")
    yaml_str = build_reynolds_config(template, solver, **overrides)
    problem = Problem.from_string(yaml_str)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    rho = problem.q[0][1:-1, 0].copy()
    return rho, elapsed


def run_reynolds_comparison(template, **common_overrides):
    results = {}
    for solver in ['fem_ns', 'fem_reynolds']:
        rho, elapsed = run_reynolds_solver(template, solver, **common_overrides)
        results[solver] = {'rho': rho, 'time': elapsed}
    return results


def print_reynolds_timing(results):
    print("Timing:")
    for solver in ['fem_ns', 'fem_reynolds']:
        print(f"  {solver:<20} {results[solver]['time']:>6.2f} s")


def run_reynolds_solver_2d(template, solver, **overrides):
    """Like run_reynolds_solver, but keeps the full 2D rho field (no y=0 slice)
    for genuinely 2D (non-periodic-in-y) geometries."""
    print(f"Running solver: {solver}")
    yaml_str = build_reynolds_config(template, solver, **overrides)
    problem = Problem.from_string(yaml_str)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    rho = problem.q[0][1:-1, 1:-1].copy()
    return rho, elapsed, problem.grid['Lx'], problem.grid['Ly']


def run_reynolds_comparison_2d(template, **common_overrides):
    results = {}
    for solver in ['fem_ns', 'fem_reynolds']:
        rho, elapsed, Lx, Ly = run_reynolds_solver_2d(template, solver, **common_overrides)
        results[solver] = {'rho': rho, 'time': elapsed, 'Lx': Lx, 'Ly': Ly}
    return results
