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


def run_reynolds_cav_solver(template, solver, topo_fn=None, **overrides):
    """Like run_reynolds_solver, but also extracts theta (cavitation
    fraction). p and theta are looked up by name via
    problem.solver.quad_mgr.nf, which indexes nodal fields by name rather
    than DOF position (position differs between fem_ns and fem_reynolds).

    topo_fn, if given, is called as topo_fn(problem) after Problem.from_string
    and before problem.run(), to inject a height field that isn't expressible
    via geometry.type (e.g. regen_twin_parabolic_slider_id)."""
    print(f"Running solver: {solver}")
    yaml_str = build_reynolds_config(template, solver, **overrides)
    problem = Problem.from_string(yaml_str)
    if topo_fn is not None:
        topo_fn(problem)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    nf = problem.solver.quad_mgr.nf
    p = nf('p')[1:-1, 0].copy()
    theta = nf('theta')[1:-1, 0].copy()
    return p, theta, elapsed


def run_reynolds_cav_comparison(template, topo_fn=None, **common_overrides):
    results = {}
    for solver in ['fem_ns', 'fem_reynolds']:
        p, theta, elapsed = run_reynolds_cav_solver(template, solver, topo_fn=topo_fn, **common_overrides)
        results[solver] = {'p': p, 'theta': theta, 'time': elapsed}
    return results


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


# =============================================================================
# FEM NS stabilization comparison (OSS / SUPG / both) vs FEM Reynolds (SUPG)
# =============================================================================

STAB_VARIANTS = {
    'ns_oss': {'reynolds': False, 'oss': True, 'mass_supg': False},
    'ns_supg': {'reynolds': False, 'oss': False, 'mass_supg': True},
    'ns_both': {'reynolds': False, 'oss': True, 'mass_supg': True},
    'rey_supg': {'reynolds': True, 'oss': False, 'mass_supg': True},
}


def build_stab_config(template, variant, Nx, **overrides):
    """Build a fem_solver config for `variant` in STAB_VARIANTS, at a given Nx.

    Fills the template's {solver}/{Nx}/{oss}/{mass_supg}/... placeholders,
    then extends fem_solver.equations.reynolds in the parsed dict (as in
    build_reynolds_config) so the reynolds flag doesn't need its own
    placeholder in the shared YAML template.
    """
    flags = STAB_VARIANTS[variant]
    params = {'solver': 'fem', 'Nx': Nx, 'oss': flags['oss'], 'mass_supg': flags['mass_supg']}
    params.update(overrides)
    yaml_str = template.format(**params)

    config = yaml.safe_load(yaml_str)
    config.setdefault('fem_solver', {}).setdefault('equations', {})['reynolds'] = flags['reynolds']
    return yaml.safe_dump(config)


def run_stab_solver(template, variant, Nx, topo_fn=None, **overrides):
    """Run one stabilization variant at a given Nx.

    topo_fn, if given, is called as topo_fn(problem) after Problem.from_string
    and before problem.run(), to inject a height field that isn't expressible
    via geometry.type (e.g. regen_twin_parabolic_slider_id). p and theta are
    looked up by name via problem.solver.quad_mgr.nf, since DOF position
    differs between the NS and Reynolds variables lists.
    """
    print(f"Running variant: {variant} (Nx={Nx})")
    yaml_str = build_stab_config(template, variant, Nx, **overrides)
    problem = Problem.from_string(yaml_str)
    if topo_fn is not None:
        topo_fn(problem)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    nf = problem.solver.quad_mgr.nf
    p = nf('p')[1:-1, 0].copy()
    theta = nf('theta')[1:-1, 0].copy()
    return p, theta, elapsed


def run_stab_comparison(template, Nx, topo_fn=None, **common_overrides):
    """Run all STAB_VARIANTS at a given Nx and collect results."""
    results = {}
    for variant in STAB_VARIANTS:
        p, theta, elapsed = run_stab_solver(template, variant, Nx, topo_fn=topo_fn, **common_overrides)
        results[variant] = {'p': p, 'theta': theta, 'time': elapsed, 'Nx': Nx}
    return results


def print_stab_timing(results):
    print("Timing:")
    for variant in STAB_VARIANTS:
        print(f"  {variant:<10} {results[variant]['time']:>6.2f} s")
