import time
from pathlib import Path

import yaml

from GaPFlow.models.pressure import eos_pressure
from GaPFlow.problem import Problem

CONFIG_DIR = Path('../../config_lib/validation')


def load_template(name):
    with open(CONFIG_DIR / f'{name}.yaml', 'r') as f:
        return f.read()


def apply_overrides(config, overrides):
    """Set keys in a nested config dict by full dotted path, e.g.
    'numerics.dt' or 'fem_solver.stabilization.oss'. Missing sections
    and keys along the path are created.
    """
    for key, value in overrides.items():
        *parents, leaf = key.split('.')
        node = config
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return config


def build_config(template, solver, **overrides):
    config = yaml.safe_load(template)
    apply_overrides(config, {'numerics.solver': solver, **overrides})
    return yaml.safe_dump(config)


def run_solver(template, solver, **overrides):
    yaml_str = build_config(template, solver, **overrides)
    problem = Problem.from_string(yaml_str)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0
    rho = problem.q[0][:, 0].copy()
    jx = problem.q[1][:, 0].copy()
    return rho, jx, elapsed


def run_pressure_profile(template, **overrides):
    yaml_str = build_config(template, 'implicit', **overrides)
    problem = Problem.from_string(yaml_str)
    problem.run()
    p = eos_pressure(problem.q[0][:, 1:-1], problem.prop).mean(axis=1)
    return problem.topo.xx[:, 0] / problem.grid['Lx'], p


def get_profile_1d(problem):
    """Normalized position, pressure [Pa] and density of a finished 1D run."""
    X = problem.topo.xx[1:-1, 1] / problem.grid['Lx']
    return X, problem.pressure.pressure[1:-1, 1].copy(), problem.q[0][1:-1, 1].copy()


def run_all_solvers(template, explicit_overrides=None, skip=None, **common_overrides):
    explicit_overrides = explicit_overrides or {}
    results = {}
    for solver in ['explicit', 'implicit']:
        if skip and solver in skip:
            continue
        overrides = {**common_overrides, **(explicit_overrides if solver == 'explicit' else {})}
        rho, jx, elapsed = run_solver(template, solver, **overrides)
        results[solver] = {'rho': rho, 'jx': jx, 'time': elapsed}
    return results


def print_timing(results):
    print("Timing:")
    for name, res in results.items():
        print(f"  {name:<10} {res['time']:>6.2f} s")


# =============================================================================
# Navier-Stokes (full momentum) vs Reynolds (scalar p-only) comparison
# =============================================================================

def run_reynolds_solver(template, reynolds, topo_fn=None, **overrides):
    overrides = {'implicit_solver.equations.reynolds': reynolds, **overrides}
    problem = Problem.from_string(build_config(template, 'implicit', **overrides))
    if topo_fn is not None:
        topo_fn(problem)
    t0 = time.perf_counter()
    problem.run()
    elapsed = time.perf_counter() - t0

    nf = problem.solver.quad_mgr.nf
    rho_eff = problem.q[0].copy()
    if problem.fem_solver['equations']['cavitation']:
        rho_eff *= 1 - nf('theta')
    return {'X': problem.topo.xx[:, 0] / problem.grid['Lx'], 'p': nf('p').copy(), 'rho_eff': rho_eff,
            'Lx': problem.grid['Lx'], 'Ly': problem.grid['Ly'], 'time': elapsed}


def run_reynolds_comparison(template, topo_fn=None, **common_overrides):
    return {name: run_reynolds_solver(template, name == 'reynolds', topo_fn=topo_fn, **common_overrides)
            for name in ['ns', 'reynolds']}


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
