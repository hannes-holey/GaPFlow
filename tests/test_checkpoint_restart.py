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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Registry-driven checkpoint round-trip test.

Rather than stepping a simulation and comparing trajectories (slow, and only
exercises whatever state a given run happens to touch), this fills every field
named in `CHECKPOINT_FIELD_REGISTRY` with known random data, saves, loads into
a freshly constructed `Problem`, and asserts each field round-trips exactly.
Walking the registry directly (instead of hardcoding a field list) means a new
registry entry is automatically covered without touching this test.

A single config with both elastic deformation and force-balance enabled is
used so every non-energy registry entry actually resolves to a real value
(the FEM solver's Newton loop itself is never run -- `_pre_run()` is enough to
allocate all fields and controller state).
"""

import numpy as np
import pytest

from GaPFlow import Problem
from GaPFlow.solver_fem.checkpoint import CHECKPOINT_FIELD_REGISTRY, _resolve, _try_resolve

CONFIG = """
options:
    output: /tmp/checkpoint_restart_test
    write_freq: 1000
    save_output: False

grid:
    Lx: 0.002
    Ly: 0.002
    Nx: 8
    Ny: 8
    xW: ['D', 'N', 'N']
    xE: ['N', 'N', 'N']
    yS: ['N', 'N', 'N']
    yN: ['N', 'N', 'N']
    xW_D: 850.0
    xE_D: 850.0

geometry:
    type: circular_contact
    Rx: 15.0e-3
    Ry: 15.0e-3
    hmin: 1.0e-6
    U: 10.
    flip: false

numerics:
    solver: fem
    dt: 1.
    tol: 0.
    max_it: 1

properties:
    EOS: Bayada
    rho0: 850.
    rho_l: 850.0
    rho_v: 0.019
    c_l: 1600.0
    c_v: 352.0
    shear: 0.039
    bulk: 0.
    elastic:
        enabled: true
        E: 210.0e9
        nu: 0.3
        alpha_underrelax: 0.1
        reference_point: [0, 0]
        n_images: 0

force_balance:
    force: 25
    pid_hold_tol: 0.05
    init_dry_contact:
        enabled: false
    rigid_height_variation:
        enabled: true
        method: bisection
        init_min_height:
        ambient_pressure: 0.0
        force_tol: 0.01
        h_min_step_divisor: 50.0

fem_solver:
    linear_solver: direct
    newton_relax: 0.2
    physics:
        gap_shear: true
        plane_shear: true
        inertia: false
    max_iter: 1
    R_norm_tol: 1.0e-9
    scaling: false
"""


def _make_problem():
    p = Problem.from_string(CONFIG)
    p._pre_run()
    return p


def _randomize_registry_state(solver, rng):
    """Overwrite every field named in the registry with known random data.

    Returns {key: value} of what was written, for later comparison. Skips
    entries whose source doesn't resolve (e.g. energy-only fields, which
    this config never activates).
    """
    written = {}
    for key, entry in CHECKPOINT_FIELD_REGISTRY.items():
        target = _try_resolve(solver, entry['source'])
        if target is None:
            continue
        field_type = entry['type']
        if field_type in ('nodal', 'nodal_P2', 'field', 'field_inplace'):
            random_values = rng.uniform(-1.0, 1.0, size=target.shape)
            target[...] = random_values
            written[key] = random_values.copy()
        else:
            # Scalars are plain Python attributes -- set them directly rather
            # than mutating the resolved value in place.
            if key == 'residual_buffer':
                value = [float(x) for x in rng.uniform(-1.0, 1.0, size=5)]
            elif key == 'rhv_history':
                value = [float(x) for x in rng.uniform(-1.0, 1.0, size=3)]
            elif key == 'fb_bisect_phase':
                value = 'bisect'
            elif key == 'step':
                value = int(rng.integers(0, 1000))
            else:
                value = float(rng.uniform(-1.0, 1.0))
            _set_scalar(solver, entry['source'], value)
            written[key] = value

    solver.problem.topo.update(suspend_fb=True, suspend_def=True)
    written['h'] = solver.problem.topo.h.copy()
    written['deformation'] = solver.problem.topo.deformation.copy()

    return written


def _set_scalar(solver, source, value):
    from GaPFlow.solver_fem.checkpoint import _resolve_parent
    obj, attr = _resolve_parent(solver, source)
    setattr(obj, attr, value)


def test_checkpoint_round_trips_every_registered_field():
    rng = np.random.default_rng(0)

    p_save = _make_problem()
    written = _randomize_registry_state(p_save.solver, rng)
    assert written, "no registry entries resolved -- config no longer exercises checkpoint state"

    p_save.save_state("/tmp/ckpt_registry_roundtrip.npz")
    del p_save

    p_load = Problem.from_string(CONFIG)
    p_load.load_state("/tmp/ckpt_registry_roundtrip.npz")

    for key, expected in written.items():
        entry = CHECKPOINT_FIELD_REGISTRY[key]
        actual = _resolve(p_load.solver, entry['source'])
        field_type = entry['type']

        if field_type in ('nodal', 'nodal_P2'):
            actual = actual[0]
        if isinstance(expected, np.ndarray):
            diff = np.abs(actual - expected).max()
            assert diff == 0.0, f"'{key}' not bit-exact after round trip: max diff {diff}"
        elif isinstance(expected, list):
            assert list(actual) == pytest.approx(expected)
        elif isinstance(expected, str):
            assert actual == expected
        else:
            assert actual == pytest.approx(expected)


def test_checkpoint_skips_unresolvable_fields():
    """Fields whose source doesn't resolve on this config (e.g. energy DOFs,
    never activated here) are silently absent from the checkpoint file, not
    written as garbage."""

    p_save = _make_problem()
    p_save.save_state("/tmp/ckpt_registry_missing.npz")

    npz = np.load("/tmp/ckpt_registry_missing.npz")
    for key in ('nodal__E', 'nodal__theta', 'nodal__xi'):
        assert key not in npz.files, f"'{key}' unexpectedly present in checkpoint"
