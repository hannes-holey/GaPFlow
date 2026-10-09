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

# flake8: noqa: W503

"""Warm-restart checkpointing for the FEM solver.

`save_state`/`load_state` persist everything needed to resume a simulation:
Newton solution variables (nodal DOFs `jx`, `jy`, `p`, ...), elastic/viscosity
state, force-balance state, and scalar time-stepping/controller state.
Independent of the regular sol.nc / topo.nc output.

`CHECKPOINT_FIELD_REGISTRY` is the single source of truth for what is saved:
each entry names one piece of state by its dotted `source` path off `solver`,
and a `type` -- 'field'/'field_P2' for a P1/P2 array (gathered/scattered with
ghosts via `DomainDecomposition.gather_global_padded`/`scatter_global_padded`,
so no ghost re-derivation is needed after restore), or 'scalar' for a plain
value written directly into the .npz.

Presence, not a separate condition flag, decides whether an entry
participates: if `source` doesn't resolve (missing attribute/dict key) or
resolves to None, save skips it; on load, a key missing from the file is
skipped too. E.g. `topo.deformation` only exists once elastic deformation is
enabled, `nodal_fields['E']` only exists once the energy equation is active --
no separate `elastic`/`force_balance` flags need to be tracked.
"""

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .solver_fem import FEMSolver

NDArray = npt.NDArray[np.floating]


CHECKPOINT_FIELD_REGISTRY = {
    # Newton DOFs
    'nodal__jx': {'type': 'nodal_P2', 'source': 'quad_mgr.nodal_fields["jx"].pg'},
    'nodal__jy': {'type': 'nodal_P2', 'source': 'quad_mgr.nodal_fields["jy"].pg'},
    'nodal__p': {'type': 'nodal', 'source': 'quad_mgr.nodal_fields["p"].pg'},
    'nodal__E': {'type': 'nodal', 'source': 'quad_mgr.nodal_fields["E"].pg'},
    'nodal__theta': {'type': 'nodal', 'source': 'quad_mgr.nodal_fields["theta"].pg'},
    'nodal__xi': {'type': 'nodal', 'source': 'quad_mgr.nodal_fields["xi"].pg'},

    # topography / elastic / viscosity state
    'h': {'type': 'field', 'source': 'problem.topo.h'},
    'deformation': {'type': 'field', 'source': 'problem.topo.deformation'},
    'h_undeformed': {'type': 'field', 'source': 'problem.topo.h_undeformed'},
    'u_prev': {'type': 'field', 'source': 'problem.topo.ElasticDeformation.u_prev'},
    'eta_prev': {'type': 'field', 'source': 'problem.viscosity.eta_prev'},
    'shear_viscosity': {'type': 'field_inplace', 'source': 'problem.viscosity.shear_viscosity'},

    # force balance
    'h0': {'type': 'scalar', 'source': 'problem.topo.h0'},
    'rhv_history': {'type': 'scalar', 'source': 'problem.topo.rhv_history'},
    'fb_h0_prev': {'type': 'scalar', 'source': 'problem.topo._fb_controller._h0_prev'},
    'fb_bisect_phase': {'type': 'scalar', 'source': 'problem.topo._fb_controller.controller.phase'},
    'fb_bisect_step': {'type': 'scalar', 'source': 'problem.topo._fb_controller.controller.step'},
    'fb_bisect_last_dh0': {'type': 'scalar',
                           'source': 'problem.topo._fb_controller.controller.last_dh0'},
    'fb_bisect_last_residual_sign': {'type': 'scalar',
                                     'source': 'problem.topo._fb_controller.controller.last_residual_sign'},
    'fb_bisect_sum_alpha': {'type': 'scalar',
                            'source': 'problem.topo._fb_controller.controller.sum_alpha'},
    'fb_bisect_h_min_step_divisor': {'type': 'scalar',
                                     'source': 'problem.topo._fb_controller.controller.h_min_step_divisor'},

    # time-stepping / convergence bookkeeping
    'step': {'type': 'scalar', 'source': 'problem.step'},
    'simtime': {'type': 'scalar', 'source': 'problem.simtime'},
    'dt': {'type': 'scalar', 'source': 'problem.dt'},
    'residual': {'type': 'scalar', 'source': 'problem.residual'},
    'residual_buffer': {'type': 'scalar', 'source': 'problem.residual_buffer'},
    'kinetic_energy_old': {'type': 'scalar', 'source': 'problem.kinetic_energy_old'},
}


def _walk(obj, part: str):
    """Advance one dotted-path segment, e.g. 'nodal_fields["jx"]' or 'topo'."""
    if '[' in part:
        attr, key = part.split('[')
        return getattr(obj, attr)[key.strip('"]')]
    return getattr(obj, part)


def _resolve(solver: "FEMSolver", source: str):
    """Resolve a dotted path off `solver`, e.g. 'problem.topo.h0' or
    'quad_mgr.nodal_fields["jx"].pg'."""
    obj = solver
    for part in source.split('.'):
        obj = _walk(obj, part)
    return obj


def _resolve_parent(solver: "FEMSolver", source: str):
    """Split a dotted path into (parent object, final attribute name), or
    (None, None) if any segment along the path is missing -- e.g. a rank
    that owns no ForceBalance controller. The final segment must be a plain
    attribute, not a bracketed key."""
    *parents, attr = source.split('.')
    obj = solver
    try:
        for part in parents:
            obj = _walk(obj, part)
    except (AttributeError, KeyError):
        return None, None
    return obj, attr


def _try_resolve(solver: "FEMSolver", source: str):
    """Resolve `source`, returning None if any segment along the path is missing."""
    try:
        return _resolve(solver, source)
    except (AttributeError, KeyError):
        return None


# =========================================================================
# Save
# =========================================================================

def save_state(solver: "FEMSolver", path: str) -> None:
    """Gather all checkpoint state to rank 0 and write a single `.npz`."""

    decomp = solver.problem.decomp
    data = {}

    for key, entry in CHECKPOINT_FIELD_REGISTRY.items():
        value = _try_resolve(solver, entry['source'])
        if value is None:
            continue
        field_type = entry['type']
        if field_type in ('nodal', 'nodal_P2'):
            value = decomp.gather_global_padded(value[0], P2=(field_type == 'nodal_P2'))
        elif field_type in ('field', 'field_inplace'):
            value = decomp.gather_global_padded(value)
        if decomp.rank == 0:
            data[key] = value

    xx = decomp.gather_global(decomp.xx)
    yy = decomp.gather_global(decomp.yy)
    rho = decomp.gather_global_padded(solver.quad_mgr.nf('rho'))

    wsx, wsy = solver.problem.wall_stress_xz, solver.problem.wall_stress_yz
    tau_xz_net = decomp.gather_global_padded(wsx.upper[wsx._out_index] - wsx.lower[wsx._out_index])
    tau_yz_net = decomp.gather_global_padded(wsy.upper[wsy._out_index] - wsy.lower[wsy._out_index])

    if decomp.rank == 0:
        data['xx'] = xx
        data['yy'] = yy
        data['rho'] = rho
        data['tau_xz_net'] = tau_xz_net
        data['tau_yz_net'] = tau_yz_net
        data['geo'] = solver.problem.geo
        data['Nx'] = solver.problem.grid['Nx']
        data['Ny'] = solver.problem.grid['Ny']
        data['Lx'] = solver.problem.grid['Lx']
        data['Ly'] = solver.problem.grid['Ly']
        np.savez(path, **data)


# =========================================================================
# Load
# =========================================================================

def load_state(solver: "FEMSolver", path: str) -> None:
    """Restore a checkpoint written by `save_state` onto a freshly constructed Problem."""

    p = solver.problem
    decomp = p.decomp
    comm = decomp._mpi_comm

    if p.step is None:
        p._pre_run(suspend_fb=True, suspend_def=True)

    npz = np.load(path, allow_pickle=True) if decomp.rank == 0 else None
    keys = comm.bcast(set(npz.keys()) if decomp.rank == 0 else None, root=0)

    def get(key):
        return npz[key] if decomp.rank == 0 else None

    for key, entry in CHECKPOINT_FIELD_REGISTRY.items():
        if key not in keys:
            continue
        field_type = entry['type']
        if field_type in ('nodal', 'nodal_P2'):
            value = decomp.scatter_global_padded(get(key), P2=(field_type == 'nodal_P2'))
            _resolve(solver, entry['source'])[0] = value
        elif field_type == 'field_inplace':
            value = decomp.scatter_global_padded(get(key))
            _resolve(solver, entry['source'])[...] = value
        elif field_type == 'field':
            value = decomp.scatter_global_padded(get(key))
            obj, attr = _resolve_parent(solver, entry['source'])
            if obj is None:
                continue
            setattr(obj, attr, value)
        else:
            value = comm.bcast(get(key), root=0)
            if key == 'rhv_history':
                value = value.tolist()
            elif key == 'residual_buffer':
                value = deque(value.tolist(), maxlen=100)
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            obj, attr = _resolve_parent(solver, entry['source'])
            if obj is None:
                continue
            setattr(obj, attr, value)

    solver.problem.pressure.pressure[:] = solver.quad_mgr.nf('p')
    solver.quad_mgr.nf('rho')[:] = solver.quad_mgr._call_computed('rho_from_p', solver.quad_mgr.nf)
    solver.quad_mgr.sync_to_problem_q()

    solver.quad_mgr.update_quad_fields()
    solver.quad_mgr.store_prev_values()

    solver.update_output_fields()
