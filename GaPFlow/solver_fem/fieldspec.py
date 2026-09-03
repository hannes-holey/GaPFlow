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
from dataclasses import dataclass


def resolve_source(problem, source: str):
    """Given a source string like 'topo.h' or 'geo["U_bot"]',
    resolve it to the actual object."""
    obj = problem
    for part in source.split('.'):
        if '[' in part:
            attr, key = part.split('[')
            obj = getattr(obj, attr)[key.strip('"]')]
        else:
            obj = getattr(obj, part)
    return obj


def categorize_registry_fields():
    """Split QUAD_FIELD_REGISTRY into nodal, scalar, and computed lists."""
    nodal, scalar, computed = [], [], []
    for name, entry in QUAD_FIELD_REGISTRY.items():
        t = entry['type']
        if t == 'nodal':
            nodal.append(name)
        elif t == 'scalar':
            scalar.append(name)
        elif t == 'computed' and entry['source'] is not None:
            computed.append(name)
    return nodal, scalar, computed


@dataclass
class FieldSpec:
    """Single source of truth for one active variable or residual."""
    name: str
    grid: str
    idx: int


VAR_GRID = {
    'jx': 'P2',
    'jy': 'P2',
    'p': 'P1',
    'E': 'P1',
    'theta': 'P1',
    'xi': 'P1',
}

RES_GRID = {
    'momentum_x': 'P2',
    'momentum_y': 'P2',
    'mass': 'P1',
    'energy': 'P1',
    'fb': 'P1',
    'R_oss': 'P1',
}

NODAL_P1 = ['rho', 'p']
NODAL_P2 = ['jx', 'jy']
NODAL_Q1 = ['h', 'dh_dx', 'dh_dy', 'rho', 'eta']

# Shared arg lists for groups of computed fields with identical inputs
_ARGS_XZ = ['rho', 'jx', 'jy', 'h', 'dh_dx', 'U_bot', 'V_bot', 'U_top',
            'V_top', 'Ls', 'theta', 'd_dx_p', 'd_dy_p', 'eta']
_ARGS_YZ = ['rho', 'jx', 'jy', 'h', 'dh_dy', 'U_bot', 'V_bot', 'U_top',
            'V_top', 'Ls', 'theta', 'd_dx_p', 'd_dy_p', 'eta']
_ARGS_T = ['rho', 'jx', 'jy', 'E']
_ARGS_S = ['h', 'eta', 'rho', 'E', 'jx', 'jy', 'U_bot', 'V_bot', 'Tb_top', 'Tb_bot']

QUAD_FIELD_REGISTRY = {
    # ------------------------------------------------------------------
    # nodal: interpolated from nodal field to quad points
    # ------------------------------------------------------------------
    'h': {'type': 'nodal', 'source': 'topo.h'},
    'dh_dx': {'type': 'nodal', 'source': 'topo.dh_dx'},
    'dh_dy': {'type': 'nodal', 'source': 'topo.dh_dy'},
    'eta': {'type': 'nodal', 'source': 'viscosity.eta'},
    'Tb_top': {'type': 'nodal', 'source': 'energy.Tb_top'},
    'Tb_bot': {'type': 'nodal', 'source': 'energy.Tb_bot'},

    # ------------------------------------------------------------------
    # scalar: broadcast scalar value to all quad points
    # ------------------------------------------------------------------
    'U_bot': {'type': 'scalar', 'source': 'geo["U_bot"]'},
    'V_bot': {'type': 'scalar', 'source': 'geo["V_bot"]'},
    'U_top': {'type': 'scalar', 'source': 'geo["U_top"]'},
    'V_top': {'type': 'scalar', 'source': 'geo["V_top"]'},
    'Ls': {'type': 'scalar', 'source': 'prop["slip_length"]'},
    'force_x': {'type': 'scalar', 'source': 'prop["force_x"]'},
    'force_y': {'type': 'scalar', 'source': 'prop["force_y"]'},

    # ------------------------------------------------------------------
    # computed: physics method called with quad field arguments
    # ------------------------------------------------------------------
    'p_from_rho': {'type': 'computed', 'source': 'pressure.p_from_rho', 'args': ['rho']},
    'rho_from_p': {'type': 'computed', 'source': 'pressure.rho_from_p', 'args': ['p']},
    'drho_dp': {'type': 'computed', 'source': 'pressure.drho_dp', 'args': ['rho']},
    'dp_drho': {'type': 'computed', 'source': 'pressure.dp_drho', 'args': ['rho']},
    'd2p_drho2': {'type': 'computed', 'source': 'pressure.d2p_drho2', 'args': ['rho_avg']},

    # wall stress xz
    'tau_xz': {'type': 'computed', 'source': 'wall_stress_xz.tau', 'args': _ARGS_XZ},
    'dtau_xz_drho': {'type': 'computed', 'source': 'wall_stress_xz.dtau_drho', 'args': _ARGS_XZ},
    'dtau_xz_djx': {'type': 'computed', 'source': 'wall_stress_xz.dtau_djx', 'args': _ARGS_XZ},
    'dtau_xz_dtheta': {'type': 'computed', 'source': 'wall_stress_xz.dtau_dtheta', 'args': _ARGS_XZ},
    'dtau_xz_dh': {'type': 'computed', 'source': 'wall_stress_xz.dtau_dh', 'args': _ARGS_XZ},
    'tau_xz_bot': {'type': 'computed', 'source': 'wall_stress_xz.tau_bot', 'args': _ARGS_XZ},
    'dtau_xz_bot_drho': {'type': 'computed', 'source': 'wall_stress_xz.dtau_bot_drho', 'args': _ARGS_XZ},
    'dtau_xz_bot_djx': {'type': 'computed', 'source': 'wall_stress_xz.dtau_bot_djx', 'args': _ARGS_XZ},
    'dtau_xz_bot_dtheta': {'type': 'computed', 'source': 'wall_stress_xz.dtau_bot_dtheta', 'args': _ARGS_XZ},
    'dtau_xz_bot_dh': {'type': 'computed', 'source': 'wall_stress_xz.dtau_bot_dh', 'args': _ARGS_XZ},

    # wall stress yz
    'tau_yz': {'type': 'computed', 'source': 'wall_stress_yz.tau', 'args': _ARGS_YZ},
    'dtau_yz_drho': {'type': 'computed', 'source': 'wall_stress_yz.dtau_drho', 'args': _ARGS_YZ},
    'dtau_yz_djy': {'type': 'computed', 'source': 'wall_stress_yz.dtau_djy', 'args': _ARGS_YZ},
    'dtau_yz_dtheta': {'type': 'computed', 'source': 'wall_stress_yz.dtau_dtheta', 'args': _ARGS_YZ},
    'dtau_yz_dh': {'type': 'computed', 'source': 'wall_stress_yz.dtau_dh', 'args': _ARGS_YZ},
    'tau_yz_bot': {'type': 'computed', 'source': 'wall_stress_yz.tau_bot', 'args': _ARGS_YZ},
    'dtau_yz_bot_drho': {'type': 'computed', 'source': 'wall_stress_yz.dtau_bot_drho', 'args': _ARGS_YZ},
    'dtau_yz_bot_djy': {'type': 'computed', 'source': 'wall_stress_yz.dtau_bot_djy', 'args': _ARGS_YZ},
    'dtau_yz_bot_dtheta': {'type': 'computed', 'source': 'wall_stress_yz.dtau_bot_dtheta', 'args': _ARGS_YZ},
    'dtau_yz_bot_dh': {'type': 'computed', 'source': 'wall_stress_yz.dtau_bot_dh', 'args': _ARGS_YZ},

    # temperature
    'T': {'type': 'computed', 'source': 'energy.T_func', 'args': _ARGS_T},
    'dT_drho': {'type': 'computed', 'source': 'energy.T_grad_rho', 'args': _ARGS_T},
    'dT_djx': {'type': 'computed', 'source': 'energy.T_grad_jx', 'args': _ARGS_T},
    'dT_djy': {'type': 'computed', 'source': 'energy.T_grad_jy', 'args': _ARGS_T},
    'dT_dE': {'type': 'computed', 'source': 'energy.T_grad_E', 'args': _ARGS_T},

    # wall heat flux
    'S': {'type': 'computed', 'source': 'energy.q_wall_sum', 'args': _ARGS_S},
    'dS_drho': {'type': 'computed', 'source': 'energy.q_wall_grad_rho', 'args': _ARGS_S},
    'dS_djx': {'type': 'computed', 'source': 'energy.q_wall_grad_jx', 'args': _ARGS_S},
    'dS_djy': {'type': 'computed', 'source': 'energy.q_wall_grad_jy', 'args': _ARGS_S},
    'dS_dE': {'type': 'computed', 'source': 'energy.q_wall_grad_E', 'args': _ARGS_S},

    # Squeeze (computed inline in _update_squeeze_quad_fields / store_prev_values)
    'h_prev': {'type': 'computed', 'source': None, 'args': []},
    'rho_prev': {'type': 'computed', 'source': None, 'args': []},
    'dp_drho_prev': {'type': 'computed', 'source': None, 'args': []},
    'p_prev': {'type': 'computed', 'source': None, 'args': []},
    'rho_avg': {'type': 'computed', 'source': None, 'args': ['rho_prev']},
    'dh_dt': {'type': 'computed', 'source': None, 'args': ['h_prev']},

    # OSS stabilization (computed inline in _update_oss_quad_fields)
    'a_vec_x': {'type': 'computed', 'source': None, 'args': ['jx']},
    'a_vec_y': {'type': 'computed', 'source': None, 'args': ['jy']},
    'tau_a_x': {'type': 'computed', 'source': None, 'args': ['jx', 'jy']},
    'tau_a_y': {'type': 'computed', 'source': None, 'args': ['jx', 'jy']},

    # Flux-capturing stabilization (computed inline in _update_fc_quad_fields)
    'fc_tau': {'type': 'computed', 'source': None,
               'args': ['jx', 'jy', 'h', 'dh_dx', 'dh_dy', 'd_dx_jx',
                        'd_dy_jy', 'd_dx_theta', 'd_dy_theta', 'rho', 'rho_prev', 'theta_prev']},

    # SUPG stabilization (computed inline in _update_supg_quad_fields)
    'f_x': {'type': 'computed', 'source': None, 'args': []},
    'f_y': {'type': 'computed', 'source': None, 'args': []},

    # Reynolds mode: mean surface velocity, SUPG stabilization
    # (computed inline in _update_reynolds_supg_quad_fields)
    'U_m': {'type': 'computed', 'source': None, 'args': []},
    'tau_supg': {'type': 'computed', 'source': None, 'args': []},
}


# TODO: we need to allow arbitrary arg-combinations
_GP_ARGS_PRESSURE = {
    'p_from_rho': ['rho', 'h'],
    'rho_from_p': ['p', 'h', 'rho'],  # rho for initial guess
    'dp_drho': ['rho', 'h'],
    'drho_dp': ['rho', 'h'],
    'd2p_drho2': ['rho', 'h'],
}

_GP_ARGS_WALL_STRESS_XZ = {
    'tau_xz': ['rho', 'jx', 'h'],
    'dtau_xz_drho': ['rho', 'jx', 'h'],
    'dtau_xz_djx': ['rho', 'jx', 'h'],
    'dtau_xz_dtheta': ['rho', 'jx', 'h'],
    'dtau_xz_dh': ['rho', 'jx', 'h'],
    'tau_xz_bot': ['rho', 'jx', 'h'],
    'dtau_xz_bot_drho': ['rho', 'jx', 'h'],
    'dtau_xz_bot_djx': ['rho', 'jx', 'h'],
    'dtau_xz_bot_dtheta': ['rho', 'jx', 'h'],
    'dtau_xz_bot_dh': ['rho', 'jx', 'h'],
}

_GP_ARGS_WALL_STRESS_YZ = {
    'tau_yz': ['rho', 'jy', 'h'],
    'dtau_yz_drho': ['rho', 'jy', 'h'],
    'dtau_yz_djy': ['rho', 'jy', 'h'],
    'dtau_yz_dtheta': ['rho', 'jy', 'h'],
    'dtau_yz_dh': ['rho', 'jy', 'h'],
    'tau_yz_bot': ['rho', 'jy', 'h'],
    'dtau_yz_bot_drho': ['rho', 'jy', 'h'],
    'dtau_yz_bot_djy': ['rho', 'jy', 'h'],
    'dtau_yz_bot_dtheta': ['rho', 'jy', 'h'],
    'dtau_yz_bot_dh': ['rho', 'jy', 'h'],
}


def patch_registry_for_gp(pressure: bool = True,
                          wall_stress_xz: bool = False,
                          wall_stress_yz: bool = False) -> None:
    for active, block in [
        (pressure, _GP_ARGS_PRESSURE),
        (wall_stress_xz, _GP_ARGS_WALL_STRESS_XZ),
        (wall_stress_yz, _GP_ARGS_WALL_STRESS_YZ),
    ]:
        if active:
            for name, args in block.items():
                QUAD_FIELD_REGISTRY[name]['args'] = args
