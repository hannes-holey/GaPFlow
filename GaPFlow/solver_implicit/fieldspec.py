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
    'Uf_bot': 'P1',
    'Vf_bot': 'P1',
    'Uf_top': 'P1',
    'Vf_top': 'P1',
}

RES_GRID = {
    'momentum_x': 'P2',
    'momentum_y': 'P2',
    'mass': 'P1',
    'energy': 'P1',
    'fb': 'P1',
    'R_oss': 'P1',
    'wallstress_top_x': 'P1',
    'wallstress_top_y': 'P1',
    'wallstress_bot_x': 'P1',
    'wallstress_bot_y': 'P1',
}

NODAL_P1 = ['rho', 'p', 'Uf_bot', 'Vf_bot', 'Uf_top', 'Vf_top']
NODAL_P2 = ['jx', 'jy']
NODAL_Q1 = ['h', 'dh_dx', 'dh_dy', 'rho', 'eta']

# Shared arg lists for groups of computed fields with identical inputs
_ARGS_WALLSTRESS = ['rho', 'theta','jx', 'jy', 'h', 'dh_dx', 'U_bot', 'V_bot', 'U_top','V_top']


_ARGS_T = ['rho', 'jx', 'jy', 'E']
_ARGS_S = ['h', 'eta', 'rho', 'E', 'jx', 'jy', 'U_bot', 'V_bot', 'Tb_top', 'Tb_bot']
_ARGS_EPS = ['rho', 'h', 'eta', 'drho_dp', 'p']
_ARGS_RHO_EFF = ['rho', 'drho_dp']

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

    # generalized Reynolds (after drho_dp: computed fields are evaluated in registry order)
    'epsilon': {'type': 'computed', 'source': 'reynolds_eps.epsilon', 'args': _ARGS_EPS},
    'depsilon_dp': {'type': 'computed', 'source': 'reynolds_eps.depsilon_dp', 'args': _ARGS_EPS},
    'depsilon_dh': {'type': 'computed', 'source': 'reynolds_eps.depsilon_dh', 'args': _ARGS_EPS},
    'rho_eff': {'type': 'computed', 'source': 'reynolds_rho_eff.rho_eff', 'args': _ARGS_RHO_EFF},
    'drho_eff_dp': {'type': 'computed', 'source': 'reynolds_rho_eff.drho_eff_dp', 'args': _ARGS_RHO_EFF},
    'drho_eff_dh': {'type': 'computed', 'source': 'reynolds_rho_eff.drho_eff_dh', 'args': _ARGS_RHO_EFF},

    # fluid wall stress xz
    'tau_xz': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_xz', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_djx': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_djx', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_dh', 'args': _ARGS_WALLSTRESS},
    'tau_xz_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_xz_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_djx': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_djx', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dh', 'args': _ARGS_WALLSTRESS},

    # fluid wall stress yz
    'tau_yz': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_yz', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_djy': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_djy', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_dh', 'args': _ARGS_WALLSTRESS},
    'tau_yz_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_yz_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_djy': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_djy', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dh', 'args': _ARGS_WALLSTRESS},

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
    'rho_eff_prev': {'type': 'computed', 'source': None, 'args': []},
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


_CUSTOM_ARGS_WALL_STRESS_XZ = {name: ['rho', 'theta', 'jx', 'h', 'U_bot', 'U_top']
                               for name in _GP_ARGS_WALL_STRESS_XZ}

_CUSTOM_ARGS_WALL_STRESS_YZ = {name: ['rho', 'theta', 'jy', 'h', 'V_bot', 'V_top']
                               for name in _GP_ARGS_WALL_STRESS_YZ}


_CUSTOM_ARGS_EPS = {name: ['p', 'h'] for name in ('epsilon', 'depsilon_dp', 'depsilon_dh')}

_CUSTOM_ARGS_RHO_EFF = {name: ['p', 'h'] for name in ('rho_eff', 'drho_eff_dp', 'drho_eff_dh')}


_DEFAULT_REGISTRY_ARGS = {
    name: list(QUAD_FIELD_REGISTRY[name]['args'])
    for block in (_GP_ARGS_PRESSURE, _GP_ARGS_WALL_STRESS_XZ, _GP_ARGS_WALL_STRESS_YZ,
                  _CUSTOM_ARGS_EPS, _CUSTOM_ARGS_RHO_EFF)
    for name in block
}


def patch_registry_for_gp(pressure: bool = True,
                          wall_stress_xz: bool = False,
                          wall_stress_yz: bool = False,
                          custom_xz: bool = False,
                          custom_yz: bool = False,
                          custom_eps: bool = False,
                          custom_rho_eff: bool = False) -> None:
    for name, args in _DEFAULT_REGISTRY_ARGS.items():
        QUAD_FIELD_REGISTRY[name]['args'] = list(args)
    for active, block in [
        (pressure, _GP_ARGS_PRESSURE),
        (wall_stress_xz, _GP_ARGS_WALL_STRESS_XZ),
        (wall_stress_yz, _GP_ARGS_WALL_STRESS_YZ),
        (custom_xz, _CUSTOM_ARGS_WALL_STRESS_XZ),
        (custom_yz, _CUSTOM_ARGS_WALL_STRESS_YZ),
        (custom_eps, _CUSTOM_ARGS_EPS),
        (custom_rho_eff, _CUSTOM_ARGS_RHO_EFF),
    ]:
        if active:
            for name, args in block.items():
                QUAD_FIELD_REGISTRY[name]['args'] = args


def patch_registry_for_slip() -> None:
    _ARGS_WALLSTRESS = ['rho', 'theta', 'jx', 'jy', 'h', 'dh_dx', 'Uf_bot', 'Vf_bot', 'Uf_top', 'Vf_top']

    QUAD_FIELD_REGISTRY.update({
    # fluid wall stress xz
    'tau_xz': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_xz', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_djx': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_djx', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_dh', 'args': _ARGS_WALLSTRESS},
    'tau_xz_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_xz_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_djx': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_djx', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dh', 'args': _ARGS_WALLSTRESS},

    # Additional derivatives wrt. Velocities
    'dtau_xz_bot_dVf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dV_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dVf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dV_top', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dUf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dU_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_bot_dUf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_bot_dU_top', 'args': _ARGS_WALLSTRESS},
    'tau_xz_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_xz_top', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_djx': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_djx', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_dh', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_dVf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_dV_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_dVf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_dV_top', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_dUf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_dU_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_xz_top_dUf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_xz_top_dU_top', 'args': _ARGS_WALLSTRESS},

        # fluid wall stress yz
    'tau_yz': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_yz', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_djy': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_djy', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_dh', 'args': _ARGS_WALLSTRESS},
    'tau_yz_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_yz_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_djy': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_djy', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dh', 'args': _ARGS_WALLSTRESS},

    # Additional derivatives wrt. Velocities
    'dtau_yz_bot_dVf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dV_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dVf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dV_top', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dUf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dU_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_bot_dUf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_bot_dU_top', 'args': _ARGS_WALLSTRESS},
    'tau_yz_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].tau_yz_top', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_drho': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_drho', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_djy': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_djy', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_dtheta': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_dtheta', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_dh': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_dh', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_dVf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_dV_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_dVf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_dV_top', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_dUf_bot': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_dU_bot', 'args': _ARGS_WALLSTRESS},
    'dtau_yz_top_dUf_top': {'type': 'computed', 'source': 'custom_models["wall_stress"].dtau_yz_top_dU_top', 'args': _ARGS_WALLSTRESS},


    # Slip velocities
    'Us_bot': {'type': 'computed', 'source': 'custom_models["operations"].difference', 'args': ['Uf_bot', 'U_bot']},
    'Vs_bot': {'type': 'computed', 'source': 'custom_models["operations"].difference', 'args': ['Vf_bot', 'V_bot']},
    'Us_top': {'type': 'computed', 'source': 'custom_models["operations"].difference', 'args': ['U_top', 'Uf_top']},
    'Vs_top': {'type': 'computed', 'source': 'custom_models["operations"].difference', 'args': ['V_top', 'Vf_top']},

    # Stress from the slip law
    'tau_slip_top_x': {'type': 'computed', 'source': 'custom_models["slip_top"].tau_x', 'args': ['Us_top', 'Vs_top']},
    'tau_slip_top_y': {'type': 'computed', 'source': 'custom_models["slip_top"].tau_y', 'args': ['Us_top', 'Vs_top']},
    'tau_slip_bot_x': {'type': 'computed', 'source': 'custom_models["slip_bot"].tau_x', 'args': ['Us_bot', 'Vs_bot']},
    'tau_slip_bot_y': {'type': 'computed', 'source': 'custom_models["slip_bot"].tau_y', 'args': ['Us_bot', 'Vs_bot']},

    'dtau_slip_top_x_dUs': {'type': 'computed', 'source': 'custom_models["slip_top"].dtau_x_dvs_x', 'args': ['Us_top', 'Vs_top']},
    'dtau_slip_top_x_dVs': {'type': 'computed', 'source': 'custom_models["slip_top"].dtau_x_dvs_y', 'args': ['Us_top', 'Vs_top']},
    'dtau_slip_top_y_dUs': {'type': 'computed', 'source': 'custom_models["slip_top"].dtau_y_dvs_x', 'args': ['Us_top', 'Vs_top']},
    'dtau_slip_top_y_dVs': {'type': 'computed', 'source': 'custom_models["slip_top"].dtau_y_dvs_y', 'args': ['Us_top', 'Vs_top']},

    'dtau_slip_bot_x_dUs': {'type': 'computed', 'source': 'custom_models["slip_bot"].dtau_x_dvs_x', 'args': ['Us_bot', 'Vs_bot']},
    'dtau_slip_bot_x_dVs': {'type': 'computed', 'source': 'custom_models["slip_bot"].dtau_x_dvs_y', 'args': ['Us_bot', 'Vs_bot']},
    'dtau_slip_bot_y_dUs': {'type': 'computed', 'source': 'custom_models["slip_bot"].dtau_y_dvs_x', 'args': ['Us_bot', 'Vs_bot']},
    'dtau_slip_bot_y_dVs': {'type': 'computed', 'source': 'custom_models["slip_bot"].dtau_y_dvs_y', 'args': ['Us_bot', 'Vs_bot']},

    }
    )