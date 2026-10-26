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

"""Linear system scaling for Taylor-Hood Q2Q1 FEM solver conditioning."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from GaPFlow import Problem
    from .assembly import Assembly

NDArray = npt.NDArray[np.floating]


@dataclass
class ScalingInfo:
    """Precomputed scaling factors for linear system conditioning.

    Transforms J·dq = -R into J*·dq* = -R* where all entries are O(1).

    Transformation:
        J*[i,j] = J[i,j] · D_C[j] / D_R[i]
        R*[i]   = R[i] / D_R[i]
        dq[j]   = dq*[j] · D_C[j]

    display_scale = 1/(D_R[r] * D_C[v]) per COO entry — used only for
    printing block norms so the table shows the equilibrated magnitudes.
    """
    coo_scale:     NDArray
    rhs_scale:     NDArray
    sol_scale:     NDArray
    display_scale: NDArray
    char_scales:   Dict[str, float]

    def scale_system(self, M_coo: NDArray,
                     R: NDArray) -> Tuple[NDArray, NDArray]:
        return M_coo * self.coo_scale, R / self.rhs_scale

    def unscale_solution(self, dq_scaled: NDArray) -> NDArray:
        return dq_scaled * self.sol_scale


def build_scaling_from_blocks(
    M_coo: NDArray,
    variables: List[str],
    residuals: List[str],
    assembly: "Assembly",
    n_iter: int = 10,
) -> ScalingInfo:
    """Ruiz equilibration on the block-norm matrix.

    Treats each (residual, variable) block as a single scalar entry equal to
    its Frobenius norm, then runs Ruiz iterations on this small (nr x nv) matrix
    until the max entry in every row and column is 1.

    Each iteration:
        D_R[i] *= sqrt(max_j B[i,j])
        D_C[j] *= sqrt(max_i B[i,j])
        B[i,j] /= D_R[i] * D_C[j]   (applied cumulatively via the accumulators)
    """
    asm = assembly
    nr = len(residuals)
    nv = len(variables)
    res_idx = {r: i for i, r in enumerate(residuals)}
    var_idx = {v: i for i, v in enumerate(variables)}

    # Build per-block Frobenius norms and COO block index arrays
    block_norms = np.zeros((nr, nv), dtype=np.float64)
    r_blk = np.empty(len(M_coo), dtype=np.int32)
    v_blk = np.empty(len(M_coo), dtype=np.int32)
    for (res, var), block in asm.block_order.items():
        s = block['nnz_idx_start']
        n = block['nb_nnz']
        ri, vi = res_idx[res], var_idx[var]
        r_blk[s:s + n] = ri
        v_blk[s:s + n] = vi
        block_norms[ri, vi] = np.linalg.norm(M_coo[s:s + n])

    # Ruiz iterations on the block-norm matrix
    D_R = np.ones(nr, dtype=np.float64)
    D_C = np.ones(nv, dtype=np.float64)
    B = block_norms.copy()
    for _ in range(n_iter):
        row_max = np.where(B.max(axis=1) > 0, B.max(axis=1), 1.0)
        col_max = np.where(B.max(axis=0) > 0, B.max(axis=0), 1.0)
        sr = np.sqrt(row_max)
        sc = np.sqrt(col_max)
        D_R *= sr
        D_C *= sc
        B /= np.outer(sr, sc)

    # Follow ScalingInfo convention: J*[i,j] = J[i,j] * D_C[j] / D_R[i]
    # so coo_scale = D_C[var] / D_R[res], rhs_scale = D_R, sol_scale = D_C,
    # and dq = dq* * D_C (unscale_solution multiplies by sol_scale).
    coo_scale = D_C[v_blk] / D_R[r_blk]
    display_scale = 1.0 / (D_R[r_blk] * D_C[v_blk])

    # RHS scale: D_R per residual DOF
    rhs_scale = D_R[asm.rhs_res_block_idx]

    # Solution unscale: D_C per variable DOF
    sol_scale = D_C[asm.rhs_res_block_idx]

    char_scales = {**{v: float(D_C[vi]) for vi, v in enumerate(variables)},
                   **{f'R_{r}': float(D_R[ri]) for ri, r in enumerate(residuals)}}

    return ScalingInfo(
        coo_scale=coo_scale,
        rhs_scale=rhs_scale,
        sol_scale=sol_scale,
        display_scale=display_scale,
        char_scales=char_scales,
    )


def build_scaling(
    problem: "Problem",
    energy: bool,
    variables: List[str],
    assembly: "Assembly",
    cavitation: bool = False,
) -> ScalingInfo:
    """Build scaling factors for linear system conditioning.

    Works for Taylor-Hood Q2Q1 where different variables can have different
    numbers of inner points (nb_inner_P2 for jx/jy, nb_inner_p for rho/e).
    The assembly.var_block_idx and res_block_idx / rhs_res_block_idx arrays
    (added in Phase 7) provide the per-entry variable/residual indices.

    Parameters
    ----------
    problem : Problem
    energy : bool
    variables : list of str
        Variable names in block order, e.g. ['jx', 'jy', 'rho'].
    assembly : Assembly
        Must have .var_block_idx, .res_block_idx, .rhs_res_block_idx.
    cavitation : bool
    """
    char_scales = compute_characteristic_scales(problem, energy, cavitation)

    q_scales = np.array([char_scales[v] for v in variables], dtype=np.float64)
    r_scales = q_scales  # residual scales match corresponding variable

    coo_scale = q_scales[assembly.var_block_idx] / r_scales[assembly.res_block_idx]
    display_scale = 1.0 / (r_scales[assembly.res_block_idx] * q_scales[assembly.var_block_idx])
    rhs_scale = r_scales[assembly.rhs_res_block_idx]
    sol_scale = q_scales[assembly.rhs_res_block_idx]

    return ScalingInfo(
        coo_scale=coo_scale,
        rhs_scale=rhs_scale,
        sol_scale=sol_scale,
        display_scale=display_scale,
        char_scales=char_scales,
    )


def compute_characteristic_scales(problem: "Problem",
                                   energy: bool,
                                   cavitation: bool = False) -> Dict[str, float]:
    """Derive characteristic scales from problem specification.

    Pressure-based solver: the mass-row scale is `p_ref = prop['P0']`, the
    reference pressure from the EoS config. Raises KeyError if P0 is not
    set (no silent default).

    theta is dimensionless and O(1) — scale 1.0.
    fb residual is also dimensionless and O(1) — scale 1.0 via r_scales = q_scales.
    """
    rho_ref = problem.prop['rho0']
    U_ref = _get_characteristic_velocity(problem)
    p_ref = 1e05  # problem.prop['P0']
    c_ref = problem.prop.get('c_l', np.sqrt(p_ref / rho_ref))
    j_ref = rho_ref * U_ref * c_ref

    scales = {'p': p_ref, 'jx': j_ref, 'jy': j_ref}

    if energy:
        cv    = problem.energy_spec['cv']
        T_ref = problem.energy_spec['T_wall']
        E_ref = rho_ref * cv * T_ref
        scales['E'] = E_ref

    if cavitation:
        scales['theta'] = 1.0
        scales['fb'] = 1.0
        scales['xi'] = 1.0
        scales['R_oss'] = 1.0

    return scales


def _get_characteristic_velocity(problem: "Problem") -> float:
    u_wall_max = max(abs(problem.geo['U_bot']), abs(problem.geo['V_bot']),
                     abs(problem.geo['U_top']), abs(problem.geo['V_top']))
    if u_wall_max > 1e-10:
        return u_wall_max

    # Body-force driven flow: estimate U ~ f * Ly^2 / (8*mu)  (Poiseuille)
    force_x = problem.prop['force_x']
    force_y = problem.prop['force_y']
    force_mag = max(abs(force_x), abs(force_y))
    if force_mag > 0.0:
        mu = problem.prop['shear']
        Ly = problem.grid['Ly']
        return force_mag * Ly**2 / (8.0 * mu)

    return 1.0
