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
Linear system scaling for FEM solver conditioning.

This module provides scaling utilities to improve the conditioning of the
linear systems arising from Newton iteration in the FEM solver.

The transformation is:
    J*[i,j] = J[i,j] · D_q[j] / D_R[i]
    R*[i] = R[i] / D_R[i]
    dq[j] = dq*[j] · D_q[j]

where D_q and D_R are characteristic scales for variables and residuals.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .assembly_layout import MatrixCOOPattern, RHSPattern
    from GaPFlow import Problem

NDArray = npt.NDArray[np.floating]


@dataclass
class ScalingInfo:
    """Precomputed scaling factors for linear system conditioning.

    Transforms the linear system J·dq = -R into a well-conditioned
    scaled system J*·dq* = -R* where all matrix entries are O(1).

    The transformation is:
        J*[i,j] = J[i,j] · D_q[j] / D_R[i]
        R*[i] = R[i] / D_R[i]
        dq[j] = dq*[j] · D_q[j]

    Attributes
    ----------
    coo_scale : NDArray
        Scale factors for COO values, shape (nnz,).
    rhs_scale : NDArray
        Scale factors for residual vector, shape (rhs_size,).
    sol_scale : NDArray
        Scale factors for solution vector, shape (sol_size,).
    char_scales : dict
        Original characteristic scales for reference.
    """
    coo_scale: NDArray
    rhs_scale: NDArray
    sol_scale: NDArray
    char_scales: Dict[str, float]

    def scale_system(self, M_coo: NDArray, R: NDArray) -> Tuple[NDArray, NDArray]:
        """Scale Jacobian COO values and residual vector.

        Parameters
        ----------
        M_coo : NDArray, shape (nnz,)
            Unscaled Jacobian in COO value format.
        R : NDArray, shape (n,)
            Unscaled residual vector.

        Returns
        -------
        M_scaled : NDArray, shape (nnz,)
            Scaled Jacobian COO values.
        R_scaled : NDArray, shape (n,)
            Scaled residual vector.
        """
        return M_coo * self.coo_scale, R / self.rhs_scale

    def unscale_solution(self, dq_scaled: NDArray) -> NDArray:
        """Recover physical solution increment from scaled solution.

        Parameters
        ----------
        dq_scaled : NDArray, shape (n,)
            Solution of the scaled system.

        Returns
        -------
        dq : NDArray, shape (n,)
            Physical solution increment.
        """
        return dq_scaled * self.sol_scale


def build_scaling(
    char_scales: Dict[str, float],
    variables: List[str],
    matrix_coo: "MatrixCOOPattern",
    rhs: "RHSPattern",
) -> ScalingInfo:
    """Build scaling factors from characteristic scales.

    Uses the block indices stored in matrix_coo and rhs patterns
    to efficiently compute per-entry scale factors.

    Parameters
    ----------
    char_scales : dict
        Characteristic scale for each variable: {'rho': ρ_ref, 'jx': j_ref, ...}
    variables : list
        Variable names in block order: ['rho', 'jx', 'jy'] or with 'E'.
    matrix_coo : MatrixCOOPattern
        Sparse matrix structure with block indices.
    rhs : RHSPattern
        RHS vector structure with block indices.

    Returns
    -------
    ScalingInfo
        Precomputed scaling factors for COO values, RHS, and solution.

    Raises
    ------
    ValueError
        If char_scales is missing entries or contains non-positive values.
    """
    # Validation
    for var in variables:
        if var not in char_scales:
            raise ValueError(f"Missing characteristic scale for '{var}'")
        if char_scales[var] <= 0:
            raise ValueError(
                f"Characteristic scale for '{var}' must be positive, "
                f"got {char_scales[var]}"
            )

    # Per-block scale arrays
    q_scales = np.array([char_scales[v] for v in variables], dtype=np.float64)
    r_scales = q_scales  # Residual scales match corresponding variable

    # COO scaling factors: J*[k] = J[k] * D_q[var_idx] / D_R[res_idx]
    coo_scale = (q_scales[matrix_coo.var_block_idx] /
                 r_scales[matrix_coo.res_block_idx])

    # RHS scaling factors
    rhs_scale = r_scales[rhs.res_block_idx]

    # Solution scaling factors (same block structure as RHS)
    sol_scale = q_scales[rhs.res_block_idx]

    return ScalingInfo(
        coo_scale=coo_scale,
        rhs_scale=rhs_scale,
        sol_scale=sol_scale,
        char_scales=char_scales,
    )


def compute_characteristic_scales(problem: "Problem", energy: bool) -> Dict[str, float]:
    """Derive characteristic scales from problem specification.

    These scales are used to condition the linear system by normalizing
    variables to O(1) magnitudes.

    Parameters
    ----------
    problem : Problem
        The problem instance containing geometry and property specifications.
    energy : bool
        Whether the energy equation is active.

    Returns
    -------
    dict
        Characteristic scale for each variable: {'rho': ρ_ref, 'jx': j_ref, ...}
    """
    rho_ref = problem.prop['rho0']
    U_ref = _get_characteristic_velocity(problem)
    j_ref = rho_ref * U_ref

    scales = {'rho': rho_ref, 'jx': j_ref, 'jy': j_ref}

    if energy:
        cv = problem.energy_spec['cv']
        # Handle both tuple format ('uniform', T) and direct value
        T0 = problem.energy_spec['T0']
        T_ref = T0[1] if isinstance(T0, tuple) else T0
        T_ref = problem.energy_spec['T_wall']
        E_ref = rho_ref * cv * T_ref
        scales['E'] = E_ref

    return scales


def _get_characteristic_velocity(problem: "Problem") -> float:
    """Compute characteristic velocity scale from wall velocities.

    Uses wall velocities (U, V from geometry) as the characteristic scale.
    This maintains backward compatibility with tuned stabilization parameters.

    Parameters
    ----------
    problem : Problem
        The problem instance.

    Returns
    -------
    float
        Characteristic velocity scale (always positive).
    """
    # Wall velocity only (backward compatible)
    u_wall_max = max(abs(problem.geo['U']), abs(problem.geo['V']))

    # Use wall velocity with a small floor to avoid division by zero
    return max(u_wall_max, 1e-10)
