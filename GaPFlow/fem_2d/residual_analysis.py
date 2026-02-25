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

"""
Residual analysis utilities for FEM 2D solver.

Provides tools to analyze and visualize the contribution of individual
physical terms to the overall residual vector.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from ..solver_fem_2d import FEMSolver2D
    from .terms import NonLinearTerm

import numpy.typing as npt
NDArray = npt.NDArray[np.floating]


# Color palette for terms (colorblind-friendly, distinct)
TERM_COLORS = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB',  # grey
    '#44AA99',  # teal
]


def _style_axis(ax) -> None:
    """Apply consistent styling to axis: full box border, light grid."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(direction='in', top=True, right=True)
    ax.grid(True, alpha=0.3, linewidth=0.5)


def compute_term_contributions(solver: "FEMSolver2D") -> Dict[str, Dict[str, float]]:
    """
    Compute the L1 norm (integral of |residual|) for each active term.

    PSPG sub-terms (R1PSPG_*) are combined into a single 'PSPG' entry per
    equation by summing the signed residual vectors first, then taking the
    L1 norm.  This avoids inflating the PSPG contribution through the
    triangle inequality (sum of |R_i| >= |sum of R_i|).

    Parameters
    ----------
    solver : FEMSolver2D
        The initialized FEM 2D solver instance.

    Returns
    -------
    dict
        Nested dictionary: {residual_name: {term_name: L1_norm}}
        Example: {'mass': {'R11x': 1.23, 'PSPG': 0.45, ...}, ...}
    """
    comm = solver.problem.decomp._mpi_comm
    dx = solver.dx
    dy = solver.dy
    dA = dx * dy

    contributions = {res: {} for res in solver.residuals}

    # Stabilization prefixes to group: sum signed vectors before L1 norm
    STAB_GROUPS = {'R1PSPG_': 'PSPG', 'R2GLS_': 'GLS'}
    # Key: (group_label, res) -> accumulated signed vector
    stab_vectors: Dict[Tuple[str, str], NDArray] = {}

    for term in solver.terms:
        local_residual = solver.residual_vector_term(term)

        group_label = None
        for prefix, label in STAB_GROUPS.items():
            if term.name.startswith(prefix):
                group_label = label
                break

        if group_label is not None:
            key = (group_label, term.res)
            if key not in stab_vectors:
                stab_vectors[key] = np.zeros_like(local_residual)
            stab_vectors[key] += local_residual
        else:
            local_L1 = np.sum(np.abs(local_residual)) * dA
            global_L1 = comm.allreduce(local_L1, op=MPI.SUM)
            contributions[term.res][term.name] = global_L1

    # Compute L1 norm of combined stabilization residual per group/equation
    for (label, res), vec in stab_vectors.items():
        local_L1 = np.sum(np.abs(vec)) * dA
        global_L1 = comm.allreduce(local_L1, op=MPI.SUM)
        contributions[res][label] = global_L1

    return contributions


def compute_residual_equation_totals(contributions: Dict[str, Dict[str, float]]
                                     ) -> Dict[str, float]:
    """
    Compute total residual magnitude per equation.

    Parameters
    ----------
    contributions : dict
        Output from compute_term_contributions().

    Returns
    -------
    dict
        {residual_name: total_L1_norm}
    """
    return {res: sum(terms.values()) for res, terms in contributions.items()}


def get_significant_terms(contributions: Dict[str, Dict[str, float]],
                          threshold: float = 0.05) -> Dict[str, List[str]]:
    """
    Get terms contributing more than threshold fraction of their equation's total.

    Parameters
    ----------
    contributions : dict
        Output from compute_term_contributions().
    threshold : float
        Fraction threshold (default 0.05 = 5%).

    Returns
    -------
    dict
        {residual_name: [list of significant term names]}
    """
    equation_totals = compute_residual_equation_totals(contributions)
    significant = {}

    for res, terms in contributions.items():
        total = equation_totals[res]
        if total > 0:
            significant[res] = [
                name for name, val in terms.items()
                if val / total >= threshold
            ]
        else:
            significant[res] = []

    return significant


def compute_term_spatial_profiles(solver: "FEMSolver2D",
                                  term: "NonLinearTerm") -> Tuple[NDArray, NDArray]:
    """
    Compute spatial profiles of term residual (signed) averaged over each direction.

    Parameters
    ----------
    solver : FEMSolver2D
        The initialized FEM 2D solver instance.
    term : NonLinearTerm
        The term to analyze.

    Returns
    -------
    profile_x : NDArray
        Mean R(x) averaged over y, shape (Nx_inner,).
    profile_y : NDArray
        Mean R(y) averaged over x, shape (Ny_inner,).
    """
    comm = solver.problem.decomp._mpi_comm

    local_residual = solver.residual_vector_term(term)
    residual_2d = local_residual.reshape(
        (solver.Nx_inner, solver.Ny_inner), order='F'
    )

    local_profile_x = np.mean(residual_2d, axis=1)
    local_profile_y = np.mean(residual_2d, axis=0)

    global_profile_x = np.zeros_like(local_profile_x)
    global_profile_y = np.zeros_like(local_profile_y)

    comm.Allreduce(local_profile_x, global_profile_x, op=MPI.SUM)
    comm.Allreduce(local_profile_y, global_profile_y, op=MPI.SUM)

    n_ranks = comm.Get_size()
    global_profile_x /= n_ranks
    global_profile_y /= n_ranks

    return global_profile_x, global_profile_y


def compute_total_residual_profile(solver: "FEMSolver2D",
                                   res_name: str) -> Tuple[NDArray, NDArray]:
    """
    Compute spatial profile of total residual for an equation.

    Parameters
    ----------
    solver : FEMSolver2D
        The initialized FEM 2D solver instance.
    res_name : str
        Name of the residual equation.

    Returns
    -------
    profile_x : NDArray
        Mean total R(x) averaged over y, shape (Nx_inner,).
    profile_y : NDArray
        Mean total R(y) averaged over x, shape (Ny_inner,).
    """
    comm = solver.problem.decomp._mpi_comm

    total_residual = np.zeros(solver.Nx_inner * solver.Ny_inner)
    for term in solver.terms:
        if term.res == res_name:
            total_residual += solver.residual_vector_term(term)

    residual_2d = total_residual.reshape(
        (solver.Nx_inner, solver.Ny_inner), order='F'
    )

    local_profile_x = np.mean(residual_2d, axis=1)
    local_profile_y = np.mean(residual_2d, axis=0)

    global_profile_x = np.zeros_like(local_profile_x)
    global_profile_y = np.zeros_like(local_profile_y)

    comm.Allreduce(local_profile_x, global_profile_x, op=MPI.SUM)
    comm.Allreduce(local_profile_y, global_profile_y, op=MPI.SUM)

    n_ranks = comm.Get_size()
    global_profile_x /= n_ranks
    global_profile_y /= n_ranks

    return global_profile_x, global_profile_y


def create_residual_analysis_plot(solver: "FEMSolver2D", output_path: str) -> None:
    """
    Create and save a residual analysis plot showing term contributions.

    Layout (3 columns, n_equations rows):
    - Column 0: Bar chart (term contributions with %)
    - Column 1: x-profile (spatial distribution)
    - Column 2: y-profile (spatial distribution)

    Parameters
    ----------
    solver : FEMSolver2D
        The initialized FEM 2D solver instance.
    output_path : str
        Full path where the plot should be saved.
    """
    comm = solver.problem.decomp._mpi_comm
    rank = comm.Get_rank()
    p = solver.problem

    contributions = compute_term_contributions(solver)
    equation_totals = compute_residual_equation_totals(contributions)
    significant_terms = get_significant_terms(contributions, threshold=0.05)

    # Compute spatial profiles for significant terms and stabilization terms
    # Stabilization sub-terms are accumulated (signed) into grouped profiles
    STAB_GROUPS = {'R1PSPG_': 'PSPG', 'R2GLS_': 'GLS'}
    spatial_profiles = {}
    stab_profiles: Dict[Tuple[str, str], Dict] = {}  # (label, res) -> profile
    for term in solver.terms:
        group_label = None
        for prefix, label in STAB_GROUPS.items():
            if term.name.startswith(prefix):
                group_label = label
                break

        if group_label is not None:
            profile_x, profile_y = compute_term_spatial_profiles(solver, term)
            key = (group_label, term.res)
            if key not in stab_profiles:
                stab_profiles[key] = {
                    'profile_x': np.zeros_like(profile_x),
                    'profile_y': np.zeros_like(profile_y),
                    'res': term.res,
                    'is_stabilization': True,
                }
            stab_profiles[key]['profile_x'] += profile_x
            stab_profiles[key]['profile_y'] += profile_y
        else:
            is_significant = term.name in significant_terms.get(term.res, [])
            is_stabilization = 'Stab' in term.name
            if is_significant or is_stabilization:
                profile_x, profile_y = compute_term_spatial_profiles(solver, term)
                spatial_profiles[term.name] = {
                    'profile_x': profile_x,
                    'profile_y': profile_y,
                    'res': term.res,
                    'is_stabilization': is_stabilization,
                }
    for (label, res), profile in stab_profiles.items():
        # Use per-equation key when a group spans multiple equations
        res_keys = [r for (l, r) in stab_profiles if l == label]
        key = f'{label} ({res})' if len(res_keys) > 1 else label
        spatial_profiles[key] = profile

    # Compute total residual profiles for each equation
    total_profiles = {}
    for res_name in solver.residuals:
        profile_x, profile_y = compute_total_residual_profile(solver, res_name)
        total_profiles[res_name] = {
            'profile_x': profile_x,
            'profile_y': profile_y,
        }

    if rank != 0:
        return

    n_equations = len(solver.residuals)
    n_cols = 3  # bar chart, x-profile, y-profile
    n_rows = n_equations

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    x_coords = np.linspace(0, p.grid['Lx'], solver.Nx_inner) * 1e3
    y_coords = np.linspace(0, p.grid['Ly'], solver.Ny_inner) * 1e3

    for i, res_name in enumerate(solver.residuals):
        ax_bar = axes[i, 0]
        ax_x = axes[i, 1]
        ax_y = axes[i, 2]

        # Column 0: Bar chart
        _plot_equation_bar_chart(ax_bar, contributions[res_name],
                                 equation_totals[res_name], res_name)

        # Columns 1-2: Spatial profiles
        # Include significant terms and stabilization terms for this equation
        res_terms = [name for name, data in spatial_profiles.items()
                     if data['res'] == res_name]

        # Plot total residual first (dashed black)
        total_prof = total_profiles[res_name]
        ax_x.plot(x_coords, total_prof['profile_x'], 'k--', lw=1.5, label='Total', alpha=0.7)
        ax_y.plot(y_coords, total_prof['profile_y'], 'k--', lw=1.5, label='Total', alpha=0.7)

        # Plot individual terms (stabilization terms with dashed lines)
        if res_terms:
            for j, term_name in enumerate(res_terms):
                profile = spatial_profiles[term_name]
                color = TERM_COLORS[j % len(TERM_COLORS)]
                linestyle = '--' if profile['is_stabilization'] else '-'
                ax_x.plot(x_coords, profile['profile_x'], label=term_name,
                          color=color, lw=1.2, linestyle=linestyle)
                ax_y.plot(y_coords, profile['profile_y'], label=term_name,
                          color=color, lw=1.2, linestyle=linestyle)

        ax_x.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax_y.axhline(0, color='k', lw=0.5, alpha=0.5)

        # Remove empty space at edges
        ax_x.set_xlim(x_coords[0], x_coords[-1])
        ax_y.set_xlim(y_coords[0], y_coords[-1])

        ax_x.legend(fontsize=7, loc='best', framealpha=0.9)
        ax_y.legend(fontsize=7, loc='best', framealpha=0.9)

        res_label = _format_residual_name(res_name)
        ax_x.set_title(f'{res_label}: x-profile', fontsize=10)
        ax_y.set_title(f'{res_label}: y-profile', fontsize=10)
        ax_x.set_xlabel('x [mm]', fontsize=9)
        ax_y.set_xlabel('y [mm]', fontsize=9)
        ax_x.set_ylabel(r'$\langle R \rangle_y$', fontsize=9)
        ax_y.set_ylabel(r'$\langle R \rangle_x$', fontsize=9)

        _style_axis(ax_x)
        _style_axis(ax_y)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Residual analysis plot saved to: {output_path}")


def _format_residual_name(name: str) -> str:
    """Format residual name for display."""
    mapping = {
        'mass': 'Mass',
        'momentum_x': 'Momentum-x',
        'momentum_y': 'Momentum-y',
        'energy': 'Energy',
    }
    return mapping.get(name, name.replace('_', ' ').title())


def _plot_equation_bar_chart(ax, terms: Dict[str, float], total: float,
                             res_name: str) -> None:
    """Plot horizontal bar chart for a single equation's terms."""
    sorted_terms = sorted(terms.items(), key=lambda x: -x[1])
    term_names = [t[0] for t in sorted_terms]
    term_values = [t[1] for t in sorted_terms]

    y_pos = np.arange(len(term_names))
    colors = [TERM_COLORS[i % len(TERM_COLORS)] for i in range(len(term_names))]

    bars = ax.barh(y_pos, term_values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(term_names, fontsize=8)
    ax.set_xlabel(r'$\int |R| \, dA$', fontsize=9)
    ax.set_title(_format_residual_name(res_name), fontsize=10, fontweight='bold')
    ax.invert_yaxis()

    # Add percentage labels
    max_val = max(term_values) if term_values else 1
    min_offset = 0.02 * max_val  # Minimum offset from y-axis for small bars
    for bar, val in zip(bars, term_values):
        pct = 100 * val / total if total > 0 else 0
        label = f'{pct:.1f}%'
        # Position label inside or outside bar depending on space
        if val > 0.3 * max_val:
            ax.text(bar.get_width() * 0.95, bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha='right', fontsize=7, color='white',
                    fontweight='bold')
        else:
            # Use minimum offset to avoid overlap with y-axis
            x_pos = max(bar.get_width() * 1.02, min_offset)
            ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha='left', fontsize=7)

    _style_axis(ax)


def plot_residual_history(R_norm_history: List[List[float]],
                          max_separators: int = 50):
    """Plot R_norm evolution across all Newton iterations.

    Parameters
    ----------
    R_norm_history : list of list of float
        Nested list: [[step1_norms], [step2_norms], ...].
    max_separators : int
        Maximum number of timestep separators to show (default 50).

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects, or (None, None) if
        no history available.
    """
    if not R_norm_history:
        print("No residual history available.")
        return None, None

    # Flatten history with timestep boundaries
    all_norms = []
    boundaries = []
    for step_history in R_norm_history:
        boundaries.append(len(all_norms))
        all_norms.extend(step_history)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(all_norms, 'b-', linewidth=0.8)

    # Add vertical separators if not too many timesteps
    if len(boundaries) <= max_separators:
        for b in boundaries[1:]:  # Skip first (at 0)
            ax.axvline(x=b, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Newton iteration (cumulative)')
    ax.set_ylabel('R_norm')
    ax.set_title(f'Residual history ({len(R_norm_history)} timesteps)')
    ax.grid(True, alpha=0.3)

    return fig, ax


def print_residual_analysis(solver: "FEMSolver2D") -> None:
    """
    Print a text summary of residual term contributions (rank 0 only).

    Parameters
    ----------
    solver : FEMSolver2D
        The initialized FEM 2D solver instance.
    """
    comm = solver.problem.decomp._mpi_comm
    rank = comm.Get_rank()

    contributions = compute_term_contributions(solver)
    equation_totals = compute_residual_equation_totals(contributions)

    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS")
    print("=" * 60)

    grand_total = sum(equation_totals.values())

    for res_name in solver.residuals:
        res_total = equation_totals[res_name]
        pct = 100 * res_total / grand_total if grand_total > 0 else 0
        print(f"\n{res_name.upper()} ({pct:.1f}% of total)")
        print("-" * 40)

        terms = contributions[res_name]
        for term_name, value in sorted(terms.items(), key=lambda x: -x[1]):
            term_pct = 100 * value / res_total if res_total > 0 else 0
            print(f"  {term_name:<12s}: {value:12.4e}  ({term_pct:5.1f}%)")

    print("\n" + "=" * 60)
    print(f"TOTAL RESIDUAL: {grand_total:.4e}")
    print("=" * 60 + "\n")
