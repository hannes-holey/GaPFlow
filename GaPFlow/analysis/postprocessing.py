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
Postprocessing utilities for GaPFlow simulations.

Provides metrics computation and overview plotting with MPI support
for scaling studies and general analysis.

Features
--------
- Metrics: Automatically printed when `output_metrics: True` in config
- Plots: Overview figure saved when `output_plots: True` in config

Metrics Output Format::

    ==================================================
    METRICS OUTPUT
    ==================================================
    coeff_friction: 0.00234
    density_max: 1.2345
    ...
    ==================================================
"""
import resource
from datetime import datetime

import numpy as np
from mpi4py import MPI

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..problem import Problem


def compute_metrics(problem: "Problem") -> dict:
    """
    Compute all characteristic metrics from a completed simulation.

    Parameters
    ----------
    problem : Problem
        The completed GaPFlow problem instance.

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    comm = MPI.COMM_WORLD

    metrics = {}
    metrics.update(_density_metrics(problem, comm))
    metrics.update(_pressure_metrics(problem, comm))
    metrics.update(_flow_metrics(problem, comm))
    metrics.update(_friction_metrics(problem, comm))
    metrics.update(_deformation_metrics(problem, comm))
    metrics.update(_energy_metrics(problem, comm))
    metrics.update(_timing_metrics(problem))
    metrics.update(_solver_metrics(problem))
    metrics.update(_memory_metrics(comm))

    return metrics


def _density_metrics(problem: "Problem", comm: MPI.Comm) -> dict:
    """Compute density field statistics."""
    rho = problem.q[0, 1:-1, 1:-1]  # Inner cells only (no ghosts)
    grid_total = problem.grid['Nx'] * problem.grid['Ny']

    return {
        "density_min": comm.allreduce(float(np.min(rho)), op=MPI.MIN),
        "density_max": comm.allreduce(float(np.max(rho)), op=MPI.MAX),
        "density_mean": comm.allreduce(float(np.sum(rho)), op=MPI.SUM) / grid_total,
    }


def _pressure_metrics(problem: "Problem", comm: MPI.Comm) -> dict:
    """Compute pressure field statistics and integral."""
    p = problem.pressure.pressure[1:-1, 1:-1]  # Inner cells only
    dx = problem.grid['dx']
    dy = problem.grid['dy']
    grid_total = problem.grid['Nx'] * problem.grid['Ny']

    return {
        "pressure_min": comm.allreduce(float(np.min(p)), op=MPI.MIN),
        "pressure_max": comm.allreduce(float(np.max(p)), op=MPI.MAX),
        "pressure_mean": comm.allreduce(float(np.sum(p)), op=MPI.SUM) / grid_total,
        "pressure_integral": comm.allreduce(float(np.sum(p)), op=MPI.SUM) * dx * dy,
    }


def _flow_metrics(problem: "Problem", comm: MPI.Comm) -> dict:
    """Compute momentum/flow statistics (integrated over domain)."""
    jx = problem.q[1, 1:-1, 1:-1]
    jy = problem.q[2, 1:-1, 1:-1]
    dx = problem.grid['dx']
    dy = problem.grid['dy']
    dA = dx * dy

    return {
        "flow_x_total": comm.allreduce(float(np.sum(jx)), op=MPI.SUM) * dA,
        "flow_y_total": comm.allreduce(float(np.sum(jy)), op=MPI.SUM) * dA,
    }


def _friction_metrics(problem: "Problem", comm: MPI.Comm) -> dict:
    """Compute friction force and coefficient of friction.

    Tracks both lower and upper wall stresses.
    CoF = Friction Force / Normal Force
    """
    # Wall shear stress (inner cells)
    tau_xz_lower = problem.wall_stress_xz.lower[4, 1:-1, 1:-1]
    tau_xz_upper = problem.wall_stress_xz.upper[4, 1:-1, 1:-1]
    tau_yz_lower = problem.wall_stress_yz.lower[3, 1:-1, 1:-1]
    tau_yz_upper = problem.wall_stress_yz.upper[3, 1:-1, 1:-1]

    p = problem.pressure.pressure[1:-1, 1:-1]
    dx = problem.grid['dx']
    dy = problem.grid['dy']
    dA = dx * dy

    # Friction forces (integral of wall shear stress)
    friction_x_lower = comm.allreduce(float(np.sum(tau_xz_lower)), op=MPI.SUM) * dA
    friction_x_upper = comm.allreduce(float(np.sum(tau_xz_upper)), op=MPI.SUM) * dA
    friction_y_lower = comm.allreduce(float(np.sum(tau_yz_lower)), op=MPI.SUM) * dA
    friction_y_upper = comm.allreduce(float(np.sum(tau_yz_upper)), op=MPI.SUM) * dA

    # Normal force (integral of pressure)
    normal_force = comm.allreduce(float(np.sum(p)), op=MPI.SUM) * dA

    # Total friction magnitude for each wall
    friction_lower = np.sqrt(friction_x_lower**2 + friction_y_lower**2)
    friction_upper = np.sqrt(friction_x_upper**2 + friction_y_upper**2)

    # Coefficient of friction
    cof_lower = friction_lower / normal_force if normal_force != 0 else None
    cof_upper = friction_upper / normal_force if normal_force != 0 else None

    return {
        "friction_x_lower": friction_x_lower,
        "friction_x_upper": friction_x_upper,
        "friction_y_lower": friction_y_lower,
        "friction_y_upper": friction_y_upper,
        "normal_force": normal_force,
        "coeff_friction_lower": cof_lower,
        "coeff_friction_upper": cof_upper,
    }


def _deformation_metrics(problem: "Problem", comm: MPI.Comm) -> dict:
    """Compute elastic deformation statistics.

    Only computed if elastic deformation is enabled.
    """
    if not getattr(problem.topo, 'elastic', False):
        return {}

    # Deformation field (inner cells)
    defo = problem.topo.deformation[1:-1, 1:-1]
    grid_total = problem.grid['Nx'] * problem.grid['Ny']

    return {
        "deformation_min": comm.allreduce(float(np.min(defo)), op=MPI.MIN),
        "deformation_max": comm.allreduce(float(np.max(defo)), op=MPI.MAX),
        "deformation_mean": comm.allreduce(float(np.sum(defo)), op=MPI.SUM) / grid_total,
    }


def _energy_metrics(problem: "Problem", comm: MPI.Comm) -> dict:
    """Compute energy and temperature statistics.

    Only computed if energy equation is enabled.
    """
    if not getattr(problem, 'bEnergy', False):
        return {}

    # Temperature field (inner cells)
    T = problem.energy.temperature[1:-1, 1:-1]
    E = problem.energy.energy[1:-1, 1:-1]
    grid_total = problem.grid['Nx'] * problem.grid['Ny']

    metrics = {
        "temperature_min": comm.allreduce(float(np.min(T)), op=MPI.MIN),
        "temperature_max": comm.allreduce(float(np.max(T)), op=MPI.MAX),
        "temperature_mean": comm.allreduce(float(np.sum(T)), op=MPI.SUM) / grid_total,
        "energy_min": comm.allreduce(float(np.min(E)), op=MPI.MIN),
        "energy_max": comm.allreduce(float(np.max(E)), op=MPI.MAX),
    }

    # Wall heat flux
    try:
        from ..models.heatflux import get_heatflux_2d
        q_top, q_bot = get_heatflux_2d(problem)
        q_top_inner = q_top[1:-1, 1:-1]
        q_bot_inner = q_bot[1:-1, 1:-1]

        metrics.update({
            "heatflux_top_min": comm.allreduce(float(np.min(q_top_inner)), op=MPI.MIN),
            "heatflux_top_max": comm.allreduce(float(np.max(q_top_inner)), op=MPI.MAX),
            "heatflux_top_mean": comm.allreduce(float(np.sum(q_top_inner)), op=MPI.SUM) / grid_total,
            "heatflux_bot_min": comm.allreduce(float(np.min(q_bot_inner)), op=MPI.MIN),
            "heatflux_bot_max": comm.allreduce(float(np.max(q_bot_inner)), op=MPI.MAX),
            "heatflux_bot_mean": comm.allreduce(float(np.sum(q_bot_inner)), op=MPI.SUM) / grid_total,
        })
    except Exception:
        # Heat flux computation may fail in some configurations
        pass

    return metrics


def _timing_metrics(problem: "Problem") -> dict:
    """Compute timing metrics."""
    metrics = {}

    # Wall time from problem._tic
    if hasattr(problem, '_tic') and problem._tic is not None:
        walltime = datetime.now() - problem._tic
        metrics["wall_time"] = walltime.total_seconds()

    return metrics


def _solver_metrics(problem: "Problem") -> dict:
    """Compute solver iteration metrics."""
    metrics = {
        "timesteps": problem.step,
    }

    solver = problem.solver

    # Total inner iterations across all timesteps
    if hasattr(solver, 'R_norm_history') and solver.R_norm_history:
        total_inner_iterations = sum(len(step_hist) for step_hist in solver.R_norm_history)
        metrics["total_inner_iterations"] = total_inner_iterations

    # Final residual
    if hasattr(problem, 'residual'):
        metrics["final_residual"] = problem.residual

    return metrics


def _memory_metrics(comm: MPI.Comm) -> dict:
    """Compute memory usage metrics.

    Uses resource.getrusage to get peak memory (max RSS).
    Reports both total (sum) and max across all MPI ranks.
    """
    # ru_maxrss is in KB on Linux, bytes on macOS
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_kb / 1024  # Convert KB to MB (Linux)

    # Total memory across all ranks (system footprint)
    total_peak_mb = comm.allreduce(peak_mb, op=MPI.SUM)
    # Max memory on any single rank (for node limits)
    max_peak_mb = comm.allreduce(peak_mb, op=MPI.MAX)

    return {
        "peak_memory_total_mb": total_peak_mb,
        "peak_memory_max_mb": max_peak_mb,
    }


def print_metrics(metrics: dict, comm: MPI.Comm) -> None:
    """
    Print metrics in a parseable format (rank 0 only).

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names and values.
    comm : MPI.Comm
        MPI communicator.
    """
    if comm.Get_rank() != 0:
        return

    print("\n" + "=" * 50)
    print("METRICS OUTPUT")
    print("=" * 50)
    for key, value in sorted(metrics.items()):
        if value is not None:
            print(f"{key}: {value}")
        else:
            print(f"{key}: N/A")
    print("=" * 50)


def create_overview_plot(problem: "Problem", output_path: str) -> None:
    """
    Create and save an overview plot with key simulation fields.

    Generates a high-resolution figure with density, mass fluxes,
    energy, and deformed height profile using sharp color boundaries.

    Parameters
    ----------
    problem : Problem
        The completed GaPFlow problem instance.
    output_path : str
        Full path where the plot should be saved (e.g., '/path/to/overview.png').

    Notes
    -----
    - Only rank 0 creates the plot (fields are gathered from all ranks)
    - Uses discrete colormap levels for sharp color boundaries
    - Saves at 300 DPI for high resolution
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Gather fields to rank 0
    g = problem.decomp.gather_global

    rho = g(problem.q[0])
    jx = g(problem.q[1])
    jy = g(problem.q[2])
    h = g(problem.topo.h)

    # Energy field (if enabled)
    has_energy = getattr(problem, 'bEnergy', False)
    if has_energy:
        E = g(problem.energy.energy)
    else:
        E = None

    # Only rank 0 creates the plot
    if rank != 0:
        return

    # Grid coordinates
    Nx, Ny = problem.grid['Nx'], problem.grid['Ny']
    Lx, Ly = problem.grid['Lx'], problem.grid['Ly']
    dx, dy = problem.grid['dx'], problem.grid['dy']

    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Number of discrete color levels for sharp boundaries
    n_levels = 20

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True, facecolor='white')

    def plot_field(ax, field, title, cmap='viridis'):
        """Plot a field with sharp color boundaries."""
        vmin, vmax = np.min(field), np.max(field)
        # Handle case where field is constant
        if vmin == vmax:
            vmax = vmin + 1e-10
        levels = np.linspace(vmin, vmax, n_levels + 1)
        norm = BoundaryNorm(levels, plt.get_cmap(cmap).N)

        cf = ax.contourf(X * 1e3, Y * 1e3, field, levels=levels,
                         cmap=cmap, norm=norm)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(title)
        ax.set_aspect('equal')
        fig.colorbar(cf, ax=ax, shrink=0.8)
        return cf

    # Plot density
    plot_field(axes[0, 0], rho, r'Density $\rho$ [kg/m³]', 'viridis')

    # Plot mass flux x
    plot_field(axes[0, 1], jx, r'Mass flux $j_x$ [kg/(m²·s)]', 'RdBu_r')

    # Plot mass flux y
    plot_field(axes[0, 2], jy, r'Mass flux $j_y$ [kg/(m²·s)]', 'RdBu_r')

    # Plot energy or placeholder
    if E is not None:
        plot_field(axes[1, 0], E, r'Energy $E$ [J/m³]', 'inferno')
    else:
        axes[1, 0].text(0.5, 0.5, 'Energy equation\nnot enabled',
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        fontsize=12, color='gray')
        axes[1, 0].set_title('Energy')
        axes[1, 0].axis('off')

    # Plot deformed height
    plot_field(axes[1, 1], h * 1e6, r'Gap height $h$ [µm]', 'plasma')

    # Info panel
    ax_info = axes[1, 2]
    ax_info.axis('off')

    info_text = (
        f"Grid: {Nx} × {Ny}\n"
        f"Domain: {Lx * 1e3:.2f} × {Ly * 1e3:.2f} mm\n"
        f"Timesteps: {problem.step}\n"
        f"\n"
        f"ρ: [{np.min(rho):.4f}, {np.max(rho):.4f}] kg/m³\n"
        f"h: [{np.min(h) * 1e6:.2f}, {np.max(h) * 1e6:.2f}] µm\n"
    )
    if E is not None:
        info_text += f"E: [{np.min(E):.2e}, {np.max(E):.2e}] J/m³\n"

    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_info.set_title('Run Info')

    # Main title
    fig.suptitle(f'GaPFlow Overview - Step {problem.step}', fontsize=14, fontweight='bold')

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Overview plot saved to: {output_path}")
