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
"""Plotting utilities for FEM tutorials."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3d projection)
from IPython.display import HTML


def animate_comparison(x_vec, t_vec, T_numeric, T_analytic, L, save_path=None):
    """Animated comparison of numeric vs analytic temperature solutions.

    Parameters
    ----------
    x_vec : array_like
        Spatial coordinates, shape (nx,)
    t_vec : array_like
        Time values, shape (nt,)
    T_numeric : ndarray
        Numeric temperature field, shape (nt, nx)
    T_analytic : ndarray
        Analytic temperature field, shape (nt, nx)
    L : float
        Domain length
    save_path : str, optional
        Path to save the animation (e.g., 'animation.mp4' or 'animation.gif').
        Requires ffmpeg for mp4 or pillow for gif.

    Returns
    -------
    HTML
        Jupyter HTML animation object
    """
    # Print error summary
    print(f"Initial max error: {np.max(np.abs(T_analytic[0] - T_numeric[0])):.2e} (should be ~0)")
    print(f"Overall max error: {np.max(np.abs(T_analytic - T_numeric)):.2e} (numerical discretization)")

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    line_fem, = ax1.plot([], [], 'b-', linewidth=2, label='FEM')
    line_ana, = ax1.plot([], [], 'r--', linewidth=2, label='Analytic')
    ax1.set_xlim(0, L)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('T [K]')
    ax1.set_title('Temperature Profiles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    line_err, = ax2.plot([], [], 'g-', linewidth=2)
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, 0.02)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('|T_FEM - T_analytic|')
    ax2.set_title('Absolute Error')
    ax2.grid(True, alpha=0.3)

    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=10)
    error_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=10)

    def init():
        line_fem.set_data([], [])
        line_ana.set_data([], [])
        line_err.set_data([], [])
        time_text.set_text('')
        error_text.set_text('')
        return line_fem, line_ana, line_err, time_text, error_text

    def animate(i):
        line_fem.set_data(x_vec, T_numeric[i])
        line_ana.set_data(x_vec, T_analytic[i])
        error = np.abs(T_numeric[i] - T_analytic[i])
        line_err.set_data(x_vec, error)
        time_text.set_text(f't = {t_vec[i]:.3f} s')
        error_text.set_text(f'max error = {np.max(error):.2e}')
        return line_fem, line_ana, line_err, time_text, error_text

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_vec),
                                  interval=50, blit=True, repeat=True)

    if save_path is not None:
        ani.save(save_path, writer='ffmpeg' if save_path.endswith('.mp4') else 'pillow')
        print(f"Animation saved to: {save_path}")

    plt.close(fig)
    return HTML(ani.to_jshtml())


PAPER_RC = {
    'font.family': 'serif',
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.grid': False,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
}


def plot_solver_comparison_rho_jx(results):
    """Plot density and momentum profiles comparing different solvers.

    Creates a side-by-side comparison of density (rho) and x-momentum (jx)
    profiles from multiple solver results.

    Parameters
    ----------
    results : dict
        Dictionary with solver names as keys and result dicts as values.
        Each result dict must contain 'rho' and 'jx' arrays.
        Expected keys: 'explicit', 'implicit'
    """
    colors = {'explicit': '#0072B2', 'implicit': '#D55E00'}
    linestyles = {'explicit': '-', 'implicit': '--'}
    labels = {'explicit': 'Explicit', 'implicit': 'Implicit'}

    Nx = len(list(results.values())[0]['rho'])
    x = np.linspace(0, 1, Nx)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=False)

        for name, res in results.items():
            kw = dict(color=colors[name], ls=linestyles[name], lw=1.5, label=labels[name])
            axes[0].plot(x, res['rho'], **kw)
            axes[1].plot(x, res['jx'], **kw)

        axes[0].set_ylabel(r'$\rho$ [kg/m$^3$]')
        axes[1].set_ylabel(r'$j_x$ [kg/(m$^2$s)]')
        for ax in axes:
            ax.set_xlabel('$X$')
            ax.set_xlim(0, 1)
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.tick_params(direction='in', top=True, right=True)

        handles, names = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_stab_comparison(results, title, p_divisor=1e5, p_unit='bar',
                         ref_csv=None, ref_label=None,
                         ref_x_scale=1.0, ref_p_scale=1.0):
    """Plot pressure and cavitation fraction (theta) profiles comparing
    the four stabilization variants (NS-OSS, NS-SUPG, NS-both, Reynolds-SUPG).

    Parameters
    ----------
    results : dict
        Dictionary with variant names as keys and result dicts as values.
        Each result dict must contain 'p' and 'theta' arrays.
        Expected keys: 'ns_oss', 'ns_supg', 'ns_both', 'rey_supg'
    title : str
        Title for the figure
    ref_csv : str or Path, optional
        Path to a CSV with columns 'x' and 'y' for reference pressure data.
    ref_label : str, optional
        Legend label for the reference data.
    ref_x_scale : float
        Multiplier applied to the reference x-column (to map onto x/L in [0, 1]).
    ref_p_scale : float
        Multiplier applied to the reference y-column to convert to Pa.

    Example
    -------
    >>> results = {
    ...     'ns_oss': {'p': p1, 'theta': th1, 'time': t1},
    ...     'ns_supg': {'p': p2, 'theta': th2, 'time': t2},
    ...     'ns_both': {'p': p3, 'theta': th3, 'time': t3},
    ...     'rey_supg': {'p': p4, 'theta': th4, 'time': t4},
    ... }
    >>> plot_stab_comparison(results, 'Twin Parabolic Slider, Identical')
    """
    fig, (ax_p, ax_theta) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'ns_oss': 'C0', 'ns_supg': 'C1', 'ns_both': 'C2', 'rey_supg': 'C3'}
    labels = {'ns_oss': 'NS-OSS', 'ns_supg': 'NS-SUPG', 'ns_both': 'NS-OSS+SUPG',
              'rey_supg': 'Reynolds-SUPG'}

    Nx = len(list(results.values())[0]['p'])
    x = np.linspace(0, 1, Nx)

    for name, res in results.items():
        ax_p.plot(x, res['p'] / p_divisor, color=colors[name], label=labels[name], lw=1.5)
        ax_theta.plot(x, res['theta'], color=colors[name], label=labels[name], lw=1.5)

    if ref_csv is not None:
        ref = pd.read_csv(ref_csv, skipinitialspace=True)
        ax_p.scatter(ref['x'].values * ref_x_scale,
                     ref['y'].values * ref_p_scale / p_divisor,
                     color='#1f77b4', s=14, zorder=5, label=ref_label)

    ax_p.set_xlabel('x / L')
    ax_p.set_ylabel(f'p [{p_unit}]')
    ax_p.legend()
    ax_p.grid(True, alpha=0.3)

    ax_theta.set_xlabel('x / L')
    ax_theta.set_ylabel(r'$\theta$ [-]')
    ax_theta.legend()
    ax_theta.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


def animate_advection(x_vec, t_vec, T_numeric, L, U_avg, T_min=None, T_max=None,
                      save_path=None):
    """Animated visualization of temperature advection.

    Parameters
    ----------
    x_vec : array_like
        Spatial coordinates, shape (nx,)
    t_vec : array_like
        Time values, shape (nt,)
    T_numeric : ndarray
        Numeric temperature field, shape (nt, nx)
    L : float
        Domain length
    U_avg : float
        Average flow velocity for advection
    T_min, T_max : float, optional
        Temperature range for y-axis. If None, computed from data.
    save_path : str, optional
        Path to save the animation (e.g., 'animation.mp4' or 'animation.gif').
        Requires ffmpeg for mp4 or pillow for gif.

    Returns
    -------
    HTML
        Jupyter HTML animation object
    """
    if T_min is None:
        T_min = np.min(T_numeric) - 0.05 * (np.max(T_numeric) - np.min(T_numeric))
    if T_max is None:
        T_max = np.max(T_numeric) + 0.05 * (np.max(T_numeric) - np.min(T_numeric))

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot initial profile as reference
    line_init, = ax.plot(x_vec, T_numeric[0], 'r--', linewidth=1.5,
                         alpha=0.5, label='Initial')
    line_T, = ax.plot([], [], 'b-', linewidth=2, label='T(x,t)')

    ax.set_xlim(0, L)
    ax.set_ylim(T_min, T_max)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('T [K]')
    ax.set_title('Temperature Advection')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
    dist_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10)

    def init():
        line_T.set_data([], [])
        time_text.set_text('')
        dist_text.set_text('')
        return line_T, time_text, dist_text

    def animate(i):
        line_T.set_data(x_vec, T_numeric[i])
        t = t_vec[i]
        distance = U_avg * t
        # Account for periodic wrapping
        distance_wrapped = distance % L
        time_text.set_text(f't = {t:.4f} s')
        dist_text.set_text(f'Δx = {distance_wrapped:.4f} m ({distance / L:.2f} periods)')
        return line_T, time_text, dist_text

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_vec),
                                  interval=50, blit=True, repeat=True)

    if save_path is not None:
        ani.save(save_path, writer='ffmpeg' if save_path.endswith('.mp4') else 'pillow')
        print(f"Animation saved to: {save_path}")

    plt.close(fig)
    return HTML(ani.to_jshtml())


def animate_3d_surface(x_vec, y_vec, t_vec, T_field, Lx=1.0, Ly=1.0,
                       T_min=None, T_max=None, title='Temperature Field',
                       view_angle=(30, -60), cmap='coolwarm', figsize=(8, 5),
                       save_path=None):
    """3D surface animation of a scalar field (e.g., temperature) over time.

    Parameters
    ----------
    x_vec : array_like
        Spatial x-coordinates, shape (nx,)
    y_vec : array_like
        Spatial y-coordinates, shape (ny,)
    t_vec : array_like
        Time values, shape (nt,)
    T_field : ndarray
        Scalar field over time, shape (nt, nx, ny)
    Lx, Ly : float
        Domain dimensions for axis labels
    T_min, T_max : float, optional
        Field value range for z-axis and colormap. If None, computed from data.
    title : str
        Title for the animation
    view_angle : tuple
        (elevation, azimuth) angles for 3D view
    cmap : str
        Colormap name (default: 'coolwarm' - red-white-blue diverging)
    figsize : tuple
        Figure size (width, height) in inches
    save_path : str, optional
        Path to save the animation (e.g., 'animation.mp4' or 'animation.gif').
        Requires ffmpeg for mp4 or pillow for gif.

    Returns
    -------
    HTML
        Jupyter HTML animation object
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    # Compute limits if not provided
    if T_min is None:
        T_min = np.min(T_field)
    if T_max is None:
        T_max = np.max(T_field)

    # Add margin to z-limits
    z_margin = 0.05 * (T_max - T_min) if T_max > T_min else 0.1
    z_min, z_max = T_min - z_margin, T_max + z_margin

    # Create meshgrid for surface plot
    X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

    # Normalize for consistent colormap across frames
    norm = Normalize(vmin=T_min, vmax=T_max)
    colormap = cm.get_cmap(cmap)

    # Create figure with 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Initial surface with facecolors based on Z values
    fcolors = colormap(norm(T_field[0]))
    ax.plot_surface(X, Y, T_field[0], facecolors=fcolors,
                    linewidth=0, antialiased=True, shade=True)

    # Set axis properties
    ax.set_xlim(x_vec.min(), x_vec.max())
    ax.set_ylim(y_vec.min(), y_vec.max())
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('T [K]')
    ax.set_title(f'{title}  |  t = {t_vec[0]:.4f} s')
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Add colorbar using ScalarMappable
    mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Temperature [K]')

    def animate(i):
        ax.clear()

        # Recreate surface with facecolors
        fcolors = colormap(norm(T_field[i]))
        ax.plot_surface(X, Y, T_field[i], facecolors=fcolors,
                        linewidth=0, antialiased=True, shade=True)

        # Restore axis properties
        ax.set_xlim(x_vec.min(), x_vec.max())
        ax.set_ylim(y_vec.min(), y_vec.max())
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('T [K]')
        ax.set_title(f'{title}  |  t = {t_vec[i]:.4f} s')
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(t_vec),
                                  interval=100, repeat=True, blit=False)

    if save_path is not None:
        ani.save(save_path, writer='ffmpeg' if save_path.endswith('.mp4') else 'pillow')
        print(f"Animation saved to: {save_path}")

    plt.close(fig)
    return HTML(ani.to_jshtml())


def animate_3d_advection(x_vec, t_vec, T_numeric, Lx, U_avg,
                         T_min=None, T_max=None, view_angle=(25, -45)):
    """3D animation of 1D temperature advection as ribbon/surface.

    Shows the spatial temperature profile evolving in time as a 3D surface
    where one axis is space (x), one is time, and height is temperature.

    Parameters
    ----------
    x_vec : array_like
        Spatial coordinates, shape (nx,)
    t_vec : array_like
        Time values, shape (nt,)
    T_numeric : ndarray
        Temperature field over time, shape (nt, nx)
    Lx : float
        Domain length
    U_avg : float
        Average flow velocity
    T_min, T_max : float, optional
        Temperature range for z-axis. If None, computed from data.
    view_angle : tuple
        (elevation, azimuth) angles for 3D view

    Returns
    -------
    HTML
        Jupyter HTML animation object
    """
    if T_min is None:
        T_min = np.min(T_numeric)
    if T_max is None:
        T_max = np.max(T_numeric)

    z_margin = 0.05 * (T_max - T_min) if T_max > T_min else 0.1
    z_min, z_max = T_min - z_margin, T_max + z_margin

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Meshgrid for the ribbon (x vs frame index)
    nx = len(x_vec)
    ribbon_depth = 5  # Number of time slices to show as ribbon

    def animate(frame):
        ax.clear()

        # Show recent history as fading ribbons
        start_idx = max(0, frame - ribbon_depth + 1)
        for i, idx in enumerate(range(start_idx, frame + 1)):
            alpha = 0.3 + 0.7 * (i / ribbon_depth)  # Fade older frames
            t_pos = t_vec[idx]
            # Use a single color based on time progression
            color = plt.cm.viridis(idx / max(len(t_vec) - 1, 1))
            ax.plot(x_vec, np.full(nx, t_pos), T_numeric[idx],
                    color=color, alpha=alpha, linewidth=2)

        # Current profile as thick line
        t_current = t_vec[frame]
        ax.plot(x_vec, np.full(nx, t_current), T_numeric[frame],
                'b-', linewidth=3, label=f't = {t_current:.4f} s')

        # Initial profile reference
        ax.plot(x_vec, np.zeros(nx), T_numeric[0],
                'r--', linewidth=1.5, alpha=0.5, label='Initial')

        ax.set_xlim(0, Lx)
        ax.set_ylim(0, t_vec[-1])
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('t [s]')
        ax.set_zlabel('T [K]')

        distance = U_avg * t_current
        periods = distance / Lx
        ax.set_title(f'Temperature Advection  |  Δx = {distance:.3f} m ({periods:.2f} periods)')
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        ax.legend(loc='upper left')

        return []

    ani = animation.FuncAnimation(fig, animate, frames=len(t_vec),
                                  interval=100, repeat=True, blit=False)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_3d_snapshot(x_vec, y_vec, T_field, Lx=1.0, Ly=1.0,
                     T_min=None, T_max=None, title='Temperature Field',
                     view_angle=(30, -60), cmap='plasma', figsize=(10, 7)):
    """Static 3D surface plot of a scalar field.

    Parameters
    ----------
    x_vec : array_like
        Spatial x-coordinates, shape (nx,)
    y_vec : array_like
        Spatial y-coordinates, shape (ny,)
    T_field : ndarray
        Scalar field, shape (nx, ny)
    Lx, Ly : float
        Domain dimensions
    T_min, T_max : float, optional
        Field value range for colormap
    title : str
        Plot title
    view_angle : tuple
        (elevation, azimuth) angles
    cmap : str
        Colormap name
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    if T_min is None:
        T_min = np.min(T_field)
    if T_max is None:
        T_max = np.max(T_field)

    X, Y = np.meshgrid(x_vec, y_vec, indexing='ij')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, T_field, cmap=cmap,
                           vmin=T_min, vmax=T_max,
                           linewidth=0, antialiased=True)

    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('T [K]')
    ax.set_title(title)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Temperature [K]')

    return fig, ax


def plot_overview_1d(problem, ref_csv=None, ref_label=None, ref_rho_csv=None,
                     ref_x_scale=1.0, ref_p_scale=1.0,
                     p_unit='bar', p_divisor=1e5):
    """Plot pressure and effective density for a 1D (y-periodic) cavitation result.

    Parameters
    ----------
    problem : Problem
        GaPFlow Problem instance after simulation.
    ref_csv : str or Path, optional
        Path to a CSV with columns 'x' and 'y' for reference pressure data.
    ref_label : str, optional
        Legend label for the reference data.
    ref_rho_csv : str or Path, optional
        Path to a CSV with columns 'x' and 'y' for reference effective density,
        given relative to the effective density at the inlet.
    ref_x_scale : float
        Multiplier applied to the reference x-column to convert to metres.
    ref_p_scale : float
        Multiplier applied to the reference y-column to convert to Pa.
    p_unit : str
        Pressure unit string used on the y-axis label (default 'bar').
    p_divisor : float
        Divisor applied to model pressure before plotting (default 1e5).
    """
    from GaPFlow.models.pressure import eos_pressure

    Nx = problem.grid['Nx']
    Lx = problem.grid['Lx']
    X = np.linspace(0, 1, Nx)

    rho = problem.q[0][1:-1, 1:-1]
    p_x = np.asarray(eos_pressure(rho, problem.prop)).mean(axis=1)

    if problem.prop.get('EOS') == 'Bayada':
        rho_l = problem.prop['rho_l']
        rho_v = problem.prop['rho_v']
        theta = np.clip((rho_l - rho) / (rho_l - rho_v), 0, 1)
        rho_eff_x = rho.mean(axis=1)
    else:
        theta = problem.q[3][1:-1, 1:-1]
        rho_eff_x = (rho * (1 - theta)).mean(axis=1)

    ref_style = dict(facecolors='none', edgecolors='0.15', marker='o', s=22, linewidths=0.8, zorder=5)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=False)

        axes[0].plot(X, p_x / p_divisor, color='#0072B2', lw=1.5, label='FEM')
        if ref_csv is not None:
            ref = pd.read_csv(ref_csv, skipinitialspace=True)
            axes[0].scatter(ref['x'].values * ref_x_scale / Lx,
                            ref['y'].values * ref_p_scale / p_divisor,
                            label=ref_label, **ref_style)
        axes[0].set_ylabel(f'$p$ [{p_unit}]')

        axes[1].plot(X, rho_eff_x, color='#0072B2', lw=1.5)
        if ref_rho_csv is not None:
            ref = pd.read_csv(ref_rho_csv, skipinitialspace=True)
            axes[1].scatter(ref['x'].values * ref_x_scale / Lx,
                            ref['y'].values * rho_eff_x[0], **ref_style)
        axes[1].set_ylabel(r'$\rho_\mathrm{eff}$ [kg/m$^3$]')

        for ax in axes:
            ax.set_xlabel('$X$')
            ax.set_xlim(0, 1)
        _style_axes(axes)

        handles, names = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_almqvist_validation(results, ref_dir=Path('../../tests/ref_data')):
    """Plot pressure profiles of the inclined slider against Almqvist reference data.

    Parameters
    ----------
    results : dict
        Keys (U, kind) with U in m/s and kind in {'comp', 'incomp'}; values (X, p)
        with normalized position X and pressure p in Pa.
    ref_dir : Path
        Directory holding the digitized Almqvist_<U>_<kind>.csv files
        (x in % of Lx, p in kPa).
    """
    colors = {5.: '#0072B2', 50.: '#D55E00', 125.: '#009E73'}
    linestyles = {5.: '-', 50.: '--', 125.: '-.'}
    markers = {5.: 'o', 50.: 's', 125.: '^'}
    panels = {'comp': 'compressible', 'incomp': 'incompressible-like'}

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=False)

        for ax, (kind, name) in zip(axes, panels.items()):
            for U in colors:
                X, p = results[(U, kind)]
                ax.plot(X, p / 1e5, color=colors[U], ls=linestyles[U], lw=1.0)

                ref = pd.read_csv(Path(ref_dir) / f'Almqvist_{U:g}_{kind}.csv', skipinitialspace=True)
                ref = ref.sort_values('x')
                ax.scatter(ref['x'] / 100, ref['y'] * 1e3 / 1e5, facecolors='none',
                           edgecolors=colors[U], marker=markers[U], s=22, linewidths=0.8, zorder=5)

            ax.text(0.05, 0.95, name, transform=ax.transAxes, ha='left', va='top')
            ax.set_xlabel('$X$')
            ax.set_ylabel('$p$ [bar]')
            ax.set_xlim(0, 1)
        _style_axes(axes)

        handles = [Line2D([0], [0], color=colors[U], ls=linestyles[U], marker=markers[U], lw=1.0,
                          markersize=5, markerfacecolor='none', markeredgewidth=0.8,
                          label=f'$U={U:g}$ m/s') for U in colors]
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_custom_eos_comparison(results):
    """Plot pressure and density profiles for different equations of state.

    Parameters
    ----------
    results : dict
        Label -> (X, p, rho) with normalized position X, pressure p in Pa and density rho.
        The first entry is drawn as line, the second as markers, the third dashed.
    """
    colors = ['#0072B2', '#D55E00', '#009E73']
    styles = [dict(ls='-'),
              dict(ls='none', marker='o', markersize=4, markerfacecolor='none', markevery=8),
              dict(ls='--')]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=False)

        for (label, (X, p, rho)), color, style in zip(results.items(), colors, styles):
            axes[0].plot(X, p / 1e5, color=color, lw=1.0, label=label, **style)
            axes[1].plot(X, rho, color=color, lw=1.0, **style)

        axes[0].set_ylabel('$p$ [bar]')
        axes[1].set_ylabel(r'$\rho$ [kg/m$^3$]')
        for ax in axes:
            ax.set_xlabel('$X$')
            ax.set_xlim(0, 1)
        _style_axes(axes)

        handles, names = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_custom_wall_stress(problem):
    """Plot pressure and wall shear stresses of a finished 1D run.

    Parameters
    ----------
    problem : Problem
        Finished 1D problem, pressure in Pa and tau_xz of the bottom and top wall in Pa.
    """
    X = problem.topo.xx[1:-1, 1] / problem.grid['Lx']
    p = problem.pressure.pressure[1:-1, 1]
    tau_bot = problem.wall_stress_xz.lower[4, 1:-1, 1]
    tau_top = problem.wall_stress_xz.upper[4, 1:-1, 1]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=False)

        axes[0].plot(X, p / 1e6, color='#0072B2', lw=1.0)
        axes[1].plot(X, tau_bot / 1e3, color='#D55E00', lw=1.0, label='bottom')
        axes[1].plot(X, tau_top / 1e3, color='#009E73', lw=1.0, ls='--', label='top')

        axes[0].set_ylabel('$p$ [MPa]')
        axes[1].set_ylabel(r'$\tau_{xz}$ [kPa]')
        for ax in axes:
            ax.set_xlabel('$X$')
            ax.set_xlim(0, 1)
        _style_axes(axes)

        handles, names = axes[1].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_reynolds_comparison_1d(results, ref_csv=None, ref_label=None,
                                ref_x_scale=1.0, ref_p_scale=1.0,
                                p_unit='bar', p_divisor=1e5):
    """Plot pressure and effective density of the Navier-Stokes and Reynolds
    solutions for a 1D (y-periodic) case.

    Parameters
    ----------
    results : dict
        Output of run_reynolds_comparison, keys 'ns' and 'reynolds'.
    ref_csv : str or Path, optional
        Path to a CSV with columns 'x' and 'y' for reference pressure data.
    ref_label : str, optional
        Legend label for the reference data.
    ref_x_scale : float
        Multiplier applied to the reference x-column to obtain the normalized position X.
    ref_p_scale : float
        Multiplier applied to the reference y-column to convert to Pa.
    p_unit, p_divisor : str, float
        Pressure unit used on the y-axis label and divisor applied to the pressure.
    """
    styles = {'ns': dict(color='#0072B2', ls='-', label='Navier-Stokes'),
              'reynolds': dict(color='#D55E00', ls='--', label='Reynolds')}

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=False)

        for name, res in results.items():
            axes[0].plot(res['X'], res['p'][:, 1:-1].mean(axis=1) / p_divisor, lw=1.5, **styles[name])
            axes[1].plot(res['X'], res['rho_eff'][:, 1:-1].mean(axis=1), lw=1.5, **styles[name])

        if ref_csv is not None:
            ref = pd.read_csv(ref_csv, skipinitialspace=True)
            axes[0].scatter(ref['x'].values * ref_x_scale, ref['y'].values * ref_p_scale / p_divisor,
                            facecolors='none', edgecolors='0.15', marker='o', s=22, linewidths=0.8,
                            zorder=5, label=ref_label)

        axes[0].set_ylabel(f'$p$ [{p_unit}]')
        axes[1].set_ylabel(r'$\rho_\mathrm{eff}$ [kg/m$^3$]')
        for ax in axes:
            ax.set_xlabel('$X$')
            ax.set_xlim(0, 1)
        _style_axes(axes)

        handles, names = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_reynolds_comparison_2d(results, p_unit='bar', p_divisor=1e5):
    """Plot the pressure fields of the Navier-Stokes and Reynolds solutions
    and their mid-sections (y = Ly/2) for a genuinely 2D case.

    Parameters
    ----------
    results : dict
        Output of run_reynolds_comparison, keys 'ns' and 'reynolds'.
    p_unit, p_divisor : str, float
        Pressure unit used on the labels and divisor applied to the pressure.
    """
    styles = {'ns': dict(color='#0072B2', ls='-', label='Navier-Stokes'),
              'reynolds': dict(color='#D55E00', ls='--', label='Reynolds')}
    p_fields = {name: res['p'][1:-1, 1:-1] / p_divisor for name, res in results.items()}
    Nx, Ny = p_fields['ns'].shape
    Lx, Ly = results['ns']['Lx'], results['ns']['Ly']
    x = (np.arange(Nx) + 0.5) / Nx
    y = (np.arange(Ny) + 0.5) / Ny * Ly * 1e3
    vmin = min(f.min() for f in p_fields.values())
    vmax = max(f.max() for f in p_fields.values())

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(9.75, 3.2), constrained_layout=True)

        for ax, name in zip(axes[:2], ['ns', 'reynolds']):
            im = ax.pcolormesh(x * Lx * 1e3, y, p_fields[name].T, cmap='viridis',
                               vmin=vmin, vmax=vmax, shading='auto')
            ax.text(0.05, 0.95, styles[name]['label'], transform=ax.transAxes, ha='left', va='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))
            ax.set_xlabel('$x$ [mm]')
            ax.set_ylabel('$y$ [mm]')
        fig.colorbar(im, ax=axes[:2], label=f'$p$ [{p_unit}]')

        for name, f in p_fields.items():
            axes[2].plot(x, f[:, Ny // 2], lw=1.5, **styles[name])
        axes[2].set_xlabel('$X$')
        axes[2].set_ylabel(f'$p$ [{p_unit}]')
        axes[2].set_xlim(0, 1)
        _style_axes(axes)

        handles, names = axes[2].get_legend_handles_labels()
        fig.legend(handles, names, loc='outside upper center', ncol=len(handles), frameon=False)
        plt.show()


def plot_overview_2d(problem, title):
    """Plot 2D pressure and fill-fraction fields for a full-2D cavitation result.

    Parameters
    ----------
    problem : Problem
        GaPFlow Problem instance after simulation.
    title : str
        Figure title.
    """
    from GaPFlow.models.pressure import eos_pressure

    Lx, Ly = problem.grid['Lx'], problem.grid['Ly']
    dx, dy = problem.grid['dx'], problem.grid['dy']
    Nx, Ny = problem.grid['Nx'], problem.grid['Ny']

    rho = problem.q[0][1:-1, 1:-1]
    p_arr = np.asarray(eos_pressure(rho, problem.prop))

    if problem.prop.get('EOS') == 'Bayada':
        rho_l = problem.prop['rho_l']
        rho_v = problem.prop['rho_v']
        theta = np.clip((rho_l - rho) / (rho_l - rho_v), 0, 1)
    else:
        theta = problem.q[3][1:-1, 1:-1]

    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='white',
                             constrained_layout=True)
    fig.suptitle(title, fontweight='bold')

    im0 = axes[0].pcolormesh(X * 1e3, Y * 1e3, p_arr / 1e5,
                             cmap='RdBu_r', shading='auto')
    plt.colorbar(im0, ax=axes[0], label='p [bar]')
    axes[0].set(xlabel='x [mm]', ylabel='y [mm]', title='Pressure')

    im1 = axes[1].pcolormesh(X * 1e3, Y * 1e3, theta,
                             cmap='RdBu_r', shading='auto', vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[1], label=r'$\theta$ [-]')
    axes[1].set(xlabel='x [mm]', ylabel='y [mm]', title=r'Fill fraction $\theta$')

    plt.show()


def plot_midsection_2d(problem, title, ref_csv, ref_label):
    """Plot the mid-section (y = Ly/2) pressure profile for a 2D cavitation result.

    Parameters
    ----------
    problem : Problem
        GaPFlow Problem instance after simulation.
    title : str
        Figure title.
    ref_csv : str or Path
        Path to a CSV with columns 'x' (normalised 0–1) and 'y' (pressure in MPa).
    ref_label : str
        Legend label for the reference data.
    """
    from GaPFlow.models.pressure import eos_pressure

    Lx = problem.grid['Lx']
    Nx, Ny = problem.grid['Nx'], problem.grid['Ny']
    dx = problem.grid['dx']

    rho = problem.q[0][1:-1, 1:-1]
    p_arr = np.asarray(eos_pressure(rho, problem.prop))

    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    mid = Ny // 2
    p_mid = p_arr[:, mid]

    ref = pd.read_csv(ref_csv, skipinitialspace=True)

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='white', constrained_layout=True)
    fig.suptitle(title, fontweight='bold')
    ax.plot(x * 1e3, p_mid / 1e6, color='0.3', lw=1.5, label='FEM')
    ax.scatter(ref['x'].values * Lx * 1e3, ref['y'].values,
               color='#1f77b4', s=14, zorder=5, label=ref_label)
    ax.legend(fontsize=8)
    ax.set(xlabel='x [mm]', ylabel='p [MPa]', title='Pressure — mid-section (y = Ly/2)')
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_lid_driven_cavity(problem, title=None):
    """Create comprehensive visualization of lid-driven cavity flow.

    Parameters
    ----------
    problem : Problem
        GaPFlow Problem instance after simulation
    title : str, optional
        Override for the figure title

    Returns
    -------
    fig : Figure
        Matplotlib figure object
    """
    # Extract fields (interior only, no ghost cells)
    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    jy = problem.q[2][1:-1, 1:-1]

    vx = jx / rho
    vy = jy / rho
    v_mag = np.sqrt(vx**2 + vy**2)

    # Pressure from EOS: P = P0 + (rho - rho0) / alpha
    P0 = problem.prop['P0']
    rho0 = problem.prop['rho0']
    alpha = problem.prop['alpha']
    pressure = P0 + (rho - rho0) / alpha

    # Grid
    Lx, Ly = problem.grid['Lx'], problem.grid['Ly']
    Nx, Ny = problem.grid['Nx'], problem.grid['Ny']
    dx, dy = problem.grid['dx'], problem.grid['dy']

    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Create figure
    fig = plt.figure(figsize=(12, 8), facecolor='white', constrained_layout=True)

    # 1. Pressure field with streamlines
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.pcolormesh(X, Y, pressure, cmap='RdBu_r', shading='auto')
    n_start = 5
    start_x = np.linspace(0.2, 0.8, n_start)
    start_y = np.linspace(0.2, 0.8, n_start)
    start_pts = np.array([[sx, sy] for sx in start_x for sy in start_y])
    ax1.streamplot(x, y, vx.T, vy.T, color='k', linewidth=0.6, density=0.6,
                   arrowsize=0.8, start_points=start_pts)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Pressure + Streamlines')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='P', shrink=0.7)

    # 2. Velocity magnitude with quiver
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.pcolormesh(X, Y, v_mag, cmap='viridis', shading='auto')
    skip = 3
    v_mag_sub = v_mag[::skip, ::skip]
    vx_norm = np.where(v_mag_sub > 0, vx[::skip, ::skip] / v_mag_sub, 0)
    vy_norm = np.where(v_mag_sub > 0, vy[::skip, ::skip] / v_mag_sub, 0)
    ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], vx_norm, vy_norm,
               color='white', alpha=0.9, scale=25, width=0.003)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Velocity Magnitude + Arrows')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='|v|', shrink=0.7)

    # 3. Density field
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.pcolormesh(X, Y, rho, cmap='plasma', shading='auto')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Density')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label=r'$\rho$', shrink=0.7)

    # 4. x-momentum (jx)
    ax4 = fig.add_subplot(2, 3, 4)
    vmax = np.abs(jx).max()
    im4 = ax4.pcolormesh(X, Y, jx, cmap='RdBu_r', shading='auto',
                         vmin=-vmax, vmax=vmax)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title(r'x-Momentum $j_x$')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label=r'$j_x$', shrink=0.7)

    # 5. y-momentum (jy)
    ax5 = fig.add_subplot(2, 3, 5)
    vmax = np.abs(jy).max()
    im5 = ax5.pcolormesh(X, Y, jy, cmap='RdBu_r', shading='auto',
                         vmin=-vmax, vmax=vmax)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title(r'y-Momentum $j_y$')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5, label=r'$j_y$', shrink=0.7)

    # 6. Vorticity
    ax6 = fig.add_subplot(2, 3, 6)
    dvx_dy = np.gradient(vx, dy, axis=1)
    dvy_dx = np.gradient(vy, dx, axis=0)
    vorticity = dvx_dy - dvy_dx
    vmax_vort = np.abs(vorticity).max()
    im6 = ax6.pcolormesh(X, Y, vorticity, cmap='RdBu_r', shading='auto',
                         vmin=-vmax_vort, vmax=vmax_vort)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title(r'Vorticity $\omega$')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6, label=r'$\omega$', shrink=0.7)

    # Title
    if title is None:
        title = 'Lid-Driven Cavity Flow'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    return fig


def _style_axes(axes):
    for ax in np.ravel(axes):
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(direction='in', top=True, right=True)


def _plot_channel_validation(problem, y, u_analytical):
    """Shared 2-panel figure for the in-plane shear cases (Poiseuille, Couette):
    velocity profile and 2D velocity field."""
    rho = problem.q[0][1:-1, 1:-1]
    jx = problem.q[1][1:-1, 1:-1]
    vx = (jx / rho).mean(axis=0)

    Lx, Nx = problem.grid['Lx'], problem.grid['Nx']
    dx = Lx / Nx

    l2_error = np.sqrt(((vx - u_analytical) ** 2).mean()) / u_analytical.max()
    print(f"L2 relative error between simulation and analytic: {l2_error:.2e}")

    x = np.linspace(dx / 2, Lx - dx / 2, Nx)
    X, Y = np.meshgrid(x, y, indexing='ij')
    me = max(1, len(vx) // 40)

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), constrained_layout=False)

        axes[0].plot(u_analytical, y, color='black', lw=1.5, label='Analytical')
        axes[0].plot(vx, y, color='#0072B2', ls='none', marker='o', ms=4, mew=0.8,
                     mfc='none', markevery=me, label='Simulated')
        axes[0].set(xlabel='$v_x(y)$ [m/s]', ylabel='$y$ [m]')

        im = axes[1].pcolormesh(X, Y, jx / rho, cmap='viridis', shading='auto')
        axes[1].set(xlabel='$x$ [m]', ylabel='$y$ [m]')
        fig.colorbar(im, ax=axes[1], label='$v_x(x,y)$ [m/s]')

        _style_axes(axes)
        handles, names = axes[0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()


def plot_poiseuille_validation(problem):
    """Compare Poiseuille simulation against the analytical parabolic profile.

    Analytical solution: u(y) = (rho * f_x) / (2*mu) * y * (Ly - y),
    with Dirichlet BCs enforced at cell faces via the ghost-cell mirror formula.

    Prints the L2 relative error and shows a 2-panel figure:
    velocity profile and 2D velocity field.
    """
    Ly, Ny = problem.grid['Ly'], problem.grid['Ny']
    dy = Ly / Ny
    mu = problem.prop['shear']
    f_x = problem.prop['force_x']
    rho_m = float(problem.q[0][1:-1, 1:-1].mean())

    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    u_analytical = (rho_m * f_x) / (2 * mu) * y * (Ly - y)
    _plot_channel_validation(problem, y, u_analytical)


def plot_couette_validation(problem):
    """Compare Couette simulation against the analytical linear profile.

    Analytical solution: u(y) = U_wall * y / Ly, where U_wall is inferred
    from the north Dirichlet BC value divided by mean density.

    Prints the L2 relative error and shows a 2-panel figure:
    velocity profile and 2D velocity field.
    """
    Ly, Ny = problem.grid['Ly'], problem.grid['Ny']
    dy = Ly / Ny
    rho_m = float(problem.q[0][1:-1, 1:-1].mean())
    U_wall = problem.grid['bc_yN_D_val'][1] / rho_m

    y = np.linspace(dy / 2, Ly - dy / 2, Ny)
    u_analytical = U_wall * y / Ly
    _plot_channel_validation(problem, y, u_analytical)


def plot_bernoulli_validation(problem,
                              x_ramp_start=0.03, x_ramp_end=0.07,
                              h_inlet=1.0e-3, h_throat=0.5e-3):
    """Compare Bernoulli venturi simulation against mass-conservation velocity
    and Bernoulli pressure predictions.

    Parameters
    ----------
    problem : Problem
        GaPFlow Problem instance after simulation.
    x_ramp_start, x_ramp_end : float
        x-coordinates [m] of the start and end of the venturi contraction.
    h_inlet, h_throat : float
        Gap heights [m] at inlet and throat.

    Prints velocity ratio and pressure drop errors, shows a 2×2 figure:
    gap height, velocity, pressure, and momentum flux.
    """
    from GaPFlow.models.pressure import eos_pressure

    rho_b = problem.q[0][1:-1, 1:-1]
    jx_b = problem.q[1][1:-1, 1:-1]
    h_b = problem.topo.h[1:-1, 1:-1]
    j_center = rho_b.shape[1] // 2

    dx = problem.grid['dx']
    x_b = np.arange(rho_b.shape[0]) * dx + dx / 2
    vx_b = jx_b[:, j_center] / rho_b[:, j_center]
    p_b = np.asarray(eos_pressure(rho_b[:, j_center], problem.prop))
    rho0 = problem.prop['rho0']

    v_inlet = float(vx_b[x_b < x_ramp_start].mean())

    # Theory: mass conservation + Bernoulli
    v_theory = v_inlet * h_inlet / h_b[:, j_center]
    p_ref = float(p_b[-1])
    v_ref = v_inlet * h_inlet / float(h_b[-1, j_center])
    p_theory = p_ref + 0.5 * rho0 * (v_ref ** 2 - v_theory ** 2)

    # Error metrics over flat inlet and throat regions
    x_throat = (x_ramp_start + x_ramp_end) / 2
    inlet_mask = x_b < x_ramp_start
    throat_mask = np.abs(x_b - x_throat) < (x_ramp_end - x_ramp_start) * 0.1

    v_in_sim = float(vx_b[inlet_mask].mean())
    v_throat_sim = float(vx_b[throat_mask].mean())
    p_in_sim = float(p_b[inlet_mask].mean())
    p_throat_sim = float(p_b[throat_mask].mean())

    v_throat_theory = v_inlet * h_inlet / h_throat
    dp_theory = 0.5 * rho0 * (v_throat_theory ** 2 - v_inlet ** 2)
    dp_sim = p_in_sim - p_throat_sim

    print(f"Velocity at throat: {v_throat_sim:.2f} m/s (theory: {v_throat_theory:.2f} m/s)")
    print(f"Velocity ratio: {v_throat_sim / v_in_sim:.3f} (theory: {v_throat_theory / v_inlet:.3f})")
    print(f"Pressure drop: {dp_sim:.1f} Pa (theory: {dp_theory:.1f} Pa)")
    print(f"Pressure drop error: {abs(dp_sim - dp_theory) / dp_theory * 100:.2f}%")

    x_mm = x_b * 1000
    x_rs, x_re = x_ramp_start * 1000, x_ramp_end * 1000

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), constrained_layout=False)

        axes[0, 0].plot(x_mm, h_b[:, j_center] * 1000, color='0.2', lw=1.5)
        axes[0, 0].set(xlabel='$x$ [mm]', ylabel='$h$ [mm]')

        axes[0, 1].plot(x_mm, vx_b, color='#D55E00', lw=1.5, label='Simulation')
        axes[0, 1].plot(x_mm, v_theory, color='#0072B2', ls='--', lw=1.5, label='Theory')
        axes[0, 1].set(xlabel='$x$ [mm]', ylabel='$v$ [m/s]')

        axes[1, 0].plot(x_mm, p_b / 1000, color='#D55E00', lw=1.5)
        axes[1, 0].plot(x_mm, p_theory / 1000, color='#0072B2', ls='--', lw=1.5)
        axes[1, 0].set(xlabel='$x$ [mm]', ylabel='$p$ [kPa]')

        axes[1, 1].plot(x_mm, jx_b[:, j_center], color='#D55E00', lw=1.5)
        axes[1, 1].set(xlabel='$x$ [mm]', ylabel='$j_x$ [kg/(m$^2$s)]')

        for ax in axes.ravel():
            ax.axvline(x_rs, color='0.6', ls=':', lw=0.8)
            ax.axvline(x_re, color='0.6', ls=':', lw=0.8)

        _style_axes(axes)
        handles, names = axes[0, 1].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False)
        plt.show()
