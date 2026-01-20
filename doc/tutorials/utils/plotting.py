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
"""Plotting utilities for FEM tutorials."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
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


def plot_solver_comparison_rho_jx(results, title):
    """Plot density and momentum profiles comparing different solvers.

    Creates a side-by-side comparison of density (rho) and x-momentum (jx)
    profiles from multiple solver results.

    Parameters
    ----------
    results : dict
        Dictionary with solver names as keys and result dicts as values.
        Each result dict must contain 'rho' and 'jx' arrays.
        Expected keys: 'explicit', 'fem_1d', 'fem_2d'
    title : str
        Title for the figure

    Example
    -------
    >>> results = {
    ...     'explicit': {'rho': rho_exp, 'jx': jx_exp, 'time': t_exp},
    ...     'fem_1d': {'rho': rho_1d, 'jx': jx_1d, 'time': t_1d},
    ...     'fem_2d': {'rho': rho_2d, 'jx': jx_2d, 'time': t_2d},
    ... }
    >>> plot_solver_comparison_rho_jx(results, 'Inclined Slider')
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = {'explicit': 'C0', 'fem_1d': 'C1', 'fem_2d': 'C2'}
    labels = {'explicit': 'Explicit', 'fem_1d': 'FEM 1D', 'fem_2d': 'FEM 2D'}

    Nx = len(list(results.values())[0]['rho'])
    x = np.linspace(0, 1, Nx)

    for name, res in results.items():
        axes[0].plot(x, res['rho'], color=colors[name], label=labels[name], lw=1.5)
        axes[1].plot(x, res['jx'], color=colors[name], label=labels[name], lw=1.5)

    axes[0].set_xlabel('x / L')
    axes[0].set_ylabel(r'$\rho$ [kg/m³]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('x / L')
    axes[1].set_ylabel(r'$j_x$ [kg/(m²s)]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

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
    surf = ax.plot_surface(X, Y, T_field[0], facecolors=fcolors,
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
