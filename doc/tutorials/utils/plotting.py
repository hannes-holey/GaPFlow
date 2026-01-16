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
from IPython.display import HTML


def animate_comparison(x_vec, t_vec, T_numeric, T_analytic, L):
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
    plt.close(fig)
    return HTML(ani.to_jshtml())


def animate_advection(x_vec, t_vec, T_numeric, L, U_avg, T_min=None, T_max=None):
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
        dist_text.set_text(f'Δx = {distance_wrapped:.4f} m ({distance/L:.2f} periods)')
        return line_T, time_text, dist_text

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(t_vec),
                                  interval=50, blit=True, repeat=True)
    plt.close(fig)
    return HTML(ani.to_jshtml())
