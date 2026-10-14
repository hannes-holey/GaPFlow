#
# Copyright 2025-2026 Hannes Holey
#           2025 Christoph Huber
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
Create line and contour plots of the simulation results.

The methods within this module are either called from the command-line interface
or directly from a :class:`GaPFlow.Problem` instance. The former reads the output
from stored NetCDF files.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import netCDF4
import numpy as np
import polars as pl

from .utils import set_axes_labels, _plot_gp, mpl_style_context, autoscale_ylim

import numpy.typing as npt
NDArray = npt.NDArray[np.floating]


#####################
# All in one figure #
#####################

@mpl_style_context
def plot_frame(file_list, dim=1, frame=-1, show=True):
    """Plot a single frame of the solution from a file.

    Parameters
    ----------
    file_list : list
        List of NetCDF files (names)
    dim : int, optional
        Dimension (the default is 1, for 2D problems this plots the solution
        along the y-centerline)
    frame : int, optional
        Frame number (the default is -1, which is the last frame)
    show : bool, optional
        Flag for plt.show() (the default is True)
    """

    if dim == 1:
        fig, ax = plt.subplots(2, 3, figsize=(12, 6))

        # one color per run, labeled by the run directory
        for i, file in enumerate(file_list):
            label = os.path.basename(os.path.dirname(os.path.abspath(file)))
            _plot_single_frame_1d(ax, file, frame, color=f'C{i}', label=label)

        # autoscale_ylim fixes the limits of the first run, rescale to all runs
        for a in ax.flat:
            a.relim()
            a.autoscale(axis='y')
            autoscale_ylim(a)

        if len(file_list) > 1:
            ax[0, 0].legend()

    elif dim == 2:
        fig, ax = plt.subplots(3, 3, figsize=(9, 9))

        for file in file_list:
            _plot_single_frame_2d(ax, file, frame)

    if show:
        plt.show()


@mpl_style_context
def plot_history(file_list,
                 gp_files=None,
                 over_step=False,
                 show=True):
    """Plot the time evolution of scalar quantities recorded in history.csv and
    of the GP model parameters, one column per kind of data (physical, numerical,
    GP models). Panels and columns without data are skipped.

    By default, all columns share time as x-axis, such that runs of different solvers
    can be compared. When plotting a single run, the other quantity (step or time) is
    shown on a secondary x-axis.

    Parameters
    ----------
    file_list : list
        List of history.csv files
    gp_files : dict, optional
        Dictionary with list of CSV files for each GP model (zz, xz, yz). Adds a
        column with database size, ratio of maximum variance to tolerance, normalised
        observation noise, and objective (line styles distinguish the models).
    over_step : bool, optional
        Use the step number instead of time as x-axis (the default is False).
    show : bool, optional
        Flag for plt.show() (the default is True)
    """

    dfs = [pl.read_csv(file) for file in file_list]
    run_dirs = [os.path.dirname(os.path.abspath(file)) for file in file_list]
    labels = [os.path.basename(d) for d in run_dirs]
    xcol, xcol_other = ('step', 'time') if over_step else ('time', 'step')

    # skip panels without data, e.g. friction in y for 1D problems
    columns = {}
    for kind in _HISTORY_PANELS:
        panels = [(label, cols, log) for label, cols, log in _HISTORY_PANELS[kind]
                  if any(c in df.columns and (df[c] != 0.).any() for df in dfs for c in cols)]
        if panels:
            columns[kind] = panels

    gp_data = {key: [_read_gp_history(f, dfs, run_dirs, xcol) for f in flist]
               for key, flist in (gp_files or {}).items() if len(flist) > 0}
    gp_panels = _gp_panels(gp_data) if gp_data else []

    if not columns and not gp_data:
        raise ValueError(f"No history data found in {file_list}")

    ncol = len(columns) + (1 if gp_data else 0)
    nrow = max([len(panels) for panels in columns.values()] + [len(gp_panels)])
    fig, ax = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2. * nrow), sharex='col', squeeze=False)

    for col, (kind, panels) in enumerate(columns.items()):
        n = len(panels)
        ax[0, col].set_title(_HISTORY_TITLES[kind])
        for i, (df, label) in enumerate(zip(dfs, labels)):
            _plot_history(ax[:n, col], df, panels, xcol, index=i, label=label)

        for i, (axis, (_, cols, _)) in enumerate(zip(ax[:n, col], panels)):
            if len(cols) > 1:
                # line styles distinguish bottom/top, colors (see first panel) the runs
                handles = [Line2D([], [], color='k', ls=ls, label=_WALL_LABELS[c.rsplit('_', 1)[-1]])
                           for c, ls in zip(cols, _WALL_LINESTYLES)]
                axis.legend(handles=handles)
            elif i == 0 and col == 0 and len(file_list) > 1:
                axis.legend()

        _hide_unused(ax[:, col], n)

    if gp_data:
        col = len(columns)
        ax[0, col].set_title('GP models')
        _plot_gp_history(ax[:len(gp_panels), col], gp_data, gp_panels, xcol)
        _hide_unused(ax[:, col], len(gp_panels))

    # secondary x-axis with step or time (mapping between both is unique only for a single run)
    if len(file_list) == 1 and dfs[0].height > 1:
        x = dfs[0][xcol].to_numpy()
        x_other = dfs[0][xcol_other].to_numpy()
        for col in range(ncol):
            sec = ax[0, col].secondary_xaxis('top', functions=(lambda v: np.interp(v, x, x_other),
                                                               lambda v: np.interp(v, x_other, x)))
            sec.set_xlabel(_XLABELS[xcol_other])
            # the y offset text would overlap with the secondary tick labels
            if isinstance(ax[0, col].yaxis.get_major_formatter(), ScalarFormatter):
                ax[0, col].yaxis.get_major_formatter().set_useOffset(False)

    if show:
        plt.show()


####################
# Separate figures #
####################

@mpl_style_context
def plot_height(file_list, dim=1, show_defo=False, show_pressure=False):
    """Plot the height profile

    Parameters
    ----------
    file_list : list
        List of NetCDF files
    dim : int, optional
        Dimension (the default is 1, for 2D problems this plots the solution
        along the y-centerline)
    show_defo: bool
        Show the displacement in a separate subfigure and the initial (default is False)
        gap height for reference
    show_pressure: bool
        Show the pressure profile in a separate subfigure (default is False)
    """

    if dim == 1:
        for file in file_list:
            _plot_height_1d(file, show_defo, show_pressure)

    elif dim == 2:
        for file in file_list:
            _plot_height_2d(file)

    plt.show()


@mpl_style_context
def plot_frames(filename, every=1):
    """Plot the time evolution of the centerline solution.

    Parameters
    ----------
    filename : str
        NetCDF file name
    every : int, optional
        Plot the solution every this many times (the default is 1)
    """

    _plot_multiple_frames_1d(filename, every)

    plt.show()

###########
# Backend #
###########


def _plot_height_1d(fname_topo: str,
                    show_defo: bool,
                    show_pressure: bool) -> None:
    """Plotting centerline height profile from file.

    Parameters
    ----------
    filename : str
        Filename of the topography file.
    show_defo: bool
        Show the displacement in a separate subfigure and the initial
        gap height for reference
    show_pressure: bool
        Show the pressure profile in a separate subfigure.
    """

    data_topo = netCDF4.Dataset(fname_topo)
    fname_sol = os.path.join(os.path.dirname(fname_topo), 'sol.nc')
    data_sol = netCDF4.Dataset(fname_sol)

    topo = np.asarray(data_topo.variables['topography'])
    press = np.asarray(data_sol.variables['pressure'])

    return _plot_height_1d_from_field(topo, press, show_defo, show_pressure)


def _plot_height_1d_from_field(topo,
                               pressure,
                               show_defo: bool,
                               show_pressure: bool) -> None:
    """Plotting centerline height profile from fields.

    Parameters
    ----------
    topo : np.ndarray
        Topography array
    pressure : np.ndarray
        Pressure array
    show_defo: bool
        Show the displacement in a separate subfigure and the initial
        gap height for reference
    show_pressure: bool
        Show the pressure profile in a separate subfigure.
    """

    nx, ny = topo.shape[-2:]
    # For 1D problems (ny=1), use index 0; otherwise compute centerline
    centerline_index = 0 if ny == 1 else ny // 2

    x = np.linspace(0, 1, nx)

    if len(topo.shape) == 5:
        h0 = topo[0, 0, 0, :, centerline_index]
        h = topo[-1, 0, 0, :, centerline_index]
        u = topo[-1, 3, 0, :, centerline_index]
    elif len(topo.shape) == 3:
        h = topo[0, :, centerline_index]
        u = topo[3, :, centerline_index]
        h0 = h - u

    if len(pressure.shape) == 3:
        p = pressure[-1, :, centerline_index]
    elif len(pressure.shape) == 2:
        p = pressure[:, centerline_index]

    columns = 1
    u_col = 0
    p_col = 0

    if show_pressure:
        columns += 1
        p_col = 1

    if show_defo:
        columns += 1
        u_col = 1
        p_col += 1

    fig, ax = plt.subplots(1, columns, figsize=(4 * columns, 3), squeeze=False)
    ax = ax.ravel()

    # height profile
    ax[0].plot(x, h, color='C0', linestyle='-', label='Deformed shape')
    ax[0].fill_between(x, h, np.ones_like(x) * 1.1 * h.max(),
                       color='0.7', lw=0.)
    ax[0].fill_between(x, np.zeros_like(x), -np.ones_like(x) * 0.1 * h.max(),
                       color='0.7', lw=0.)
    ax[0].plot(x, h, color='C0')
    ax[0].plot(x, np.zeros_like(h), color='C0')

    ax[0].set_xlabel('$x/L_x$')
    ax[0].set_ylabel('Gap height $h$')

    # initial height and deformation
    if show_defo:
        ax[0].plot(x, h0, color='C0', linestyle='--', label='Initial shape')
        ax[0].legend(loc='upper center')

        ax[u_col].plot(x, u, color='C1')
        ax[u_col].set_xlabel('$x/L_x$')
        ax[u_col].set_ylabel('Deformation $u$')

    # pressure
    if show_pressure:
        ax[p_col].plot(x, p, color='C2')
        ax[p_col].set_xlabel('$x/L_x$')
        ax[p_col].set_ylabel('Pressure $p$')

    return fig, ax


def _plot_height_2d(filename):

    data = netCDF4.Dataset(filename)
    topo = np.asarray(data.variables['topography'])

    return _plot_height_2d_from_field(topo)


def _plot_height_2d_from_field(topo):

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    if len(topo.shape) == 5:
        h = topo[0, 0, 0, 1:-1, 1:-1].T
        dh_dx = topo[0, 1, 0, 1:-1, 1:-1].T
        dh_dy = topo[0, 2, 0, 1:-1, 1:-1].T
    elif len(topo.shape) == 3:
        h = topo[0, 1:-1, 1:-1].T
        dh_dx = topo[1, 1:-1, 1:-1].T
        dh_dy = topo[2, 1:-1, 1:-1].T

    imshow_args = {'origin': 'lower', 'extent': (0., 1., 0., 1.)}
    ax[0].imshow(h, **imshow_args)
    ax[1].imshow(dh_dx, **imshow_args)
    ax[2].imshow(dh_dy, **imshow_args)

    titles = [r'$h$', r'$\partial h/ \partial x$', r'$\partial h/ \partial y$']
    for (a, title) in zip(ax.flat, titles):
        a.set_xlabel(r'$x/L_x$')
        a.set_ylabel(r'$y/L_y$')
        a.set_title(title)

    return fig, ax


def _plot_single_frame_1d(ax, filename, frame=-1, disc=None, color=None, label=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])[frame]
    p_nc = np.asarray(data.variables['pressure'])[frame]
    shear_nc = np.asarray(data.variables['wall_stress_xz'])[frame]

    sol = q_nc[:, 0]
    press = p_nc
    lower = shear_nc[4, 0]
    upper = shear_nc[10, 0]

    if 'pressure_var' in data.variables.keys():
        var_press = np.asarray(data.variables['pressure_var'])[frame]
    else:
        var_press = None

    if 'wall_stress_xz_var' in data.variables.keys():
        var_shear = np.asarray(data.variables['wall_stress_xz_var'])[frame]
    else:
        var_shear = None

    _plot_sol_from_field_1d(sol, press, lower, upper, var_press, var_shear, ax=ax,
                            color=color, label=label)


@mpl_style_context
def _plot_sol_from_field_1d(q,
                            pressure,
                            lower,
                            upper,
                            var_press=None,
                            var_shear=None,
                            var_tol_press=None,
                            var_tol_shear=None,
                            energy=None,
                            ax=None,
                            color=None,
                            label=None,
                            ):

    col = 3 if energy is None else 4

    if ax is None:
        fig, ax = plt.subplots(2, col, figsize=(1.7 * col, 3), sharex=True)
    nx, ny = q.shape[-2:]

    # Handle both live data (with ghosts, ny >= 3) and NetCDF data (interior only, ny=1 for 1D)
    if ny == 1:
        # NetCDF data: interior only, no slicing needed
        ci = 0
        s = [slice(None), ci]
        x = np.linspace(0., 1., nx)
    else:
        # Live data: includes ghosts, need to slice them out
        ci = max(1, (ny - 2) // 2)
        s = [slice(1, -1), ci]
        x = np.linspace(0., 1., nx - 2)

    # a single color for all quantities, e.g. to distinguish runs, or one per quantity
    if color is None:
        color_q, color_p, color_t, color_e = 'C0', 'C1', 'C2', 'C3'
    else:
        color_q = color_p = color_t = color_e = color

    ax[0, 0].plot(x, q[(0, *s)], color=color_q, label=label)
    ax[0, 1].plot(x, q[(1, *s)], color=color_q)
    ax[0, 2].plot(x, q[(2, *s)], color=color_q)

    # pressure
    if var_press is None:
        ax[1, 0].plot(x, pressure[(*s,)], color=color_p)
    else:
        _plot_gp(ax[1, 0],
                 x,
                 pressure[(*s,)],
                 var_press[(*s,)],
                 tol=None,
                 color=color_p)

    if var_tol_press is not None:
        ax[1, 0].plot(x, pressure[(*s,)] + 1.96 * np.sqrt(var_tol_press), '--', color=color_p)
        ax[1, 0].plot(x, pressure[(*s,)] - 1.96 * np.sqrt(var_tol_press), '--', color=color_p)

    # shear stress
    if var_shear is None:
        ax[1, 1].plot(x, lower[(*s,)], color=color_t)
        ax[1, 2].plot(x, upper[(*s,)], color=color_t)
    else:
        _plot_gp(ax[1, 1],
                 x,
                 lower[(*s,)],
                 var_shear[(*s,)],
                 tol=None,
                 color=color_t)

        _plot_gp(ax[1, 2],
                 x,
                 upper[(*s,)],
                 var_shear[(*s,)],
                 tol=None,
                 color=color_t)

    if var_tol_shear is not None:
        ax[1, 1].plot(x, lower[(*s,)] + 1.96 * np.sqrt(var_tol_shear), '--', color=color_t)
        ax[1, 1].plot(x, lower[(*s,)] - 1.96 * np.sqrt(var_tol_shear), '--', color=color_t)
        ax[1, 2].plot(x, upper[(*s,)] + 1.96 * np.sqrt(var_tol_shear), '--', color=color_t)
        ax[1, 2].plot(x, upper[(*s,)] - 1.96 * np.sqrt(var_tol_shear), '--', color=color_t)

    if energy is not None:
        tot_energy, temperature = energy[0], energy[1]
        ax[0, 3].plot(x, tot_energy[(*s,)], color=color_e)
        ax[0, 3].set_ylabel(r'Energy $E$')
        ax[1, 3].plot(x, temperature[(*s,)], color=color_e)
        ax[1, 3].set_ylabel(r'Temperature $T$')

    for a in ax.flat:
        autoscale_ylim(a)
    set_axes_labels(ax)


def _plot_single_frame_2d(ax, filename, frame=-1, disc=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])[frame]
    p_nc = np.asarray(data.variables['pressure'])[frame]
    shear_xz_nc = np.asarray(data.variables['wall_stress_xz'])[frame]
    shear_yz_nc = np.asarray(data.variables['wall_stress_yz'])[frame]

    sol = q_nc[:, 0]
    press = p_nc
    lower_xz = shear_xz_nc[4, 0]
    upper_xz = shear_xz_nc[10, 0]
    lower_yz = shear_yz_nc[3, 0]
    upper_yz = shear_yz_nc[9, 0]

    _plot_sol_from_field_2d(sol,
                            press,
                            lower_xz,
                            upper_xz,
                            lower_yz,
                            upper_yz,
                            var_press=None,
                            var_shear_xz=None,
                            var_shear_yz=None,
                            ax=ax)


def _plot_sol_from_field_2d(q,
                            pressure,
                            lower_xz,
                            upper_xz,
                            lower_yz,
                            upper_yz,
                            var_press=None,
                            var_shear_xz=None,
                            var_shear_yz=None,
                            ax=None):

    if ax is None:
        fig, ax = plt.subplots(3, 3, figsize=(12, 10))

    s = [slice(1, -1), slice(1, -1)]

    imshow_args = {'origin': 'lower', 'extent': (0., 1., 0., 1.)}

    ax[0, 0].imshow(q[(0, *s,)].T, **imshow_args)
    ax[0, 1].imshow(q[(1, *s,)].T, **imshow_args)
    ax[0, 2].imshow(q[(2, *s,)].T, **imshow_args)

    ax[1, 0].imshow(pressure[(*s,)].T, **imshow_args)
    ax[1, 1].imshow(lower_xz[(*s,)].T, **imshow_args)
    ax[1, 2].imshow(upper_xz[(*s,)].T, **imshow_args)

    ax[2, 0].imshow(pressure[(*s,)].T, **imshow_args)
    ax[2, 1].imshow(lower_yz[(*s,)].T, **imshow_args)
    ax[2, 2].imshow(upper_yz[(*s,)].T, **imshow_args)

    titles = [r'$\rho$', r'$j_x$', r'$j_y$',
              r'$p$', r'$\tau_{xz}^\text{bot}$', r'$\tau_{xz}^\text{top}$',
              r'$p$', r'$\tau_{yz}^\text{bot}$', r'$\tau_{yz}^\text{top}$', ]

    for (a, title) in zip(ax.flat, titles):
        a.set_xlabel(r'$x/L_x$')
        a.set_ylabel(r'$y/L_y$')
        a.set_title(title)


def _plot_multiple_frames_1d(filename, every=1):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_nc = np.asarray(data.variables['wall_stress_xz'])

    nt, nc, _, nx, ny = q_nc.shape

    x = np.arange(nx - 2) / (nx - 2)
    x += x[1] / 2.

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

    for i in range(nt)[::every]:
        color_q = plt.cm.Blues(i / nt)
        ax[0, 0].plot(x, q_nc[i, 0, 0, 1:-1, ny // 2], color=color_q)
        ax[0, 1].plot(x, q_nc[i, 1, 0, 1:-1, ny // 2], color=color_q)
        ax[0, 2].plot(x, q_nc[i, 2, 0, 1:-1, ny // 2], color=color_q)

        color_p = plt.cm.Greens(i / nt)
        color_t = plt.cm.Oranges(i / nt)

        ax[1, 0].plot(x, p_nc[i, 1:-1, ny // 2], color=color_p)
        ax[1, 1].plot(x, tau_nc[i, 4, 0, 1:-1, ny // 2], color=color_t)
        ax[1, 2].plot(x, tau_nc[i, 10, 0, 1:-1, ny // 2], color=color_t)

    set_axes_labels(ax)

    return fig, ax


_HISTORY_TITLES = {'phys': 'Physical', 'num': 'Numerical'}
_XLABELS = {'time': 'Time', 'step': 'Step'}
_WALL_LABELS = {'bot': 'bottom', 'top': 'top'}
_WALL_LINESTYLES = ['-', '--']

# (label, columns, log scale) of the history panels for each kind of history.csv data.
# Not all columns are recorded by both solvers; panels without data are skipped.
_HISTORY_PANELS = {
    'phys': [
        ('Mass', ['mass'], False),
        ('Kinetic energy', ['ekin'], False),
        ('Load', ['load'], False),
        ('Friction force x', ['friction_x_bot', 'friction_x_top'], False),
        ('Friction force y', ['friction_y_bot', 'friction_y_top'], False),
        ('Max. sound velocity', ['vsound'], False),
    ],
    'num': [
        ('Residual', ['residual'], True),
        ('Time step', ['dt'], True),
        ('Newton residual', ['R_newton'], True),
        ('Newton iterations', ['inner_iterations'], False),
    ],
}


def _plot_history(ax, df, panels, xcol='time', index=0, label=None):

    for axis, (ylabel, cols, log) in zip(ax, panels):
        for col, ls in zip(cols, _WALL_LINESTYLES):
            if col in df.columns:
                # bot/top variants of a quantity share a panel, distinguished by line style
                axis.plot(df[xcol], df[col], ls, color=f'C{index}', label=label if len(cols) == 1 else None)
        axis.set_ylabel(ylabel)
        if log:
            axis.set_yscale('log')

    ax[-1].set_xlabel(_XLABELS[xcol])


def _hide_unused(ax, n):
    """Hide the axes below the first n of a column and restore the x tick labels
    of the last visible one (removed by sharex)."""
    for axis in ax[n:]:
        axis.set_visible(False)
    ax[n - 1].xaxis.set_tick_params(labelbottom=True)


# line styles distinguish the GP models, colors the runs
_GP_LINESTYLES = {'zz': '-', 'xz': '--', 'yz': ':'}


def _read_gp_history(filename, dfs, run_dirs, xcol='time'):
    """Read a GP history file and return (run index, data, x-values)."""

    df = pl.read_csv(filename)

    run_dir = os.path.dirname(os.path.abspath(filename))
    index = run_dirs.index(run_dir) if run_dir in run_dirs else 0
    df_hist = dfs[index] if run_dir in run_dirs else None

    if xcol == 'step':
        x = df['step'].to_numpy()
    elif 'time' in df.columns:
        x = df['time'].to_numpy()
    elif df_hist is not None:
        # older output without time column: interpolate from history.csv of the same run
        x = np.interp(df['step'].to_numpy(), df_hist['step'].to_numpy(), df_hist['time'].to_numpy())
    else:
        raise ValueError(f"Cannot determine time for {filename}")

    return index, df, x


def _gp_panels(gp_data):
    """Panels of the GP column that have data."""

    panels = ['db', 'var', 'noise']
    if any('objective' in df.columns for runs in gp_data.values() for _, df, _ in runs):
        panels.append('obj')

    return panels


def _plot_gp_history(ax, gp_data, panels, xcol='time'):
    """Plot all GP models into a single column of axes.

    Parameters
    ----------
    ax : array of matplotlib.axes.Axes
        Axes of the column, one per panel.
    gp_data : dict
        For each GP model (zz, xz, yz), a list of (run index, data, x-values).
    panels : list
        Panels to plot, see :func:`_gp_panels`.
    xcol : str, optional
        Quantity on the x-axis, 'time' or 'step' (the default is 'time').
    """

    axes = dict(zip(panels, ax))

    # the database is shared between all GP models, plot its size once per run
    db_runs = set()
    for runs in gp_data.values():
        for index, df, x in runs:
            if index not in db_runs:
                db_runs.add(index)
                axes['db'].plot(x, df['database_size'], color=f'C{index}')
    axes['db'].set_ylabel('DB size')

    for key, runs in gp_data.items():
        ls = _GP_LINESTYLES[key]
        for index, df, x in runs:
            # active learning is triggered when the ratio exceeds one
            axes['var'].plot(x, df['maximum_variance'] / df['variance_tol'], ls, color=f'C{index}')
            axes['noise'].plot(x, df['obs_stddev'], ls, color=f'C{index}')
            if 'obj' in axes and 'objective' in df.columns:
                axes['obj'].plot(x, df['objective'], ls, color=f'C{index}')

    axes['var'].axhline(1., color='k', lw=0.8)
    axes['var'].set_yscale('log')
    axes['var'].set_ylabel('Max. variance / tol.')
    axes['var'].legend(handles=[Line2D([], [], color='k', ls=_GP_LINESTYLES[key], label=key)
                                for key in gp_data])

    axes['noise'].set_ylabel('Noise (normalised)')
    if 'obj' in axes:
        axes['obj'].set_ylabel('Objective')

    ax[len(panels) - 1].set_xlabel(_XLABELS[xcol])
