#
# Copyright 2025 Hannes Holey
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
import os
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import polars as pl

from GaPFlow.viz.utils import set_axes_labels, _get_centerline_coords, _plot_gp, mpl_style_context

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

        for file in file_list:
            _plot_single_frame_1d(ax, file, frame)

    elif dim == 2:
        fig, ax = plt.subplots(3, 3, figsize=(9, 9))

        for file in file_list:
            _plot_single_frame_2d(ax, file, frame)

    if show:
        plt.show()


@mpl_style_context
def plot_history(file_list,
                 gp_files_0=[],
                 gp_files_1=[],
                 show=True):
    """Plot the time evolution of scalar quantities, 
    e.g. kinetic energy, mass, GP variance etc.

    Parameters
    ----------
    file_list : list
        List of CSV files
    gp_files_0 : list, optional
        List of gp history files plotted in a separate column of the figure
        (the default is [], which means not a GP simulation)
    gp_files_1 : list, optional
        List of gp history files plotted in a separate column of the figure
        (the default is [], which means not a GP simulation)
    show : bool, optional
        Flag for plt.show() (the default is True)
    """

    ncol = 1
    if len(gp_files_0) > 0:
        ncol += 1

    if len(gp_files_1) > 0:
        ncol += 1

    fig, ax = plt.subplots(3, ncol, figsize=(ncol * 4, 9), sharex='col')

    for file in file_list:
        _plot_history(ax[:, 0] if ncol > 1 else ax,
                      file)

    col = 1
    for gp_file, k in gp_files_0:
        _plot_gp_history(ax[:, col], gp_file, k)

    col = 2 if len(gp_files_0) > 0 else 1
    for gp_file, k in gp_files_1:
        _plot_gp_history(ax[:, col], gp_file, k)

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


def _plot_height_1d(filename: str,
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

    data = netCDF4.Dataset(filename)
    topo = np.asarray(data.variables['topography'])

    nx, ny = topo.shape[-2:]
    centerline_index = max(1, (ny - 2) // 2)

    x = np.linspace(0, 1, nx - 2)
    h0 = topo[0, 0, 0, 1:-1, centerline_index]
    h = topo[-1, 0, 0, 1:-1, centerline_index]
    u = topo[-1, 3, 0, 1:-1:, centerline_index]

    columns = 1
    u_col = 0
    p_col = 0

    if show_pressure:
        fname_sol = os.path.join(os.path.dirname(filename), 'sol.nc')
        data_sol = netCDF4.Dataset(fname_sol)
        p = np.asarray(data_sol.variables['pressure'])[-1, 1:-1, centerline_index]
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


def _plot_height_1d_from_field(topo,
                               pressure,
                               show_defo: bool,
                               show_pressure: bool) -> None:
    """Plotting centerline height profile from topography.

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

    nx, ny = topo.shape[-2:]
    centerline_index = max(1, (ny - 2) // 2)

    x = np.linspace(0, 1, nx - 2)
    h = topo[0, 1:-1, centerline_index]
    u = topo[3, 1:-1:, centerline_index]
    h0 = h - u

    columns = 1
    u_col = 0
    p_col = 0

    if show_pressure:
        p = pressure[1:-1:, centerline_index]
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

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))

    h = topo[0, 0, 0, 1:-1, 1:-1].T
    dh_dx = topo[0, 1, 0, 1:-1, 1:-1].T
    dh_dy = topo[0, 2, 0, 1:-1, 1:-1].T

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


def _plot_single_frame_1d(ax, filename, frame=-1, disc=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_nc = np.asarray(data.variables['wall_stress_xz'])

    nt, nc, _, nx, ny = q_nc.shape
    x, _ = _get_centerline_coords(nx, ny, disc)

    color_q = 'C0'
    color_p = 'C1'
    color_t = 'C2'

    ax[0, 0].plot(x, q_nc[frame, 0, 0, 1:-1, ny // 2], color=color_q)
    ax[0, 1].plot(x, q_nc[frame, 1, 0, 1:-1, ny // 2], color=color_q)
    ax[0, 2].plot(x, q_nc[frame, 2, 0, 1:-1, ny // 2], color=color_q)

    if 'pressure_var' in data.variables.keys():
        pvar_nc = np.asarray(data.variables['pressure_var'])

        _plot_gp(ax[1, 0],
                 x, p_nc[frame, 1:-1, ny // 2],
                 pvar_nc[frame, 1:-1, ny // 2], tol=None,
                 color=color_p)

    else:
        ax[1, 0].plot(x, p_nc[frame, 1:-1, ny // 2], color=color_p)

    if 'wall_stress_xz_var' in data.variables.keys():
        tauvar_nc = np.asarray(data.variables['wall_stress_xz_var'])

        _plot_gp(ax[1, 1],
                 x, tau_nc[frame, 4, 0, 1:-1, ny // 2],
                 tauvar_nc[frame, 1:-1, ny // 2], tol=None,
                 color=color_t)

        _plot_gp(ax[1, 2],
                 x, tau_nc[frame, 10, 0, 1:-1, ny // 2],
                 tauvar_nc[frame, 1:-1, ny // 2], tol=None,
                 color=color_t)
    else:
        ax[1, 1].plot(x, tau_nc[frame, 4, 0, 1:-1, ny // 2], color=color_t)
        ax[1, 2].plot(x, tau_nc[frame, 10, 0, 1:-1, ny // 2], color=color_t)

    set_axes_labels(ax)


def _plot_single_frame_2d(ax, filename, frame=-1, disc=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_xz_nc = np.asarray(data.variables['wall_stress_xz'])
    tau_yz_nc = np.asarray(data.variables['wall_stress_yz'])

    nt, nc, _, nx, ny = q_nc.shape

    imshow_args = {'origin': 'lower', 'extent': (0., 1., 0., 1.)}

    ax[0, 0].imshow(q_nc[frame, 0, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[0, 1].imshow(q_nc[frame, 1, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[0, 2].imshow(q_nc[frame, 2, 0, 1:-1, 1:-1].T, **imshow_args)

    ax[1, 0].imshow(p_nc[frame, 1:-1, 1:-1].T, **imshow_args)
    ax[1, 1].imshow(tau_xz_nc[frame, 4, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[1, 2].imshow(tau_yz_nc[frame, 3, 0, 1:-1, 1:-1].T, **imshow_args)

    ax[2, 0].imshow(p_nc[frame, 1:-1, 1:-1].T, **imshow_args)
    ax[2, 1].imshow(tau_xz_nc[frame, 10, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[2, 2].imshow(tau_yz_nc[frame, 9, 0, 1:-1, 1:-1].T, **imshow_args)

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


def _plot_history(ax, filename='history.csv'):

    df = pl.read_csv(filename)

    ax[0].plot(df['time'], df['ekin'])
    ax[0].set_ylabel('Kinetic energy')

    ax[1].plot(df['time'], df['residual'])
    ax[1].set_yscale('log')
    ax[1].set_ylabel('Residual')

    ax[2].plot(df['time'], df['vsound'])
    ax[2].set_ylabel('Max. sound velocity')
    ax[2].set_ylim(0.,)

    ax[-1].set_xlabel('Time')


def _plot_gp_history(ax, filename='history.csv', index=0):

    df = pl.read_csv(filename)

    ax[0].plot(df['step'], df['database_size'], color=f'C{index}')
    ax[0].set_ylabel('DB size')

    ax[1].plot(df['step'], df['maximum_variance'], color=f'C{index}')
    ax[1].plot(df['step'], df['variance_tol'], '--', color=f'C{index}')
    ax[1].set_ylabel('Variance')

    ax[-1].set_xlabel('Step')
