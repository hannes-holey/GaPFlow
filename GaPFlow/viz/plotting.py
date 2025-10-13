import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

from GaPFlow.gap import create_midpoint_grid
from GaPFlow.viz.utils import set_axes_labels, _get_centerline_coords, _plot_gp


def plot_evolution(filename, every=1, savefig=False, show=True, disc=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_nc = np.asarray(data.variables['wall_stress_xz'])

    nt, nc, _, nx, ny = q_nc.shape

    if disc is not None:
        xx, yy = create_midpoint_grid(disc)
        x = xx[1:-1, ny // 2]
    else:
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

    if savefig:
        fig.savefig(filename + '.pdf')

    if show:
        plt.show()

    return fig, ax


def plot_height(filename, disc=None):

    fig, ax = plt.subplots(1)

    data = netCDF4.Dataset(filename)
    h_nc = np.asarray(data.variables['gap'])
    _, _, _, nx, ny = h_nc.shape

    x, _ = _get_centerline_coords(nx, ny, disc)

    gap_height_1d = h_nc[0, 0, 0, 1:-1, ny // 2]

    ax.fill_between(x,
                    gap_height_1d,
                    np.ones_like(x) * 1.1 * gap_height_1d.max(),
                    color='0.7', lw=0.)
    ax.fill_between(x,
                    np.zeros_like(x),
                    -np.ones_like(x) * 0.1 * gap_height_1d.max(),
                    color='0.7', lw=0.)

    ax.plot(x, np.zeros_like(gap_height_1d), color='C0')

    ax.plot(x, gap_height_1d, color='C0')
    ax.plot(x, np.zeros_like(gap_height_1d), color='C0')

    ax.set_ylabel('Gap height $h$')
    ax.set_xlabel('$x/L_x$' if disc is None else '$x$')

    plt.show()


def plot_single_frame(file_list, frame=-1, savefig=False, show=True, disc=None):

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    for file in file_list:
        _plot_single_frame(ax, file, frame, disc)

    if savefig:
        fig.savefig('out_nc_last.pdf')

    if show:
        plt.show()


def _plot_single_frame(ax, filename, frame=-1, disc=None):

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


def plot_history(file_list,
                 gp_files_0=[],
                 gp_files_1=[],
                 show=True,
                 savefig=False):

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

    if savefig:
        fig.savefig('out_csv.pdf')

    if show:
        plt.show()


def _plot_history(ax, filename='history.csv'):

    df = pd.read_csv(filename)

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

    df = pd.read_csv(filename)

    ax[0].plot(df['step'], df['database_size'], color=f'C{index}')
    ax[0].set_ylabel('DB size')

    ax[1].plot(df['step'], df['maximum_variance'], color=f'C{index}')
    ax[1].plot(df['step'], df['variance_tol'], '--', color=f'C{index}')
    ax[1].set_ylabel('Variance')

    ax[-1].set_xlabel('Step')
