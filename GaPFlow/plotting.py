import os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import netCDF4
import numpy as np
import pandas as pd

from GaPFlow.gap import create_midpoint_grid


def get_pipeline(path='.', silent=False, mode='select', name='sol.nc'):

    folders = []

    for root, dirs, files in os.walk(path, topdown=False):
        if any([file.endswith(name) for file in files]):
            folders.append(root)

    folders = sorted(folders)

    for i, folder in enumerate(folders):
        date = time.strftime('%d/%m/%Y %H:%M', time.localtime(os.path.getmtime(folder)))
        if not silent:
            print(f"{i:3d}: {folder:<50} {date}")

    if mode == "select":
        inp = input("Enter keys (space separated or range [start]-[end] or combination of both): ")

        if inp.split('-') == 2:
            s, e = inp.split('-')
            mask = np.arange(int(s), int(e) + 1).tolist()
        else:
            mask = [int(i) for i in inp.split()]

        files = [os.path.join(folders[i], name) for i in mask]

    elif mode == "all":
        files = [os.path.join(folder, name) for folder in folders]

    elif mode == "single":
        inp = input("Enter key: ")
        files = os.path.join(folders[int(inp)], name)

    return files


def _get_centerline_coords(nx, ny, disc=None):
    if disc is not None:
        xx, yy = create_midpoint_grid(disc)
        x = xx[1:-1, ny // 2]
        y = yy[nx // 2, 1:-1]
    else:
        x = np.arange(nx - 2) / (nx - 2)
        x += x[0] / 2.
        y = np.arange(ny - 2) / (ny - 2)
        y += y[0] / 2.

    return x, y


def set_axes_labels(ax):

    ax[1, 0].set_xlabel(r"$x$")
    ax[1, 1].set_xlabel(r"$x$")
    ax[1, 2].set_xlabel(r"$x$")

    ax[0, 0].set_ylabel(r"Density $\rho$")
    ax[0, 1].set_ylabel(r"Mass flux $j_x$")
    ax[0, 2].set_ylabel(r"Mass flux $j_y$")

    ax[1, 0].set_ylabel(r"Pressure $p$")
    ax[1, 1].set_ylabel(r"Shear stress $\tau_{xz}^\mathsf{bot}$")
    ax[1, 2].set_ylabel(r"Shear stress $\tau_{xz}^\mathsf{top}$")


def set_axes_limits(ax, q, tol=None):

    q_min = q.min()
    q_max = q.max()

    if np.isclose(q_min, q_max):
        if np.isclose(q_min, 0.):
            q_min = -1.
            q_max = 1.
        else:
            q_min = 0.95 * q_min
            q_max = 1.05 * q_max

    if tol is not None:
        q_min -= tol
        q_max += tol

    ax.set_ylim(q_min, q_max)


def plot_evolution(filename, every=1, savefig=False, show=True, disc=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_nc = np.asarray(data.variables['wall_stress'])

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


def _plot_gp(ax, x, mean, var, tol=None, color='C0'):

    ax.fill_between(x,
                    mean + 1.96 * np.sqrt(var),
                    mean - 1.96 * np.sqrt(var),
                    color=color,
                    lw=0.,
                    alpha=0.3)

    ax.plot(x, mean, color=color)

    if tol is not None:
        ax.plot(x, mean + 1.96 * tol, '--', color=color)
        ax.plot(x, mean - 1.96 * tol, '--', color=color)


def _plot_single_frame(ax, filename, frame=-1, disc=None):

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_nc = np.asarray(data.variables['wall_stress'])

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

    if 'wall_stress_var' in data.variables.keys():
        tauvar_nc = np.asarray(data.variables['wall_stress_var'])

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


def animate(filename, seconds=10, save=False, show=True, disc=None, tol_p=None, tol_s=None):
    if tol_p is not None:
        tol_p = np.sqrt(tol_p[0])
        tol_p_max = tol_p.max()
    else:
        tol_p_max = 0.

    if tol_s is not None:
        tol_t = np.sqrt(tol_s[0])
        tol_t_max = tol_t.max()
    else:
        tol_t_max = 0.

    def update_lines(i, q, p, vp, tau, vt):

        ax[0, 0].get_lines()[0].set_ydata(q[i, 0, 0, 1:-1, ny // 2])
        ax[0, 1].get_lines()[0].set_ydata(q[i, 1, 0, 1:-1, ny // 2])
        ax[0, 2].get_lines()[0].set_ydata(q[i, 2, 0, 1:-1, ny // 2])

        ax[1, 0].cla()
        ax[1, 1].cla()
        ax[1, 2].cla()

        # Pressure
        _plot_gp(ax[1, 0], x, p[i, 1:-1, ny // 2], vp[i, 1:-1, ny // 2], tol=tol_p, color=color_p)

        # Shear stress
        _plot_gp(ax[1, 1], x, tau[i, 4, 0, 1:-1, ny // 2], vt[i, 1:-1, ny // 2], tol=tol_t, color=color_t)
        _plot_gp(ax[1, 2], x, tau[i, 10, 0, 1:-1, ny // 2], vt[i, 1:-1, ny // 2], tol=tol_t, color=color_t)

        set_axes_labels(ax)
        set_axes_limits(ax[1, 0], p[1:, 1:-1, ny // 2], tol=1.96 * tol_p_max)
        set_axes_limits(ax[1, 1], tau[1:, 4, 0, 1:-1, ny // 2], tol=1.96 * tol_t_max)
        set_axes_limits(ax[1, 2], tau[1:, 10, 0, 1:-1, ny // 2], tol=1.96 * tol_t_max)

    # adaptive_ylim(ax)

    data = netCDF4.Dataset(filename)
    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    pvar_nc = np.asarray(data.variables['pressure_var'])
    tau_nc = np.asarray(data.variables['wall_stress'])
    tauvar_nc = np.asarray(data.variables['wall_stress_var'])

    nt, nc, _, nx, ny = q_nc.shape

    if disc is not None:
        xx, yy = create_midpoint_grid(disc)
        x = xx[1:-1, ny // 2]
    else:
        x = np.arange(nx - 2) / (nx - 2)
        x += x[1] / 2.

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    color_q = 'C0'
    color_p = 'C1'
    color_t = 'C2'

    ax[0, 0].plot(x, q_nc[0, 0, 0, 1:-1, ny // 2], color=color_q)
    ax[0, 1].plot(x, q_nc[0, 1, 0, 1:-1, ny // 2], color=color_q)
    ax[0, 2].plot(x, q_nc[0, 2, 0, 1:-1, ny // 2], color=color_q)

    update_lines(0, q_nc, p_nc, pvar_nc, tau_nc, tauvar_nc)

    set_axes_labels(ax)

    set_axes_limits(ax[0, 0], q_nc[:, 0, 0, 1:-1, ny // 2])
    set_axes_limits(ax[0, 1], q_nc[:, 1, 0, 1:-1, ny // 2])
    set_axes_limits(ax[0, 2], q_nc[:, 2, 0, 1:-1, ny // 2])

    ani = animation.FuncAnimation(fig,
                                  update_lines,
                                  frames=nt,
                                  fargs=(q_nc, p_nc, pvar_nc, tau_nc, tauvar_nc),
                                  interval=100,
                                  repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(nt / seconds),
                        codec='libx264',
                        extra_args=['-pix_fmt', 'yuv420p', '-crf', '25'])

        outfile = os.path.join(os.path.dirname(filename),
                               os.path.dirname(filename).split(os.sep)[-1]) + ".mp4"

        ani.save(outfile, writer=writer, dpi=600)

    plt.show()


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


def adaptive_ylim(ax):

    offset = lambda x, y: 0.05 * (x - y) if (x - y) != 0 else 1.

    try:
        axes = ax.flat
    except AttributeError:
        axes = [ax, ]

    for a in axes:
        y_min = np.amin(a.lines[0].get_ydata())
        y_max = np.amax(a.lines[0].get_ydata())

        a.set_ylim(y_min - offset(y_max, y_min), y_max + offset(y_max, y_min))

    return ax
