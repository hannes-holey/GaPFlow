import os
import time
import numpy as np


from GaPFlow.topography import create_midpoint_grid


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
