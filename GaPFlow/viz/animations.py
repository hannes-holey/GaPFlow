#
# Copyright 2025 Hannes Holey
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
from cmath import tau
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, Video
import netCDF4

from GaPFlow.topography import create_midpoint_grid
from GaPFlow.viz.utils import set_axes_labels, set_axes_limits, _plot_gp, mpl_style_context


def animate(filename, seconds=10, save=False, show=True, disc=None):

    def update_lines(i, q, p, tau):

        ax[0, 0].get_lines()[0].set_ydata(q[i, 0, 0, 1:-1, ny // 2])
        ax[0, 1].get_lines()[0].set_ydata(q[i, 1, 0, 1:-1, ny // 2])
        ax[0, 2].get_lines()[0].set_ydata(q[i, 2, 0, 1:-1, ny // 2])

        # Pressure & shear stress
        ax[1, 0].get_lines()[0].set_ydata(p[i, 1:-1, ny // 2])
        ax[1, 1].get_lines()[0].set_ydata(tau[i, 4, 0, 1:-1, ny // 2])
        ax[1, 2].get_lines()[0].set_ydata(tau[i, 10, 0, 1:-1, ny // 2])

        set_axes_labels(ax)
        set_axes_limits(ax[1, 0], p[1:, 1:-1, ny // 2])
        set_axes_limits(ax[1, 1], tau[1:, 4, 0, 1:-1, ny // 2])
        set_axes_limits(ax[1, 2], tau[1:, 10, 0, 1:-1, ny // 2])

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

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))

    color_q = 'C0'
    color_p = 'C1'
    color_t = 'C2'

    ax[0, 0].plot(x, q_nc[0, 0, 0, 1:-1, ny // 2], color=color_q)
    ax[0, 1].plot(x, q_nc[0, 1, 0, 1:-1, ny // 2], color=color_q)
    ax[0, 2].plot(x, q_nc[0, 2, 0, 1:-1, ny // 2], color=color_q)

    ax[1, 0].plot(x, p_nc[0, 1:-1, ny // 2], color=color_p)
    ax[1, 1].plot(x, tau_nc[0, 4, 0, 1:-1, ny // 2], color=color_t)
    ax[1, 2].plot(x, tau_nc[0, 10, 0, 1:-1, ny // 2], color=color_t)

    set_axes_labels(ax)

    set_axes_limits(ax[0, 0], q_nc[:, 0, 0, 1:-1, ny // 2])
    set_axes_limits(ax[0, 1], q_nc[:, 1, 0, 1:-1, ny // 2])
    set_axes_limits(ax[0, 2], q_nc[:, 2, 0, 1:-1, ny // 2])

    ani = animation.FuncAnimation(fig,
                                  update_lines,
                                  frames=nt,
                                  fargs=(q_nc, p_nc, tau_nc),
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


def animate_gp(filename, seconds=10, save=False, show=True, disc=None, tol_p=None, tol_s=None):
    if tol_p is not None:
        tol_p = np.sqrt(tol_p)
        tol_p_max = tol_p.max()
    else:
        tol_p_max = 0.

    if tol_s is not None:
        tol_t = np.sqrt(tol_s)
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
        _plot_gp(ax[1, 0], x, p[i, 1:-1, ny // 2], vp[i, 1:-1, ny // 2], tol=tol_p[i], color=color_p)

        # Shear stress
        _plot_gp(ax[1, 1], x, tau[i, 4, 0, 1:-1, ny // 2], vt[i, 1:-1, ny // 2], tol=tol_t[i], color=color_t)
        _plot_gp(ax[1, 2], x, tau[i, 10, 0, 1:-1, ny // 2], vt[i, 1:-1, ny // 2], tol=tol_t[i], color=color_t)

        set_axes_labels(ax)
        set_axes_limits(ax[1, 0], p[1:, 1:-1, ny // 2], tol=1.96 * tol_p_max)
        set_axes_limits(ax[1, 1], tau[1:, 4, 0, 1:-1, ny // 2], tol=1.96 * tol_t_max)
        set_axes_limits(ax[1, 2], tau[1:, 10, 0, 1:-1, ny // 2], tol=1.96 * tol_t_max)

    # adaptive_ylim(ax)

    data = netCDF4.Dataset(filename)

    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    pvar_nc = np.asarray(data.variables['pressure_var'])
    tau_nc = np.asarray(data.variables['wall_stress_xz'])
    tauvar_nc = np.asarray(data.variables['wall_stress_xz_var'])

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


def animate2d(filename, seconds=10, save=False, show=True, disc=None):

    def update_fields(i, q, p, tau):

        # firstr row (q)
        im, = ax[0, 0].get_images()
        im.set_array(q[i, 0, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=q[:, 0].min(), vmax=q[:, 0].max())

        im, = ax[0, 1].get_images()
        im.set_array(q[i, 1, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=q[:, 1].min(), vmax=q[:, 1].max())

        im, = ax[0, 2].get_images()
        im.set_array(q[i, 2, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=q[:, 2].min(), vmax=q[:, 2].max())

        # second row (p, tau_xz)
        im, = ax[1, 0].get_images()
        im.set_array(p[i, 1:-1, 1:-1].T)
        im.set_clim(vmin=p.min(), vmax=p.max())

        im, = ax[1, 1].get_images()
        im.set_array(tau[i, 4, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=tau[:, 4].min(), vmax=tau[:, 4].max())

        im, = ax[1, 2].get_images()
        im.set_array(tau[i, 10, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=tau[:, 10].min(), vmax=tau[:, 10].max())

        # third row (p, tau_yz)
        im, = ax[2, 0].get_images()
        im.set_array(p[i, 1:-1, 1:-1].T)
        im.set_clim(vmin=p.min(), vmax=p.max())

        im, = ax[2, 1].get_images()
        im.set_array(tau[i, 3, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=tau[:, 3].min(), vmax=tau[:, 3].max())

        im, = ax[2, 2].get_images()
        im.set_array(tau[i, 9, 0, 1:-1, 1:-1].T)
        im.set_clim(vmin=tau[:, 9].min(), vmax=tau[:, 9].max())

    data = netCDF4.Dataset(filename)
    q_nc = np.asarray(data.variables['solution'])
    p_nc = np.asarray(data.variables['pressure'])
    tau_xz_nc = np.asarray(data.variables['wall_stress_xz'])
    tau_yz_nc = np.asarray(data.variables['wall_stress_yz'])
    tau_nc = tau_xz_nc + tau_yz_nc

    nt, nc, _, nx, ny = q_nc.shape

    fig, ax = plt.subplots(3, 3, figsize=(9, 9))

    imshow_args = {'origin': 'lower', 'extent': (0., 1., 0., 1.)}

    ax[0, 0].imshow(q_nc[0, 0, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[0, 1].imshow(q_nc[0, 1, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[0, 2].imshow(q_nc[0, 2, 0, 1:-1, 1:-1].T, **imshow_args)

    ax[1, 0].imshow(p_nc[0, 1:-1, 1:-1].T, **imshow_args)
    ax[1, 1].imshow(tau_nc[0, 4, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[1, 2].imshow(tau_nc[0, 10, 0, 1:-1, 1:-1].T, **imshow_args)

    ax[2, 0].imshow(p_nc[0, 1:-1, 1:-1].T, **imshow_args)
    ax[2, 1].imshow(tau_nc[0, 3, 0, 1:-1, 1:-1].T, **imshow_args)
    ax[2, 2].imshow(tau_nc[0, 9, 0, 1:-1, 1:-1].T, **imshow_args)

    titles = [r'$\rho$', r'$j_x$', r'$j_y$',
              r'$p$', r'$\tau_{xz}^\text{bot}$', r'$\tau_{xz}^\text{top}$',
              r'$p$', r'$\tau_{yz}^\text{bot}$', r'$\tau_{yz}^\text{top}$', ]

    for (a, title) in zip(ax.flat, titles):
        a.set_xlabel(r'$x/L_x$')
        a.set_ylabel(r'$y/L_y$')
        a.set_title(title)

    ani = animation.FuncAnimation(fig,
                                  update_fields,
                                  frames=nt,
                                  fargs=(q_nc, p_nc, tau_nc),
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

@mpl_style_context
def animate_1d(filename_sol: str,
                     filename_topo: str,
                     seconds: float = 10.,
                     save: bool = False,
                     show_notebook: bool = False
                     ) -> None:
    """Animation of solution process for 1D simulations.
    - Option 1: Default. Showing in a matplotlib window.
    - Option 2: Showing in Jupyter notebook (show_notebook=True).
    - Option 3: Saving as mp4 file (save=True).

    Parameters
    ----------
    filename_sol : str
        Relative path to the solution NetCDF file.
    filename_topo : str
        Relative path to the topography NetCDF file.
    seconds : float, optional
        Duration of the animation in seconds, by default 10.
    save : bool, optional
        Whether to save the animation as an mp4 file, by default False.
    show_notebook : bool, optional
        Whether to show the animation in a Jupyter notebook, by default False.
    """
    assert not (save and show_notebook), "Cannot both save and show in notebook."

    data_sol = netCDF4.Dataset(filename_sol)
    q_nc = np.asarray(data_sol.variables['solution'])
    p_nc = np.asarray(data_sol.variables['pressure'])
    tau_nc = np.asarray(data_sol.variables['wall_stress_xz'])

    data_topo = netCDF4.Dataset(filename_topo)
    topo_nc = np.asarray(data_topo.variables['topography'])

    nt, nc, _, nx, ny = q_nc.shape
    x = np.linspace(0, 1, nx-2)

    bDef = True if topo_nc.shape[0] > 1 else False
    fig, ax = plt.subplots(2, 3 + int(bDef), figsize=(10, 4))
    title = fig.suptitle("Simulation Animation 1D", fontsize=12)

    color_q, color_p, color_t, color_h = 'C0', 'C1', 'C2', 'C3'

    (line_rho,) = ax[0, 0].plot([], [], color=color_q)
    (line_jx,) = ax[0, 1].plot([], [], color=color_q)
    (line_jy,) = ax[0, 2].plot([], [], color=color_q)
    (line_p,) = ax[1, 0].plot([], [], color=color_p)
    (line_tauxz_bot,) = ax[1, 1].plot([], [], color=color_t)
    (line_tauxz_top,) = ax[1, 2].plot([], [], color=color_t)
    if bDef:
        (line_h,) = ax[0, 3].plot([], [], color=color_h)
        (line_def,) = ax[1, 3].plot([], [], color=color_h)
        ax[0, 3].plot(x, topo_nc[0, 0, 0, 1:-1, ny // 2], color=color_h,
                      linestyle='--', label='Initial')
        ax[0, 3].legend(loc='upper center')

    set_axes_limits(ax[0, 0], q_nc[:, 0, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
    set_axes_limits(ax[0, 1], q_nc[:, 1, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
    set_axes_limits(ax[0, 2], q_nc[:, 2, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
    set_axes_limits(ax[1, 0], p_nc[1:, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
    set_axes_limits(ax[1, 1], tau_nc[1:, 4, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
    set_axes_limits(ax[1, 2], tau_nc[1:, 10, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
    if bDef:
        set_axes_limits(ax[0, 3], topo_nc[:, 0, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)
        set_axes_limits(ax[1, 3], topo_nc[:, 3, 0, 1:-1, ny // 2], x=(0, 1), rel_tol=0.05)

    set_axes_labels(ax, bDef)

    def init():
        line_rho.set_data([], [])
        line_jx.set_data([], [])
        line_jy.set_data([], [])
        line_p.set_data([], [])
        line_tauxz_bot.set_data([], [])
        line_tauxz_top.set_data([], [])
        if bDef:
            line_h.set_data([], [])
            line_def.set_data([], [])
        return (line_rho, line_jx, line_jy, line_p, line_tauxz_bot,
                line_tauxz_top, line_h, line_def)

    def update(i):
        line_rho.set_data(x, q_nc[i, 0, 0, 1:-1, ny // 2])
        line_jx.set_data(x, q_nc[i, 1, 0, 1:-1, ny // 2])
        line_jy.set_data(x, q_nc[i, 2, 0, 1:-1, ny // 2])
        line_p.set_data(x, p_nc[i, 1:-1, ny // 2])
        line_tauxz_bot.set_data(x, tau_nc[i, 4, 0, 1:-1, ny // 2])
        line_tauxz_top.set_data(x, tau_nc[i, 10, 0, 1:-1, ny // 2])
        if bDef:
            line_h.set_data(x, topo_nc[i, 0, 0, 1:-1, ny // 2])
            line_def.set_data(x, topo_nc[i, 3, 0, 1:-1, ny // 2])
        return (line_rho, line_jx, line_jy, line_p, line_tauxz_bot,
                line_tauxz_top, line_h, line_def)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=nt,
        init_func=init,
        blit=True,
        interval=100,
        repeat=True
    )

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=max(1, int(nt / seconds)),
                        codec='libx264',
                        extra_args=['-pix_fmt', 'yuv420p', '-crf', '25'])

        outfile = os.path.join(os.path.dirname(filename_sol),
                                os.path.dirname(filename_sol).split(os.sep)[-1]) + ".mp4"

        ani.save(outfile, writer=writer, dpi=150)
        print(f"Saved animation to {outfile}")

    elif show_notebook:
        plt.close(fig)
        return HTML(ani.to_jshtml())

    else:
        plt.show()
