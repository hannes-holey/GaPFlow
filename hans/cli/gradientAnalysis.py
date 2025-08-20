from hans.plottools import DatasetSelector
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import scienceplots
import numpy as np
import warnings

mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
plt.style.use('science')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

def get_AxisLimits(y):
    """Determines min and max values and adds 5% margin

    Parameters
    ----------
    y : 1D or 2D numpy array
        function values to plot

    Returns
    -------
    min, max : float
        values for axis y_lim
    """

    y = np.asarray(y).flatten()

    if np.isnan(y).any() or np.isinf(y).any():
        return -1., 1.

    y_min = y.min() - 0.05*np.ptp(y)
    y_max = y.max() + 0.05 * np.ptp(y)

    return y_min, y_max


def get_group_box(fig, axes, group_indices, color='blue', lw=2, alpha=0.3):
    """Draws a box to group subplots for easier understanding.
    Box is drawn around subplots of given indices.

    Parameters
    ----------
    fig : matplotlib figure
        figure
    axes : matplotlib axes
        axis vector
    group_indices : list[int]
        indices of selected subplots
    color : str, optional
        box color, by default 'blue'
    lw : int, optional
        linewidth, by default 2
    alpha : float, optional
        alpha value for blending, by default 0.3

    Returns
    -------
    matplotlib.patches.Rectangle
        rectangle object
    """

    bboxes = [axes[i].get_position() for (i) in group_indices]
    xmin = min(b.x0 for b in bboxes) - 0.035
    xmax = max(b.x1 for b in bboxes) + 0.003
    ymin = min(b.y0 for b in bboxes) - 0.03
    ymax = max(b.y1 for b in bboxes) + 0.03
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                     transform=fig.transFigure, color=color, alpha=alpha,
                     lw=lw, fill=True, zorder=0)
    rect.set_edgecolor("black")
    rect.set_linewidth(0.5)
    return rect


def init_contributions(contribution_groups, ax):
    """Init for the contribution boxes. 
    Draws the outline rectangles and returns the fill-rectangles.

    Parameters
    ----------
    contribution_groups : list[list[int]]
        contains lists that hold the indices of subplots that belong together
    ax : matplotlib axes
        axis object

    Returns
    -------
    contribution_fills : dict
        contains rectangle objects of the fill-rectangles 
        accesible with the indice of the corresponding subplot
    """

    cmap = cm.get_cmap('viridis')
    frame_x, frame_y = 0.88, 0.93
    frame_width, frame_height = 0.10, 0.04
    contribution_fills = {}

    for contribution_indices in contribution_groups:
        for idx in contribution_indices:
            frame = Rectangle(
                (frame_x, frame_y),
                frame_width,
                frame_height,
                transform=ax[idx].transAxes,
                facecolor='none',
                edgecolor='black',
                linewidth=0.5,
                zorder=5,
                clip_on=False
                )
            ax[idx].add_patch(frame)

            fill = Rectangle(
                (frame_x, frame_y),
                0.0,
                frame_height,
                transform=ax[idx].transAxes,
                facecolor=cmap(0.0),
                edgecolor='none',
                zorder=4,
                alpha=0.8,
                clip_on=False
            )
            ax[idx].add_patch(fill)
            contribution_fills[idx] = fill

    return contribution_fills


def update_contributions(ydata, list, frame, contribution_groups, contribution_fills, validity_check=None):
    """Fills the fill-rectangles depending on their relative contribution.
    Contribution is simply the integral over x of the absolute value

    Parameters
    ----------
    ydata : dict
        ydata for calculating integrals
    list : list[str]
        keys for data selection
    frame : int
        current frame for data selection
    contribution_groups : list[list[int]]
        groups with indices for relative contribution calculation
    contribution_fills : dict
        dict with rectangle objects accessible by int indices corresponding to contribution groups
    validity_check : list[int]
        if given, it is checked that the individual terms add up to the overall gradient (given by corresponding index)
        For example: contribution group = [[1,2]] and validity_check = [3] means that ydata of [1,2] should add up to ydata of [3]
    """

    cmap = cm.get_cmap('viridis')
    frame_width, frame_height = 0.10, 0.04
    imp = 0.5

    imp = {}
    # determine contribution
    for i,contribution_indices in enumerate(contribution_groups):
        group_sum = 0.
        if validity_check:
            arr_sum = np.zeros_like((ydata[list[0]])[frame])
        # calculate sum/integral
        for idx in contribution_indices:
            arr = (ydata[list[idx]])[frame]
            res = np.sum(np.abs(arr))
            imp[idx] = res
            group_sum += res
            if validity_check:
                arr_sum += arr
        # check the sum against the directly calculated overall gradient
        if validity_check:
            check_arr = (ydata[list[validity_check[i]]])[frame]
            #np.testing.assert_allclose(check_arr, arr_sum, rtol=1e-3, atol=1e-5)
        # normalize
        if group_sum > 1e-12:
            for idx in contribution_indices:
                imp[idx] = imp[idx] / group_sum

    # fill
    for contribution_indices in contribution_groups:
        for idx in contribution_indices:
            width = frame_width * imp[idx]
            color = cmap(imp[idx])
            contribution_fills[idx].set_width(width)
            contribution_fills[idx].set_facecolor(color)

    return


def gradientAnalysis(filename, savename):
    """Extensive overview of state variables and gradient contributions of rho and rho u

    Parameters
    ----------
    filename : string
        filename for DataSelector
    savename : string
        filename for saving animation. Can contain folder: r"folder/filename"

    When changing be aware that
    - list and bTime Series are consistent
    - 'list' is the main list for iterators
    - all variables that additionally contain predictor and corrector are in 'pre_cor_list'
    - indices for grouping (patches and background color) are hardcoded - rather 
    append new items in a new column to not disturb the order
    - items in list need to be valid keys (netCDF savefile)
    - new keys have to be added to the 'allowed-keys' in the get_centerlines() function
    """

    # items / keys
    list = ['rho', 'ga_rho_t', 'ga_jx_x', 'ga_h_t_rho', 'ga_h_x_jx',
            'jx', 'ga_jx_t', 'ga_p_x', 'ga_uxjx_x', 'ga_tauxx_x', 
            'ga_tauxz', 'ga_h_x_uxjx', 'ga_h_x_tauxx', 'ga_h_t_jx', 'ga_ux',
            'height', 'u', 'p', 'wallforce_dp', 'wallforce_dh',
            'dh_dt', 'g_u', 'p_prev', 'ga_mass', 'ga_ux'
            ]
    
    # specify items that additionally have a predictor and corrector value
    pre_cor_list = ['ga_jx_x', 'ga_h_t_rho', 'ga_h_x_jx',
                    'ga_p_x', 'ga_uxjx_x', 'ga_tauxx_x', 
                    'ga_tauxz', 'ga_h_x_uxjx', 'ga_h_x_tauxx', 'ga_h_t_jx']

    # 0:plotting over x | 1:plotting over t
    bTimeSeries = [0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,0,0,
                   0,0,0,1,1,
                   0,0,0,1,0]

    # y-labels
    ylabels = [
        r"$\rho(x)$ in $\frac{\mathrm{kg}}{\mathrm{m}^3}$",
        r"$\frac{\partial \rho}{\partial t}$ in $\frac{\mathrm{kg}}{\mathrm{m}^3 \mathrm{s}}$",
        r"$- \frac{\partial \rho u}{\partial x}$",
        r"$- \frac{1}{h} \frac{\partial h}{\partial t} \rho$",
        r"$- \frac{1}{h} \frac{\partial h}{\partial x} \rho u$", #
        r"$j_x$ in $\frac{\mathrm{kg}}{\mathrm{m}^2 \mathrm{s}}$",
        r"$\frac{\partial \rho u}{\partial t}$ in $\frac{\mathrm{kg}}{\mathrm{m}^2 \mathrm{s}^2}$",
        r"$-\frac{\partial p}{\partial x}$",
        r"$-u \frac{\partial \rho u}{\partial x}$",
        r"$\frac{\partial \tau_{xx}}{\partial x}$", #
        r"$\frac{1}{h} \Delta \tau_{xz}$",
        r"$-\frac{1}{h} \frac{\partial h}{\partial x} \rho u^2$",
        r"$-\frac{1}{h} \frac{\partial h}{\partial x} \left( \tau_{xx}\vert_{h2} - \overline{\tau_{xx}} \right)$",
        r"$- \frac{1}{h}  \frac{\partial h}{\partial t} \rho u$", 
        r"$u_x$ in m/s", #
        r"$h$ in m",
        r"$u$ in m",
        r"$p$ in Pa",
        r"$p_{ext}-p_{inner}$ in Pa",
        r"$\Delta h(t)$ in m", #
        r"$\frac{\partial h}{\partial t}$ in $\frac{m}{s}$",
        r"$g(u)$ in m",
        r"$p_{t-1}$ in Pa",
        r"overall mass $m$ in kg",
        r"$u_x$ in m/s"
    ]

    # get file and data
    files = DatasetSelector("data", mode="name", fname=[filename])
    array_dict = {}
    array_dict['ydata'] = {}
    array_dict['min_max'] = {}

    # Additional arrays required for variables with predictor and corrector
    array_dict['ydata_pred'] = {}
    array_dict['ydata_cor'] = {}

    # Fetch data for all variables, distinguishing between time series and non-time series data
    for idx, item in enumerate(list):
        array_dict[item] = {}
        if item in pre_cor_list:
            time, xdata, ydata_pred, ydata_cor = files.get_pre_cor_centerlines(key=item)[0]
            array_dict['ydata_pred'][item] = ydata_pred
            array_dict['ydata_cor'][item] = ydata_cor
            combined = (ydata_pred + ydata_cor) / 2
            array_dict['ydata'][item] = combined
            combined_data = np.concatenate([ydata_pred, ydata_cor, combined])
            array_dict['min_max'][item] = get_AxisLimits(combined_data)
        elif bTimeSeries[idx]:
            full_arr = np.array(files.get_scalar(key=item)).reshape(-1)
            array_dict['ydata'][item] = full_arr[len(full_arr)//2:]
            array_dict['min_max'][item] = get_AxisLimits(array_dict['ydata'][item])
        else:
            time, xdata, ydata = files.get_centerlines(key=item)[0]
            array_dict['ydata'][item] = ydata
            array_dict['min_max'][item] = get_AxisLimits(ydata)
    tmax = time[-1]
    xmax = xdata[-1]

    # create figure
    fig, axes = plt.subplots(5, 5, figsize=(20,12))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08, hspace=0.5, wspace=0.4)
    axes = axes.flatten(order='F')
    
    # axis labels and limits
    for ax, label in zip(axes, ylabels):
        ax.set_ylabel(label, fontsize=14)
    for i, item in enumerate(list):
        if bTimeSeries[i]:
            axes[i].set_xlim(0, tmax)
            axes[i].set_ylim(array_dict['min_max'][item])
        else:
            axes[i].set_xlim(0, xmax)

    # label format
    for ax in axes:
        y_formatter = mticker.ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 2))
        y_formatter.set_scientific(True)
        y_formatter.set_useOffset(True)
        ax.yaxis.set_major_formatter(y_formatter)
        x_formatter = mticker.ScalarFormatter(useMathText=True)
        x_formatter.set_powerlimits((-3, 3))
        x_formatter.set_scientific(True)
        x_formatter.set_useOffset(True)
        ax.xaxis.set_major_formatter(x_formatter)

    # grouping boxes
    rect_rho = get_group_box(fig, axes, [0,1,2,3,4], color='lightblue', alpha=0.15)
    rect_rhou = get_group_box(fig, axes, [5,6,7,8,9,10,11,12,13,14], color='green', alpha=0.075)
    rect_deform = get_group_box(fig, axes, [16,17,21,22], color='orange', alpha=0.1)
    fig.patches.append(rect_rho)
    fig.patches.append(rect_rhou)
    fig.patches.append(rect_deform)

    # main subplots
    axes[0].set_facecolor('#e0f7fa')
    axes[1].set_facecolor('#e0f7fa')
    axes[5].set_facecolor("#e0fae8")
    axes[6].set_facecolor("#e0fae8")
    #axes[12].set_facecolor("#faf7e0")
    #axes[13].set_facecolor("#faf7e0")
    #axes[19].set_facecolor("#faf7e0")

    # contribution bars
    contribution_indices_rho = [2,3,4]
    contribution_indices_rhou = [7,8,11,13,9,10,12] # stresses: 9,10,12
    contribution_groups = [contribution_indices_rho, contribution_indices_rhou]
    validity_check = [1,6] # indices of the overall gradient for validity check
    contribution_fills = init_contributions(contribution_groups, axes) # draws emtpy rectangles, returns fill objects

    lines = []
    for i, item in enumerate(list):
        axis_lines = []
        if item in pre_cor_list:
            line1, = axes[i].plot([], [], color='orange')
            line2, = axes[i].plot([], [], color='green')
            line3, = axes[i].plot([], [], color=colors[0])
            axis_lines = [line1, line2, line3]
        else:
            line, = axes[i].plot([], [])
            axis_lines = [line]
        lines.append(axis_lines)

    # Add legend across the top of the figure
    # Create custom legend entries
    blue_patch = mpatches.Patch(color="#5ee5f79d", label='Density')
    green_patch = mpatches.Patch(color="#6eee94", label='Momentum')
    yellow_patch = mpatches.Patch(color='orange', label='Elastic Deformation')
    fig.legend(handles=[blue_patch, green_patch, yellow_patch],
            loc='upper center',
            bbox_to_anchor=(0.8, 0.99),
            ncol=3,
            fontsize='large',
            frameon=False)

    # frame update
    def update(frame):
        y = array_dict['ydata']
        y_pred = array_dict['ydata_pred']
        y_cor = array_dict['ydata_cor']
        for idx, item in enumerate(list):
            if bTimeSeries[idx]:
                lines[idx][0].set_data(time[0:frame+1], (y[item])[0:frame+1])
            elif item in pre_cor_list:
                lines[idx][0].set_data(xdata, y_pred[item][frame])
                lines[idx][1].set_data(xdata, y_cor[item][frame])
                lines[idx][2].set_data(xdata, y[item][frame])
                combined_data = np.concatenate([y_pred[item][frame], 
                                                y_cor[item][frame], 
                                                y[item][frame]])
                min_max = get_AxisLimits(combined_data)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Attempting to set identical low and high ylims.*")
                    axes[idx].set_ylim(min_max)
            else:
                min_max = get_AxisLimits((y[item])[frame])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Attempting to set identical low and high ylims.*")
                    axes[idx].set_ylim(min_max)
                    lines[idx][0].set_data(xdata, (y[item])[frame])

        fig.suptitle(
            r"\textbf{Gradient Analysis} \textbar\ Frame: %d \textbar\ Time: %.2e s" % (frame, time[frame]),
            fontsize=16,
            color='#003366' # dark blue
        )

        # update contribution fills
        update_contributions(y, list, frame, contribution_groups, contribution_fills, validity_check=validity_check)

        return lines

    # init frame
    def init():
        for idx, line in enumerate(lines):
            if bTimeSeries[idx]:
                line[0].set_xdata(time)
            else:
                if item in pre_cor_list:
                    line[0].set_xdata(xdata)
                    line[1].set_xdata(xdata)
                    line[2].set_xdata(xdata)
                else:
                    line[0].set_xdata(xdata)
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, repeat=True)
    ani.save(savename + '.mp4', writer='ffmpeg')
    return


def h_rho_j_p_u(filename, savename):
    """Extensive overview of state variables and gradient contributions of rho and rho u

    Parameters
    ----------
    filename : string
        filename for DataSelector
    savename : string
        filename for saving animation. Can contain folder: r"folder/filename"

    When changing be aware that
    - list and bTime Series are consistent
    - 'list' is the main list for iterators
    - all variables that additionally contain predictor and corrector are in 'pre_cor_list'
    - indices for grouping (patches and background color) are hardcoded - rather 
    append new items in a new column to not disturb the order
    - items in list need to be valid keys (netCDF savefile)
    - new keys have to be added to the 'allowed-keys' in the get_centerlines() function
    """

    # items / keys
    list = ['height', 'rho', 'p', 'u']
    
    # specify items that additionally have a predictor and corrector value
    pre_cor_list = []

    # 0:plotting over x | 1:plotting over t
    bTimeSeries = [0,0,0,0]

    # y-labels
    ylabels = [
        r"$h$ in m",
        r"$\rho$ in $\frac{\mathrm{kg}}{\mathrm{m}^3}$",
        r"$p$ in Pa",
        r"$u$ in m"
    ]

    # get file and data
    files = DatasetSelector("data", mode="name", fname=[filename])
    array_dict = {}
    array_dict['ydata'] = {}
    array_dict['min_max'] = {}

    # Additional arrays required for variables with predictor and corrector
    array_dict['ydata_pred'] = {}
    array_dict['ydata_cor'] = {}

    # Fetch data for all variables, distinguishing between time series and non-time series data
    for idx, item in enumerate(list):
        array_dict[item] = {}
        if item in pre_cor_list:
            time, xdata, ydata_pred, ydata_cor = files.get_pre_cor_centerlines(key=item)[0]
            array_dict['ydata_pred'][item] = ydata_pred
            array_dict['ydata_cor'][item] = ydata_cor
            combined = (ydata_pred + ydata_cor) / 2
            array_dict['ydata'][item] = combined
            combined_data = np.concatenate([ydata_pred, ydata_cor, combined])
            array_dict['min_max'][item] = get_AxisLimits(combined_data)
        elif bTimeSeries[idx]:
            full_arr = np.array(files.get_scalar(key=item)).reshape(-1)
            array_dict['ydata'][item] = full_arr[len(full_arr)//2:]
            array_dict['min_max'][item] = get_AxisLimits(array_dict['ydata'][item])
        else:
            time, xdata, ydata = files.get_centerlines(key=item)[0]
            array_dict['ydata'][item] = ydata
            array_dict['min_max'][item] = get_AxisLimits(ydata)
    tmax = time[-1]
    xmax = xdata[-1]

    # create figure
    fig, axes = plt.subplots(2, 2, figsize=(7,7))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08, hspace=0.2, wspace=0.4)
    axes = axes.flatten(order='F')
    
    # axis labels and limits
    for ax, label in zip(axes, ylabels):
        ax.set_ylabel(label, fontsize=14)
    for i, item in enumerate(list):
        if bTimeSeries[i]:
            axes[i].set_xlim(0, tmax)
            axes[i].set_ylim(array_dict['min_max'][item])
        else:
            axes[i].set_xlim(0, xmax)
    axes[1].set_xlabel(r"$x$ in m", fontsize=14)
    axes[3].set_xlabel(r"$x$ in m", fontsize=14)

    # label format
    for ax in axes:
        y_formatter = mticker.ScalarFormatter(useMathText=True)
        y_formatter.set_powerlimits((-2, 2))
        y_formatter.set_scientific(True)
        y_formatter.set_useOffset(True)
        ax.yaxis.set_major_formatter(y_formatter)
        x_formatter = mticker.ScalarFormatter(useMathText=True)
        x_formatter.set_powerlimits((-3, 3))
        x_formatter.set_scientific(True)
        x_formatter.set_useOffset(True)
        ax.xaxis.set_major_formatter(x_formatter)

    lines = []
    for i, item in enumerate(list):
        axis_lines = []
        if item in pre_cor_list:
            line1, = axes[i].plot([], [], color='orange')
            line2, = axes[i].plot([], [], color='green')
            line3, = axes[i].plot([], [], color=colors[0])
            axis_lines = [line1, line2, line3]
        else:
            line, = axes[i].plot([], [])
            axis_lines = [line]
        lines.append(axis_lines)

    # frame update
    def update(frame):
        y = array_dict['ydata']
        y_pred = array_dict['ydata_pred']
        y_cor = array_dict['ydata_cor']
        for idx, item in enumerate(list):
            if bTimeSeries[idx]:
                lines[idx][0].set_data(time[0:frame+1], (y[item])[0:frame+1])
            elif item in pre_cor_list:
                lines[idx][0].set_data(xdata, y_pred[item][frame])
                lines[idx][1].set_data(xdata, y_cor[item][frame])
                lines[idx][2].set_data(xdata, y[item][frame])
                combined_data = np.concatenate([y_pred[item][frame], 
                                                y_cor[item][frame], 
                                                y[item][frame]])
                min_max = get_AxisLimits(combined_data)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Attempting to set identical low and high ylims.*")
                    axes[idx].set_ylim(min_max)
            else:
                min_max = get_AxisLimits((y[item])[frame])
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Attempting to set identical low and high ylims.*")
                    axes[idx].set_ylim(min_max)
                    lines[idx][0].set_data(xdata, (y[item])[frame])

        fig.suptitle(
            r"\textbf{Parabolic Slider}",
            fontsize=16,
            color='#003366' # dark blue
        )

        return lines

    # init frame
    def init():
        for idx, line in enumerate(lines):
            if bTimeSeries[idx]:
                line[0].set_xdata(time)
            else:
                if item in pre_cor_list:
                    line[0].set_xdata(xdata)
                    line[1].set_xdata(xdata)
                    line[2].set_xdata(xdata)
                else:
                    line[0].set_xdata(xdata)
        return lines

    ani = animation.FuncAnimation(fig, update, frames=len(time), init_func=init, repeat=True)
    ani.save(savename + '.mp4', writer='ffmpeg', dpi=300)
    return