import matplotlib.pyplot as plt
from hans_mugrid.plotting import plot_single_frame, plot_evolution, plot_history, animate, get_pipeline

plt.style.use("gpjax.mplstyle")

if __name__ == "__main__":

    nc_files = get_pipeline()
    plot_single_frame(nc_files)

    # csv_files = get_pipeline(suffix='.csv')
    # plot_history(csv_files)
    # for file in nc_files:
    #     plot_single_frame(file)
    #     # plot_evolution(savefig=True)
    # animate(file, save=True, seconds=10, tol_f=0.1)
