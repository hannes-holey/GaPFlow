from importlib import resources
import matplotlib.pyplot as plt
from GaPFlow.plotting import plot_single_frame, get_pipeline


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def main():

    files = get_pipeline(name='sol.nc')
    plot_single_frame(files)
