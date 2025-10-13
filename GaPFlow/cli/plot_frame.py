from importlib import resources
import matplotlib.pyplot as plt

from GaPFlow.viz.utils import get_pipeline
from GaPFlow.viz.plotting import plot_single_frame

plt.style.use(resources.files("GaPFlow.resources").joinpath("gapflow.mplstyle"))


def main():

    files = get_pipeline(name='sol.nc')
    plot_single_frame(files)
