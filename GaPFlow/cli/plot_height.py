from importlib import resources
import matplotlib.pyplot as plt
from GaPFlow.plotting import plot_height, get_pipeline


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def main():
    nc_files = get_pipeline(name='gap.nc')
    for file in nc_files:
        plot_height(file)
