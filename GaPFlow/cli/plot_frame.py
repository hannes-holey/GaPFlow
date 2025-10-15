from importlib import resources
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from GaPFlow.viz.utils import get_pipeline
from GaPFlow.viz.plotting import plot_single_frame

plt.style.use(resources.files("GaPFlow.resources").joinpath("gapflow.mplstyle"))


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('-d', '--dim', type=int, default=1)

    return parser


def main():

    args = get_parser().parse_args()

    files = get_pipeline(name='sol.nc')

    plot_single_frame(files, args.dim)
