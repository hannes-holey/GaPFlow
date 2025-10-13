from importlib import resources
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from GaPFlow.viz.utils import get_pipeline
from GaPFlow.viz.plotting import plot_evolution


plt.style.use(resources.files("GaPFlow.resources").joinpath("gapflow.mplstyle"))


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('-e', '--every', type=int, default=1)

    return parser


def main():

    args = get_parser().parse_args()

    file = get_pipeline(name='sol.nc', mode='single')
    plot_evolution(file, every=args.every)
