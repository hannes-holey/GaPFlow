import os
from argparse import ArgumentParser
from importlib import resources
import matplotlib.pyplot as plt
from GaPFlow.plotting import plot_history, get_pipeline


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('-g', '--gp', action='store_true', default=False)

    return parser


def main():

    args = get_parser().parse_args()

    files = get_pipeline(name='history.csv')

    files_gp_press = []
    files_gp_shear = []
    if args.gp:
        files_gp_press = [(os.path.join(os.path.dirname(file), 'gp_zz.csv'), i)
                          for i, file in enumerate(files)
                          if 'gp_zz.csv' in os.listdir(os.path.dirname(file))]

        files_gp_shear = [(os.path.join(os.path.dirname(file), 'gp_xz.csv'), i)
                          for i, file in enumerate(files)
                          if 'gp_xz.csv' in os.listdir(os.path.dirname(file))]

    plot_history(files, files_gp_press, files_gp_shear)
