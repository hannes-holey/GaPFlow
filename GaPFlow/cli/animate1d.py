import os
import numpy as np
from importlib import resources
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from GaPFlow.plotting import animate, animate_gp, get_pipeline

import pandas as pd


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', default=False)

    return parser


def main():

    args = get_parser().parse_args()

    file = get_pipeline(name='sol.nc', mode='single')

    gp_p = os.path.join(os.path.dirname(file), 'gp_zz.csv')
    gp_s = os.path.join(os.path.dirname(file), 'gp_xz.csv')

    try:
        df_p = pd.read_csv(gp_p)
        tol_p = np.array(df_p['variance_tol'])
    except FileNotFoundError:
        tol_p = None

    try:
        df_s = pd.read_csv(gp_s)
        tol_s = np.array(df_s['variance_tol'])
    except FileNotFoundError:
        tol_s = None

    # TODO: should also work if not all are stress models
    if tol_s is None or tol_p is None:
        animate(file, save=args.save)
    else:
        animate_gp(file, save=args.save, tol_p=tol_p, tol_s=tol_s)
