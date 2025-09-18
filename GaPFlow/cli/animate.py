import os
import numpy as np
from importlib import resources
import matplotlib.pyplot as plt
from GaPFlow.plotting import animate, animate_gp, get_pipeline

import pandas as pd


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def main():

    file = get_pipeline(name='sol.nc', mode='single')

    gp_p = os.path.join(os.path.dirname(file), 'gp_press.csv')
    gp_s = os.path.join(os.path.dirname(file), 'gp_shear.csv')

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

    if tol_s is None and tol_p is None:
        animate(file)
    else:
        animate_gp(file, tol_p=tol_p, tol_s=tol_s)
