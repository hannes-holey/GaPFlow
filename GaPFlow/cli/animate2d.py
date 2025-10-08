import os
import numpy as np
from importlib import resources
import matplotlib.pyplot as plt
from GaPFlow.plotting import animate2d, get_pipeline

import pandas as pd


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def main():

    file = get_pipeline(name='sol.nc', mode='single')

    gp_p = os.path.join(os.path.dirname(file), 'gp_zz.csv')
    gp_s_xz = os.path.join(os.path.dirname(file), 'gp_xz.csv')
    gp_s_yz = os.path.join(os.path.dirname(file), 'gp_yz.csv')

    try:
        df_p = pd.read_csv(gp_p)
        tol_p = np.array(df_p['variance_tol'])
    except FileNotFoundError:
        tol_p = None

    try:
        df_s_xz = pd.read_csv(gp_s_xz)
        tol_s_xz = np.array(df_s_xz['variance_tol'])
    except FileNotFoundError:
        tol_s_xz = None

    try:
        df_s_yz = pd.read_csv(gp_s_yz)
        tol_s_yz = np.array(df_s_yz['variance_tol'])
    except FileNotFoundError:
        tol_s_yz = None

    animate2d(file)

    # if tol_s_xz is None and tol_s_yz is None and tol_p is None:
    #     animate(file)
    # else:
    #     animate_gp(file, tol_p=tol_p, tol_s=tol_s_xz, tol)
