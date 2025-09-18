import os
import numpy as np
from importlib import resources
import matplotlib.pyplot as plt
from GaPFlow.plotting import animate, get_pipeline

import pandas as pd


plt.style.use(resources.files("GaPFlow.resources").joinpath("gpjax.mplstyle"))


def main():

    file = get_pipeline(name='sol.nc', mode='single')

    gp_p = os.path.join(os.path.dirname(file), 'gp_press.csv')
    gp_s = os.path.join(os.path.dirname(file), 'gp_shear.csv')

    df_p = pd.read_csv(gp_p)
    tol_p = np.array(df_p['variance_tol'])

    df_s = pd.read_csv(gp_s)
    tol_s = np.array(df_s['variance_tol'])

    # print(tol_p, tol_s)

    animate(file, tol_p=tol_p, tol_s=tol_s)
