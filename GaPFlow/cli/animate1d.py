#
# Copyright 2025 Hannes Holey
#           2025 Christoph Huber
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import polars as pl
import numpy as np
from argparse import ArgumentParser

from GaPFlow.viz.utils import get_pipeline
from GaPFlow.viz.animations import animate_1d, animate_1d_gp


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', default=False)
    parser.add_argument('-p', '--path', type=str, default='.')
    parser.add_argument('-m', '--mode', type=str, default='single')

    return parser


def main(cli=True, path='.', save=False, mode='single'):

    if cli:
        # overwrite defaults with cmdline args
        args = get_parser().parse_args()
        save = args.save
        path = args.path
        mode = args.mode

    file_sol = get_pipeline(path=path, name='sol.nc', mode=mode)
    file_topo = file_sol.replace('sol.nc', 'topo.nc')

    gp_p = os.path.join(os.path.dirname(file_sol), 'gp_zz.csv')
    gp_s = os.path.join(os.path.dirname(file_sol), 'gp_xz.csv')

    try:
        df_p = pl.read_csv(gp_p)
        tol_p = np.array(df_p['variance_tol'])
    except FileNotFoundError:
        tol_p = None

    try:
        df_s = pl.read_csv(gp_s)
        tol_s = np.array(df_s['variance_tol'])
    except FileNotFoundError:
        tol_s = None

    has_gp_models = tol_s is not None and tol_p is not None

    if has_gp_models:
        animate_1d_gp(file_sol, save=save, tol_p=tol_p, tol_s=tol_s)
    else:
        animate_1d(file_sol, file_topo, save=save)
