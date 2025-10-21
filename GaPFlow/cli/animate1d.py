#
# Copyright 2025 Hannes Holey
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
from GaPFlow.viz.animations import animate, animate_gp


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
        df_p = pl.read_csv(gp_p)
        tol_p = np.array(df_p['variance_tol'])
    except FileNotFoundError:
        tol_p = None

    try:
        df_s = pl.read_csv(gp_s)
        tol_s = np.array(df_s['variance_tol'])
    except FileNotFoundError:
        tol_s = None

    # TODO: should also work if not all are stress models
    if tol_s is None or tol_p is None:
        animate(file, save=args.save)
    else:
        animate_gp(file, save=args.save, tol_p=tol_p, tol_s=tol_s)
