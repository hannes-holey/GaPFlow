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
from argparse import ArgumentParser

from GaPFlow.viz.utils import get_pipeline
from GaPFlow.viz.plotting import plot_history


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
