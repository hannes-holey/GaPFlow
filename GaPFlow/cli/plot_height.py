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
from argparse import ArgumentParser

from GaPFlow.viz.utils import get_pipeline
from GaPFlow.viz.plotting import plot_height


def get_parser():

    parser = ArgumentParser()
    parser.add_argument('-d', '--dim', type=int, default=1)
    parser.add_argument('--show-defo', action='store_true', default=False)
    parser.add_argument('--show-pressure', action='store_true', default=False)

    return parser


def main(cli=True, dim=1, show_defo=False, show_pressure=False):

    if cli:
        # overwrite defaults with cmdline args
        args = get_parser().parse_args()
        dim = args.dim
        show_defo = args.show_defo
        show_pressure = args.show_pressure

    nc_files = get_pipeline(name='topo.nc')
    plot_height(nc_files,
                dim=dim,
                show_defo=show_defo,
                show_pressure=show_pressure)
