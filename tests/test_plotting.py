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

from GaPFlow import Problem
from GaPFlow.viz.plotting import _plot_height_1d, _plot_height_2d, _plot_multiple_frames_1d, plot_frame


def test_plot_1d(tmp_path):

    sim = f"""
options:
    output: {tmp_path}
    write_freq: 10
    silent: False
grid:
    dx: 1.e-5
    dy: 1.
    Nx: 100
    Ny: 1
    xE: ['P', 'P', 'P']
    xW: ['P', 'P', 'P']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
geometry:
    type: journal
    CR: 1.e-2
    eps: 0.7
    U: 0.1
    V: 0.
numerics:
    CFL: 0.25
    adaptive: 1
    tol: 1e-8
    dt: 1e-10
    max_it: 100
properties:
    shear: 0.0794
    bulk: 0.
    EOS: DH
    P0: 101325.
    rho0: 877.7007
    T0: 323.15
    C1: 3.5e10
    C2: 1.23
    elastic:
        E: 5e09
        v: 0.3
        alpha_underrelax: 1e-04
"""

    myProblem = Problem.from_string(sim)
    myProblem.run()

    filename = os.path.join(myProblem.outdir, 'topo.nc')

    fig, axes = _plot_height_1d(filename, show_defo=False, show_pressure=False)
    assert len(axes) == 1

    fig, axes = _plot_height_1d(filename, show_defo=True, show_pressure=False)
    assert len(axes) == 2

    fig, axes = _plot_height_1d(filename, show_defo=False, show_pressure=True)
    assert len(axes) == 2

    fig, axes = _plot_height_1d(filename, show_defo=True, show_pressure=True)
    assert len(axes) == 3

    filename = os.path.join(myProblem.outdir, 'sol.nc')

    fig, axes = _plot_multiple_frames_1d(filename, every=1)
    for ax in axes.flat:
        assert len(ax.get_lines()) == 11

    fig, axes = _plot_multiple_frames_1d(filename, every=2)
    for ax in axes.flat:
        assert len(ax.get_lines()) == 6

    # Just check that plotting does not raise any error
    plot_frame([filename, ], dim=1, show=False)


def test_plot_2d(tmp_path):
    sim = f"""
options:
    output: {tmp_path}
    write_freq: 1
    use_tstamp: True
grid:
    Lx: 1470.
    Ly: 1470.
    Nx: 100
    Ny: 100
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['D', 'N', 'N']
    yN: ['D', 'N', 'N']
    xE_D: 0.8
    xW_D: 0.8
    yS_D: 0.8
    yN_D: 0.8
geometry:
    type: asperity
    hmin: 12.
    hmax: 60.
    U: 0.12
    V: 0.
numerics:
    CFL: 0.5
    adaptive: 1
    tol: 1e-8
    dt: 0.05
    max_it: 1
properties:
    shear: 2.15
    bulk: 0.
    EOS: BWR
    T: 1.0
    rho0: 0.8
"""
    myProblem = Problem.from_string(sim)
    myProblem.run()

    fname = os.path.join(myProblem.outdir, 'topo.nc')

    fig, axes = _plot_height_2d(fname)
    assert len(axes) == 3

    # Just check that plotting does not raise any error
    fname = os.path.join(myProblem.outdir, 'sol.nc')
    plot_frame([fname, ], dim=2, show=False)
