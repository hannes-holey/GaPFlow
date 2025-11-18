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
import shutil
import matplotlib
import polars as pl
import numpy as np
from GaPFlow.problem import Problem
from GaPFlow.viz.animations import (_create_animation_1d, _create_animation_1d_gp,
                                    _create_animation_2d, _display_animation)


def test_animation_1d_elastic(tmp_path):

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
    max_it: 100 # 40_000
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

    ani = _create_animation_1d(filename_sol=os.path.join(myProblem.outdir, 'sol.nc'),
                               filename_topo=os.path.join(myProblem.outdir, 'topo.nc'))

    assert isinstance(ani, matplotlib.animation.FuncAnimation)
    assert ani._save_count == 11
    assert len(ani._fig.axes) == 8


def test_animation_1d(tmp_path):

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
"""

    myProblem = Problem.from_string(sim)
    myProblem.run()

    ani = _create_animation_1d(filename_sol=os.path.join(myProblem.outdir, 'sol.nc'),
                               filename_topo=os.path.join(myProblem.outdir, 'topo.nc'))

    assert isinstance(ani, matplotlib.animation.FuncAnimation)
    assert ani._save_count == 11
    assert len(ani._fig.axes) == 6


def test_animation_1d_gp(tmp_path):

    sim = f"""
options:
    output: {tmp_path}
    write_freq: 1
    use_tstamp: True
grid:
    Lx: 1470.
    Ly: 1.
    Nx: 200
    Ny: 1
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 0.8
    xW_D: 0.8
geometry:
    type: parabolic
    hmin: 12.
    hmax: 60.
    U: 0.12
    V: 0.
numerics:
    CFL: 0.5
    adaptive: 1
    tol: 1e-8
    dt: 0.05
    max_it: 10
properties:
    shear: 2.15
    bulk: 0.
    EOS: BWR
    T: 1.0
    rho0: 0.8
gp:
    press:
        fix_noise: True
        atol: 1.5
        rtol: 0.
        obs_stddev: 2.e-2
        max_steps: 10
        active_learning: False
    shear:
        fix_noise: True
        atol: 1.5
        rtol: 0.
        obs_stddev: 4.e-3
        max_steps: 10
        active_learning: False
db:
    init_size: 5
    init_method: rand
    init_width: 0.01 # default (for density)
"""

    myProblem = Problem.from_string(sim)
    myProblem.run()

    fname_sol = os.path.join(myProblem.outdir, 'sol.nc')

    gp_p = os.path.join(os.path.dirname(fname_sol), 'gp_zz.csv')
    gp_s = os.path.join(os.path.dirname(fname_sol), 'gp_xz.csv')

    tol_p = np.array(pl.read_csv(gp_p)['variance_tol'])
    tol_s = np.array(pl.read_csv(gp_s)['variance_tol'])

    ani = _create_animation_1d_gp(filename=fname_sol,
                                  tol_p=tol_p,
                                  tol_s=tol_s)

    assert isinstance(ani, matplotlib.animation.FuncAnimation)
    assert ani._save_count == 11
    assert len(ani._fig.axes) == 6

    save = False if shutil.which('ffmpeg') is None else True
    _display_animation(ani, fname_sol, seconds=2., save=save, show=False)


def test_animation_2d(tmp_path):

    sim = f"""
options:
    output: {tmp_path}
    write_freq: 10
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
    max_it: 100
properties:
    shear: 2.15
    bulk: 0.
    EOS: BWR
    T: 1.0
    rho0: 0.8
"""
    myProblem = Problem.from_string(sim)
    myProblem.run()

    fname = os.path.join(myProblem.outdir, 'sol.nc')

    ani = _create_animation_2d(filename=fname)

    assert isinstance(ani, matplotlib.animation.FuncAnimation)
    assert ani._save_count == 11
    assert len(ani._fig.axes) == 9  # 3x3

    save = False if shutil.which('ffmpeg') is None else True
    _display_animation(ani, fname, seconds=2., save=save, show=False)
