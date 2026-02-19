import os
import shutil
import matplotlib
import polars as pl
import numpy as np
from GaPFlow.problem import Problem
from GaPFlow.viz.animations import _create_animation_1d_gp, _display_animation


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
