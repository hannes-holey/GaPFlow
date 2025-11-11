
import os
import matplotlib
from GaPFlow.problem import Problem
from GaPFlow.viz.animations import _create_animation_1d


def test_animation_elastic(tmp_path):

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


def test_animation(tmp_path):

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


def test_animation_gp():
    pass
