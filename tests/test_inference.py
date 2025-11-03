import pytest
import numpy as np
from GaPFlow import Problem


def test_predict(tmp_path):
    sim = f"""
options:
    output: {tmp_path}
    write_freq: 100
    use_tstamp: False
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
    max_it: 5_000
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
        active_learning: True
    shear:
        fix_noise: True
        atol: 1.5
        rtol: 0.
        obs_stddev: 4.e-3
        max_steps: 10
        active_learning: True
db:
    # dtool_path: data/train  # defaults to options['output']/train
    init_size: 5
    init_method: rand
    init_width: 0.01 # default (for density)
"""

    testProblem = Problem.from_string(sim)
    testProblem.pre_run()

    mean1, var1 = testProblem.pressure.predict(predictor=True,
                                               compute_var=True)

    # Uses cached Cholesky factors and variance
    mean2, var2 = testProblem.pressure.predict(predictor=True,
                                               compute_var=False)

    np.testing.assert_almost_equal(np.asarray(mean1),
                                   np.asarray(mean2))

    np.testing.assert_almost_equal(np.asarray(var1),
                                   np.asarray(var2))

    mean3, var3 = testProblem.wall_stress_xz.predict(predictor=True,
                                                     compute_var=True)

    # Uses cached Cholesky factors and variance
    mean4, var4 = testProblem.wall_stress_xz.predict(predictor=True,
                                                     compute_var=False)

    np.testing.assert_almost_equal(np.asarray(mean3),
                                   np.asarray(mean4))

    np.testing.assert_almost_equal(np.asarray(var3),
                                   np.asarray(var4))
