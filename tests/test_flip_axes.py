from copy import deepcopy
import io
import numpy as np
from GaPFlow.problem import Problem
from GaPFlow.io import read_yaml_input

sim = """
options:
    output: data/journal
    write_freq: 1000
    silent: True
grid:
    dx: 1.e-5
    dy: 1.e-5
    Nx: 100
    Ny: 100
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
    CFL: 0.5
    adaptive: 1
    tol: 1e-8
    dt: 1e-10
    max_it: 10_000
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


def test_x_y():

    with io.StringIO(sim) as file:
        input_x = read_yaml_input(file)
    input_y = deepcopy(input_x)

    # Swap axes
    input_y['geometry']['U'] = 0.
    input_y['geometry']['V'] = input_x['geometry']['U']
    input_y['geometry']['flip'] = True

    problem_x = Problem.from_dict(input_x)
    problem_y = Problem.from_dict(input_y)

    for _ in range(5):
        problem_x.update()
        problem_y.update()

        np.testing.assert_almost_equal(problem_x.q[0, 1:-1, 1:-1],
                                       problem_y.q[0, 1:-1, 1:-1].T)

        np.testing.assert_almost_equal(problem_x.q[1, 1:-1, 1:-1],
                                       problem_y.q[2, 1:-1, 1:-1].T)

        np.testing.assert_almost_equal(problem_x.q[2, 1:-1, 1:-1],
                                       problem_y.q[1, 1:-1, 1:-1].T)
