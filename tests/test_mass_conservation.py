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
    dx: 2.e-5
    dy: 2.e-5
    Nx: 50
    Ny: 50
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
        input_dict = read_yaml_input(file)

    problem = Problem(input_dict)

    mass_before = problem.mass.copy()

    for _ in range(50):
        problem.update()

    assert np.isclose(problem.mass, mass_before)
