import os
import numpy as np
import netCDF4
import pytest

from GaPFlow.problem import Problem

sim = """
options:
    output: data/journal
    write_freq: 1000
    silent: True
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


@pytest.fixture(scope="session")
def setup():

    problem = Problem.from_string(sim)
    problem.run()

    datafile = os.path.join("tests",
                            "regression_tests",
                            "testdata",
                            "2025-09-16_journal.nc")

    nc = netCDF4.Dataset(datafile, 'r')

    yield problem, nc


def test_pressure(setup):
    problem, nc = setup
    p_new = problem.pressure.pressure[1:-1, 1]
    p_old = np.asarray(nc.variables['pressure'])[-1, 1:-1, 1]

    np.testing.assert_almost_equal(p_new / p_new[0], p_old / p_old[0], decimal=4)


def test_density(setup):
    problem, nc = setup
    r_new = problem.density
    r_old = np.asarray(nc.variables['solution'])[-1, 0, 0, 1:-1, 1]

    np.testing.assert_almost_equal(r_new / r_new[0], r_old / r_old[0], decimal=4)
