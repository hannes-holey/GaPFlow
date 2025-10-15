import os
import numpy as np
import netCDF4
import pytest

from GaPFlow.problem import Problem

sim = """
options:
    output: data/inclined
    write_freq: 1000
    silent: True
grid:
    Lx: 0.1
    Ly: 1.
    Nx: 100
    Ny: 1
    xE: ['D', 'N', 'N']
    xW: ['D', 'N', 'N']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
    xE_D: 1.1853
    xW_D: 1.1853
geometry:
    type: inclined
    hmax: 6.6e-5
    hmin: 1e-5
    U: 50.
    V: 0.
numerics:
    adaptive: 1
    CFL: 0.4
    tol: 1e-6
    dt: 1e-8
    max_it: 20_000
properties:
    EOS: PL
    shear: 1.846e-5
    bulk: 0.
    P0: 101325
    rho0: 1.1853
    alpha: 0.
"""


@pytest.fixture(scope="session")
def setup():

    problem = Problem.from_string(sim)
    problem.run()

    datafile = os.path.join("tests",
                            "regression_tests",
                            "testdata",
                            "2025-09-09_inclined.nc")

    nc = netCDF4.Dataset(datafile, 'r')

    yield problem, nc


def test_pressure(setup):
    problem, nc = setup
    p_new = problem.pressure.pressure[1:-1, 1]
    p_old = np.asarray(nc.variables['pressure'])[-1, 1:-1, 1]

    np.testing.assert_almost_equal(p_new / p_new[0], p_old / p_old[0], decimal=4)


def test_density(setup):
    problem, nc = setup
    r_new = problem.centerline_mass_density
    r_old = np.asarray(nc.variables['solution'])[-1, 0, 0, 1:-1, 1]

    np.testing.assert_almost_equal(r_new / r_new[0], r_old / r_old[0], decimal=4)
