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
import io
import pytest
import numpy as np

from GaPFlow.io import read_yaml_input
from GaPFlow import Problem


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
    C1: 3.5e12
    C2: 1.23
"""


def sommerfeld_solution(x, Lx, mu, U, clearance_ratio, eps, P0):
    """Analytical solution to the journal bearing problem for incompressible fluids.

    Parameters
    ----------
    x : np.ndarray
        Circumferential coordinate
    Lx : float
        Circumference
    mu : float
        Viscosity
    U : float
        Velocity
    clearance_ratio : float
        Clearance ratio
    eps : float
        Eccentricity ratio
    P0 : float
        Boundary pressure

    Returns
    -------
    np.ndarray
        Pressure distribution
    """

    Rb = Lx / (2. * np.pi)
    c = clearance_ratio * Rb
    omega = U / Rb

    prefac = 6. * mu * omega * (Rb / c)**2 * eps

    P = P0 + prefac * np.sin(x / Rb) * (2. + eps * np.cos(x / Rb)) / ((2. + eps**2) * (1. + eps * np.cos(x / Rb))**2)

    return P


@pytest.fixture(scope='session')
def setup():
    with io.StringIO(sim) as ymlfile:
        input_dict = read_yaml_input(ymlfile)

    yield input_dict


@pytest.mark.parametrize('eps', [0.5, 0.7, 0.9])
def test_pressure_profile(setup, eps):

    input_dict = setup
    input_dict['geometry']['eps'] = eps

    problem = Problem._from_dict(input_dict)
    problem.run()

    p_num = problem.pressure.pressure[1:-1, 1]

    Lx = problem.grid['Lx']
    U = problem.pressure.geo['U']
    CR = problem.pressure.geo['CR']
    mu = problem.pressure.prop['shear']

    Nx = 100
    x_ana = np.linspace(0., Lx, Nx + 1)
    x_num = (x_ana[1:] + x_ana[:-1]) / 2.

    dp = p_num[1] - p_num[0]

    p_ana = sommerfeld_solution(x_num, Lx, mu, U, CR, eps, p_num[0] - dp / 2)

    # Numerical solution is for almost incompressible fluid
    rel_err = np.linalg.norm(p_ana - p_num) / np.linalg.norm(p_ana)
    assert rel_err < 0.02
