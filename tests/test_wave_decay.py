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
    output: data/decay
    write_freq: 100
    use_tstamp: False
    silent: True
grid:
    Lx: 3.2e-7
    Ly: 1
    Nx: 256
    Ny: 1
    xE: ['P', 'P', 'P']
    xW: ['P', 'P', 'P']
    yS: ['P', 'P', 'P']
    yN: ['P', 'P', 'P']
geometry:
    type: inclined
    hmin: 5e-9
    hmax: 5e-9
    U: 0.
    V: 0.
numerics:
    adaptive: 0
    CFL: 0.5
    dt: 1e-13
    max_it: 5_000
properties:
    EOS: cubic
    shear: 3.92293e-05    # N s/m^2
    bulk: 0.              # N s/m^2
    rho0: 762.8617        # kg/m^3
    a: 1.33030e-1
    b: -1.41778e2
    c: 8.35134e4
    d: -2.86532e6
"""


def longitudinal_decay_underdamped(t, a, b, c):
    return np.exp(-t / a) * (np.cos(b * t) - c * np.sin(b * t))


def longitudinal_decay_overdamped(t, a, b, c):
    return np.exp(-t / a) * (np.cosh(b * t) - c * np.sinh(b * t))


@pytest.fixture(scope='session')
def setup():
    with io.StringIO(sim) as ymlfile:
        input_dict = read_yaml_input(ymlfile)

    yield input_dict


@pytest.mark.parametrize('n', [1, 2, 4, 8])
def test_shear_wave_decay(setup, n):

    input_dict = setup
    problem = Problem._from_dict(input_dict)
    problem._pre_run()

    h = problem.geo['hmin']
    kin_visc = problem.prop['shear'] / problem.prop['rho0']
    Lx = problem.grid['Lx']

    kn = n * 2. * np.pi / Lx
    tau = h**2 / (6 * kin_visc)

    x = problem.topo.x[1:-1, 1]

    problem.q[2, 1:-1, :] = np.sin(kn * x)[:, None]
    problem.kinetic_energy_old = problem.kinetic_energy

    for i in range(200):
        problem.update()
        jy_ana = np.sin(kn * x) * np.exp(-2 * problem.simtime / tau)
        jy_num = problem.q[2, 1:-1, 1]

        np.testing.assert_almost_equal(jy_num, jy_ana, decimal=4)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_sound_wave_decay(setup, n):

    input_dict = setup
    problem = Problem._from_dict(input_dict)
    problem._pre_run()

    h = problem.geo['hmin']
    kin_visc = problem.prop['shear'] / problem.prop['rho0']
    Lx = problem.grid['Lx']

    kn = n * 2. * np.pi / Lx
    tau = h**2 / (6 * kin_visc)
    cT = problem.pressure.v_sound

    x = problem.topo.x[1:-1, 1]
    problem.q[1, 1:-1, :] = np.sin(kn * x)[:, None]
    problem.kinetic_energy_old = problem.kinetic_energy

    k_crit = 6. * kin_visc / (h**2 * cT)

    for i in range(400):
        problem.update()

        if kn > k_crit:
            sT = np.sqrt(cT**2 - (1 / tau / kn)**2)
            jx_ana = np.sin(kn * x) * longitudinal_decay_underdamped(problem.simtime,
                                                                     tau, sT * kn, 1 / (tau * sT * kn))
        else:
            isT = np.sqrt((1 / tau / kn)**2 - cT**2)
            jx_ana = np.sin(kn * x) * longitudinal_decay_overdamped(problem.simtime,
                                                                    tau, isT * kn, 1 / (tau * isT * kn))

        jx_num = problem.q[1, 1:-1, 1]
        np.testing.assert_almost_equal(jx_num, jx_ana, decimal=3)
