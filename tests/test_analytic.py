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
import pytest
import numpy as np

from GaPFlow.models.profiles import get_velocity_profiles, get_stress_profiles
from GaPFlow.models.viscous import stress_avg, stress_top, stress_bottom


@pytest.mark.parametrize('slip, Ls', [('both', 0.),
                                      ('both', 0.5),
                                      ('top', 0.),
                                      ('top', 0.5),
                                      ('bottom', 0.),
                                      ('bottom', 0.5),
                                      ])
def test_flow_rate(slip, Ls):

    Nz = 10_000
    hmax = 2.

    z = np.linspace(0., hmax, Nz)
    q = np.array([1., 2., 1.])

    u, v = get_velocity_profiles(z, q, Ls=Ls, U=1., V=1., slip=slip)

    assert np.isclose(np.trapezoid(u, z) / hmax, q[1])
    assert np.isclose(np.trapezoid(v, z) / hmax, q[2])


@pytest.mark.parametrize('slip, Ls', [('both', 0.),
                                      ('both', 0.5),
                                      ('top', 0.),
                                      ('top', 0.5),
                                      ])
def test_avg_stress(slip, Ls):

    q_test = np.array([1.0, 0.75, 0.25])
    h_test = np.array([1.0, 0.01, 0.01])  # "small slopes"

    Nz = 10_000
    z = np.linspace(0., 1., Nz)

    tau_xx, tau_yy, _, _, _, tau_xy = get_stress_profiles(z,
                                                          h_test,
                                                          q_test,
                                                          np.zeros(3),
                                                          np.zeros(3),
                                                          U=1.,
                                                          V=1.,
                                                          eta=1.,
                                                          zeta=1.,
                                                          Ls=Ls,
                                                          mode=slip)

    tau_avg = stress_avg(q_test, h_test, U=1., V=1., eta=1., zeta=1., Ls=Ls, slip=slip)

    assert np.isclose(np.trapezoid(tau_xx, z) / tau_avg[0], 1.)
    assert np.isclose(np.trapezoid(tau_yy, z) / tau_avg[1], 1.)
    assert np.isclose(np.trapezoid(tau_xy, z) / tau_avg[2], 1.)


@pytest.mark.parametrize('slip, Ls', [('both', 0.),
                                      ('both', 0.5),
                                      ('top', 0.),
                                      ('top', 0.5),
                                      ])
def test_wall_stress(slip, Ls):

    q_test = np.array([1.0, 0.75, 0.25])
    h_test = np.array([1.0, 0.01, 0.01])  # "small slopes"

    Nz = 10_000
    z = np.linspace(0., 1., Nz)

    tau_xx, tau_yy, tau_zz, tau_yz, tau_xz, tau_xy = get_stress_profiles(z,
                                                                         h_test,
                                                                         q_test,
                                                                         np.zeros(3),
                                                                         np.zeros(3),
                                                                         U=1.,
                                                                         V=1.,
                                                                         eta=1.,
                                                                         zeta=1.,
                                                                         Ls=Ls,
                                                                         mode=slip)

    tau_top = stress_top(q_test, h_test, U=1., V=1., eta=1., zeta=1., Ls=Ls, slip=slip)
    tau_bot = stress_bottom(q_test, h_test, U=1., V=1., eta=1., zeta=1., Ls=Ls, slip=slip)

    assert np.isclose(tau_bot[0], tau_xx[0])
    assert np.isclose(tau_top[0], tau_xx[-1])
    assert np.isclose(tau_bot[1], tau_yy[0])
    assert np.isclose(tau_top[1], tau_yy[-1])
    assert np.isclose(tau_bot[2], tau_zz[0])
    assert np.isclose(tau_top[2], tau_zz[-1])

    assert np.isclose(tau_bot[3], tau_yz[0])
    assert np.isclose(tau_top[3], tau_yz[-1])
    assert np.isclose(tau_bot[4], tau_xz[0])
    assert np.isclose(tau_top[4], tau_xz[-1])
    assert np.isclose(tau_bot[5], tau_xy[0])
    assert np.isclose(tau_top[5], tau_xy[-1])
