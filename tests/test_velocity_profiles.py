#
# Copyright 2024-2025 Hannes Holey
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


import numpy as np
import pytest


from hans.models.velocity_profile import get_velocity_profiles


def trapezoid(u, z):
    return np.sum((u[1:] + u[:-1]) / 2. * (z[1:] - z[:-1]))


@pytest.mark.parametrize('slip', ['both', 'top', 'bottom', 'none'])
def test_flow_rate(slip):

    Nz = 10000
    hmax = 2.

    z = np.linspace(0., hmax, Nz)
    q = np.array([1., 2., 1.])

    Ls = 0.5

    u, v = get_velocity_profiles(z, q, Ls=Ls, U=1., V=1., slip=slip)

    assert np.isclose(trapezoid(u, z) / hmax, q[1])
    assert np.isclose(trapezoid(v, z) / hmax, q[2])
