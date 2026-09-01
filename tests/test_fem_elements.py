#
# Copyright 2026 Christoph Huber
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

from GaPFlow.solver_fem.elements import TaylorHoodQ2Q1


@pytest.fixture(scope='module')
def element():
    return TaylorHoodQ2Q1(dx=1.0, dy=1.0)


def test_q1_partition_of_unity(element):
    """Q1 shape functions sum to 1 at every quadrature point."""
    assert np.allclose(element.Q1.N.sum(axis=1), 1.0)


def test_q2_partition_of_unity(element):
    """Q2 shape functions sum to 1 at every quadrature point."""
    assert np.allclose(element.Q2.N.sum(axis=1), 1.0)


def test_quadrature_weights(element):
    """Quadrature weights integrate the unit square exactly (area = 1.0)."""
    assert np.isclose(element.Quadrature.weights.sum(), 1.0)


def test_q1_dN_dx_sum_zero(element):
    """Derivatives of Q1 shape functions sum to zero (constant reproduction)."""
    assert np.allclose(element.Q1.dN_dx.sum(axis=1), 0.0)
    assert np.allclose(element.Q1.dN_dy.sum(axis=1), 0.0)


def test_q2_dN_dx_sum_zero(element):
    """Derivatives of Q2 shape functions sum to zero (constant reproduction)."""
    assert np.allclose(element.Q2.dN_dx.sum(axis=1), 0.0)
    assert np.allclose(element.Q2.dN_dy.sum(axis=1), 0.0)
