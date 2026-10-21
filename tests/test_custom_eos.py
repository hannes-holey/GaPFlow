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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Custom equation of state for the FEM solver."""

import numpy as np
import pytest

from GaPFlow import Problem

CONFIG = """
options:
    output: /tmp/custom_eos_test
    write_freq: 1000
    save_output: False
    print_progress: False
    print_config: False

grid:
    Lx: 0.02
    Ly: 0.01
    Nx: 16
    Ny: 8
    xW: ['D', 'N', 'N']
    xE: ['D', 'N', 'N']
    yS: ['N', 'N', 'N']
    yN: ['N', 'N', 'N']
    xW_D: 1.0
    xE_D: 1.0

geometry:
    type: inclined
    hmin: 1.0e-5
    hmax: 2.0e-5
    U: 1.0
    V: 0.0

numerics:
    solver: implicit
    dt: 1.0e-4
    tol: 0.
    max_it: 4

properties:
    EOS: PL
    rho0: 1.0
    P0: 1.0e5
    alpha: 0.5
    shear: 0.1
    bulk: 0.

implicit_solver:
    max_iter: 10
    physics:
        squeeze: True
"""

P0 = 1.0e5
EXPONENT = 1.0 / (1.0 - 0.5 * 0.5)


def _run(eos=None):
    problem = Problem.from_string(CONFIG)
    if eos is not None:
        problem.set_custom_eos(eos)
    problem.run()
    return problem.q.copy()


def test_custom_eos_matches_builtin_power_law():
    q_ref = _run()
    q_user = _run(lambda rho: P0 * rho**EXPONENT)
    np.testing.assert_allclose(q_user, q_ref, rtol=1e-8, atol=1e-12)


def test_custom_eos_with_gap_height():
    problem = Problem.from_string(CONFIG)
    problem.set_custom_eos(lambda rho, h: P0 * (1.0 + 0.1 * h / 1.0e-5) * rho**EXPONENT)
    problem.run()

    nf = problem.solver.quad_mgr.nf
    p, rho, h = nf('p'), nf('rho'), nf('h')
    p_expected = problem.pressure.p_from_rho(rho, h)
    np.testing.assert_allclose(p, p_expected, rtol=1e-8)
    assert not np.allclose(h, h.flat[0])


def test_custom_eos_not_increasing_raises():
    problem = Problem.from_string(CONFIG)
    problem.set_custom_eos(lambda rho: P0 * (2.0 - rho)**2)
    with pytest.raises(ValueError, match="strictly increasing"):
        problem.run()
