#
# Copyright 2026 Hannes Holey
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
"""Selection of the linear algebra backend (implicit_solver.linear_backend)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from GaPFlow import Problem, HAS_PETSC
from GaPFlow.solver_implicit.scipy_system import ScipySystem

CONFIG = Path(__file__).parents[1] / "config_lib" / "validation" / "parabolic_slider.yaml"


def _make_problem(backend, max_it=5):
    config = yaml.safe_load(CONFIG.read_text())
    config['numerics']['max_it'] = max_it
    config.setdefault('implicit_solver', {})['linear_backend'] = backend
    problem = Problem.from_string(yaml.safe_dump(config))
    problem._pre_run()
    return problem


def test_scipy_backend_selected():
    problem = _make_problem('scipy')
    assert isinstance(problem.solver.linear_solver, ScipySystem)


def test_auto_backend_selected():
    problem = _make_problem('auto')
    assert isinstance(problem.solver.linear_solver, ScipySystem) != HAS_PETSC


def test_invalid_backend():
    with pytest.raises(IOError):
        _make_problem('foo')


@pytest.mark.skipif(HAS_PETSC, reason="petsc4py is installed")
def test_petsc_backend_missing():
    with pytest.raises(RuntimeError):
        _make_problem('petsc')


@pytest.mark.skipif(not HAS_PETSC, reason="petsc4py is not installed")
def test_scipy_matches_petsc():
    results = {}
    for backend in ['scipy', 'petsc']:
        problem = _make_problem(backend)
        problem.run()
        results[backend] = problem.q[:2].copy()  # rho, jx (jy is round-off noise in 1D)

    np.testing.assert_allclose(results['scipy'], results['petsc'], rtol=1e-8)
