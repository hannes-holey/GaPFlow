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
from copy import deepcopy
import io
import numpy as np
from GaPFlow import Problem
from GaPFlow.io import read_yaml_input

sim = """
options:
    output: data/journal
    write_freq: 1000
    silent: True
grid:
    dx: 1.e-5
    dy: 1.e-5
    Nx: 100
    Ny: 100
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
        input_x = read_yaml_input(file)
    input_y = deepcopy(input_x)

    # Swap axes
    input_y['geometry']['U'] = 0.
    input_y['geometry']['V'] = input_x['geometry']['U']
    input_y['geometry']['flip'] = True

    problem_x = Problem._from_dict(input_x)
    problem_y = Problem._from_dict(input_y)

    problem_x._pre_run()
    problem_y._pre_run()

    for _ in range(5):
        problem_x.update()
        problem_y.update()

        np.testing.assert_almost_equal(problem_x.q[0, 1:-1, 1:-1],
                                       problem_y.q[0, 1:-1, 1:-1].T)

        np.testing.assert_almost_equal(problem_x.q[1, 1:-1, 1:-1],
                                       problem_y.q[2, 1:-1, 1:-1].T)

        np.testing.assert_almost_equal(problem_x.q[2, 1:-1, 1:-1],
                                       problem_y.q[1, 1:-1, 1:-1].T)
