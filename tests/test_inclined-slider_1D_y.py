#
# Copyright 2020-2022, 2025 Hannes Holey
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


import os
import numpy as np
import pytest

from hans.input import Input
from hans.plottools import DatasetSelector


@pytest.fixture(scope="session")
def setup(tmpdir_factory):
    config_file = os.path.join("tests", "inclined-slider1D_y_ideal-gas.yaml")
    tmp_dir = tmpdir_factory.mktemp("tmp")

    myTestProblem = Input(config_file).getProblem()
    myTestProblem.run(out_dir=tmp_dir)

    file = DatasetSelector("", mode="name", fname=[str(tmp_dir.join(os.path.basename(myTestProblem.outpath)))])

    data = file.get_centerline(dir="y")

    yield data


@pytest.mark.long
def test_pressure(setup):

    p_ref = np.loadtxt(os.path.join("tests", "inclined-slider1D_ideal-gas_U50_s5.6e-4.dat"), unpack=True, usecols=(2,))

    for xdata, ydata in setup:
        p = ydata["p"]
        np.testing.assert_almost_equal(p / 1e6, p_ref / 1e6, decimal=1)


@pytest.mark.long
def test_density(setup):

    rho_ref = np.loadtxt(os.path.join("tests", "inclined-slider1D_ideal-gas_U50_s5.6e-4.dat"), unpack=True, usecols=(1,))

    for xdata, ydata in setup:
        rho = ydata["rho"]
        np.testing.assert_almost_equal(rho, rho_ref, decimal=1)
