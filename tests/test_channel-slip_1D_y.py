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


def p_channel_nondimensional(x, b):
    """Reference solution, cf Savio et al., Sci Adv. 2 (2016)"""

    kappa = 5 * b / (2 + 5 * b)
    slope = 6 * kappa / 5

    return (abs(x - 1) * slope - slope / 2) / kappa


@pytest.fixture(scope="session", params=[1e-4, 1e-5, 1e-6])
def setup(tmpdir_factory, request):
    config_file = os.path.join("tests", "channel-slip1D_y_incompressible.yaml")
    tmp_dir = tmpdir_factory.mktemp("tmp")

    myTestProblem = Input(config_file).getProblem()
    myTestProblem.surface["lslip"] = request.param
    myTestProblem.run(out_dir=tmp_dir)

    ds = DatasetSelector(tmp_dir, mode="name", fname=[tmp_dir.join(os.path.basename(myTestProblem.outpath))])

    data = ds.get_centerline(key="p", dir="y")[0]
    meta = ds.get_metadata()[0]

    yield data, meta


def test_pressure(setup):

    data, meta = setup

    y, p = data

    Nx = meta["disc"]["Nx"]
    Ny = meta["disc"]["Ny"]
    Ly = meta["disc"]["Ly"]
    lslip = meta["surface"]["lslip"]
    h = meta["geometry"]["h1"]
    eta = meta["material"]["shear"]
    V = meta["geometry"]["V"]

    # reference solution
    yref = 2. * y / Ly
    b = lslip / h
    p_ref = p_channel_nondimensional(yref, b)

    kappa = 5 * b / (2 + 5 * b)
    scalef = h**2 / (eta * V * Ly / 2)
    p *= scalef / kappa

    np.testing.assert_almost_equal(p, p_ref, decimal=2)
