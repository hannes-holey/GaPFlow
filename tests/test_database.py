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

from GaPFlow import Database
from GaPFlow.md import Mock


@pytest.mark.parametrize('method', ['rand', 'lhc', 'sobol'])
def test_addition(tmp_path, method):

    db_config = {'init_size': 4, 'init_width': 0.01, 'init_method': method,
                 'init_seed': 42, 'dtool_path': str(tmp_path)}
    geo = {'U': 1., 'V': 0.}
    prop = {'shear': 1., 'bulk': 0., 'EOS': 'PL'}
    gp = {'press_gp': False, 'shear_gp': False}

    md = Mock(prop, geo, gp)

    db = Database(md, db_config, num_extra_features=1)

    Xtest = np.random.uniform(size=(100, 7))
    db.initialize(Xtest)

    assert db.size == db_config['init_size']

    Xnew = np.random.uniform(size=(10, 7))
    db.add_data(Xnew)
    assert db.size == 14

    new_db = Database(md, db_config, num_extra_features=1)
    assert new_db.size == 14

    # A bare database object does not have an output path to write into
    # new_db.write()
    # Xtrain = np.load(os.path.join(tmp_path, 'Xtrain.npy'))
    # Ytrain = np.load(os.path.join(tmp_path, 'Ytrain.npy'))
    # Ytrain_err = np.load(os.path.join(tmp_path, 'Ytrain_err.npy'))

    # np.testing.assert_almost_equal(Xtrain, new_db._Xtrain)
    # np.testing.assert_almost_equal(Ytrain, new_db._Ytrain)
    # np.testing.assert_almost_equal(Ytrain_err, new_db._Ytrain_err)
