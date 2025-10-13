import os
import pytest
import numpy as np

from GaPFlow.db import Database
from GaPFlow.md import Mock


@pytest.mark.parametrize('method', ['rand', 'lhc', 'sobol'])
def test_addition(tmp_path, method):

    db_config = {'init_size': 4, 'init_width': 0.01, 'init_method': method, 'init_seed': 42}
    geo = {'U': 1., 'V': 0.}
    prop = {'shear': 1., 'bulk': 0., 'EOS': 'PL'}
    gp = {'press_gp': False, 'shear_gp': False}

    md = Mock(prop, geo, gp)

    db = Database(str(tmp_path), md, db_config)

    Xtest = np.random.uniform(size=(100, 6))
    db.initialize(Xtest)

    assert db.size == db_config['init_size']

    Xnew = np.random.uniform(size=(10, 6))
    db.add_data(Xnew)
    assert db.size == 14

    new_db = Database(str(tmp_path), md, db_config)
    assert new_db.size == 14

    new_db.write()

    Xtrain = np.load(os.path.join(tmp_path, 'Xtrain.npy'))
    Ytrain = np.load(os.path.join(tmp_path, 'Ytrain.npy'))
    Ytrain_err = np.load(os.path.join(tmp_path, 'Ytrain_err.npy'))

    np.testing.assert_almost_equal(Xtrain, new_db._Xtrain)
    np.testing.assert_almost_equal(Ytrain, new_db._Ytrain)
    np.testing.assert_almost_equal(Ytrain_err, new_db._Ytrain_err)
