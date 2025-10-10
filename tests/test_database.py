import os
import numpy as np

from GaPFlow.db import Database
from GaPFlow.md import Mock


def test_addition(tmp_path):

    db_config = {'init_size': 5, 'init_width': 0.01, 'init_method': 'rand'}
    geo = {'U': 1., 'V': 0.}
    prop = {'shear': 1., 'bulk': 0., 'EOS': 'PL'}
    gp = {'press_gp': False, 'shear_gp': False}

    md = Mock(prop, geo, gp)

    db = Database.from_dtool(db_config, md, str(tmp_path))

    Xtest = np.random.uniform(size=(100, 6))
    db.fill_missing(Xtest)

    assert db.size == db_config['init_size']

    Xnew = np.random.uniform(size=(10, 6))
    db.add_data(Xnew)
    assert db.size == 15

    new_db = Database.from_dtool(db_config, md, str(tmp_path))
    assert new_db.size == 15

    new_db.write()

    Xtrain = np.load(os.path.join(tmp_path, 'Xtrain.npy'))
    Ytrain = np.load(os.path.join(tmp_path, 'Ytrain.npy'))
    Ytrain_err = np.load(os.path.join(tmp_path, 'Ytrain_err.npy'))

    np.testing.assert_almost_equal(Xtrain, new_db._Xtrain)
    np.testing.assert_almost_equal(Ytrain, new_db._Ytrain)
    np.testing.assert_almost_equal(Ytrain_err, new_db._Ytrain_err)
