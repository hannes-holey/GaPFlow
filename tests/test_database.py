import numpy as np

from GaPFlow.db import Database


def test_addition():

    db_config = {'dtool': False, 'init_size': 5, 'init_width': 0.01}
    db = Database(db_config, outdir=None)

    Xnew = np.random.uniform(size=(10, 6))
    geo = {'U': 1., 'V': 0.}
    prop = {'shear': 1., 'bulk': 0., 'EOS': 'PL'}
    db.add_data(Xnew, prop=prop, geo=geo)

    assert db.size == 10
