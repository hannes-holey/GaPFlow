import numpy as np

from GaPFlow.db import Database


def test_addition():

    db = Database(0)

    Xnew = np.random.uniform(size=(10, 6))
    geo = {'U': 1., 'V': 0.}
    prop = {'shear': 1., 'bulk': 0., 'EOS': 'PL'}
    db.add_data(Xnew.T, prop=prop, geo=geo, dtool=False)

    assert db.size == 10
