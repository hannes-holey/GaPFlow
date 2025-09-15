import numpy as np

from GaPFlow.db import Database


def test_addition():

    db = Database(0)

    Xnew = np.random.uniform(size=(10, 6))
    Ynew = np.random.uniform(size=(10, 13))

    db.add_data(Xnew, Ynew)

    assert db.size == 10
