import os
import jax.numpy as jnp


class Database:

    def __init__(self, minimum_size, outdir=None):

        self.outdir = outdir

        self.minimum_size = minimum_size

        self._Xtrain = jnp.empty((0, 6))
        self._Ytrain = jnp.empty((0, 13))

        if self.size == 0:
            self.X_scale = jnp.ones((6,))
            self.Y_scale = jnp.ones((13,))
        else:
            self.X_scale = self.normalizer(self._Xtrain)  # shape=(6, )
            self.Y_scale = self.normalizer(self._Ytrain)  # shape=(13, )

    # TODO
    @classmethod
    def from_numpy(cls, directory):
        raise NotImplementedError

    def add_data(self, Xnew, Ynew):

        self._Xtrain = jnp.vstack([self._Xtrain, Xnew])
        self._Ytrain = jnp.vstack([self._Ytrain, Ynew])

        self.X_scale = self.normalizer(self._Xtrain)
        self.Y_scale = self.normalizer(self._Ytrain)

        self.write()

    def normalizer(self, x):
        return jnp.maximum(jnp.max(jnp.abs(x), axis=0), 1e-12)

    @property
    def Xtrain(self):
        return self._Xtrain / self.X_scale

    @property
    def Ytrain(self):
        return self._Ytrain / self.Y_scale

    @property
    def size(self):
        return self._Xtrain.shape[0]

    def write(self):
        if self.outdir is not None:
            jnp.save(os.path.join(self.outdir, 'Xtrain.npy'), self._Xtrain)
            jnp.save(os.path.join(self.outdir, 'Ytrain.npy'), self._Ytrain)
