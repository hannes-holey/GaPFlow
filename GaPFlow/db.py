import jax.numpy as jnp


class Database:

    def __init__(self, minimum_size, location=None):

        self.minimum_size = minimum_size

        self._Xtrain = jnp.empty((0, 6))
        self._Ytrain = jnp.empty((0, 13))

        self.X_scale = jnp.ones((6,))
        self.Y_scale = jnp.ones((13,))

    def add_data(self, Xnew, Ynew):

        self._Xtrain = jnp.vstack([self._Xtrain, Xnew])
        self._Ytrain = jnp.vstack([self._Ytrain, Ynew])

        self.X_scale = jnp.maximum(jnp.max(jnp.abs(self._Xtrain), axis=0), 1e-12)  # shape=(6, )
        self.Y_scale = jnp.maximum(jnp.max(jnp.abs(self._Ytrain), axis=0), 1e-12)  # shape=(13, )

    @property
    def Xtrain(self):
        return self._Xtrain / self.X_scale

    @property
    def Ytrain(self):
        return self._Ytrain / self.Y_scale

    @property
    def size(self):
        return self._Xtrain.shape[0]
