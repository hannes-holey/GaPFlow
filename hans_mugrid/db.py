import jax.numpy as jnp
from jax import config
from jaxtyping import install_import_hook
config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


class Database:

    def __init__(self, minimum_size, location=None):

        self.minimum_size = minimum_size

        self.data = gpx.Dataset(X=jnp.empty((0, 6)),
                                y=jnp.empty((0, 13)),)  # X=Xtrain, y=Ytrain

        self.X_scale = jnp.ones((6,))
        self.Xext_scale = jnp.ones((7,))
        self.y_scale = jnp.ones((13,))

    @property
    def size(self):
        try:
            return self.data.n
        except AttributeError:
            return 0

    def add_data(self, X, Y):

        new_data = gpx.Dataset(X=X, y=Y)
        self.data += new_data

        # Normalization constants
        self.X_scale = jnp.maximum(jnp.max(jnp.abs(self.data.X), axis=0), 1e-12)  # shape=(dim, )
        self.y_scale = jnp.maximum(jnp.max(jnp.abs(self.data.y), axis=0), 1e-12)  # shape=(13, )

        self.Xext_scale = jnp.hstack([self.X_scale, jnp.ones((1, ))])  # shape = (dim + 1, )

        # Pressure data
        self.data_press = gpx.Dataset(X=self.data.X / self.X_scale,
                                      y=(self.data.y[:, 0] / self.y_scale[0])[:, None])

        # Shear stress
        yz_ids = jnp.array([4, 10])
        xz_ids = jnp.array([5, 11])

        # extend data for multi output kernel
        X_ext = jnp.vstack([  # shape = (2n, dim)
            jnp.hstack([self.data.X / self.X_scale, jnp.zeros((self.size, 1))]),
            jnp.hstack([self.data.X / self.X_scale, jnp.ones((self.size, 1))])
        ]
        )

        y_ext_xz = (self.data.y[:, xz_ids] / self.y_scale[xz_ids]).T.reshape(-1)[:, None]  # shape=(2n, 1)
        y_ext_yz = (self.data.y[:, yz_ids] / self.y_scale[yz_ids]).T.reshape(-1)[:, None]  # shape=(2n, 1)

        self.data_shear_xz = gpx.Dataset(X=X_ext,
                                         y=y_ext_xz)

        self.data_shear_yz = gpx.Dataset(X=X_ext,
                                         y=y_ext_yz)
