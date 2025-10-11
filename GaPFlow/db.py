import os
from typing import Any
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float
from jax import Array

from GaPFlow.dtool import get_readme_list_local

# ----------------------------------------------------------------------
# Fixed-shape array type aliases
# ----------------------------------------------------------------------
ArrayX = Float[Array, "Ntrain 6"]   # Input features
ArrayY = Float[Array, "Ntrain 13"]  # Output features


class Database:
    """
    Container for GP training datasets.

    Handles dataset initialization, normalization, data addition,
    and optional dtool integration for persistent dataset storage.

    Parameters
    ----------
    db : dict
        Configuration dictionary with keys:
        - ``'dtool'`` : bool, use dtool datasets if True.
        - ``'init_size'`` : int, minimum dataset size.
        - ``'init_width'`` : float, relative sampling width.
    Xtrain : jax.Array, optional
        Initial training inputs of shape (Ntrain, 6).
    Ytrain : jax.Array, optional
        Initial training outputs of shape (Ntrain, 13).
    Ytrain_err : jax.Array, optional
        Observation errors of shape (Ntrain, 3).
    outdir : str or None, optional
        Directory for local dataset output.

    Attributes
    ----------
    outdir : str or None
        Output directory for saving arrays.
    minimum_size : int
        Minimum number of samples to maintain in the dataset.
    db_init_width : float
        Relative sampling width for database initialization.
    X_scale, Y_scale : jax.Array
        Feature-wise normalization factors.
    """

    def __init__(
        self,
        output_path: str,
        md: Any,
        db: dict,
    ) -> None:

        self.output_path = output_path
        self.training_path = db.get('dtool_path')
        if self.training_path is None:
            self.training_path = os.path.join(self.output_path, "train")

        readme_list = get_readme_list_local(self.training_path)

        if len(readme_list) > 0:
            Xtrain, Ytrain, Yerr = [], [], []
            for rm in readme_list:
                Xtrain.append(jnp.array(rm["X"]))
                Ytrain.append(jnp.array(rm["Y"]))
                Yerr.append(jnp.array(rm["Yerr"]))

            Xtrain = jnp.array(Xtrain)
            Ytrain = jnp.array(Ytrain)
            Yerr = jnp.array(Yerr)
        else:
            print(f"Start with empty dtool database in {self.training_path}")
            Xtrain = jnp.empty((0, 6))
            Ytrain = jnp.empty((0, 13))
            Yerr = jnp.empty((0, 13))

        # self.minimum_size = init_size
        self.minimum_size = db['init_size']
        self.init_method = db["init_method"]
        self.init_width = db["init_width"]

        self.md = md
        self.md.dtool_basepath = self.training_path

        self._Xtrain = Xtrain
        self._Ytrain = Ytrain
        self._Ytrain_err = Yerr

        if self.size == 0:
            self.X_scale = jnp.ones((6,))
            self.Y_scale = jnp.ones((13,))
        else:
            self.X_scale = self.normalizer(self._Xtrain)  # shape=(6, )
            self.Y_scale = self.normalizer(self._Ytrain)  # shape=(13, )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def Xtrain(self) -> ArrayX:
        """Normalized input features of shape (Ntrain, 6)."""
        return self._Xtrain / self.X_scale

    @property
    def Ytrain(self) -> ArrayY:
        """Normalized output features of shape (Ntrain, 13)."""
        return self._Ytrain / self.Y_scale

    @property
    def Ytrain_err(self) -> ArrayY:
        """Normalized observation error of shape (Ntrain, 13)."""
        return self._Ytrain_err / self.Y_scale

    @property
    def size(self) -> int:
        """Number of training samples currently stored."""
        return self._Xtrain.shape[0]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def normalizer(self, x: ArrayX) -> ArrayX:
        """Compute feature-wise normalization factors."""
        return jnp.maximum(jnp.max(jnp.abs(x), axis=0), 1e-12)

    def write(self) -> None:
        """Write the dataset arrays to disk (if outdir is specified)."""
        if self.output_path is not None:
            jnp.save(os.path.join(self.output_path, "Xtrain.npy"), self._Xtrain)
            jnp.save(os.path.join(self.output_path, "Ytrain.npy"), self._Ytrain)
            jnp.save(os.path.join(self.output_path, "Ytrain_err.npy"), self._Ytrain_err)

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def fill_missing(
        self,
        Xtest: ArrayX,
    ) -> None:
        """
        Fill the database to reach the minimum required number of samples.

        Parameters
        ----------
        Xtest : ndarray
            Candidate test points of shape (n_test, 6).
        """
        num_missing = self.minimum_size - self.size
        Xnew = get_new_training_input(Xtest, Nsample=num_missing, width=self.init_width)

        self.add_data(Xnew)

    def add_data(
        self,
        Xnew: ArrayX,
    ) -> None:
        """
        Add new data entries to the database.

        Parameters
        ----------
        Xnew : jax.Array
            New samples of shape (n_new, 6).
        """
        size_before = self.size

        for X in Xnew:
            size_before += 1

            Y, Ye = self.md.run(X, size_before)

            self._Xtrain = jnp.vstack([self._Xtrain, X])
            self._Ytrain = jnp.vstack([self._Ytrain, Y])
            self._Ytrain_err = jnp.vstack([self._Ytrain_err, Ye])

            self.X_scale = self.normalizer(self._Xtrain)
            self.Y_scale = self.normalizer(self._Ytrain)

        self.write()


def get_new_training_input(
    Xtest: ArrayX,
    Nsample: int,
    width: float = 1e-2,
) -> ArrayX:
    """
    Generate new input samples for initial database population.

    Parameters
    ----------
    Xtest : jax.Array
        Candidate test points of shape (n_test, 6).
    Nsample : int
        Number of new samples to generate.
    width : float, optional
        Relative sampling width for density and flux components.

    Returns
    -------
    Xnew : jax.Array
        New input samples of shape (n_new, 6).
    """
    if Nsample > 0:
        jabs = jnp.hypot(jnp.mean(Xtest[:, 4]), jnp.mean(Xtest[:, 5]))
        rho = jnp.mean(Xtest[:, 3])

        l_bounds = jnp.array([(1.0 - width) * rho, 0.5 * jabs, -0.5 * jabs])
        u_bounds = jnp.array([(1.0 + width) * rho, 1.5 * jabs, +0.5 * jabs])

        key = jr.key(123)
        key, subkey = jr.split(key)
        choice = jr.choice(key, Xtest.shape[0], shape=(Nsample,), replace=False).tolist()

        scaled_samples = jr.uniform(
            key, shape=(Nsample, 3),
            minval=l_bounds[None, :],
            maxval=u_bounds[None, :],
        )

        Xnew = jnp.column_stack([
            Xtest[choice, :3],  # gap height and derivatives
            scaled_samples,     # rho, flux_x, flux_y
        ])
    else:
        Xnew = jnp.empty((0, 6))

    return Xnew
