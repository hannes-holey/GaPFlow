import os
from typing import Tuple, Self
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float
from jax import Array

from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top
from GaPFlow.dtool import get_readme_list_local, init_dataset, write_readme
from GaPFlow.md import write_input_files, run_serial, run_parallel, read_output_files
from GaPFlow.utils import bordered_text

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
    use_dtool : bool
        Whether to use dtool for dataset storage.
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
        db: dict,
        md: dict | None = None,
        Xtrain: ArrayX = jnp.empty((0, 6)),
        Ytrain: ArrayY = jnp.empty((0, 13)),
        Ytrain_err: ArrayY = jnp.empty((0, 13)),
        outdir: str | None = None,
    ) -> None:

        self.use_dtool = db['dtool']
        self.outdir = outdir
        self.minimum_size = db['init_size']
        self.db_init_width = db['init_width']  # density only

        self.md = md
        # if self.system == "lj":
        #     self._stress_scale = 1.
        # else:  # moltemplate
        #     # Alkane/gold
        #     # from kcal/mol/A^3 to g/mol/A/fs^2
        #     self._stress_scale = float(self.md.get('scaleRM', sci.calorie * 1e-4))

        self._Xtrain = Xtrain
        self._Ytrain = Ytrain
        self._Ytrain_err = Ytrain_err

        if self.size == 0:
            self.X_scale = jnp.ones((6,))
            self.Y_scale = jnp.ones((13,))
        else:
            self.X_scale = self.normalizer(self._Xtrain)  # shape=(6, )
            self.Y_scale = self.normalizer(self._Ytrain)  # shape=(13, )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_numpy(cls, directory: str) -> Self:
        """
        Load a local NumPy-based dataset.

        Parameters
        ----------
        directory : str
            Directory containing ``Xtrain.npy`` and ``Ytrain.npy``.

        Returns
        -------
        Database
            Loaded database instance.
        """
        raise NotImplementedError

    @classmethod
    def from_dtool(cls, db: dict, md: dict | None, outdir: str) -> Self:
        """
        Load or initialize a dtool-based dataset.

        Parameters
        ----------
        db : dict
            Database configuration dictionary.
        outdir : str
            Output directory for dataset initialization.

        Returns
        -------
        Database
            Database instance linked to a dtool dataset.
        """
        cls.dtool_basepath = (
            db["dtool_path"]
            if db["dtool_path"] is not None
            else os.path.join(outdir, "train")
        )

        readme_list = get_readme_list_local(cls.dtool_basepath)

        if len(readme_list) > 0:
            Xtrain, Ytrain, Yerr = [], [], []
            for rm in readme_list:
                Xtrain.append(jnp.array(rm["X"]))
                Ytrain.append(jnp.array(rm["Y"]))
                Yerr.append(rm.get("Yerr", 13 * [0.]))

            Xtrain = jnp.array(Xtrain)
            Ytrain = jnp.array(Ytrain)
            Yerr = jnp.array(Yerr)
        else:
            print(f"Start with empty dtool database in {cls.dtool_basepath}")
            Xtrain = jnp.empty((0, 6))
            Ytrain = jnp.empty((0, 13))
            Yerr = jnp.empty((0, 13))

        return cls(db, md, Xtrain, Ytrain, Yerr, outdir)

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
        if self.outdir is not None:
            jnp.save(os.path.join(self.outdir, "Xtrain.npy"), self._Xtrain)
            jnp.save(os.path.join(self.outdir, "Ytrain.npy"), self._Ytrain)

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def fill_missing(
        self,
        Xtest: ArrayX,
        prop: dict | None = None,
        geo: dict | None = None,
        noise: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """
        Fill the database to reach the minimum required number of samples.

        Parameters
        ----------
        Xtest : ndarray
            Candidate test points of shape (n_test, 6).
        prop : dict, optional
            Physical properties used to generate outputs.
        geo : dict, optional
            Geometric parameters used to generate outputs.
        noise : tuple of float, optional
            (pressure_noise, stress_noise) standard deviations.
        """
        num_missing = self.minimum_size - self.size
        Xnew = get_new_training_input(Xtest, Nsample=num_missing, width=self.db_init_width)
        self.add_data(Xnew, prop=prop, geo=geo, noise=noise)

    def add_data(
        self,
        Xnew: ArrayX,
        prop: dict | None = None,
        geo: dict | None = None,
        noise: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """
        Add new data entries to the database.

        Parameters
        ----------
        Xnew : jax.Array
            New samples of shape (n_new, 6).
        prop : dict, optional
            Material property parameters.
        geo : dict, optional
            Geometry-related parameters.
        noise : tuple of float, optional
            Noise standard deviations for pressure and stress.
        """
        size_before = self.size

        for X in Xnew:
            size_before += 1

            if self.use_dtool:
                proto_ds, proto_ds_path = init_dataset(self.dtool_basepath, size_before)
                proto_ds_datapath = os.path.join(proto_ds_path, 'data')

            if self.md is None:
                Y = get_new_training_output_mock(X[None, :], prop, geo, noise_stddev=noise).squeeze()
                Ye = jnp.zeros((13,))
            else:
                write_input_files(X, proto_ds, self.md)  # into dtool datapath

                # Move to dtool directory
                basedir = os.getcwd()
                os.chdir(proto_ds_datapath)

                # Run MD
                pretty_print(proto_ds_path, X)

                if self.md['ncpu'] > 1:
                    run_parallel(self.md['ncpu'])
                else:
                    run_serial()

                # Read output
                Y, Ye = read_output_files()

                # Return to base directory
                os.chdir(basedir)

            # Finalize dtool dataset
            if self.use_dtool:
                write_readme(proto_ds_path, X, Y, Ye)
                proto_ds.freeze()

            self._Xtrain = jnp.vstack([self._Xtrain, X])
            self._Ytrain = jnp.vstack([self._Ytrain, Y])
            self._Ytrain_err = jnp.vstack([self._Ytrain_err, Ye])

            self.X_scale = self.normalizer(self._Xtrain)
            self.Y_scale = self.normalizer(self._Ytrain)

        self.write()


def pretty_print(proto_datapath, X):
    ascii_art = r"""
  _        _    __  __ __  __ ____  ____
 | |      / \  |  \/  |  \/  |  _ \/ ___|
 | |     / _ \ | |\/| | |\/| | |_) \___ \
 | |___ / ___ \| |  | | |  | |  __/ ___) |
 |_____/_/   \_\_|  |_|_|  |_|_|   |____/

"""
    text = ['Run next MD simulation in:', f'{proto_datapath}']
    text.append(ascii_art)
    text.append('---')
    for i, Xi in enumerate(X):
        text.append(f'Input {i+1}: {Xi:.3g}')
    print(bordered_text('\n'.join(text)))


# ----------------------------------------------------------------------
# Mock data generation
# ----------------------------------------------------------------------


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


def get_new_training_output_mock(
    X: ArrayX,
    prop: dict,
    geo: dict,
    noise_stddev: Tuple[float, float] = (0.0, 0.0),
) -> ArrayY:
    """
    Generate mock outputs based on constitutive models and additive noise.

    Parameters
    ----------
    X : jax.Array
        Input array of shape (Ntrain, 6).
    prop : dict
        Material properties with shear and bulk viscosities.
    geo : dict
        Geometric parameters such as velocities ``U`` and ``V``.
    noise_stddev : tuple of float, optional
        Standard deviations for (pressure, stress) noise.

    Returns
    -------
    Y : jax.Array
        Mock output data of shape (Ntrain, 13).
    """
    key = jr.key(123)
    key, subkey = jr.split(key)
    noise_p = jr.normal(key, shape=X.shape[0]) * noise_stddev[0]
    noise_s0 = jr.normal(key, shape=X.shape[0]) * noise_stddev[1]
    noise_s1 = jr.normal(key, shape=X.shape[0]) * noise_stddev[1]

    U, V = geo["U"], geo["V"]
    eta, zeta = prop["shear"], prop["bulk"]

    X = X.T
    tau_bot = stress_bottom(X[3:], X[:3], U, V, eta, zeta, 0.0) + noise_s0
    tau_top = stress_top(X[3:], X[:3], U, V, eta, zeta, 0.0) + noise_s1
    press = eos_pressure(X[3:4], prop) + noise_p

    return jnp.vstack([press, tau_bot, tau_top]).T
