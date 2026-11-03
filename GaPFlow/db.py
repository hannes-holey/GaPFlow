#
# Copyright 2026 Dan Waxman
#           2025 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import re
import dtoolcore
from ruamel.yaml import YAML
from dtool_lookup_api import query
from typing import Any
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float
from jax import Array
from scipy.stats import qmc

from .utils import progressbar
from .logging import get_logger

logger = get_logger("gapflow.md")

ArrayX = Float[Array, "Ntrain Nfeat"]   # Input features
ArrayY = Float[Array, "Ntrain 13"]  # Output features

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)

# Schema version of the dtool README layout (bump when the meaning/shape of
# the stored 'X' columns changes, and add a migration step to
# GaPFlow/cli/migrate_readme.py).
README_SCHEMA_VERSION = 1

# Maps human-readable names to base feature indices.
# Index layout: 0-2 solution (rho, jx, jy), 3-5 topography (h, dhdx, dhdy),
#               6-7 wall velocities (U, V), 8+ extra features.
_FEATURE_NAMES = {
    'rho': 0, 'jx': 1, 'jy': 2,
    'h': 3, 'dhdx': 4, 'dhdy': 5,
    'U': 6, 'V': 7,
}


def _eval_derived_expr(expr: str, features: Array) -> Array:
    """Evaluate a derived feature expression against a feature array.

    ``features`` must be indexed along its first axis (feature index), e.g.
    shape ``(num_features, N)`` for a batch or ``(num_features, Nx, Ny)`` for
    spatial fields.

    Supports named tokens (rho, jx, jy, h, dhdx, dhdy, U, V) and index
    notation ``f[i]`` for extra features (i >= 8). Any valid Python arithmetic
    expression using ``jnp`` is allowed.

    Examples: ``"U / h"``, ``"f[8] / h"``, ``"1. / rho"``,
              ``"(jx**2 + jy**2)**0.5 / rho"``
    """
    code = re.sub(r'f\[(\d+)\]', r'features[\1]', expr)
    for name, idx in _FEATURE_NAMES.items():
        code = re.sub(rf'\b{name}\b', f'features[{idx}]', code)
    return eval(code, {"__builtins__": {}}, {"features": features, "jnp": jnp})


def _check_readme_version(rm: dict, source: str = "dataset") -> None:
    """Fail loudly if a loaded README is not in the schema this code expects.

    READMEs are versioned via a top-level 'schema_version' tag (missing means
    version 0, i.e. predating the tag). This code does not attempt to migrate
    them on the fly; run the 'gpf_migrate_readme' CLI script on the dataset's
    dtool base URI first.
    """
    version = rm.get('schema_version', 0)

    if version != README_SCHEMA_VERSION:
        raise ValueError(
            f"README for '{source}' has schema_version {version}, but this "
            f"installation expects {README_SCHEMA_VERSION}. Run "
            "'gpf_migrate_readme' on the dataset's dtool base URI before loading it."
        )


class Database:
    """
    Container for GP training datasets.

    Handles dataset initialization, normalization, data addition,
    and optional dtool integration for persistent dataset storage.

    Parameters
    ----------
    md : GaPFlow.md.MolecularDynamics
        An instance of the MD runner object. Adding a data point will lead to calling its `run` method.
    db : dict
        Configuration dictionary with keys:

        - ``'dtool_path'`` : str, path where training data is stored and loaded from.
        - ``'init_size'`` : int, minimum dataset size.
        - ``'init_width'`` : float, relative sampling width.
        - ``'init_method'`` : str, name of the (quasi-)random initialization method ('lhc', 'rand', 'sobol').
        - ``'init_seed'`` : int, random seed for initialization.
    num_extra_features : int, number of additional features (next to solution, gap height + gradients)
        stored with the database (default is 1)

    """

    def __init__(
        self,
        md: Any,
        db: dict,
        num_extra_features: int = 0,
        num_derived_features: int = 0,
        derived_expressions: list[str] = []
    ) -> None:

        self._md = md
        self._db = db
        self._derived_expressions = derived_expressions

        self._output_path = None
        _training_path = db.get('dtool_path')

        if _training_path is not None:
            self._temporary_training_path = False

            if isinstance(_training_path, list):
                _training_paths = _training_path
            else:
                _training_paths = [_training_path]

            readme_list = []
            for path in _training_paths:
                readme_list.extend(self.get_readme_list_local(path))

            self.set_training_path(_training_paths[0])
        else:
            self._temporary_training_path = True
            self.set_training_path('/tmp/')
            readme_list = []

        if len(readme_list) > 0:
            Xtrain, Ytrain, Yerr = [], [], []
            for rm in readme_list:
                Xtrain.append(jnp.array(rm["X"]))
                Ytrain.append(jnp.array(rm["Y"]))
                Yerr.append(jnp.array(rm["Yerr"]))

            Xtrain = jnp.array(Xtrain)
            Ytrain = jnp.array(Ytrain)
            Yerr = jnp.array(Yerr)

            # Infer num_extra_features from the loaded data so the database is
            # self-describing regardless of the constructor default.
            num_extra_features = Xtrain.shape[1] - 8

        else:
            # Empty arrays use base feature count; derived cols are computed on-the-fly.
            Xtrain = jnp.empty((0, 8 + num_extra_features))
            Ytrain = jnp.empty((0, 13))
            Yerr = jnp.empty((0, 13))

        self._num_extra_features = num_extra_features
        # 6 base features (rho, jx, jy, h, dhdx, dhdy)
        # + 2 wall velocities (U, V)
        # + extra features (e.g. slip length)
        self._num_base_features = 8 + num_extra_features
        # + derived features (computed combinations, never stored persistently)
        self._num_features = self._num_base_features + num_derived_features

        self._Xtrain = Xtrain
        self._Xtrain_target = Xtrain
        self._Ytrain = Ytrain
        self._Ytrain_err = Yerr

        if self.size == 0:
            input_norm = 'none'
            output_norm = 'none'
        else:
            input_norm = self._db['normalizer_X']
            output_norm = self._db['normalizer_Y']

        # Normalizer covers all features (base + derived).
        self._X_shift, self._X_scale = self._normalizer(
            self._augment_with_derived(self._Xtrain), mode=input_norm)
        self._Y_shift, self._Y_scale = self._normalizer(self._Ytrain, mode=output_norm)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def config(self) -> dict:
        """Configuration parameters of the database object."""
        return self._db

    @property
    def md_config(self) -> dict:
        """Configuration parameters of the attached MD runner object."""
        return self._md.params

    def _augment_with_derived(self, X: ArrayX) -> ArrayX:
        """Append derived feature columns to X (base features only).

        Accepts either a 1-D array of shape ``(num_base_features,)`` for a
        single sample, or a 2-D array of shape ``(N, num_base_features)``.
        Returns an array with ``num_derived`` extra columns appended.
        """
        if not self._derived_expressions:
            return X
        is_1d = X.ndim == 1
        if is_1d:
            X = X[jnp.newaxis, :]
        # features indexed along first axis so features[i] selects feature i.
        features = X.T
        derived_cols = jnp.vstack([
            _eval_derived_expr(expr, features)
            for expr in self._derived_expressions
        ]).T
        result = jnp.concatenate([X, derived_cols], axis=1)
        return result.squeeze(0) if is_1d else result

    @property
    def Xtrain(self) -> ArrayX:
        """Normalized input features of shape (Ntrain, Nfeat)."""
        return (self._augment_with_derived(self._Xtrain) - self._X_shift) / self.X_scale

    @property
    def Xtrain_target(self) -> ArrayX:
        """Normalized target input features of shape (Ntrain, Nfeat)."""
        return (self._augment_with_derived(self._Xtrain_target) - self._X_shift) / self.X_scale

    @property
    def Ytrain(self) -> ArrayY:
        """Normalized observations of shape (Ntrain, 13)."""
        return (self._Ytrain - self._Y_shift) / self.Y_scale

    @property
    def Ytrain_err(self) -> ArrayY:
        """Normalized observation error of shape (Ntrain, 13)."""
        return self._Ytrain_err / self.Y_scale

    @property
    def size(self) -> int:
        """Number of training samples currently stored."""
        return self._Xtrain.shape[0]

    @property
    def X_scale(self) -> ArrayX:
        """Normalization constants for input features."""
        return self._X_scale

    @property
    def Y_scale(self) -> ArrayY:
        """Normalization constants for observations"""
        return self._Y_scale

    @property
    def X_shift(self) -> ArrayX:
        """Normalization constants for input features."""
        return self._X_shift

    @property
    def Y_shift(self) -> ArrayY:
        """Normalization constants for observations"""
        return self._Y_shift

    @property
    def num_features(self) -> int:
        """Number of possible features, actual ones are selected from GP's active_dims."""
        return self._num_features

    @property
    def num_extra_features(self) -> int:
        """Number of extra (statically provided) features beyond the 8 base features."""
        return self._num_extra_features

    @property
    def has_mock_md(self) -> bool:
        """Flag that indicates whether the attached MD runner is a 'mock' object."""
        return self._md.is_mock

    @property
    def output_path(self) -> str:
        """Simulation output path"""
        return self._output_path

    @output_path.setter
    def output_path(self, path) -> None:
        """Simulation output path setter."""
        self._output_path = path

    @property
    def training_path(self) -> str:
        """Local storage location of dtool datasets."""
        return self._training_path

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_readme_list_local(self, training_path=None):
        """Get list of dtool README files for existing MD runs
        from a local directory.

        Returns
        -------
        list
            List of dicts containing the readme content
        """

        if training_path is None:
            training_path = self.training_path

        readme_list = []
        for ds in dtoolcore.iter_datasets_in_base_uri(training_path):
            rm = yaml.load(ds.get_readme_content())
            _check_readme_version(rm, source=ds.name)
            readme_list.append(rm)
            logger.info('- %s (%s)', ds.uuid, ds.name)

        logger.info("Loading %d local datasets in '%s'.", len(readme_list), training_path)

        return readme_list

    def get_readme_list_remote(self):
        """Get list of dtool README files for existing MD runs
        from a remote data server (via dtool_lookup_api)

        In the future, one should be able to pass a valid MongoDB
        query string to select data.

        Returns
        -------
        list
            List of dicts containing the readme content
        """

        # TODO: Pass a textfile w/ uuids or yaml with query string
        query_dict = {"readme.description": {"$regex": "Dummy"}}
        remote_ds_list = query(query_dict)

        remote_ds = [dtoolcore.DataSet.from_uri(ds['uri'])
                     for ds in progressbar(remote_ds_list,
                                           prefix="Loading remote datasets based on dtool query: ")]

        readme_list = []
        for ds in remote_ds:
            rm = yaml.load(ds.get_readme_content())
            _check_readme_version(rm, source=ds.name)
            readme_list.append(rm)

        return readme_list

    def set_training_path(self,
                          new_path: str,
                          check_temporary: bool = False) -> None:
        """Set training path.

        This modifies the storage location of dtool basepaths,
        also for the attached MD runner object.

        Parameters
        ----------
        new_path : str
            Training path
        """

        if check_temporary:
            overwrite = self._temporary_training_path
        else:
            overwrite = True

        if overwrite:
            if not os.path.exists(new_path):
                os.makedirs(new_path)

            self._training_path = new_path
            self._md._dtool_basepath = new_path
            self._db['dtool_path'] = new_path

    def _normalizer(self, x: ArrayX, mode) -> ArrayX:
        """Compute feature-wise normalization factors."""

        shift = jnp.zeros(x.shape[1])
        scale = jnp.ones(x.shape[1])

        if mode == 'max':
            scale = jnp.maximum(jnp.max(jnp.abs(x - shift), axis=0), 1e-12)
        elif mode == 'minmax':
            shift = jnp.min(x, axis=0)
            scale = jnp.maximum(jnp.max(x - shift, axis=0), 1e-12)
        elif mode == 'standard':
            shift = jnp.mean(x, axis=0)
            scale = jnp.maximum(jnp.std(x, axis=0), 1e-12)

        return shift, scale

    def write(self) -> None:
        """Write the dataset arrays to disk (if the simulation output path is specified)."""
        if self.output_path is not None:
            jnp.save(os.path.join(self.output_path, "Xtrain.npy"), self._Xtrain)
            jnp.save(os.path.join(self.output_path, "Ytrain.npy"), self._Ytrain)
            jnp.save(os.path.join(self.output_path, "Ytrain_err.npy"), self._Ytrain_err)

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------
    def initialize(
        self,
        Xtest: ArrayX,
        dim: int = 1
    ) -> ArrayX:
        """
        Initialize database.

        Parameters
        ----------
        Xtest : jax.Array
            Candidate test points of shape (n_test, 6).
        dim : int
            Dimension of the fluid problem (either 1 or 2, defaults to 1)
        """

        init_method = self._db['init_method']
        init_halfwidth = self._db['init_halfwidth']
        init_seed = self._db['init_seed']
        init_size = self._db['init_size']

        Nsample = init_size - self.size

        if Nsample > 0:
            logger.info("Database contains less than %d MD runs.", init_size)
            logger.info("Generate new training data in %s", self.training_path)

            if dim == 1:
                active = jnp.array([0, 1])
            else:
                active = jnp.array([0, 1, 2])

            central_values = jnp.mean(Xtest, axis=0)
            ref_values = jnp.array([central_values[0],
                                    jnp.hypot(central_values[1], central_values[2]),
                                    jnp.hypot(central_values[1], central_values[2])])[active]
            half_widths = jnp.array(init_halfwidth)[active]
            l_bounds = central_values[active] - half_widths * ref_values
            u_bounds = central_values[active] + half_widths * ref_values

            key = jr.key(init_seed)
            key, subkey = jr.split(key)

            if init_method == 'rand':
                _samples = _get_random_samples(subkey, Nsample, l_bounds, u_bounds)
            elif init_method == 'lhc':
                _samples = _get_lhc_samples(Nsample, l_bounds, u_bounds)
            elif init_method == 'sobol':
                _samples = _get_sobol_samples(Nsample, l_bounds, u_bounds)
                Nsample = _samples.shape[0]

            key, subkey = jr.split(key)
            choice = jr.choice(subkey, Xtest.shape[0], shape=(Nsample,), replace=False).tolist()

            Xnew = jnp.column_stack([
                jnp.hstack([_samples, jnp.zeros((Nsample, 1))]) if len(active) == 2 else _samples,  # rho, jx, jy
                Xtest[choice, 3:],  # h dh_dx dh_dy + ...
            ])

            self.add_data(Xnew)
        else:
            # Also write training data to file when no new data is added
            self.write()

    def add_data(
        self,
        Xnew: ArrayX,
    ) -> None:
        """
        Add new data entries to the database.

        Parameters
        ----------
        Xnew : jax.Array
            New samples of shape (Nnew, Nfeat).
        """
        size_before = self.size

        for Xi in Xnew:
            size_before += 1

            # Strip any derived feature columns — MD only needs base features.
            Xi_base = Xi[:self._num_base_features]

            self._Xtrain_target = jnp.vstack([self._Xtrain_target, Xi_base])

            X, Y, Ye = self._md.run(Xi_base, size_before)

            self._Xtrain = jnp.vstack([self._Xtrain, X])
            self._Ytrain = jnp.vstack([self._Ytrain, Y])
            self._Ytrain_err = jnp.vstack([self._Ytrain_err, Ye])

            self._X_shift, self._X_scale = self._normalizer(
                self._augment_with_derived(self._Xtrain), mode=self._db['normalizer_X'])
            self._Y_shift, self._Y_scale = self._normalizer(self._Ytrain, mode=self._db['normalizer_Y'])

        self.write()


def _get_random_samples(key, N, lo, hi):
    """Random samples

    Parameters
    ----------
    key : int
        Random seed
    N : int
        Number of samples
    lo : array-like
        Lower bounds
    hi : array-like
        Upper bounds

    Returns
    -------
    numpy.ndarray
        Scaled samples
    """
    dim = len(lo)
    samples = jr.uniform(
        key,
        shape=(N, dim),
        minval=lo[None, :],
        maxval=hi[None, :],
    )

    return samples


def _get_lhc_samples(N, lo, hi):
    """Latin hypercube sampler

    Parameters
    ----------
    N : int
        Number of samples
    lo : array-like
        Lower bounds
    hi : array-like
        Upper bounds

    Returns
    -------
    numpy.ndarray
        Scaled samples
    """

    dim = len(lo)
    sampler = qmc.LatinHypercube(d=dim)
    sample = sampler.random(n=N)
    scaled_samples = qmc.scale(sample, lo, hi)

    return scaled_samples


def _get_sobol_samples(N, lo, hi):
    """Sobol sampler

    Parameters
    ----------
    N : int
        Number of samples
    lo : array-like
        Lower bounds
    hi : array-like
        Upper bounds

    Returns
    -------
    numpy.ndarray
        Scaled samples
    """

    dim = len(lo)
    sampler = qmc.Sobol(d=dim)
    m = int(jnp.log2(N))
    if int(2**m) != N:
        m = int(jnp.ceil(jnp.log2(N)))
        logger.info("Sample size should be a power of 2 for Sobol sampling. Use Ninit=%d.", 2**m)
    sample = sampler.random_base2(m=m)
    scaled_samples = qmc.scale(sample, lo, hi)

    return scaled_samples
