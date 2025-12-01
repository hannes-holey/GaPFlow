#
# Copyright 2025 Hannes Holey
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

ArrayX = Float[Array, "Ntrain Nfeat"]   # Input features
ArrayY = Float[Array, "Ntrain 13"]  # Output features

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)


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
        num_extra_features: int = 1
    ) -> None:

        self._md = md
        self._db = db

        self._num_features = 6 + num_extra_features

        self._output_path = None
        _training_path = db.get('dtool_path')

        if _training_path is not None:
            self._temporary_training_path = False
            self.set_training_path(_training_path)
            readme_list = self.get_readme_list_local()
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

        else:
            Xtrain = jnp.empty((0, self.num_features))
            Ytrain = jnp.empty((0, 13))
            Yerr = jnp.empty((0, 13))

        self._Xtrain = Xtrain
        self._Ytrain = Ytrain
        self._Ytrain_err = Yerr

        if self.size == 0:
            self._X_scale = jnp.ones((self.num_features,))
            self._Y_scale = jnp.ones((13,))
        else:
            self._X_scale = self._normalizer(self._Xtrain)
            self._Y_scale = self._normalizer(self._Ytrain)

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

    @property
    def Xtrain(self) -> ArrayX:
        """Normalized input features of shape (Ntrain, Nfeat)."""
        return self._Xtrain / self.X_scale

    @property
    def Ytrain(self) -> ArrayY:
        """Normalized observations of shape (Ntrain, 13)."""
        return self._Ytrain / self.Y_scale

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
    def num_features(self) -> int:
        """Number of possible features, actual ones are selected from GP's active_dims."""
        return self._num_features

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

    def get_readme_list_local(self):
        """Get list of dtool README files for existing MD runs
        from a local directory.

        Returns
        -------
        list
            List of dicts containing the readme content
        """

        readme_list = [yaml.load(ds.get_readme_content())
                       for ds in dtoolcore.iter_datasets_in_base_uri(self.training_path)]

        print(f"Loading {len(readme_list)} local datasets in '{self.training_path}'.")
        for ds in dtoolcore.iter_datasets_in_base_uri(self.training_path):
            print(f'- {ds.uuid} ({ds.name})')

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

        readme_list = [yaml.load(ds.get_readme_content()) for ds in remote_ds]

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

    def _normalizer(self, x: ArrayX) -> ArrayX:
        """Compute feature-wise normalization factors."""
        return jnp.maximum(jnp.max(jnp.abs(x), axis=0), 1e-12)

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
        init_width = self._db['init_width']
        init_seed = self._db['init_seed']
        init_size = self._db['init_size']

        Nsample = init_size - self.size

        if Nsample > 0:
            print(f"Database contains less than {init_size} MD runs.")
            print(f"Generate new training data in {self.training_path}")

            if dim == 1:
                flux = jnp.mean(Xtest[:, 1])
                active = jnp.array([0, 1])
            else:
                flux = jnp.hypot(jnp.mean(Xtest[:, 1]), jnp.mean(Xtest[:, 2]))
                active = jnp.array([0, 1, 2])

            rho = jnp.mean(Xtest[:, 0])

            l_bounds = jnp.array([(1.0 - init_width) * rho,
                                  0.5 * flux,
                                  -0.5 * flux])[active]

            u_bounds = jnp.array([(1.0 + init_width) * rho,
                                  1.5 * flux,
                                  0.5 * flux])[active]

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

        for X in Xnew:
            size_before += 1

            Y, Ye = self._md.run(X, size_before)

            self._Xtrain = jnp.vstack([self._Xtrain, X])
            self._Ytrain = jnp.vstack([self._Ytrain, Y])
            self._Ytrain_err = jnp.vstack([self._Ytrain_err, Ye])

            self._X_scale = self._normalizer(self._Xtrain)
            self._Y_scale = self._normalizer(self._Ytrain)

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
        print(f'Sample size should be a power of 2 for Sobol sampling. Use Ninit={2**m}.')
    sample = sampler.random_base2(m=m)
    scaled_samples = qmc.scale(sample, lo, hi)

    return scaled_samples
