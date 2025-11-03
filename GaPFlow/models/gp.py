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
import abc
import warnings
import numpy as np
from copy import deepcopy
from datetime import datetime
from typing import Tuple, List

import jax
import jax.numpy as jnp

# jaxopt is deprecated, may switch to optax or similar
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jaxopt

from tinygp import GaussianProcess, kernels, transforms

JAXArray = jax.Array


class MultiOutputKernel(kernels.Kernel):
    """
    Multi-output Gaussian process kernel.

    Combines multiple latent kernels and a projection matrix to handle
    vector-valued (multi-output) functions.

    Parameters
    ----------
    kernels : list of tinygp.kernels.Kernel or tinygp.transforms.Linear
        List of kernels corresponding to latent GPs.
    projection : jax.Array, shape (num_outputs, num_latents)
        Projection matrix mapping latent processes to outputs.

    Methods
    -------
    evaluate(X1, X2)
        Evaluate the kernel covariance between input sets `X1` and `X2`.
    """

    kernels: List[kernels.Kernel | transforms.Linear]
    projection: JAXArray  # shape = (num_outputs, num_latents)

    def evaluate(self,
                 X1: Tuple[JAXArray, JAXArray],
                 X2: Tuple[JAXArray, JAXArray],
                 ) -> JAXArray:
        """
        Compute the covariance matrix between two sets of inputs.

        Parameters
        ----------
        X1 : tuple
            Tuple `(t1, idx1)` where `t1` is input array and `idx1` 
            indexes the corresponding outputs.
        X2 : tuple
            Tuple `(t2, idx2)` analogous to `X1`.

        Returns
        -------
        jax.Array
            Covariance matrix computed via the kernel projections.
        """
        t1, idx1 = X1
        t2, idx2 = X2
        latents = jnp.stack([k.evaluate(t1, t2) for k in self.kernels])
        return (self.projection[idx1] * self.projection[idx2]) @ latents


class GaussianProcessSurrogate:
    """
    Abstract base class for Gaussian Process (GP) surrogate models.

    Implements GP training, inference, and active learning routines.
    Subclasses must define abstract properties that describe the kernel 
    hyperparameters, data arrays, and noise models.

    Parameters
    ----------
    fc : muGrid.GlobalFieldCollection
        Field container with accessors for real fields such as 'solution' and 'gap'.
    database : GaPFlow.db.Database
        Training database providing `initialize`, `add_data`, and `size` attributes.

    Attributes
    ----------
    step : int
        Current training step.
    history : dict
        Stores GP hyperparameter evolution over time.
    cumtime_train, cumtime_infer : datetime.timedelta
        Cumulative training and inference times.
    """

    __metaclass__ = abc.ABCMeta

    # Expected subclass attributes
    name: str
    is_gp_model: bool
    active_dims: list[int]
    build_gp: callable
    rtol: float
    atol: float
    max_steps: int
    pause_steps: int
    params_init: dict
    noise: Tuple[float, float]
    prop: dict
    geo: dict

    def __init__(self, fc, database):
        self.step = 0
        self.__solution = fc.get_real_field('solution')
        self.__topo = fc.get_real_field('topography')
        self.__extra = fc.get_real_field('extra')

        if self.is_gp_model:
            self.database = database
            self.last_fit_train_size = 0
            self.pause = 0

            # Initialize timers
            ref = datetime.now()
            self.cumtime_train = datetime.now() - ref
            self.cumtime_infer = datetime.now() - ref

            # History of hyperparameters
            self.history = {
                'step': [],
                'database_size': [],
                'variance': [],
                'obs_stddev': [],
                'maximum_variance': [],
                'variance_tol': []
            }

            for li in self.active_dims:
                self.history[f'lengthscale_{li}'] = []

    def init_database(self, dim):
        if self.is_gp_model:
            self.database.initialize(self._Xtest, dim)

    # ------------------------------------------------------------------
    # Abstract Properties (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def kernel_lengthscale(self):
        """Return kernel lengthscale(s)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kernel_variance(self):
        """Return kernel variance."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def obs_stddev(self):
        """Return observation noise standard deviation."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Xtrain(self):
        """Return training inputs."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ytrain(self):
        """Return training outputs."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Xtest(self):
        """Return test inputs."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yscale(self):
        """Return output scaling factor."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yerr(self):
        """Return training observation error."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def solution(self):
        """Return full solution field."""
        return self.__solution.p

    @property
    def topography(self) -> JAXArray:
        """Return the topography (height and gradients)."""
        return self.__topo.p

    @property
    def extra(self):
        """Return extra constant field, which can be used as input."""
        return self.__extra.p

    @property
    def height(self):
        return self.__topo.p[0]

    @property
    def dh_dx(self):
        return self.__topo.p[1]

    @property
    def dh_dy(self):
        return self.__topo.p[2]

    @property
    def trusted(self) -> bool:
        """Return True if model predictive variance is below tolerance."""
        return self.maximum_variance < self.variance_tol

    # ------------------------------------------------------------------
    # Logging and summary
    # ------------------------------------------------------------------
    def write(self) -> None:
        """Log current GP hyperparameters and diagnostics."""
        if self.is_gp_model:
            self.history['step'].append(self.step)
            self.history['database_size'].append(self.database.size)
            self.history['variance'].append(self.kernel_variance)
            self.history['obs_stddev'].append(self.obs_stddev)
            self.history['maximum_variance'].append(self.maximum_variance)
            self.history['variance_tol'].append(self.variance_tol)

            for i, l in enumerate(self.active_dims):
                self.history[f'lengthscale_{l}'].append(self.kernel_lengthscale[i])

    def print_opt_summary(self, obj: float) -> None:
        """Print summary of optimization results."""
        print(f'# Objective    : {obj:.5g}')
        print("# Hyperparam   :", end=' ')
        print(f"{self.kernel_variance:.5e}", end=' ')
        print(f"{self.obs_stddev:.5e}", end=' ')
        for li in self.kernel_lengthscale:
            print(f"{li:.5e}", end=' ')
        print()

    # ------------------------------------------------------------------
    # Training and Inference
    # ------------------------------------------------------------------
    def _train(self, reason: int = 0) -> None:
        """
        Train the Gaussian process model via marginal likelihood maximization.

        Parameters
        ----------
        reason : int, optional
            Training reason code: `0` = database update, `1` = active learning.
        """
        self.last_fit_train_size = deepcopy(self.database.size)
        reasons = ['DB', "AL"]

        print('#' + 17 * '-' + f"GP TRAINING ({self.name.upper()})" + 17 * '-')
        print('# Timestep     :', self.step)
        print('# Reason       :', reasons[reason])
        print('# Database size:', self.database.size)

        @jax.jit
        def loss(params, X, yerr):
            return -self.build_gp(params, X, yerr).log_probability(self.Ytrain)

        solver = jaxopt.ScipyMinimize(fun=loss)
        soln = solver.run(self.params_init, X=self.Xtrain, yerr=self.Yerr)
        self.params = soln.params
        self.gp = self.build_gp(self.params, self.Xtrain, self.Yerr)

        obj = -self.gp.log_probability(self.Ytrain)
        self.print_opt_summary(obj)

        if self.step > 0:
            self.write()

        if reason == 0:
            print('#' + 50 * '-')

    def _infer(self) -> Tuple[JAXArray, JAXArray]:
        """
        Perform GP prediction on test data.

        Returns
        -------
        predictive_mean : jax.Array
            Predicted mean field.
        predictive_var : jax.Array
            Predicted variance field.
        """
        m, v = self.gp.predict(self.Ytrain, self.Xtest, return_var=True)
        _, nx, ny = self.solution.shape

        predictive_mean = m.reshape(-1, nx, ny).squeeze() * self.Yscale
        predictive_var = v.reshape(-1, nx, ny).squeeze() * self.Yscale**2

        self.variance_tol = jnp.maximum(
            self.atol * self.Yerr * self.Yscale, self.rtol * self.Yscale
        ) ** 2
        self.maximum_variance = np.max(predictive_var)

        return predictive_mean, predictive_var

    # ------------------------------------------------------------------
    # Active Learning
    # ------------------------------------------------------------------
    def _active_learning(self, var: JAXArray) -> None:
        """
        Select new training point using maximum variance criterion.

        Parameters
        ----------
        var : jax.Array
            Predictive variance field.
        """
        imax = np.argmax(var)
        Xnew = self._Xtest[imax, :][None, :]
        self.database.add_data(Xnew)

    # ------------------------------------------------------------------
    # Main Predict/Active Loop
    # ------------------------------------------------------------------
    def predict(self, predictor: bool = True) -> Tuple[JAXArray, JAXArray]:
        """
        Perform GP prediction, optionally updating the model via active learning
        (only in predictor step of the predictor-corrector time integration scheme)

        Parameters
        ----------
        predictor : bool, default=True
            Whether to perform active learning updates.

        Returns
        -------
        m : jax.Array
            Predictive mean.
        v : jax.Array
            Predictive variance.
        """
        if predictor:
            self.step += 1
            self.pause = max(-1, self.pause - 1)
            if self.last_fit_train_size < self.database.size:
                tic = datetime.now()
                self._train(reason=0)
                toc = datetime.now()
                self.cumtime_train += toc - tic

        tic = datetime.now()
        m, v = self._infer()
        toc = datetime.now()
        self.cumtime_infer += toc - tic

        # Active learning loop
        if predictor and self.pause < 0:
            counter = 0
            before = deepcopy(self.maximum_variance / self.variance_tol)
            while not self.trusted and counter < self.max_steps:
                counter += 1
                self._active_learning(v)

                # retrain
                tic = datetime.now()
                self._train(reason=1)
                toc = datetime.now()
                self.cumtime_train += toc - tic

                # predict again
                m, v = self._infer()
                tic = datetime.now()
                self.cumtime_infer += tic - toc

                after = self.maximum_variance / self.variance_tol
                print(f"# AL {counter:2d}/{self.max_steps:2d}     : {before:.3f} --> {after:.3f}")
                print('#' + 50 * '-')

            if counter == self.max_steps:
                print("# Active learning loop missed uncertainty threshold")
                print(f"# Pause for {self.pause_steps} steps...")
                print('#' + 50 * '-')
                self.pause = self.pause_steps

        return m, v

    # ------------------------------------------------------------------
    # Helper Property
    # ------------------------------------------------------------------
    @property
    def _Xtest(self) -> JAXArray:
        """
        Construct flattened test input array from physical fields.

        Returns
        -------
        jax.Array
            Test inputs of shape (n_samples, 6).
        """
        return jnp.vstack([
            self.solution,
            self.topography,
            self.extra
        ]).reshape(self.database.num_features, -1).T


# ----------------------------------------------------------------------
# Utility kernel builders
# ----------------------------------------------------------------------
def multi_in_single_out(params: dict,
                        X: JAXArray,
                        yerr: float | JAXArray) -> GaussianProcess:
    """
    Build a single-output anisotropic Matern GP.

    Parameters
    ----------
    params : dict
        Dictionary with kernel hyperparameters. Must contain:
        - ``log_amp`` : logarithm of amplitude.
        - ``log_scale`` : logarithm of length scale.
    X : jax.Array
        Input data.
    yerr : float or jax.Array
        Observation noise standard deviation.

    Returns
    -------
    tinygp.GaussianProcess
        Configured single-output GP model.
    """
    kernel = jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.exp(-params["log_scale"]),
        kernels.stationary.Matern32(distance=kernels.distance.L2Distance()),
    )
    return GaussianProcess(kernel, X, diag=yerr**2)


def multi_in_multi_out(params: dict,
                       X: JAXArray,
                       yerr: float | JAXArray) -> GaussianProcess:
    """
    Build a multi-output anisotropic Matern GP.

    Parameters
    ----------
    params : dict
        Dictionary with kernel hyperparameters (same as for single-output).
    X : jax.Array
        Input data.
    yerr : float or jax.Array
        Observation noise standard deviation.

    Returns
    -------
    tinygp.GaussianProcess
        Multi-output Gaussian process using `MultiOutputKernel`.
    """
    k = [
        jnp.exp(params["log_amp"]) * transforms.Linear(
            jnp.exp(-params["log_scale"]),
            kernels.stationary.Matern32(distance=kernels.distance.L2Distance()),
        )
        for _ in range(2)
    ]

    kernel = MultiOutputKernel(kernels=k, projection=jnp.eye(2))
    return GaussianProcess(kernel, X, diag=yerr**2)
