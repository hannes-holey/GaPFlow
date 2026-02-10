#
# Copyright 2026 Christoph Huber
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
import abc
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Tuple

import jax
import jax.numpy as jnp
from muGrid.Field import wrap_field

# jaxopt is deprecated, may switch to optax or similar
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jaxopt

from tinygp import GaussianProcess, kernels, transforms

JAXArray = jax.Array


class GaussianProcessSurrogate:
    """
    Abstract base class for Gaussian Process (GP) surrogate models.

    Implements GP training, inference, and active learning routines.
    Subclasses must define abstract properties that describe the kernel
    hyperparameters, data arrays, and noise models.

    """

    __metaclass__ = abc.ABCMeta

    # Expected subclass attributes
    name: str
    is_gp_model: bool
    active_dims: list[int]
    use_active_learning: bool
    rtol: float
    atol: float
    tol: str
    max_steps: int
    pause_steps: int
    params_init: dict
    noise: Tuple[float, float]
    prop: dict
    geo: dict

    def __init__(self, fc, database):
        """Constructor.

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field container with accessors for real fields such as 'solution' and 'topography'.
        database : GaPFlow.db.Database
            Training database providing `initialize`, `add_data`, and `size` attributes.
        """

        self._step = 0
        self.__solution = wrap_field(fc.get_real_field('solution'))
        self.__topo = wrap_field(fc.get_real_field('topography'))
        self.__extra = wrap_field(fc.get_real_field('extra'))

        if self.is_gp_model:
            self._cache = None
            self._database = database
            self._last_fit_train_size = 0
            self._pause = self.pause_steps

            # Initialize timers
            ref = datetime.now()
            self._cumtime_train = datetime.now() - ref
            self._cumtime_infer = datetime.now() - ref

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

    def init_database(self, dim: int) -> None:
        """Triggers the first database initialization.

        Parameters
        ----------
        dim : int
            Dimension of the fluid problem.
        """
        if self.is_gp_model:
            self._database.initialize(self._Xtest, dim)

    # ------------------------------------------------------------------
    # Abstract Properties (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def kernel_lengthscale(self):
        """Kernel lengthscale(s)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kernel_variance(self):
        """Kernel variance."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def obs_stddev(self):
        """Observation noise standard deviation."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Xtrain(self):
        """Training inputs (only active dimensions."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ytrain(self):
        """Training observations (only active dimensions, scaled)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Xtest(self):
        """Test inputs."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yscale(self):
        """Observations scaling factor."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yshift(self):
        """Observations shift."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yerr(self):
        """Observations standard error."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def database(self):
        """The database holding the training data for the GP surrogate model."""
        return self._database

    @property
    def last_fit_train_size(self):
        """Size of the training database at the last fit."""
        return self._last_fit_train_size

    @property
    def solution(self):
        """Return full solution field."""
        return self.__solution.p

    @property
    def height_and_slopes(self) -> JAXArray:
        """Return the topography (height and gradients)."""
        return self.__topo.p[:3]

    @property
    def height(self):
        """Return the gap height field."""
        return self.__topo.p[0]

    @property
    def extra(self):
        """Return constant extra field, which can be used as additional input."""
        return self.__extra.p

    @property
    def trusted(self) -> bool:
        """Return True if model predictive variance is below tolerance."""
        return self.maximum_variance < self.variance_tol

    @property
    def cumtime_train(self):
        """Cumulative time spent for training of the GP (fitting hyperparameters)"""
        return self._cumtime_train

    @property
    def cumtime_infer(self):
        """Cumulative time spent for inference from the GP (making predictions)"""
        return self._cumtime_infer

    @property
    def _Xtest(self) -> JAXArray:
        """
        Flattened test input array from physical fields.
        """
        return jnp.vstack([
            self.solution,
            self.height_and_slopes,
            self.extra
        ]).reshape(self._database.num_features, -1).T

    @property
    def has_multi_output(self):
        return self.Ytrain.ndim > 1

    # ------------------------------------------------------------------
    # Logging and summary
    # ------------------------------------------------------------------

    def write(self) -> None:
        """Log current GP hyperparameters and diagnostics."""
        if self.is_gp_model:
            self.history['step'].append(self._step)
            self.history['database_size'].append(self._database.size)
            self.history['variance'].append(self.kernel_variance)
            self.history['obs_stddev'].append(self.obs_stddev)
            self.history['maximum_variance'].append(self.maximum_variance)
            self.history['variance_tol'].append(self.variance_tol)

            for i, l in enumerate(self.active_dims):
                self.history[f'lengthscale_{l}'].append(self.kernel_lengthscale[i])

    def _print_opt_summary(self, obj: float) -> None:
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
    def build_gp(self,
                 params: dict,
                 X: JAXArray,
                 yerr: float | JAXArray) -> GaussianProcess:
        """Default GP build method.

        Parameters
        ----------
        params : dict
            Dictionary with hyperparameters
        X : jax.Array
            Input data.
        yerr : jax.Array
            Observation noise (standard deviation).

        Returns
        -------
        tinygp.GaussianProcess
            Single-output GP model.
        """
        return multi_in_single_out(params, X, yerr)

    def _train(self, reason: int = 0) -> None:
        """
        Train the Gaussian process model via marginal likelihood maximization.

        Parameters
        ----------
        reason : int, optional
            Training reason code: `0` = database update, `1` = active learning.
        """
        self._last_fit_train_size = deepcopy(self._database.size)
        reasons = ['DB', "AL"]

        print('#' + 17 * '-' + f"GP TRAINING ({self.name.upper()})" + 17 * '-')
        print('# Timestep     :', self._step)
        print('# Reason       :', reasons[reason])
        print('# Database size:', self._database.size)

        @jax.jit
        def loss_so(params, X, Y, yerr):
            gp = self.build_gp(params, X, yerr)
            return -gp.log_probability(Y)

        @jax.jit
        def loss_mo(params, X, Y, yerr):
            gp = self.build_gp(params, X, yerr)
            obj = 0.
            for Yi in Y.T:
                obj -= gp.log_probability(Yi)
            return obj

        solver = jaxopt.ScipyMinimize(fun=loss_mo if self.has_multi_output else loss_so)
        soln = solver.run(self.params_init, X=self.Xtrain, Y=self.Ytrain, yerr=self.Yerr)

        self.gp = self.build_gp(soln.params, self.Xtrain, self.Yerr)

        obj = soln.state.fun_val
        self._print_opt_summary(obj)

        if self._step > 0:
            self.write()

        if reason == 0:
            print('#' + 50 * '-')

        # Delete cache to force inference step with new training data
        self._cache = None

    def _predict(self):
        if self.has_multi_output:
            return _predict_multi_output(self.gp, self.Ytrain, self.Xtest)
        else:
            return _predict_single_output(self.gp, self.Ytrain, self.Xtest)

    def _infer_mean(self) -> JAXArray:
        """Infer mean for new test inputs.

        Uses cached quantities of the previous inference step, if available.

        Returns
        -------
        JAXArray
            Predictive mean
        """

        if self._cache is None:
            m, _, alpha, noise = self._predict()
            self._cache = (alpha, noise)
        else:
            m = _repredict_mean(self.gp, self._cache, self.Xtest)

        predictive_mean = m.reshape(-1, *self.solution.shape[-2:]).squeeze() * self.Yscale + self.Yshift

        return predictive_mean

    def _infer_mean_var(self) -> Tuple[JAXArray, JAXArray]:
        """Infer mean and variance for new test inputs.

        Uses cached quantities of the previous inference step, if available.

        Returns
        -------
        JAXArray
            Predictive mean
        JAXArray
            Predictive variance
        """

        if self._cache is None:
            m, v, alpha, noise = self._predict()
            self._cache = (alpha, noise)
        else:
            m, v = _repredict_mean_var(self.gp, self._cache, self.Xtest)

        predictive_mean = m.reshape(-1, *self.solution.shape[-2:]).squeeze() * self.Yscale + self.Yshift
        predictive_var = v.reshape(-1, *self.solution.shape[-2:]).squeeze() * self.Yscale**2

        return predictive_mean, predictive_var

    def _infer(self,
               compute_var: bool = True) -> Tuple[JAXArray, JAXArray]:
        """
        Perform GP prediction on test data.

        Parameters
        ----------
        compute_var : bool
            Flag to re-compute predictive variance, default is True.

        Returns
        -------
        predictive_mean : jax.Array
            Predicted mean field.
        predictive_var : jax.Array
            Predicted variance field.
        """

        if compute_var:
            predictive_mean, self._predictive_var = self._infer_mean_var()
            self.maximum_variance = jnp.max(self._predictive_var)
            self.variance_tol = self._get_tolerance()
        else:
            predictive_mean = self._infer_mean()

        return predictive_mean, self._predictive_var

    # ------------------------------------------------------------------
    # Active Learning
    # ------------------------------------------------------------------
    def _get_tolerance(self):

        noise = self.Yerr * self.Yscale
        Y = self.Ytrain * self.Yscale + self.Yshift

        if self.tol == 'delta':
            Ys = jnp.max(Y) - jnp.min(Y)
        elif self.tol == 'absmax':
            Ys = jnp.max(jnp.abs(Y))
        elif self.tol == 'snr':
            Ys = jnp.mean(Y) / noise
        else:
            raise RuntimeError('No tolerance calculation configured.')

        std_tol = jnp.maximum(
            self.atol * noise,  # "lower bound", multiple of observation noise
            self.rtol * Ys  # grows with Ys,
        )

        variance_tol = std_tol**2

        return variance_tol

    def _active_learning(self, var: JAXArray) -> None:
        """
        Apply active learning by adding new point to the training database.

        Parameters
        ----------
        var : jax.Array
            Predictive variance field.
        """

        next_train_index = self._select_next_point(var)
        _Xnew = self._Xtest[next_train_index, :][None, :]
        self._database.add_data(_Xnew)

    def _select_next_point(self,
                           var: JAXArray,
                           similarity_check: bool = True) -> int:
        """
        Select new training point using maximum variance criterion.
        Returns flattened index

        Parameters
        ----------
        var : jax.Array
            Predictive variance field.
        similarity_check : bool, optional
            If true, check similarity between existing and new training points
            (and avoid too similar points). The default is True.
        """

        # from large to small
        sorted_indices = jnp.argsort(var, axis=None)[::-1]

        # start with largest variance (currently only implemented strategy)
        imax = sorted_indices[0]

        if similarity_check:
            skipped = 0
            min_similarity = 1.

            for i in sorted_indices:
                Xnew = self.Xtest[i][None, :]
                similarity_score = self.gp.kernel(Xnew, self.Xtrain) / self.kernel_variance
                if jnp.any(jnp.isclose(similarity_score, 1., rtol=0., atol=1e-8)):
                    # Too similar point exists
                    skipped += 1
                    min_similarity = min(min_similarity, jnp.min(similarity_score))
                    continue
                else:
                    # Found suitable test point
                    imax = i
                    break

            if skipped > 0 and skipped < var.size:
                print(f"Skipped {skipped} largest variance points ({min_similarity})")

        return imax

    # ------------------------------------------------------------------
    # Main Predict/Active Loop
    # ------------------------------------------------------------------

    def predict(self,
                predictor: bool = True,
                compute_var: bool = True) -> Tuple[JAXArray, JAXArray]:
        """
        Perform GP prediction, optionally updating the model via active learning
        (only in predictor step of the predictor-corrector time integration scheme)

        Parameters
        ----------
        predictor : bool
            Whether to perform active learning updates (only in predictor step, default is True).
        compute_var : bool
            If true (default), preditive variance is re-computed.

        Returns
        -------
        m : jax.Array
            Predictive mean.
        v : jax.Array
            Predictive variance.
        """

        # Update hyperparameters
        if predictor:
            self._step += 1
            self._pause = max(-1, self._pause - 1)
            if self._last_fit_train_size < self._database.size:
                tic = datetime.now()
                self._train(reason=0)
                toc = datetime.now()
                self._cumtime_train += toc - tic

        tic = datetime.now()
        m, v = self._infer(compute_var=compute_var and predictor)
        toc = datetime.now()
        self._cumtime_infer += toc - tic

        if self.use_active_learning \
                and predictor \
                and self._pause < 0:

            counter = 0
            before = deepcopy(self.maximum_variance / self.variance_tol)

            # Active learning loop
            while not self.trusted and counter < self.max_steps:
                counter += 1
                self._active_learning(v)

                # retrain
                tic = datetime.now()
                self._train(reason=1)
                toc = datetime.now()
                self._cumtime_train += toc - tic

                # predict again
                tic = datetime.now()
                m, v = self._infer(compute_var=True)
                toc = datetime.now()
                self._cumtime_infer += tic - toc

                after = self.maximum_variance / self.variance_tol
                print(f"# AL {counter:2d}/{self.max_steps:2d}     : {before:.3f} --> {after:.3f}")
                print('#' + 50 * '-')

            if counter == self.max_steps and after > 1.:
                print("# Active learning loop missed uncertainty threshold")
                print(f"# Pause for {self.pause_steps} steps...")
                print('#' + 50 * '-')
                self._pause = self.pause_steps

        return m, v


@jax.jit
def _repredict_mean_var(gp: GaussianProcess,
                        cache: tuple,
                        Xtest: JAXArray) -> Tuple[JAXArray, JAXArray]:

    alpha, noise = cache

    Ks = gp.kernel(gp.X, Xtest)
    mean = Ks.T @ alpha
    v = gp.solver.solve_triangular(Ks)
    Kss = gp.kernel(Xtest) + noise
    var = Kss - jnp.sum(v**2, axis=0)

    return mean.T, var


@jax.jit
def _repredict_mean(gp: GaussianProcess,
                    cache: tuple,
                    Xtest: JAXArray) -> Tuple[JAXArray, JAXArray] | JAXArray:

    alpha, _ = cache

    Ks = gp.kernel(gp.X, Xtest)
    mean = Ks.T @ alpha

    return mean.T


@jax.jit
def _predict_multi_output(gp: GaussianProcess,
                          Ytrain: JAXArray,
                          Xtest: JAXArray) -> Tuple[JAXArray, JAXArray, GaussianProcess]:

    Ytest = []
    alpha = []

    for Yi in Ytrain.T:
        cond_gp = gp.condition(Yi, Xtest).gp
        Ytest.append(cond_gp.loc)
        alpha.append(cond_gp.mean_function.alpha)

    Ytest = jnp.array(Ytest)
    alpha = jnp.array(alpha).T
    noise = cond_gp.noise.diag
    Yvar = cond_gp.variance

    return Ytest, Yvar, alpha, noise


@jax.jit
def _predict_single_output(gp: GaussianProcess,
                           Ytrain: JAXArray,
                           Xtest: JAXArray) -> Tuple[JAXArray, JAXArray, GaussianProcess]:

    cond_gp = gp.condition(Ytrain, Xtest).gp
    alpha = cond_gp.mean_function.alpha
    noise = cond_gp.noise.diag
    Ytest = cond_gp.loc
    Yvar = cond_gp.variance

    return Ytest, Yvar, alpha, noise


# ----------------------------------------------------------------------
# Utility kernel builders
# ----------------------------------------------------------------------
def multi_in_single_out(params: dict,
                        X: JAXArray,
                        yerr: float | JAXArray) -> GaussianProcess:
    """
    Build a single-output GP with anisotropic Matérn kernel.

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
