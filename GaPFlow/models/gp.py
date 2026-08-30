#
# Copyright 2025-2026 Hannes Holey
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
from typing import Tuple, Deque

import jax
import jax.numpy as jnp
from muGrid.Field import wrap_field

# jaxopt is deprecated, may switch to optax or similar
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jaxopt

from tinygp import GaussianProcess, kernels, transforms

from ..db import _eval_derived_expr
from ..logging import get_logger
from ..utils import above_tolerance

logger = get_logger("gapflow.gp")

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
    fix_noise: bool
    tolerance_protocol: str
    rtol: float
    atol: float
    atol_reduction_factor: float
    tol_rmid: float
    tol_alpha: float
    max_steps: int
    pause_steps: int
    similarity_check: bool
    allowed_skips: int
    pause_on_high_residual: bool
    params_init: dict
    noise: Tuple[float, float]
    prop: dict
    geo: dict
    derived_expressions: list[str]

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
            self._pause = 0
            self._tol_ratio = 0.
            self._objective = jnp.inf

            first_derived_idx = 8 + database.num_extra_features
            if self.use_active_learning and any(d >= first_derived_idx for d in self.active_dims):
                raise ValueError(
                    f"Active learning cannot be used when derived features appear in "
                    f"active_dims (derived feature indices start at {first_derived_idx}). "
                    f"Set active_learning: false or remove derived feature indices from active_dims."
                )

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
                'variance_tol': [],
                'objective': []
            }

            for li in self.active_dims:
                self.history[f'lengthscale_{li}'] = []

            if self.fix_noise:
                self.func = multi_in_single_out
            else:
                self.func = multi_in_single_out_jitter

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
        """Return the core (jx, jy, rho) solution field, excluding any
        solver-internal extra state (e.g. cavitation theta, energy E)."""
        return self.__solution.pg[:3]

    @property
    def height_and_slopes(self) -> JAXArray:
        """Return the topography (height and gradients)."""
        return self.__topo.pg[:3]

    @property
    def height(self):
        """Return the gap height field."""
        return self.__topo.pg[0]

    @property
    def extra(self):
        """Return constant extra field, which can be used as additional input."""
        return self.__extra.pg

    @property
    def _derived(self) -> JAXArray:
        """Compute derived features from base feature expressions defined in config."""
        base = jnp.vstack([
            self.solution,
            self.height_and_slopes,
            jnp.full((1, *self.height.shape), self.geo['U_bot']),  # TODO: also add top wall velocities
            jnp.full((1, *self.height.shape), self.geo['V_bot']),
            self.extra
        ])
        return jnp.vstack([
            _eval_derived_expr(expr, base)[jnp.newaxis]
            for expr in self.derived_expressions
        ])

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
    def objective(self):
        """Optimization objective (negative marginal log likelihood)"""
        return self._objective

    @property
    def _Xtest(self) -> JAXArray:
        """Flattened test input array from physical fields."""
        parts = [
            self.solution,
            self.height_and_slopes,
            jnp.full((1, *self.height.shape), self.geo['U_bot']),  # TODO: top wall
            jnp.full((1, *self.height.shape), self.geo['V_bot']),
            self.extra,
        ]
        if self.derived_expressions:
            parts.append(self._derived)
        return jnp.vstack(parts).reshape(self._database.num_features, -1).T

    @property
    def Xtrain(self) -> JAXArray:
        """Training inputs (normalized)."""
        return self._database.Xtrain[:, self.active_dims]

    @property
    def Xtrain_target(self) -> JAXArray:
        """Training inputs (normalized)."""
        return self._database.Xtrain_target[:, self.active_dims]

    @property
    def has_multi_output(self):
        return self.Ytrain.ndim > 1

    # ------------------------------------------------------------------
    # Logging and summary
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Log current GP hyperparameters and diagnostics."""
        if self.is_gp_model:
            self.history['step'].append(self._step)
            self.history['database_size'].append(self._database.size)
            self.history['variance'].append(self.kernel_variance)
            self.history['obs_stddev'].append(self.obs_stddev)
            self.history['maximum_variance'].append(self.maximum_variance)
            self.history['objective'].append(self.objective)
            self.history['variance_tol'].append(self.variance_tol)

            for i, l in enumerate(self.active_dims):
                self.history[f'lengthscale_{l}'].append(self.kernel_lengthscale[i])

    def _print_opt_summary(self, params) -> None:
        """Print summary of optimization results."""

        msg = f"# Objective    : {self.objective:.5g}\n"

        msg += "# Hyperparam   : \n"
        scale = [f'{s:.5f}' for s in params['log_scale']]
        msg += f"# - Log scale  : {' '.join(scale)}\n"
        msg += f"# - Log amp    : {params['log_amp']:.5f}\n"
        msg += f"# - Log noise  : {jnp.log(self.obs_stddev**2):.5f}"

        if not self.fix_noise:
            msg += f"\n# - Log jitter : {params['log_jitter']:.5f}"

        logger.info(msg)

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
        return self.func(params, X, yerr)

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

        logger.info('#' + 17 * '-' + f"GP TRAINING ({self.name.upper()})" + 17 * '-')
        logger.info(f'# Timestep     : {self._step}')
        logger.info(f'# Reason       : {reasons[reason]}')
        logger.info(f'# Database size: {self._database.size}')

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

        self._objective = soln.state.fun_val
        self._print_opt_summary(soln.params)

        if self._step > 0:
            self.save_state()

        if reason == 0:
            logger.info('#' + 50 * '-')

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

            # Dummy value overwritten by active learning
            self.variance_tol = self.maximum_variance
        else:
            predictive_mean = self._infer_mean()

        return predictive_mean, self._predictive_var

    # ------------------------------------------------------------------
    # Active Learning
    # ------------------------------------------------------------------
    def _get_tolerance(self, m, residuals):
        """Compute the variance tolerance based on the current prediction.

        Parameters
        ----------
        m : jax.Array
            Predictive mean
        residual : jax.Array
            Current residual, used by the sigmoid and linear tolerance protocols.

        Returns
        -------
        jax.Array
            Maximum allowed variance tolerance
        """

        if residuals is not None:
            residual_mean = jnp.mean(jnp.array(residuals))

        noise = self.Yerr * self.Yscale
        atol = self.atol * noise  # "lower bound", multiple of observation noise

        if self.tolerance_protocol == 'rtol_delta':
            delta = jnp.max(m) - jnp.min(m)
            rtol = self.rtol * delta
            std_tol = jnp.maximum(atol, rtol)

        elif self.tolerance_protocol == 'sigmoid':
            atol_init = atol
            atol_final = self.atol_reduction_factor * atol
            x = self.tol_alpha * jnp.log(residual_mean / self.tol_rmid)
            std_tol = atol_final + (atol_init - atol_final) / (1. + jnp.exp(-x))

        elif self.tolerance_protocol == 'linear':
            atol_init = atol
            atol_final = self.atol_reduction_factor * atol
            x = jnp.clip(self.tol_alpha * jnp.log(residual_mean / self.tol_rmid), -1., 1.)
            std_tol = atol_final + (atol_init - atol_final) * (x + 1.) / 2.

        else:
            raise RuntimeError('No tolerance calculation configured.')

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

        Xnew = self._select_next_point(var, similarity_check=self.similarity_check)
        self._database.add_data(Xnew)

    def _select_next_point(self,
                           var: JAXArray,
                           similarity_check: bool = True) -> int:
        """
        Select new training point using maximum variance criterion.
        If `similarity_check=True`, we try to avoid points that are
        too similar to the existing database. If all candidate points
        are too close, we fall back to the point with largest variance.

        Parameters
        ----------
        var : jax.Array
            Predictive variance field.
        similarity_check : bool, optional
            If true, check similarity between existing and new training points
            (and avoid too similar points). The default is True.

        Returns
        -------
        jax.Array
            Selected point
        """

        # Consicer only interior cells (no ghosts)
        nx, ny = var.shape
        n_inner = (nx - 2) * (ny - 2)

        # normalized
        Xtest = self.Xtest.reshape(nx, ny, -1)[1:-1, 1:-1].reshape(n_inner, -1)
        # not normalized
        _Xtest = self._Xtest.reshape(nx, ny, -1)[1:-1, 1:-1].reshape(n_inner, -1)

        # from large to small
        sorted_indices = jnp.argsort(var[1:-1, 1:-1], axis=None)[::-1]

        # start with largest variance (currently only implemented strategy)
        selected = sorted_indices[0]

        if similarity_check:
            skipped = 0

            # Loop over candidate points
            for i in sorted_indices:
                Xnew = Xtest[i][None, :]

                # Similarity between selected point and all requested training points
                similarity_score = self.gp.kernel(Xnew, self.Xtrain_target) / self.kernel_variance
                smax = similarity_score.max()
                smin = similarity_score.min()

                if jnp.isclose(smax, 1., rtol=0., atol=1e-6):
                    # Too similar point exists already
                    skipped += 1
                    continue
                else:
                    # Found suitable test point
                    logger.info(f'New input selected (skipped {skipped}): {smax:.3e}, {smin:.3e} (max, min)')
                    selected = i
                    break

            if skipped > self.allowed_skips:
                logger.info('No suitable test point found. Using max. variance point.')

        # Test point from index
        _Xnew = _Xtest[selected, :][None, :]

        return _Xnew

    # ------------------------------------------------------------------
    # Main Predict/Active Loop
    # ------------------------------------------------------------------

    def predict(self,
                residuals: Deque | None = None,
                predictor: bool = True,
                compute_var: bool = True) -> Tuple[JAXArray, JAXArray]:
        """
        Perform GP prediction, optionally updating the model via active learning
        (only in predictor step of the predictor-corrector time integration scheme)

        Parameters
        ----------
        residuals : Deque
            Residual buffer of the main simulation loop.
        predictor : bool, optional
            Whether to perform active learning updates (only in predictor step, default is True).
        compute_var : bool, optional
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

        after_failed_attempt = self._pause >= 0

        if residuals is None:
            cooldown = False
        else:
            cooldown = above_tolerance(residuals, tol=1e-3) and self.pause_on_high_residual
        pause_acquisition = after_failed_attempt or cooldown

        if self.use_active_learning \
                and predictor \
                and not pause_acquisition:

            # Compute variance tolerance
            self.variance_tol = self._get_tolerance(m, residuals)
            before = deepcopy(self.maximum_variance / self.variance_tol)

            # Active learning loop
            counter = 0
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

                # Re-compute variance tolerance
                self.variance_tol = self._get_tolerance(m, residuals)

                # AL step output summary
                after = self.maximum_variance / self.variance_tol
                msg = f"# AL {counter:2d}/{self.max_steps:2d}     : {before:.3f} --> {after:.3f}"
                logger.info(msg)
                logger.info('#' + 50 * '-')

            if not self.trusted:  # AL loop failed
                logger.warning("# Active learning loop missed uncertainty threshold")
                logger.info(f"# Pause for {self.pause_steps} steps...")
                logger.info('#' + 50 * '-')
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


def multi_in_single_out_jitter(params: dict,
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

    diag = yerr**2 + jnp.exp(params['log_jitter'])

    return GaussianProcess(kernel, X, diag=diag)
