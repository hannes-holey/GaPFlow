import abc
import warnings
import numpy as np
from copy import deepcopy
from datetime import datetime

import jax
import jax.numpy as jnp

# jaxopt is no longer maintained
# may switch to optax or other optimization library
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import jaxopt

from tinygp import GaussianProcess, kernels, transforms


class MultiOutputKernel(kernels.Kernel):
    kernels: list[kernels.Kernel | transforms.Linear]
    projection: jax.Array  # shape = (num_classes, num_latents)

    def evaluate(self, X1, X2):
        t1, idx1 = X1
        t2, idx2 = X2
        latents = jnp.stack([k.evaluate(t1, t2) for k in self.kernels])
        return (self.projection[idx1] * self.projection[idx2]) @ latents


class GaussianProcessSurrogate:

    __metaclass__ = abc.ABCMeta

    name: str
    is_gp_model: bool
    active_dims: list[int]
    build_gp: callable
    rtol: float
    atol: float
    max_steps: int
    params_init: dict
    noise: float
    prop: dict
    geo: dict

    def __init__(self, fc, database):

        self.step = 0

        self.__solution = fc.get_real_field('solution')
        self.__gap = fc.get_real_field('gap')

        if self.is_gp_model:
            self.database = database
            self.database.fill_missing(self._Xtest, self.prop, self.geo, self.noise)
            self.last_fit_train_size = self.database.size

            ref = datetime.now()
            self.cumtime_train = datetime.now() - ref
            self.cumtime_infer = datetime.now() - ref

            self.history = {'step': [],
                            'database_size': [],
                            'variance': [],
                            'obs_stddev': [],
                            'maximum_variance': [],
                            'variance_tol': []}

            for li in self.active_dims:
                self.history[f'lengthscale_{li}'] = []

    @property
    @abc.abstractmethod
    def kernel_lengthscale(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def kernel_variance(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def obs_stddev(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Xtrain(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Ytrain(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Xtest(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yscale(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Yerr(self):
        raise NotImplementedError

    @property
    def mass_density(self):
        return self.__solution.p[0]

    @property
    def density(self):
        return self.__solution.p

    @property
    def gap(self):
        return self.__gap.p

    @property
    def trusted(self):
        return self.maximum_variance < self.variance_tol

    def write(self):
        if self.is_gp_model:
            self.history['step'].append(self.step)
            self.history['database_size'].append(self.database.size)
            self.history['variance'].append(self.kernel_variance)
            self.history['obs_stddev'].append(self.obs_stddev)
            self.history['maximum_variance'].append(self.maximum_variance)
            self.history['variance_tol'].append(self.variance_tol)

            for i, l in enumerate(self.active_dims):
                self.history[f'lengthscale_{l}'].append(self.kernel_lengthscale[i])

    def print_opt_summary(self, obj):

        print(f'# Objective    : {obj:.5g}')
        print("# Hyperparam   :", end=' ')

        print(f"{self.kernel_variance:.5e}", end=' ')
        print(f"{self.obs_stddev:.5e}", end=' ')

        for li in self.kernel_lengthscale:
            print(f"{li:.5e}", end=' ')

        print()

    def _train(self, reason=0):

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

        obj = self.gp.log_probability(self.Ytrain)

        self.print_opt_summary(obj)

        if self.step > 0:
            self.write()

        if reason == 0:
            print('#' + 50 * '-')

    def _infer(self):

        m, v = self.gp.predict(self.Ytrain, self.Xtest, return_var=True)

        _, nx, ny = self.density.shape

        predictive_mean = m.reshape(-1, nx, ny).squeeze() * self.Yscale
        predictive_var = v.reshape(-1, nx, ny).squeeze() * self.Yscale**2

        self.variance_tol = jnp.maximum(self.atol * self.Yerr * self.Yscale, self.rtol * self.Yscale)**2
        self.maximum_variance = np.max(predictive_var)

        return predictive_mean, predictive_var

    def _active_learning(self, var):

        imax = np.argmax(var)
        Xnew = self._Xtest[imax, :][:, None]

        self.database.add_data(Xnew, prop=self.prop, geo=self.geo, noise=self.noise)

    def predict(self, predictor=True):

        if predictor:
            self.step += 1
            if self.last_fit_train_size < self.database.size:
                tic = datetime.now()
                self._train(reason=0)
                toc = datetime.now()
                self.cumtime_train += toc - tic

        tic = datetime.now()
        m, v = self._infer()
        toc = datetime.now()
        self.cumtime_infer += toc - tic

        # Active learning
        if predictor:
            counter = 0
            before = deepcopy(self.maximum_variance / self.variance_tol)
            while not self.trusted and counter < self.max_steps:
                counter += 1

                # add new data
                self._active_learning(v)

                # train again
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

        return m, v

    @property
    def _Xtest(self):
        return jnp.vstack([self.gap,
                           self.density[0][None, :, :],
                           self.density[1][None, :, :],
                           self.density[2][None, :, :]  # * jnp.sign(self.density[2][None, :, :])
                           ]).reshape(6, -1).T


def multi_in_single_out(params, X, yerr):
    """
    Anisotropic Matern kernel, single output
    """

    kernel = jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.exp(-params["log_scale"]), kernels.stationary.Matern32(
            distance=kernels.distance.L2Distance())
    )

    return GaussianProcess(kernel, X, diag=yerr**2)


def multi_in_multi_out(params, X, yerr):
    """
    Anisotropic Matern kernel, multi output
    """

    k = [jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.exp(-params["log_scale"]),
        kernels.stationary.Matern32(
            distance=kernels.distance.L2Distance())
    ) for i in range(2)
    ]

    kernel = MultiOutputKernel(kernels=k,
                               projection=jnp.eye(2)
                               )

    return GaussianProcess(kernel, X, diag=yerr**2)
