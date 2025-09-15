import abc
import numpy as np
from copy import deepcopy
from datetime import datetime

from GaPFlow.utils import get_new_training_input
from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top, stress_avg

import jax
import jax.numpy as jnp
import jax.random as jr
import jaxopt

from tinygp import GaussianProcess, kernels, transforms


class MultiOutputKernel(kernels.Kernel):
    kernels: list[kernels.Kernel, ...]
    projection: jax.Array  # shape = (num_classes, num_latents)

    def evaluate(self, X1, X2):
        t1, idx1 = X1
        t2, idx2 = X2
        latents = jnp.stack([k.evaluate(t1, t2) for k in self.kernels])
        return (self.projection[idx1] * self.projection[idx2]) @ latents


class GaussianProcessSurrogate:

    __metaclass__ = abc.ABCMeta

    name: str

    def __init__(self, fc, prop, database):

        self.prop = prop
        self.step = 0

        self.__solution = fc.get_real_field('solution')
        self.__gap = fc.get_real_field('gap')

        # should come from yaml
        # active_dims = [0, 3, 4, ]  # gap, density, flux_x
        # active_dims = [0, 3, 4, 5] # gap, density, flux_x, flux_y
        # active_dims = [0, 1, 3, 4] # gap, gradient, density, flux_x

        if self.is_gp_model:
            self.database = database
            Xnew = get_new_training_input(self._Xtest.T,
                                          self.database.minimum_size - self.database.size)

            # For mock data from known constitutive laws
            Ynew = get_new_training_output_mock(Xnew, prop, noise_stddev=self.noise)
            # ...or from MD
            # Ynew = get_new_training_output_MD(Xnew)

            self.database.add_data(Xnew.T, Ynew.T)

            self.last_fit_train_size = self.database.size

            ref = datetime.now()
            self.cumtime_train = datetime.now() - ref
            self.cumtime_infer = datetime.now() - ref

    @abc.abstractmethod
    def model_setup(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_training_data(self):
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
        pass

    def print_opt_summary(self, obj):

        print(f'# Objective    : {obj:.5g}')
        print("# Hyperparam   :", end=' ')

        print(f"{self.kernel_variance:.5e}", end=' ')
        print(f"{self.obs_stddev:.5e}", end=' ')

        for l in self.kernel_lengthscale:
            print(f"{l:.5e}", end=' ')

        print()

    def _train(self, opt='scipy', reason=0):

        self.last_fit_train_size = deepcopy(self.database.size)

        reasons = ['DB', "AL"]

        print('#' + 15 * '-' + f"GP TRAINING ({self.name.upper()})" + 16 * '-')
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

        if reason == 0:
            print('#' + 50 * '-')

    def _infer(self):

        m, v = self.gp.predict(self.Ytrain, self.Xtest, return_var=True)

        _, nx, ny = self.density.shape

        predictive_mean = m.reshape(-1, nx, ny).squeeze() * self.Yscale
        predictive_var = v.reshape(-1, nx, ny).squeeze() * self.Yscale**2

        self.variance_tol = (self.std_tol_norm * self.Yscale)**2
        self.maximum_variance = np.max(predictive_var)

        return predictive_mean, predictive_var

    def _active_learning(self, var):

        imax = np.argmax(var)

        Xnew = self._Xtest[imax, :][:, None]
        Ynew = get_new_training_output_mock(Xnew, self.prop,
                                            noise_stddev=self.noise)  # replace w/ MD call

        self.database.add_data(Xnew.T, Ynew.T)

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

        max_steps = 5  # TODO not hardcoded

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

                print(f"# AL {counter}/{max_steps}       : {before:.3f} --> {after:.3f}")
                print('#' + 50 * '-')

        return m, v

    @property
    def _Xtest(self):
        Xtest = jnp.vstack([self.gap,
                            self.density[0][None, :, :],
                            self.density[1][None, :, :],
                            self.density[2][None, :, :] * jnp.sign(self.density[2][None, :, :])
                            ]).reshape(6, -1).T

        return Xtest


def get_new_training_output_mock(X, prop, noise_stddev=0.):

    key = jr.key(123)
    key, subkey = jr.split(key)
    noise = jr.normal(key, shape=X.shape[1]) * noise_stddev

    # For MD data: call update method from database (or external)

    # Shear stress
    U = 0.1
    V = 0.
    eta = prop['shear']
    zeta = prop['bulk']

    tau_bot = stress_bottom(X[3:],  # q
                            X[:3],  # h, dhdx, dhdy
                            U, V, eta, zeta, 0.)

    tau_top = stress_top(X[3:],  # q
                         X[:3],  # h, dhdx, dhdy
                         U, V, eta, zeta, 0.)

    # Pressure
    press = eos_pressure(X[3], prop)[None, :] + noise

    return np.vstack([press,
                      tau_bot,
                      tau_top])


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
