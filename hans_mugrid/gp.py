from datetime import datetime
from hans_mugrid.utils import get_new_training_input
from hans_mugrid.models import dowson_higginson_pressure, stress_bottom, stress_top, stress_avg
import abc
import numpy as np
import jax.numpy as jnp
import jax.random as jr

from copy import deepcopy

from jax import config
from jaxtyping import install_import_hook
config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


class MultiOutputKernel(gpx.kernels.AbstractKernel):
    def __init__(
        self,
        kernel: gpx.kernels.AbstractKernel = gpx.kernels.Matern32(active_dims=[0, 1, 2],
                                                                  lengthscale=jnp.ones(3),
                                                                  variance=1.),
    ):
        self.kernel = kernel
        self.lengthscale = self.kernel.lengthscale
        self.variance = self.kernel.variance

        super().__init__()

    def __call__(self, X, Xp):
        # standard RBF-SE kernel is x and x' are on the same output, otherwise returns 0

        z = jnp.array(X[6], dtype=int)
        zp = jnp.array(Xp[6], dtype=int)

        # achieve the correct value via 'switches' that are either 1 or 0
        k0_switch = ((z + 1) % 2) * ((zp + 1) % 2)
        k1_switch = z * zp

        return k0_switch * self.kernel(X, Xp) + k1_switch * self.kernel(X, Xp)


class GaussianProcessSurrogate:

    __metaclass__ = abc.ABCMeta

    name: str

    def __init__(self, fc, database, gp_config, prop):

        self.prop = prop

        self.__solution = fc.get_real_field('solution')
        self.__gap = fc.get_real_field('gap')

        # should come from yaml
        active_dims = [0, 3, 4, ]  # gap, density, flux_x
        # active_dims = [0, 3, 4, 5] # gap, density, flux_x, flux_y
        # active_dims = [0, 1, 3, 4] # gap, gradient, density, flux_x

        self.active_dims = active_dims

        self.database = database
        Xnew = get_new_training_input(self._Xtest,
                                      self.database.minimum_size - self.database.size)

        # For mock data from known constitutive laws
        noise = 0.
        self.noise = noise
        Ynew = get_new_training_output_mock(Xnew, prop, noise_stddev=noise)
        # ...or from MD
        # Ynew = get_new_training_output_MD(Xnew)

        self.database.add_data(Xnew.T, Ynew.T)
        self.update_training_data()
        self.last_fit_train_size = 0

        ref = datetime.now()
        self.cumtime_train = datetime.now() - ref
        self.cumtime_infer = datetime.now() - ref

        self.std_tol_norm = 0.1

        self.model_setup()

        self.step = 0

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
    def kernel_variance(self):
        return self.opt_posterior.prior.kernel.variance

    @property
    def kernel_lengthscale(self):
        return self.opt_posterior.prior.kernel.lengthscale

    @property
    def obs_stddev(self):
        return self.opt_posterior.likelihood.obs_stddev

    @property
    def trusted(self):
        return self.maximum_variance < self.variance_tol

    def write(self):
        pass

    def print_opt_summary(self, obj):

        print(f'# Objective    : {obj:.5g}')
        print("# Hyperparam   :", end=' ')
        # print(f"{self.database.data.n: 3d}", end=' ')

        print(f"{self.kernel_variance.value:.5e}", end=' ')
        print(f"{self.obs_stddev.value:.5e}", end=' ')

        for l in self.kernel_lengthscale:
            print(f"{l:.5e}", end=' ')

        print()

    def _train(self, opt='scipy', reason=0):

        reasons = ['DB', "AL"]

        print('#' + 15 * '-' + f"GP TRAINING ({self.name.upper()})" + 16 * '-')
        print('# Timestep     :', self.step)
        print('# Reason       :', reasons[reason])
        print('# Database size:', self.database.data.n)

        self.update_training_data()

        fit_func = {'scipy': gpx.fit_scipy,
                    'lbfgs': gpx.fit_lbfgs}[opt]

        opt_posterior, history = fit_func(
            model=self._posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
            train_data=self.train_data,
            verbose=False
        )

        self.last_fit_train_size = deepcopy(self.database.size)

        # TODO: write history
        self.opt_posterior = opt_posterior

        self.print_opt_summary(history[-1] if opt == "scipy" else history)

        if reason == 0:
            print('#' + 50 * '-')

    def _predict(self):

        latent_dist = self.opt_posterior.predict(self.Xtest.T / self.X_scale,
                                                 train_data=self.train_data)

        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        n, nt = self.train_data.y.shape
        _, nx, ny = self.density.shape

        predictive_mean = np.asarray(predictive_dist.mean).reshape(-1, nx, ny).squeeze() * self.y_scale
        predictive_var = np.asarray(predictive_dist.variance).reshape(-1, nx, ny).squeeze() * self.y_scale**2
        noise_variance = (self.opt_posterior.likelihood.obs_stddev * self.y_scale)**2
        predictive_var -= noise_variance

        self.variance_tol = (self.std_tol_norm * self.y_scale)**2
        self.maximum_variance = np.max(predictive_var)

        return predictive_mean, predictive_var

    def _active_learning(self, var):

        imax = np.argmax(var)
        # ix, iy = np.unravel_index(np.argmax(var), shape=var.shape)

        Xnew = self._Xtest[:, imax][:, None]
        Ynew = get_new_training_output_mock(Xnew, self.prop,
                                            noise_stddev=self.noise)  # replace w/ MD call

        self.database.add_data(Xnew.T, Ynew.T)

        # assert 0

    def infer(self, predictor=True):

        if predictor:
            self.step += 1
            if self.last_fit_train_size < self.database.size:
                tic = datetime.now()
                self._train(reason=0)
                toc = datetime.now()
                self.cumtime_train += toc - tic

        tic = datetime.now()
        m, v = self._predict()
        toc = datetime.now()
        self.cumtime_infer += toc - tic

        max_steps = 5  # TODO not hardcoded

        # Active learning
        if predictor:
            counter = 0
            before = deepcopy(self.maximum_variance / self.variance_tol)
            while not self.trusted and counter < max_steps:
                counter += 1

                # add new data
                self._active_learning(v)

                # train again
                tic = datetime.now()
                self._train(reason=1)
                toc = datetime.now()
                self.cumtime_train += toc - tic

                # predict again
                m, v = self._predict()
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
                            ]).reshape(6, -1)

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

    rho0 = prop['rho0']
    p0 = prop['P0']
    C1 = prop['C1']
    C2 = prop['C2']
    press = dowson_higginson_pressure(X[3],
                                      rho0, p0, C1, C2)[None, :] + noise

    return np.vstack([press,
                      tau_bot,
                      tau_top])
