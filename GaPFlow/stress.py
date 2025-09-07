import numpy as np
import jax.numpy as jnp
from GaPFlow.gp import GaussianProcessSurrogate, MultiOutputKernel
from GaPFlow.models import dowson_higginson_pressure, stress_bottom, stress_top, stress_avg

from jax import config
from jaxtyping import install_import_hook
config.update("jax_enable_x64", True)

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx


class WallStress(GaussianProcessSurrogate):

    name = "shear"

    def __init__(self, fc, prop, data=None, gp_config=None):
        self.__field = fc.real_field('wall_stress', (12,))
        self.__field_variance = fc.real_field('wall_stress_var')
        self.prop = prop

        super().__init__(fc, data, gp_config, prop)

    @property
    def full(self):
        return self.__field.p

    @property
    def upper(self):
        return self.__field.p[6:]

    @property
    def lower(self):
        return self.__field.p[:6]

    @property
    def Xtest(self):

        # Overrides base class method, because we need extended input dimensions for multi output kernel
        Xtest_extended = jnp.hstack([
            jnp.vstack([self._Xtest, jnp.zeros_like(self._Xtest[0])]),
            jnp.vstack([self._Xtest, jnp.ones_like(self._Xtest[0])])
        ])

        return Xtest_extended

    def model_setup(self):
        # intial lengthscales
        l0 = np.std(self.database.data_press.X, axis=0)[jnp.array(self.active_dims)]

        scalar_kernel = gpx.kernels.Matern32(active_dims=self.active_dims,
                                             lengthscale=l0,
                                             variance=1.)

        self._kernel = MultiOutputKernel(scalar_kernel)

        self._mean_func = gpx.mean_functions.Zero()

        self._prior = gpx.gps.Prior(mean_function=self._mean_func,
                                    kernel=self._kernel,
                                    jitter=1e-6
                                    )

        self._likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.database.size,
                                                    # obs_stddev=gpx.parameters.Static(noise)
                                                    )

        self._posterior = self._prior * self._likelihood

    def update_training_data(self):
        self.train_data = self.database.data_shear_xz
        self.y_scale = self.database.y_scale[jnp.array([5, 11])].max()
        self.X_scale = self.database.Xext_scale

    def get_output(self, X):

        # For MD data: call update method from database (or external)

        U = 0.1
        V = 0.
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        Ybot = stress_bottom(X[3:],  # q
                             X[:3],  # h, dhdx, dhdy
                             U, V, eta, zeta, 0.)

        Ytop = stress_top(X[3:],  # q
                          X[:3],  # h, dhdx, dhdy
                          U, V, eta, zeta, 0.)

        return np.vstack([Ybot[4],
                          Ytop[4]])

    def update(self, gp=False, predictor=False):

        if gp:
            mean, var = self.infer(predictor)

            self.__field.p[4] = mean[0, :, :]
            self.__field.p[10] = mean[1, :, :]
            self.__field_variance.p = var[0, :, :]
        else:
            U = 0.1
            V = 0.
            eta = self.prop['shear']
            zeta = self.prop['bulk']

            self.__field.p[:6] = stress_bottom(self.density,
                                               self.gap,
                                               U, V, eta, zeta, 0.)

            self.__field.p[6:] = stress_top(self.density,
                                            self.gap,
                                            U, V, eta, zeta, 0.)


class BulkStress(GaussianProcessSurrogate):

    name = "bulk"

    def __init__(self, fc, prop, data=None, gp_config=None):
        self.__field = fc.real_field('bulk_viscous_stress', (3,))
        self.prop = prop

        super().__init__(fc, data, gp_config, prop)

    @property
    def stress(self):
        return self.__field.p

    def model_setup(self):
        pass

    def update_training_data(self):
        pass

    def get_output(self, X):

        # For MD data: call update method from database (or external)

        U = 0.1
        V = 0.
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        Y = stress_avg(X[3:],  # q
                       X[:3],  # h, dhdx, dhdy
                       U, V, eta, zeta, 0.)

        return Y

    def update(self):

        U = 0.1
        V = 0.
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        self.__field.p = stress_avg(self.density,
                                    self.gap,
                                    U, V, eta, zeta, 0.)


class Pressure(GaussianProcessSurrogate):

    name = "press"

    def __init__(self, fc, prop, data=None, gp_config=None):
        self.prop = prop
        self.__field = fc.real_field('pressure')
        self.__field_variance = fc.real_field('pressure_var')

        super().__init__(fc, data, gp_config, prop)

    @property
    def pressure(self):
        return self.__field.p

    @property
    def Xtest(self):
        return self._Xtest

    def model_setup(self):
        # intial lengthscales
        l0 = np.std(self.database.data_press.X, axis=0)[jnp.array(self.active_dims)]
        # l0 = jnp.ones(len(self.active_dims))

        self._kernel = gpx.kernels.Matern32(active_dims=self.active_dims,
                                            lengthscale=l0,
                                            variance=1.)

        self._mean_func = gpx.mean_functions.Zero()

        self._prior = gpx.gps.Prior(mean_function=self._mean_func,
                                    kernel=self._kernel,
                                    jitter=1e-6
                                    )

        self._likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.database.size,
                                                    # obs_stddev=gpx.parameters.Static(noise)
                                                    )

        self._posterior = self._prior * self._likelihood

    def update_training_data(self):
        self.train_data = self.database.data_press
        self.y_scale = self.database.y_scale[0]
        self.X_scale = self.database.X_scale

    def get_output(self, X):

        # For MD data: call update method from database (or external)

        rho0 = self.prop['rho0']
        p0 = self.prop['P0']
        C1 = self.prop['C1']
        C2 = self.prop['C2']
        return dowson_higginson_pressure(X[3],
                                         rho0, p0, C1, C2)[None, :]

    def update(self, gp=False, predictor=False):

        if gp:
            mean, var = self.infer(predictor)
            self.__field.p = mean
            self.__field_variance.p = var
        else:
            rho0 = self.prop['rho0']
            p0 = self.prop['P0']
            C1 = self.prop['C1']
            C2 = self.prop['C2']
            self.__field.p = dowson_higginson_pressure(self.density[0],
                                                       rho0, p0, C1, C2)


# Utility functions

def get_all_outputs(X, prop):

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
                                      rho0, p0, C1, C2)[None, :]

    return np.vstack([press,
                      tau_bot,
                      tau_top])
