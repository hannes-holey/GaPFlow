import jax.numpy as jnp
from jax import vmap, grad

from GaPFlow.gp import GaussianProcessSurrogate
from GaPFlow.gp import multi_in_single_out, multi_in_multi_out
from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top, stress_avg
from GaPFlow.models.sound import eos_sound_velocity


class WallStress(GaussianProcessSurrogate):

    name = "shear"

    def __init__(self, fc, prop, geo, data=None, gp=None):
        self.__field = fc.real_field('wall_stress', (12,))
        self.geo = geo

        if gp is not None:
            self.__field_variance = fc.real_field('wall_stress_var')
            self.active_dims = [0, 3, 4, ]  # TODO: from yaml
            self.noise = gp['obs_stddev']
            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.max_steps = gp['max_steps']
            self.is_gp_model = True
            self.build_gp = multi_in_multi_out
        else:
            self.is_gp_model = False

        super().__init__(fc, prop, data)

        if self.is_gp_model:
            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.std(self.Xtrain[0], axis=0))
            }

            self._train()

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

        X = jnp.concatenate([(self._Xtest / self.database.X_scale)[:, self.active_dims],
                             (self._Xtest / self.database.X_scale)[:, self.active_dims]])

        flag = jnp.concatenate([jnp.zeros(X.shape[0] // 2, dtype=int),
                                jnp.ones(X.shape[0] // 2, dtype=int)])

        return X, flag

    @property
    def Xtrain(self):

        X = jnp.concatenate([self.database.Xtrain[:, self.active_dims],
                             self.database.Xtrain[:, self.active_dims]])

        flag = jnp.concatenate([jnp.zeros(X.shape[0] // 2, dtype=int),
                                jnp.ones(X.shape[0] // 2, dtype=int)])

        return X, flag

    @property
    def Ytrain(self):
        return jnp.concatenate([self.database.Ytrain[:self.last_fit_train_size, 5],
                                self.database.Ytrain[:self.last_fit_train_size, 11]])

    @property
    def Yscale(self):
        return jnp.max(self.database.Y_scale[jnp.array([5, 11], dtype=int)])

    @property
    def Yerr(self):
        return self.noise / self.Yscale

    @property
    def kernel_variance(self):
        return self.gp.kernel.kernels[0].kernel1.value

    @property
    def kernel_lengthscale(self):
        return self.gp.kernel.kernels[0].kernel2.scale

    @property
    def obs_stddev(self):
        return self.Yerr

    def get_output(self, X):

        # For MD data: call update method from database (or external)

        U = self.geo['U']
        V = self.geo['V']
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        Ybot = stress_bottom(X[3:],  # q
                             X[:3],  # h, dhdx, dhdy
                             U, V, eta, zeta, 0.)

        Ytop = stress_top(X[3:],  # q
                          X[:3],  # h, dhdx, dhdy
                          U, V, eta, zeta, 0.)

        return jnp.vstack([Ybot[4],
                           Ytop[4]])

    def update(self, predictor=False):

        if self.is_gp_model:
            mean, var = self.predict(predictor)
            self.__field.p[4] = mean[0, :, :]
            self.__field.p[10] = mean[1, :, :]
            self.__field_variance.p = var[0, :, :]
        else:
            U = self.geo['U']
            V = self.geo['V']
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

    def __init__(self, fc, prop, geo, data=None, gp=None):
        self.__field = fc.real_field('bulk_viscous_stress', (3,))
        self.geo = geo

        self.is_gp_model = False
        self.noise = 0.

        super().__init__(fc, prop, data)

    @property
    def stress(self):
        return self.__field.p

    def model_setup(self):
        pass

    def get_output(self, X):

        # For MD data: call update method from database (or external)

        U = self.geo['U']
        V = self.geo['V']
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        Y = stress_avg(X[3:],  # q
                       X[:3],  # h, dhdx, dhdy
                       U, V, eta, zeta, 0.)

        return Y

    def update(self):
        U = self.geo['U']
        V = self.geo['V']
        eta = self.prop['shear']
        zeta = self.prop['bulk']

        self.__field.p = stress_avg(self.density,
                                    self.gap,
                                    U, V, eta, zeta, 0.)


class Pressure(GaussianProcessSurrogate):

    name = "press"

    def __init__(self, fc, prop, geo, data=None, gp=None):

        self.__field = fc.real_field('pressure')
        self.geo = geo

        if gp is not None:
            self.active_dims = [0, 3, 4, ]  # TODO: from yaml
            self.__field_variance = fc.real_field('pressure_var')
            self.noise = gp['obs_stddev']
            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.max_steps = gp['max_steps']
            self.is_gp_model = True
            self.build_gp = multi_in_single_out

            # for sound speed
            self.eos = lambda x: self.gp.predict(self.Ytrain, x[None, :]).squeeze()

        else:
            self.is_gp_model = False

        super().__init__(fc, prop, data)

        if self.is_gp_model:
            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.std(self.Xtrain, axis=0))
            }

            self._train()

    @property
    def pressure(self):
        return self.__field.p

    @property
    def v_sound(self) -> float:
        if self.is_gp_model:
            eos_grad = vmap(grad(self.eos))
            vsound_squared = eos_grad(self.Xtest)[:, 1].max() * self.Yscale / self.database.X_scale[3]
            vsound = jnp.sqrt(vsound_squared)
            return vsound
        else:
            return eos_sound_velocity(self.mass_density, self.prop).max()

    @property
    def Xtest(self):
        # not normalized
        return (self._Xtest / self.database.X_scale)[:, self.active_dims]

    @property
    def Xtrain(self):
        # normalized
        return self.database.Xtrain[:, self.active_dims]

    @property
    def Ytrain(self):
        # normalized
        return self.database.Ytrain[:self.last_fit_train_size, 0]

    @property
    def Yscale(self):
        return self.database.Y_scale[0]

    @property
    def Yerr(self):
        return self.noise / self.Yscale

    @property
    def kernel_variance(self):
        return self.gp.kernel.kernel1.value

    @property
    def kernel_lengthscale(self):
        return self.gp.kernel.kernel2.scale

    @property
    def obs_stddev(self):
        return self.Yerr

    def get_output(self, X):
        # For MD data: call update method from database (or external)
        return eos_pressure(X[3], self.prop)[None, :]

    def update(self, predictor=False):

        if self.is_gp_model:
            mean, var = self.predict(predictor)
            self.__field.p = mean
            self.__field_variance.p = var
        else:
            self.__field.p = eos_pressure(self.density[0], self.prop)
