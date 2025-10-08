import jax.numpy as jnp
from jax import vmap, grad

from GaPFlow.gp import GaussianProcessSurrogate
from GaPFlow.gp import multi_in_single_out, multi_in_multi_out
from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top, stress_avg
from GaPFlow.models.sound import eos_sound_velocity
from GaPFlow.models.viscosity import piezoviscosity, shear_thinning_factor, shear_rate_avg


class WallStress(GaussianProcessSurrogate):

    def __init__(self, fc, prop, geo, direction='x', data=None, gp=None):
        self.__field = fc.real_field(f'wall_stress_{direction}z', (12,))
        self.__pressure = fc.get_real_field('pressure')
        self.__gap = fc.get_real_field('gap')
        self.__x = fc.get_real_field('x')
        self.__y = fc.get_real_field('y')

        self.geo = geo
        self.prop = prop
        self.name = f'{direction}z'

        if direction == 'x':
            self.active_dims = [0, 3, 4, ]  # TODO: from yaml
            self._out_index = 4
        elif direction == 'y':
            self.active_dims = [0, 3, 5, ]
            self._out_index = 3

        if gp is not None:
            self.noise = (gp['press']['obs_stddev'] if gp['press_gp'] else 0.,
                          gp['shear']['obs_stddev'] if gp['shear_gp'] else 0.)

            gp = gp['shear']

            self.__field_variance = fc.real_field(f'wall_stress_{direction}z_var')

            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.max_steps = gp['max_steps']
            self.is_gp_model = True
            self.build_gp = multi_in_multi_out
        else:
            self.is_gp_model = False

        super().__init__(fc, data)

        if self.is_gp_model:
            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.std(self.Xtrain[0], axis=0))
            }

            self._train()
            self._infer()
            # self.maximum_variance = self.kernel_variance

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
    def pressure(self):
        return self.__pressure.p

    @property
    def dp_dx(self):
        return jnp.gradient(self.pressure, self.__x.p[:, 0], axis=0)

    @property
    def dp_dy(self):
        return jnp.gradient(self.pressure, self.__y.p[0, :], axis=1)

    @property
    def height(self):
        return self.__gap.p[0]

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
        return jnp.concatenate([self.database.Ytrain[:self.last_fit_train_size, self._out_index + 1],
                                self.database.Ytrain[:self.last_fit_train_size, self._out_index + 7]])

    @property
    def Yscale(self):
        return jnp.max(self.database.Y_scale[jnp.array([5, 11], dtype=int)])

    @property
    def Yerr(self):
        return self.noise[1] / self.Yscale

    @property
    def kernel_variance(self):
        return self.gp.kernel.kernels[0].kernel1.value

    @property
    def kernel_lengthscale(self):
        return self.gp.kernel.kernels[0].kernel2.scale

    @property
    def obs_stddev(self):
        return self.Yerr

    def update(self, predictor=False):

        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(self.pressure,
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(self.dp_dx,
                                        self.dp_dy,
                                        self.height,
                                        self.geo['U'],
                                        self.geo['V'],
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        s_bot = stress_bottom(self.density,
                              self.gap,
                              self.geo['U'],
                              self.geo['V'],
                              shear_viscosity,
                              self.prop['bulk'],
                              0.  # slip length
                              )

        s_top = stress_top(self.density,
                           self.gap,
                           self.geo['U'],
                           self.geo['V'],
                           shear_viscosity,
                           self.prop['bulk'],
                           0.  # slip length
                           )

        # FIXME: this is probably wrong if only one direction (xz or yz) is a GP model
        self.__field.p[:3] = s_bot[:3] / 2.
        self.__field.p[6:9] = s_top[:3] / 2.

        self.__field.p[5] = s_bot[-1] / 2.
        self.__field.p[11] = s_top[-1] / 2.

        if self.is_gp_model:
            mean, var = self.predict(predictor)
            self.__field.p[self._out_index] = mean[0, :, :]
            self.__field.p[self._out_index + 6] = mean[1, :, :]
            self.__field_variance.p = var[0, :, :]
        else:
            self.__field.p[self._out_index] = s_bot[self._out_index]
            self.__field.p[self._out_index + 6] = s_top[self._out_index]


class BulkStress(GaussianProcessSurrogate):

    name = "bulk"

    def __init__(self, fc, prop, geo, data=None, gp=None):
        self.__field = fc.real_field('bulk_viscous_stress', (3,))
        self.__pressure = fc.get_real_field('pressure')
        self.__gap = fc.get_real_field('gap')
        self.__x = fc.get_real_field('x')
        self.__y = fc.get_real_field('y')

        self.geo = geo
        self.prop = prop
        self.is_gp_model = False
        self.noise = 0.

        super().__init__(fc, data)

    @property
    def stress(self):
        return self.__field.p

    @property
    def pressure(self):
        return self.__pressure.p

    @property
    def dp_dx(self):
        return jnp.gradient(self.pressure, self.__x.p[:, 0], axis=0)

    @property
    def dp_dy(self):
        return jnp.gradient(self.pressure, self.__y.p[0, :], axis=1)

    @property
    def height(self):
        return self.__gap.p[0]

    def update(self):
        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(self.pressure,
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(self.dp_dx,
                                        self.dp_dy,
                                        self.height,
                                        self.geo['U'],
                                        self.geo['V'],
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        self.__field.p = stress_avg(self.density,
                                    self.gap,
                                    self.geo['U'],
                                    self.geo['V'],
                                    shear_viscosity,
                                    self.prop['bulk'],
                                    0.  # slip length
                                    )


class Pressure(GaussianProcessSurrogate):

    name = "zz"

    def __init__(self, fc, prop, geo, data=None, gp=None):

        self.__field = fc.real_field('pressure')
        self.geo = geo
        self.prop = prop

        if gp is not None:
            self.noise = (gp['press']['obs_stddev'] if gp['press_gp'] else 0.,
                          gp['shear']['obs_stddev'] if gp['shear_gp'] else 0.)

            gp = gp['press']

            self.active_dims = [0, 3, 4, ]  # TODO: from yaml
            self.__field_variance = fc.real_field('pressure_var')
            # self.noise = gp['obs_stddev']
            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.max_steps = gp['max_steps']
            self.is_gp_model = True
            self.build_gp = multi_in_single_out

            # for sound speed
            self.eos = lambda x: self.gp.predict(self.Ytrain, x[None, :]).squeeze()

        else:
            self.is_gp_model = False

        super().__init__(fc, data)

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
        return self.noise[0] / self.Yscale

    @property
    def kernel_variance(self):
        return self.gp.kernel.kernel1.value

    @property
    def kernel_lengthscale(self):
        return self.gp.kernel.kernel2.scale

    @property
    def obs_stddev(self):
        return self.Yerr

    def update(self, predictor=False):

        if self.is_gp_model:
            mean, var = self.predict(predictor)
            self.__field.p = mean
            self.__field_variance.p = var
        else:
            self.__field.p = eos_pressure(self.density[0], self.prop)
