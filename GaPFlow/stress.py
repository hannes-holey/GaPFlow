import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from jax import vmap, grad
from jax import Array
from typing import Optional, Tuple, Any

from GaPFlow.gp import GaussianProcessSurrogate
from GaPFlow.gp import multi_in_single_out, multi_in_multi_out
from GaPFlow.models.pressure import eos_pressure
from GaPFlow.models.viscous import stress_bottom, stress_top, stress_avg
from GaPFlow.models.sound import eos_sound_velocity
from GaPFlow.models.viscosity import piezoviscosity, shear_thinning_factor, shear_rate_avg

NDArray = npt.NDArray[np.floating]
JAXArray = Array


class WallStress(GaussianProcessSurrogate):
    """
    Wall stress model (wall shear/stress in xz or yz direction).

    This class can operate in two modes:
    - Deterministic: compute wall/boundary stresses from viscous models.
    - GP-based surrogate: train/predict wall stress using GaussianProcessSurrogate.

    Parameters
    ----------
    fc : muGrid.GlobalFieldCollection
        Field collection that provides access to 'pressure', 'gap', 'x', 'y', etc.
    prop : dict
        Physical properties (e.g., 'shear', 'bulk', optionally 'piezo', 'thinning').
    geo : dict
        Geometry and flow parameters (e.g., 'U', 'V').
    direction : {'x', 'y'}, optional
        Direction of the wall stress ('x' -> xz component, 'y' -> yz component).
    data : Database or None, optional
        Training database if using GP surrogates.
    gp : dict or None, optional
        GP configuration dictionary (if using GP surrogates).
    """

    def __init__(
        self,
        fc: Any,
        prop: dict,
        geo: dict,
        direction: str = 'x',
        data: Optional[Any] = None,
        gp: Optional[dict] = None
    ) -> None:
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

    # -------------------------
    # Properties
    # -------------------------
    @property
    def full(self) -> NDArray:
        """
        Full wall stress field including upper and lower components.

        Returns
        -------
        ndarray
            Array holding 12 components for wall stress (6 lower + 6 upper).
        """
        return self.__field.p

    @property
    def upper(self) -> NDArray:
        """
        Upper-wall stress slice.

        Returns
        -------
        ndarray
            Upper half of the wall stress components (6 entries).
        """
        return self.__field.p[6:]

    @property
    def lower(self) -> NDArray:
        """
        Lower-wall stress slice.

        Returns
        -------
        ndarray
            Lower half of the wall stress components (6 entries).
        """
        return self.__field.p[:6]

    @property
    def pressure(self) -> NDArray:
        """
        Local pressure field.

        Returns
        -------
        ndarray
            Pressure field from the field collection.
        """
        return self.__pressure.p

    @property
    def dp_dx(self) -> NDArray:
        """
        Partial derivative of pressure with respect to x (∂p/∂x).

        Returns
        -------
        ndarra
            Gradient along x computed with jnp.gradient.
        """
        return np.gradient(self.pressure, self.__x.p[:, 0], axis=0)

    @property
    def dp_dy(self) -> NDArray:
        """
        Partial derivative of pressure with respect to y (∂p/∂y).

        Returns
        -------
        ndarray
            Gradient along y computed with jnp.gradient.
        """
        return np.gradient(self.pressure, self.__y.p[0, :], axis=1)

    @property
    def height(self) -> NDArray:
        """
        Gap height field.

        Returns
        -------
        ndarray
            Gap height array (typically the first component of gap field).
        """
        return self.__gap.p[0]

    @property
    def Xtest(self) -> Tuple[JAXArray, JAXArray]:
        """
        Test inputs for GP prediction.

        Returns
        -------
        tuple of jax.Array
            (X, flag) where X contains duplicated and scaled features and flag
            indicates which output (lower/upper) the sample corresponds to.
        """
        X = jnp.concatenate([(self._Xtest / self.database.X_scale)[:, self.active_dims],
                             (self._Xtest / self.database.X_scale)[:, self.active_dims]])

        flag = jnp.concatenate([jnp.zeros(X.shape[0] // 2, dtype=int),
                                jnp.ones(X.shape[0] // 2, dtype=int)])

        return X, flag

    @property
    def Xtrain(self) -> Tuple[JAXArray, JAXArray]:
        """
        Training inputs for GP (duplicated to account for two outputs).

        Returns
        -------
        tuple of jax.Array
            (X, flag) where X contains duplicated input rows and flag labels.
        """
        X = jnp.concatenate([self.database.Xtrain[:, self.active_dims],
                             self.database.Xtrain[:, self.active_dims]])

        flag = jnp.concatenate([jnp.zeros(X.shape[0] // 2, dtype=int),
                                jnp.ones(X.shape[0] // 2, dtype=int)])

        return X, flag

    @property
    def Ytrain(self) -> JAXArray:
        """
        Training outputs for GP corresponding to the lower and upper wall stress.

        Returns
        -------
        jax.Array
            Concatenated array of training outputs (lower then upper).
        """
        return jnp.concatenate([self.database.Ytrain[:self.last_fit_train_size, self._out_index + 1],
                                self.database.Ytrain[:self.last_fit_train_size, self._out_index + 7]])

    @property
    def Yscale(self) -> JAXArray:
        """
        Output scaling factor used for normalization.

        Returns
        -------
        jax.Array
            Scalar-like array representing the maximum of selected Y scales.
        """
        return jnp.max(self.database.Y_scale[jnp.array([5, 11], dtype=int)])

    @property
    def Yerr(self) -> JAXArray:
        """
        Observational error (normalized by Yscale).

        Returns
        -------
        jax.Array
            Observation noise standard deviation normalized by Yscale.
        """
        return self.noise[1] / self.Yscale

    @property
    def kernel_variance(self) -> JAXArray:
        """Return kernel variance (JAX scalar or array)."""
        return self.gp.kernel.kernels[0].kernel1.value

    @property
    def kernel_lengthscale(self) -> JAXArray:
        """Return kernel lengthscale(s)."""
        return self.gp.kernel.kernels[0].kernel2.scale

    @property
    def obs_stddev(self) -> JAXArray:
        """Observation standard deviation (normalized)."""
        return self.Yerr

    # -------------------------
    # Update
    # -------------------------
    def update(self, predictor: bool = False) -> None:
        """
        Update wall stress: compute deterministic stresses and, if enabled,
        perform GP prediction and place predicted mean and variance into the
        appropriate field entries.

        Parameters
        ----------
        predictor : bool, optional
            Whether this update is part of the predictor stage.
        """
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
    """
    Bulk (gap-averaged) viscous stress model.

    This model currently operates deterministically (no GP surrogate).
    """

    name = "bulk"

    def __init__(self,
                 fc: Any,
                 prop: dict,
                 geo: dict,
                 data: Optional[Any] = None,
                 gp: Optional[dict] = None) -> None:

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
    def stress(self) -> NDArray:
        """Return the bulk viscous stress field."""
        return self.__field.p

    @property
    def pressure(self) -> NDArray:
        """Return the pressure field."""
        return self.__pressure.p

    @property
    def dp_dx(self) -> NDArray:
        """Return ∂p/∂x."""
        return np.gradient(self.pressure, self.__x.p[:, 0], axis=0)

    @property
    def dp_dy(self) -> NDArray:
        """Return ∂p/∂y."""
        return np.gradient(self.pressure, self.__y.p[0, :], axis=1)

    @property
    def height(self) -> NDArray:
        """Return gap height array."""
        return self.__gap.p[0]

    def update(self) -> None:
        """Compute and store bulk viscous stress using viscous model."""
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
    """
    Pressure model.

    Supports deterministic pressure via eos_pressure or a GP surrogate.
    """

    name = "zz"

    def __init__(self,
                 fc: Any,
                 prop: dict,
                 geo: dict,
                 data: Optional[Any] = None,
                 gp: Optional[dict] = None) -> None:

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
    def pressure(self) -> NDArray:
        """Pressure field."""
        return self.__field.p

    @property
    def v_sound(self) -> NDArray | JAXArray:
        """
        Effective sound speed computed from the GP-based eos (if available)
        or from the analytic eos_sound_velocity.

        Returns
        -------
        jax.Array or scalar-like
            Sound speed (may be a JAX array/scalar).
        """
        if self.is_gp_model:
            eos_grad = vmap(grad(self.eos))
            vsound_squared = eos_grad(self.Xtest)[:, 1].max() * self.Yscale / self.database.X_scale[3]
            vsound = jnp.sqrt(vsound_squared)
            return vsound
        else:
            return eos_sound_velocity(self.mass_density, self.prop).max()

    @property
    def Xtest(self) -> JAXArray:
        """Test inputs for pressure GP (not normalized)."""
        # not normalized
        return (self._Xtest / self.database.X_scale)[:, self.active_dims]

    @property
    def Xtrain(self) -> JAXArray:
        """Training inputs for pressure GP (normalized)."""
        # normalized
        return self.database.Xtrain[:, self.active_dims]

    @property
    def Ytrain(self) -> JAXArray:
        """Training outputs for pressure GP (normalized)."""
        # normalized
        return self.database.Ytrain[:self.last_fit_train_size, 0]

    @property
    def Yscale(self) -> JAXArray:
        """Output scale for pressure (scalar-like jax.Array)."""
        return self.database.Y_scale[0]

    @property
    def Yerr(self) -> JAXArray:
        """Observation noise (normalized) for pressure."""
        return self.noise[0] / self.Yscale

    @property
    def kernel_variance(self) -> JAXArray:
        """Kernel variance for pressure GP."""
        return self.gp.kernel.kernel1.value

    @property
    def kernel_lengthscale(self) -> JAXArray:
        """Kernel lengthscale for pressure GP."""
        return self.gp.kernel.kernel2.scale

    @property
    def obs_stddev(self) -> JAXArray:
        """Observation standard deviation for pressure GP."""
        return self.Yerr

    def update(self, predictor: bool = False) -> None:
        """
        Update the pressure field: perform GP inference if enabled or
        compute analytic pressure via eos_pressure.
        """
        if self.is_gp_model:
            mean, var = self.predict(predictor)
            self.__field.p = mean
            self.__field_variance.p = var
        else:
            self.__field.p = eos_pressure(self.density[0], self.prop)
