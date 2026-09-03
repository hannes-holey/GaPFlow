#
# Copyright 2026 Dan Waxman
#           2025-2026 Hannes Holey
#           2025-2026 Christoph Huber
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
import numpy as np
import numpy.typing as npt
import jax.numpy as jnp
from jax import vmap, grad, jit, lax
from jax import Array
from typing import Optional, Any, Deque
from muGrid import Field

from .gp import GaussianProcessSurrogate
from .pressure import eos_pressure, eos_rho  # noqa: F401
from .viscous import (stress_bottom, stress_top, stress_avg,  # noqa: F401
                      stress_top_xz, stress_bottom_xz,
                      stress_top_yz, stress_bottom_yz,
                      get_shear_viscosity)
from .sound import eos_sound_velocity
from .viscosity import piezoviscosity, shear_thinning_factor, shear_rate_avg

NDArray = npt.NDArray[np.floating]
JAXArray = Array


class WallStress(GaussianProcessSurrogate):
    """
    Wall stress model (wall shear/stress in xz or yz direction).

    This class can operate in two modes:

    - Deterministic: compute wall/boundary stresses from viscous models.
    - GP-based surrogate: train/predict wall stress using GaussianProcessSurrogate.

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
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field collection that provides access to 'pressure', 'topography', etc.
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        direction : {'x', 'y'}, optional
            Direction of the wall stress ('x' -> xz component (default), 'y' -> yz component).
        data : Database or None, optional
            Training database if using GP surrogates.
        gp : dict or None, optional
            GP configuration dictionary (if using GP surrogates).
        """

        self.__field = fc.real_field(f'wall_stress_{direction}z', (12,))
        self.__pressure = Field(fc.get_real_field('pressure'))
        self.__x = Field(fc.get_real_field('x'))
        self.__y = Field(fc.get_real_field('y'))

        self.geo = geo
        self.prop = prop
        self.direction = direction
        self.name = f'{direction}z'

        self._out_index = {'x': 4, 'y': 3}[direction]

        if gp is not None:
            self.is_gp_model = True
            self.active_dims = {'x': gp.get('active_dims_x', [0, 1, 3]),
                                'y': gp.get('active_dims_y', [0, 2, 3])}[direction]
            self.derived_expressions = gp.get('derived_features', [])

            self.__field_variance = fc.real_field(f'wall_stress_{direction}z_var')

            # Active learning parameters
            self.tolerance_protocol = gp['tolerance_protocol']
            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.atol_reduction_factor = gp['atol_reduction_factor']
            self.tol_rmid = gp['tol_rmid']
            self.tol_alpha = gp['tol_alpha']
            self.max_steps = gp['max_steps']
            self.pause_steps = gp['pause_steps']
            self.use_active_learning = gp['active_learning']
            self.similarity_check = gp['similarity_check']
            self.allowed_skips = gp['allowed_skips']
            self.fix_noise = gp['fix_noise']
            self.pause_on_high_residual = gp['pause_on_high_residual']
        else:
            self.is_gp_model = False
            self.use_active_learning = False
            self.derived_expressions = []

        super().__init__(fc, data)

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
        return self.__field.pg

    @property
    def upper(self) -> NDArray:
        """
        Upper-wall stress slice.

        Returns
        -------
        ndarray
            Upper half of the wall stress components (6 entries).
        """
        return self.__field.pg[6:]

    @property
    def lower(self) -> NDArray:
        """
        Lower-wall stress slice.

        Returns
        -------
        ndarray
            Lower half of the wall stress components (6 entries).
        """
        return self.__field.pg[:6]

    @property
    def variance(self) -> NDArray:
        """
        Variance of the shear stress field

        """
        return self.__field_variance.pg

    @property
    def pressure(self) -> NDArray:
        """
        Local pressure field.

        Returns
        -------
        ndarray
            Pressure field from the field collection.
        """
        return self.__pressure.pg

    @property
    def dp_dx(self) -> NDArray:
        """
        Partial derivative of pressure with respect to x (∂p/∂x).

        Returns
        -------
        ndarra
            Gradient along x computed with jnp.gradient.
        """
        return np.gradient(self.pressure, self.__x.pg[:, 0], axis=0)

    @property
    def dp_dy(self) -> NDArray:
        """
        Partial derivative of pressure with respect to y (∂p/∂y).

        Returns
        -------
        ndarray
            Gradient along y computed with jnp.gradient.
        """
        return np.gradient(self.pressure, self.__y.pg[0, :], axis=1)

    @property
    def Xtest(self) -> JAXArray:
        """Test inputs for shear stress GP (normalized)."""
        return ((self._Xtest - self.database.X_shift) / self.database.X_scale)[:, self.active_dims]

    @property
    def _Ytrain(self) -> JAXArray:
        """
        Training outputs for GP corresponding to the lower and upper wall stress.

        Returns
        -------
        jax.Array
            Concatenated array of training outputs (lower then upper).
        """
        return jnp.vstack([self.database._Ytrain[:self.last_fit_train_size, self._out_index + 1],
                           self.database._Ytrain[:self.last_fit_train_size, self._out_index + 7]]).T

    @property
    def Ytrain(self) -> JAXArray:
        """
        Normalized training outputs for GP corresponding to the lower and upper wall stress.

        Returns
        -------
        jax.Array
            Concatenated array of training outputs (lower then upper).
        """
        return (self._Ytrain - self.Yshift) / self.Yscale

    @property
    def Yscale(self) -> JAXArray:
        """
        Output scaling factor used for normalization.

        Returns
        -------
        jax.Array
            Scalar-like array representing the maximum of selected Y scales.
        """
        indices = jnp.array([self._out_index + 1,
                             self._out_index + 7], dtype=int)

        return jnp.max(self.database.Y_scale[indices])

    @property
    def Yshift(self) -> JAXArray:
        """
        Output scaling factor used for normalization.

        Returns
        -------
        jax.Array
            Scalar-like array representing the maximum of selected Y scales.
        """
        indices = jnp.array([self._out_index + 1,
                             self._out_index + 7], dtype=int)

        return jnp.max(self.database.Y_shift[indices])

    @property
    def Yerr(self) -> JAXArray:
        """
        Observational error (normalized by Yscale).

        Returns
        -------
        jax.Array
            Observation noise standard deviation normalized by Yscale.
        """

        Yerr_all = jnp.vstack([self.database._Ytrain_err[:self.last_fit_train_size, self._out_index + 1],
                               self.database._Ytrain_err[:self.last_fit_train_size, self._out_index + 7]]).T

        return jnp.mean(Yerr_all / self.Yscale)

    @property
    def kernel_variance(self) -> JAXArray:
        """Return kernel variance (JAX scalar or array)."""
        return self.gp.kernel.kernel1.value

    @property
    def kernel_lengthscale(self) -> JAXArray:
        """Return kernel lengthscale(s)."""
        return self.gp.kernel.kernel2.scale

    @property
    def obs_stddev(self) -> JAXArray:
        """Observation standard deviation (normalized)."""
        return self.Yerr

    # -------------------------
    # Update
    # -------------------------
    def init(self) -> None:
        """Run first training and inference."""
        if self.is_gp_model:
            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.ones(len(self.active_dims)) * 0.5),
                "log_jitter": jnp.log(1e-3)
            }

            self._train()
            self._infer()

    def update(self,
               residuals: Deque,
               predictor: bool = False,
               compute_var: bool = False,
               ) -> None:
        """
        Update wall stress: compute deterministic stresses and, if enabled,
        perform GP prediction and place predicted mean and variance into the
        appropriate field entries.

        Parameters
        ----------
        residuals : Deque
            Residual buffer of the main simulation loop
        predictor : bool, optional
            Whether this update is part of the predictor stage.
        compute_var : bool, optional
            Flag for re-computing the variance (the default is False which uses
            the stored variance from previous steps).
        """

        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(self.pressure if not self.prop['EOS'] == 'Bayada' else self.solution[0],
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(self.dp_dx,
                                        self.dp_dy,
                                        self.height,
                                        np.hypot(self.geo['U_bot'], self.geo['V_bot']),
                                        np.hypot(self.geo['U_top'], self.geo['V_top']),
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        s_bot = stress_bottom(self.solution,
                              self.height_and_slopes,
                              self.geo['U_bot'],
                              self.geo['V_bot'],
                              self.geo['U_top'],
                              self.geo['V_top'],
                              shear_viscosity,
                              self.prop['bulk'],
                              Ls_bot=0.0,
                              Ls_top=self.extra if self.extra.shape == 1 else 0.
                              )

        s_top = stress_top(self.solution,
                           self.height_and_slopes,
                           self.geo['U_bot'],
                           self.geo['V_bot'],
                           self.geo['U_top'],
                           self.geo['V_top'],
                           shear_viscosity,
                           self.prop['bulk'],
                           Ls_bot=0.0,
                           Ls_top=self.extra if self.extra.shape == 1 else 0.
                           )

        self.__field.pg[:3] = s_bot[:3] / 2.
        self.__field.pg[6:9] = s_top[:3] / 2.

        self.__field.pg[5] = s_bot[-1] / 2.
        self.__field.pg[11] = s_top[-1] / 2.

        if self.is_gp_model:
            mean, var = self.predict(
                residuals=residuals,
                predictor=predictor,
                compute_var=self.use_active_learning or compute_var,
            )

            self.__field.pg[self._out_index] = mean[0, :, :]
            self.__field.pg[self._out_index + 6] = mean[1, :, :]
            self.__field_variance.pg[:] = var

        else:
            self.__field.pg[self._out_index] = s_bot[self._out_index]
            self.__field.pg[self._out_index + 6] = s_top[self._out_index]

    def build_grad(self) -> None:
        """Build JIT-compiled gradient functions for wall stress."""

        if self.is_gp_model:
            jmom_name = 'j' + self.direction
            X_shift = self.database.X_shift[jnp.array(self.active_dims)]
            X_scale = self.database.X_scale[jnp.array(self.active_dims)]

            Ytrain_bot = self.Ytrain[:, 0]
            Ytrain_top = self.Ytrain[:, 1]

            def _f(Ytrain_col, rho, jmom, h):
                x_norm = jnp.array([
                    (rho - X_shift[0]) / X_scale[0],
                    (jmom - X_shift[1]) / X_scale[1],
                    (h - X_shift[2]) / X_scale[2],
                ])
                return self.gp.predict(Ytrain_col, x_norm[None, :]).squeeze() * self.Yscale + self.Yshift

            f_bot = lambda rho, jmom, h: _f(Ytrain_bot, rho, jmom, h)
            f_top = lambda rho, jmom, h: _f(Ytrain_top, rho, jmom, h)
            f_gap = lambda rho, jmom, h: f_top(rho, jmom, h) - f_bot(rho, jmom, h)

            f_gap_drho = grad(f_gap, argnums=0)
            f_gap_djmom = grad(f_gap, argnums=1)
            f_gap_dh = grad(f_gap, argnums=2)
            f_bot_drho = grad(f_bot, argnums=0)
            f_bot_djmom = grad(f_bot, argnums=1)
            f_bot_dh = grad(f_bot, argnums=2)
            zero = lambda rho, jmom, h: jnp.zeros(())

            vmap2 = lambda f: jit(vmap(vmap(f, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)))

            self.tau = vmap2(f_gap)
            self.tau_bot = vmap2(f_bot)
            self.dtau_drho = vmap2(f_gap_drho)
            self.dtau_dtheta = vmap2(zero)
            self.dtau_dh = vmap2(f_gap_dh)
            self.dtau_bot_drho = vmap2(f_bot_drho)
            self.dtau_bot_dtheta = vmap2(zero)
            self.dtau_bot_dh = vmap2(f_bot_dh)
            setattr(self, f'dtau_d{jmom_name}', vmap2(f_gap_djmom))
            setattr(self, f'dtau_bot_d{jmom_name}', vmap2(f_bot_djmom))

        else:
            stress_top_fn = globals()[f'stress_top_{self.direction}z']
            stress_bot_fn = globals()[f'stress_bottom_{self.direction}z']
            der_vars = ['rho', 'j' + self.direction, 'theta', 'h']
            der_arg_idx = [0, 1 if self.direction == 'x' else 2, 10, 3]
            freeze_gradient = self.prop['viscosity']['freeze_gradient']
            underrelax_gradient_value = self.prop['viscosity']['underrelax_gradient_value']
            alpha_underrelax = self.prop['viscosity']['alpha_underrelax']

            def _eta(rho, dp_dx, dp_dy, h, eta_in):
                # eta_in is the (possibly underrelaxed) viscosity field from Viscosity.eta.
                if freeze_gradient:
                    return eta_in
                p = eos_pressure(rho, self.prop)
                eta_live = get_shear_viscosity(self, p, dp_dx, dp_dy, h)
                if underrelax_gradient_value:
                    # Value = underrelaxed eta_in (matches the residual used elsewhere),
                    # but gradient = alpha_underrelax * d(eta_live)/d(rho), i.e. the analytic
                    # deta/dp from the live pressure, SCALED by the same alpha_underrelax
                    # factor Viscosity.update() uses to blend eta_prev/eta_live into eta_in.
                    # Without this scaling the Jacobian saw the full, unscaled live gradient
                    # while the value it multiplies lagged ~1/alpha_underrelax behind it --
                    # an inconsistent linearization that blew up Newton for small alpha
                    # (e.g. alpha_underrelax=0.001 in relax_eta). eta_scaled -
                    # stop_gradient(eta_scaled) is exactly zero in value, so this adds no
                    # bias to eta_in; it only injects the scaled gradient into the tangent.
                    eta_scaled = eta_live * alpha_underrelax
                    return eta_in + (eta_scaled - lax.stop_gradient(eta_scaled))
                return eta_live

            # central functions: only argument difference for x/y is dh
            def _tau(rho, jx, jy, h, dh, U_bot, V_bot, U_top, V_top, Ls, theta, dp_dx, dp_dy, eta_in):
                eta = _eta(rho, dp_dx, dp_dy, h, eta_in)
                q = jnp.array([rho * (1.0 - theta), jx * (1.0 - theta), jy * (1.0 - theta)])
                h_arr = jnp.array([h, dh])
                tau_top = stress_top_fn(q, h_arr, U_bot, V_bot, U_top, V_top, eta, self.prop['bulk'], 0.0, Ls)
                tau_bot = stress_bot_fn(q, h_arr, U_bot, V_bot, U_top, V_top, eta, self.prop['bulk'], 0.0, Ls)
                return (1.0 - theta) * (tau_top - tau_bot)

            # required for energy
            def _tau_bot(rho, jx, jy, h, dh, U_bot, V_bot, U_top, V_top, Ls, theta, dp_dx, dp_dy, eta_in):
                eta = _eta(rho, dp_dx, dp_dy, h, eta_in)
                q = jnp.array([rho * (1.0 - theta), jx * (1.0 - theta), jy * (1.0 - theta)])
                h_arr = jnp.array([h, dh])
                tau_bot = stress_bot_fn(q, h_arr, U_bot, V_bot, U_top, V_top, eta, self.prop['bulk'], 0.0, Ls)
                return (1.0 - theta) * tau_bot

            vmap2 = lambda f: vmap(vmap(f))  # no map_axes required since we broadcast all args to quad fields

            # tau and its gradients
            setattr(self, 'tau', jit(vmap2(_tau)))
            for i, var in enumerate(der_vars):
                idx = der_arg_idx[i]
                setattr(self, f'dtau_d{var}', jit(vmap2(grad(_tau, argnums=idx))))

            # tau_bot and its gradients
            setattr(self, 'tau_bot', jit(vmap2(_tau_bot)))
            for i, var in enumerate(der_vars):
                idx = der_arg_idx[i]
                setattr(self, f'dtau_bot_d{var}', jit(vmap2(grad(_tau_bot, argnums=idx))))


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
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field collection that provides access to 'pressure', 'topography', etc.
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        data : Database or None, optional
            Training database if using GP surrogates.
        gp : dict or None, optional
            GP configuration dictionary (if using GP surrogates).
        """

        self.__field = fc.real_field('bulk_viscous_stress', (3,))
        self.__pressure = Field(fc.get_real_field('pressure'))
        self.__x = Field(fc.get_real_field('x'))
        self.__y = Field(fc.get_real_field('y'))

        self.geo = geo
        self.prop = prop
        self.is_gp_model = False

        super().__init__(fc, data)

    @property
    def stress(self) -> NDArray:
        """Return the bulk viscous stress field."""
        return self.__field.pg

    @property
    def pressure(self) -> NDArray:
        """Return the pressure field."""
        return self.__pressure.pg

    @property
    def dp_dx(self) -> NDArray:
        """Return ∂p/∂x."""
        return np.gradient(self.pressure, self.__x.pg[:, 0], axis=0)

    @property
    def dp_dy(self) -> NDArray:
        """Return ∂p/∂y."""
        return np.gradient(self.pressure, self.__y.pg[0, :], axis=1)

    def update(self) -> None:
        """Compute and store bulk viscous stress using viscous model."""

        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(self.pressure if not self.prop['EOS'] == 'Bayada' else self.solution[0],
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(self.dp_dx,
                                        self.dp_dy,
                                        self.height,
                                        np.hypot(self.geo['U_bot'], self.geo['V_bot']),
                                        np.hypot(self.geo['U_top'], self.geo['V_top']),
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        self.__field.pg[:] = stress_avg(self.solution,
                                        self.height_and_slopes,
                                        self.geo['U_bot'],
                                        self.geo['V_bot'],
                                        self.geo['U_top'],
                                        self.geo['V_top'],
                                        shear_viscosity,
                                        self.prop['bulk'],
                                        Ls_bot=0.0,
                                        Ls_top=self.extra if self.extra.shape == 1 else 0.
                                        )


class Pressure(GaussianProcessSurrogate):
    """
    Pressure model.

    Supports deterministic pressure via lookup equation of state or a GP surrogate.
    """

    name = "zz"

    def __init__(self,
                 fc: Any,
                 prop: dict,
                 geo: dict,
                 data: Optional[Any] = None,
                 gp: Optional[dict] = None) -> None:
        """Constructor

        Parameters
        ----------
        fc : muGrid.GlobalFieldCollection
            Field collection that provides access to 'pressure', 'topography', etc.
        prop : dict
            Physical fluid properties (e.g., shear viscosity).
        geo : dict
            Geometry parameters.
        data : Database or None, optional
            Training database if using GP surrogates.
        gp : dict or None, optional
            GP configuration dictionary (if using GP surrogates).
        """

        self.__field = Field(fc.get_real_field('pressure'))
        self.geo = geo
        self.prop = prop

        if gp is not None:
            self.is_gp_model = True
            self.active_dims = gp.get('active_dims', [0, 3])
            self.derived_expressions = gp.get('derived_features', [])
            self.__field_variance = fc.real_field('pressure_var')

            # Active learning parameters
            self.tolerance_protocol = gp['tolerance_protocol']
            self.atol = gp['atol']
            self.rtol = gp['rtol']
            self.atol_reduction_factor = gp['atol_reduction_factor']
            self.tol_rmid = gp['tol_rmid']
            self.tol_alpha = gp['tol_alpha']
            self.max_steps = gp['max_steps']
            self.pause_steps = gp['pause_steps']
            self.use_active_learning = gp['active_learning']
            self.similarity_check = gp['similarity_check']
            self.allowed_skips = gp['allowed_skips']
            self.fix_noise = gp['fix_noise']
            self.pause_on_high_residual = gp['pause_on_high_residual']
        else:
            self.is_gp_model = False
            self.use_active_learning = False
            self.derived_expressions = []

        super().__init__(fc, data)

    @property
    def pressure(self) -> NDArray:
        """Pressure field."""
        return self.__field.pg

    @property
    def variance(self) -> NDArray:
        """Variance of the pressure field."""
        return self.__field_variance.pg

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
            vsound_squared = eos_grad(self.Xtest)[:, 0].max() * self.Yscale / self.database.X_scale[0]
            vsound = jnp.sqrt(vsound_squared)
            return vsound
        else:
            return eos_sound_velocity(self.solution[0], self.prop).max()

    @property
    def Xtest(self) -> JAXArray:
        """Test inputs for pressure GP (normalized)."""
        return ((self._Xtest - self.database.X_shift) / self.database.X_scale)[:, self.active_dims]

    @property
    def _Ytrain(self) -> JAXArray:
        """Training outputs for pressure GP (not normalized)."""
        return self.database._Ytrain[:self.last_fit_train_size, 0]

    @property
    def Ytrain(self) -> JAXArray:
        """Training outputs for pressure GP (normalized)."""
        return (self._Ytrain - self.Yshift) / self.Yscale

    @property
    def Yshift(self) -> JAXArray:
        """Training outputs for pressure GP (not normalized)."""
        return self.database.Y_shift[0]

    @property
    def Yscale(self) -> JAXArray:
        """Output scale for pressure (scalar-like jax.Array)."""
        return self.database.Y_scale[0]

    @property
    def Yerr(self) -> float:
        """Observation noise (normalized) for pressure."""
        return jnp.mean(self.database.Ytrain_err[:self.last_fit_train_size, 0])

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

    def init(self) -> None:
        """Run first training and inference."""
        if self.is_gp_model:
            # for sound speed
            self.eos = lambda x: self.gp.predict(self.Ytrain, x[None, :]).squeeze()

            self.params_init = {
                "log_amp": jnp.log(1.),
                "log_scale": jnp.log(jnp.ones(len(self.active_dims)) * 0.5),
                "log_jitter": jnp.log(1e-3)
            }

            self._train()
            self._infer()

    def update(self,
               residuals: Deque,
               predictor: bool = False,
               compute_var: bool = False) -> None:
        """
        Update pressure: compute deterministic stresses and, if enabled,
        perform GP prediction and place predicted mean and variance into the
        appropriate field entries.

        Parameters
        ----------
        residuals : Deque
            Residual buffer of the main simulation loop
        predictor : bool, optional
            Whether this update is part of the predictor stage.
        compute_var : bool, optional
            Flag for re-computing the variance (the default is False which uses
            the stored variance from previous steps).
        """
        if self.is_gp_model:
            mean, var = self.predict(residuals=residuals,
                                     predictor=predictor,
                                     compute_var=self.use_active_learning or compute_var)

            self.__field.pg[:] = mean
            self.__field_variance.pg[:] = var
        else:
            self.__field.pg[:] = eos_pressure(self.solution[0], self.prop)

    def build_grad(self) -> None:

        if self.is_gp_model:
            X_shift = self.database.X_shift[jnp.array(self.active_dims)]  # [rho_shift, h_shift]
            X_scale = self.database.X_scale[jnp.array(self.active_dims)]  # [rho_scale, h_scale]

            def f0(rho, h):
                x_norm = jnp.array([(rho - X_shift[0]) / X_scale[0],
                                    (h - X_shift[1]) / X_scale[1]])
                return self.gp.predict(self.Ytrain, x_norm[None, :]).squeeze() * self.Yscale + self.Yshift

            f1 = grad(f0, argnums=0)
            f2 = grad(f1, argnums=0)

            def _newton_invert(p_target, h, rho_init, n_steps=10):
                def step(rho, _):
                    return rho - (f0(rho, h) - p_target) / f1(rho, h), None
                rho, _ = lax.scan(step, rho_init, None, length=n_steps)
                return rho

            vmap2 = lambda f: jit(vmap(vmap(f, in_axes=(0, 0)), in_axes=(0, 0)))
            vmap2_inv = lambda f: jit(vmap(vmap(f, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)))

            self.p_from_rho = vmap2(f0)
            self.dp_drho = vmap2(f1)
            self.d2p_drho2 = vmap2(f2)
            self.drho_dp = vmap2(lambda rho, h: 1.0 / f1(rho, h))
            self.rho_from_p = vmap2_inv(_newton_invert)

        else:

            f0 = lambda rho: eos_pressure(rho, self.prop)
            f1 = grad(f0, argnums=0)
            f2 = grad(f1, argnums=0)

            vmap2 = lambda f: jit(vmap(vmap(f, in_axes=0), in_axes=0))

            self.p_from_rho = vmap2(f0)
            self.dp_drho = vmap2(f1)
            self.d2p_drho2 = vmap2(f2)
            self.drho_dp = vmap2(lambda rho: 1.0 / f1(rho))
            self.rho_from_p = vmap2(lambda p: eos_rho(p, self.prop))


class Viscosity():

    def __init__(self,
                 fc: Any,
                 prop: dict) -> None:
        self.__field = fc.real_field('shear_viscosity')
        self.prop = prop
        self.eta_prev = None
        # Fixed relaxation target (e.g. piezoviscosity at a frozen, converged
        # pressure). None (default): relax towards the live piezoviscosity
        # computed below, as before. Set externally to relax towards this
        # instead, ignoring the freshly computed shear_viscosity.
        self.eta_target = None

    @property
    def shear_viscosity(self) -> NDArray:
        """Shear viscosity field."""
        return self.__field.pg

    def update(self,
               pressure: NDArray,
               dp_dx: NDArray,
               dp_dy: NDArray,
               height: NDArray,
               U_bot: float,
               V_bot: float,
               U_top: float = 0.0,
               V_top: float = 0.0) -> None:
        """Update shear viscosity field using piezoviscosity and shear-thinning."""
        # piezoviscosity
        if 'piezo' in self.prop.keys():
            mu0 = piezoviscosity(pressure,
                                 self.prop['shear'],
                                 self.prop['piezo'])
        else:
            mu0 = self.prop['shear']

        # shear-thinning
        if 'thinning' in self.prop.keys():
            shear_rate = shear_rate_avg(dp_dx,
                                        dp_dy,
                                        height,
                                        np.hypot(U_bot, V_bot),
                                        np.hypot(U_top, V_top),
                                        mu0)

            shear_viscosity = mu0 * shear_thinning_factor(shear_rate, mu0,
                                                          self.prop['thinning'])
        else:
            shear_viscosity = mu0

        shear_viscosity = np.broadcast_to(shear_viscosity, pressure.shape)

        if self.eta_target is not None:
            shear_viscosity = self.eta_target

        # underrelaxation
        alpha = self.prop['viscosity']['alpha_underrelax']
        if self.eta_prev is None:
            self.eta_prev = shear_viscosity
        shear_viscosity = (1 - alpha) * self.eta_prev + alpha * shear_viscosity
        self.eta_prev = np.copy(shear_viscosity)

        self.__field.pg[:] = shear_viscosity

    @property
    def eta(self) -> NDArray:
        """Alias for shear_viscosity for cleaner access."""
        return self.__field.pg
