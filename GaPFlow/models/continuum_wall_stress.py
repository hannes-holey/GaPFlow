
import jax.numpy as jnp
from jax import vmap, grad, jit, lax
from jax import Array

class Operations():

    def difference(self, a, b):
        return a - b

    def difference_da(self, a, b):
        return jnp.ones_like(a)

    def difference_db(self, a, b):
        return - jnp.ones_like(b)

    def build_grad(self):
        pass


class Gap:
    """
    Important ! This class should be treated as immutable, or at least its parameters used in the jited functions !

    """

    # def __init__(self):
    #
    #
    #     self._step = 0
    #     # Simulation time getter for the history output, set by the Problem
    #     self.clock = lambda: float('nan')
    #     self.__solution = wrap_field(fc.get_real_field('solution'))
    #     self.__topo = wrap_field(fc.get_real_field('topography'))
    #     self.__extra = wrap_field(fc.get_real_field('extra'))
    #
    # I will probably need all the internal fields storage machinery analogous to the one in the GaussianProcess surogate and the WallStress classes
    #
    # Along with it a update method

    def wall_stress(self, rho, theta, j, h, U_bot, U_top):
        raise NotImplementedError

    def wall_stress_2D(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        raise NotImplementedError

    def tau_xz(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        return self.tau_xz_top(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top) - self.tau_xz_bot(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top)

    def tau_yz(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        return self.tau_yz_top(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top) - self.tau_yz_bot(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top)

    def tau_xz_bot(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        return self.wall_stress_2D(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top)[0]

    def tau_yz_bot(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        return self.wall_stress_2D(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top)[2]

    def tau_xz_top(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        return self.wall_stress_2D(rho, theta, jx, jy, h, dh_dx,  U_bot, U_top, V_bot, V_top)[1]

    def tau_yz_top(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        return self.wall_stress_2D(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top)[3]

    def build_grad(self, ) -> None:
        vmap2 = lambda f: jit(vmap(vmap(f)))

        for argnum, var in enumerate(['rho', 'theta', 'jx', 'jy', 'h', 'dh_dx', 'U_bot' , 'U_top', 'V_bot', 'V_top']):
            for direction in ['xz', 'yz']:
                setattr(self, f'dtau_{direction}_d{var}', vmap2(grad(getattr(self, f'tau_{direction}'), argnums=argnum)))
                setattr(self, f'dtau_{direction}_bot_d{var}', vmap2(grad(getattr(self, f'tau_{direction}_bot'), argnums=argnum)))
                setattr(self, f'dtau_{direction}_top_d{var}', vmap2(grad(getattr(self, f'tau_{direction}_top'), argnums=argnum)))

class ParabolicFlowFluidGap(Gap):

    @staticmethod
    def _shear_rates_2D(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):

        u_mean_x = jx / (rho * (1.0 - theta))
        u_mean_y = jy / (rho * (1.0 - theta))

        gam_bot_x = 2.0 / h * (3.0 * u_mean_x - 2.0 * U_bot - U_top)
        gam_top_x = 2.0 / h * (U_bot + 2.0 * U_top - 3.0 * u_mean_x)
        gam_bot_y = 2.0 / h * (3.0 * u_mean_y - 2.0 * V_bot - V_top)
        gam_top_y = 2.0 / h * (V_bot + 2.0 * V_top - 3.0 * u_mean_y)

        return gam_bot_x, gam_top_x, gam_bot_y, gam_top_y

    @staticmethod
    def velocity_profile_function(rho, theta,  jx, jy, h, U_bot, U_top, V_bot, V_top):
        u_mean_x = jx / (rho * (1.0 - theta))
        u_mean_y = jy / (rho * (1.0 - theta))

        def velocity(z):
            couette_x = U_bot + z / h * (U_top - U_bot)
            couette_y = V_bot + z / h * (V_top - V_bot)

            #                                       v mean velocity of the poiseulle term = total mean - couette mean
            poiseulle_x = z * (h- z) * 6 / h ** 2 * (u_mean_x - 0.5 * (U_bot + U_top) * (1 - theta))
            poiseulle_y = z * (h- z) * 6 / h ** 2 * (u_mean_y - 0.5 * (V_bot + V_top) * (1 - theta))

            return poiseulle_x + couette_x, poiseulle_y + couette_y

        return velocity

class NewtonParabolicFlowFluidGap(ParabolicFlowFluidGap):
    def __init__(self, mu0):
        self.mu0 = mu0

    def wall_stress(self, rho, theta, j, h, U_bot, U_top):
        u_mean = j / (rho * (1.0 - theta))
        etapt_cav = self.mu0 * (1.0 - theta)
        tau_bot = 2.0 * etapt_cav / h * (3.0 * u_mean - 2.0 * U_bot - U_top)
        tau_top = 2.0 * etapt_cav / h * (U_bot + 2.0 * U_top - 3.0 * u_mean)
        return tau_bot, tau_top

    def wall_stress_2D(self, rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top):
        etapt_cav = self.mu0 * (1.0 - theta)
        return [_ * etapt_cav for _ in self._shear_rates_2D(rho, theta, jx, jy, h, dh_dx, U_bot, U_top, V_bot, V_top) ]


class Slip:
    def stress(self, slip_velocity: float) -> float:
        raise NotImplementedError

    def stress_2D(self, slip_velocity_x: float, slip_velocity_y: float):
        raise NotImplementedError

    def build_grad(self, ) -> None:
        vmap2 = lambda f: jit(vmap(vmap(f)))
        self.tau_x = lambda *args: self.stress_2D(*args)[0]
        self.tau_y = lambda *args: self.stress_2D(*args)[1]

        for argnum, var in enumerate(['vs_x', 'vs_y']):
            for direction in ['x', 'y']:
                setattr(self, f'dtau_{direction}_d{var}', vmap2(grad(getattr(self, f'tau_{direction}'), argnums=argnum)))


class SlipLengthSlip(Slip):
    def __init__(self, slip_length, viscosity):
        self.slip_length = slip_length
        self.viscosity  = viscosity # TODO: here eta should be as argument

    def stress_2D(self, slip_velocity_x: float, slip_velocity_y: float):
        # Regularised magnitudes (avoids 0/0 at zero slip; 1e-20 is negligible vs physical speeds)
        slip_velocity = jnp.sqrt(slip_velocity_x ** 2 + slip_velocity_y ** 2 + 1e20)
        stress = self.viscosity / self.slip_length * slip_velocity
        stress_x = stress * slip_velocity_x / slip_velocity
        stress_y = stress * slip_velocity_y / slip_velocity
        return stress_x, stress_y

class TTSlip(Slip):
    """Thompson-Troian slip law."""

    def __init__(self, tc=10, vc=10):
        self.tc = tc
        self.vc = vc

    def stress(self, slip_velocity: float):
        v = slip_velocity / self.vc
        return 2 * self.tc * (jnp.sqrt(1 + v ** 2) - jnp.abs(v)) * v

    def stress_2D(self, slip_velocity_x: float, slip_velocity_y: float):
        # Regularised magnitudes (avoids 0/0 at zero slip; 1e-20 is negligible vs physical speeds)
        slip_velocity = jnp.sqrt(slip_velocity_x ** 2 + slip_velocity_y ** 2 + 1e20)
        v = slip_velocity / self.vc
        stress =  2 * self.tc * (jnp.sqrt(1 + v ** 2) - jnp.abs(v)) * v
        stress_x = stress * slip_velocity_x / slip_velocity
        stress_y = stress * slip_velocity_y / slip_velocity
        return stress_x, stress_y

