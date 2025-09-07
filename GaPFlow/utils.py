import jax.numpy as jnp
import jax.random as jr


def get_new_training_input(Xtest, Nsample):

    if Nsample > 0:
        # Bounds for quasi random sampling of initial database
        jabs = jnp.hypot(jnp.mean(Xtest[4, :]), jnp.mean(Xtest[5, :]))
        rho = jnp.mean(Xtest[3, :])

        l_bounds = jnp.array([(1. - 1e-6) * rho, 0.5 * jabs, 0.])
        u_bounds = jnp.array([(1. + 1e-6) * rho, 1.5 * jabs, 0.5 * jabs])

        dim = len(l_bounds)

        # Constant properties are randomly sampled collectively (e.g. from the available profiles)
        key = jr.key(123)
        key, subkey = jr.split(key)
        choice = jr.choice(key, Xtest.shape[1], shape=(Nsample,), replace=False).tolist()

        # random samples (other strategies later...)
        scaled_samples = jr.uniform(key, shape=(Nsample, dim),
                                    minval=l_bounds[None, :],
                                    maxval=u_bounds[None, :]).T

        Xnew = jnp.vstack([
            Xtest[:3, choice],  # gap height
            scaled_samples[0],  # density
            scaled_samples[1],  # flux_x
            scaled_samples[2],  # flux_y
        ])

    else:
        Xnew = jnp.empty((6, 0))

    return Xnew
