import numpy as np
import jax.numpy as jnp


if __name__ == "__main__":

    # see discussion on: https://github.com/jax-ml/jax/issues/1961

    jnp_array = jnp.ones(3)
    onp_asarray = np.asarray(jnp_array)

    print(id(jnp_array), type(jnp_array))
    print(id(onp_asarray), type(onp_asarray))
    print(id(jnp_array.__array__()), type(jnp_array.__array__()))

    buffer_pointer = jnp_array.addressable_data(0).unsafe_buffer_pointer()

    print(np.lib.array_utils.byte_bounds(onp_asarray)[0])
    print(np.lib.array_utils.byte_bounds(onp_asarray)[0] == buffer_pointer)
    # True

    # Explicit copy with np.array
    onp_array = np.array(jnp_array)
    print(np.lib.array_utils.byte_bounds(onp_array)[0] == buffer_pointer)
    # False
