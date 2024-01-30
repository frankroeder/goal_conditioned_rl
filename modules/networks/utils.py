import flax.linen as nn
import jax


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

    return _init


def default_init(scale: float = jax.numpy.sqrt(2)):
    return nn.initializers.orthogonal(scale)
