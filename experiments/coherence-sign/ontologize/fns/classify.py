import jax
import jax.numpy as jnp

from jaxtyping import Array, Float

def softmax_cl(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]:
    return jax.nn.softmax(x, *args, **kwargs, axis=-1)

def ste(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]:
    """Straight-through estimator. Use `argmax` for forward pass but
    `softmax` for backward pass."""
    k = x.shape[-1]
    P_soft = jax.nn.softmax(x, axis=-1)
    P_hard = jax.nn.one_hot(jnp.argmax(x, axis=-1), k)
    P = jax.lax.stop_gradient(P_hard - P_soft) + P_soft
    return P
