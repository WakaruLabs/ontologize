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

def topk_cl(x: Float[Array, "... n"], k: int) -> Float[Array, "... n"]:
    """Softmax over the top-k logits; the rest are masked to -inf, so
    their probabilities are exactly 0 and carry no gradient (the mask
    condition is piecewise constant, so no gradient flows through the
    threshold either). The support is invariant to positive logit
    scaling, so `DictBlock.cluster`'s temperature only sharpens within
    the survivor set. Ties at the threshold all survive. `k` must be
    static (`jax.lax.top_k` requires a concrete k); reached through
    `get_activation`'s "top<k>" keys ("top1", "top4", ...)."""
    v = jax.lax.top_k(x, k)[0][..., -1:]
    return jax.nn.softmax(jnp.where(x >= v, x, -jnp.inf), axis=-1)
