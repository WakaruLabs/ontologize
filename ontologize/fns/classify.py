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

def sparsemax(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]:
    """Sparsemax (Martins & Astudillo 2016): the Euclidean projection of
    the logits onto the probability simplex, max(x - tau, 0) with tau set
    so that the result sums to 1. Logits below tau get exactly zero
    probability, and the support varies from row to row. On the support S
    the Jacobian is I - 1 1^T / |S|, and it is zero elsewhere, so a row
    whose support is one entry passes no gradient to its logits.
    `DictBlock.cluster` divides the logits by the temperature first, so a
    lower temperature gives a smaller support, with the argmax as its
    limit."""
    k = x.shape[-1]
    z = jnp.flip(jnp.sort(x, axis=-1), axis=-1)
    r = jnp.arange(1, k + 1, dtype=x.dtype)
    cs = jnp.cumsum(z, axis=-1)
    # the support's size: the largest r with 1 + r z_r above the sum of the
    # r largest logits (true for every r up to it, false after)
    n = jnp.sum(1 + r * z > cs, axis=-1, keepdims=True)
    tau = (jnp.take_along_axis(cs, n - 1, axis=-1) - 1) / n.astype(x.dtype)
    return jnp.maximum(x - tau, 0)

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
