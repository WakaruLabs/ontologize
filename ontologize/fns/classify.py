"""Classification activation functions and straight-through estimators.

This module provides activation functions used for discrete ontofeature selection
and categorical probability estimation over dictionary entries, including standard
softmax classification and a straight-through estimator (STE) for discrete one-hot
hard selection with differentiable softmax backward gradients.
"""
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float

def softmax_cl(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]:
    """Compute softmax probabilities along the trailing class/feature dimension.

    Applies `jax.nn.softmax` along `axis=-1`, converting unnormalized logits into
    a normalized probability distribution over dictionary tags.

    Args:
        x: Logit tensor of shape `(..., k)`, where `k` is the number of classes/tags.
        *args: Additional positional arguments forwarded to `jax.nn.softmax`.
        **kwargs: Additional keyword arguments forwarded to `jax.nn.softmax`.

    Returns:
        Probability distribution tensor of shape `(..., k)` summing to 1 along `axis=-1`.
    """
    return jax.nn.softmax(x, *args, **kwargs, axis=-1)

def ste(x: Float[Array, "... k"], *args, **kwargs) -> Float[Array, "... k"]:
    """Straight-through estimator. Use `argmax` for forward pass but
    `softmax` for backward pass."""
    k = x.shape[-1]
    P_soft = jax.nn.softmax(x, axis=-1)
    P_hard = jax.nn.one_hot(jnp.argmax(x, axis=-1), k)
    P = jax.lax.stop_gradient(P_hard - P_soft) + P_soft
    return P
