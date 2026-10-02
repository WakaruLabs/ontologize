"""Loss functions, similarity metrics, information criteria, noise generators, and ghost gradients.

This module provides the core mathematical functions used for dictionary learning objectives,
regularization, and dead-feature revival:
- Similarity and distance metrics: cosine similarity (`cossim`, `bcossim`), L0, L1, and L2 norms.
- Information theory: Shannon entropy (`entropy`), Akaike (`aic`) and Bayesian (`bic`) information criteria.
- Noise injection: Gaussian noise (`addnoise`), per-sample norm-scaled noise (`addnoise_batchnorm`),
  and batch feature-variance scaled noise (`addnoise_featvar`).
- Dead-feature ghost gradients: auxiliary gradient pathways (`l2_ghost`, `ghost`, `ghostgrad`)
  preventing dictionary entries and linear units from permanently dying.
"""
import jax
import jax.numpy as jnp
from typing import Any, Callable, List, Optional, Tuple, TypeVar
from jaxtyping import Array, Bool, Float, UInt, PRNGKeyArray
import einops

T = TypeVar('T') # generic type
def identity(x: T, *args, **kwargs) -> T:
    """Identity function returning input unchanged.

    Used as a default passthrough activation, noise, or loss function.

    Args:
        x: Any input tensor or value.
        *args: Ignored additional positional arguments.
        **kwargs: Ignored additional keyword arguments.

    Returns:
        The input `x` unmodified.
    """
    return x

def cossim(x: Float[Array,  "d"], y: Float[Array, "d"]) -> Float[Array, ""]:
    """Pairwise cosine similarity of two 1D arrays.

    Computes `dot(x, y) / (||x|| * ||y||)` with machine epsilon added to prevent
    division by zero.

    Args:
        x: First 1D vector of shape `(d,)`.
        y: Second 1D vector of shape `(d,)`.

    Returns:
        Scalar float array `()` containing the cosine similarity in `[-1.0, 1.0]`.
    """
    eps = jnp.finfo(x.dtype).eps
    norm_x = jnp.sqrt(jnp.sum(x**2) + eps)
    norm_y = jnp.sqrt(jnp.sum(y**2) + eps)
    return jnp.dot(x, y) / (norm_x * norm_y)

def bcossim(X: Float[Array, "... b d"]) -> Float[Array, "... b b"]:
    """Cosine similarity matrix for all samples. The -2 axis is assumed to be the batch dimension."""
    norms = jnp.sqrt((X ** 2).sum(axis=-1, keepdims=True) + jnp.finfo(X.dtype).eps)
    X_norm = X / norms
    return jnp.einsum("...id,...jd->...ij", X_norm, X_norm) 

def entropy(X: Float[Array, "... d"]) -> Float[Array, "..."]:
    """Shannon entropy."""
    X = X + jnp.finfo(X.dtype).eps
    X = X / jnp.sum(X, axis=-1, keepdims=True)
    return -jnp.sum(X * jnp.log2(X), axis=-1)

def l0(x: Float[Array, "..."], reduction: Callable = jnp.sum
       ) -> Float[Array, ""]:
    """Applies `reduction` to `abs(x) > 0`. When `reduction=jnp.sum` this gives L0 loss.

    Measures sparsity by counting non-zero entries in `x`.

    Args:
        x: Input tensor of arbitrary shape `(...)`.
        reduction: Aggregation function (default `jnp.sum`) reducing boolean array to a scalar.

    Returns:
        Scalar float array `()` with the reduced non-zero count.
    """
    return reduction(jnp.abs(x) > 0)

def l1(x: Float[Array, "..."], reduction: Callable = jnp.sum
       ) -> Float[Array, ""]:
    """Applies `reduction` to `abs(x)`. When `reduction=jnp.sum`, this gives L1 loss.

    Computes the L1 norm or mean absolute error of `x`.

    Args:
        x: Input tensor of arbitrary shape `(...)`.
        reduction: Aggregation function (default `jnp.sum`) reducing array to a scalar.

    Returns:
        Scalar float array `()` containing the reduced L1 magnitude.
    """
    return reduction(jnp.abs(x))

def l2(x: Float[Array, "..."], y: Float[Array, "..."], 
       reduction: Callable = jnp.mean) -> Float[Array,""]:
    """Applies `reduction` to `(x - y) ** 2`. When `reduction=jnp.mean`, this gives
    MSE/L2 loss."""
    return reduction((x - y)**2)

def errorct(pred: Float[Array, "... d"], target: UInt[Array, "..."]
            ) -> UInt[Array, ""]:
    """Count classification errors between predictions and integer class targets.

    Computes `argmax(pred, axis=-1)` and counts mismatches with `target`.

    Args:
        pred: Predicted class logits or probabilities of shape `(..., d)`.
        target: Ground-truth integer labels of shape `(...)`.

    Returns:
        Scalar unsigned integer array `()` containing total misclassification count.
    """
    return jnp.sum(jnp.argmax(pred, -1) != target)

def aic(k, L):
    """Akaike information criterion.
    k is the number of parameters.
    L is the maximised value of the likelihood function.
    """
    return 2 * k - 2 * jnp.log(L)

def bic(k, n, L):
    """Bayesian information criterion.
    k is the number of parameters.
    n is the number of data points.
    L is the maximised value of the likelihood function.
    """
    return k * jnp.log(n) - 2 * jnp.log(L)

def addnoise(x: Float[Array, "..."], rng: PRNGKeyArray,  
             stddev: float=0.0) -> Float[Array, "..."]:
    """Adds Gaussian noise with given standard deviation to `x` using a given `PRNGKey`.

    Uses `jax.lax.cond` (rather than a Python `if`) so `stddev` can be traced; the
    noise is always sampled, and is only added when `stddev > 0.0`.

    Args:
        x: Input tensor of arbitrary shape `(...)`.
        rng: JAX pseudo-random number generator key.
        stddev: Standard deviation of normal noise. If `<= 0.0`, returns `x` unchanged.

    Returns:
        Tensor of shape `(...)` with additive Gaussian noise.
    """
    # use jax.lax.cond instead of Python if for JAX tracing compatibility
    noise = jax.random.normal(rng, shape=x.shape, dtype=x.dtype)
    return jax.lax.cond(
        stddev > 0.0,
        lambda _: x + (noise * stddev),
        lambda _: x,
        operand=None
    )

def addnoise_batchnorm(x: Float[Array, "... d"], rng: PRNGKeyArray, 
                       stddev: float=0.0) -> Float[Array, "... d"]:
    """As `addnoise`, but scaled by batchnorm."""
    noise = jax.random.normal(rng, shape=x.shape, dtype=x.dtype)
    
    # Calculate the per-sample norm
    # For SONAR this is 1.0, but this makes the code robust to other data
    # (stop_gradient: a fixed noise scale, and norm has a NaN gradient at 0)
    x_norm = jax.lax.stop_gradient(jnp.linalg.norm(x, axis=-1, keepdims=True))
    
    # Scale noise so it's a percentage of the signal magnitude
    return x + (noise * stddev * x_norm / jnp.sqrt(x.shape[-1]))

def addnoise_featvar(x: Float[Array, "b ..."], rng: PRNGKeyArray, 
                     stddev: float=0.0) -> Float[Array, "b ..."]:
    """As `addnoise`, but each feature is scaled by its standard deviation in the bach."""
    noise = jax.random.normal(rng, shape=x.shape, dtype=x.dtype)
    
    # Standard deviation of each feature in the current batch, as a constant
    # noise scale: stop_gradient so the model can't shrink feature variance
    # to evade its own noise, and an eps in the sqrt because tiny features
    # underflow the f32 variance to exactly 0, where the unguarded gradient
    # (x - mean) / (b * std) is 0/0 = NaN (killed a run at step 300476)
    feature_var = jnp.var(x, axis=0, keepdims=True)
    feature_std = jax.lax.stop_gradient(
            jnp.sqrt(feature_var + jnp.finfo(x.dtype).eps))

    return x + (noise * stddev * feature_std)


def l2_ghost(W: Float[Array, "d_in d_out"], X: Float[Array, "... b d_in"], 
             Y: Float[Array, "... b d_out"], Yhat:Float[Array, "... b d_out"],
             ) -> Float[Array, ""]:
    """Ghost grad MSE/L2 loss."""
    isdead = jnp.all(Yhat == 0.0, axis=-2, keepdims=True)
    if jnp.any(isdead):
        R = Y - jax.lax.stop_gradient(Yhat)
        X_clipped = jnp.clip(X, -10.0, 10.0)
        X_ghost = jnp.exp(X_clipped) * isdead
        Y_ghost = jnp.einsum("ij, ...i -> ...j", W, X_ghost)
        return l2(Y_ghost, R)
    return jnp.array(0.0, dtype=Y.dtype)

def ghost(X: Float[Array, "... b d"], isdead: Bool[Array, "d"],
          lbound: float=-10.0, ubound: float=10.0) -> Float[Array, "... b d"]:
    """Ghost grad activation. Masks active features; 
    clips and applies exponential to dead features."""
    # Clip X heavily to prevent overflow when applying exp and scaling linearly
    X_clipped = jnp.clip(X, lbound, ubound)
    return jnp.exp(X_clipped) * isdead

def ghostgrad(W: Float[Array, "d_out d_in"],
          X: Float[Array, "... d_in"], Y: Float[Array, "... d_out"],
          lbound: float=-10.0, ubound: float=10.0) -> Float[Array, "... d_out"]:
    """Ghost grad for dead features in `Y`. Computes `exp(W @ X)` with dead features in
    `W` masked. Clips `W @ X` to (lbound, ubound)` to prevent overflow."""
    isdead = jnp.all(Y == 0.0, axis=-2, keepdims=True)
    Y_ghost = jnp.einsum("oi, ...i -> ...o", W, X)
    return ghost(Y_ghost, isdead, lbound, ubound)
