import jax
import jax.numpy as jnp
from typing import Any, Callable, List, Optional, Tuple, TypeVar
from jaxtyping import Array, Bool, Float, UInt, PRNGKeyArray
import einops

T = TypeVar('T') # generic type
def identity(x: T, *args, **kwargs) -> T:
    return x

def recip_norm(sq: Float[Array, "..."]) -> Float[Array, "..."]:
    """`1/sqrt(sq)` where `sq` is positive, and 0 where it is not.

    The degenerate case is handled by definition rather than by adding an
    epsilon to `sq`, because that epsilon is absolute while the quantities
    here carry no guaranteed scale: once `sq` falls to the same order as
    `jnp.finfo.eps` the guard dominates and the cosine is silently
    deflated toward zero. That makes any cosine statistic built this way
    reducible by shrinking the vectors rather than by separating their
    directions -- which an optimizer under pressure will find. Defining a
    zero vector's cosine as 0 costs nothing (it has no direction) and
    leaves every nonzero vector exact at any scale.

    The `where` is doubled so the gradient is also finite at `sq = 0`: the
    inner one keeps the reciprocal-sqrt off the singular branch, and the
    outer one selects it away."""
    safe = jnp.where(sq > 0, sq, jnp.ones_like(sq))
    return jnp.where(sq > 0, jax.lax.rsqrt(safe), jnp.zeros_like(sq))

def cossim(x: Float[Array,  "d"], y: Float[Array, "d"]) -> Float[Array, ""]:
    """Pairwise cosine similarity of two 1D arrays."""
    return (jnp.dot(x, y) * recip_norm(jnp.sum(x ** 2))
            * recip_norm(jnp.sum(y ** 2)))

def bcossim(X: Float[Array, "... b d"]) -> Float[Array, "... b b"]:
    """Cosine similarity matrix for all samples. The -2 axis is assumed to be the batch dimension."""
    X_norm = X * recip_norm((X ** 2).sum(axis=-1, keepdims=True))
    return jnp.einsum("...id,...jd->...ij", X_norm, X_norm) 

def entropy(X: Float[Array, "... d"]) -> Float[Array, "..."]:
    """Shannon entropy."""
    X = X + jnp.finfo(X.dtype).eps
    X = X / jnp.sum(X, axis=-1, keepdims=True)
    return -jnp.sum(X * jnp.log2(X), axis=-1)

def l0(x: Float[Array, "..."], reduction: Callable = jnp.sum
       ) -> Float[Array, ""]:
    """Applies `reduction` to `abs(x) > 0`. When `reduction=jnp.sum` this gives L0 loss."""
    return reduction(jnp.abs(x) > 0)

def batchmean(X: Float[Array, "b ..."]) -> Float[Array, ""]:
    """Reduction for `l1` (which has already taken the absolute value):
    each sample's total over its feature axes, averaged over the batch
    (axis 0). Independent of batch size, so a weight calibrated against it
    keeps its strength when `b` changes -- a bare `jnp.sum` does not.
    Matches sae.py's convention for the same quantity (`|z|_1` per sample,
    meaned over the batch), which makes the Ontologizer and SAE L1 columns
    directly comparable."""
    return X.reshape(X.shape[0], -1).sum(-1).mean()

def l1(x: Float[Array, "..."], reduction: Callable = jnp.sum
       ) -> Float[Array, ""]:
    """Applies `reduction` to `abs(x)`. When `reduction=jnp.sum`, this gives L1 loss."""
    return reduction(jnp.abs(x))

def l2(x: Float[Array, "..."], y: Float[Array, "..."], 
       reduction: Callable = jnp.mean) -> Float[Array,""]:
    """Applies `reduction` to `(x - y) ** 2`. When `reduction=jnp.mean`, this gives
    MSE/L2 loss."""
    return reduction((x - y)**2)

def errorct(pred: Float[Array, "... d"], target: UInt[Array, "..."]
            ) -> UInt[Array, ""]:
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
    """Adds Gaussian noise with given standard deviation to `x` using a given `PRNGKey`."""
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
