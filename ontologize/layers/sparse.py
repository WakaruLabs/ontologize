"""Base layer providing conditional gradient gating and statistical metrics.

Defines the `Sparse` base Flax Linen module, which handles string-keyed
activation and dtype resolution, noise injection, and conditional stop-gradient
operations for L1 sparsity, batch cosine similarity, and Shannon entropy.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import einops
from typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from ontologize.fns.keys import get_activation, get_dtype, get_noise
from ontologize.fns.loss import ghostgrad, l1, bcossim, entropy

class Sparse(nn.Module):
    """Superclass providing methods to conditionally stop gradients when computing 
    sparsity statistics and handling specification of activation functions, noise functions,
    and dtypes as strings."""
    activation: str = "none"

    sparse: bool = False
    entropy_loss: bool = False
    cossim_loss: bool = False
    bcossim_loss: bool = False

    noise: str = "none"
    sd: float = 0.0 # ignored; passed as argument to addnoise instead

    dtype_str: str = "bfloat16"
    dtype_p_str: str = "float32"

    def setup(self):
        """Resolves string keys into callables and JAX dtypes.

        Initializes `self.fn`, `self.fn_noise`, `self.dtype`, and `self.dtype_p`.
        """
        self.fn = get_activation(self.activation)
        self.fn_noise = get_noise(self.noise)
        self.dtype = get_dtype(self.dtype_str)
        self.dtype_p = get_dtype(self.dtype_p_str)

    def l1(self, F: Float[Array, "..."]) -> Float[Array, ""]:
        """Computes the L1 norm. If `sparse=False`, blocks gradients so it 
        is only tracked as a validation metric and doesn't affect backprop.

        Args:
            F: Input array of arbitrary shape `(...)`.

        Returns:
            Scalar float array `()` with total L1 norm.
        """
        F = F.astype(self.dtype)
        if self.sparse:
            return l1(F)

        return l1(jax.lax.stop_gradient(F))

    def cossim(self, F: Float[Array, "... d"],
               S: Optional[Float[Array, "..."]]=None,
               isloss: Optional[bool]=None) -> Float[Array, ""]:
        """Computes mean absolute batch cosine similarity over the sample axis.

        `isloss` overrides the gradient gate; defaults to `self.cossim_loss`
        (`self.bcossim` passes `self.bcossim_loss` instead).

        Args:
            F: Activation tensor of shape `(..., d)` where dim -2 is sample axis.
            S: Optional weighting tensor broadcasting with the batch similarity matrix.
            isloss: If True, allows gradients through `F`; if False, stops gradients.

        Returns:
            Scalar float array `()` with mean absolute cosine similarity.
        """
        if isloss is None:
            isloss = self.cossim_loss
        F = F.astype(self.dtype)
        if isloss:
            D = bcossim(F)
        else:
            D = bcossim(jax.lax.stop_gradient(F))
        if S is not None:
            S = S.astype(self.dtype)
            D = D * S
        return jnp.abs(D).mean()

    def bcossim(self, Fs: Float[Array, "... h d"],
                *args, **kwargs) -> Float[Array, ""]:
        """Computes batch cosine similarity across multihead activations.

        Flattens multihead dimensions `(..., h, d)` to `(..., h * d)` and delegates
        to `self.cossim` with `isloss=self.bcossim_loss`.

        Args:
            Fs: Multihead activation tensor of shape `(..., h, d)`.
            *args: Positional arguments forwarded to `self.cossim`.
            **kwargs: Keyword arguments forwarded to `self.cossim`.

        Returns:
            Scalar float array `()`.
        """
        Fs = Fs.astype(self.dtype)
        F = einops.rearrange(Fs, "... h d -> ... (h d)")
        return self.cossim(F, *args, isloss=self.bcossim_loss, **kwargs)

    def entropy(self, K: Float[Array, "... d"],
                S: Optional[Float[Array, "..."]] = None) -> Float[Array, ""]:
        """Computes mean Shannon entropy in bits across the trailing feature axis.

        If `self.entropy_loss` is False, stops gradients through `K`.

        Args:
            K: Classification probability tensor of shape `(..., d)`.
            S: Optional weighting tensor broadcasting with `K.shape[:-1]`.

        Returns:
            Scalar float array `()` containing mean entropy in bits.
        """
        K = K.astype(self.dtype)
        if self.entropy_loss:
            H = entropy(K)
        else:
            H = entropy(jax.lax.stop_gradient(K))
        if S is not None:
            S = S.astype(self.dtype)
            H = H * S
        return H.mean()

    def addnoise(self, X: Float[Array, "..."], sd: float=0.0,
                 rng: Optional[PRNGKeyArray]=None
                 ) -> Tuple[Float[Array, "..."],
                            Optional[PRNGKeyArray]]:
        """Injects random noise into tensor `X` if PRNG key is provided.

        Args:
            X: Input tensor of arbitrary shape `(...)`.
            sd: Noise standard deviation multiplier.
            rng: Optional JAX PRNG key. If None, no noise is injected.

        Returns:
            Tuple of `(X_noisy, rng_next)` where `X_noisy` matches `X.shape`, and
            `rng_next` is the advanced PRNG key (or None if `rng` was None).
        """
        X = X.astype(self.dtype)
        if rng is not None:
            rng_next, rng = jax.random.split(rng)
            return self.fn_noise(X, rng, sd), rng_next
        return X, None

    def addbias(self, Y: Float[Array, "... d"],
                bias: Optional[Float[Array, "d"]]
                ) -> Float[Array, "... d"]:
        """Adds bias vector to trailing dimension `d` if present.

        Args:
            Y: Activation tensor of shape `(..., d)`.
            bias: Optional bias vector of shape `(d,)`.

        Returns:
            Tensor of shape `(..., d)` with bias added, or `Y` unchanged if `bias` is None.
        """
        if bias is not None:
            bias = bias.astype(self.dtype)
            return Y + bias
        return Y

    def haddbias(self, Y: Float[Array, "... h d"],
                bias: Optional[Float[Array, "h d"]]
                ) -> Float[Array, "... h d"]:
        """Adds multihead bias tensor to trailing dimensions `(h, d)` if present.

        Args:
            Y: Multihead activation tensor of shape `(..., h, d)`.
            bias: Optional multihead bias tensor of shape `(h, d)`.

        Returns:
            Tensor of shape `(..., h, d)` with bias added, or `Y` unchanged if `bias` is None.
        """
        if bias is not None:
            bias = bias.astype(self.dtype)
            return Y + bias
        return Y

    def ghost(self, X: Float[Array, "..."], Y: Float[Array, "..."], *args, **kwargs
              ) -> Optional[Float[Array, "..."]]:
        """Computes ghost gradient activations for dead features.

        Base placeholder implementation returning None. Subclasses override this method
        to generate ghost gradient signals reactivating inactive dictionary entries.

        Args:
            X: Layer input tensor `(...)`.
            Y: Layer output tensor `(...)`.
            *args: Additional arguments forwarded to subclass implementations.
            **kwargs: Additional keyword arguments.

        Returns:
            None in base class `Sparse`. Subclasses return ghost output tensor `(...)`.
        """
        return None

