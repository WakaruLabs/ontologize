import jax
import jax.numpy as jnp
import flax.linen as nn
import einops
from typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Float, PRNGKeyArray

from ontologize.fns.keys import get_activation, get_dtype, get_noise
from ontologize.fns.loss import ghostgrad, l1, batchmean, bcossim, entropy

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
        self.fn = get_activation(self.activation)
        self.fn_noise = get_noise(self.noise)
        self.dtype = get_dtype(self.dtype_str)
        self.dtype_p = get_dtype(self.dtype_p_str)

    def l1(self, F: Float[Array, "b ..."]) -> Float[Array, ""]:
        """Mean over the batch of each sample's L1 norm (`fns.loss.batchmean`),
        so the stat does not change strength with batch size. 
        If `sparse=False`, blocks gradients so it is only tracked as a 
        validation metric and doesn't affect backprop."""
        F = F.astype(self.dtype)
        if not self.sparse:
            F = jax.lax.stop_gradient(F)
        return l1(F, batchmean)

    def cossim(self, F: Float[Array, "... d"],
               S: Optional[Float[Array, "..."]]=None,
               isloss: Optional[bool]=None) -> Float[Array, ""]:
        """`isloss` overrides the gradient gate; defaults to `self.cossim_loss`
        (`self.bcossim` passes `self.bcossim_loss` instead)."""
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
        """As `cossim`, but flattens the last two dimensions."""
        Fs = Fs.astype(self.dtype)
        F = einops.rearrange(Fs, "... h d -> ... (h d)")
        return self.cossim(F, *args, isloss=self.bcossim_loss, **kwargs)

    def entropy(self, K: Float[Array, "... d"],
                S: Optional[Float[Array, "..."]] = None) -> Float[Array, ""]:
        """Shannon entropy over last dimension of `K`. Blocks gradient if
        `self.entropy_loss=False`."""
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
        """Adds noise to `X` using `self.fn_noise`. If `rng` is provided,
        returns split key as its second value."""
        X = X.astype(self.dtype)
        if rng is not None:
            rng_next, rng = jax.random.split(rng)
            return self.fn_noise(X, rng, sd), rng_next
        return X, None

    def addbias(self, Y: Float[Array, "... d"],
                bias: Optional[Float[Array, "d"]]
                ) -> Float[Array, "... d"]:
        """Adds 1D bias vector to `Y`. `bias` is a separate argument, as 
        not all subclasses of `Sparse` support bias."""
        if bias is not None:
            bias = bias.astype(self.dtype)
            return Y + bias
        return Y

    def haddbias(self, Y: Float[Array, "... h d"],
                bias: Optional[Float[Array, "h d"]]
                ) -> Float[Array, "... h d"]:
        """Adds 2D bias vector to `Y`. `bias` is a separate argument, as 
        not all subclasses of `Sparse` support bias."""
        if bias is not None:
            bias = bias.astype(self.dtype)
            return Y + bias
        return Y

    def ghost(self, X: Float[Array, "..."], Y: Float[Array, "..."], *args, **kwargs
              ) -> Optional[Float[Array, "..."]]:
        return None

