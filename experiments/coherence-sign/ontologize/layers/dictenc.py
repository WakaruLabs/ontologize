import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Tuple, List
from jaxtyping import Array, Float, UInt, PRNGKeyArray
import einops

from ontologize.fns.loss import ghost, identity
from .dictblock import DictBlock
from .dictblock import entropy, l1, bcossim
from .nlinear import NLinear, NLinearBlock, Bilinear, BilinearBlock
from .linear import Linear, get_activation, get_dtype

class DictEnc(nn.Module):
    """Dictionary Encoder: one `Ontologizer` layer, composed of a multihead
    classifier (`BilinearBlock` for `n=2`, else `NLinearBlock`), a
    `DictBlock`, and an optional per-head scaling (`Bilinear` or `NLinear`).
    The encoder and decoder belong to the `Ontologizer`."""
    d_in: int = 0
    d_out: int = 0
    k: int = 0
    h: int = 0

    n: int = 2
    gate: str = "none"
    activation_cl: str = "none"
    biased_cl: bool = False

    select: str = "softmax"
    activation_dict: str = "none"

    scaled: bool = False
    n_sc: int = 2
    activation_scale: str = "none"
    gate_scale: str = "none"
    biased_scale: bool = False

    sparse_K: bool = False
    sparse_F: bool = False
    entropy_loss: bool = False
    cossim_loss: bool = False
    bcossim_loss: bool = False
    hmean_loss: bool = False
    pwak_loss: bool = False

    noise_K: str = "none"
    noise_F: str = "none"
    sd_K: float = 0.0
    sd_F: float = 0.0

    dtype_str: str = "bfloat16"
    dtype_p_str: str = "float32"

    def setup(self):
        self.n_tags = self.k * self.h
        self.n_feat = self.d_in * self.h
        self.dtype = get_dtype(self.dtype_str)
        self.dtype_p = get_dtype(self.dtype_p_str)

        ClType = BilinearBlock if self.n == 2 else NLinearBlock
        ScType = Bilinear if self.n == 2 else NLinear

        self.classifier = ClType(
            d_in=self.d_in, 
            d_out=self.k, 
            h=self.h, n=self.n,
            biased=self.biased_cl, 
            gate=self.gate, 
            activation=self.activation_cl,
            sparse=self.sparse_K,
            noise=self.noise_K, sd=self.sd_K,
            dtype_str=self.dtype_str,
            dtype_p_str=self.dtype_p_str
        )
        self.dict = DictBlock(
            k=self.k, 
            d=self.d_out, 
            h=self.h,
            select=self.select,
            activation=self.activation_dict,
            sparse=self.sparse_F, entropy_loss=self.entropy_loss,
            cossim_loss=self.cossim_loss, bcossim_loss=self.bcossim_loss,
            hmean_loss=self.hmean_loss, pwak_loss=self.pwak_loss,
            noise=self.noise_F, sd=self.sd_F,
            dtype_str=self.dtype_str,
            dtype_p_str=self.dtype_p_str
        )

        if self.scaled:
            self.scaling = ScType(
                d_in=self.d_in, 
                d_out=self.h, n=self.n,
                biased=self.biased_scale, 
                gate=self.gate_scale, 
                activation=self.activation_scale
            )

    def fwd_dict(self, P: Float[Array, "... h k"], *args, **kwargs
                 ) -> Float[Array, "... d_out"]:
        """Forward `DictBlock` pass followed by `self.fwd_dec`, which this
        class does not define (the decoder is the `Ontologizer`'s), so
        calling it raises `AttributeError`. Unused."""
        return self.fwd_dec(self.dict.fwd(P, *args, **kwargs))

    def fwd(self, E: Float[Array, "... d_in"], *args, **kwargs
               ) -> Float[Array, "... d_out"]:
        """Forward pass starting from `self.classifier` with only unbiased linear
        transformations. Ends in `fwd_dict`, so it raises too. Unused."""
        K = self.classifier.fwd(E)
        if self.scaled:
            S = self.scaling.fwd(E)
            return self.fwd_dict(K, S, *args, **kwargs)
        return self.fwd_dict(K, *args, **kwargs)

    def rev_dict(self, Y: Float[Array, "... d_out"], *args, **kwargs
                 ) -> Float[Array, "... h k"]:
        """`DictBlock.rev`: projects a dictionary-space output onto each
        head's atoms, giving `(..., h, k)`."""
        return self.dict.rev(Y, *args, **kwargs)

    def rev(self, Y: Float[Array, "... d_out"], *args, **kwargs
               ) -> Float[Array, "... d_in"]:
        """`rev_dict`, then the classifier's `rev` (`BilinearBlock.rev` for
        `n=2`), back to the layer input. Does not account for
        `self.scaling`. `S` must be passed explicitly."""
        K = self.rev_dict(Y, *args, **kwargs)
        return self.classifier.rev(K)

    def classify(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... h k"]:
        """Forward pass up to `DictBlock.cluster`. Returns the classification tensor."""
        K = self.classifier(E)
        return self.dict.cluster(K, *args, **kwargs)

    def scale(self, E: Float[Array, "... d_in"]) -> Float[Array, "... h"]:
        """Forward pass for `self.scaling`, which returns the scaling vector for the output of
        `self.dict`. If `self.scaled=False`, returns vector of all 1s."""
        if self.scaled:
            return jnp.abs(self.scaling(E))
        shape = E.shape[:-1] + (self.h,)
        return jnp.full(shape, 1.0, dtype=self.dtype)

    def __call__(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... d_out"]:
        """Forward pass through `self.dict.fwd`. Reconstructs but does not unembed the input.
        """
        P = self.classify(E)
        if self.scaled:
            S = self.scaling(E)
            return self.dict.fwd(P, S, *args, **kwargs)
        return self.dict.fwd(P, *args, **kwargs)

    def tags(self) -> Float[Array, "n_tags d_out"]:
        """Flattens `self.dict.dicts()` to shape `(h * k, d)`. This allows the weights to be
        treated as a batch of synthetic data."""
        return self.dict.tags()

    def decodeUniform(self, *args, **kwargs) -> Float[Array, "n_tags d_out"]:
        """`DictBlock.fwd` of `self.dict.uniformTags()`, in dictionary space
        (not decoded). After one all-uniform output, for each tag this
        generates a synthetic output where that tag has probability 1, other
        tags in the same head have probability 0, and all other tags have
        probability `1 / k`."""
        K = self.dict.uniformTags()
        return self.dict.fwd(K, *args, **kwargs)

    def withClusts(self, R: Float[Array, "... d_out"], 
                   E: Float[Array, "... d_in"], *args, **kwargs
                 ) -> Tuple[Float[Array, "... d_out"], Float[Array, "... n_tags"]]:
        """Forward pass that adds the output to a residual tensor `R` and returns
        the flattened classifications `K` as the second value. This is used to build an
        `Ontologizer` from sequential `DictEnc`s, each of which adds its output to
        `R` and returns `K`, which the next `DictEnc` reads under labels
        forwarding."""
        K_0 = self.classifier(E)
        if self.scaled:
            S = self.scale(E)
            F, K = self.dict.withClusts(K_0, S, *args, **kwargs)
        else:
            F, K = self.dict.withClusts(K_0, *args, **kwargs)
        Y = self.dict.combine(F)
        return R + Y, einops.rearrange(K, "... h k -> ... (h k)")
    
    def withStats(self, R: Float[Array, "... d_out"],
                  E: Float[Array, "... d_in"], sd_K: float=0.0, sd_F: float=0.0,
                  rng: Optional[PRNGKeyArray]=None,
                  *args, p_drop: float=0.0, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "7"], PRNGKeyArray]:
        """As `withClusts`, but also returns validation statistics
        `[L1_K, L1_F, entropy, cossim_batch, cossim_heads, KL_mean, KL_pwak]`.
        When `rng` is given, adds noise of scale `sd_K` (the classifier's
        noise function) to `K` before passing it to `self.dict`, which adds
        noise of scale `sd_F` to its features. `L1_K` is taken on the clean
        `K`. The returned `R + F` carries the feature noise, so it reaches
        the residual input of every later layer.
        Splits `rng` so the two noises use different keys."""
        K = self.classifier(E)

        K_n, rng_F = self.classifier.addnoise(K, sd_K, rng)
        L1_K = self.classifier.l1(K)

        if self.scaled:
            S = self.scale(E)
        else:
            S = None

        F, K, stats, rng_next = self.dict.withStats(
                K_n, S, *args, **kwargs, sd=sd_F, rng=rng_F, p_drop=p_drop,
                E_cl=E)

        K = einops.rearrange(K, "... h k -> ... (h k)")
        stats = jnp.insert(stats, 0, L1_K)
        return R + F, K, stats, rng_next

    def withGhost(self, R: Float[Array, "... d_out"], R_g: Float[Array, "... d_out"],
                  E: Float[Array, "... d_in"], E_g: Optional[Float[Array, "... d_in"]],
                  temperature: float=1.0, sd_K: float=0.0, sd_F: float=0.0,
                  rng: Optional[PRNGKeyArray]=None, *args, p_drop: float=0.0,
                  **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "7"], PRNGKeyArray]:
        """As `withStats`, but also returns ghost gradient output."""
        K = self.classifier(E)

        K_n, rng_F = self.classifier.addnoise(K, sd_K, rng)
        L1_K = self.classifier.l1(K)
        K_g = self.classifier.ghost(E, K)
        if E_g is not None:
            K_g = K_g + self.classifier.fwd(E_g)
        if self.scaled:
            S = self.scale(E)
            S_g = self.scaling.ghost(E, S)
        else:
            S = None
            S_g = None

        F, P, stats, rng_next = self.dict.withStats(
                K_n, S, sd=sd_F, rng=rng_F, temperature=temperature,
                p_drop=p_drop, E_cl=E, *args, **kwargs)
        F_R = self.dict.fwd(K_g, S_g, *args, **kwargs)
        F_g = self.dict.ghost(K, F, S) + F_R

        P = einops.rearrange(P, "... h k -> ... (h k)")
        
        stats = jnp.insert(stats, 0, L1_K)

        return R + F, R_g + F_g, P, stats, rng_next

    def intervene(self, E: Float[Array, "... d_in"],
                  *args, temperature: float=1.0, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "... d_in"]]:
        """Classifies `E`, applies `DictBlock.intervene` arguments to the
        classification, and reverses the intervened classification through the
        classifier to reconstruct an input that would produce it. Returns the
        intervened layer output, the flattened intervened classification, and
        the reconstructed input."""
        K = self.classifier(E)
        P = self.dict.cluster(K, temperature)
        P_int = self.dict.intervene(P, *args, **kwargs)
        E_int = self.classifier.rev(P_int)
        if self.scaled:
            S = self.scale(E)
        else:
            S = None
        # P_int is already intervened; do not pass *args again or additive/
        # scaling interventions would be applied twice.
        Fs = self.dict.hfwd(P_int, S)
        F = self.dict.combine(Fs)
        return F, einops.rearrange(P_int, "... h k -> ... (h k)"), E_int

