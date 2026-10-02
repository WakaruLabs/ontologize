"""Single-layer dictionary encoder combining classifier, dictionary lookup, and scaling.

Defines `DictEnc`, a composite Flax Linen module that wraps:
- A multihead classifier (`BilinearBlock` or `NLinearBlock`) mapping `(..., d_in) -> (..., h, k)`
- An ontofeature dictionary (`DictBlock`) mapping `(..., h, k) -> (..., d_out)`
- An optional scaling module (`Bilinear` or `NLinear`) mapping `(..., d_in) -> (..., h)`

Supports training with per-layer statistics, ghost gradients for dead feature reactivation,
and causal intervention round-trips.
"""
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
from .headsparse import HeadSparseEncoder, HeadBilinear

class DictEnc(nn.Module):
    """Dictionary Encoder composed of a dense linear encoder, multihead bilinear classifier,
    `DictBlock`, dense linear decoder, and optional bilinear scaling.
    TODO: move encoder/decoder to `Ontologizer`"""
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

    noise_K: str = "none"
    noise_F: str = "none"
    sd_K: float = 0.0
    sd_F: float = 0.0

    dtype_str: str = "bfloat16"
    dtype_p_str: str = "float32"

    # standardize logits per head across `k` (zero mean, unit std) before
    # noise, winner dropout and the temperature. Bilinear logits have a free
    # scale, so without this the classifier can shrink its weights to cancel
    # any temperature schedule; with it, `temperature` alone sets the spread.
    logit_norm: bool = False
    # per-head sparse encoder in front of the classifier: each head first
    # encodes the input as `m_h` JumpReLU (or top-k_z) features and classifies
    # with a small per-head bilinear map on them. Its L0 replaces the L1_K
    # stat, so `Hyperparams.s_L1K` acts as the sparsity coefficient.
    head_sparse: str = "none"  # "none" | "jumprelu" | "topk"
    # one encoder per layer shared by all heads (features broadcast to every
    # head's classifier) instead of one per head
    hs_shared: bool = False
    m_h: int = 32
    k_z: int = 4
    hs_bandwidth: float = 1e-3
    hs_init_threshold: float = 1e-3
    # auxiliary SAE objective on the sparse encoder (see
    # `HeadSparseEncoder.decoder`); its loss fills the `aux` stats slot,
    # weighted by `Hyperparams.s_aux`
    hs_decoder: bool = False
    hs_auxk: int = 0
    # see `DictBlock.fiber_rank`: the per-head coordinates c_h are a linear
    # read of the layer input, c_h = A_h E
    fiber_rank: int = 0
    fiber_drop: float = 0.0
    fiber_bound: float = 0.0
    fiber_eps: float = 0.0
    fiber_eps_layer: float = 0.0
    fiber_radial: bool = False
    # one coordinate vector per layer, shared by all heads (each selected
    # entry still has its own basis), so the continuous channel is only
    # fiber_rank numbers per layer
    fiber_shared: bool = False
    # see `DictBlock.fast_stats`, `.private`, `.signed`, `.p_head_drop`
    fast_stats: bool = False
    private_dict: bool = False
    signed_dict: bool = False
    p_head_drop: float = 0.0

    def setup(self):
        """Initializes classifier, dictionary, and optional scaling submodules.

        Instantiates `self.classifier` (`BilinearBlock` or `NLinearBlock`),
        `self.dict` (`DictBlock`), and optional `self.scaling` (`Bilinear` or `NLinear`).
        """
        self.n_tags = self.k * self.h
        self.n_feat = self.d_in * self.h
        self.dtype = get_dtype(self.dtype_str)
        self.dtype_p = get_dtype(self.dtype_p_str)

        ClType = BilinearBlock if self.n == 2 else NLinearBlock
        ScType = Bilinear if self.n == 2 else NLinear

        if self.head_sparse != "none":
            self.hs_encoder = HeadSparseEncoder(
                h=1 if self.hs_shared else self.h,
                d_in=self.d_in, m=self.m_h, mode=self.head_sparse,
                k_z=self.k_z, bandwidth=self.hs_bandwidth,
                init_threshold=self.hs_init_threshold, dtype_str=self.dtype_str,
                decoder=self.hs_decoder, auxk=self.hs_auxk)
            self.classifier = HeadBilinear(
                h=self.h, m=self.m_h, k=self.k, biased=self.biased_cl,
                sparse=self.sparse_K, noise=self.noise_K, sd=self.sd_K,
                dtype_str=self.dtype_str, dtype_p_str=self.dtype_p_str)
        else:
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
        if self.fiber_rank:
            n_read = 1 if self.fiber_shared else self.h
            self.fiber_read = self.param(
                'fiber_read',
                nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
                (n_read, self.fiber_rank, self.d_in), dtype=self.dtype_p)
        self.dict = DictBlock(
            k=self.k, 
            d=self.d_out, 
            h=self.h,
            select=self.select,
            activation=self.activation_dict,
            sparse=self.sparse_F, entropy_loss=self.entropy_loss,
            cossim_loss=self.cossim_loss, bcossim_loss=self.bcossim_loss,
            hmean_loss=self.hmean_loss,
            noise=self.noise_F, sd=self.sd_F,
            fast_stats=self.fast_stats,
            private=self.private_dict, signed=self.signed_dict,
            p_head_drop=self.p_head_drop, fiber_rank=self.fiber_rank,
            fiber_drop=self.fiber_drop, fiber_bound=self.fiber_bound,
            fiber_eps=self.fiber_eps, fiber_eps_layer=self.fiber_eps_layer, fiber_radial=self.fiber_radial,
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
        """Forward `DictBlock` pass followed by `fwd_dec`.

        Note: References `self.fwd_dec`, which is defined on `Ontologizer` rather than `DictEnc`.

        Args:
            P: Classification probabilities of shape `(..., h, k)`.
            *args: Positional arguments forwarded to `self.dict.fwd`.
            **kwargs: Keyword arguments forwarded to `self.dict.fwd`.

        Returns:
            Reconstruction tensor of shape `(..., d_out)`.
        """
        return self.fwd_dec(self.dict.fwd(P, *args, **kwargs))

    def fwd(self, E: Float[Array, "... d_in"], *args, **kwargs
               ) -> Float[Array, "... d_out"]:
        """Forward pass starting from `self.classifier` with only unbiased linear
        transformations."""
        K = self.classifier.fwd(E)
        if self.scaled:
            S = self.scaling.fwd(E)
            return self.fwd_dict(K, S, *args, **kwargs)
        return self.fwd_dict(K, *args, **kwargs)

    def rev_dict(self, Y: Float[Array, "... d_out"], *args, **kwargs
                 ) -> Float[Array, "... h k"]:
        """Reversal of `fwd_dec` and `fwd_dict`.

        Args:
            Y: Output tensor of shape `(..., d_out)`.
            *args: Positional arguments forwarded to `self.dict.rev`.
            **kwargs: Keyword arguments forwarded to `self.dict.rev`.

        Returns:
            Classification projection tensor of shape `(..., h, k)`.
        """
        return self.dict.rev(Y, *args, **kwargs)

    def rev(self, Y: Float[Array, "... d_out"], *args, **kwargs
               ) -> Float[Array, "... d_in"]:
        """Reversal of `fwd_dec`, `fwd_dict`, and `fwd_cl`. Does not account for 
        `self.scaling`. `S` must be passed explicitly."""
        K = self.rev_dict(Y, *args, **kwargs)
        return self.classifier.rev(K)

    def logits(self, E: Float[Array, "... d_in"]) -> Float[Array, "... h k"]:
        """Classifier logits, standardized per head when `logit_norm`."""
        if self.head_sparse != "none":
            Z = self.hs_encoder(E)          # (..., 1, m) when hs_shared; the classifier broadcasts
            K = self.classifier(Z)
        else:
            K = self.classifier(E)
        if not self.logit_norm:
            return K
        K = K.astype(self.dtype)
        mu = K.mean(-1, keepdims=True)
        # a fixed tiny eps: bilinear logits at init have variance ~1/d_in^2,
        # comparable to finfo.eps, which would bias the standardization
        sd = jnp.sqrt(((K - mu) ** 2).mean(-1, keepdims=True) + 1e-12)
        return (K - mu) / sd

    def fiber_coords(self, E: Float[Array, "... d_in"]
                     ) -> Optional[Float[Array, "... h r"]]:
        """Per-head fiber coordinates c_h = A_h E (None without fibers)."""
        if not self.fiber_rank:
            return None
        A = self.fiber_read.astype(self.dtype)
        C = jnp.einsum("...i,hri->...hr", E.astype(self.dtype), A)
        if self.fiber_shared:
            C = jnp.broadcast_to(C, C.shape[:-2] + (self.h, self.fiber_rank))
        return C

    def stat_aux(self, E) -> Float[Array, ""]:
        """The `aux` stats slot: the sparse encoder's SAE loss, or 0."""
        if self.head_sparse != "none" and self.hs_decoder:
            return self.hs_encoder.recon_loss(E)
        return jnp.zeros((), self.dtype)

    def stat_K(self, E, K):
        """The L1_K stat slot: the classifier logits' L1, or with `head_sparse`
        the encoders' L0 (differentiable via the JumpReLU STE), so that
        `s_L1K` weights the sparsity penalty."""
        if self.head_sparse != "none":
            return self.hs_encoder.l0(E)
        return self.classifier.l1(K)

    def classify(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... h k"]:
        """Forward pass up to `DictBlock.cluster`. Returns the classification tensor.

        Args:
            E: Input tensor of shape `(..., d_in)`.
            *args: Arguments forwarded to `self.dict.cluster`.
            **kwargs: Keyword arguments forwarded to `self.dict.cluster` (e.g. `temperature`).

        Returns:
            Probability tensor of shape `(..., h, k)`.
        """
        K = self.logits(E)
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

        Args:
            E: Input tensor of shape `(..., d_in)`.
            *args: Arguments forwarded to `self.dict.fwd`.
            **kwargs: Keyword arguments forwarded to `self.dict.fwd`.

        Returns:
            Reconstructed output tensor of shape `(..., d_out)`.
        """
        P = self.classify(E)
        C = self.fiber_coords(E)
        if self.scaled:
            S = self.scaling(E)
            return self.dict.fwd(P, S, *args, C=C, **kwargs)
        return self.dict.fwd(P, *args, C=C, **kwargs)

    def tags(self) -> Float[Array, "n_tags d_out"]:
        """Flattens `sel.dict.dicts()` to shape `(h * k, d)`. This allows the weights to be
        treated as a batch of synthetic data."""
        return self.dict.tags()

    def decodeUniform(self, *args, **kwargs) -> Float[Array, "n_tags d_out"]:
        """Unembeds `self.dict.uniformTags()`. For each tag, this generates a synthetic 
        output where that tag has probability 1, other tags in the same head have 
        probability 0, and all other tags have probability `1 / k`."""
        K = self.dict.uniformTags()
        return self.dict.fwd(K, *args, **kwargs)

    def withClusts(self, R: Float[Array, "... d_out"], 
                   E: Float[Array, "... d_in"], *args, **kwargs
                 ) -> Tuple[Float[Array, "... d_out"], Float[Array, "... n_tags"]]:
        """Forward pass that adds the output to a residual tensor `R` and returns
        the flattened classifications `K` as the second value. This is used to build an
        `Ontologizer` from sequential `DictEnc`s, each of which adds its output to
        `R` and passes `K` to the next `DictEnc`."""
        K_0 = self.logits(E)
        C = self.fiber_coords(E)
        if self.scaled:
            S = self.scale(E)
            F, K = self.dict.withClusts(K_0, S, *args, C=C, **kwargs)
        else:
            F, K = self.dict.withClusts(K_0, *args, C=C, **kwargs)
        Y = self.dict.combine(F)
        return R + Y, einops.rearrange(K, "... h k -> ... (h k)")
    
    def withStats(self, R: Float[Array, "... d_out"],
                  E: Float[Array, "... d_in"], sd_K: float=0.0, sd_F: float=0.0,
                  rng: Optional[PRNGKeyArray]=None,
                  *args, p_drop: float=0.0, return_base: bool = False, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "6"], PRNGKeyArray]:
        """As `withClusts`, but also returns validation statistics
        `[L1_K, L1_F, entropy, cossim_batch, cossim_heads, KL_mean, aux]`.
        If `noisefn_K` and `sd_K` are specified, adds
        noise to `K` before passing it to `self.dict`.
        Splits `rng` so `noisefn_K` and `noisefn_F` use different seeds."""
        K = self.logits(E)

        K_n, rng_F = self.classifier.addnoise(K, sd_K, rng)
        L1_K = self.stat_K(E, K)

        if self.scaled:
            S = self.scale(E)
        else:
            S = None

        out = self.dict.withStats(
                K_n, S, *args, **kwargs, sd=sd_F, rng=rng_F, p_drop=p_drop,
                C=self.fiber_coords(E), return_base=return_base)
        F, K, stats, rng_next = out[:4]

        K = einops.rearrange(K, "... h k -> ... (h k)")
        stats = jnp.insert(stats, 0, L1_K)
        stats = jnp.append(stats, self.stat_aux(E))
        if return_base:  # the layer's fiber-free output, for a base-only loss
            return R + F, K, stats, rng_next, out[4]
        return R + F, K, stats, rng_next

    def withGhost(self, R: Float[Array, "... d_out"], R_g: Float[Array, "... d_out"],
                  E: Float[Array, "... d_in"], E_g: Optional[Float[Array, "... d_in"]],
                  temperature: float=1.0, sd_K: float=0.0, sd_F: float=0.0,
                  rng: Optional[PRNGKeyArray]=None, *args, p_drop: float=0.0,
                  **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "6"], PRNGKeyArray]:
        """As `withStats`, but also returns ghost gradient output."""
        K = self.logits(E)

        K_n, rng_F = self.classifier.addnoise(K, sd_K, rng)
        L1_K = self.stat_K(E, K)
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
                p_drop=p_drop, C=self.fiber_coords(E), *args, **kwargs)
        F_R = self.dict.fwd(K_g, S_g, *args, **kwargs)
        F_g = self.dict.ghost(K, F, S) + F_R

        P = einops.rearrange(P, "... h k -> ... (h k)")
        
        stats = jnp.insert(stats, 0, L1_K)
        stats = jnp.append(stats, self.stat_aux(E))

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
        the reconstructed input.

        Args:
            E: Input tensor of shape `(..., d_in)`.
            *args: Positional arguments forwarded to `self.dict.intervene`.
            temperature: Softmax temperature scalar (default: 1.0).
            **kwargs: Keyword arguments forwarded to `self.dict.intervene`.

        Returns:
            Tuple of:
                - `F`: Intervened reconstruction output of shape `(..., d_out)`.
                - `P_int`: Flattened intervened classification of shape `(..., h * k)`.
                - `E_int`: Reconstructed steered input of shape `(..., d_in)`.
        """
        K = self.logits(E)
        P = self.dict.cluster(K, temperature)
        P_int = self.dict.intervene(P, *args, **kwargs)
        E_int = self.classifier.rev(P_int)
        if self.scaled:
            S = self.scale(E)
        else:
            S = None
        # P_int is already intervened; do not pass *args again or additive/
        # scaling interventions would be applied twice.
        Fs = self.dict.hfwd(P_int, S, C=self.fiber_coords(E))
        F = self.dict.combine(Fs)
        return F, einops.rearrange(P_int, "... h k -> ... (h k)"), E_int

