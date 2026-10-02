"""Multihead ontofeature dictionary lookup and causal intervention layer.

Defines `DictBlock`, a Flax Linen module that stores learned non-negative concept
dictionaries `(h, k, d)` and reconstructs continuous embeddings from discrete or
categorical multihead classification probabilities. Also implements causal
feature interventions (clamping, scaling, adding, subtracting, and uniform/zero ablation).
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from jaxtyping import Array, Float, UInt, PRNGKeyArray
import einops

from ontologize.fns.loss import identity, cossim, bcossim, entropy, l1, ghostgrad
from ontologize.fns.loss import addnoise, addnoise_batchnorm, addnoise_featvar
from ontologize.fns.keys import get_activation
from .sparse import Sparse

class DictBlock(Sparse):
    """Multihead ontofeature submodule. It does not have bias or an activation function.
    It expects an *input* which has an `h` axis, to which it applies `softmax` *before*
    `fwd`. If the preceding layer is interpreted as a classifier, a `DictBlock` returns
    a weighted sum of representative vectors for each classification based on the 
    probability of that classification."""
    k: int = 0
    d: int = 0
    h: int = 0

    select: str = "softmax"
    activation: str = "none"

    sparse: bool = False
    entropy_loss: bool = False
    cossim_loss: bool = False
    bcossim_loss: bool = False
    hmean_loss: bool = False

    noise: str = "none"
    sd: float = 0.0

    # compute `withStats` without materializing the per-head outputs
    # (..., h, d). With non-negative weights Q = P*S and dictionaries |W|,
    # the L1 stat, the head-cosine stat and the per-head noise all have exact
    # closed forms (`_withStatsFast`). Differences from the reference path:
    # L1 is taken on the clean per-head outputs rather than the noised ones,
    # and the noise is drawn at the combined level (same distribution,
    # different random stream). Only used when no intervention is passed.
    # With `signed` entries, L1 and the head cosine come from a 32-row subset.
    fast_stats: bool = False

    # head-private spaces: head i's entries live only in the i-th block of
    # d // h coordinates (a block-diagonal dictionary), so the heads' outputs
    # are concatenated, not summed in a shared space, before the decoder
    private: bool = False
    # keep the dictionary's sign (by default abs() makes entries non-negative)
    signed: bool = False
    # head dropout: in training, with this probability per sample and head,
    # replace the head's classification by its batch-mean classification (the
    # head's origin), so heads cannot rely on co-adaptation and ablating a head
    # to its origin stays in distribution. Inverted (kept deviations scaled
    # by 1/(1-p)) so the expected output is unchanged.
    p_head_drop: float = 0.0
    # tangent-bundle fibers: each entry (h, k) carries a local basis U_hk of
    # `fiber_rank` directions, and the head adds U_hk @ c_h to its output,
    # where c_h (`C`, read linearly from the layer input by `DictEnc`) are the
    # entry's local coordinates. Signed, unlike the entries. The stats
    # (L1, cosines) describe the base output only.
    fiber_rank: int = 0
    # fiber dropout: in training, zero a head's fiber term with this
    # probability per sample, so the base entry must reconstruct on its own
    # and the fiber can only be a correction
    fiber_drop: float = 0.0
    # bound each fiber coordinate to (-fiber_bound, fiber_bound) via tanh
    # (0 = unbounded); an unbounded fiber can run away with the gain
    fiber_bound: float = 0.0
    # cap each head's fiber output norm at fiber_eps (0 = no cap). With
    # gain-shape the layer's output is later scaled by the residual norm, so
    # this caps the correction at fiber_eps of what is left to explain, and
    # zeroing the fibers can move each layer's output by at most that much
    fiber_eps: float = 0.0
    # joint cap: the sum over heads of the fiber output norms is capped at
    # fiber_eps_layer (a group-lasso ball: L2 within a head, L1 across
    # heads). Under gain-shape this bounds the whole layer's correction at
    # fiber_eps_layer of the residual left to explain
    fiber_eps_layer: float = 0.0
    # radial fiber (needs fiber_rank == 1): no basis parameters; the single
    # coordinate rescales the selected entry, F_h *= 1 + c_h (a per-head gain)
    fiber_radial: bool = False

    dtype_str: str = "bfloat16"
    dtype_p_str: str = "float32"

    def setup(self):
        """Initializes dictionary weight tensor and resolves selection function.

        Allocates parameter `weights` of shape `(h, k, d)` using LeCun normal initialization.
        Computes `n_tags = k * h` and `n_feat = d * h`.
        """
        super().setup()
        self.fn_sel = get_activation(self.select)
        self.n_tags = self.k * self.h
        self.n_feat = self.d * self.h
        # Shape is (h k d) rather than (h d k) so that tags() is contiguous
        if self.private:
            assert self.d % self.h == 0, "private heads need d divisible by h"
        self.weights = self.param(
            'weights',
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.h, self.k, self.d // self.h if self.private else self.d),
            dtype=self.dtype_p
        )
        if self.fiber_rank and not self.fiber_radial:
            e = self.d // self.h if self.private else self.d
            # same per-element scale as the entries, spread over the rank
            self.fiber = self.param(
                'fiber', nn.initializers.normal(stddev=(e * self.fiber_rank) ** -0.5),
                (self.h, self.k, e, self.fiber_rank), dtype=self.dtype_p)

    def dicts(self) -> Float[Array, "h k d"]:
        """Returns `abs(self.weights)`. This prevents subtractive ontofeatures.
        With `signed`, the raw weights; with `private`, each head's block
        embedded in its own d // h coordinates."""
        W = self.weights.astype(self.dtype)
        if not self.signed:
            W = jnp.abs(W)
        if self.private:
            W = jnp.einsum("hke,hg->hkge", W, jnp.eye(self.h, dtype=W.dtype))
            W = W.reshape(self.h, self.k, self.d)
        return W

    def fiber_out(self, P: Float[Array, "... h k"],
                  C: Optional[Float[Array, "... h r"]], per_head: bool
                  ) -> Optional[Array]:
        """Fiber term sum_k P_hk U_hk c_h, per head `(..., h, d)` or summed
        over heads `(..., d)`. None when there is no fiber (or no `C`)."""
        if not self.fiber_rank or C is None:
            return None
        C = C.astype(self.dtype)
        if self.fiber_bound:
            C = self.fiber_bound * jnp.tanh(C / self.fiber_bound)
        if self.fiber_radial:
            Fs = jnp.einsum("...hk,hkd->...hd", P, self.dicts())
            Fb = Fs * C[..., :1]                                   # per-head gain
            if self.fiber_eps or self.fiber_eps_layer:
                Fb = self._cap(Fb)
            return Fb if per_head else Fb.sum(-2)
        U = self.fiber.astype(self.dtype)
        if self.private:
            # each head's block only: (..., h, e) is small
            Fb = jnp.einsum("...hk,hker,...hr->...he", P, U, C)
            if self.fiber_eps or self.fiber_eps_layer:
                Fb = self._cap(Fb)
            if per_head:  # embed each head's block in its own coordinates
                Fb = jnp.einsum("...he,hg->...hge", Fb, jnp.eye(self.h, dtype=Fb.dtype))
                return Fb.reshape(*Fb.shape[:-3], self.h, self.d)
            return Fb.reshape(*Fb.shape[:-2], self.d)
        if per_head or self.fiber_eps or self.fiber_eps_layer:
            # the cap needs per-head norms, so materialize (..., h, d) here
            Fb = jnp.einsum("...hk,hkdr,...hr->...hd", P, U, C)
            if self.fiber_eps or self.fiber_eps_layer:
                Fb = self._cap(Fb)
            return Fb if per_head else Fb.sum(-2)
        return jnp.einsum("...hk,hkdr,...hr->...d", P, U, C)     # one contraction

    def _cap(self, Fb):
        """Rescale fiber vectors: each head's norm <= fiber_eps, and/or the
        sum of the heads' norms <= fiber_eps_layer."""
        n = jnp.linalg.norm(Fb, axis=-1, keepdims=True)
        if self.fiber_eps:
            Fb = Fb * jnp.minimum(1.0, self.fiber_eps / (n + 1e-12))
            n = jnp.linalg.norm(Fb, axis=-1, keepdims=True)
        if self.fiber_eps_layer:
            tot = n.sum(-2, keepdims=True)
            Fb = Fb * jnp.minimum(1.0, self.fiber_eps_layer / (tot + 1e-12))
        return Fb

    def fiber_dropout(self, C, rng):
        """Zero a head's fiber coordinates with prob `fiber_drop` per sample
        (training only; see `fiber_drop`)."""
        if C is None or not self.fiber_drop or rng is None:
            return C, rng
        rng, r = jax.random.split(rng)
        drop = jax.random.bernoulli(r, self.fiber_drop, C.shape[:-1])[..., None]
        return jnp.where(drop, 0.0, C), rng

    def head_drop(self, P: Float[Array, "... h k"],
                  rng: Optional[PRNGKeyArray]
                  ) -> Tuple[Float[Array, "... h k"], Optional[PRNGKeyArray]]:
        """Head dropout (see `p_head_drop`). No-op at inference (rng None)."""
        if not self.p_head_drop or rng is None:
            return P, rng
        rng, r = jax.random.split(rng)
        drop = jax.random.bernoulli(r, self.p_head_drop, P.shape[:-1])[..., None]
        origin = jax.lax.stop_gradient(P.reshape(-1, self.h, self.k).mean(0))
        # inverted: kept heads' deviations from the origin are scaled by
        # 1/(1-p), so the expected output matches inference (no dropout)
        kept = origin + (P - origin) / (1.0 - self.p_head_drop)
        return jnp.where(drop, origin, kept), rng

    def cluster(self, K: Float[Array, "... h k"], temperature: float=1.0,
                hardness: Optional[float] = None) -> Float[Array, "... h k"]:
        """Converts logits to probabilistic classifications, used for the weighted 
        sum of ontofeatures. With `select="anneal"`, a scheduled convex mix
        (1-hardness) * softmax + hardness * straight-through argmax; with
        `select="topm"`, the renormalized top-m of the softmax with m going
        from k to 1 as `hardness` goes 0 -> 1. Either way the code hardens
        structurally (argmax = normalized top-1) rather than through the
        temperature; `hardness` defaults to 1 (fully hard, inference)."""
        K = K.astype(self.dtype)
        if self.select == "anneal":
            P = jax.nn.softmax(K / temperature, axis=-1)
            hard = jax.nn.one_hot(jnp.argmax(K, -1), self.k, dtype=P.dtype)
            P_ste = jax.lax.stop_gradient(hard - P) + P
            lam = 1.0 if hardness is None else hardness
            return (1.0 - lam) * P + lam * P_ste
        if self.select == "topm":
            # normalized top-m of the softmax, with m shrinking geometrically
            # from k (soft) to 1 (argmax) as `hardness` goes 0 -> 1; the
            # backward pass is straight-through to the softmax (a renormalized
            # top-1 is the constant 1 and would have no gradient of its own)
            P = jax.nn.softmax(K / temperature, axis=-1)
            f = 1.0 if hardness is None else hardness
            m = jnp.round(jnp.asarray(self.k, jnp.float32) ** (1.0 - f)).astype(jnp.int32)
            srt = jnp.sort(P, axis=-1)[..., ::-1]
            thr = jnp.take_along_axis(srt, jnp.broadcast_to(m - 1, srt.shape[:-1])[..., None], -1)
            kept = jnp.where(P >= thr, P, 0.0)
            P_m = kept / (kept.sum(-1, keepdims=True) + 1e-12)
            return jax.lax.stop_gradient(P_m - P) + P
        return self.fn_sel(K / temperature)

    def fwd(self, P_0: Float[Array, "... h k"],
                S: Optional[Float[Array, "... h"]] = None, *args,
                C: Optional[Float[Array, "... h r"]] = None, **kwargs
               ) -> Float[Array, "... d"]:
        """Forward pass. Input `P_0` is expected to be probabilites from a multihead classifier.
        Accepts an optional scaling vector `S`, which should apply a scalar multiple
        to each slice along the `h` axis. Allows causal interventions on ontofeatures
        by passing arguments to `DictBlock.intervene`."""
        P = P_0.astype(self.dtype)
        P = self.intervene(P, *args, **kwargs)
        if S is not None:
            P = P * S.astype(self.dtype)[..., None]
        F = jnp.einsum("...hk,hkd->...d", P, self.dicts())
        Fb = self.fiber_out(P, C, per_head=False)
        return F if Fb is None else F + Fb

    def hfwd(self, P_0: Float[Array, "... h k"],
             S: Optional[Float[Array, "... h"]] = None,
             *args, C: Optional[Float[Array, "... h r"]] = None,
             **kwargs) -> Float[Array, "... h d"]:
        """As `DictBlock.fwd`, but returns the output without summing the `h` axis.
        This allows calculation of summary statistics and interventions on specific 
        heads."""
        P = P_0.astype(self.dtype)
        P = self.intervene(P, *args, **kwargs)
        if S is not None:
            P = P * S.astype(self.dtype)[..., None]
        Fs = jnp.einsum("...hk,hkd->...hd", P, self.dicts())
        Fb = self.fiber_out(P, C, per_head=True)
        return Fs if Fb is None else Fs + Fb

    def combine(self, Y: Float[Array, "... h d"]) -> Float[Array, "... d"]:
        """Sum over `h` axis."""
        return jnp.einsum("...hd->...d", Y)

    def __call__(self, K: Float[Array, "... h k"], 
                 S: Optional[Float[Array, "... h"]] = None,
                 temperature: float=1.0, *args, **kwargs
                 ) -> Float[Array, "... d"]:
        """Full forward evaluation: clusters logits, runs `fwd`, and applies activation.

        Args:
            K: Classification logits of shape `(..., h, k)`.
            S: Optional per-head scaling tensor of shape `(..., h)`.
            temperature: Softmax temperature scalar (default: 1.0).
            *args: Positional arguments forwarded to `self.intervene`.
            **kwargs: Keyword arguments forwarded to `self.intervene`.

        Returns:
            Output continuous activation vector of shape `(..., d)`.
        """
        P = self.cluster(K, temperature)
        F = self.fwd(P, S, *args, **kwargs)
        return self.fn(F)

    def hrev(self, F: Float[Array, "... h d"]) -> Float[Array, "... h k"]:
        """Reverse pass. Requires input to have an `h` axis."""
        F = F.astype(self.dtype)
        return jnp.einsum("hkd, ...hd -> ...hk", self.dicts(), F)

    def rev(self, F: Float[Array, "... d"],
                S: Optional[Float[Array, "... h"]] = None, *args, **kwargs
               ) -> Float[Array, "... h k"]:
        """Reverse pass with optional scale and interventions."""
        F = F.astype(self.dtype)
        if S is None:
            P = jnp.einsum("...d,hkd->...hk", F, self.dicts())
        else:
            S = S.astype(self.dtype)
            Fs = jnp.einsum("...d,...h->...hd", F, S)
            P = jnp.einsum("...hd,hkd->...hk", Fs, self.dicts())

        return self.intervene(P, *args, **kwargs)

    def ghost(self, K: Float[Array, "... h k"], F: Float[Array, "... d"],
              S: Optional[Float[Array, "... h"]]=None, *args, **kwargs
              ) -> Float[Array, "... h d"]:
        """Map `fns.loss.ghostgrad` over `h`."""
        W = self.dicts()
        K = K.astype(self.dtype)
        F = F.astype(self.dtype)
        def f(W_i, K_i):
            # Transpose W_i to (d, k) because ghostgrad expects (d_out, d_in)
            return ghostgrad(W_i.T, K_i, F, *args, **kwargs)

        Fs_g = jax.vmap(f, (0, -2), -2)(W, K)
        if S is None:
            return self.combine(Fs_g)
        return jnp.einsum("...hd, ...h -> ...d", Fs_g, S)


    def tagGram(self) -> Float[Array, "h k k"]:
        """Per-head Gram matrices of the dictionaries, `G_h = W_h @ W_h.T`.

        Returns:
            Gram matrix tensor of shape `(h, k, k)` containing pairwise inner products of dictionary entries.
        """
        W = self.dicts()
        return jnp.einsum("hkd,hjd->hkj", W, W)

    def bcossim_tags(self, P_0: Float[Array, "... h k"],
                     S: Optional[Float[Array, "... h"]] = None
                     ) -> Float[Array, ""]:
        """Batch cosine similarity of the per-head reconstructions
        `Fs = P @ dicts()`, computed in tag space: `Fs` has rank <= k per
        head, so `<F_i, F_j> = sum_h P_i.T G_h P_j` with the k x k
        `tagGram` -- O(h k^2 d + b^2 h k) instead of the O(b^2 h d) of
        materializing the flattened Gram (~40x cheaper at b=256, d=4096).
        Identical in value and gradient to `Sparse.bcossim(hfwd(P, S))`.
        Note: computed from the classification, so (unlike the previous
        `Fs`-based stat) `intervene` arguments do not enter it.

        Args:
            P_0: Classification probability tensor of shape `(..., h, k)`.
            S: Optional per-head scaling tensor of shape `(..., h)`.

        Returns:
            Scalar float array `()` with mean absolute batch cosine similarity.
        """
        P = P_0.astype(self.dtype)
        if S is not None:
            P = P * S[..., None].astype(self.dtype)
        if not self.bcossim_loss:
            P = jax.lax.stop_gradient(P)
            G = jax.lax.stop_gradient(self.tagGram())
        else:
            G = self.tagGram()
        P = P.reshape(-1, self.h, self.k)
        M = jnp.einsum("ihk,hkj->ihj", P, G)
        C = jnp.einsum("ihj,qhj->iq", M, P)
        n = jnp.sqrt(jnp.diagonal(C) + jnp.finfo(P.dtype).eps)
        C = C / (n[:, None] * n[None, :])
        return jnp.abs(C).mean()

    def hmean_kl(self, P_0: Float[Array, "... h k"]) -> Float[Array, ""]:
        """KL(batch-mean classification ‖ uniform) in bits, averaged over
        heads. Zero exactly when each head's E_batch[p] is uniform;
        minimizing it maximizes the entropy of the *mean* (per-sample
        classifications can stay arbitrarily hard, unlike the per-sample
        `entropy` stat). Serves two roles: load balancing (a frozen
        constant head costs the full log2(k) bits, and the pressure acts
        before the softmax saturates and its gradient vanishes) and
        anchoring the corpus-generic code point to the uniform mixture
        (nothing else ties E[p] to 1/k, and the resid_nc probe showed the
        non-negative dictionary amplifying a 0.1-bit drift into an
        off-manifold decode). Ignores the scaling vector: this measures
        classification usage, not scaled output.

        Args:
            P_0: Classification probability tensor of shape `(..., h, k)`.

        Returns:
            Scalar float array `()` with mean KL divergence in bits.
        """
        P = P_0.astype(self.dtype)
        if not self.hmean_loss:
            P = jax.lax.stop_gradient(P)
        Pm = P.reshape(-1, self.h, self.k).mean(0)
        logk = jnp.log2(jnp.asarray(self.k, Pm.dtype))
        return (logk - entropy(Pm)).mean()

    def withClusts(self, K: Float[Array, "... h k"],
                   S: Optional[Float[Array, "... h"]] = None,
                   temperature: float=1.0, *args,
                   C: Optional[Float[Array, "... h r"]] = None,
                   hardness: Optional[float] = None, **kwargs
                   ) -> Tuple[Float[Array, "... h d"], 
                              Float[Array, "... h k"]]:
        """As `DictBlock.hfwd`, but first applies `DictBlock.cluster` to the input, 
        which it returns as a second value.

        Args:
            K: Classification logits of shape `(..., h, k)`.
            S: Optional per-head scaling tensor of shape `(..., h)`.
            temperature: Softmax temperature scalar (default: 1.0).
            *args: Positional arguments forwarded to `self.hfwd`.
            **kwargs: Keyword arguments forwarded to `self.hfwd`.

        Returns:
            Tuple of:
                - `Fs`: Per-head reconstruction tensor of shape `(..., h, d)`.
                - `P`: Classification probabilities of shape `(..., h, k)`.
        """
        P = self.cluster(K, temperature, hardness)
        F = self.hfwd(P, S, *args, C=C, **kwargs)
        return F, P

    def withEntropy(self, K: Float[Array, "... h k"], 
                    S: Optional[Float[Array, "... h"]] = None,
                    *args, **kwargs
                    ) -> Tuple[Float[Array, "... h k"], Float[Array, ""]]:
        """As `DictBlock.withClusts`, but also calculates `DictBlock.entropy`
        on the classifications. If `isloss=False`, backpropagation is stopped for the
        `entropy` calculation."""
        F, P = self.withClusts(K, S, *args, **kwargs)
        return F, P, self.entropy(P, S)

    def withL1(self, Fs: Float[Array, "... h d"], isloss: bool=False
               ) -> Tuple[Float[Array, "... d"], Float[Array, ""]]:
        """Sums input over `h` axis, then returns the L1 loss for the result.
        If `isloss=False`, backpropagation is stopped before `L1` calculation.

        Args:
            Fs: Per-head reconstruction tensor of shape `(..., h, d)`.
            isloss: If True, allows gradients through L1 calculation (default: False).

        Returns:
            Tuple of:
                - `F`: Combined reconstruction vector of shape `(..., d)`.
                - `L1`: Scalar L1 norm `()`.
        """
        F = self.combine(Fs)
        return F, self.l1(Fs)

    def drop_winners(self, K: Float[Array, "... h k"], p_drop: float=0.0,
                     rng: Optional[PRNGKeyArray]=None
                     ) -> Tuple[Float[Array, "... h k"],
                                Optional[PRNGKeyArray]]:
        """Winner dropout: with probability `p_drop` per head per sample, masks
        the argmax entry's logit to `-inf`, so `cluster` renormalizes over the
        remaining entries and the runner-up receives the gradient. Guarantees
        non-winning entries get training signal regardless of logit margins.
        No-op when `rng` is `None` (inference).

        Args:
            K: Classification logits of shape `(..., h, k)`.
            p_drop: Probability of dropping the winning argmax entry (default: 0.0).
            rng: Optional PRNG key. If None, operation is a no-op.

        Returns:
            Tuple of `(K_dropped, rng_next)` where `K_dropped` matches `K.shape`, and
            `rng_next` is the updated PRNG key (or None).
        """
        if rng is None:
            return K, None
        rng_next, rng = jax.random.split(rng)
        K = K.astype(self.dtype)
        win = jax.nn.one_hot(jnp.argmax(K, -1), self.k, dtype=bool)
        drop = jax.random.bernoulli(rng, p_drop, K.shape[:-1])[..., None]
        return jnp.where(win & drop, -jnp.inf, K), rng_next

    def withStats(self, K: Float[Array, "... b h k"],
                  S: Optional[Float[Array, "... b h"]] = None,
                  sd: float=0.0, rng: Optional[PRNGKeyArray]=None,
                  *args, p_drop: float=0.0,
                  C: Optional[Float[Array, "... b h r"]] = None,
                  return_base: bool = False, **kwargs
                  ) -> Tuple[Float[Array, "... b d"], Float[Array, "... b h k"],
                             Float[Array, "5"],
                             PRNGKeyArray]:
        """As `withEntropy`, but also calculates `L1`, `cossim`, and
        `hmean_kl`. If `noisefn_F` and `sd_F` are specified, adds noise to
        the result. If `p_drop` is specified, applies `drop_winners` to
        `K` first."""
        if self.fast_stats and not args and set(kwargs) <= {"temperature", "hardness"}:
            return self._withStatsFast(K, S, sd, rng, p_drop,
                                       kwargs.get("temperature", 1.0), C,
                                       kwargs.get("hardness"), return_base)
        K, rng = self.drop_winners(K, p_drop, rng)
        # withEntropy, inlined so head dropout can act between the
        # classification (which the stats describe) and the lookup
        temperature = kwargs.pop("temperature", 1.0)
        hardness = kwargs.pop("hardness", None)
        P = self.cluster(K, temperature, hardness)
        H = self.entropy(P, S)
        P_used, rng = self.head_drop(P, rng)
        C, rng = self.fiber_dropout(C, rng)
        Fs = self.hfwd(P_used, S, *args, **kwargs)   # base output; stats describe it

        Fs_n, rng_next = self.addnoise(Fs, sd, rng)
        cossim_b = self.bcossim_tags(P, S)
        cossim_h = self.cossim(Fs)
        KL_m = self.hmean_kl(P)

        F, L1 = self.withL1(Fs_n)
        F_base = F
        Q = P_used if S is None else P_used * S.astype(self.dtype)[..., None]
        Fb = self.fiber_out(self.intervene(Q, *args, **kwargs), C, per_head=False)
        if Fb is not None:
            F = F + Fb
        stats = [L1, H, cossim_b, cossim_h, KL_m]
        if return_base:
            return F, P, jnp.stack(stats), rng_next, F_base
        return F, P, jnp.stack(stats), rng_next

    def _combined_noise(self, Q, W, F, sd, rng):
        """Noise for F = sum_h Fs_h with the distribution of `self.fn_noise`
        applied to each head's output Fs_h = Q_h @ W_h independently and then
        summed (a sum of independent Gaussians), without materializing Fs."""
        if rng is None:
            return F, None
        rng_next, rng = jax.random.split(rng)
        sg = jax.lax.stop_gradient
        eps = jnp.finfo(F.dtype).eps
        if self.fn_noise is addnoise:
            std = jnp.sqrt(jnp.asarray(self.h, F.dtype)) * sd
        elif self.fn_noise is addnoise_featvar:
            # per-head batch variance of Fs_hd: diag(W_h^T Cov_b(Q_h) W_h)
            Qc = sg(Q) - sg(Q).mean(0, keepdims=True)
            C = jnp.einsum("bhk,bhj->hkj", Qc, Qc) / Q.shape[0]
            var = jnp.einsum("hkd,hkj,hjd->hd", sg(W), C, sg(W))
            std = sd * jnp.sqrt((var + eps).sum(0))           # (d,)
        elif self.fn_noise is addnoise_batchnorm:
            G = jnp.einsum("hkd,hjd->hkj", sg(W), sg(W))
            n2 = jnp.einsum("...hk,hkj,...hj->...h", sg(Q), G, sg(Q))
            std = sd * jnp.sqrt(n2.sum(-1, keepdims=True) / W.shape[-1])
        else:  # identity
            return F, rng_next
        noise = jax.random.normal(rng, F.shape, F.dtype)
        return F + noise * std, rng_next

    def _withStatsFast(self, K, S, sd, rng, p_drop, temperature, C=None, hardness=None,
                       return_base=False):
        """`withStats` for the no-intervention case, computed from P and the
        dictionary directly (see `fast_stats`)."""
        sg = jax.lax.stop_gradient
        K, rng = self.drop_winners(K, p_drop, rng)
        P = self.cluster(K, temperature, hardness)
        H = self.entropy(P, S)
        P_used, rng = self.head_drop(P, rng)
        C, rng = self.fiber_dropout(C, rng)
        Q = P_used if S is None else P_used * S.astype(self.dtype)[..., None]
        W = self.dicts()
        F = jnp.einsum("...hk,hkd->...d", Q, W)
        F_n, rng_next = self._combined_noise(Q, W, F, sd, rng)
        F_base = F_n
        Fb = self.fiber_out(Q, C, per_head=False)
        if Fb is not None:
            F_n = F_n + Fb

        if self.signed:
            # the closed forms below need non-negative entries; with signed
            # entries, take the two stats from a materialized 32-row subset
            # (L1 rescaled to the full batch)
            Qs = Q.reshape(-1, self.h, self.k)[:32]
            Fs = jnp.einsum("bhk,hkd->bhd", Qs, W)
            L1 = self.l1(Fs) * (Q.reshape(-1, self.h, self.k).shape[0] / Qs.shape[0])
            cossim_h = self.cossim(Fs)
        else:
            # L1 of the per-head outputs: every term is >= 0, so
            # sum |Fs| = sum_{hk} Q_hk ||W_hk||_1
            Ql, Wl = (Q, W) if self.sparse else (sg(Q), sg(W))
            L1 = jnp.einsum("...hk,hk->", Ql, Wl.sum(-1))

            # mean |cos| between the heads' outputs, diagonal included (as in
            # `Sparse.cossim`): all cosines are >= 0 and
            # sum_{h,h'} <F^_h, F^_h'> = ||sum_h F^_h||^2
            Qc, Wc = (Q, W) if self.cossim_loss else (sg(Q), sg(W))
            G = jnp.einsum("hkd,hjd->hkj", Wc, Wc)
            n = jnp.sqrt(jnp.einsum("...hk,hkj,...hj->...h", Qc, G, Qc)
                         + jnp.finfo(self.dtype).eps)
            Z = jnp.einsum("...hk,hkd->...d", Qc / n[..., None], Wc)
            cossim_h = ((Z ** 2).sum(-1) / self.h ** 2).mean()

        cossim_b = self.bcossim_tags(P, S)
        KL_m = self.hmean_kl(P)
        stats = [L1, H, cossim_b, cossim_h, KL_m]
        if return_base:
            return F_n, P, jnp.stack(stats), rng_next, F_base
        return F_n, P, jnp.stack(stats), rng_next

    def tags(self) -> Float[Array, "n_tags d"]:
        """Flattens `self.dicts()` to shape `(h * k, d)`. This allows the weights to be
        treated as a batch of synthetic data."""
        return jnp.concat(self.dicts(), axis=0)

    def uniform(self, b: int = 1) -> Float[Array, "... b"]:
        """Returns synthetic classifications of a specified batchsize with all values
        set to `1 / k`."""
        p = 1.0 / self.k
        shape = (b, self.h, self.k)
        return jnp.full(shape, p, dtype=self.dtype)

    def hmask(self, P: Float[Array, "... h k"], heads: UInt[Array, "n"]
              ) -> Float[Array, "... h k"]:
        """Accepts classifications `P` and vector of indices `heads`.
        Returns a mask of `P` which is 1 at `[..., heads, :]` and 0 elsewhere."""
        P = P.astype(self.dtype)
        return jnp.zeros_like(P).at[..., heads, :].set(1.0)

    def kmask(self, P: Float[Array, "... h k"], tags:UInt[Array, "n"]
              ) -> Float[Array, "... h k"]:
        """Accepts classifications `P` and vector of indices `tags`.
        Returns a mask of `P` which is 1 at `[..., tags]` and 0 elsewhere."""
        P = P.astype(self.dtype)
        return jnp.zeros_like(P).at[..., tags].set(1.0)

    def mask(self, P: Float[Array, "... h k"], 
             heads: UInt[Array, "n_head"], tags: UInt[Array, "n_tag"], 
             ) -> Float[Array, "... h k"]:
        """Accepts classifications `P` and vectors of indices `heads` and `tags`.
        Returns a mask of `P` which is 1 at `[..., heads, tags]` and 0 elsewhere."""
        P = P.astype(self.dtype)
        return jnp.zeros_like(P).at[..., heads, tags].set(1.0)

    def uniformAblate(self, P_0: Float[Array, "... h k"], 
                      heads: Optional[UInt[Array, "n_head"]] = None
                      ) -> Float[Array, "... h k"]:
        """Accepts classifications `P_0` and vector of indices `heads`.
        Returns `P` with `[..., heads, :]` set to `1 / k`."""
        P = P_0.astype(self.dtype)
        p = 1.0 / self.k
        if heads is not None:
            P = P.at[..., heads, :].set(p)
        return P

    def zeroAblate(self, P_0: Float[Array, "... h k"], 
                   heads: Optional[UInt[Array, "n_head"]] = None
                   ) -> Float[Array, "... h k"]:
        """Accepts classifications `P_0` and vector of indices `heads`.
        Returns `P` with `[..., heads, :]` set to 0."""
        P = P_0.astype(self.dtype)
        if heads is not None:
            P = P.at[..., heads, :].set(0.0)
        return P

    def set(self, P_0: Float[Array, "... h k"],
            heads: UInt[Array, "n_head"], tags: UInt[Array, "n_head"]
            ) -> Float[Array, "... h k"]:
        """Accepts classifications `P_0` and vectors of indices `heads` and `tags`.
        Returns `P` with `[..., heads, tags]` set to 1 and `[..., heads, not tags]`
        set to 0."""
        P = P_0.astype(self.dtype)
        P = P.at[..., heads, :].set(0.0)
        P = P.at[..., heads, tags].set(1.0)
        return P


    def uniformTags(self) -> Float[Array, "n_tags h k"]:
        """For each combination of indices `i, j` in `range(h)` × `range(k)`,
        returns `P_ij` of shape `(h, k)` with `[i, j]` set to 1, `[i, not j]` set to 0,
        and all other indices set to `1 / k`."""
        # b = 1 (uniform) + k * h (one-hot variations)
        
        # 1. Create the base uniform array (shape: 1, h, k)
        p = 1.0 / self.k
        uniform_base = jnp.full((1, self.h, self.k), p, dtype=self.dtype)
        
        # 2. Define a function that takes a (tag, head) pair and returns a (k, h) tensor
        def make_one_hot_tag(head, tag):
            # Start with uniform (k, h)
            base = jnp.full((self.h, self.k), p, dtype=self.dtype)
            # Create a one-hot vector for the specific tag/head
            one_hot_head = jnp.zeros(self.k, dtype=self.dtype).at[tag].set(1.0)
            # Replace the specific head's distribution with the one-hot vector
            return base.at[head, :].set(one_hot_head)
            
        # 3. Create all combinations of tags and heads
        heads, tags = jnp.meshgrid(jnp.arange(self.h), jnp.arange(self.k), indexing='ij')
        heads_flat = heads.flatten()
        tags_flat = tags.flatten()
        
        # 4. Use vmap to apply the function over all combinations
        # Result shape: (k*h, k, h)
        one_hot_variations = jax.vmap(make_one_hot_tag)(heads_flat, tags_flat)
        
        # 5. Concatenate the uniform base with the variations
        return jnp.concatenate([uniform_base, one_hot_variations], axis=0)

    def intervene(self, P_0: Float[Array, "... h k"],
                  scale: Optional[Float[Array, "h"]] = None,
                  k_set: Optional[UInt[Array, "n_set"]] = None,
                  h_set: Optional[UInt[Array, "n_set"]] = None,
                  h_unif: Optional[UInt[Array, "n_unif"]] = None,
                  h_zero: Optional[UInt[Array, "n_zero"]] = None,
                  k_add: Optional[UInt[Array, "n_add"]] = None,
                  h_add:Optional[UInt[Array, "n_add"]] = None,
                  k_sub: Optional[UInt[Array, "n_sub"]] = None,
                  h_sub:Optional[UInt[Array, "n_sub"]] = None,
                  ) -> Float[Array, "... h k"]:
        """Interface for intervening on ontofeatures.
        Applies `P_0 |> uniformAblate(h_unif) |> zeroAblate(h_zero) |> set(h_set, k_set)`.
        Then adds `mask(P, h_add, k_add)`, subtracts `mask(P, h_sub, k_sub)`, 
        then scales `P` by `S`."""
        P = P_0.astype(self.dtype)
        P = self.uniformAblate(P, h_unif)
        P = self.zeroAblate(P, h_zero)
        if h_set is not None and k_set is not None:
            P = self.set(P, h_set, k_set)

        if h_add is not None and k_add is not None:
            P = P + self.mask(P, h_add, k_add)

        if h_sub is not None and k_sub is not None:
            P = P - self.mask(P, h_sub, k_sub)

        if scale is not None:
            S = scale.astype(self.dtype)
            P = jnp.einsum("...hk,h->...hk", P, S)

        return P
