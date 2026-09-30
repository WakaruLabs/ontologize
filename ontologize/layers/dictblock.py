import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from jaxtyping import Array, Float, UInt, PRNGKeyArray
import einops

import re

from ontologize.fns.loss import (identity, cossim, bcossim, entropy, l1,
                                 ghostgrad, recip_norm)
from ontologize.fns.keys import get_activation
from .sparse import Sparse

class DictBlock(Sparse):
    """Multihead ontofeature submodule. It does not have bias or an activation function.
    While it inherits `activation` from `Sparse`, it is applied only by `__call__`,
    which is not used. This ensured that the output is linear w.r.t. classifications.
    It expects an *input* which has an `h` axis, to which it applies `softmax` *before*
    `fwd`. If the preceding layer is interpreted as a classifier, a `DictBlock` returns
    a weighted sum of representative vectors for each classification based on the
    probability of that classification."""
    k: int = 0
    d: int = 0
    h: int = 0

    # logits -> probabilities rule for `cluster`: "softmax" (dense),
    # "argmax"/"ste" (straight-through hard), or "top<k>" ("top4"):
    # softmax over the top-k logits per head, the rest masked to -inf
    # (exact-zero probability, no gradient). The top-k support is chosen
    # on the logits `cluster` receives -- after sd_K noise and winner
    # dropout -- so both still explore support membership.
    select: str = "softmax"
    activation: str = "none"

    # unit-L2-normalize each `(h, k)` dictionary row in `dicts()`, so a tag
    # carries a direction and `cluster` alone sets its magnitude. Because
    # the rows are non-negative and `P` sums to 1 per head, this puts a
    # floor of `h` on the layer's `L1`: no cancellation means
    # `|F|_1 = sum_hk P_hk |W_hk|_1`, and `|W_hk|_1 >= |W_hk|_2 = 1`.
    norm_rows: bool = False

    # drop the `abs()` in `dicts()`, letting atoms subtract. See there for
    # why this changes the reachable output set and not just the sign.
    signed: bool = False

    sparse: bool = False
    entropy_loss: bool = False
    cossim_loss: bool = False
    bcossim_loss: bool = False
    kcossim_loss: bool = False
    flatcos_loss: bool = False
    hmean_loss: bool = False

    noise: str = "none"
    sd: float = 0.0

    dtype_str: str = "bfloat16"
    dtype_p_str: str = "float32"

    def setup(self):
        super().setup()
        match = re.fullmatch(r"top(\d+)", self.select.lower())
        if match:
            k_sel = int(match.group(1))
            if not 1 <= k_sel <= self.k:
                raise ValueError(
                    f"select={self.select!r} needs 1 <= k <= {self.k} "
                    f"(tags per head)")
        elif self.select.lower() in ("argmax", "ste"):
            # one entry survives the forward, so the support is 1. Setting
            # this to `k` made `revive_dead`'s `k_sel >= k` guard treat
            # `ste` as dense and switch revival off for the rule the live
            # arm uses. It is not dense where it matters: the
            # straight-through estimator restores gradient to what PRODUCED
            # the code, not to what consumes it. `fwd` is `P @ dicts()` and
            # P's forward value is one-hot, so `d/dW[h,j]` is proportional
            # to `P[..., h, j]` and is exactly zero for an entry that never
            # won -- measured identical to `top<k>`, where `softmax` leaves
            # every entry a little mass.
            k_sel = 1
        else:
            k_sel = self.k
        # tags kept per head by `select`; `k` only for the dense rules,
        # which is what makes `revive_dead` a no-op for them
        self.k_sel = k_sel
        self.fn_sel = get_activation(self.select)
        self.n_tags = self.k * self.h
        self.n_feat = self.d * self.h
        self.d_head = self.head_width()
        # Shape is (h k d) rather than (h d k) so that tags() is contiguous
        self.weights = self.param(
            'weights',
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.h, self.k, self.d_head),
            dtype=self.dtype_p
        )

    def head_width(self) -> int:
        """Output width of one head. The whole of `d` here, since heads sum
        into a shared space; `ConcatDictBlock` narrows it."""
        return self.d

    def dicts(self) -> Float[Array, "h k d"]:
        """Returns `abs(self.weights)`. This prevents subtractive ontofeatures.
        Under `norm_rows`, each row is then scaled to unit L2 norm. Doing it
        here rather than by projecting after the optimizer step makes the
        radial direction an exact null direction of the forward pass, so no
        gradient is produced along it and none has to be projected out.

        `signed` drops the `abs`, which is a larger change than it looks.
        Non-negativity is an ELEMENTWISE constraint and so basis-dependent,
        which couples the decoder's row space to its null space: the visible
        component `P a` of an atom generically has negative entries, so null
        content is what lifts `a` into the non-negative orthant and makes
        that visible component representable at all. Signed atoms need no
        such lift, so the reachable set of output directions becomes the
        decoder's column SPAN rather than the cone its columns generate."""
        W = self.weights.astype(self.dtype)
        if not self.signed:
            W = jnp.abs(W)
        if not self.norm_rows:
            return W
        n = jnp.linalg.norm(W, axis=-1, keepdims=True)
        return W / (n + jnp.finfo(self.dtype).eps)

    def cluster(self, K: Float[Array, "... h k"], temperature: float=1.0) -> Float[Array, "... h k"]:
        """Converts logits to probabilistic classifications, used for the weighted 
        sum of ontofeatures."""
        K = K.astype(self.dtype)
        return self.fn_sel(K / temperature)

    def fwd(self, P_0: Float[Array, "... h k"],
                S: Optional[Float[Array, "... h"]] = None, *args, **kwargs
               ) -> Float[Array, "... d"]:
        """Forward pass. Input `P_0` is expected to be probabilites from a multihead classifier.
        Accepts an optional scaling vector `S`, which should apply a scalar multiple
        to each slice along the `h` axis. Allows causal interventions on ontofeatures
        by passing arguments to `DictBlock.intervene`."""
        P = P_0.astype(self.dtype)
        P = self.intervene(P, *args, **kwargs)
        if S is None:
            return jnp.einsum("...hk,hkd->...d", P, self.dicts())
        S = S.astype(self.dtype)
        return jnp.einsum("...hk,hkd,...h->...d", P, self.dicts(), S)

    def hfwd(self, P_0: Float[Array, "... h k"],
             S: Optional[Float[Array, "... h"]] = None,
             *args, **kwargs) -> Float[Array, "... h d"]:
        """As `DictBlock.fwd`, but returns the output without summing the `h` axis.
        This allows calculation of summary statistics and interventions on specific 
        heads."""
        P = P_0.astype(self.dtype)
        P = self.intervene(P, *args, **kwargs)
        if S is None:
            return jnp.einsum("...hk,hkd->...hd", P, self.dicts())
        S = S.astype(self.dtype)
        return jnp.einsum("...hk,hkd,...h->...hd", P, self.dicts(), S)

    def combine(self, Y: Float[Array, "... h d"]) -> Float[Array, "... d"]:
        """Sum over `h` axis."""
        return jnp.einsum("...hd->...d", Y)

    def __call__(self, K: Float[Array, "... h k"], 
                 S: Optional[Float[Array, "... h"]] = None,
                 temperature: float=1.0, *args, **kwargs
                 ) -> Float[Array, "... d"]:
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
        """Per-head Gram matrices of the dictionaries, `G_h = W_h @ W_h.T`."""
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
        `Fs`-based stat) `intervene` arguments do not enter it."""
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
        r = recip_norm(jnp.diagonal(C))
        C = C * r[:, None] * r[None, :]
        return jnp.abs(C).mean()

    def rowcos(self) -> Float[Array, "h"]:
        """Mean off-diagonal cosine between a head's dictionary rows,
        averaged over heads. Zero when a head's entries have disjoint
        supports, one when they are all the same direction.

        This is head *liveness*, which `hmean_kl` does not measure.
        `hmean_kl` scores usage balance -- whether `E_batch[p]` is uniform
        -- and a head whose rows are all the same direction produces an
        output independent of which entry wins, so its classifier is free
        to spread usage perfectly and score zero while the head carries no
        information. A head has two routes to contributing a constant:
        saturate the classification, which `hmean_kl` prices at the full
        log2(k) bits, or collapse the dictionary, which costs it nothing.
        This closes the second.

        Reads the weights alone, so unlike the other cossim stats it is a
        property of the dictionary rather than of a batch. Rows are
        non-negative (`dicts`), so the cosines are already in [0, 1].

        Returns the per-head values rather than their mean: a head is
        collapsed or it is not, and averaging lets a few collapsed heads
        hide among healthy ones. `perhead` keeps that visible to a caller
        deciding what to reduce with -- `withStats` sums for the penalty
        (every head gets gradient) and maxes for the reported stat."""
        G = self.tagGram()
        if not self.kcossim_loss:
            G = jax.lax.stop_gradient(G)
        r = recip_norm(jnp.diagonal(G, axis1=-2, axis2=-1))
        C = G * r[..., :, None] * r[..., None, :]
        diag = jnp.diagonal(C, axis1=-2, axis2=-1).sum(-1)
        return (C.sum((-2, -1)) - diag) / (self.k * (self.k - 1))

    def flatcos(self) -> Float[Array, ""]:
        """Mean off-diagonal cosine over the dictionary FLATTENED: every
        pair of rows, not just pairs inside a head.

        `rowcos` is within-head by construction and cannot see the other
        redundancy: heads whose entries are individually diverse but
        whose rows are parallel to another head's. Nothing else measures
        that -- `cossim_h` scores head CONTRIBUTIONS after selection, not
        the rows themselves.

        The two are complements, not substitutes. Of the `h*k - 1`
        partners a row has, only `k - 1` share its head, so this mean is
        dominated by the between-head pairs and a single collapsed head
        barely moves it where `rowcos`'s per-head max saturates. Report
        both.

        Needs no Gram: sum_{i!=j} u_i.u_j = |sum_i u_i|^2 - N over unit
        rows, which is O(N d) against the per-head Gram's O(h k^2 d)."""
        W = self.dicts()
        if not self.flatcos_loss:
            W = jax.lax.stop_gradient(W)
        U = W.reshape(-1, W.shape[-1])
        U = U * recip_norm(jnp.sum(U * U, -1))[:, None]
        n = U.shape[0]
        tot = jnp.sum(jnp.square(jnp.sum(U, 0))) - n
        return tot / (n * (n - 1))

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
        classification usage, not scaled output."""
        P = P_0.astype(self.dtype)
        if not self.hmean_loss:
            P = jax.lax.stop_gradient(P)
        Pm = P.reshape(-1, self.h, self.k).mean(0)
        logk = jnp.log2(jnp.asarray(self.k, Pm.dtype))
        return (logk - entropy(Pm)).mean()

    def withClusts(self, K: Float[Array, "... h k"],
                   S: Optional[Float[Array, "... h"]] = None,
                   temperature: float=1.0, *args, **kwargs
                   ) -> Tuple[Float[Array, "... h d"], 
                              Float[Array, "... h k"]]:
        """As `DictBlock.hfwd`, but first applies `DictBlock.cluster` to the input, 
        which it returns as a second value."""
        P = self.cluster(K, temperature)
        F = self.hfwd(P, S, *args, **kwargs)
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
        If `isloss=False`, backpropagation is stopped before `L1` calculation."""
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
        No-op when `rng` is `None` (inference)."""
        if rng is None:
            return K, None
        rng_next, rng = jax.random.split(rng)
        K = K.astype(self.dtype)
        win = jax.nn.one_hot(jnp.argmax(K, -1), self.k, dtype=bool)
        drop = jax.random.bernoulli(rng, p_drop, K.shape[:-1])[..., None]
        return jnp.where(win & drop, -jnp.inf, K), rng_next

    def revive_dead(self, K: Float[Array, "... h k"], p_revive: float=0.0,
                    revive_frac: float=0.5,
                    rng: Optional[PRNGKeyArray]=None
                    ) -> Tuple[Float[Array, "... h k"],
                               Optional[PRNGKeyArray]]:
        """Tag-axis starved-entry revival, the counterpart of `drop_winners`:
        where that promotes the runner-up, this promotes an entry the batch
        has all but stopped selecting.

        A hard `select` rule makes death absorbing. Under `top<k>` an entry
        outside every sample's support has probability exactly 0, so it
        receives exactly no gradient, so its logits cannot move, so it can
        never return -- unlike the dense rules, where every entry keeps some
        mass and can always recover. With probability `p_revive` per head
        per sample, this lifts one starved entry to that head's leading
        logit, which puts it in the support and splits the head's mass with
        the incumbent.

        An entry is starved when it is selected at less than `revive_frac`
        of the rate uniform usage would give it (`k_sel / k`). The test has
        to be a RATE rather than absence from the batch: a head makes
        `b * k_sel` selections over `k` entries, so at any realistic batch
        size essentially every entry is still picked at least once long
        after usage has become badly skewed, and a test for absence would
        simply never fire. `revive_frac` below 1 leaves a merely
        below-average entry alone, so the mechanism is a genuine no-op on a
        healthy head.

        Promoting to a tie rather than past the leader is deliberate: merely
        landing in the support with a low logit would leave the entry a
        vanishing softmax share at the annealed temperature (so no useful
        gradient), while dominating outright would saturate the softmax (so
        no gradient to the classifier, only to the dictionary). A tie leaves
        both paths live.

        Promotion displaces the weakest incumbent from the support, which is
        the same trade `drop_winners` makes: at a sane `p_revive` only that
        fraction of (sample, head) pairs is disturbed.

        No-op when `rng` is `None` (inference) or for a dense `select`,
        where no entry is ever masked out. `p_revive` is traced (it is not
        static on `update`), so a zero is handled by the Bernoulli drawing
        no hits rather than by branching on it."""
        if rng is None or self.k_sel >= self.k:
            return K, rng
        rng_next, r_pick, r_hit = jax.random.split(rng, 3)
        K = K.astype(self.dtype)
        # the support is fixed by logit rank, so the selection rate is
        # temperature-invariant and can be read straight off the logits
        thr = jax.lax.top_k(K, self.k_sel)[0][..., -1:]
        freq = (K >= thr).reshape(-1, self.h, self.k).mean(0)
        dead = freq < revive_frac * self.k_sel / self.k
        # one dead entry per (sample, head), uniform over that head's dead
        # set; Gumbel argmax picks it without materializing a categorical
        g = jax.random.gumbel(r_pick, K.shape, K.dtype)
        pick = jnp.argmax(jnp.where(dead, g, -jnp.inf), -1)
        hit = (jax.random.bernoulli(r_hit, p_revive, K.shape[:-1])
               & dead.any(-1))
        boost = jax.nn.one_hot(pick, self.k, dtype=bool) & hit[..., None]
        return jnp.where(boost, K.max(-1, keepdims=True), K), rng_next

    def withStats(self, K: Float[Array, "... b h k"],
                  S: Optional[Float[Array, "... b h"]] = None,
                  sd: float=0.0, rng: Optional[PRNGKeyArray]=None,
                  *args, p_drop: float=0.0, p_revive: float=0.0,
                  revive_frac: float=0.5, **kwargs
                  ) -> Tuple[Float[Array, "... b d"], Float[Array, "... b h k"],
                             Float[Array, "7"],
                             PRNGKeyArray]:
        """As `withEntropy`, but also calculates `L1`, `cossim`, and
        `hmean_kl`. If `noisefn_F` and `sd_F` are specified, adds noise to
        the result. If `p_drop` is specified, applies `drop_winners` to
        `K` first. The pwak stats are not here: they are functions of the
        classifications and the layer INPUT, which is `DictEnc`'s, so
        `DictEnc.withStats` appends them to this row."""
        # revive first: deadness is read off the logits the classifier
        # produced, before drop_winners masks a winner out of them
        K, rng = self.revive_dead(K, p_revive, revive_frac, rng)
        K, rng = self.drop_winners(K, p_drop, rng)
        Fs, P, H = self.withEntropy(K, S, *args, **kwargs)

        Fs_n, rng_next = self.addnoise(Fs, sd, rng)
        cossim_b = self.bcossim_tags(P, S)
        cossim_h = self.cossim(Fs)
        cossim_k = self.rowcos()      # (h,) per-head
        cos_flat = self.flatcos()     # every row pair, within and between
        KL_m = self.hmean_kl(P)

        F, L1 = self.withL1(Fs_n)
        # the mean is the penalty (every head gets gradient); the max is
        # the constraint (a head is collapsed or it is not, and averaging
        # lets a few hide among healthy ones) and carries weight 0
        stats = [L1, H, cossim_b, cossim_h, cossim_k.mean(),
                 cossim_k.max(), KL_m, cos_flat]
        return F, P, jnp.stack(stats), rng_next

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


class ConcatDictBlock(DictBlock):
    """Heads write disjoint slices of the output instead of summing into
    a shared one.

    Each head gets `d // h` of the output and the slices concatenate, so
    `fwd` still returns `(..., d)` and the layer is a drop-in. Three
    things change:

      parameters   the dictionary is `h*k*(d//h) = k*d` rather than
                   `h*k*d`, a factor of `h` fewer at matched `d`.
      gauge        a head reaches the output only through the decoder's
                   own `(d_out, d//h)` column block, which has full
                   column rank whenever `d//h <= d_out`. Then that block
                   has no null space, so every dictionary direction is
                   visible downstream. The summing form instead sends
                   every head through the whole `(d_out, d)` decoder,
                   which for `d > d_out` has a null space an atom is free
                   to sit in, changing no output and taking no gradient.
      cossim_h     between-head cosine is 0 by construction, so the
                   `s_hcossim` penalty has nothing left to do.

    The two together fix the useful slice width: wide enough that `h` of
    them span `d_out`, narrow enough to stay under it. Widening a slice
    to the full `d` (so `d = e*h`) gives that up -- `D_h` is then
    surjective, so every head can realize any `k` output vectors exactly
    as in the summing form, and the orthogonality is pure gauge over a
    per-head null space.

    A subclass rather than a flag because the per-head shapes differ --
    `dicts`, `hfwd` and `hrev` carry `d // h` -- and because three
    statistics would otherwise return plausible wrong numbers rather
    than fail: `cossim` and `flatcos` compare vectors from disjoint
    slices as though they shared a space, and `tags` hands out atoms
    that are no longer in the layer's output space. `ghost` would hand
    each head the whole output rather than its own slice, and does at
    least fail loudly. Each is overridden below. `rowcos`, `tagGram` and
    `bcossim_tags` need no change: they are within-head, and a block's
    inner products are the same computed in the slice or in the whole.
    """

    def head_width(self) -> int:
        if self.h and self.d % self.h:
            raise ValueError(
                f"concat needs d divisible by h; got d={self.d}, h={self.h}. "
                f"Each head writes d // h of the output, and a truncating "
                f"division here would silently drop {self.d % self.h} dims")
        return self.d // self.h if self.h else self.d

    def combine(self, Y: Float[Array, "... h d_head"]
                ) -> Float[Array, "... d"]:
        """Concatenate the `h` axis instead of summing it."""
        return einops.rearrange(Y, "... h d -> ... (h d)")

    def split(self, F: Float[Array, "... d"]) -> Float[Array, "... h d_head"]:
        """Inverse of `combine`."""
        return einops.rearrange(F, "... (h d) -> ... h d", h=self.h)

    def fwd(self, P_0: Float[Array, "... h k"],
            S: Optional[Float[Array, "... h"]] = None, *args, **kwargs
            ) -> Float[Array, "... d"]:
        return self.combine(self.hfwd(P_0, S, *args, **kwargs))

    def rev(self, F: Float[Array, "... d"],
            S: Optional[Float[Array, "... h"]] = None, *args, **kwargs
            ) -> Float[Array, "... h k"]:
        Fs = self.split(F.astype(self.dtype))
        if S is not None:
            Fs = jnp.einsum("...hd,...h->...hd", Fs, S.astype(self.dtype))
        return self.intervene(self.hrev(Fs), *args, **kwargs)

    def cossim(self, F: Float[Array, "... h d_head"],
               S: Optional[Float[Array, "..."]]=None,
               isloss: Optional[bool]=None) -> Float[Array, ""]:
        """The parent's value, computed in closed form. Heads occupy
        disjoint coordinates, so every off-diagonal cosine is exactly 0
        whatever the atoms contain and only the diagonal survives.

        That diagonal is not 0, so this does not return 0: the parent
        means over the full `(h, h)` matrix, giving `1/h` when a head is
        nonzero and dropping its entry when a head's output vanishes,
        since `recip_norm` defines a zero vector's cosine as 0. Returning
        0 instead would put this layer's `cossim_h` column on a different
        scale from a summing layer's and make the two unreadable side by
        side. Costs O(b h d) rather than the parent's O(b h^2 d), and
        takes no gradient because there is none to take -- which is the
        point of the construction.

        `S` and `isloss` are accepted and ignored: both scale or gate a
        quantity that is constant here."""
        Fs = F.astype(self.dtype)
        live = jnp.sum(Fs * Fs, -1) > 0
        return jnp.mean(live.astype(self.dtype)) / self.h

    def ghost(self, K: Float[Array, "... h k"], F: Float[Array, "... d"],
              S: Optional[Float[Array, "... h"]]=None, *args, **kwargs
              ) -> Float[Array, "... d"]:
        """As the parent, but each head's deadness is read off its own
        slice of the output. The parent hands every head the whole of `F`,
        which here is neither the right width nor the right dims."""
        W = self.dicts()
        K = K.astype(self.dtype)
        Fs = self.split(F.astype(self.dtype))
        def f(W_i, K_i, F_i):
            # Transpose W_i to (d_head, k) because ghostgrad expects (d_out, d_in)
            return ghostgrad(W_i.T, K_i, F_i, *args, **kwargs)

        Fs_g = jax.vmap(f, (0, -2, -2), -2)(W, K, Fs)
        if S is not None:
            Fs_g = jnp.einsum("...hd,...h->...hd", Fs_g, S.astype(self.dtype))
        return self.combine(Fs_g)

    def flatcos(self) -> Float[Array, ""]:
        """Mean off-diagonal cosine over the flattened dictionary, with
        the block structure accounted for.

        Of the `hk(hk-1)` ordered off-diagonal pairs, only the `hk(k-1)`
        inside a head can be nonzero; every cross-head pair is exactly 0.
        So this is `rowcos`'s per-head sum rescaled by the pair counts,
        and it stays a strictly weaker signal than `rowcos` exactly as it
        is in the parent."""
        n = self.k * self.h
        within = jnp.sum(self.rowcos()) * self.k * (self.k - 1)
        return within / (n * (n - 1))

    def tags(self) -> Float[Array, "n_tags d"]:
        """Atoms embedded in the layer's full output space, zero outside
        their own head's slice. The parent's raw concatenation would hand
        out `(n_tags, d // h)` rows that are not in the space every
        caller reads them as."""
        W = self.dicts()                                # (h, k, d_head)
        E = jnp.zeros((self.h, self.k, self.h, self.d_head), W.dtype)
        E = E.at[jnp.arange(self.h), :, jnp.arange(self.h), :].set(W)
        return einops.rearrange(E, "h k g d -> (h k) (g d)")
