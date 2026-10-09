import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple
from jaxtyping import Array, Float, UInt, PRNGKeyArray
import einops

from ontologize.fns.loss import identity, cossim, bcossim, entropy, l1, ghostgrad
from ontologize.fns.keys import get_activation
from ontologize.fns.pwak import pwak_kl
from .sparse import Sparse

class DictBlock(Sparse):
    """Multihead ontofeature submodule. It has no bias, and its `activation`
    defaults to none. It expects logits with an `h` axis, which `cluster`
    turns into classifications with the `select` rule (softmax by default)
    *before* `fwd`. If the preceding layer is interpreted as a classifier, a
    `DictBlock` returns a weighted sum of representative vectors for each
    classification based on the probability of that classification."""
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
    pwak_loss: bool = False

    noise: str = "none"
    sd: float = 0.0

    dtype_str: str = "bfloat16"
    dtype_p_str: str = "float32"

    def setup(self):
        super().setup()
        self.fn_sel = get_activation(self.select)
        self.n_tags = self.k * self.h
        self.n_feat = self.d * self.h
        # Shape is (h k d) rather than (h d k) so that tags() is contiguous
        self.weights = self.param(
            'weights',
            nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
            (self.h, self.k, self.d),
            dtype=self.dtype_p
        )

    def dicts(self) -> Float[Array, "h k d"]:
        """Returns `abs(self.weights)`. This prevents subtractive ontofeatures."""
        W = self.weights.astype(self.dtype)
        return jnp.abs(W)

    def cluster(self, K: Float[Array, "... h k"], temperature: float=1.0) -> Float[Array, "... h k"]:
        """Converts logits to probabilistic classifications, used for the weighted 
        sum of ontofeatures."""
        K = K.astype(self.dtype)
        return self.fn_sel(K / temperature)

    def fwd(self, P_0: Float[Array, "... h k"],
                S: Optional[Float[Array, "... h"]] = None, *args, **kwargs
               ) -> Float[Array, "... d"]:
        """Forward pass. Input `P_0` is expected to be probabilities from a multihead classifier.
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
        classification usage, not scaled output."""
        P = P_0.astype(self.dtype)
        if not self.hmean_loss:
            P = jax.lax.stop_gradient(P)
        Pm = P.reshape(-1, self.h, self.k).mean(0)
        logk = jnp.log2(jnp.asarray(self.k, Pm.dtype))
        return (logk - entropy(Pm)).mean()

    def pwak_kl(self, P: Float[Array, "... h k"],
                E: Optional[Float[Array, "... d_in"]] = None,
                pwak_s: int = 0, pwak_tau: float = 0.2) -> Float[Array, ""]:
        """KL(neighbourhood-consensus target ‖ classification), bits. The
        target is the batch's partition-gated diffusion of `P` over the
        affinity of `E` (the classifier input); design intent, measured
        behaviour (it softens rather than hardens -- it earns its keep as a
        regularizer), and caveats are recorded in `ontologize.fns.pwak`.
        Gated on the static `pwak_loss` flag: the target costs a dense
        (h, b, b) affinity graph, and `pwak_s` arrives traced from the
        schedule, so the graph cannot be pruned at compile time -- pwak-off
        models must return 0 here without tracing it at all. Also 0 when
        `E` is absent (inference paths), and exactly 0 at `pwak_s=0`."""
        if E is None or not self.pwak_loss:
            return jnp.zeros((), self.dtype)
        return pwak_kl(P.astype(self.dtype), E.astype(self.dtype),
                       pwak_s, pwak_tau, self.pwak_loss)

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
        on the classifications. Backpropagation reaches the `entropy`
        calculation only when `entropy_loss` is set."""
        F, P = self.withClusts(K, S, *args, **kwargs)
        return F, P, self.entropy(P, S)

    def withL1(self, Fs: Float[Array, "... h d"], isloss: bool=False
               ) -> Tuple[Float[Array, "... d"], Float[Array, ""]]:
        """Sums input over `h` axis, and returns the sum with the L1 of the
        per-head input `Fs` itself (`Sparse.l1`: summed over every element,
        batch included). Backpropagation reaches `L1` only when `sparse` is
        set; the `isloss` argument is unused."""
        F = self.combine(Fs)
        return F, self.l1(Fs)

    def drop_winners(self, K: Float[Array, "... h k"], p_drop: float=0.0,
                     rng: Optional[PRNGKeyArray]=None
                     ) -> Tuple[Float[Array, "... h k"],
                                Optional[PRNGKeyArray]]:
        """Winner dropout: with probability `p_drop` per head per sample, masks
        the argmax entry's logit to `-inf`, so `cluster` renormalizes over the
        remaining entries and the runner-up receives the gradient, however
        large the winner's margin. No-op when `rng` is `None` (inference)."""
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
                  E_cl: Optional[Float[Array, "... b d_in"]] = None,
                  pwak_s: int = 0, pwak_tau: float = 0.2, **kwargs
                  ) -> Tuple[Float[Array, "... b d"], Float[Array, "... b h k"],
                             Float[Array, "6"],
                             PRNGKeyArray]:
        """As `withEntropy`, but also calculates `L1`, two cosines, `hmean_kl`
        and `pwak_kl`. Applies `drop_winners` to `K` first (at rate
        `p_drop`). When `rng` is given, adds noise of scale `sd` (this
        block's `noise` function) to the per-head features. Returns their
        noised sum over heads, the classifications `P`, the statistics
        `[L1, entropy, cossim_b, cossim_h, KL_m, KL_pwak]` and the next key.
        `L1` is taken on the noised features. `cossim_b` (`bcossim_tags`)
        and `KL_m` (`hmean_kl`) come from `P`. `cossim_h` is the mean `abs`
        cosine between heads' clean outputs on each sample, self-pairs
        included. `KL_pwak` is 0 unless `pwak_loss` is set."""
        K, rng = self.drop_winners(K, p_drop, rng)
        Fs, P, H = self.withEntropy(K, S, *args, **kwargs)

        Fs_n, rng_next = self.addnoise(Fs, sd, rng)
        cossim_b = self.bcossim_tags(P, S)
        cossim_h = self.cossim(Fs)
        KL_m = self.hmean_kl(P)
        KL_p = self.pwak_kl(P, E_cl, pwak_s, pwak_tau)

        F, L1 = self.withL1(Fs_n)
        stats = [L1, H, cossim_b, cossim_h, KL_m, KL_p]
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
        and all other indices set to `1 / k`. These follow one all-uniform
        classification, so there are `h * k + 1` in all."""
        # b = 1 (uniform) + k * h (one-hot variations)

        # 1. Create the base uniform array (shape: 1, h, k)
        p = 1.0 / self.k
        uniform_base = jnp.full((1, self.h, self.k), p, dtype=self.dtype)

        # 2. Define a function that takes a (head, tag) pair and returns an (h, k) tensor
        def make_one_hot_tag(head, tag):
            # Start with uniform (h, k)
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
        # Result shape: (h*k, h, k)
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
        then scales each head of `P` by `scale`."""
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
