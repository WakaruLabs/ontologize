import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Optional, Tuple, List
from jaxtyping import Array, Float, UInt, PRNGKeyArray
import einops

import re

from ontologize.fns.loss import ghost, identity
from ontologize.fns.pwak import pwak_kl, pwak_l2
from .dictblock import DictBlock, ConcatDictBlock
from .dictblock import entropy, l1, bcossim
from .nlinear import NLinear, NLinearBlock, Bilinear, BilinearBlock
from .linear import Linear, get_activation, get_dtype

class DictEnc(nn.Module):
    """Dictionary Encoder composed of a dense linear encoder, 
    multihead `NLinear` classifier, `DictBlock`, dense linear decoder,
    and optional bilinear router."""
    d_in: int = 0
    d_out: int = 0
    k: int = 0
    h: int = 0

    # gain-shape input separation (set from `Ontologizer.resid_gain`):
    # classify the unit-normalized input direction (shape) and scale
    # the layer's output contribution by the measured input norm (gain).
    # The trailing `n_const` coordinates (resid_const's constant)
    # are excluded from the norm and passed through untouched.
    gainshape: bool = False
    n_const: int = 0

    # placement in a latent wider than this layer's own output (set from
    # `Ontologizer.per_layer_dec`): `gained` writes the `d_out`-wide
    # contribution at offset `e_off` of an `e_lat`-wide vector, so each
    # layer owns a block of the decoder's input. 0 leaves it unplaced.
    e_off: int = 0
    e_lat: int = 0

    n: int = 2
    gate: str = "none"
    activation_cl: str = "none"
    biased_cl: bool = False

    select: str = "softmax"
    activation_dict: str = "none"
    norm_rows: bool = False
    signed: bool = False

    # heads write disjoint slices of the output and concatenate, instead
    # of each getting the whole of it and summing. Selects
    # ConcatDictBlock, whose docstring has the consequences. Kept as a
    # flag rather than a constructor argument because the checkpoint spec
    # is dataclasses.asdict(model) restored through Ontologizer(**spec),
    # which serializes a bool and not a class.
    concat: bool = False

    scaled: bool = False
    n_sc: int = 2
    activation_router: str = "none"
    gate_router: str = "none"
    biased_router: bool = False

    # let a head's gain go negative; see `scale`
    router_signed: bool = False

    # see `DictBlock.p_head_drop`
    p_head_drop: float = 0.0
    # see `DictBlock.fiber_rank` and the fields after it. The per-head
    # coordinates are a linear read of the classifier's input,
    # c_h = A_h U: the shaped input under `gainshape`, so the fiber, like
    # the entry it corrects, is scaled by the layer's gain afterwards
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

    sparse_K: bool = False
    sparse_F: bool = False
    sparse_S: bool = False
    entropy_loss: bool = False
    cossim_loss: bool = False
    bcossim_loss: bool = False
    kcossim_loss: bool = False
    flatcos_loss: bool = False
    support_loss: bool = False
    hmean_loss: bool = False
    pwak_loss: bool = False
    l2pwak_loss: bool = False

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

        if self.pwak_loss and re.fullmatch(r"top\d+", self.select.lower()):
            raise ValueError(
                "top-k selection and pwak_loss are mutually exclusive: "
                "the consensus target diffuses mass onto tags outside a "
                "sample's exact-zero support, where log2(P + eps) makes "
                "KL_pwak explode. `l2pwak_loss` has no log and composes "
                "with top-k selection.")

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
        self.dict = (ConcatDictBlock if self.concat else DictBlock)(
            k=self.k, 
            d=self.d_out, 
            h=self.h,
            select=self.select,
            activation=self.activation_dict,
            norm_rows=self.norm_rows, signed=self.signed,
            sparse=self.sparse_F, entropy_loss=self.entropy_loss,
            cossim_loss=self.cossim_loss, bcossim_loss=self.bcossim_loss,
            kcossim_loss=self.kcossim_loss,
            flatcos_loss=self.flatcos_loss,
            support_loss=self.support_loss,
            hmean_loss=self.hmean_loss,
            noise=self.noise_F, sd=self.sd_F,
            p_head_drop=self.p_head_drop,
            fiber_rank=self.fiber_rank, fiber_drop=self.fiber_drop,
            fiber_bound=self.fiber_bound, fiber_eps=self.fiber_eps,
            fiber_eps_layer=self.fiber_eps_layer,
            fiber_radial=self.fiber_radial,
            dtype_str=self.dtype_str,
            dtype_p_str=self.dtype_p_str
        )
        if self.fiber_rank:
            self.fiber_read = self.param(
                'fiber_read',
                nn.initializers.lecun_normal(in_axis=-1, out_axis=-2, batch_axis=(0,)),
                (1 if self.fiber_shared else self.h, self.fiber_rank, self.d_in),
                dtype=self.dtype_p)

        if self.scaled:
            self.router = ScType(
                d_in=self.d_in, 
                d_out=self.h, n=self.n,
                biased=self.biased_router,
                gate=self.gate_router,
                activation=self.activation_router,
                sparse=self.sparse_S,
                dtype_str=self.dtype_str,
                dtype_p_str=self.dtype_p_str
            )

    def fwd_dict(self, P: Float[Array, "... h k"], *args, **kwargs
                 ) -> Float[Array, "... d_out"]:
        """Forward `DictBlock` pass followed by `fwd_dec`."""
        return self.fwd_dec(self.dict.fwd(P, *args, **kwargs))

    def fwd(self, E: Float[Array, "... d_in"], *args, **kwargs
               ) -> Float[Array, "... d_out"]:
        """Forward pass starting from `self.classifier` with only unbiased linear
        transformations."""
        K = self.classifier.fwd(E)
        if self.scaled:
            S = self.router.fwd(E)
            return self.fwd_dict(K, S, *args, **kwargs)
        return self.fwd_dict(K, *args, **kwargs)

    def rev_dict(self, Y: Float[Array, "... d_out"], *args, **kwargs
                 ) -> Float[Array, "... h k"]:
        """Reversal of `fwd_dec` and `fwd_dict`."""
        return self.dict.rev(Y, *args, **kwargs)

    def rev(self, Y: Float[Array, "... d_out"], *args, **kwargs
               ) -> Float[Array, "... d_in"]:
        """Reversal of `fwd_dec`, `fwd_dict`, and `fwd_cl`. Does not account for 
        `self.router`. `S` must be passed explicitly."""
        K = self.rev_dict(Y, *args, **kwargs)
        return self.classifier.rev(K)

    def gainshape_in(self, E: Float[Array, "... d_in"]
                     ) -> Tuple[Float[Array, "... d_in"],
                                Optional[Float[Array, "... 1"]]]:
        """Gain-shape split of the layer input. Returns the input with its
        first `d_in - n_const` coordinates unit-normalized (the shape; the
        trailing constant coordinates are excluded from the norm and pass
        through untouched) and the measured norm (the gain, `(..., 1)`).
        The gain is stop-gradiented: it is measured, not trained, so it
        cannot become a gradient side-channel (the residual input is
        already stop-gradiented above layer 0; this also covers layer 0's
        encoder path). Identity `(E, None)` when `gainshape` is off."""
        if not self.gainshape:
            return E, None
        E = E.astype(self.dtype)
        d = E.shape[-1] - self.n_const
        D = E[..., :d]
        n = jnp.linalg.norm(D, axis=-1, keepdims=True)
        U = D / (n + jnp.finfo(self.dtype).eps)
        if self.n_const:
            U = jnp.concatenate([U, E[..., d:]], axis=-1)
        return U, jax.lax.stop_gradient(n)

    def pwak_in(self, U: Float[Array, "... d_in"]) -> Float[Array, "... d"]:
        """The semantic part of the shaped classifier input: the trailing
        `n_const` coordinates are constant, so they would distort the
        affinity's cosine and pad the prediction target with a dimension
        every neighbourhood predicts exactly. Dropped from both pwak
        stats."""
        if not self.n_const:
            return U
        return U[..., :U.shape[-1] - self.n_const]

    def withPWAK(self, stats: Float[Array, "9"], L1_K: Float[Array, ""],
                 P: Float[Array, "... h k"], U: Float[Array, "... d_in"],
                 pwak_s: int = 0, pwak_tau: float = 0.2,
                 L1_S: Float[Array, ""] = 0.0
                 ) -> Float[Array, "13"]:
        """Completes a `DictBlock.withStats` row into the `DictEnc` one:
        the classifier's `L1_K` in front, the two pwak stats after the
        seventh `DictBlock` entry, `cossim_flat`'s slot, `L1_S`, and then
        whatever `DictBlock` appended after that slot.

        `Hyperparams.s_loss`'s weight vector pairs against this order
        positionally, so a stat inserted anywhere but the end makes every
        weight beyond it multiply the wrong quantity. Each stat therefore
        keeps the position it had when it was added: `L1_S` stays at 11
        and `support_overlap`, added after it, goes at 12.

        `L1_S` is the router's own L1, which `L1_F` cannot stand in for:
        `L1_F` reduces `hfwd(P, S)`, so it sees the product `S * |W| * c`
        and shrinking either factor pays it down. Penalising `S` alone is
        the separation a gated architecture is built around -- the router
        decides how much each head speaks, the dictionary decides what it
        says. Zero when `scaled` is off, where there is no router to
        read."""
        return jnp.concatenate([
            jnp.stack([L1_K]), stats[:7],
            jnp.stack([self.pwak_kl(P, U, pwak_s, pwak_tau),
                       self.pwak_l2(P, U, pwak_s, pwak_tau)]),
            stats[7:8], jnp.stack([jnp.asarray(L1_S, stats.dtype)]),
            stats[8:]])

    def pwak_kl(self, P: Float[Array, "... h k"],
                E: Optional[Float[Array, "... d_in"]] = None,
                pwak_s: int = 0, pwak_tau: float = 0.2) -> Float[Array, ""]:
        """KL(neighbourhood-consensus target ‖ classification), bits. The
        target is the batch's partition-gated diffusion of `P` over the
        affinity of this layer's input.
        Gated on the static `pwak_loss` flag: the target costs a dense
        (h, b, b) affinity graph per layer, so a pwak-off model should not
        build it at all. Also 0 when `E` is absent (inference paths), and
        exactly 0 at `pwak_s=0`."""
        if E is None or not self.pwak_loss:
            return jnp.zeros((), self.dtype)
        return pwak_kl(P.astype(self.dtype),
                       self.pwak_in(E).astype(self.dtype),
                       pwak_s, pwak_tau, self.pwak_loss)

    def pwak_l2(self, P: Float[Array, "... h k"],
                E: Optional[Float[Array, "... d_in"]] = None,
                pwak_s: int = 0, pwak_tau: float = 0.2) -> Float[Array, ""]:
        """Leave-one-out (noise2self-style, not J-invariant) error of this
        layer's own input predicted from its
        partition-gated neighbourhood, as a fraction of the batch's
        spread -- 0 if neighbours predict a sample exactly, 1 if the graph
        does no better than the batch mean. See
        `ontologize.fns.pwak.pwak_l2`.

        The layer input is both the affinity source and the thing being
        predicted, which under `forward="resid"` is exactly what this
        layer is tasked with reconstructing. It is stop-gradiented
        so the only gradient path is the partition gate. The term scores
        the clustering, not the dictionary, and it cannot be reduced by
        changing what is being predicted. Gated on the static
        `l2pwak_loss` flag, 0 when `E` is absent (inference paths), and
        exactly 0 at `pwak_s=0`."""
        if E is None or not self.l2pwak_loss:
            return jnp.zeros((), self.dtype)
        return pwak_l2(P.astype(self.dtype),
                       self.pwak_in(E).astype(self.dtype),
                       pwak_s, pwak_tau)

    def fiber_coords(self, U: Float[Array, "... d_in"]
                     ) -> Optional[Float[Array, "... h r"]]:
        """Per-head fiber coordinates c_h = A_h U, read from the classifier's
        input (None without fibers)."""
        if not self.fiber_rank:
            return None
        C = jnp.einsum("...i,hri->...hr", U.astype(self.dtype),
                       self.fiber_read.astype(self.dtype))
        if self.fiber_shared:
            C = jnp.broadcast_to(C, C.shape[:-2] + (self.h, self.fiber_rank))
        return C

    def place(self, Y: Float[Array, "... d_out"]) -> Float[Array, "... e_lat"]:
        """Zero-pad a layer-output vector into this layer's block of the
        `e_lat`-wide latent; the identity when `e_lat` is 0."""
        if not self.e_lat:
            return Y
        pad = [(0, 0)] * (Y.ndim - 1) + [
            (self.e_off, self.e_lat - self.e_off - Y.shape[-1])]
        return jnp.pad(Y, pad)

    def gained(self, Y: Array, G: Optional[Float[Array, "... 1"]]) -> Array:
        """Scale a layer contribution by the measured input gain and
        `place` it. A zero gain (perfectly reconstructed residual) zeroes
        the contribution, so the eps-normalized zero direction never
        reaches the output.

        Every path that adds a layer's contribution to the running
        residual goes through here, so this is where a per-layer decoder's
        block placement happens. Analyses that decode dictionary vectors
        without a gain (atoms, entries) call `place` themselves."""
        if G is not None:
            Y = Y * G.astype(self.dtype)
        return self.place(Y)

    def classify(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... h k"]:
        """Forward pass up to `DictBlock.cluster`. Returns the classification tensor.
        With `gainshape`, classifies the unit shape of `E` (idempotent up to eps
        if `E` is already shaped)."""
        U, _ = self.gainshape_in(E)
        K = self.classifier(U)
        return self.dict.cluster(K, *args, **kwargs)

    def head_outputs(self, U: Float[Array, "... d_in"],
                     P: Float[Array, "... h k"]) -> Float[Array, "... h e"]:
        """Per-head dictionary output for a given classification `P` of the
        shaped input `U`, with the router gain and fiber coordinates applied
        as `withClusts` applies them, and the layer gain not yet applied.

        For analyses that hold or force a classification and need what the
        layer then writes: rebuilding it as `dict.hfwd(P)` drops the router
        and the fibers, and silently computes a different model's output
        once either is on."""
        S = self.scale(U) if self.scaled else None
        return self.dict.hfwd(P, S, C=self.fiber_coords(U))

    def scale(self, E: Float[Array, "... d_in"]) -> Float[Array, "... h"]:
        """Forward pass for `self.router`, which returns the router vector for the output of
        `self.dict`. If `self.scaled=False`, returns vector of all 1s.

        The sole site deciding the router's sign convention; every path that
        needs `S` routes through here so they cannot disagree. `withStats`
        alone repeats the rule, because it needs `L1_S` from the same call.

        The `abs` keeps a head's contribution additive: `S_h < 0` flips that
        head's whole output, and since `S` is a function of the input the
        sign would be per-sample, so a tag would mean presence for one
        sample and negation for another. `router_signed` allows that.

        No `gate_router` makes the `abs` redundant. The router computes
        `fn(gate(Ys[0]) * prod(Ys[1:]))`, so a non-negative gate bounds the
        gate factor while the magnitude factor stays a signed linear term
        and `S` still straddles zero."""
        if self.scaled:
            S = self.router(E)
            return S if self.router_signed else jnp.abs(S)
        shape = E.shape[:-1] + (self.h,)
        return jnp.full(shape, 1.0, dtype=self.dtype)

    def __call__(self, E: Float[Array, "... d_in"], *args, **kwargs) -> Float[Array, "... d_out"]:
        """Forward pass through `self.dict.fwd`. Reconstructs but does not unembed the input.
        """
        U, G = self.gainshape_in(E)
        P = self.dict.cluster(self.classifier(U))
        C = self.fiber_coords(U)
        if self.scaled:
            S = self.scale(U)
            return self.gained(self.dict.fwd(P, S, *args, C=C, **kwargs), G)
        return self.gained(self.dict.fwd(P, *args, C=C, **kwargs), G)

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
        U, G = self.gainshape_in(E)
        K_0 = self.classifier(U)
        C = self.fiber_coords(U)
        if self.scaled:
            S = self.scale(U)
            F, K = self.dict.withClusts(K_0, S, *args, C=C, **kwargs)
        else:
            F, K = self.dict.withClusts(K_0, *args, C=C, **kwargs)
        Y = self.dict.combine(F)
        return R + self.gained(Y, G), einops.rearrange(K, "... h k -> ... (h k)")
    
    def withStats(self, R: Float[Array, "... d_out"],
                  E: Float[Array, "... d_in"], sd_K: float=0.0, sd_F: float=0.0,
                  rng: Optional[PRNGKeyArray]=None,
                  *args, p_drop: float=0.0, p_revive: float=0.0,
                  revive_frac: float=0.5,
                  pwak_s: int = 0, pwak_tau: float = 0.2,
                  return_base: bool = False, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "11"], PRNGKeyArray]:
        """As `withClusts`, but also returns validation statistics
        `[L1_K, L1_F, entropy, cossim_batch, cossim_heads, cossim_tags,
        cossim_tags_max, KL_mean,
        KL_pwak, L2_pwak]`. If `noisefn_K` and `sd_K` are specified, adds
        noise to `K` before passing it to `self.dict`.
        Splits `rng` so `noisefn_K` and `noisefn_F` use different seeds.
        The two pwak stats are appended here rather than in `DictBlock`
        because both read the layer input `U` alongside `P`. With
        `return_base`, also returns this layer's gained fiber-free
        contribution, for `Ontologizer.base_aux`."""
        U, G = self.gainshape_in(E)
        K, L1_K = self.classifier.withL1(U)

        K_n, rng_F = self.classifier.addnoise(K, sd_K, rng)

        if self.scaled:
            # `scale`'s rule, repeated because `L1_S` comes from the same call
            S, L1_S = self.router.withL1(U)
            if not self.router_signed:
                S = jnp.abs(S)
        else:
            S, L1_S = None, 0.0

        out = self.dict.withStats(
                K_n, S, *args, **kwargs, sd=sd_F, rng=rng_F, p_drop=p_drop,
                p_revive=p_revive, revive_frac=revive_frac,
                C=self.fiber_coords(U), return_base=return_base)
        F, P, stats, rng_next = out[:4]

        stats = self.withPWAK(stats, L1_K, P, U, pwak_s, pwak_tau, L1_S)
        K = einops.rearrange(P, "... h k -> ... (h k)")
        if return_base:
            return (R + self.gained(F, G), K, stats, rng_next,
                    self.gained(out[4], G))
        return R + self.gained(F, G), K, stats, rng_next

    def withGhost(self, R: Float[Array, "... d_out"], R_g: Float[Array, "... d_out"],
                  E: Float[Array, "... d_in"], E_g: Optional[Float[Array, "... d_in"]],
                  temperature: float=1.0, sd_K: float=0.0, sd_F: float=0.0,
                  rng: Optional[PRNGKeyArray]=None, *args, p_drop: float=0.0,
                  p_revive: float=0.0, revive_frac: float=0.5,
                  pwak_s: int = 0, pwak_tau: float = 0.2, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "11"], PRNGKeyArray]:
        """As `withStats`, but also returns ghost gradient output."""
        U, G = self.gainshape_in(E)
        K = self.classifier(U)

        K_n, rng_F = self.classifier.addnoise(K, sd_K, rng)
        L1_K = self.classifier.l1(K)
        K_g = self.classifier.ghost(U, K)
        if E_g is not None:
            if G is not None:
                # the classifier consumed E / (n + eps); with n
                # stop-gradiented the normalization is linear in E, so the
                # encoder's ghost input takes the same rescale (only layer
                # 0 receives E_g; its constant coordinate, if any, arrives
                # as ghost-0, which the rescale preserves)
                E_g = E_g / (G + jnp.finfo(self.dtype).eps)
            K_g = K_g + self.classifier.fwd(E_g)
        if self.scaled:
            S = self.scale(U)
            L1_S = self.router.l1(S)
            S_g = self.router.ghost(U, S)
        else:
            S = None
            S_g = None
            L1_S = 0.0

        F, P, stats, rng_next = self.dict.withStats(
                K_n, S, sd=sd_F, rng=rng_F, temperature=temperature,
                p_drop=p_drop, p_revive=p_revive, revive_frac=revive_frac,
                C=self.fiber_coords(U), *args, **kwargs)
        # the ghost path is the linear surrogate of the base lookup and
        # carries no fiber
        F_R = self.dict.fwd(K_g, S_g, *args, **kwargs)
        F_g = self.dict.ghost(K, F, S) + F_R

        stats = self.withPWAK(stats, L1_K, P, U, pwak_s, pwak_tau)
        P = einops.rearrange(P, "... h k -> ... (h k)")

        # the ghost contribution takes the same gain as the live one: G is
        # a stop-gradiented per-sample constant, so this is a pure rescale
        # of both paths and the ghost algebra is unchanged
        return R + self.gained(F, G), R_g + self.gained(F_g, G), P, stats, rng_next

    def intervene(self, E: Float[Array, "... d_in"],
                  *args, temperature: float=1.0, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... (h k)"],
                             Float[Array, "... d_in"]]:
        """Classifies `E`, applies `DictBlock.intervene` arguments to the
        classification, and reverses the intervened classification through the
        classifier to reconstruct an input that would produce it. Returns the
        intervened layer output, the flattened intervened classification, and
        the reconstructed input. With `gainshape` the output keeps the
        sample's own measured gain (the intervention chooses a direction
        only), and the reconstructed input lives in shaped (unit-norm)
        input space."""
        U, G = self.gainshape_in(E)
        K = self.classifier(U)
        P = self.dict.cluster(K, temperature)
        P_int = self.dict.intervene(P, *args, **kwargs)
        E_int = self.classifier.rev(P_int)
        if self.scaled:
            S = self.scale(U)
        else:
            S = None
        # P_int is already intervened; do not pass *args again or additive/
        # router interventions would be applied twice.
        Fs = self.dict.hfwd(P_int, S, C=self.fiber_coords(U))
        F = self.dict.combine(Fs)
        return self.gained(F, G), einops.rearrange(P_int, "... h k -> ... (h k)"), E_int

