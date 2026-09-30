import jax
import jax.numpy as jnp
import flax.linen as nn
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from jaxtyping import Array, Bool, Float, UInt, PRNGKeyArray
import einops

from .layers.sparse import Sparse
from .layers.dictenc import DictEnc
from .layers.linear import Linear
from .fns.keys import get_activation, get_dtype

@dataclass
class DictIntervention:
    """Container for arguments to `DictBlock.intervene`."""
    scale: Optional[Float[Array, "h"]] = None
    k_set: Optional[UInt[Array, "n_set"]] = None
    h_set: Optional[UInt[Array, "n_set"]] = None
    h_unif: Optional[UInt[Array, "n_unif"]] = None
    h_zero: Optional[UInt[Array, "n_zero"]] = None
    k_add: Optional[UInt[Array, "n_add"]] = None
    h_add:Optional[UInt[Array, "n_add"]] = None
    k_sub: Optional[UInt[Array, "n_sub"]] = None
    h_sub:Optional[UInt[Array, "n_sub"]] = None

    def show(self):
        print(f"scale: {self.scale}   set: {self.h_set}->{self.k_set}   "
              f"unif: {self.h_unif}   zero: {self.h_zero}   "
              f"add: {self.h_add}->{self.k_add}   sub: {self.h_sub}->{self.k_sub}")

@dataclass
class OntologizerIntervention:
    """Container for specifying which `Ontologizer` layer to apply a `DictIntervention` to."""
    layer: int
    method: DictIntervention

    def show(self):
        print(f"ontologizer layer: {self.layer}")
        self.method.show()

@dataclass
class Ontologizer(nn.Module):
    """Multilayer Ontologizer where each layer is a `DictEnc`. Uses `DictEnc.withClusts`
    to add the output of each layer to a residual and pass either the flattened
    classifications or the running reconstruction residual to the next layer
    (see `forward`)."""
    d_in: int = 0
    d_out: int = 0
    e_dec: int = 0
    k: int = 0
    h: int = 0
    l: int = 0

    # layer-to-layer interface. "labels": layer i+1 consumes layer i's
    # flattened classification (h*k). "resid": layer i+1 consumes the
    # stop-gradiented output-space residual X - decode(R) (RVQ-style;
    # requires d_in == d_out). Labels quantized hard from the start starve
    # upper layers (the residual is code-orthogonal, so frozen constant
    # heads are the loss-optimal response); residual forwarding gives every
    # layer continuous input. The stop_gradient prevents lower layers from
    # writing a communication code into the residual instead of reducing it.
    # "resid_labels": the residual with layer i's flattened classification
    # concatenated onto it, so the next layer keeps the continuous input it
    # needs and gains the symbolic context labels forwarding was wanted
    # for. Both blocks are stop-gradiented, and upper classifiers widen by
    # h*k, so it is not parameter-matched to "resid" at equal h. See
    # `nextinput`.
    forward: str = "labels"
    # when True, `withStats`/`withGhost` return the per-prefix decodes
    # stacked on a leading axis (l, ..., d_out) instead of the final decode:
    # every prefix must reconstruct (deep supervision). `Hyperparams.loss`
    # needs no change -- the MSE broadcast averages over the prefix axis.
    # Recommended with forward="resid" to enforce monotone refinement.
    deepsup: bool = False
    # stagewise deep supervision (only meaningful with deepsup=True): each
    # prefix decode stop-gradients the residual accumulated by earlier
    # layers, so layer i trains only against its own residual target
    # sg(X - decode(R_{i-1})) -- pure RVQ/boosting credit assignment with
    # clean per-layer semantics. Prefix *values* are unchanged; only the
    # gradient path differs. When False (joint), earlier layers also
    # receive gradient from every later prefix loss.
    deepsup_sg: bool = False
    # residual input conditioning (forward="resid" only). `resid_norm`
    # classifies the unit-normalized residual direction: bilinear logits
    # scale with ||input||^2, so raw residuals (norm ~0.25 vs the unit-norm
    # layer-0 input) leave upper classifiers with logit spreads far below
    # the temperature -- flat softmax, vanishing gradients, weights stuck
    # at initialization (the uniform-classification collapse of the first
    # resid run). `resid_const` appends a constant 1 coordinate: unbiased
    # bilinear logits are even in their input (blind to residual sign; the
    # NLinear bias is added after the product, so biased_cl cannot fix
    # this), and the constant coordinate gives the quadratic form linear
    # terms in the input while keeping the pure-bilinear
    # eigendecomposition analysis (over d+1 dims). Every classifier input
    # gets the coordinate: `nextinput` appends it to the residual and
    # `constinput` to the encoder output -- layer-0 logits lack linear
    # terms just the same, and the anisotropic input cone can only emulate
    # them through cross-terms with the corpus mean. Changes classifier
    # input dims.
    resid_norm: bool = False
    resid_const: bool = False
    # gain-shape separation at each DictEnc input (forward="resid" only).
    # Moves resid_norm's normalization inside the layer: each DictEnc
    # unit-normalizes its own input (the shape), classifies that, and
    # scales its residual contribution by the measured input norm (the
    # gain, stop-gradiented -- measured, not trained). Closes the gap
    # resid_norm leaves open: the normalized classifier input no longer
    # says how large the residual is, forcing the dictionary to bake in a
    # corpus-average scale; here the magnitude is carried through exactly
    # (stagewise gain-shape VQ), and interventions pick a direction while
    # the sample keeps its own gain. Supersedes resid_norm's `nextinput`
    # normalization (skipped when set, so the DictEnc sees the raw
    # magnitude); resid_const's constant coordinate is appended to the RAW
    # residual and excluded from the norm inside the layer. Applies to
    # layer 0 too -- the identity for unit-norm SONAR input.
    resid_gain: bool = True

    # classifier (NLinearBlock) args
    n: int = 2
    gate: str = "none"
    activation_cl: str = "none"
    biased_cl: bool = False

    # DictBlock args
    select: str = "softmax"
    activation_dict: str = "none"
    norm_rows: bool = False
    # see ConcatDictBlock: heads take disjoint slices of e_dec
    # instead of summing into all of it. Needs e_dec % h == 0.
    concat: bool = False

    # decoder (Linear) args
    activation_dec: str = "none"
    biased_dec: bool = False

    # encoder (Linear) args
    encoded: bool = False
    e_enc: int = 0
    activation_enc: str = "none"
    biased_enc: bool = False

    #scaling (NLinear) args
    scaled: bool = False
    n_sc: int = 2
    activation_router: str = "none"
    gate_router: str = "none"
    biased_router: bool = False

    sparse_K: bool = False
    sparse_F: bool = False
    sparse_S: bool = False
    entropy_loss: bool = False
    cossim_loss: bool = False
    bcossim_loss: bool = False
    kcossim_loss: bool = False
    flatcos_loss: bool = False
    hmean_loss: bool = False
    pwak_loss: bool = False
    l2pwak_loss: bool = False

    noise_in: str = "none"
    noise_K: str = "none"
    noise_F: str = "none"
    sd_in: float = 0.0
    sd_K: float = 0.0
    sd_F: float = 0.0

    dtype_str: str = "float32"
    dtype_p_str: str = "float32"

    def dictenc(self, d_enc: int, n_const: int = 0) -> DictEnc:
        """Initialize `DictEnc`s for a specified input dimension with all other properties
        taken from the `Ontologizer`. `n_const` is the number of trailing
        constant input coordinates (resid_const) the layer's gain-shape
        split must exclude from the norm."""
        return DictEnc(
            d_enc, self.e_dec, self.k, self.h,
            gainshape=self.resid_gain, n_const=n_const,
            n=self.n,
            gate=self.gate, activation_cl=self.activation_cl, 
            biased_cl=self.biased_cl, 
            select=self.select, 
            activation_dict=self.activation_dict,
            norm_rows=self.norm_rows, concat=self.concat,
            scaled=self.scaled, n_sc=self.n_sc,
            activation_router=self.activation_router, gate_router=self.gate_router, 
            biased_router=self.biased_router,
            sparse_K=self.sparse_K, sparse_F=self.sparse_F,
            sparse_S=self.sparse_S,
            entropy_loss=self.entropy_loss,
            cossim_loss=self.cossim_loss, bcossim_loss=self.bcossim_loss,
            kcossim_loss=self.kcossim_loss,
            flatcos_loss=self.flatcos_loss,
            hmean_loss=self.hmean_loss,
            pwak_loss=self.pwak_loss, l2pwak_loss=self.l2pwak_loss,
            noise_K=self.noise_K, noise_F=self.noise_F,
            sd_K=self.sd_K, sd_F=self.sd_F,
            dtype_str=self.dtype_str, dtype_p_str=self.dtype_p_str)

    def setup(self):
        """Layer 0's input is the encoder output (`d_in`, or `e_enc` when
        `encoded`); later layers consume the layer-to-layer interface per
        `forward` (the `h * k` labels, or the `d_out` residual). With
        `resid_const`, every layer's input gains the trailing constant
        coordinate."""
        self.n_tags = self.k * self.h * self.l
        self.n_feat = self.e_dec * self.h * self.l
        self.dtype = get_dtype(self.dtype_str)
        self.dtype_p = get_dtype(self.dtype_p_str)

        if self.resid_gain and self.forward not in ("resid", "resid_labels"):
            raise ValueError(
                "resid_gain measures the residual magnitude and only makes "
                "sense with a residual-carrying forward mode ('resid' or "
                "'resid_labels'); labels forwarding hands the next layer a "
                "probability vector, whose norm is not a gain")

        if self.encoded and self.e_enc > 0:
            d_enc = self.e_enc
            self.encoder = Linear(
                d_in=self.d_in, 
                d_out=self.e_enc, 
                biased=self.biased_enc, 
                activation=self.activation_enc,
                noise=self.noise_in, sd=self.sd_in,
                dtype_str=self.dtype_str,
                dtype_p_str=self.dtype_p_str
            )
        else:
            d_enc = self.d_in
            self.encoder = Sparse(
                activation=self.activation_enc,
                noise=self.noise_in, sd=self.sd_in,
                dtype_str=self.dtype_str,
                dtype_p_str=self.dtype_p_str
            )

        # layer 0 never has a previous code, so its pass-through tail is
        # only resid_const's coordinate; the upper layers' tail also
        # carries the concatenated code under "resid_labels". Ordering is
        # [residual | const | code], so `gainshape_in` normalizes exactly
        # the residual and measures the gain on it alone.
        n_const = int(self.resid_const)
        if self.forward == "labels":
            d_next = self.k * self.h
            n_const = n_up = 0
        elif self.forward == "resid_labels":
            n_up = n_const + self.k * self.h
            d_next = self.d_out + n_up
        else:
            n_up = n_const
            d_next = self.d_out + n_const

        dictencs = [self.dictenc(d_enc + n_const, n_const)]
        for _ in range(self.l - 1):
            dictencs.append(self.dictenc(d_next, n_up))

        self.dictencs = dictencs

        self.decoder = Linear(
            d_in=self.e_dec, 
            d_out=self.d_out, 
            biased=self.biased_dec, 
            activation=self.activation_dec,
            dtype_str=self.dtype_str,
            dtype_p_str=self.dtype_p_str
        )

        self.interventions = [OntologizerIntervention(i, []) for i in range(self.l+1)]

    def fwd_dec(self, F: Float[Array, "... e_dec"]) -> Float[Array, "... d_out"]:
        """Forward decoder pass without applying bias or activation function"""
        return self.decoder.fwd(F)

    def prefix(self, R: Float[Array, "... e_dec"],
               R_0: Float[Array, "... e_dec"]) -> Float[Array, "... e_dec"]:
        """Residual prefix handed to the deepsup decode. With `deepsup_sg`,
        gradients stop at the residual accumulated before this layer (`R_0`),
        so only the newest layer's contribution trains against this prefix;
        the value is unchanged."""
        if self.deepsup_sg:
            return R - R_0 + jax.lax.stop_gradient(R_0)
        return R

    def resid(self, X: Float[Array, "... d_in"]
              ) -> Float[Array, "... e_dec"]:
        shape = list(X.shape)
        shape[-1] = self.e_dec
        R = jnp.zeros(shape, self.dtype)
        return R

    def encode(self, X: Float[Array, "... d_in"],
               sd: float=0.0, rng: Optional[PRNGKeyArray]=None
               ) -> Tuple[Float[Array, "... e_enc"],
                          Optional[PRNGKeyArray]]:
        X = X.astype(self.dtype)
        X_n, rng_K = self.encoder.addnoise(X, sd, rng)
        if self.encoded:
            return self.encoder(X_n), rng_K
        return X_n, rng_K

    def constinput(self, E: Array, const: float = 1.0) -> Array:
        """Appends resid_const's constant coordinate to the layer-0
        classifier input (the encoder output); `nextinput` does the same
        for the upper layers. `const` is the value appended: 1 for live
        inputs, 0 for ghost inputs (the coordinate is constant, so its
        gradient surrogate is zero). No-op unless a residual-carrying
        forward mode is set and `resid_const`; a `None` ghost input passes
        through."""
        if (E is None or not self.resid_const
                or self.forward not in ("resid", "resid_labels")):
            return E
        c = jnp.full(E.shape[:-1] + (1,), const, E.dtype)
        return jnp.concatenate([E, c], axis=-1)

    def nextinput(self, X: Float[Array, "... d_out"],
                  R: Float[Array, "... e_dec"],
                  K: Float[Array, "... (h k)"]) -> Array:
        """Input handed to the next layer, per `self.forward`: the flattened
        classification, or the stop-gradiented output-space residual
        (unit-normalized when `resid_norm`; with a constant 1 coordinate
        appended when `resid_const` -- see the field comments). With
        `resid_gain` the residual is passed RAW (const coordinate aside):
        the DictEnc normalizes internally so it can measure the gain.

        `resid_labels` appends the previous layer's flattened
        classification, giving `[residual | const | code]`. The code is
        carried alongside the residual direction, never instead of it:
        the residual is by construction the part of the input the code
        failed to explain, so its conditional mean given the code alone
        is near zero and a code-only layer has nothing to reduce. This
        mode buys the symbolic context labels forwarding was wanted for,
        at the cost of a wider upper-layer classifier."""
        if self.forward in ("resid", "resid_labels"):
            D = jax.lax.stop_gradient(
                X.astype(self.dtype) - self.decode(R))
            if self.resid_norm and not self.resid_gain:
                D = D / (jnp.linalg.norm(D, axis=-1, keepdims=True)
                         + jnp.finfo(self.dtype).eps)
            if self.resid_const:
                ones = jnp.ones(D.shape[:-1] + (1,), D.dtype)
                D = jnp.concatenate([D, ones], axis=-1)
            if self.forward == "resid_labels":
                if K is None:
                    raise ValueError(
                        "forward='resid_labels' needs the previous layer's "
                        "classification; this caller passed K=None, which "
                        "only 'resid' can ignore")
                D = jnp.concatenate(
                    [D, jax.lax.stop_gradient(K.astype(D.dtype))], axis=-1)
            return D
        return K

    def classify(self, E: Float[Array, "... e_enc"], *args,
                 X_ref: Optional[Float[Array, "... d_out"]]=None,
                 rng: Optional[PRNGKeyArray]=None, **kwargs
                 ) -> Tuple[
                         Float[Array, "... e_dec"],
                         Float[Array, "l ... (h k)"]]:
        """`X_ref` is the reconstruction target used as the residual
        reference when `forward == "resid"`; defaults to `E` (exact when the
        encoder is a noiseless passthrough)."""
        if X_ref is None:
            X_ref = E
        R = self.resid(E)
        E = self.constinput(E)
        Ps = []
        for i, dictenc in enumerate(self.dictencs):
            R, K = dictenc.withClusts(R, E, *args, **kwargs)
            Ps.append(K)
            if i < self.l - 1:
                E = self.nextinput(X_ref, R, K)

        return R, jnp.stack(Ps)

    def decode(self, E: Float[Array, "... e_dec"]) -> Float[Array, "... d_out"]:
        """`self.decoder` forward pass. Unembeds a reconstructed output of `self.dict`."""
        return self.decoder(E)

    def __call__(self, X: Float[Array, "... d_in"],
                 *args,
                 sd_in: float=0.0, rng: Optional[PRNGKeyArray]=None,
                 arglist: Optional[List[DictIntervention]] = None,
                 **kwargs,
                 ) -> Float[Array, "... d_out"]:
        E, rng_K = self.encode(X, sd_in, rng)
        F, Ps = self.classify(E, *args, X_ref=X, rng=rng_K, **kwargs)
        return self.decode(F)

    def withStats(self, X: Float[Array, "... d_in"],
                  sd_in: float=0.0, rng: Optional[PRNGKeyArray]=None, *args, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "l 8"]]:
        """Iterate `DictEnc.withStats` sequentially over layers.

        `Y` is `(..., d_out)`, or `(l, ..., d_out)` under `deepsup`: the
        per-prefix decodes stacked on a leading axis. `Hyperparams.loss`
        needs no change either way -- the target broadcasts against the
        prefix axis and the MSE mean averages over it, so every prefix
        carries weight `1/l`."""
        E, rng_K = self.encode(X, sd_in, rng)
        R = self.resid(E)
        E = self.constinput(E)
        stats, Ys = [], []
        for i, dictenc in enumerate(self.dictencs):
            R_0 = R
            R, K, stat, rng_K = dictenc.withStats(R, E, rng=rng_K, *args, **kwargs)
            stats.append(stat)
            if self.deepsup:
                Ys.append(self.decode(self.prefix(R, R_0)))
            if i < self.l - 1:
                E = self.nextinput(X, R, K)

        Y = jnp.stack(Ys) if self.deepsup else self.decode(R)
        return Y, jnp.stack(stats), rng_K

    def withGhost(self, X: Float[Array, "... d_in"],
                  sd_in: float=0.0, rng: Optional[PRNGKeyArray]=None, *args, **kwargs
                  ) -> Tuple[Float[Array, "... d_out"],
                             Float[Array, "... d_out"],
                             Float[Array, "l 8"],
                             Optional[PRNGKeyArray]]:
        """Iterate `DictEnc.withGhost` sequentially over layers.

        `Y` is the decode and `Y_g` its ghost-gradient counterpart. Without
        `deepsup` both are `(..., d_out)`; with `deepsup` both gain the
        leading prefix axis and are `(l, ..., d_out)`, so `Y_g[i]` pairs
        with `Y[i]`. The pairing is load-bearing: `Hyperparams.loss` forms
        `L2_g = lossfn(Y_g, X - Y)`, and decoding only the final ghost
        residual would leave `Y_g` a single `(..., d_out)` array that
        broadcasts silently against every prefix, scoring the last layer's
        ghost output against reconstruction errors it did not produce."""
        E, rng_K = self.encode(X, sd_in, rng)
        E_g = self.encoder.ghost(X, E)

        R = self.resid(E)
        R_g = self.resid(E)
        E = self.constinput(E)
        E_g = self.constinput(E_g, 0.0)
        stats, Ys, Ys_g = [], [], []
        for i, dictenc in enumerate(self.dictencs):
            R_0, R_g_0 = R, R_g
            R, R_g, K, stat, rng_K = dictenc.withGhost(
                    R, R_g, E, E_g, rng=rng_K, *args, **kwargs)
            stats.append(stat)
            E_g = None # only first layer should accept encoder ghost grad
            if self.deepsup:
                # `prefix` is the identity in value and only redirects
                # gradient, so the ghost accumulator takes the same
                # treatment as the reconstruction accumulator and
                # `deepsup_sg` credit assignment holds on both paths.
                Rp, Rp_g = self.prefix(R, R_0), self.prefix(R_g, R_g_0)
                Y_i = self.decode(Rp)
                Ys.append(Y_i)
                Ys_g.append(self.decoder.ghost(Rp, Y_i)
                            + self.decoder.fwd(Rp_g))
            if i < self.l - 1:
                E = self.nextinput(X, R, K)

        if self.deepsup:
            Y, Y_g = jnp.stack(Ys), jnp.stack(Ys_g)
        else:
            Y = self.decode(R)
            Y_g = self.decoder.ghost(R, Y) + self.decoder.fwd(R_g)

        # return the threaded key, not the input one, so the caller's next
        # step draws fresh noise (ontostate.train feeds this back in)
        return Y, Y_g, jnp.stack(stats), rng_K

    def withArgs(self, X: Float[Array, "... d_in"],
                 arglist: List[DictIntervention], *args,
                 sd_in: float=0.0, rng: Optional[PRNGKeyArray]=None, **kwargs
                 ) -> Tuple[Float[Array, "... d_out"],
                            Float[Array, "... (h k)"],
                            Float[Array, "l 8"], Optional[PRNGKeyArray]]:
        """As `withClust`, but accepts a list of `DictIntervention`s of length `l`.
        Each is passed as arguments to `DictBlock.intervene` for the corresponding
        layer."""
        E, rng_K = self.encode(X, sd_in, rng)

        R = self.resid(E)
        E = self.constinput(E)
        stats = []
        K = None
        for i, (dictenc, kwargs_l) in enumerate(zip(self.dictencs, arglist)):
            import dataclasses
            if dataclasses.is_dataclass(kwargs_l):
                intervention_dict = dataclasses.asdict(kwargs_l)
                intervention_dict = {k: v for k, v in intervention_dict.items() if v is not None}
            else:
                intervention_dict = kwargs_l
            R, K, stat, rng_K = dictenc.withStats(R, E, *args, rng=rng_K, **intervention_dict, **kwargs)
            stats.append(stat)
            if i < self.l - 1:
                E = self.nextinput(X, R, K)

        return self.decode(R), K, jnp.stack(stats), rng_K

    def decodeLayerEntries(self, layer: int, *args, **kwargs
                      ) -> Tuple[Float[Array, "(h k) d_out"],
                                 Float[Array, "(h k) (h k)"]]:
        """Takes `DictEnc.tags()` for a specified layer, then runs a forward
        pass of the remaining layers. The classification for each is taken to be a
        1-hot vector of shape `h * k`. With `forward == "resid"` there is no
        label path into later layers -- contributions are additive through
        the shared decoder -- so entries decode directly."""
        dictenc = self.dictencs[layer]
        R = dictenc.tags()
        K = jnp.eye(self.h * self.k, dtype=R.dtype)
        if self.forward == "labels":
            for i in range(layer + 1, self.l):
                R, K = self.dictencs[i].withClusts(R, K, *args, **kwargs)
        return self.decode(R), K

    def decodeEntries(self, *args, **kwargs
                      ) -> Tuple[Float[Array, "(l h k) d_out"],
                                 Float[Array, "l (h k) (h k)"]]:
        """Runs `decodeLayerEntries` for each layer."""
        Rs = []
        Ps = []
        for i in range(self.l):
            R, P = self.decodeLayerEntries(i, *args, **kwargs)
            Rs.append(R)
            Ps.append(P)

        return jnp.concatenate(Rs), jnp.stack(Ps)

    def decodeLayerUnif(self, layer: int, *args, **kwargs) -> Float[Array, "(h k) d_out"]:
        """Takes `DictEnc.decodeUniform()` for a specified layer, then runs a forward
        pass of the remaining layers. The classification for each is taken to be a
        1-hot vector of shape `h * k`."""
        dictenc = self.dictencs[layer]
        R = dictenc.decodeUniform()
        K = dictenc.dict.uniformTags()
        import einops
        K = einops.rearrange(K, "... h k -> ... (h k)")
        if self.forward == "labels":
            for i in range(layer + 1, self.l):
                R, K = self.dictencs[i].withClusts(R, K, *args, **kwargs)
        return self.decode(R)

    def decodeUniform(self, *args, **kwargs) -> Float[Array, "(l h k) d_out"]:
        """Runs `decodeLayerUnif` for each layer."""
        Rs = []
        for i in range(self.l):
            R = self.decodeLayerUnif(i, *args, **kwargs)
            Rs.append(R)

        return jnp.concatenate(Rs)

    def intervene(self, X: Float[Array, "... d_in"], layer: int,
                 *args,
                 temperature:float=1.0,
                 strength: float=1.0,
                 sd_in: float=0.0, rng: Optional[PRNGKeyArray]=None,
                 **kwargs,
                 ) -> Tuple[Float[Array, "... d_out"],
                            Float[Array, "... d_in"],
                            Float[Array, "l ... (h k)"]]:
        """Applies `DictBlock.intervene` arguments to the classification at
        `layer`, reverses the *change* in classification through the preceding
        classifiers (and encoder, if present) into input space, and adds it to
        `X` as a steering delta. The steered input is rescaled to the norm of
        `X` and run through a clean forward pass, so — unlike `withArgs`,
        which only affects layers after `layer` — the perturbation reaches
        every layer.

        Reversing only the change (rather than the full intervened
        classification) makes a no-op intervention the identity, and leaves
        heads already agreeing with the intervention untouched. `strength`
        scales the delta; because the adjoint chain does not preserve scale,
        the value needed to realize a tag assignment depends on the trained
        weights — compare the returned classifications against the intended
        intervention to verify, and increase `strength` if it is not
        realized.

        Returns the decoded output, the steered input, and the realized
        per-layer classifications of the clean pass."""
        E, rng_K = self.encode(X, sd_in, rng)

        # forward pass up to `layer` to obtain its natural input
        R = self.resid(E)
        E = self.constinput(E)
        for i in range(layer):
            R, K = self.dictencs[i].withClusts(R, E, temperature=temperature)
            E = self.nextinput(X, R, K)

        dictenc = self.dictencs[layer]
        P = dictenc.classify(E, temperature)
        P_int = dictenc.dict.intervene(P, *args, **kwargs)

        # reverse the classification delta through preceding layers. With
        # labels forwarding, the input of layer i+1 is the flattened
        # classification of layer i; with residual forwarding, the layer's
        # input X - decode(R) passes input-space deltas through identically,
        # so the classifier adjoint already lands in input space.
        D = dictenc.classifier.rev(P_int - P)
        if self.forward in ("resid", "resid_labels"):
            # keep only the residual block of the adjoint; the constant
            # and, under resid_labels, the code block are not input-space
            # directions (layer 0's input has neither)
            D = D[..., :self.d_out] if layer else D[..., :D.shape[-1]
                                                    - int(self.resid_const)]
        if self.forward == "labels":
            for i in reversed(range(layer)):
                D_i = einops.rearrange(D, "... (h k) -> ... h k", h=self.h)
                D = self.dictencs[i].classifier.rev(D_i)
        if self.encoded:
            D = self.encoder.rev(D)

        n_X = jnp.linalg.norm(X, axis=-1, keepdims=True)
        X_int = X + strength * D
        X_int = X_int * n_X / (jnp.linalg.norm(X_int, axis=-1, keepdims=True)
                               + 1e-8)

        E_int, _ = self.encode(X_int)
        F, Ps = self.classify(E_int, temperature=temperature)
        return self.decode(F), X_int, Ps
