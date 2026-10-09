"""Sparse autoencoders, the field-standard baselines the Ontologizer is
compared against, as `ontologize` modules.

`SAE` is a top-k SAE (Gao et al. 2024) with a pre-subtracted decoder bias,
unit-norm decoder rows, tied initialization and aux-k dead-latent revival,
or a ReLU+L1 SAE when `topk=0`. `enc` selects how a latent's
pre-activation is read from the input: a linear map, the gated encoder of
Rajamanoharan et al. (2024), or a bilinear form (`BilinearSAE`). `Grouped`
(`ontologize.grouped`) replaces the global top-k with competition inside
groups of latents. `prefixes` adds Matryoshka nested-prefix losses to any
of them.

These are SAEs, not Ontologizers: the decoder is signed and linear, the
code carries magnitudes, and nothing is a dictionary lookup. The root
`sae.py` trains them; its `params.npz` keeps the flat key layout every
downstream script reads (`W_enc`, `W_dec`, ...), and `from_legacy` /
`to_legacy` convert between that and the parameter tree here.

Layout, following `Linear` (`weights` are `(d_out, d_in)`):

  encoder  Linear(d -> m)        weights (m, d), bias b_enc    "linear"
           Bilinear(d + 1 -> m)  weight (2, m, d+1), bias b_enc "bilinear"
  gate     Linear(d -> m)        weights (m, d), bias b_gate   "gated"
  r_mag, b_mag                   (m,)                          "gated"
  decoder  Linear(m -> d)        weights (d, m), bias b_dec

The decoder bias is subtracted from the input before encoding, so `b_dec`
is both the reconstruction's offset and the encoder's centring.
"""
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .layers.sparse import Sparse
from .layers.linear import Linear
from .layers.nlinear import Bilinear

AUX_COEF = 1 / 32  # aux-k loss scale (Gao et al. 2024)

ENCODERS = ("linear", "bilinear", "gated")

Params = Dict[str, Array]


def topk_relu(pre: Float[Array, "... m"], topk: int) -> Float[Array, "... m"]:
    """ReLU, then zero all but the `topk` largest activations per sample
    (`topk=0` keeps every positive one). Ties at the threshold all
    survive, so L0 can exceed `topk` only through exact ties."""
    a = jax.nn.relu(pre)
    if topk:
        thr = jax.lax.top_k(a, topk)[0][..., -1:]
        a = jnp.where(a >= thr, a, 0.0)
    return a


def eigenfeatures(W1: Float[Array, "m d1"], W2: Float[Array, "m d1"],
                  W_dec: Float[Array, "m d"]) -> Float[Array, "m d"]:
    """Input-space top eigenvector of each bilinear latent's symmetric form
    `sym(w1 w2^T)`. The form has rank at most 2, so the top-|eigenvalue|
    eigenvector is closed-form, `w1/|w1| + sign(w1.w2) w2/|w2|`, with no
    `eigh`. When w1.w2 < 0 its eigenvalue is negative: the form falls along
    it, so it is a direction the latent rejects, and the positive-eigenvalue
    eigenvector `w1/|w1| + w2/|w2|` is the one that excites it. The
    constant coordinate is dropped and the result unit-normalized and
    sign-aligned to the latent's decoder row (the form is even, so its sign
    is otherwise arbitrary)."""
    Un = W1 / (jnp.linalg.norm(W1, axis=-1, keepdims=True) + 1e-9)
    Vn = W2 / (jnp.linalg.norm(W2, axis=-1, keepdims=True) + 1e-9)
    c = jnp.where((W1 * W2).sum(-1, keepdims=True) >= 0, 1.0, -1.0)
    E = (Un + c * Vn)[:, :-1]
    E = E / (jnp.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)
    s = (E * W_dec).sum(-1, keepdims=True)
    return E * jnp.where(s >= 0, 1.0, -1.0)


class SAE(Sparse):
    """Sparse autoencoder over `d`-dimensional inputs with `m` latents.

    `topk` selects the activation rule (0 = plain ReLU, sparsified by
    `s_l1`); `s_l1` is the L1 coefficient, priced against the same whitened
    MSE as the reconstruction; `aux_k` is the number of dead latents that
    reconstruct the residual each step (0 = off); `prefixes` splits the
    latents into that many contiguous blocks and trains every nested prefix
    to reconstruct (1 = off). `enc` is "linear", "gated" or "bilinear".

    The gated encoder sets its own sparsity, so it needs `topk=0` and
    `prefixes=1`, and to train, `s_l1 > 0` and `aux_k=0`; see
    `gated_loss`."""
    d: int = 0
    m: int = 0
    topk: int = 32
    s_l1: float = 0.0
    aux_k: int = 512
    prefixes: int = 1
    enc: str = "linear"

    dtype_str: str = "float32"
    dtype_p_str: str = "float32"

    def setup(self):
        super().setup()
        self.check()
        kw = dict(dtype_str=self.dtype_str, dtype_p_str=self.dtype_p_str)
        if self.enc == "gated":
            # one tied encoder direction read twice: the gate decides WHICH
            # latents fire, the magnitude path (same direction, scaled by
            # exp(r_mag)) how much
            self.gate = Linear(d_in=self.d, d_out=self.m, biased=True, **kw)
            self.r_mag = self.param("r_mag", nn.initializers.zeros,
                                    (self.m,), self.dtype_p)
            self.b_mag = self.param("b_mag", nn.initializers.zeros,
                                    (self.m,), self.dtype_p)
        elif self.enc == "bilinear":
            # factors act on [x - b_dec; 1]: the constant coordinate gives
            # the quadratic form linear terms (resid_const's sign fix)
            self.encoder = Bilinear(d_in=self.d + 1, d_out=self.m,
                                    biased=True, **kw)
        else:
            self.encoder = Linear(d_in=self.d, d_out=self.m, biased=True,
                                  **kw)
        self.decoder = Linear(d_in=self.m, d_out=self.d, biased=True, **kw)

    def check(self):
        """Raises on a configuration the loss cannot honour."""
        if self.enc not in ENCODERS:
            raise ValueError(f"enc must be one of {ENCODERS}; got {self.enc!r}")
        if self.prefixes < 1 or self.m % self.prefixes:
            raise ValueError(f"m={self.m} must be a positive multiple of "
                             f"prefixes={self.prefixes}")
        if self.enc == "gated":
            if self.topk:
                raise ValueError("the gated encoder sets its own sparsity; "
                                 "needs topk=0")
            if self.prefixes != 1:
                raise ValueError("the gated encoder does not implement "
                                 "prefixes")

    @property
    def unit_rows(self) -> bool:
        """Whether the decoder rows are held at unit norm. Fixes the feature
        scale, so coefficients carry magnitude, and closes the L1
        norm-gaming loophole."""
        return True

    # ---- forward ----------------------------------------------------------

    def center(self, X: Float[Array, "... d"]) -> Float[Array, "... d"]:
        return X.astype(self.dtype) - self.decoder.bias.astype(self.dtype)

    def gated_pre(self, X: Float[Array, "... d"]
                  ) -> Tuple[Float[Array, "... m"], Float[Array, "... m"]]:
        """The gated encoder's gate and magnitude logits, from one tied
        direction: the magnitude path is the gate's projection scaled by
        `exp(r_mag)`, which is what lets an L1 on the gate set sparsity
        without shrinking the magnitudes it selects."""
        pre = self.gate.fwd(self.center(X))
        return (pre + self.gate.bias.astype(self.dtype),
                pre * jnp.exp(self.r_mag.astype(self.dtype))
                + self.b_mag.astype(self.dtype))

    def preacts(self, X: Float[Array, "... d"]) -> Float[Array, "... m"]:
        """Encoder logits. For the gated form, the magnitude path: the one
        whose ReLU carries the coefficient."""
        if self.enc == "gated":
            return self.gated_pre(X)[1]
        Xc = self.center(X)
        if self.enc == "bilinear":
            Xc = jnp.concatenate([Xc, jnp.ones_like(Xc[..., :1])], -1)
        return self.encoder(Xc)

    def activate(self, pre: Float[Array, "... m"]) -> Float[Array, "... m"]:
        """Encoder logits to latent activations."""
        return topk_relu(pre, self.topk)

    def encode(self, X: Float[Array, "... d"]) -> Float[Array, "... m"]:
        if self.enc == "gated":
            gate, mag = self.gated_pre(X)
            return jnp.where(gate > 0, jax.nn.relu(mag), 0.0)
        return self.activate(self.preacts(X))

    def decode(self, z: Float[Array, "... m"]) -> Float[Array, "... d"]:
        return self.decoder(z)

    def __call__(self, X: Float[Array, "... d"]) -> Float[Array, "... d"]:
        return self.decode(self.encode(X))

    def prefix_decodes(self, z: Float[Array, "b m"]) -> Float[Array, "p b d"]:
        """The reconstruction from each nested prefix of the `prefixes`
        latent blocks; the last is the full decode."""
        W = self.decoder.weights.astype(self.dtype)                # (d, m)
        zb = z.astype(self.dtype).reshape(z.shape[0], self.prefixes, -1)
        Wb = W.reshape(W.shape[0], self.prefixes, -1)
        parts = jnp.einsum("bpj,dpj->pbd", zb, Wb)
        return jnp.cumsum(parts, 0) + self.decoder.bias.astype(self.dtype)

    def eigenfeatures(self) -> Float[Array, "m d"]:
        """Each latent's closed-form top eigenvector (bilinear encoders
        only); see the module-level `eigenfeatures`."""
        if self.enc != "bilinear":
            raise ValueError("eigenfeatures need a bilinear encoder")
        W1, W2 = self.encoder.weight_ubind()
        return eigenfeatures(W1, W2, self.decoder.weights.T)

    # ---- objective ----------------------------------------------------------

    def loss(self, X: Float[Array, "b d"], w_sqrt: Float[Array, "d"],
             dead: Bool[Array, "m"]
             ) -> Tuple[Float[Array, ""],
                        Tuple[Float[Array, ""], Bool[Array, "m"]]]:
        """`(loss, (mse, fired))`: the training objective, its whitened
        reconstruction term, and which latents fired on this batch. `dead`
        marks the latents aux-k may recruit."""
        if self.enc == "gated":
            return self.gated_loss(X, w_sqrt)
        pre = self.preacts(X)
        z = self.activate(pre)
        if self.prefixes > 1:
            # every prefix of latent blocks must reconstruct on its own
            # (joint, no stop-gradient between blocks)
            recons = self.prefix_decodes(z)
            mse = (((recons - X[None]) * w_sqrt) ** 2).mean()
            recon = recons[-1]
        else:
            recon = self.decode(z)
            mse = (((recon - X) * w_sqrt) ** 2).mean()
        loss = mse
        if self.s_l1:
            loss = loss + self.s_l1 * jnp.abs(z).sum(-1).mean()
        if self.aux_k:
            # the top aux_k currently-dead latents reconstruct the
            # stop-gradiented residual, so they receive gradient
            a_dead = jnp.where(dead, jax.nn.relu(pre), 0.0)
            thr = jax.lax.top_k(a_dead, self.aux_k)[0][..., -1:]
            z_aux = jnp.where(a_dead >= thr, a_dead, 0.0)
            resid = jax.lax.stop_gradient(X - recon)
            aux = (((self.decoder.fwd(z_aux) - resid) * w_sqrt) ** 2).mean()
            loss = loss + AUX_COEF * aux * jnp.any(dead)
        return loss, (mse, (z > 0.0).any(0))

    def gated_loss(self, X: Float[Array, "b d"], w_sqrt: Float[Array, "d"]
                   ) -> Tuple[Float[Array, ""],
                              Tuple[Float[Array, ""], Bool[Array, "m"]]]:
        """Reconstruction, an L1 on the gate, and the auxiliary
        reconstruction that keeps the gate honest.

        The three are not separable. The gate reaches the code only through
        a step function, so reconstruction gives it no gradient and the L1
        alone would shut every gate; the auxiliary term asks `ReLU(gate)` to
        reconstruct through a frozen decoder, which is what teaches the gate
        what is worth opening for. Without it the model trains, reports a
        loss, and encodes nothing.

        Raises unless `s_l1 > 0` (the gate's L1 is the only sparsity) and
        `aux_k == 0` (aux-k acts on the magnitude path, which cannot reopen
        a shut gate, and would only report deadness as fixed). Encoding
        needs neither, so they are checked here rather than in `check`."""
        if not self.s_l1 > 0:
            raise ValueError("the gated encoder needs s_l1 > 0, the gate's "
                             "L1 coefficient")
        if self.aux_k:
            raise ValueError("the gated encoder needs aux_k=0")
        gate, mag = self.gated_pre(X)
        z = jnp.where(gate > 0, jax.nn.relu(mag), 0.0)
        mse = (((self.decode(z) - X) * w_sqrt) ** 2).mean()
        g = jax.nn.relu(gate)
        # frozen decoder: this term trains the gate, not the dictionary
        W = jax.lax.stop_gradient(self.decoder.weights.astype(self.dtype))
        b = jax.lax.stop_gradient(self.decoder.bias.astype(self.dtype))
        aux = (((jnp.einsum("...m,dm->...d", g, W) + b - X) * w_sqrt)
               ** 2).mean()
        loss = mse + self.s_l1 * g.sum(-1).mean() + aux
        return loss, (mse, (z > 0.0).any(0))

    # ---- operations on a parameter tree (no `apply` needed) -----------------

    def renorm(self, params: Params) -> Params:
        """`params` with each decoder row at unit norm, when `unit_rows`."""
        if not self.unit_rows:
            return params
        W = params["decoder"]["weights"]                           # (d, m)
        W = W / (jnp.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
        return {**params, "decoder": {**params["decoder"], "weights": W}}

    def initialize(self, rng: PRNGKeyArray, x_mean: Float[np.ndarray, "d"],
                   x_scale: float = 1.0) -> Params:
        """Parameters at the field-standard initialization: random unit
        decoder rows, `b_dec` at the data mean, and a tied encoder (the
        decoder's transpose; the gated form ties both paths to it). The
        bilinear factors are scaled so their product matches the tied
        linear init's pre-activation scale, `|x| / sqrt(d)`, on centered
        inputs of norm `x_scale`: bilinear logits grow with `|x|^2`, and
        top-k picks the extreme tail of `m` products, so an unscaled start
        puts the reconstruction orders of magnitude off the data. Draws
        match `sae.py`'s historical `init_params` for a given key."""
        r1, r2, r3 = jax.random.split(rng, 3)
        W = jax.random.normal(r1, (self.m, self.d), jnp.float32)
        W = W / jnp.linalg.norm(W, axis=-1, keepdims=True)         # rows
        legacy = {"W_dec": W, "b_dec": jnp.asarray(x_mean, jnp.float32)}
        if self.enc == "gated":
            legacy.update(W_gate=W.T, r_mag=jnp.zeros(self.m),
                          b_gate=jnp.zeros(self.m), b_mag=jnp.zeros(self.m))
        elif self.enc == "bilinear":
            s = (x_scale * self.d ** 0.5) ** -0.5
            legacy.update(
                b_enc=jnp.zeros(self.m),
                W_enc1=s * jax.random.normal(r2, (self.d + 1, self.m)),
                W_enc2=s * jax.random.normal(r3, (self.d + 1, self.m)))
        else:
            legacy.update(b_enc=jnp.zeros(self.m), W_enc=W.T)
        return from_legacy(legacy)

    def resample(self, params: Params, moments: Tuple[Params, ...],
                 X: Float[Array, "n d"], w_sqrt: Float[Array, "d"],
                 dead: Bool[np.ndarray, "m"], rng: np.random.Generator
                 ) -> Tuple[Params, Tuple[Params, ...]]:
        """Neuron resampling (Bricken et al. 2023): point each dead latent
        at an example the model reconstructs badly. Examples are drawn with
        probability proportional to the squared whitened loss; a dead
        latent's decoder row becomes the drawn example's centered unit
        direction, its encoder row the same direction at 0.2x the mean live
        encoder-row norm, and its per-latent biases (and the gated form's
        `r_mag`) go back to 0. `moments` are optimizer statistics shaped
        like `params` (Adam's mu and nu); every touched entry is zeroed, so
        stale momentum cannot drag the fresh direction away.

        For a gated model this is the only way back: a shut gate gets no
        gradient from reconstruction, and resetting `b_gate` reopens it. A
        bilinear factor pair has no defined resample direction."""
        if self.enc == "bilinear":
            raise ValueError("a bilinear factor pair has no defined resample "
                             "direction")
        dead = np.asarray(dead, bool)
        alive = ~dead
        z = self.apply({"params": params}, X, method=type(self).encode)
        R = self.apply({"params": params}, z, method=type(self).decode)
        err = np.asarray((((R - X) * w_sqrt) ** 2).sum(-1))
        p = err ** 2
        p = p / p.sum() if p.sum() > 0 else np.full(len(err), 1 / len(err))
        idx = rng.choice(len(err), size=int(dead.sum()), replace=True, p=p)
        v = np.asarray(X)[idx] - np.asarray(params["decoder"]["bias"])
        v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

        di = np.flatnonzero(dead)
        enc = "gate" if self.enc == "gated" else "encoder"
        rows = np.linalg.norm(np.asarray(params[enc]["weights"]), axis=-1)
        scale = 0.2 * (rows[alive].mean() if alive.any() else 1.0)
        v = jnp.asarray(v)

        new = {**params,
               "decoder": {**params["decoder"],
                           "weights": params["decoder"]["weights"]
                           .at[:, di].set(v.T)},
               enc: {"weights": params[enc]["weights"].at[di].set(scale * v),
                     "bias": params[enc]["bias"].at[di].set(0.0)}}
        if self.enc == "gated":
            for k in ("r_mag", "b_mag"):
                new[k] = params[k].at[di].set(0.0)

        keep = jnp.asarray(alive)
        mask = {"decoder": {"weights": keep[None, :],
                            "bias": jnp.ones_like(params["decoder"]["bias"],
                                                  bool)},
                enc: {"weights": keep[:, None], "bias": keep}}
        if self.enc == "gated":
            mask.update(r_mag=keep, b_mag=keep)
        moments = tuple(jax.tree_util.tree_map(lambda m, k: m * k, mo, mask)
                        for mo in moments)
        return new, moments


class BilinearSAE(SAE):
    """`SAE` with a bilinear encoder (Pearce et al. 2025): each latent's
    pre-activation is `(w1 . [x - b_dec; 1]) (w2 . [x - b_dec; 1]) + b`, the
    form the Ontologizer's classifier uses. Its latents admit closed-form
    analysis from the weights alone (`eigenfeatures`, and the encoder's
    `Bilinear.decompose`). Expect to need a lower learning rate."""
    enc: str = "bilinear"

    def check(self):
        if self.enc != "bilinear":
            raise ValueError(f"BilinearSAE has enc='bilinear'; got {self.enc!r}")
        super().check()


# ---- the flat `params.npz` layout --------------------------------------------

def legacy_encoder(legacy: Params) -> str:
    """Which encoder a flat parameter dict holds; carried by its keys, so
    consumers of a checkpoint need no flag."""
    if "W_gate" in legacy:
        return "gated"
    if "W_enc2" in legacy:
        return "bilinear"
    return "linear"


def from_legacy(legacy: Params) -> Params:
    """`sae.py`'s flat parameter dict to this module's tree. Applies to any
    tree shaped like the parameters (e.g. Adam's moments) as well."""
    tree = {"decoder": {"weights": legacy["W_dec"].T, "bias": legacy["b_dec"]}}
    enc = legacy_encoder(legacy)
    if enc == "gated":
        tree["gate"] = {"weights": legacy["W_gate"].T,
                        "bias": legacy["b_gate"]}
        tree["r_mag"] = legacy["r_mag"]
        tree["b_mag"] = legacy["b_mag"]
    elif enc == "bilinear":
        tree["encoder"] = {"weight": jnp.stack([legacy["W_enc1"].T,
                                                legacy["W_enc2"].T]),
                           "bias": legacy["b_enc"]}
    else:
        tree["encoder"] = {"weights": legacy["W_enc"].T,
                           "bias": legacy["b_enc"]}
    return tree


def to_legacy(tree: Params) -> Params:
    """The inverse of `from_legacy`."""
    legacy = {"W_dec": tree["decoder"]["weights"].T,
              "b_dec": tree["decoder"]["bias"]}
    if "gate" in tree:
        legacy.update(W_gate=tree["gate"]["weights"].T,
                      b_gate=tree["gate"]["bias"],
                      r_mag=tree["r_mag"], b_mag=tree["b_mag"])
    elif "weight" in tree["encoder"]:
        legacy.update(W_enc1=tree["encoder"]["weight"][0].T,
                      W_enc2=tree["encoder"]["weight"][1].T,
                      b_enc=tree["encoder"]["bias"])
    else:
        legacy.update(W_enc=tree["encoder"]["weights"].T,
                      b_enc=tree["encoder"]["bias"])
    return legacy
