"""The Base-LCM (LCM team 2024, arXiv:2412.08821, Sec. 2.3.1) in Flax, and
the document batches it trains on.

The model reads a document's run of SONAR embeddings and predicts, at each
position n, the embedding at n + 1:
- **robust scaler** (the paper's Eq. 4): `normalize(x) = (x - median) / IQR`
  per dimension, fitted on a sample of training rows and held in the
  `scaler` collection, outside the trained parameters;
- **PreNet**: a linear map from the normalized embedding to d_model (Eq. 1);
- **blocks**: a causal decoder-only transformer, pre-normalized with
  RMSNorm, with rotary position embeddings over the row's position in its
  run, SwiGLU, and dropout on each residual branch;
- **PostNet**: RMSNorm, a linear map back to SONAR's width, then
  `denormalize` (Eq. 2), so the prediction is in SONAR space and the loss
  (Eqs. 5-6) is the squared error there.

Shapes are fixed for the TPU: a batch is `b` documents, each padded to `t`
rows, with a mask. Attention reaches only earlier rows of the same
document, so a state at n does not depend on rows after n, on padding, or
on other documents in the batch. `states(..., block)` returns the residual
stream after `block` blocks (1 to `layers`), before PostNet's norm: the
rows a dictionary is trained on.
"""
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

NEG = -1e30          # the attention score of a key a query may not see


def rope(x: Float[Array, "b t h k"], pos: Int[Array, "b t"],
         base: float = 10000.0) -> Float[Array, "b t h k"]:
    """Rotary position embedding: rotate each pair (x_i, x_{i + k/2}) by
    `pos` times its frequency."""
    half = x.shape[-1] // 2
    freq = base ** (-jnp.arange(half, dtype=jnp.float32) / half)
    ang = pos[..., None].astype(jnp.float32) * freq
    cos, sin = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def allowed(mask: Bool[Array, "b t"]) -> Bool[Array, "b t t"]:
    """Which key j each query i may attend to: j <= i, and j a real row."""
    t = mask.shape[1]
    causal = jnp.tril(jnp.ones((t, t), bool))
    return causal[None] & mask[:, None, :]


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        g = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        return x * jax.lax.rsqrt((x * x).mean(-1, keepdims=True) + self.eps) * g


class Attention(nn.Module):
    d: int
    heads: int

    @nn.compact
    def __call__(self, x: Float[Array, "b t d"], pos: Int[Array, "b t"],
                 allow: Bool[Array, "b t t"]) -> Float[Array, "b t d"]:
        b, t, _ = x.shape
        k = self.d // self.heads
        qkv = nn.Dense(3 * self.d, use_bias=False, name="qkv")(x)
        qkv = qkv.reshape(b, t, 3, self.heads, k)
        q, key, v = rope(qkv[:, :, 0], pos), rope(qkv[:, :, 1], pos), qkv[:, :, 2]
        s = jnp.einsum("bihk,bjhk->bhij", q, key) / jnp.sqrt(k)
        a = jax.nn.softmax(jnp.where(allow[:, None], s, NEG), axis=-1)
        o = jnp.einsum("bhij,bjhk->bihk", a, v).reshape(b, t, self.d)
        return nn.Dense(self.d, use_bias=False, name="out")(o)


class SwiGLU(nn.Module):
    d: int
    d_ff: int

    @nn.compact
    def __call__(self, x: Float[Array, "b t d"]) -> Float[Array, "b t d"]:
        gate = nn.Dense(self.d_ff, use_bias=False, name="gate")(x)
        up = nn.Dense(self.d_ff, use_bias=False, name="up")(x)
        return nn.Dense(self.d, use_bias=False, name="down")(nn.silu(gate) * up)


class Block(nn.Module):
    d: int
    heads: int
    d_ff: int
    dropout: float

    @nn.compact
    def __call__(self, h: Float[Array, "b t d"], pos: Int[Array, "b t"],
                 allow: Bool[Array, "b t t"], train: bool) -> Float[Array, "b t d"]:
        a = Attention(self.d, self.heads, name="attn")(RMSNorm(name="norm_attn")(h), pos, allow)
        h = h + nn.Dropout(self.dropout)(a, deterministic=not train)
        f = SwiGLU(self.d, self.d_ff, name="mlp")(RMSNorm(name="norm_mlp")(h))
        return h + nn.Dropout(self.dropout)(f, deterministic=not train)


class BaseLCM(nn.Module):
    d_sonar: int = 1024
    d_model: int = 1024
    layers: int = 8
    heads: int = 16
    d_ff: int = 2816
    dropout: float = 0.1

    def setup(self):
        self.median = self.variable("scaler", "median", jnp.zeros, (self.d_sonar,))
        self.iqr = self.variable("scaler", "iqr", jnp.ones, (self.d_sonar,))
        self.pre = nn.Dense(self.d_model)
        self.blocks = [Block(self.d_model, self.heads, self.d_ff, self.dropout)
                       for _ in range(self.layers)]
        self.norm = RMSNorm()
        self.post = nn.Dense(self.d_sonar)

    def normalize(self, x: Float[Array, "... d_s"]) -> Float[Array, "... d_s"]:
        return (x - self.median.value) / self.iqr.value

    def denormalize(self, y: Float[Array, "... d_s"]) -> Float[Array, "... d_s"]:
        return self.median.value + self.iqr.value * y

    def residual(self, x: Float[Array, "b t d_s"], pos: Int[Array, "b t"],
                 mask: Bool[Array, "b t"], upto: int, train: bool = False
                 ) -> Float[Array, "b t d_m"]:
        """The residual stream after the first `upto` blocks."""
        allow = allowed(mask)
        h = self.pre(self.normalize(x))
        for blk in self.blocks[:upto]:
            h = blk(h, pos, allow, train)
        return h

    def __call__(self, x: Float[Array, "b t d_s"], pos: Int[Array, "b t"],
                 mask: Bool[Array, "b t"], train: bool = False
                 ) -> Float[Array, "b t d_s"]:
        """The prediction at each position of the next row, in SONAR space."""
        h = self.residual(x, pos, mask, self.layers, train)
        return self.denormalize(self.post(self.norm(h)))

    def states(self, x: Float[Array, "b t d_s"], pos: Int[Array, "b t"],
               mask: Bool[Array, "b t"], block: int) -> Float[Array, "b t d_m"]:
        """The residual stream after block `block` (1 to `layers`)."""
        if not 1 <= block <= self.layers:
            raise ValueError(f"block {block}: 1 to {self.layers}")
        return self.residual(x, pos, mask, block)


def fit_scaler(x: Float[np.ndarray, "n d"], floor: float = 1e-6
               ) -> Tuple[Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    """Per-dimension median and interquartile range of `x`. An IQR below
    `floor` times the median IQR is raised to it, so no dimension divides
    by zero."""
    x = np.asarray(x, np.float64)
    q25, med, q75 = np.percentile(x, [25, 50, 75], axis=0)
    iqr = q75 - q25
    iqr = np.maximum(iqr, floor * np.median(iqr))
    return med.astype(np.float32), iqr.astype(np.float32)


def next_targets(mask: Bool[Array, "b t"], adjacent: Bool[Array, "b t-1"],
                 adjacent_only: bool) -> Bool[Array, "b t-1"]:
    """Where the prediction at n is scored against row n + 1: both rows
    real, and, if `adjacent_only`, row n + 1 the next segment of the text
    (runs skip unusable segments, so a neighbouring row is not always the
    next sentence)."""
    both = mask[:, :-1] & mask[:, 1:]
    return both & adjacent if adjacent_only else both


def next_loss(pred: Float[Array, "b t d"], x: Float[Array, "b t d"],
              target: Bool[Array, "b t-1"]) -> Tuple[Float[Array, ""], Float[Array, ""]]:
    """The paper's loss: the squared error between the prediction at n and
    row n + 1, summed over dimensions and averaged over scored targets;
    and how many targets were scored."""
    err = ((pred[:, :-1] - x[:, 1:]) ** 2).sum(-1)
    n = target.sum()
    return jnp.where(target, err, 0.0).sum() / jnp.maximum(n, 1), n


def doc_spans(docstart: Int[np.ndarray, "n_docs"], n: int, lo: int, hi: int,
              min_len: int = 1) -> Tuple[Int[np.ndarray, "m"], Int[np.ndarray, "m"]]:
    """(first row, length) of the documents with at least `min_len` rows,
    all within [lo, hi). Raises unless `lo` and `hi` are document
    boundaries."""
    docstart = np.asarray(docstart, np.int64)
    for v in (lo, hi):
        i = int(np.searchsorted(docstart, v))
        if v not in (0, n) and not (i < len(docstart) and docstart[i] == v):
            raise ValueError(f"row {v} is not a document boundary")
    ends = np.r_[docstart[1:], n]
    length = ends - docstart
    keep = (docstart >= lo) & (ends <= hi) & (length >= min_len)
    return docstart[keep], length[keep]


def gather(arr: np.ndarray, index: Int[np.ndarray, "n 3"],
           starts: Int[np.ndarray, "b"], lengths: Int[np.ndarray, "b"], t: int
           ) -> Tuple[Float[np.ndarray, "b t d"], Int[np.ndarray, "b t"],
                      Bool[np.ndarray, "b t"], Bool[np.ndarray, "b t-1"]]:
    """A batch of documents padded to `t` rows: the rows as float32, each
    row's position in its run (index column 1), the mask of real rows, and
    whether row n + 1 is the next segment after row n (index column 2). A
    document of length 0 is an all-padding slot."""
    lengths = np.asarray(lengths, np.int64)
    if len(lengths) and lengths.max() > t:
        raise ValueError(f"a document has {lengths.max()} rows, more than t = {t}")
    off = np.arange(t)
    mask = off[None] < lengths[:, None]
    rows = (np.asarray(starts, np.int64)[:, None] + off[None])[mask]
    x = np.zeros((len(lengths), t, arr.shape[1]), np.float32)
    x[mask] = arr[rows]
    ix = np.asarray(index[rows])
    pos = np.zeros((len(lengths), t), np.int32)
    seg = np.full((len(lengths), t), -2, np.int64)
    pos[mask], seg[mask] = ix[:, 1], ix[:, 2]
    adjacent = mask[:, 1:] & mask[:, :-1] & (seg[:, 1:] == seg[:, :-1] + 1)
    return x, pos, mask, adjacent


def batches(starts: Int[np.ndarray, "m"], lengths: Int[np.ndarray, "m"], b: int
            ) -> list:
    """`(starts, lengths)` in consecutive groups of `b`, the last padded
    with empty documents so every batch has the same shape."""
    out = []
    for i in range(0, len(starts), b):
        s, n = starts[i:i + b], lengths[i:i + b]
        if len(s) < b:
            s = np.r_[s, np.zeros(b - len(s), np.int64)]
            n = np.r_[n, np.zeros(b - len(n), np.int64)]
        out.append((s, n))
    return out
