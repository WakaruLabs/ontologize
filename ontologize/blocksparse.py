"""`BlockSparse`: the vanilla block-sparse featurizer of Fel et al. (2026),
the block-structured baseline.

The `m` latents are partitioned into `m // block` contiguous blocks of
`block` latents. The code keeps the `topk` blocks whose pre-codes have the
largest L2 norm, signs and all, and zeroes the rest: the paper's block
projection, the block analogue of TopK. A block is a subspace with free
signed coordinates, chosen sparsely and varying continuously inside its
span, the opposite unit to an Ontologizer head, which is always active and
names one of its entries. The paper's model of an activation is a sum of a
few points on low-dimensional manifolds, to which block sparsity is the
matched prior.

What the paper fixes, and this keeps:
- a free linear encoder and a linear decoder, untied;
- each decoder block on the unit Frobenius ball (`renorm`), which fixes the
  scale the code and the dictionary would otherwise trade, and with it the
  top-k selection;
- a tied start, the encoder scaled so block norms begin at the input's
  scale (`initialize`);
- reconstruction plus the block analogue of Gao et al.'s auxiliary loss at
  weight 1 / topk, in which the runner-up blocks reconstruct what the
  selected ones missed (`loss`).

What it changes, as every `sae.py` model does: the reconstruction is the
whitened MSE, and the decoder bias is subtracted from the input before
encoding. The paper's decoder has no bias, and its inputs are rescaled to
mean norm sqrt(d) instead.

The rows inside a block are one basis of its span. Rotating them inside the
block changes no reconstruction, so a single row means nothing alone.

Layout: the encoder is held by block, as `W_blk` (d, G, block) and `b_blk`
(G, block) with G = m // block, in the parameter tree and in `sae.py`'s
flat `params.npz` alike, in place of `W_enc` and `b_enc`. Block g's
latents are `g * block` to `(g + 1) * block - 1`. The decoder is
`SAE`'s. `W_blk`'s shape carries the block size, so a checkpoint names its
own code (`ontologize.sae.legacy_block`).
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from .layers.linear import Linear
from .layers.sparse import Sparse
from .sae import SAE, Params, from_legacy


def block_topk(pre: Float[Array, "... m"], topk: int, block: int
               ) -> Float[Array, "... m"]:
    """Keep the `topk` blocks of `block` consecutive latents with the
    largest L2 norm, signs and all, and zero the rest. Ties at the
    threshold all survive, as in `topk_relu`."""
    g = pre.reshape(*pre.shape[:-1], -1, block)
    n = (g ** 2).sum(-1)
    thr = jax.lax.top_k(n, topk)[0][..., -1:]
    return jnp.where((n >= thr)[..., None], g, 0.0).reshape(pre.shape)


class BlockSparse(SAE):
    """`SAE` whose code keeps the `topk` largest-norm blocks of `block`
    latents. `topk` counts blocks, so L0 is `topk * block` latents. The
    encoder is linear and `prefixes` is not implemented. Training needs
    `aux_k=0` and `s_l1=0` (see `loss`); encoding needs neither."""
    block: int = 0

    def setup(self):
        Sparse.setup(self)
        self.check()
        G = self.m // self.block
        self.W_blk = self.param(
            "W_blk", nn.initializers.lecun_normal(in_axis=0, out_axis=(1, 2)),
            (self.d, G, self.block), self.dtype_p)
        self.b_blk = self.param("b_blk", nn.initializers.zeros,
                                (G, self.block), self.dtype_p)
        self.decoder = Linear(d_in=self.m, d_out=self.d, biased=True,
                              dtype_str=self.dtype_str,
                              dtype_p_str=self.dtype_p_str)

    def check(self):
        super().check()
        if self.enc != "linear":
            raise ValueError("the block-sparse featurizer's encoder is "
                             f"linear; got {self.enc!r}")
        if self.block < 1 or self.m % self.block:
            raise ValueError(f"m={self.m} must be a positive multiple of "
                             f"block={self.block}")
        if self.prefixes != 1:
            raise ValueError("BlockSparse does not implement prefixes")

    @property
    def unit_rows(self) -> bool:
        """Blocks, not rows, are held (`renorm`)."""
        return False

    # ---- forward ----------------------------------------------------------

    def preacts(self, X: Float[Array, "... d"]) -> Float[Array, "... m"]:
        """The pre-codes `(x - b_dec) W + b`, on the latent axis."""
        pre = jnp.einsum("...d,dgj->...gj", self.center(X),
                         self.W_blk.astype(self.dtype))
        pre = pre + self.b_blk.astype(self.dtype)
        return pre.reshape(*pre.shape[:-2], self.m)

    def activate(self, pre: Float[Array, "... m"]) -> Float[Array, "... m"]:
        """The block projection (`block_topk`). Checked here rather than in
        `check`, since decoding needs no `topk`."""
        if not 1 <= self.topk <= self.m // self.block:
            raise ValueError(f"topk={self.topk} must count between 1 and "
                             f"{self.m // self.block} blocks")
        return block_topk(pre, self.topk, self.block)

    # ---- objective ----------------------------------------------------------

    def loss(self, X: Float[Array, "b d"], w_sqrt: Float[Array, "d"],
             dead: Bool[Array, "m"]
             ) -> Tuple[Float[Array, ""],
                        Tuple[Float[Array, ""], Bool[Array, "m"]]]:
        """`(loss, (mse, fired))`, as in `SAE.loss`. The whitened
        reconstruction, plus the paper's auxiliary term at weight 1 / topk.
        In that term the next `topk` blocks by norm after the selected ones
        reconstruct the stop-gradiented residual. It reaches every sample's
        near misses rather than the dead latents, so `dead` is unused.

        Raises unless `aux_k == 0` (the auxiliary term replaces aux-k),
        `s_l1 == 0` (the code's only sparsity is the top-k over blocks) and
        the runner-ups fit (`2 * topk` blocks at most)."""
        if self.aux_k or self.s_l1:
            raise ValueError("the block-sparse featurizer trains with "
                             "aux_k=0 and s_l1=0")
        G = self.m // self.block
        if 2 * self.topk > G:
            raise ValueError(f"topk={self.topk} blocks and as many "
                             f"runner-ups need 2 * topk <= {G} blocks")
        pre = self.preacts(X)
        z = self.activate(pre)
        recon = self.decode(z)
        mse = (((recon - X) * w_sqrt) ** 2).mean()
        g = pre.reshape(*pre.shape[:-1], G, self.block)
        n = (g ** 2).sum(-1)
        thr = jax.lax.top_k(n, 2 * self.topk)[0][..., -1:]
        idle = (z.reshape(g.shape) == 0.0).all(-1)
        z_run = jnp.where(((n >= thr) & idle)[..., None], g, 0.0
                          ).reshape(pre.shape)
        resid = jax.lax.stop_gradient(X - recon)
        aux = (((self.decoder.fwd(z_run) - resid) * w_sqrt) ** 2).mean()
        return mse + aux / self.topk, (mse, (z != 0.0).any(0))

    # ---- operations on a parameter tree (no `apply` needed) -----------------

    def renorm(self, params: Params) -> Params:
        """`params` with each decoder block projected onto the unit
        Frobenius ball. Blocks inside it keep their norm."""
        W = params["decoder"]["weights"]                           # (d, m)
        Wb = W.reshape(W.shape[0], -1, self.block)
        n = jnp.sqrt((Wb ** 2).sum((0, 2), keepdims=True))
        Wb = Wb / jnp.maximum(n, 1.0)
        return {**params, "decoder": {**params["decoder"],
                                      "weights": Wb.reshape(W.shape)}}

    def initialize(self, rng: PRNGKeyArray, x_mean: Float[np.ndarray, "d"],
                   x_scale: float = 1.0) -> Params:
        """Parameters at the paper's tied initialization.
        - Each decoder block is a random orthonormal frame scaled to
          Frobenius norm 1.
        - `b_dec` is at the data mean.
        - The encoder is the decoder's transpose times `block`, so each
          block decodes the orthogonal projection of the centred input onto
          its span, at the input's scale.

        At `block=1` this is `SAE.initialize`'s tied initialization, and
        `x_scale` is unused for the same reason."""
        G, b = self.m // self.block, self.block
        F = jnp.linalg.qr(jax.random.normal(rng, (G, self.d, b),
                                            jnp.float32))[0]    # (G, d, b)
        return from_legacy({
            "W_dec": (jnp.swapaxes(F, 1, 2) / b ** 0.5).reshape(self.m,
                                                                self.d),
            "b_dec": jnp.asarray(x_mean, jnp.float32),
            "W_blk": b ** 0.5 * jnp.transpose(F, (1, 0, 2)),
            "b_blk": jnp.zeros((G, b), jnp.float32)})

    def resample(self, params: Params, moments: Tuple[Params, ...],
                 X: Float[Array, "n d"], w_sqrt: Float[Array, "d"],
                 dead: Bool[np.ndarray, "m"], rng: np.random.Generator
                 ) -> Tuple[Params, Tuple[Params, ...]]:
        """Not defined: the runner-up term in `loss` is this model's
        revival."""
        raise ValueError("a block-sparse featurizer has no resampling; its "
                         "runner-up auxiliary loss revives blocks")
