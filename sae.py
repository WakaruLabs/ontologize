"""Standalone SAE baseline trainer for the Ontologizer comparison.

Trains a top-k sparse autoencoder (Gao et al. 2024) -- or a vanilla ReLU+L1
SAE with --l1 -- on the same precomputed SONAR embedding cache and the same
whitened MSE objective as sonar.py, so FVU_w numbers are directly
comparable. Deliberately NOT built from DictEnc: the abs()'d dictionary
and softmax selection make it structurally not an SAE, and the baseline
should be the field-standard architecture at its best-practice
configuration (pre-subtracted decoder bias, unit-norm decoder rows, tied
init, aux-k dead-latent revival -- or, for the ReLU+L1 arm,
--resample-every swaps in that lineage's neuron resampling), not a
handicapped reimplementation. The models are `ontologize.sae.SAE`,
`BilinearSAE` and `ontologize.grouped.Grouped`; this script is their
trainer. Checkpoints keep the flat key layout (W_enc, W_dec, ...) every
downstream script reads, and the module-level functions here (`encode`,
`decode`, `preacts`, ...) take that layout and delegate to the classes
(`from_legacy` / `to_legacy` convert).

Matching the resid_nc Ontologizer (d=1024, e_dec=2048, k=32, h=32, l=5;
~23M params, l*h*k = 5120 dictionary entries, hard code = 800 bits/sample):
  matched dictionary:  --m 5120   (default)
  matched parameters:  --m 11264
  code-size sweep:     --topk 16 / 32 / 64 / 128
                       (top-k bits/sample ~= L0 * (log2 m + coeff bits),
                       so topk=32 at m=5120 sits at the 800-bit point)

Unlike sonar.py this takes CLI flags (it exists to be swept). The final
eval slice is the cache TAIL (--eval-rows, never trained on here).
sonar.py now holds out the same tail (its `holdout` config, same 32768
default), so the tail is out-of-sample for both models -- but Ontologizer
checkpoints trained before that holdout landed saw the full cache, so for
writeups comparing against those, re-evaluate both models on freshly
encoded data.

Structural-ablation rungs (each adds one Ontologizer-style commitment to
the standard architecture, so the interpretability battery can attribute
gains to specific structure rather than to the whole architecture):

  --groups G [--group-fn top1|softmax]
      Partition the m latents into G contiguous groups with per-group
      competition (head analogue). top1: the winning latent per group
      keeps its ReLU magnitude, so L0 <= G (a group whose winner is
      negative stays silent) -- the "hard code, trained natively" rung.
      softmax: each group emits a full distribution over its members
      (dense, sums to 1 per group; decoder rows NOT unit-normalized in
      this mode, since coefficients are bounded and the rows must carry
      magnitude) -- one DictEnc-like layer without the bilinear
      classifier. Mutually exclusive with --topk.
  --prefixes P
      Matryoshka/deepsup analogue (layer analogue): latents are ordered
      into P contiguous blocks and the loss is the mean whitened MSE over
      the P nested prefix reconstructions (joint, no stop-gradient --
      matching the validated deepsup_sg=False configuration). The
      activation rule (topk/groups) is applied once, globally, before
      prefix masking. NOTE: the loss.csv mse_w column becomes the mean
      over prefixes. Composes with --groups when G is a multiple of P.

  --enc bilinear
      Bilinear encoder (Pearce et al. 2025 style, matching the
      Ontologizer's classifier): each latent's pre-activation is
      (w1.[x-b_dec; 1]) * (w2.[x-b_dec; 1]) + b -- the constant
      coordinate gives the quadratic form linear terms, the same
      sign-blindness fix as sonar.py's resid_const. Latents then admit
      parameter-based "eigenfeature" analysis: eigenfeatures() returns
      each latent's closed-form top eigenvector (the rank-2 symmetric
      form sym(w1 w2^T) needs no eigh), which autointerp.py decodes
      through M2M100 as a third description mode ("eig") next to the
      W_dec row ("params") and top activations ("acts"). Composes with
      any activation rule; expect to need a lower --lr.

Block-sparse featurizer (Fel et al. 2026, the vanilla form;
`ontologize.blocksparse.BlockSparse`), a block-structured baseline rather
than a rung:

  --block B
      The m latents form m/B blocks of B signed coordinates, and the code
      keeps the --topk blocks of largest L2 norm, so --topk counts blocks
      and L0 is topk*B latents. Each decoder block is held on the unit
      Frobenius ball, the encoder starts tied, and the next --topk blocks
      by norm reconstruct the residual at weight 1/topk in place of aux-k,
      so it needs --aux-k 0. Linear encoder only; no --groups, --prefixes,
      --l1 or --resample-every. The encoder is saved by block, as W_blk
      (d, m/B, B) and b_blk, and encode/preacts refuse such a checkpoint
      unless passed the same block=B: a consumer that would read --topk as
      latents, or price the support per latent, fails instead of
      mis-encoding it. dead, revived and new_dead count latents, B per
      block.

  uv run python sae.py                          # m=5120 topk=32
  uv run python sae.py --m 11264 --topk 64
  uv run python sae.py --topk 0 --l1 3e-4       # vanilla ReLU+L1
  uv run python sae.py --topk 0 --l1 3e-4 --aux-k 0 --resample-every 5000
                                                # canonical L1-lineage revival
  uv run python sae.py --m 5120 --topk 0 --groups 160          # 160 "heads"
  uv run python sae.py --m 5120 --topk 0 --groups 160 --group-fn softmax
  uv run python sae.py --m 5120 --topk 32 --prefixes 5         # 5 "layers"
  uv run python sae.py --m 5120 --topk 32 --enc bilinear --lr 1e-4
  uv run python sae.py --m 5120 --block 4 --topk 8 --aux-k 0
                                                # 8 of 1280 blocks: 32
                                                # coefficients, as --topk 32

Resumable: state is snapshotted atomically every --save-each steps and
picked up automatically on restart. loss.csv columns:
step, loss, mse_w, fvu_eval, l0_eval, l1_eval, dead, revived, new_dead,
use_ent, hifreq, wdec_maxcos, wdec_meancos, fvu_train
(runs from before l1_eval have 6 columns ending at dead; runs from
before the health stats have 7).
  l1_eval        held-out mean per-sample |z|_1, logged unpenalized in
                 every mode -- its trajectory separates penalty-driven
                 sparsity (--l1) from what the task and the activation
                 rule produce on their own
  revived,       dead-mask churn between consecutive evals: revival
  new_dead       efficacy (resampling predicts step-drops that stick;
                 aux-k on the L1 arm predicts standing churn at the
                 firing boundary, since revival fights the penalty)
  use_ent        effective number of latents in use (exp-entropy of the
                 firing-frequency distribution; KL_m's SAE counterpart)
  hifreq         latents firing on >10% of eval samples (dense features,
                 which autointerp reliably scores poorly)
  wdec_maxcos,   off-diagonal |cos| of unit-normalized decoder rows:
  wdec_meancos   duplication/splitting forming over training (cossim_h's
                 counterpart; headcoh/splitting measure this post hoc)
  fvu_train      whitened FVU on a fixed held-in slice with its own
                 constant-predictor baseline -- fvu_train vs fvu_eval is
                 the generalization gap
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import functools
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Tuple
from tqdm import tqdm

from ontologize.sae import (AUX_COEF, SAE, BilinearSAE, from_legacy,
                            legacy_block, legacy_encoder, to_legacy,
                            topk_relu)
from ontologize.sae import eigenfeatures as _eigenfeatures
from ontologize.grouped import Grouped, group_softmax, group_top1
from ontologize.blocksparse import BlockSparse, block_topk


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=5120,
                   help="latents (5120 = matched dictionary, 11264 = matched params)")
    p.add_argument("--topk", type=int, default=32,
                   help="active latents per sample; 0 = vanilla ReLU+L1 mode")
    p.add_argument("--l1", type=float, default=0.0,
                   help="L1 coefficient (used when --topk 0)")
    p.add_argument("--groups", type=int, default=0,
                   help="partition latents into G competing groups "
                        "(head analogue; requires --topk 0)")
    p.add_argument("--group-fn", choices=["top1", "softmax"], default="top1",
                   help="per-group competition (with --groups)")
    p.add_argument("--prefixes", type=int, default=1,
                   help="nested prefix losses over P latent blocks "
                        "(matryoshka/deepsup analogue; 1 = off)")
    p.add_argument("--enc", choices=["linear", "bilinear", "gated"],
                   default="linear",
                   help="encoder form; bilinear latents admit eigenfeature "
                        "analysis")
    p.add_argument("--block", type=int, default=0,
                   help="vanilla block-sparse featurizer: blocks of B signed "
                        "latents, --topk of them active (0 = off; needs "
                        "--aux-k 0)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after N steps regardless of epochs (0 = off)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--adam-eps", type=float, default=1e-8,
                   help="Adam's eps (optax's default 1e-8). A floor in the "
                        "loss's own units: a parameter whose gradient scale "
                        "is well below it takes about that scale / eps of a "
                        "full step")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (excluded from training)")
    p.add_argument("--train-rows", type=int, default=0,
                   help="train on only the first N rows of the cache (0 = every "
                        "row before the held-out tail); the tail is unchanged")
    p.add_argument("--block-rows", type=int, default=0,
                   help="block-shuffled batches: split the training rows into "
                        "contiguous blocks of N rows, take them in random order "
                        "--buffer-blocks at a time into memory, and draw batches "
                        "from a shuffle of each buffer, so a cache larger than "
                        "memory is read sequentially (0 = a uniform permutation; "
                        "N a multiple of --b; rows past the last whole block are "
                        "not trained on)")
    p.add_argument("--buffer-blocks", type=int, default=16,
                   help="blocks per shuffle buffer (with --block-rows)")
    p.add_argument("--aux-k", type=int, default=512,
                   help="dead latents used to reconstruct the residual (0 = off)")
    p.add_argument("--resample-every", type=int, default=0,
                   help="resample dead latents every N steps toward "
                        "high-loss examples (Bricken et al. 2023, the "
                        "ReLU+L1 lineage's revival; needs --aux-k 0 and "
                        "--enc linear; pick N > --dead-steps; 0 = off)")
    p.add_argument("--dead-steps", type=int, default=1000,
                   help="steps without firing before a latent counts as dead")
    p.add_argument("--sd-pre", type=float, default=0.0,
                   help="selection noise, training only: the top k are "
                        "chosen from the pre-activations plus Gaussian noise "
                        "of this many times each sample's pre-activation RMS, "
                        "and keep their clean values (the Ontologizer's "
                        "logit noise, --sd-k with --noise-k batchnorm)")
    p.add_argument("--p-drop", type=float, default=0.0,
                   help="winner dropout, training only: each chosen latent "
                        "is replaced by the runner-up with this probability "
                        "per sample (the Ontologizer's --p-drop)")
    p.add_argument("--sd-out", type=float, default=0.0,
                   help="output noise, training only: the sum of noise on "
                        "each latent's output of this many times its batch "
                        "std per coordinate (the Ontologizer's --sd-f)")
    p.add_argument("--save-each", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="output dir (default data/out/sonar/sae/<auto name>)")
    p.add_argument("--overwrite", action="store_true",
                   help="ignore an existing state snapshot instead of resuming")
    return p.parse_args()


def run_name(cfg):
    if cfg.block:
        name = f"m{cfg.m}_bsf{cfg.block}_k{cfg.topk}"
    elif cfg.groups:
        name = f"m{cfg.m}_g{cfg.groups}{cfg.group_fn}"
    elif cfg.topk:
        name = f"m{cfg.m}_k{cfg.topk}"
    else:
        name = f"m{cfg.m}_l1{cfg.l1:g}"
    if cfg.enc == "bilinear":
        name += "_bl"
    if cfg.enc == "gated":
        name += "_gated"
    if cfg.prefixes > 1:
        name += f"_p{cfg.prefixes}"
    if cfg.adam_eps != 1e-8:
        name += f"_eps{cfg.adam_eps:g}"
    for flag, tag in (("sd_pre", "sdpre"), ("p_drop", "pdrop"), ("sd_out", "sdout")):
        if getattr(cfg, flag):
            name += f"_{tag}{getattr(cfg, flag):g}"
    if cfg.seed != 42:
        name += f"_s{cfg.seed}"  # seed-stability replicas get their own dir
    return name


def model_for(params=None, *, d=0, m=0, enc=None, topk=0, l1=0.0, groups=0,
              group_fn="top1", prefixes=1, aux_k=0, block=0,
              sd_pre=0.0, p_drop=0.0, sd_out=0.0):
    """The `ontologize` model that a flat `params.npz` dict (or, without
    one, `d`/`m`/`enc`) and the run's encode settings describe. The encoder
    form comes off the params' keys, so a checkpoint needs no flag; the
    gated form sets its own sparsity, so `topk` and `groups` are ignored
    for it, as they always were here.

    A block-sparse featurizer's block size comes off its params too, but
    the caller must pass the same `block` or this raises: a caller that
    does not know about blocks would read its `topk` as latents."""
    if params is not None:
        enc = legacy_encoder(params)
        m, d = params["W_dec"].shape
        held = legacy_block(params)
        if held != block:
            raise ValueError(
                f"these parameters are a block-sparse featurizer with "
                f"block={held}; pass block={held}" if held else
                f"block={block} was passed, but these parameters are not a "
                f"block-sparse featurizer")
    enc = enc or "linear"
    kw = dict(d=d, m=m, s_l1=l1, aux_k=aux_k, prefixes=prefixes, enc=enc,
              sd_pre=sd_pre, p_drop=p_drop, sd_out=sd_out)
    if block:
        return BlockSparse(topk=topk, block=block, **kw)
    if enc == "gated":
        return SAE(topk=0, **kw)
    if groups:
        return Grouped(topk=0, groups=groups, group_fn=group_fn, **kw)
    return (BilinearSAE if enc == "bilinear" else SAE)(topk=topk, **kw)


def model_from_cfg(cfg, params):
    """`model_for` with a training run's settings."""
    return model_for(params, topk=cfg.topk, l1=cfg.l1, groups=cfg.groups,
                     group_fn=cfg.group_fn, prefixes=cfg.prefixes,
                     aux_k=cfg.aux_k, block=cfg.block, sd_pre=cfg.sd_pre,
                     p_drop=cfg.p_drop, sd_out=cfg.sd_out)


def _apply(model, params, method, *args):
    return model.apply({"params": from_legacy(params)}, *args,
                       method=getattr(type(model), method))


def init_params(rng, d, m, x_mean, enc="linear", x_scale=1.0, block=0):
    """Field-standard initialization (`SAE.initialize`) in the flat layout:
    unit-norm decoder rows, b_dec at the data mean, tied encoder, bilinear
    factors scaled to the linear init's pre-activation scale. A gated
    checkpoint has no `b_enc`: its two paths carry their own biases. With
    `block`, `BlockSparse.initialize`: orthonormal blocks at Frobenius
    norm 1, read by the decoder's transpose times `block`."""
    return to_legacy(model_for(d=d, m=m, enc=enc, block=block).initialize(
        rng, x_mean, x_scale))


def gated_pre(params, X):
    """The gated encoder's gate and magnitude logits (`SAE.gated_pre`)."""
    return _apply(model_for(params), params, "gated_pre", X)


def preacts(params, X, block=0):
    """Raw encoder logits for any encoder form (`SAE.preacts`); for the
    gated form, the magnitude path; for a block-sparse featurizer (pass
    its `block`), the signed pre-codes."""
    return _apply(model_for(params, block=block), params, "preacts", X)


def eigenfeatures(params):
    """Each bilinear latent's closed-form top eigenvector, (m, d)
    (`ontologize.sae.eigenfeatures`)."""
    return _eigenfeatures(params["W_enc1"].T, params["W_enc2"].T,
                          params["W_dec"])


def activate(pre, topk, groups=0, group_fn="top1", block=0):
    """Raw encoder logits -> latent activations for every SAE variant;
    `topk` counts blocks when `block` is set."""
    if block:
        return block_topk(pre, topk, block)
    if groups:
        return (group_softmax if group_fn == "softmax" else group_top1)(
            pre, groups)
    return topk_relu(pre, topk)


def encode(params, X, topk, groups=0, group_fn="top1", block=0):
    return _apply(model_for(params, topk=topk, groups=groups,
                            group_fn=group_fn, block=block),
                  params, "encode", X)


def decode(params, z):
    """The decoder reads every model's code alike, so a block-sparse
    checkpoint decodes without its caller passing `block`."""
    return _apply(model_for(params, block=legacy_block(params)), params,
                  "decode", z)


def gated_loss(params, X, w_sqrt, l1):
    """The gated objective (`SAE.gated_loss`): reconstruction, the gate's
    L1 and the frozen-decoder auxiliary reconstruction."""
    return _apply(model_for(params, l1=l1), params, "gated_loss", X, w_sqrt)


def make_step(tx, cfg):
    """One optimizer step on the flat layout, with the model's loss
    (`SAE.loss`) and decoder renormalization (`SAE.renorm`). The model is
    built from the params inside the trace, where their shapes are fixed.
    `rng` reaches the loss only when a training noise is set; the other
    models' losses take none."""
    noisy = bool(cfg.sd_pre or cfg.p_drop or cfg.sd_out)

    def loss_fn(params, X, w_sqrt, dead, rng):
        model = model_from_cfg(cfg, params)
        args = (X, w_sqrt, dead, rng) if noisy else (X, w_sqrt, dead)
        return _apply(model, params, "loss", *args)

    @jax.jit
    def step(params, opt_state, X, w_sqrt, dead, rng):
        (loss, (mse, fired)), g = jax.value_and_grad(
            loss_fn, has_aux=True)(params, X, w_sqrt, dead, rng)
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = to_legacy(model_from_cfg(cfg, params).renorm(
            from_legacy(params)))
        return params, opt_state, loss, mse, fired

    return step


def make_eval(topk, groups=0, group_fn="top1", block=0):
    @jax.jit
    def eval_batch(params, X, w_sqrt):
        z = encode(params, X, topk, groups, group_fn, block)
        err = ((decode(params, z) - X) * w_sqrt) ** 2
        # l1 is logged unpenalized in every mode, so its trajectory
        # separates penalty-driven sparsity (--l1) from what the task and
        # the activation rule produce on their own. A block code is
        # signed, so a latent fires when it is nonzero
        return (err.mean(0), (z != 0.0).sum(-1).mean(),
                jnp.abs(z).sum(-1).mean(), (z != 0.0).sum(0))

    return eval_batch


def evaluate(eval_batch, params, X_eval, w_sqrt, base_w, b):
    """Returns (fvu, l0, l1, freq): whitened FVU, mean per-sample L0 and
    |z|_1, and each latent's firing frequency over the evaluated rows."""
    b = min(b, len(X_eval))
    mse = np.zeros(X_eval.shape[1])
    l0 = 0.0
    l1 = 0.0
    fires = 0
    nb = 0
    for i in range(0, len(X_eval) - b + 1, b):
        e, l, a, f = eval_batch(params, jnp.asarray(X_eval[i:i + b]), w_sqrt)
        mse += np.asarray(e)
        l0 += float(l)
        l1 += float(a)
        fires = fires + np.asarray(f)
        nb += 1
    return (mse / nb).mean() / base_w, l0 / nb, l1 / nb, fires / (nb * b)


def usage_stats(freq, hi=0.1):
    """Usage-balance summaries from per-latent firing frequencies:
    the effective number of latents in use (exp of the entropy of the
    normalized firing-frequency distribution -- m when usage is uniform,
    1 when one latent does everything) and the count of dense latents
    (firing on more than `hi` of eval samples; dense features score
    poorly under autointerp)."""
    f = np.asarray(freq, np.float64)
    s = f.sum()
    if s <= 0:
        return 0.0, 0
    p = f[f > 0] / s
    return float(np.exp(-(p * np.log(p)).sum())), int((f > hi).sum())


def wdec_cos(params, chunk=1024):
    """(max, mean) off-diagonal |cosine| between decoder rows -- the
    training-time duplication/splitting signal headcoh.py and
    splitting.py measure post hoc. Rows are unit-normalized first so the
    stat is comparable across renorm modes; chunked so the (m, m)
    similarity matrix never fully materializes."""
    W = params["W_dec"]
    Wn = W / (jnp.linalg.norm(W, axis=-1, keepdims=True) + 1e-9)
    m = Wn.shape[0]
    mx, tot = 0.0, 0.0
    for i in range(0, m, chunk):
        C = jnp.abs(Wn[i:i + chunk] @ Wn.T)
        r = jnp.arange(C.shape[0])
        C = C.at[r, r + i].set(0.0)  # drop self-similarity
        mx = max(mx, float(C.max()))
        tot += float(C.sum())
    return mx, tot / (m * (m - 1))


def save_state(path, params, opt_state, step, last_fired):
    leaves = jax.tree_util.tree_leaves((params, opt_state))
    # np.savez appends .npz to names that lack it, so the temp name must
    # keep the extension for os.replace to find the written file
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez(tmp, step=step, last_fired=last_fired,
             **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)})
    os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt the snapshot


def resample_dead(params, opt_state, X, w_sqrt, dead, rng,
                  topk=0, groups=0, group_fn="top1"):
    """Neuron resampling (`SAE.resample`, Bricken et al. 2023) on the flat
    layout, with Adam's moments zeroed for every touched entry. Linear and
    gated forms only."""
    model = model_for(params, topk=topk, groups=groups, group_fn=group_fn)
    adam = opt_state[0]
    tree, (mu, nu) = model.resample(
        from_legacy(params), (from_legacy(adam.mu), from_legacy(adam.nu)),
        jnp.asarray(X), w_sqrt, np.asarray(dead), rng)
    adam = adam._replace(mu=to_legacy(mu), nu=to_legacy(nu))
    return to_legacy(tree), (adam,) + tuple(opt_state[1:])


def load_state(path, params, opt_state):
    dat = np.load(path)
    template = (params, opt_state)
    treedef = jax.tree_util.tree_structure(template)
    n = len(jax.tree_util.tree_leaves(template))
    leaves = [jnp.asarray(dat[f"leaf_{i}"]) for i in range(n)]
    params, opt_state = jax.tree_util.tree_unflatten(treedef, leaves)
    return params, opt_state, int(dat["step"]), dat["last_fired"]


def splits(n: int, eval_rows: int, train_rows: int) -> Tuple[int, int]:
    """The number of training rows and the first row of the held-out tail.
    The tail is the last `eval_rows` rows; training takes the rows before
    it, only the first `train_rows` of them when that is set."""
    head = n - eval_rows
    return (min(head, train_rows) if train_rows else head), head


def uniform_batches(mm, n_train: int, b: int, epochs: int, rng, step: int,
                    total: int) -> Iterator[np.ndarray]:
    """Batches from a fresh uniform permutation of the training rows each
    epoch, from batch `step` (a resumed run skips those already consumed;
    the permutations are seed-deterministic, so this replays the schedule)
    up to batch `total`."""
    if step >= total:
        return
    per_epoch = n_train // b
    for epoch in range(epochs):
        if epoch * per_epoch >= total:
            return
        perm = rng.permutation(n_train)
        for i in range(0, n_train - b + 1, b):
            j = epoch * per_epoch + i // b
            if j >= total:
                return
            if j < step:
                continue
            idx = np.sort(perm[i:i + b])  # sorted for memmap locality
            yield np.asarray(mm[idx], dtype=np.float32)


def block_schedule(rng, n: int, block_rows: int, buffer_blocks: int
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """One epoch of block-shuffled sampling over the first n rows: the
    starts of the whole blocks of `block_rows` rows in random order, taken
    `buffer_blocks` at a time, each group with a permutation of positions in
    its buffer (which holds the group's blocks end to end). Rows past the
    last whole block are left out."""
    order = rng.permutation(n // block_rows) * block_rows
    for g in range(0, len(order), buffer_blocks):
        starts = order[g:g + buffer_blocks]
        yield starts, rng.permutation(len(starts) * block_rows)


def block_groups(rng, n_train: int, b: int, block_rows: int, buffer_blocks: int,
                 epochs: int, step: int, total: int
                 ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """The block-shuffled schedule between batches `step` and `total`: for
    each buffer with a batch left to run, the index of its first batch, the
    starts of its blocks and the permutation of its rows. Every epoch's
    groups are drawn, so a resumed run replays the same schedule."""
    per_epoch = (n_train // block_rows) * (block_rows // b)
    for epoch in range(epochs):
        j = epoch * per_epoch
        if j >= total:
            return
        for starts, perm in block_schedule(rng, n_train, block_rows, buffer_blocks):
            if j >= total:
                return
            nb = len(perm) // b
            if j + nb > step:
                yield j, starts, perm
            j += nb


def block_reader(path: str, block_rows: int) -> Callable[[np.ndarray], np.ndarray]:
    """Reads a buffer of whole blocks of a float32 .npy into memory with
    pread, dropping each block from the page cache once read, since a pass
    over a cache larger than memory would otherwise evict other files."""
    X = np.load(path, mmap_mode="r")
    if X.dtype != np.float32 or X.ndim != 2 or not X.flags.c_contiguous:
        raise ValueError(f"{path}: expected a C-ordered 2-d float32 array")
    d, off = X.shape[1], X.offset
    del X

    def read(starts: np.ndarray) -> np.ndarray:
        buf = np.empty((len(starts) * block_rows, d), np.float32)
        nb = block_rows * d * 4
        fd = os.open(path, os.O_RDONLY)
        try:
            for i, s in enumerate(starts):
                at = off + int(s) * d * 4
                view = memoryview(buf[i * block_rows:(i + 1) * block_rows]).cast("B")
                if os.preadv(fd, [view], at) != nb:
                    raise IOError(f"{path}: short read at row {int(s)}")
                os.posix_fadvise(fd, at, nb, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return buf
    return read


def block_batches(groups: Iterator[Tuple[int, np.ndarray, np.ndarray]],
                  read: Callable[[np.ndarray], np.ndarray], b: int, step: int,
                  total: int) -> Iterator[np.ndarray]:
    """The batches of `groups` from batch `step` up to batch `total`, each
    buffer read in a background thread while the one before it trains."""
    with ThreadPoolExecutor(1) as pool:
        it = iter(groups)
        nxt = next(it, None)
        fut = pool.submit(read, nxt[1]) if nxt is not None else None
        while nxt is not None:
            (j, _, perm), cur = nxt, fut
            nxt = next(it, None)
            fut = pool.submit(read, nxt[1]) if nxt is not None else None
            buf = cur.result()
            for k in range(len(perm) // b):
                if j + k >= total:
                    return
                if j + k >= step:
                    yield buf[perm[k * b:(k + 1) * b]]


def train(cfg):
    if cfg.groups:
        assert not cfg.topk, "--groups needs --topk 0 (mutually exclusive)"
        assert cfg.m % cfg.groups == 0, "--m must be a multiple of --groups"
    if cfg.prefixes > 1:
        assert cfg.m % cfg.prefixes == 0, "--m must be a multiple of --prefixes"
    if cfg.resample_every:
        assert not cfg.aux_k, ("--resample-every needs --aux-k 0 "
                               "(one revival mechanism at a time)")
        assert cfg.enc != "bilinear", \
            "--resample-every cannot serve --enc bilinear: a factor pair " \
            "has no defined resample direction"
    if cfg.enc == "gated":
        assert not cfg.topk and not cfg.groups, \
            "--enc gated sets its own sparsity; needs --topk 0 and no --groups"
        assert cfg.l1 > 0, "--enc gated needs --l1 (the gate's L1 coefficient)"
        assert cfg.prefixes == 1, "--enc gated does not implement --prefixes"
        # the paper's revival mechanism is the auxiliary reconstruction in
        # `gated_loss`, which is always on; aux_k would give the MAGNITUDE
        # path gradient for latents whose gate is shut, which cannot reopen
        # a gate and would only report deadness as fixed
        assert not cfg.aux_k, "--enc gated needs --aux-k 0 (see gated_loss)"
    if cfg.block:
        assert cfg.m % cfg.block == 0, "--m must be a multiple of --block"
        assert cfg.topk >= 1 and 2 * cfg.topk <= cfg.m // cfg.block, \
            "--block needs --topk >= 1 blocks, and room for as many " \
            "runner-ups among the m/block blocks"
        assert not cfg.groups and cfg.enc == "linear" and cfg.prefixes == 1, \
            "--block takes the linear encoder, without --groups or --prefixes"
        assert not cfg.l1, \
            "--block's only sparsity is its top-k over blocks; needs --l1 0"
        # the paper's revival is the runner-up term in BlockSparse.loss
        assert not cfg.aux_k, "--block needs --aux-k 0 (see BlockSparse.loss)"
        assert not cfg.resample_every, "--block has no resampling"
    if cfg.sd_pre or cfg.p_drop or cfg.sd_out:
        assert cfg.topk and not cfg.groups and not cfg.block \
            and cfg.enc == "linear" and cfg.prefixes == 1, \
            "--sd-pre, --p-drop and --sd-out need a flat linear top-k SAE"

    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/sae") / run_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.npz"
    # architecture record so downstream consumers (pareto.py, autointerp.py,
    # headstruct.py) can rebuild the encode rule without parsing run names
    (out / "meta.json").write_text(json.dumps(
        {"m": cfg.m, "topk": cfg.topk, "l1": cfg.l1, "groups": cfg.groups,
         "group_fn": cfg.group_fn, "prefixes": cfg.prefixes,
         "enc": cfg.enc, "resample_every": cfg.resample_every,
         "block": cfg.block, "adam_eps": cfg.adam_eps,
         "sd_pre": cfg.sd_pre, "p_drop": cfg.p_drop, "sd_out": cfg.sd_out}))

    mm = np.load(cfg.cache, mmap_mode="r")
    n_train, n_head = splits(mm.shape[0], cfg.eval_rows, cfg.train_rows)
    d = mm.shape[1]
    X_eval = np.asarray(mm[n_head:], dtype=np.float32)

    w = np.load(cfg.mse_weights) if cfg.mse_weights else np.ones(d)
    w_sqrt = jnp.asarray(np.sqrt(w), jnp.float32)
    base_w = (X_eval.var(0) * w).mean()  # whitened constant-predictor MSE

    x_slice = np.asarray(mm[:min(n_train, 1 << 18)], dtype=np.float32)
    x_mean = x_slice.mean(0)
    x_scale = float(np.linalg.norm(x_slice - x_mean, axis=1).mean())
    # fixed held-in slice, same size as the tail, with its own
    # constant-predictor baseline: fvu_train vs fvu_eval reads as the
    # generalization gap
    X_tr = x_slice[:cfg.eval_rows]
    base_tr = (X_tr.var(0) * w).mean()
    params = init_params(jax.random.PRNGKey(cfg.seed), d, cfg.m, x_mean,
                         cfg.enc, x_scale, cfg.block)
    tx = optax.adam(cfg.lr, eps=cfg.adam_eps)
    opt_state = tx.init(params)
    step_fn = make_step(tx, cfg)
    eval_fn = make_eval(cfg.topk, cfg.groups, cfg.group_fn, cfg.block)
    # the training noises draw from their own stream, keyed by step, so a
    # noised run shares the stock run's initialization and batch order and
    # a resume draws what the uninterrupted run would have
    noise_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 1)

    step = 0
    last_fired = np.zeros(cfg.m, np.int64)
    if state_path.exists() and not cfg.overwrite:
        params, opt_state, step, last_fired = load_state(
            state_path, params, opt_state)
        print(f"resumed {state_path} at step {step}")

    loss_path = out / "loss.csv"
    prev_dead = step - last_fired > cfg.dead_steps
    rng = np.random.default_rng(cfg.seed)
    if cfg.block_rows:
        assert cfg.block_rows % cfg.b == 0, "--block-rows must be a multiple of --b"
        assert cfg.block_rows <= n_train, "--block-rows exceeds the training rows"
        per_epoch = (n_train // cfg.block_rows) * (cfg.block_rows // cfg.b)
    else:
        per_epoch = n_train // cfg.b
    total = cfg.epochs * per_epoch
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    if cfg.block_rows:
        batches = block_batches(
            block_groups(rng, n_train, cfg.b, cfg.block_rows, cfg.buffer_blocks,
                         cfg.epochs, step, total),
            block_reader(cfg.cache, cfg.block_rows), cfg.b, step, total)
    else:
        batches = uniform_batches(mm, n_train, cfg.b, cfg.epochs, rng, step, total)

    with tqdm(total=total, initial=step, desc=run_name(cfg)) as pbar:
        for batch in batches:
            X = jnp.asarray(batch)
            dead = jnp.asarray(step - last_fired > cfg.dead_steps)
            params, opt_state, L, mse, fired = step_fn(
                params, opt_state, X, w_sqrt, dead,
                jax.random.fold_in(noise_key, step))
            last_fired[np.asarray(fired)] = step
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=float(L), mse_w=float(mse))

            if step % cfg.save_each == 0:
                fvu, l0, l1, freq = evaluate(eval_fn, params, X_eval,
                                             w_sqrt, base_w, cfg.b)
                fvu_tr = evaluate(eval_fn, params, X_tr, w_sqrt,
                                  base_tr, cfg.b)[0]
                dead_now = step - last_fired > cfg.dead_steps
                revived = int((prev_dead & ~dead_now).sum())
                new_dead = int((~prev_dead & dead_now).sum())
                prev_dead = dead_now
                use_ent, hifreq = usage_stats(freq)
                maxcos, meancos = wdec_cos(params)
                with open(loss_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [step, float(L), float(mse), fvu, l0, l1,
                         int(dead_now.sum()), revived, new_dead,
                         use_ent, hifreq, maxcos, meancos, fvu_tr])
                save_state(state_path, params, opt_state, step, last_fired)

            # after the eval/save block, so logged numbers never show
            # freshly reset latents; never on the final step, which
            # would send untrained directions into the final eval
            if (cfg.resample_every and step % cfg.resample_every == 0
                    and step < total):
                dead_now = step - last_fired > cfg.dead_steps
                if dead_now.any():
                    params, opt_state = resample_dead(
                        params, opt_state, X, w_sqrt, dead_now,
                        np.random.default_rng(cfg.seed + step),
                        cfg.topk, cfg.groups, cfg.group_fn)
                    last_fired[dead_now] = step

    fvu, l0, l1, _ = evaluate(eval_fn, params, X_eval, w_sqrt, base_w, cfg.b)
    n_dead = int((step - last_fired > cfg.dead_steps).sum())
    save_state(state_path, params, opt_state, step, last_fired)
    np.savez(out / "params.npz", **{k: np.asarray(v) for k, v in params.items()})
    print(f"\n{run_name(cfg)} step {step}: FVU_w {fvu:.4f}  L0 {l0:.1f}  "
          f"L1 {l1:.2f}  dead {n_dead}/{cfg.m}")
    return fvu, l0, n_dead


if __name__ == "__main__":
    train(parse_args())
