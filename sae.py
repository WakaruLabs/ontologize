"""Standalone SAE baseline trainer for the Ontologizer comparison.

Trains a top-k sparse autoencoder (Gao et al. 2024) -- or a vanilla ReLU+L1
SAE with --l1 -- on the same precomputed SONAR embedding cache and the same
whitened MSE objective as sonar.py, so FVU_w numbers are directly
comparable. Deliberately NOT built from the ontologize layers: the abs()'d
dictionary and softmax selection make DictEnc structurally not an SAE, and
the baseline should be the field-standard architecture at its
best-practice configuration (pre-subtracted decoder bias, unit-norm
decoder rows, tied init, aux-k dead-latent revival), not a handicapped
reimplementation.

Matching the resid_nc Ontologizer (d=1024, e_dec=2048, k=32, h=32, l=5;
~23M params, l*h*k = 5120 dictionary entries, hard code = 800 bits/sample):
  matched dictionary:  --m 5120   (default)
  matched parameters:  --m 11264
  code-size sweep:     --topk 16 / 32 / 64 / 128
                       (top-k bits/sample ~= L0 * (log2 m + coeff bits),
                       so topk=32 at m=5120 sits at the 800-bit point)

Unlike sonar.py this takes CLI flags (it exists to be swept). The final
eval slice is the cache TAIL (--eval-rows, never trained on here) -- note
the Ontologizer runs trained on the full cache, so for the writeup
re-evaluate both models on freshly encoded data.

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

  uv run python sae.py                          # m=5120 topk=32
  uv run python sae.py --m 11264 --topk 64
  uv run python sae.py --topk 0 --l1 3e-4       # vanilla ReLU+L1
  uv run python sae.py --m 5120 --topk 0 --groups 160          # 160 "heads"
  uv run python sae.py --m 5120 --topk 0 --groups 160 --group-fn softmax
  uv run python sae.py --m 5120 --topk 32 --prefixes 5         # 5 "layers"
  uv run python sae.py --m 5120 --topk 32 --enc bilinear --lr 1e-4

Resumable: state is snapshotted atomically every --save-each steps and
picked up automatically on restart. loss.csv columns:
step, loss, mse_w, fvu_eval, l0_eval, dead.
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
from pathlib import Path
from tqdm import tqdm

AUX_COEF = 1 / 32  # aux-k loss scale (Gao et al. 2024)


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
    p.add_argument("--enc", choices=["linear", "bilinear"], default="linear",
                   help="encoder form; bilinear latents admit eigenfeature "
                        "analysis")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=0,
                   help="stop after N steps regardless of epochs (0 = off)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache (excluded from training)")
    p.add_argument("--aux-k", type=int, default=512,
                   help="dead latents used to reconstruct the residual (0 = off)")
    p.add_argument("--dead-steps", type=int, default=1000,
                   help="steps without firing before a latent counts as dead")
    p.add_argument("--save-each", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="output dir (default data/out/sonar/sae/<auto name>)")
    p.add_argument("--overwrite", action="store_true",
                   help="ignore an existing state snapshot instead of resuming")
    return p.parse_args()


def run_name(cfg):
    if cfg.groups:
        name = f"m{cfg.m}_g{cfg.groups}{cfg.group_fn}"
    elif cfg.topk:
        name = f"m{cfg.m}_k{cfg.topk}"
    else:
        name = f"m{cfg.m}_l1{cfg.l1:g}"
    if cfg.enc == "bilinear":
        name += "_bl"
    if cfg.prefixes > 1:
        name += f"_p{cfg.prefixes}"
    if cfg.seed != 42:
        name += f"_s{cfg.seed}"  # seed-stability replicas get their own dir
    return name


def init_params(rng, d, m, x_mean, enc="linear", x_scale=1.0):
    """Unit-norm decoder rows, b_dec at the data mean; linear encoder is
    tied-init, bilinear factors are scaled so each is unit-variance on
    centered inputs of norm x_scale (bilinear logits grow with |x|^2, so
    unscaled init runs hot on any data that isn't unit-norm)."""
    r1, r2, r3 = jax.random.split(rng, 3)
    W_dec = jax.random.normal(r1, (m, d), jnp.float32)
    W_dec = W_dec / jnp.linalg.norm(W_dec, axis=-1, keepdims=True)
    params = {"b_enc": jnp.zeros(m), "W_dec": W_dec,
              "b_dec": jnp.asarray(x_mean, jnp.float32)}
    if enc == "bilinear":
        # factors act on [x - b_dec; 1]: the constant coordinate gives the
        # quadratic form linear terms (resid_const's sign-blindness fix).
        # Scale so the product matches the tied linear init's pre-activation
        # std (|x|/sqrt(d)) -- top-k picks the extreme tail of m products,
        # so a unit-scale product start puts the reconstruction orders of
        # magnitude off the data
        s = (x_scale * d ** 0.5) ** -0.5
        params["W_enc1"] = s * jax.random.normal(r2, (d + 1, m), jnp.float32)
        params["W_enc2"] = s * jax.random.normal(r3, (d + 1, m), jnp.float32)
    else:
        params["W_enc"] = W_dec.T
    return params


def preacts(params, X):
    """Raw encoder logits for either encoder form (which form a params
    dict uses is carried by its keys, so downstream consumers need no
    flag)."""
    Xc = X - params["b_dec"]
    if "W_enc2" in params:
        Xa = jnp.concatenate([Xc, jnp.ones_like(Xc[..., :1])], -1)
        return (Xa @ params["W_enc1"]) * (Xa @ params["W_enc2"]) \
            + params["b_enc"]
    return Xc @ params["W_enc"] + params["b_enc"]


def eigenfeatures(params):
    """Input-space top eigenvector of each bilinear latent's symmetric
    form B_j = sym(w1_j w2_j^T). Rank 2, so the top-|eigenvalue|
    eigenvector is closed-form: w1/|w1| + sign(w1.w2) w2/|w2| (no eigh).
    Returns (m, d): constant coordinate dropped, unit-normalized,
    sign-aligned to the latent's decoder row (the form is even, so the
    sign is otherwise arbitrary)."""
    U = params["W_enc1"].T
    V = params["W_enc2"].T
    Un = U / (jnp.linalg.norm(U, axis=-1, keepdims=True) + 1e-9)
    Vn = V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)
    c = jnp.where((U * V).sum(-1, keepdims=True) >= 0, 1.0, -1.0)
    E = (Un + c * Vn)[:, :-1]
    E = E / (jnp.linalg.norm(E, axis=-1, keepdims=True) + 1e-9)
    s = (E * params["W_dec"]).sum(-1, keepdims=True)
    return E * jnp.where(s >= 0, 1.0, -1.0)


def activate(pre, topk, groups=0, group_fn="top1"):
    """Raw encoder logits -> latent activations for every SAE variant."""
    if groups:
        g = pre.reshape(*pre.shape[:-1], groups, -1)
        if group_fn == "softmax":
            z = jax.nn.softmax(g, -1)
        else:
            # winner per group keeps its ReLU magnitude; a group whose
            # winner is negative stays silent (unlike an Ontologizer head,
            # a group may abstain)
            top = g.max(-1, keepdims=True)
            z = jnp.where((g >= top) & (g > 0), g, 0.0)
        return z.reshape(pre.shape)
    a = jax.nn.relu(pre)
    if topk:
        thr = jax.lax.top_k(a, topk)[0][..., -1:]
        a = jnp.where(a >= thr, a, 0.0)
    return a


def encode(params, X, topk, groups=0, group_fn="top1"):
    return activate(preacts(params, X), topk, groups, group_fn)


def decode(params, z):
    return z @ params["W_dec"] + params["b_dec"]


def make_step(tx, cfg):
    topk, l1, aux_k = cfg.topk, cfg.l1, cfg.aux_k
    groups, group_fn, P = cfg.groups, cfg.group_fn, cfg.prefixes
    # softmax coefficients are bounded, so rows must carry magnitude and
    # the L1 norm-gaming loophole doesn't exist
    renorm = not (groups and group_fn == "softmax")

    def loss_fn(params, X, w_sqrt, dead):
        pre = preacts(params, X)
        z = activate(pre, topk, groups, group_fn)
        if P > 1:
            # nested prefix reconstructions (matryoshka/deepsup, joint):
            # every prefix of latent blocks must reconstruct on its own
            zb = z.reshape(z.shape[0], P, -1)
            Wb = params["W_dec"].reshape(P, -1, params["W_dec"].shape[-1])
            recons = (jnp.cumsum(jnp.einsum("bpj,pjd->pbd", zb, Wb), 0)
                      + params["b_dec"])
            mse = (((recons - X[None]) * w_sqrt) ** 2).mean()
            recon = recons[-1]
        else:
            recon = decode(params, z)
            mse = (((recon - X) * w_sqrt) ** 2).mean()
        loss = mse
        if l1:
            loss = loss + l1 * jnp.abs(z).sum(-1).mean()
        if aux_k:
            # reconstruct the (stop-gradiented) residual with the top
            # aux_k currently-dead latents so they receive gradient
            a_dead = jnp.where(dead, jax.nn.relu(pre), 0.0)
            thr_d = jax.lax.top_k(a_dead, aux_k)[0][..., -1:]
            z_aux = jnp.where(a_dead >= thr_d, a_dead, 0.0)
            resid = jax.lax.stop_gradient(X - recon)
            aux = (((z_aux @ params["W_dec"] - resid) * w_sqrt) ** 2).mean()
            loss = loss + AUX_COEF * aux * jnp.any(dead)
        fired = (z > 0.0).any(0)
        return loss, (mse, fired)

    @jax.jit
    def step(params, opt_state, X, w_sqrt, dead):
        (loss, (mse, fired)), g = jax.value_and_grad(
            loss_fn, has_aux=True)(params, X, w_sqrt, dead)
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)
        if renorm:
            # keep decoder rows unit-norm: fixes the feature scale
            # (coefficients carry magnitude) and closes the L1
            # norm-gaming loophole
            n = jnp.linalg.norm(params["W_dec"], axis=-1, keepdims=True)
            params = {**params, "W_dec": params["W_dec"] / (n + 1e-9)}
        return params, opt_state, loss, mse, fired

    return step


def make_eval(topk, groups=0, group_fn="top1"):
    @jax.jit
    def eval_batch(params, X, w_sqrt):
        z = encode(params, X, topk, groups, group_fn)
        err = ((decode(params, z) - X) * w_sqrt) ** 2
        return err.mean(0), (z > 0.0).sum(-1).mean()

    return eval_batch


def evaluate(eval_batch, params, X_eval, w_sqrt, base_w, b):
    b = min(b, len(X_eval))
    mse = np.zeros(X_eval.shape[1])
    l0 = 0.0
    nb = 0
    for i in range(0, len(X_eval) - b + 1, b):
        e, l = eval_batch(params, jnp.asarray(X_eval[i:i + b]), w_sqrt)
        mse += np.asarray(e)
        l0 += float(l)
        nb += 1
    return (mse / nb).mean() / base_w, l0 / nb


def save_state(path, params, opt_state, step, last_fired):
    leaves = jax.tree_util.tree_leaves((params, opt_state))
    # np.savez appends .npz to names that lack it, so the temp name must
    # keep the extension for os.replace to find the written file
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez(tmp, step=step, last_fired=last_fired,
             **{f"leaf_{i}": np.asarray(x) for i, x in enumerate(leaves)})
    os.replace(tmp, path)  # atomic: a crash mid-write can't corrupt the snapshot


def load_state(path, params, opt_state):
    dat = np.load(path)
    template = (params, opt_state)
    treedef = jax.tree_util.tree_structure(template)
    n = len(jax.tree_util.tree_leaves(template))
    leaves = [jnp.asarray(dat[f"leaf_{i}"]) for i in range(n)]
    params, opt_state = jax.tree_util.tree_unflatten(treedef, leaves)
    return params, opt_state, int(dat["step"]), dat["last_fired"]


def train(cfg):
    if cfg.groups:
        assert not cfg.topk, "--groups needs --topk 0 (mutually exclusive)"
        assert cfg.m % cfg.groups == 0, "--m must be a multiple of --groups"
    if cfg.prefixes > 1:
        assert cfg.m % cfg.prefixes == 0, "--m must be a multiple of --prefixes"

    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/sae") / run_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.npz"
    # architecture record so downstream consumers (pareto.py, autointerp.py,
    # headstruct.py) can rebuild the encode rule without parsing run names
    (out / "meta.json").write_text(json.dumps(
        {"m": cfg.m, "topk": cfg.topk, "l1": cfg.l1, "groups": cfg.groups,
         "group_fn": cfg.group_fn, "prefixes": cfg.prefixes,
         "enc": cfg.enc}))

    mm = np.load(cfg.cache, mmap_mode="r")
    n_train = mm.shape[0] - cfg.eval_rows
    d = mm.shape[1]
    X_eval = np.asarray(mm[n_train:], dtype=np.float32)

    w = np.load(cfg.mse_weights) if cfg.mse_weights else np.ones(d)
    w_sqrt = jnp.asarray(np.sqrt(w), jnp.float32)
    base_w = (X_eval.var(0) * w).mean()  # whitened constant-predictor MSE

    x_slice = np.asarray(mm[:min(n_train, 1 << 18)], dtype=np.float32)
    x_mean = x_slice.mean(0)
    x_scale = float(np.linalg.norm(x_slice - x_mean, axis=1).mean())
    params = init_params(jax.random.PRNGKey(cfg.seed), d, cfg.m, x_mean,
                         cfg.enc, x_scale)
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)
    step_fn = make_step(tx, cfg)
    eval_fn = make_eval(cfg.topk, cfg.groups, cfg.group_fn)

    step = 0
    last_fired = np.zeros(cfg.m, np.int64)
    if state_path.exists() and not cfg.overwrite:
        params, opt_state, step, last_fired = load_state(
            state_path, params, opt_state)
        print(f"resumed {state_path} at step {step}")

    loss_path = out / "loss.csv"
    rng = np.random.default_rng(cfg.seed)
    total = cfg.epochs * (n_train // cfg.b)
    if cfg.max_steps:
        total = min(total, cfg.max_steps)

    with tqdm(total=total, initial=step, desc=run_name(cfg)) as pbar:
        for epoch in range(cfg.epochs):
            if step >= total:
                break
            perm = rng.permutation(n_train)
            for i in range(0, n_train - cfg.b + 1, cfg.b):
                if step >= total:
                    break
                # skip batches already consumed by a resumed run (the perm
                # stream is seed-deterministic, so this replays the schedule)
                if step > epoch * (n_train // cfg.b) + i // cfg.b:
                    continue
                idx = np.sort(perm[i:i + cfg.b])  # sorted for memmap locality
                X = jnp.asarray(np.asarray(mm[idx], dtype=np.float32))
                dead = jnp.asarray(step - last_fired > cfg.dead_steps)
                params, opt_state, L, mse, fired = step_fn(
                    params, opt_state, X, w_sqrt, dead)
                last_fired[np.asarray(fired)] = step
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=float(L), mse_w=float(mse))

                if step % cfg.save_each == 0:
                    fvu, l0 = evaluate(eval_fn, params, X_eval, w_sqrt,
                                       base_w, cfg.b)
                    n_dead = int((step - last_fired > cfg.dead_steps).sum())
                    with open(loss_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [step, float(L), float(mse), fvu, l0, n_dead])
                    save_state(state_path, params, opt_state, step, last_fired)

    fvu, l0 = evaluate(eval_fn, params, X_eval, w_sqrt, base_w, cfg.b)
    n_dead = int((step - last_fired > cfg.dead_steps).sum())
    save_state(state_path, params, opt_state, step, last_fired)
    np.savez(out / "params.npz", **{k: np.asarray(v) for k, v in params.items()})
    print(f"\n{run_name(cfg)} step {step}: FVU_w {fvu:.4f}  L0 {l0:.1f}  "
          f"dead {n_dead}/{cfg.m}")
    return fvu, l0, n_dead


if __name__ == "__main__":
    train(parse_args())
