"""Sparse probing of language identity (SAEBench-style k-sparse probing).

The embedding cache's .langs.npy sidecar labels every row with its mC4
language, giving ~86 free binary concepts. For each language this script
selects the top latents by two-sample t-statistic on training rows, then
scores one-vs-rest probes on held-out tail rows:

  k=1      the single best latent, threshold fit on train (does one
           latent = one language?)
  k>1      logistic probe over the top-k latents
  dense    logistic probe on the raw 1024-d embedding -- the ceiling any
           latent basis is trying to reach

Macro-F1 over languages is the headline number; per-language detail goes
to <out>/langs.csv. Works on sae.py runs and Ontologizer checkpoints
alike (autointerp activation convention), so "is language a latent-level
concept" is answerable per rung.

--only-langs keeps only the rows of the listed sidecar languages, in
training and test alike, so each probe separates its language from the
others listed and from nothing else. The training rows are still the
first --train-rows of the cache, so a short list wants a larger window.

  uv run python langprobe.py --model data/out/sonar/sae_conv/m5120_k32/params.npz
  uv run python langprobe.py --model data/out/sonar/multilingual/resid_nc
  uv run python langprobe.py --model data/out/sonar/multilingual/resid_nc \\
      --only-langs en fr es de zh --train-rows 262144 --test-rows 32768

Probe training rows come from the cache head, test rows from the tail
(the sae.py eval split).
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Int, Shaped
from pathlib import Path
from typing import Sequence

from splitting import load_model


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="sae params.npz or onto checkpoint dir")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--langs", default=None,
                   help="default: <cache>.langs.npy sidecar")
    p.add_argument("--train-rows", type=int, default=65536)
    p.add_argument("--test-rows", type=int, default=16384)
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--min-count", type=int, default=200,
                   help="skip languages with fewer training rows")
    p.add_argument("--only-langs", nargs="+", default=None,
                   help="keep only the rows of these sidecar languages")
    p.add_argument("--no-dense", action="store_true",
                   help="skip the raw-embedding ceiling probes")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default data/out/sonar/langprobe/<model name>")
    return p.parse_args()


def t_stats(S, S2, cnt, N):
    """Two-sample t per (language, latent) from per-language activation
    sums/sumsq/counts. Positive t = latent fires more on the language."""
    cnt = cnt[:, None].astype(np.float64)
    tS, tS2 = S.sum(0, keepdims=True), S2.sum(0, keepdims=True)
    mu1 = S / np.maximum(cnt, 1)
    mu0 = (tS - S) / np.maximum(N - cnt, 1)
    v1 = np.maximum(S2 / np.maximum(cnt, 1) - mu1 ** 2, 0)
    v0 = np.maximum((tS2 - S2) / np.maximum(N - cnt, 1) - mu0 ** 2, 0)
    se = np.sqrt(v1 / np.maximum(cnt, 1) + v0 / np.maximum(N - cnt, 1))
    return (mu1 - mu0) / (se + 1e-12)


def best_f1_threshold(scores, y):
    """Max F1 over all thresholds of the form score >= t, and that t."""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(y[order])
    fp = np.cumsum(1 - y[order])
    f1 = 2 * tp / np.maximum(tp + fp + y.sum(), 1e-12)
    i = int(np.argmax(f1))
    return float(f1[i]), float(scores[order][i])


def f1_at(scores, y, thr):
    pred = scores >= thr
    tp = float((pred & (y > 0)).sum())
    return 2 * tp / max(pred.sum() + y.sum(), 1e-12)


def fit_logistic(X, y, steps=300, lr=0.1, seed=0):
    """Full-batch logistic regression (standardized inputs, class-balanced
    loss); returns a scoring function. Small and dependency-free."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = jnp.asarray((X - mu) / sd)
    yj = jnp.asarray(y.astype(np.float32))
    pos = float(y.mean())
    wgt = jnp.where(yj > 0, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    w = jnp.zeros(X.shape[1])
    b = jnp.array(0.0)

    def loss(wb):
        w, b = wb
        z = Xn @ w + b
        return (wgt * (jnp.logaddexp(0.0, z) - yj * z)).mean()

    grad = jax.jit(jax.grad(loss))
    m = (jnp.zeros_like(w), jnp.array(0.0))
    v = (jnp.zeros_like(w), jnp.array(0.0))
    wb = (w, b)
    for i in range(steps):  # plain adam
        g = grad(wb)
        m = jax.tree_util.tree_map(lambda a, b_: 0.9 * a + 0.1 * b_, m, g)
        v = jax.tree_util.tree_map(lambda a, b_: 0.999 * a + 0.001 * b_ ** 2,
                                   v, g)
        t = i + 1
        wb = jax.tree_util.tree_map(
            lambda p, mi, vi: p - lr * (mi / (1 - 0.9 ** t))
            / (jnp.sqrt(vi / (1 - 0.999 ** t)) + 1e-8), wb, m, v)

    def score(Xq):
        return np.asarray(jnp.asarray((Xq - mu) / sd) @ wb[0] + wb[1])

    return score


def only_rows(langs: Shaped[np.ndarray, "n"], rows: Int[np.ndarray, "r"],
              keep: Sequence[str]) -> Int[np.ndarray, "s"]:
    """The rows whose sidecar language is in `keep`, in their order."""
    labels = np.asarray(langs[rows])
    missing = set(keep) - set(labels.tolist())
    if missing:
        raise SystemExit(f"--only-langs: no rows of {sorted(missing)}")
    return rows[np.isin(labels, list(keep))]


def collect(acts_fn, mm, rows, b, cols):
    """Activation columns `cols` for the given row slice."""
    out = np.empty((len(rows), len(cols)), np.float32)
    for i in range(0, len(rows), b):
        X = jnp.asarray(np.asarray(mm[rows[i:i + b]], dtype=np.float32))
        out[i:i + b] = np.asarray(acts_fn(X))[:, cols]
    return out


def main():
    cfg = parse_args()
    acts_fn, F, _, _, name = load_model(cfg.model, cfg)
    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/langprobe") / name
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    langs_path = cfg.langs or str(Path(cfg.cache).with_name(
        Path(cfg.cache).name.replace(".npy", ".langs.npy")))
    all_langs = np.load(langs_path, mmap_mode="r")
    tr_rows = np.arange(cfg.train_rows // cfg.b * cfg.b)
    te_rows = np.arange(len(mm) - cfg.test_rows, len(mm))
    if cfg.only_langs:
        tr_rows = only_rows(all_langs, tr_rows, cfg.only_langs)
        te_rows = only_rows(all_langs, te_rows, cfg.only_langs)
    n_tr = len(tr_rows)
    y_tr_lang = np.asarray(all_langs[tr_rows])
    y_te_lang = np.asarray(all_langs[te_rows])

    codes, y_tr = np.unique(y_tr_lang, return_inverse=True)
    L = len(codes)
    onehot = jnp.asarray(np.eye(L, dtype=np.float32)[y_tr])
    print(f"{name}: {F} features, {L} languages, "
          f"{n_tr} train / {len(te_rows)} test rows")

    # pass 1: per-language activation moments -> t-statistics
    S = np.zeros((L, F))
    S2 = np.zeros((L, F))
    for i in range(0, n_tr, cfg.b):
        A = acts_fn(jnp.asarray(np.asarray(mm[tr_rows[i:i + cfg.b]],
                                           dtype=np.float32)))
        O = onehot[i:i + cfg.b]
        S += np.asarray(O.T @ A, np.float64)
        S2 += np.asarray(O.T @ (A * A), np.float64)
    cnt = np.bincount(y_tr, minlength=L)
    T = t_stats(S, S2, cnt, n_tr)

    keep = [i for i in range(L) if cnt[i] >= cfg.min_count
            and (y_te_lang == codes[i]).sum() >= 20]
    kmax = max(cfg.ks)
    top = np.argsort(-np.abs(T), axis=1)[:, :kmax]     # (L, kmax)

    # pass 2: gather the union of selected columns
    cols = np.unique(top[keep].ravel())
    col_of = {c: i for i, c in enumerate(cols)}
    A_tr = collect(acts_fn, mm, tr_rows, cfg.b, cols)
    A_te = collect(acts_fn, mm, te_rows, cfg.b, cols)
    X_tr = np.asarray(mm[tr_rows], dtype=np.float32) if not cfg.no_dense else None
    X_te = np.asarray(mm[te_rows], dtype=np.float32) if not cfg.no_dense else None

    rows_out = []
    for li in keep:
        ytr = (y_tr_lang == codes[li]).astype(np.float32)
        yte = (y_te_lang == codes[li]).astype(np.float32)
        rec = {"lang": codes[li], "n_train": int(cnt[li]),
               "n_test": int(yte.sum()), "latent": int(top[li, 0])}
        for k in cfg.ks:
            feats = [col_of[c] for c in top[li, :k]]
            if k == 1:
                sgn = np.sign(T[li, top[li, 0]]) or 1.0
                s_tr, s_te = sgn * A_tr[:, feats[0]], sgn * A_te[:, feats[0]]
            else:
                score = fit_logistic(A_tr[:, feats], ytr, seed=cfg.seed)
                s_tr, s_te = score(A_tr[:, feats]), score(A_te[:, feats])
            _, thr = best_f1_threshold(s_tr, ytr)
            rec[f"f1@{k}"] = f1_at(s_te, yte, thr)
        if not cfg.no_dense:
            score = fit_logistic(X_tr, ytr, seed=cfg.seed)
            _, thr = best_f1_threshold(score(X_tr), ytr)
            rec["f1@dense"] = f1_at(score(X_te), yte, thr)
        rows_out.append(rec)
        print(f"{rec['lang']:>8} " + " ".join(
            f"f1@{k}={rec[f'f1@{k}']:.3f}" for k in cfg.ks)
            + (f" dense={rec['f1@dense']:.3f}" if not cfg.no_dense else ""))

    fields = list(rows_out[0].keys())
    with open(out / "langs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nmacro means over {len(rows_out)} languages:")
    for k in fields[4:]:
        print(f"  {k}: {np.mean([r[k] for r in rows_out]):.3f}")
    (out / "meta.json").write_text(json.dumps(
        {"model": str(cfg.model), "train_rows": n_tr,
         "test_rows": len(te_rows), "ks": cfg.ks,
         "only_langs": cfg.only_langs}))
    print(f"-> {out / 'langs.csv'}")


if __name__ == "__main__":
    main()
