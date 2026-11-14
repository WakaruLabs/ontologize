"""Is a head an informationally closed level of the document process?

Rosas (2026) calls a coarse-graining Z = phi(X) of a process informationally
closed when the macro past predicts the macro future as well as the micro
past does: I(X_past; Z_future) = I(Z_past; Z_future). Here X_t is a
sentence's embedding, Z_t a head's entry for it (a function of X_t), and
the test is truncated to one step of past and future. Since Z_t is a
function of X_t, the gap is a conditional mutual information,

    gap = I(X_t; Z_t+1) - I(Z_t; Z_t+1) = I(X_t; Z_t+1 | Z_t) >= 0,

zero for a closed head. A head that keeps its entry between sentences is
slow; a closed one also leaves nothing about its next entry for the rest
of the embedding to add. Replication across seeds asks whether two runs
found the same head; closure asks, of each run's head alone, whether it is
an autonomous level.

Estimated per head on adjacent sentence pairs (`pairs.adjacent`), in bits,
as held-out cross-entropies of three predictors of Z_t+1:
  marginal  the next entry's frequencies;
  table     the k x k transition counts, given Z_t (add-1/2 smoothing);
  probe     a softmax regression on Z_t (one-hot) and the standardized
            embedding X_t, started at the table with the embedding weights
            at zero. Per head, the L2 penalty on the embedding weights, or
            the table alone, is chosen by cross-entropy on a
            document-held-out tenth of the fit pairs.
gap = CE(table) - CE(probe); the information Z_t carries is
CE(marginal) - CE(table), and X_t's is CE(marginal) - CE(probe). A head
whose embedding weights do not help on the selection pairs keeps the
table, so its gap is 0. A linear probe bounds what X_t adds from below.

The predictors are fitted on adjacent pairs from --fit-rows, rows no pair
run trained on (`train_pair.py --rows` keeps its pairs below 4M), and
scored on the adjacent pairs of the held-out tail, with a document
bootstrap for each group's mean. Groups per checkpoint: its first
--slow-heads heads of layer 0 (the pair runs' slow heads; in a control
run, the same positions) and --rest heads drawn from the others. Two groups without a model:
--lsh random-projection heads (argmax over k random directions of the
centered embedding), the coarse-grainings of the same cardinality that
Rosas's Fig. 2E compares against; and the top --slow-heads directions of
`sfa.py`'s fit (--sfa), each cut into k equal-count bins, the slowest
linear coarse-grainings of that cardinality.

    uv run python experiments/next-sentence/closure.py \\
        --ckpt data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38 \\
               data/out/sonar/next-sentence/long/doc2_sp0_ph38
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

# partition.py sets the XLA environment on import, as running it would
from partition import codes  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jaxtyping import Array, Float, Int  # noqa: E402

from ontologize.data.loaders import doc_holdout  # noqa: E402
from pairs import adjacent  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
LN2 = math.log(2.0)


def marginal_ce(zn_fit: Int[np.ndarray, "p h"], zn: Int[np.ndarray, "q h"],
                k: int, alpha: float = 0.5) -> Float[np.ndarray, "q h"]:
    """Per pair and head, -log2 of the next entry's smoothed frequency."""
    h = zn_fit.shape[1]
    cnt = np.stack([np.bincount(zn_fit[:, j], minlength=k) for j in range(h)])
    p = (cnt + alpha) / (cnt.sum(1, keepdims=True) + alpha * k)
    return -np.log2(p[np.arange(h)[None], zn])


def table(zt_fit: Int[np.ndarray, "p h"], zn_fit: Int[np.ndarray, "p h"],
          k: int, alpha: float = 0.5) -> Float[np.ndarray, "h k k"]:
    """Per head, p(z_t+1 | z_t) from transition counts, add-alpha smoothed."""
    h = zt_fit.shape[1]
    T = np.zeros((h, k, k))
    for j in range(h):
        np.add.at(T[j], (zt_fit[:, j], zn_fit[:, j]), 1.0)
    return (T + alpha) / (T.sum(2, keepdims=True) + alpha * k)


def table_ce(P: Float[np.ndarray, "h k k"], zt: Int[np.ndarray, "q h"],
             zn: Int[np.ndarray, "q h"]) -> Float[np.ndarray, "q h"]:
    """Per pair and head, -log2 p(z_t+1 | z_t) under the table `P`."""
    return -np.log2(P[np.arange(P.shape[0])[None], zt, zn])


def probe_logits(params: Dict[str, Array], x: Float[Array, "b d"],
                 zt: Int[Array, "b h"]) -> Float[Array, "b h k"]:
    """logits[b, j] = x_b W_j + E_j[z_t] + c_j, one softmax regression per
    head."""
    h = zt.shape[1]
    return (jnp.einsum("bd,hdk->bhk", x, params["W"])
            + params["E"][jnp.arange(h)[None], zt] + params["c"])


def probe_fit(x: Float[np.ndarray, "p d"], zt: Int[np.ndarray, "p h"],
              zn: Int[np.ndarray, "p h"], P0: Float[np.ndarray, "h k k"],
              wd: float, steps: int, batch: int, lr: float, seed: int
              ) -> Dict[str, Array]:
    """Adam on the mean cross-entropy over heads plus wd * |W|^2 (the
    embedding weights only; the entry table is left free). Starts at the
    transition table `P0`, with W = 0, so the probe begins as the table and
    only the embedding can move it away."""
    p, d = x.shape
    h, k = P0.shape[0], P0.shape[1]
    params = {"W": jnp.zeros((h, d, k)),
              "E": jnp.asarray(np.log(P0), jnp.float32),
              "c": jnp.zeros((h, k))}
    opt = optax.adam(lr)
    state = opt.init(params)

    def loss(prm, xb, ztb, znb):
        lp = jax.nn.log_softmax(probe_logits(prm, xb, ztb), -1)
        nll = -jnp.take_along_axis(lp, znb[..., None], -1).mean()
        return nll + wd * (prm["W"] ** 2).sum() / h

    @jax.jit
    def step(prm, st, xb, ztb, znb):
        g = jax.grad(loss)(prm, xb, ztb, znb)
        up, st = opt.update(g, st, prm)
        return optax.apply_updates(prm, up), st

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        b = rng.integers(0, p, min(batch, p))
        params, state = step(params, state, jnp.asarray(x[b]),
                             jnp.asarray(zt[b]), jnp.asarray(zn[b]))
    return params


def probe_ce(params: Dict[str, Array], x: Float[np.ndarray, "q d"],
             zt: Int[np.ndarray, "q h"], zn: Int[np.ndarray, "q h"],
             batch: int = 4096) -> Float[np.ndarray, "q h"]:
    """Per pair and head, -log2 of the probe's probability of z_t+1."""
    f = jax.jit(lambda prm, xb, ztb, znb: -jnp.take_along_axis(
        jax.nn.log_softmax(probe_logits(prm, xb, ztb), -1),
        znb[..., None], -1)[..., 0] / LN2)
    return np.concatenate([np.asarray(f(params, jnp.asarray(x[i:i + batch]),
                                        jnp.asarray(zt[i:i + batch]),
                                        jnp.asarray(zn[i:i + batch])))
                           for i in range(0, len(x), batch)])


def fit_select(x: Float[np.ndarray, "p d"], zt: Int[np.ndarray, "p h"],
               zn: Int[np.ndarray, "p h"], val: np.ndarray, k: int,
               wds: Sequence[float], steps: int, batch: int, lr: float,
               seed: int
               ) -> Tuple[Float[np.ndarray, "h k k"], List[Dict[str, Array]],
                          Int[np.ndarray, "h"]]:
    """The transition table and one probe per penalty, all fitted on the
    pairs outside `val`, and each head's choice among them by cross-entropy
    on `val`: index i < len(wds) is probe i, len(wds) the table alone (the
    probe with W held at 0)."""
    fit = ~val
    P0 = table(zt[fit], zn[fit], k)
    fits, vce = [], []
    for wd in wds:
        prm = probe_fit(x[fit], zt[fit], zn[fit], P0, wd, steps, batch, lr,
                        seed)
        fits.append(prm)
        vce.append(probe_ce(prm, x[val], zt[val], zn[val]).mean(0))
    vce.append(table_ce(P0, zt[val], zn[val]).mean(0))
    return P0, fits, np.argmin(np.stack(vce), 0)


def doc_bootstrap(v: Float[np.ndarray, "q"], doc: Int[np.ndarray, "q"],
                  draws: int, seed: int) -> Tuple[float, float]:
    """The mean of per-pair values `v`, and its standard error with whole
    documents resampled."""
    ud, inv = np.unique(doc, return_inverse=True)
    s = np.bincount(inv, v)
    n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ud), (draws, len(ud)))
    means = s[pick].sum(1) / n[pick].sum(1)
    return float(v.mean()), float(means.std())


def lsh_codes(X: Float[np.ndarray, "n d"], mu: Float[np.ndarray, "d"],
              W: Float[np.ndarray, "h k d"], batch: int = 8192
              ) -> Int[np.ndarray, "h n"]:
    return np.concatenate(
        [np.argmax(np.einsum("bd,hkd->bhk", X[i:i + batch] - mu, W), -1)
         for i in range(0, len(X), batch)]).T


def quantile_codes(Y: Float[np.ndarray, "n r"], edges: Float[np.ndarray, "r km1"]
                   ) -> Int[np.ndarray, "r n"]:
    """Each column of Y cut at its own k - 1 inner edges: bins 0 .. k-1."""
    return np.stack([np.searchsorted(edges[j], Y[:, j], side="right")
                     for j in range(Y.shape[1])])


def sfa_projections(X: Float[np.ndarray, "n d"], npz: str, design: str,
                    top: int) -> Float[np.ndarray, "n top"]:
    """The top `top` directions of `sfa.py`'s fit `design` (e.g. doc_plain),
    on the direction x / |x| that fit read."""
    z = np.load(npz)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    return (U - z[f"{design}_means"][0]) @ z[f"{design}_V"][:, :top]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[])
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--fit-rows", default="4000000:4400000",
                   help="lo:hi, cache rows no pair run trained on")
    p.add_argument("--holdout", type=int, default=32768)
    p.add_argument("--slow-heads", type=int, default=38)
    p.add_argument("--rest", type=int, default=38,
                   help="other heads of layer 0, drawn at random")
    p.add_argument("--lsh", type=int, default=38)
    p.add_argument("--k", type=int, default=32,
                   help="bins of the random-projection and SFA heads")
    p.add_argument("--sfa", default="data/out/sonar/next-sentence/sfa/sfa_dir.npz",
                   help="sfa.py's fit; its top --slow-heads directions of "
                        "--sfa-design, each cut into --k equal-count bins, "
                        "form a group ('' to skip)")
    p.add_argument("--sfa-design", default="doc_plain")
    p.add_argument("--wd", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 1e-1])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--fit-batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/next-sentence/closure")
    cfg = p.parse_args()

    cache = Path(cfg.cache)
    arr = np.load(cache, mmap_mode="r")
    n = len(arr)
    index = np.load(cache.with_name(cache.stem + ".index.npy"), mmap_mode="r")
    lo, hi = (int(v) for v in cfg.fit_rows.split(":"))
    tail = doc_holdout(cache, cfg.holdout)
    if hi > n - tail:
        raise SystemExit("--fit-rows reaches into the held-out tail")
    fi = adjacent(index, lo, hi)                     # pair (i, i + 1)
    ti = adjacent(index, n - tail, n)
    Xf = np.asarray(arr[lo:hi], np.float32)
    Xt = np.asarray(arr[n - tail:], np.float32)
    f0, t0 = fi - lo, ti - (n - tail)
    fdoc = np.asarray(index[fi, 0], np.int64)
    tdoc = np.asarray(index[ti, 0], np.int64)
    rng = np.random.default_rng(cfg.seed)
    udoc = np.unique(fdoc)
    val = np.isin(fdoc, rng.choice(udoc, int(len(udoc) * cfg.val_frac),
                                   replace=False))
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    xf = (Xf[f0] - mu) / sd
    xt = (Xt[t0] - mu) / sd
    print(f"fit: rows {lo}-{hi}, {len(fi)} adjacent pairs from "
          f"{len(udoc)} documents ({val.sum()} for selection); test: the "
          f"{tail}-row tail, {len(ti)} pairs from {len(np.unique(tdoc))} "
          f"documents")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = []                                      # (run, group, codes)
    if cfg.lsh:
        W = rng.standard_normal((cfg.lsh, cfg.k, Xf.shape[1])
                                ).astype(np.float32)
        W /= np.linalg.norm(W, axis=-1, keepdims=True)
        groups.append(("random-projection", "lsh", cfg.k,
                       lsh_codes(Xf, mu, W), lsh_codes(Xt, mu, W)))
    if cfg.sfa:
        Yf = sfa_projections(Xf, cfg.sfa, cfg.sfa_design, cfg.slow_heads)
        Yt = sfa_projections(Xt, cfg.sfa, cfg.sfa_design, cfg.slow_heads)
        edges = np.quantile(Yf, np.arange(1, cfg.k) / cfg.k, axis=0).T
        groups.append((f"sfa {cfg.sfa_design}", "sfa", cfg.k,
                       quantile_codes(Yf, edges), quantile_codes(Yt, edges)))
    for ckpt in cfg.ckpt:
        Af, layer, k, step = codes(ckpt, cfg.step, cfg.temperature, Xf,
                                   cfg.batch)
        At, _, _, _ = codes(ckpt, step, cfg.temperature, Xt, cfg.batch)
        l0 = np.flatnonzero(layer == 0)
        slow = l0[:cfg.slow_heads]
        rest = np.sort(rng.choice(l0[cfg.slow_heads:], cfg.rest,
                                  replace=False))
        name = f"{Path(ckpt).name}@{step}"
        groups.append((name, "slow", k, Af[slow], At[slow]))
        groups.append((name, "rest", k, Af[rest], At[rest]))

    results = []
    for name, group, k, Af, At in groups:
        zt_f, zn_f = Af[:, f0].T, Af[:, f0 + 1].T
        zt_t, zn_t = At[:, t0].T, At[:, t0 + 1].T
        ce_m = marginal_ce(zn_f[~val], zn_t, k)
        P0, fits, best = fit_select(xf, zt_f, zn_f, val, k, cfg.wd,
                                    cfg.steps, cfg.fit_batch, cfg.lr,
                                    cfg.seed)
        ce_t = table_ce(P0, zt_t, zn_t)
        ce_all = np.stack([probe_ce(prm, xt, zt_t, zn_t) for prm in fits]
                          + [ce_t])
        ce_p = ce_all[best, :, np.arange(len(best))].T          # (q, h)
        gap = ce_t - ce_p
        info_z = ce_m - ce_t
        info_x = ce_m - ce_p
        keep = (zt_t == zn_t).mean(0)
        choice = list(cfg.wd) + ["table"]
        heads = []
        for j in range(gap.shape[1]):
            heads.append({"head": j, "keep": float(keep[j]),
                          "H_next": float(ce_m[:, j].mean()),
                          "info_z": float(info_z[:, j].mean()),
                          "info_x": float(info_x[:, j].mean()),
                          "gap": float(gap[:, j].mean()),
                          "chosen": choice[best[j]]})
        row = {"run": name, "group": group, "heads": len(heads)}
        for key, v in (("gap", gap), ("info_z", info_z), ("info_x", info_x),
                       ("H_next", ce_m)):
            m, se = doc_bootstrap(v.mean(1), tdoc, cfg.bootstrap, cfg.seed)
            row[key], row[f"{key}_se"] = m, se
        row["keep"] = float(keep.mean())
        row["table_chosen"] = int((best == len(cfg.wd)).sum())
        # the share of X_t's information that Z_t misses, where there is
        # information to share
        g = np.array([hd["gap"] for hd in heads])
        ix = np.array([hd["info_x"] for hd in heads])
        row["gap_share"] = (float(g.sum() / ix.sum()) if ix.sum() > 0.01
                            else None)
        ok = ix > 0.01
        row["gap_share_heads"] = ([float(x) for x in np.quantile(
            g[ok] / ix[ok], [0.1, 0.5, 0.9])] if ok.any() else None)
        row["per_head"] = heads
        results.append(row)
        share = (f"{row['gap_share']:.1%} of x_t's; per head 10/50/90% "
                 + ", ".join(f"{x:.1%}" for x in row["gap_share_heads"])
                 if row["gap_share"] is not None else "no information to share")
        print(f"\n{name} {group} ({len(heads)} heads): keep "
              f"{row['keep']:.3f}, H(next) {row['H_next']:.3f} bits; "
              f"I(z_t; z_t+1) >= {row['info_z']:.4f}, "
              f"I(x_t; z_t+1) >= {row['info_x']:.4f}; gap "
              f"{row['gap']:.4f} +- {row['gap_se']:.4f} bits ({share}); "
              f"table chosen for {row['table_chosen']} heads")

    print(f"\n{'run':>34} {'group':>5} {'keep':>6} {'H next':>7} "
          f"{'I(z)':>7} {'I(x)':>7} {'gap':>7} {'+-':>6} {'share':>6}")
    for r in results:
        share = (f"{r['gap_share']:6.1%}" if r["gap_share"] is not None
                 else f"{'-':>6}")
        print(f"{r['run']:>34} {r['group']:>5} {r['keep']:6.3f} "
              f"{r['H_next']:7.3f} {r['info_z']:7.4f} {r['info_x']:7.4f} "
              f"{r['gap']:7.4f} {r['gap_se']:6.4f} {share}")
    (out / "summary.json").write_text(json.dumps(
        {"cache": str(cache), "fit_rows": [lo, hi], "fit_pairs": int(len(fi)),
         "test_pairs": int(len(ti)), "wd": cfg.wd, "steps": cfg.steps,
         "results": results}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
