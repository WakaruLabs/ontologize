"""Cross-model feature-splitting measurement.

Quantifies feature splitting between a coarse model A and a finer model B
(canonically: same-k SAEs at two widths, e.g. m5120_k32 -> m11264_k32) via
activation containment on shared cache rows:

  child       B-latent j with P(A_i fires | B_j fires) >= tau but
              P(B_j | A_i) < tau -- j fires only inside i's territory and
              covers a fraction of it
  match       both directions >= tau -- the same feature in both models
  split       a parent with >= 2 proper children; quality checks are
              union coverage (children jointly account for the parent's
              firing set) and child disjointness (mean pairwise overlap)

Multiplicity is reported over a tau sweep (a single threshold cherry-picks),
per-parent detail is written at --tau. When both models expose decoder rows,
two geometric statistics are added: decoder-cosine confirmation is implicit
in the absorption check (B-latents geometrically near a parent that do NOT
co-fire with it -- the absorption pathology), and each model gets a
nearest-neighbour decoder-cosine distribution against a random-directions
baseline (the within-model splitting proxy: split families form tight
geometric clusters).

The match columns double as the seed-stability benchmark: run two
same-architecture SAEs differing only in --seed through this script and
"A matched"/"B matched" are the fractions of features that reproduce
across initializations (feature-level claims inherit these error bars).

Either side may be an Ontologizer checkpoint directory instead of a sae.py
params.npz: tags use the autointerp activation convention (flattened P at
temperature --temperature, fire threshold 2/k), so onto<->SAE granularity
is measured in the same frame.

  uv run python splitting.py --a data/out/sonar/sae/m5120_k32/params.npz \\
                             --b data/out/sonar/sae/m11264_k32/params.npz
  uv run python splitting.py --a data/out/sonar/multilingual/resid_nc \\
                             --b data/out/sonar/sae/m11264_k32/params.npz

Writes <out>/parents.csv (per-parent multiplicity, coverage, overlap at
--tau) and meta.json; prints the tau sweep and distributions.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path

import sae


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True,
                   help="parent (coarse) model: sae params.npz or onto dir")
    p.add_argument("--b", required=True,
                   help="child (fine) model: sae params.npz or onto dir")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=524288)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--tau", type=float, default=0.7,
                   help="containment threshold for the per-parent detail")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9], help="sweep for the summary")
    p.add_argument("--min-fires", type=int, default=50,
                   help="latents firing fewer times than this are excluded")
    p.add_argument("--step", type=int, default=0, help="onto checkpoint step")
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--out", default=None,
                   help="default data/out/sonar/splitting/<a>__<b>")
    return p.parse_args()


def load_model(path, cfg):
    """-> (jitted acts fn, n_features, fire threshold, W_dec|None, name)"""
    import jax
    p = Path(path)
    if p.suffix == ".npz":
        params = {k: jnp.asarray(v) for k, v in np.load(p).items()}
        meta_path = p.parent / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            topk, groups, gfn = meta["topk"], meta["groups"], meta["group_fn"]
        else:
            from pareto import parse_sae_name
            _, topk = parse_sae_name(p)
            groups, gfn = 0, "top1"
        acts = jax.jit(lambda X: sae.encode(params, X, topk, groups, gfn))
        return (acts, int(params["W_dec"].shape[0]), 0.0,
                np.asarray(params["W_dec"]), p.parent.name)
    from autointerp import onto_acts_fn
    acts, _, F, meta = onto_acts_fn(str(p), cfg.step, cfg.temperature)
    return acts, F, 2.0 / meta["k"], None, p.name


def split_children(C, na, nb, tau, min_fires):
    """Proper children and matches per parent from the (m1, m2) co-fire
    count matrix. P(i|j) = C_ij / nb_j must reach tau for any child; the
    reverse direction P(j|i) splits children (below tau) from matches."""
    pa = C / np.maximum(nb[None, :], 1.0)
    pb = C / np.maximum(na[:, None], 1.0)
    contained = (pa >= tau) & (nb[None, :] >= min_fires)
    proper = contained & (pb < tau)
    match = contained & (pb >= tau)
    return ([np.where(r)[0] for r in proper],
            [np.where(r)[0] for r in match])


def match_fractions(matches, live_a, nb, min_fires):
    """Seed-stability statistic: fraction of live A-latents with >= 1
    match, and fraction of live B-latents that are some A-latent's match.
    For two same-architecture runs differing only in seed, these are the
    reproducible-feature fractions."""
    n_ma = np.array([len(m_) for m_ in matches])
    a_frac = float((n_ma[live_a] >= 1).mean()) if live_a.any() else 0.0
    live_b = nb >= min_fires
    if not live_b.any():
        return a_frac, 0.0
    matched = np.zeros(len(nb), bool)
    for m_ in matches:
        matched[m_] = True
    return a_frac, float(matched[live_b].mean())


def union_cover_counts(Fa, Fb, M):
    """Per-parent count of parent-firing samples where any selected child
    fires, for one batch. M is the (m2, m1) child-of indicator."""
    hit = (Fb @ M) > 0
    return np.asarray((hit * (Fa > 0)).sum(0))


def child_overlap(Cbb, nb, children):
    """Mean pairwise child-child overlap per parent:
    C_bb[j,j'] / min(n_j, n_j'). 0 = perfectly disjoint children."""
    out = np.full(len(children), np.nan)
    for i, ch in enumerate(children):
        if len(ch) < 2:
            continue
        sub = Cbb[np.ix_(ch, ch)] / np.minimum.outer(
            np.maximum(nb[ch], 1.0), np.maximum(nb[ch], 1.0))
        iu = np.triu_indices(len(ch), 1)
        out[i] = sub[iu].mean()
    return out


def absorption_candidates(cos, pa, nb, min_fires, cos_thr=0.6, p_thr=0.2):
    """Per-parent count of B-latents geometrically near the parent's
    decoder row that do NOT co-fire with it (feature absorption)."""
    return ((cos >= cos_thr) & (pa < p_thr)
            & (nb[None, :] >= min_fires)).sum(1)


def nn_cosine(W):
    """Each row's nearest-neighbour |cosine| within the dictionary."""
    Wn = jnp.asarray(W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9),
                     jnp.float32)
    S = np.abs(np.asarray(Wn @ Wn.T))
    np.fill_diagonal(S, 0.0)
    return S.max(1)


def main():
    cfg = parse_args()
    acts_a, m1, thr_a, W_a, name_a = load_model(cfg.a, cfg)
    acts_b, m2, thr_b, W_b, name_b = load_model(cfg.b, cfg)
    out = Path(cfg.out) if cfg.out else \
        Path("data/out/sonar/splitting") / f"{name_a}__{name_b}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"A (parent): {name_a} ({m1} features)  "
          f"B (child): {name_b} ({m2} features)")

    mm = np.load(cfg.cache, mmap_mode="r")
    n = min(cfg.rows, mm.shape[0]) // cfg.batch * cfg.batch

    C = jnp.zeros((m1, m2), jnp.float32)
    Cbb = jnp.zeros((m2, m2), jnp.float32)
    na = jnp.zeros(m1, jnp.float32)
    nb = jnp.zeros(m2, jnp.float32)
    for i in range(0, n, cfg.batch):
        X = jnp.asarray(np.asarray(mm[i:i + cfg.batch], dtype=np.float32))
        Fa = (acts_a(X) > thr_a).astype(jnp.float32)
        Fb = (acts_b(X) > thr_b).astype(jnp.float32)
        C = C + Fa.T @ Fb
        Cbb = Cbb + Fb.T @ Fb
        na = na + Fa.sum(0)
        nb = nb + Fb.sum(0)
    C, Cbb = np.asarray(C), np.asarray(Cbb)
    na, nb = np.asarray(na), np.asarray(nb)
    live = na >= cfg.min_fires
    print(f"{n} rows; {live.sum()}/{m1} parents fire >= {cfg.min_fires}x")

    print(f"\n{'tau':>5} {'parents split':>14} {'mean children':>14} "
          f"{'mean matches':>13} {'A matched':>10} {'B matched':>10}")
    for tau in cfg.taus:
        ch, ma = split_children(C, na, nb, tau, cfg.min_fires)
        n_ch = np.array([len(c) for c in ch])[live]
        n_ma = np.array([len(x) for x in ma])[live]
        fa, fb = match_fractions(ma, live, nb, cfg.min_fires)
        print(f"{tau:>5.2f} {(n_ch >= 2).mean():>14.3f} "
              f"{n_ch.mean():>14.2f} {n_ma.mean():>13.2f} "
              f"{fa:>10.3f} {fb:>10.3f}")

    # detail at --tau: union coverage (second pass) + child disjointness
    children, matches = split_children(C, na, nb, cfg.tau, cfg.min_fires)
    M = np.zeros((m2, m1), np.float32)
    for i, ch in enumerate(children):
        M[ch, i] = 1.0
    Mj = jnp.asarray(M)
    cover = np.zeros(m1)
    for i in range(0, n, cfg.batch):
        X = jnp.asarray(np.asarray(mm[i:i + cfg.batch], dtype=np.float32))
        Fa = (acts_a(X) > thr_a).astype(jnp.float32)
        Fb = (acts_b(X) > thr_b).astype(jnp.float32)
        cover += union_cover_counts(Fa, Fb, Mj)
    cover = cover / np.maximum(na, 1.0)
    overlap = child_overlap(Cbb, nb, children)

    n_ch = np.array([len(c) for c in children])
    split = live & (n_ch >= 2)
    print(f"\nat tau={cfg.tau}: {split.sum()} parents split "
          f"(multiplicity median {np.median(n_ch[split]) if split.any() else 0:.0f}); "
          f"coverage {np.nanmean(cover[split]) if split.any() else 0:.3f}, "
          f"child overlap {np.nanmean(overlap[split]) if split.any() else 0:.3f}")

    absorb = None
    if W_a is not None and W_b is not None:
        An = W_a / (np.linalg.norm(W_a, axis=1, keepdims=True) + 1e-9)
        Bn = W_b / (np.linalg.norm(W_b, axis=1, keepdims=True) + 1e-9)
        cos = np.asarray(jnp.asarray(An) @ jnp.asarray(Bn).T)
        pa = C / np.maximum(nb[None, :], 1.0)
        absorb = absorption_candidates(cos, pa, nb, cfg.min_fires)
        print(f"absorption candidates (cos>=0.6, P(parent|child)<0.2): "
              f"{absorb[live].sum()} across {(absorb[live] > 0).sum()} parents")
        rng = np.random.default_rng(0)
        for name, W in ((name_a, W_a), (name_b, W_b)):
            nn = nn_cosine(W)
            base = nn_cosine(rng.normal(size=W.shape).astype(np.float32))
            q = lambda x: np.percentile(x, [50, 90, 99])
            print(f"NN decoder cosine {name}: p50/p90/p99 "
                  f"{'/'.join(f'{v:.3f}' for v in q(nn))} "
                  f"(random {'/'.join(f'{v:.3f}' for v in q(base))})")

    with open(out / "parents.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["parent", "fire_rate", "n_children", "n_matches",
                    "coverage", "child_overlap", "absorption"])
        for i in np.where(live)[0]:
            w.writerow([i, na[i] / n, len(children[i]), len(matches[i]),
                        cover[i], overlap[i],
                        absorb[i] if absorb is not None else ""])
    (out / "meta.json").write_text(json.dumps(
        {"a": str(cfg.a), "b": str(cfg.b), "rows": n, "tau": cfg.tau,
         "min_fires": cfg.min_fires, "thr_a": thr_a, "thr_b": thr_b}))
    print(f"\n-> {out / 'parents.csv'}")


if __name__ == "__main__":
    main()
