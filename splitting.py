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
Containment is a strict criterion, so the field's own agreement statistics
are reported beside it, each both ways over --thetas, in agreement.json:
  MPPC           each live latent's largest Pearson correlation of
                 activations with any latent of the other model (Wang et
                 al. 2024); no threshold, a maximum, not a matching
  decoder share  the fraction of live latents whose decoder row has cosine
                 above theta with some latent of the other model (Leask et
                 al.'s reconstruction latents; 0.94 at 0.7 for same-size
                 GPT-2 SAEs from two seeds); SAE pairs only
  PW-MCC         the mean |cosine| of decoder rows matched one to one by
                 the Hungarian algorithm over every latent, against random
                 directions (Song et al. 2025); SAE pairs only

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
    p.add_argument("--thetas", type=float, nargs="+", default=[0.4, 0.5, 0.7, 0.9],
                   help="cosine and correlation thresholds for the decoder share, "
                        "PW-MCC's matched pairs and MPPC (Leask et al. use 0.7 on "
                        "GPT-2 and 0.4 on Gemma)")
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


def decoder_shares(cos, live_a, live_b, thetas):
    """Leask et al.'s reconstruction share, both ways: for each theta, the
    fraction of live A-latents whose decoder row has cosine above theta
    with some live B-latent's, and of live B-latents against A. A maximum
    over every latent of the other model, not a one-to-one match, so the
    most lenient of the agreement statistics here."""
    c = cos[np.ix_(live_a, live_b)]
    best_a, best_b = c.max(1), c.max(0)
    return {float(t): (float((best_a > t).mean()), float((best_b > t).mean()))
            for t in thetas}


def pw_mcc(cos):
    """Song et al.'s PW-MCC: the mean |cosine| between decoder rows matched
    one to one by the Hungarian algorithm, over every latent (dead and
    rare ones count, as in the paper), with the matched |cosines|."""
    from scipy.optimize import linear_sum_assignment
    a = np.abs(np.asarray(cos, np.float64))
    r, c = linear_sum_assignment(a, maximize=True)
    return float(a[r, c].mean()), a[r, c]


def max_corr(n, sa, sb, saa, sbb, sab):
    """MPPC (Wang et al. 2024, Eq. 8), both ways, from running sums of the
    activations over n rows: each latent's largest Pearson correlation with
    any latent of the other model. A latent whose activation never varies
    has no correlation and reads nan."""
    mu_a, mu_b = sa / n, sb / n
    sd_a = np.sqrt(np.maximum(saa / n - mu_a ** 2, 0.0))
    sd_b = np.sqrt(np.maximum(sbb / n - mu_b ** 2, 0.0))
    sd = np.outer(sd_a, sd_b)
    r = np.where(sd > 0, (sab / n - np.outer(mu_a, mu_b)) / np.where(sd > 0, sd, 1.0),
                 -np.inf)
    best_a, best_b = r.max(1), r.max(0)
    return (np.where(np.isfinite(best_a), best_a, np.nan),
            np.where(np.isfinite(best_b), best_b, np.nan))


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
    # activation sums for MPPC, carried in float64 across batches
    sa, sb, saa, sbb = (np.zeros(m, np.float64) for m in (m1, m2, m1, m2))
    sab = np.zeros((m1, m2), np.float64)
    for i in range(0, n, cfg.batch):
        X = jnp.asarray(np.asarray(mm[i:i + cfg.batch], dtype=np.float32))
        Aa, Ab = acts_a(X).astype(jnp.float32), acts_b(X).astype(jnp.float32)
        Fa = (Aa > thr_a).astype(jnp.float32)
        Fb = (Ab > thr_b).astype(jnp.float32)
        C = C + Fa.T @ Fb
        Cbb = Cbb + Fb.T @ Fb
        na = na + Fa.sum(0)
        nb = nb + Fb.sum(0)
        sa += np.asarray(Aa.sum(0), np.float64)
        sb += np.asarray(Ab.sum(0), np.float64)
        saa += np.asarray((Aa * Aa).sum(0), np.float64)
        sbb += np.asarray((Ab * Ab).sum(0), np.float64)
        sab += np.asarray(Aa.T @ Ab, np.float64)
    C, Cbb = np.asarray(C), np.asarray(Cbb)
    na, nb = np.asarray(na), np.asarray(nb)
    live = na >= cfg.min_fires
    live_b = nb >= cfg.min_fires
    print(f"{n} rows; {live.sum()}/{m1} parents fire >= {cfg.min_fires}x")

    print(f"\n{'tau':>5} {'parents split':>14} {'mean children':>14} "
          f"{'mean matches':>13} {'A matched':>10} {'B matched':>10}")
    agreement = {"containment": {}}
    for tau in cfg.taus:
        ch, ma = split_children(C, na, nb, tau, cfg.min_fires)
        n_ch = np.array([len(c) for c in ch])[live]
        n_ma = np.array([len(x) for x in ma])[live]
        fa, fb = match_fractions(ma, live, nb, cfg.min_fires)
        agreement["containment"][float(tau)] = (fa, fb)
        print(f"{tau:>5.2f} {(n_ch >= 2).mean():>14.3f} "
              f"{n_ch.mean():>14.2f} {n_ma.mean():>13.2f} "
              f"{fa:>10.3f} {fb:>10.3f}")

    # the same question in the activation frame without thresholds
    mppc_a, mppc_b = max_corr(n, sa, sb, saa, sbb, sab)
    agreement["mppc"] = {
        "mean": (float(np.nanmean(mppc_a[live])), float(np.nanmean(mppc_b[live_b]))),
        "above": {float(t): (float((mppc_a[live] > t).mean()),
                             float((mppc_b[live_b] > t).mean())) for t in cfg.thetas}}
    print(f"\nMPPC (each live latent's best activation correlation with the other "
          f"model), A->B / B->A: mean {agreement['mppc']['mean'][0]:.3f} / "
          f"{agreement['mppc']['mean'][1]:.3f}; above "
          + ", ".join(f"{t}: {a:.3f} / {b:.3f}"
                      for t, (a, b) in agreement["mppc"]["above"].items()))

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
        shares = decoder_shares(cos, live, live_b, cfg.thetas)
        agreement["decoder_share"] = shares
        print("decoder-cosine reconstruction share (Leask et al.), A->B / B->A: "
              + ", ".join(f"cos > {t}: {a:.3f} / {b:.3f}" for t, (a, b) in shares.items()))
        mcc, matched = pw_mcc(cos)
        rng = np.random.default_rng(0)
        Ra = rng.normal(size=W_a.shape)
        Rb = rng.normal(size=W_b.shape)
        Ra /= np.linalg.norm(Ra, axis=1, keepdims=True)
        Rb /= np.linalg.norm(Rb, axis=1, keepdims=True)
        mcc_null, _ = pw_mcc(Ra @ Rb.T)
        agreement["pw_mcc"] = {"value": mcc, "random_directions": mcc_null,
                               "matched_above": {float(t): float((matched > t).mean())
                                                 for t in cfg.thetas}}
        print(f"PW-MCC (Song et al.; Hungarian-matched |cos|, every latent): {mcc:.3f} "
              f"(random directions {mcc_null:.3f}); matched pairs above "
              + ", ".join(f"{t}: {v:.3f}"
                          for t, v in agreement["pw_mcc"]["matched_above"].items()))
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
    (out / "agreement.json").write_text(json.dumps(
        {"rows": n, "live": [int(live.sum()), int(live_b.sum())], **agreement},
        indent=1))
    print(f"\n-> {out / 'parents.csv'}")


if __name__ == "__main__":
    main()
