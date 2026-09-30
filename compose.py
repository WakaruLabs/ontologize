"""Intervention composition (additivity): treating heads as candidate
causal variables, do set-interventions on two distinct heads compose
additively in decode space? For each pair of forced assignments
(l1,h1)->k1, (l2,h2)->k2 we compare the joint intervention's decode delta
D12 against the sum D1 + D2 of the single-intervention deltas, per eval
row: cosine(D12, D1+D2) and the relative error ||D12-(D1+D2)||/||D1+D2||,
under both raw SONAR cosine and the training objective's inverse-variance
metric.

The decode is exactly linear in the flattened classification
(Y = P_flat G), so two set-interventions in the FINAL layer compose
exactly additively by construction -- that stratum is the machinery
check. Any non-additivity elsewhere measures downstream reclassification:
later layers re-reading the intervened residual. Same-layer pairs are
reported per layer; cross-layer pairs by the earlier layer of the pair.

The final-layer stratum also measures the pipeline's NUMERICAL floor,
which is much larger than float32 eps: ||R|| ~ 31 decodes to ||Y|| ~ 1,
so the decoder einsum cancels ~30x and each forward carries ~4e-3
absolute rounding noise that does not cancel between differently-shaped
(differently-compiled) intervention variants. Verified: P- and R-space
additivity at the final layer are exact to rounding, and a float64
recompute of the deltas from R gives rel ~3e-6, while the float32
pipeline reads ~0.02. Treat rel ~0.02 as the floor; values well above it
are structural.

  uv run python compose.py --ckpt data/out/sonar/multilingual/resid_nc
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import numpy as np
from itertools import combinations
from pathlib import Path


# ---------- pure helpers (unit-tested) ----------

def additivity_stats(D1, D2, D12, w=None):
    """Median per-row additivity of D12 vs D1+D2, rows (b, d): cosine and
    relative error, optionally under a diagonal metric w (all deltas
    scaled by sqrt(w) first)."""
    if w is not None:
        s = np.sqrt(w)
        D1, D2, D12 = D1 * s, D2 * s, D12 * s
    S = D1 + D2
    nS = np.linalg.norm(S, axis=-1)
    cos = (D12 * S).sum(-1) / (np.linalg.norm(D12, axis=-1) * nS + 1e-12)
    rel = np.linalg.norm(D12 - S, axis=-1) / (nS + 1e-12)
    return float(np.median(cos)), float(np.median(rel))


def sample_singles(l, h, k, per_layer, seed):
    """Per layer, `per_layer` distinct heads with one random entry each:
    [(layer, head, entry), ...]."""
    rng = np.random.default_rng(seed)
    singles = []
    for li in range(l):
        for hi in rng.choice(h, min(per_layer, h), replace=False):
            singles.append((li, int(hi), int(rng.integers(k))))
    return singles


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768)
    p.add_argument("--rows", type=int, default=512,
                   help="eval rows sampled from the cache tail")
    p.add_argument("--singles-per-layer", type=int, default=8)
    p.add_argument("--cross-pairs", type=int, default=60)
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    cfg = p.parse_args()

    import jax.numpy as jnp
    from pareto import load_onto
    from ontologize.ontologizer import Ontologizer, DictIntervention

    model, mparams, step = load_onto(cfg.ckpt, cfg.step)
    l, h, k = model.l, model.h, model.k
    out = Path(cfg.out or Path(cfg.ckpt) / "compose")
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    tail = mm[len(mm) - cfg.eval_rows:]
    rng = np.random.default_rng(cfg.seed)
    X = jnp.asarray(np.array(
        tail[np.sort(rng.choice(cfg.eval_rows, cfg.rows, replace=False))],
        dtype=np.float32))
    w = np.load(cfg.mse_weights)

    def run(arglist):
        Y, _, _, _ = model.apply({"params": mparams}, X, arglist,
                                 temperature=cfg.temperature,
                                 method=Ontologizer.withArgs)
        return np.asarray(Y)

    def arglist_for(*fs):
        args = [DictIntervention() for _ in range(l)]
        for li in set(f[0] for f in fs):
            hs = [f[1] for f in fs if f[0] == li]
            ks = [f[2] for f in fs if f[0] == li]
            args[li] = DictIntervention(h_set=jnp.array(hs),
                                        k_set=jnp.array(ks))
        return args

    Y0 = run([DictIntervention() for _ in range(l)])
    singles = sample_singles(l, h, k, cfg.singles_per_layer, cfg.seed)
    D = {f: run(arglist_for(f)) - Y0 for f in singles}

    by_layer = [[f for f in singles if f[0] == li] for li in range(l)]
    pairs = [pr for li in range(l) for pr in combinations(by_layer[li], 2)]
    cross = [(a, b) for a, b in combinations(singles, 2) if a[0] != b[0]]
    pairs += [cross[i] for i in
              rng.choice(len(cross), min(cfg.cross_pairs, len(cross)),
                         replace=False)]

    rows = []
    for f1, f2 in pairs:
        D12 = run(arglist_for(f1, f2)) - Y0
        cos_r, rel_r = additivity_stats(D[f1], D[f2], D12)
        cos_w, rel_w = additivity_stats(D[f1], D[f2], D12, w)
        # median whitened norm of the summed delta: small-magnitude pairs
        # (e.g. frozen layer-4 heads) have rel inflated by float noise
        mag = float(np.median(np.linalg.norm(
            (D[f1] + D[f2]) * np.sqrt(w), axis=-1)))
        rows.append({"l1": f1[0], "h1": f1[1], "k1": f1[2],
                     "l2": f2[0], "h2": f2[1], "k2": f2[2],
                     "cos_raw": cos_r, "rel_raw": rel_r,
                     "cos_w": cos_w, "rel_w": rel_w, "mag_w": mag})

    with open(out / "pairs.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    def agg(rs):
        return {"n": len(rs),
                "median_cos_w": float(np.median([r["cos_w"] for r in rs])),
                "median_rel_w": float(np.median([r["rel_w"] for r in rs])),
                "median_cos_raw": float(np.median([r["cos_raw"] for r in rs])),
                "median_rel_raw": float(np.median([r["rel_raw"] for r in rs]))}

    summary = {"ckpt": str(cfg.ckpt), "step": int(step), "rows": cfg.rows,
               "temperature": cfg.temperature, "seed": cfg.seed,
               "same_layer": {}, "cross_by_earlier_layer": {}}
    for li in range(l):
        rs = [r for r in rows if r["l1"] == li and r["l2"] == li]
        if rs:
            summary["same_layer"][str(li)] = agg(rs)
            a = summary["same_layer"][str(li)]
            print(f"same-layer L{li}: n={a['n']} cos_w {a['median_cos_w']:.4f} "
                  f"rel_w {a['median_rel_w']:.4f}")
    for li in range(l):
        rs = [r for r in rows if r["l1"] != r["l2"]
              and min(r["l1"], r["l2"]) == li]
        if rs:
            summary["cross_by_earlier_layer"][str(li)] = agg(rs)
            a = summary["cross_by_earlier_layer"][str(li)]
            print(f"cross from L{li}: n={a['n']} cos_w {a['median_cos_w']:.4f} "
                  f"rel_w {a['median_rel_w']:.4f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
