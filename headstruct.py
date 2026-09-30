"""Post-hoc head-structure analysis: do SAE latents organize into units
analogous to Ontologizer heads?

An Ontologizer head is a categorical variable by construction: k entries
that are exhaustive (every sample gets a distribution over them) and
exclusive (softmax competition). This script asks whether a trained SAE's
latents can be *discovered* to form such units: it clusters latents by the
paradigmatic-substitution signal -- pairs that fire in the same contexts
(similar co-fire profiles with the rest of the dictionary) but rarely fire
together (pointwise mutual information < 1) -- and scores the resulting
groups on the three properties heads have by construction:

  exhaustiveness  P(group fires at all)                 heads: 1
  exclusivity     E[# latents firing | >=1]             heads: 1
  sum stability   CV of the group's summed activation   heads: 0 (sums to 1)

Each metric is compared against size-matched random groups drawn from the
same live-latent pool (--nulls permutations), so the printed z-scores mean
"more head-like than chance". The signature of head structure is
exhaustiveness ABOVE null with exclusivity BELOW null (mutually exclusive
members raise P(any) and pin #fired at 1); topical co-firing clusters show
the opposite pattern. Split-half design: groups are discovered on the
first half of --rows and scored on the second, so selection noise cannot
masquerade as structure. NOTE: with very sparse codes (k=32 at m=11264,
p ~ 0.3%) pairwise co-fire estimates need a lot of data -- use --rows
~1M on a free GPU before believing a verdict.

--trained-groups skips discovery and scores the contiguous partition
recorded in the run's meta.json instead -- the validation mode for
sae.py --groups runs (rung 2 of the structural-ablation ladder), where
grouping was trained in rather than discovered.

For --group-fn softmax runs every latent fires every sample; pass a
--fire-thr (e.g. 2/group width, matching autointerp's tag threshold) so
"fires" means "meaningfully above its resting share".

  uv run python headstruct.py --sae data/out/sonar/sae/m11264_k32/params.npz
  uv run python headstruct.py --sae .../m5120_g160top1/params.npz --trained-groups

Writes <out>/groups.csv (per-group size + metrics), assignment.npy, and
prints the summary table.
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
    p.add_argument("--sae", required=True, help="params.npz of a sae.py run")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=65536,
                   help="training rows to accumulate statistics over")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--size", type=int, default=32,
                   help="group size cap for discovery (Ontologizer k)")
    p.add_argument("--topn", type=int, default=64,
                   help="candidate neighbours kept per latent")
    p.add_argument("--min-fire", type=float, default=1e-3,
                   help="latents firing less often than this stay ungrouped")
    p.add_argument("--fire-thr", type=float, default=0.0,
                   help="activation threshold defining 'fires'")
    p.add_argument("--nulls", type=int, default=20,
                   help="size-matched random partitions for the null")
    p.add_argument("--trained-groups", action="store_true",
                   help="score the meta.json --groups partition instead of "
                        "discovering one")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="output dir (default <run dir>/headstruct)")
    return p.parse_args()


def affinity_edges(C, n_fire, N, min_fire=1e-3, topn=64):
    """Candidate same-head pairs: high co-fire-profile similarity (fire with
    the same *other* latents) x anti-correlation (rarely fire together).
    C is the m x m co-fire count matrix, n_fire the per-latent fire counts.
    Returns (ii, jj, vals) with i < j, strongest first."""
    N = float(N)
    p = np.asarray(n_fire, np.float64) / N
    live = p > min_fire
    eps = 1e-12

    co = np.asarray(C, np.float64) / N
    pmi = co / (p[:, None] * p[None, :] + eps)
    R = co / np.sqrt(p[:, None] * p[None, :] + eps)
    np.fill_diagonal(R, 0.0)
    R[~live] = 0.0
    R /= np.linalg.norm(R, axis=1, keepdims=True) + eps
    S = np.asarray(jnp.asarray(R, jnp.float32) @ jnp.asarray(R.T, jnp.float32),
                   np.float64)

    A = S * np.clip(1.0 - pmi, 0.0, None)  # similar contexts, never together
    A[~live] = 0.0
    A[:, ~live] = 0.0
    np.fill_diagonal(A, 0.0)

    m = A.shape[0]
    topn = min(topn, m - 1)
    nbr = np.argpartition(-A, topn, axis=1)[:, :topn]
    ii = np.repeat(np.arange(m), topn)
    jj = nbr.ravel()
    vals = A[ii, jj]
    keep = (vals > 0) & (ii < jj)  # dedupe: each unordered pair once
    ii, jj, vals = ii[keep], jj[keep], vals[keep]
    order = np.argsort(-vals)
    return ii[order], jj[order], vals[order]


def greedy_groups(ii, jj, vals, m, cap):
    """Agglomerate edges strongest-first under a group-size cap.
    Returns labels (m,): groups of size >= 2 numbered 0..G-1, everything
    else -1."""
    parent = np.arange(m)
    size = np.ones(m, np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= cap:
            parent[rb] = ra
            size[ra] += size[rb]

    roots = np.array([find(x) for x in range(m)])
    labels = np.full(m, -1, np.int64)
    g = 0
    for r in np.unique(roots):
        members = roots == r
        if members.sum() >= 2:
            labels[members] = g
            g += 1
    return labels


def group_moments(z, fire, labels, n_groups):
    """Per-batch accumulators for the head metrics: for each group, counts
    of samples with any fire, total fires, and first two moments of the
    summed activation. z/fire are (b, m); labels (m,) with -1 = ungrouped."""
    onehot = jnp.asarray(
        (labels[:, None] == np.arange(n_groups)[None, :]).astype(np.float32))
    cnt = fire @ onehot           # (b, G) latents firing per group
    s = z @ onehot                # (b, G) summed activation per group
    # batch reductions in f32 on device, accumulation in f64 on host
    return tuple(np.asarray(x, np.float64) for x in
                 ((cnt > 0).sum(0), cnt.sum(0), s.sum(0), (s ** 2).sum(0)))


def metrics(any_c, cnt, s1, s2, n):
    """(exhaustiveness, exclusivity, sum-CV) per group from the moments."""
    exh = any_c / n
    exc = cnt / np.maximum(any_c, 1.0)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 0.0)
    cv = np.sqrt(var) / np.maximum(mean, 1e-12)
    return exh, exc, cv


def main():
    cfg = parse_args()
    path = Path(cfg.sae)
    out = Path(cfg.out) if cfg.out else path.parent / "headstruct"
    out.mkdir(parents=True, exist_ok=True)

    meta_path = path.parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:  # pre-meta.json run: recover the encode rule from the name
        from pareto import parse_sae_name
        m_, topk_ = parse_sae_name(path)
        meta = {"m": m_, "topk": topk_, "groups": 0, "group_fn": "top1"}
    params = {k: jnp.asarray(v) for k, v in np.load(path).items()}
    m = meta["m"]
    enc = lambda X: sae.encode(params, X, meta["topk"], meta["groups"],
                               meta["group_fn"])

    mm = np.load(cfg.cache, mmap_mode="r")
    n = min(cfg.rows, mm.shape[0]) // cfg.b * cfg.b
    assert n >= 2 * cfg.b, "--rows must cover at least two batches"
    # split-half: discover groups on [0, n1), score them on [n1, n)
    n1 = max(n // cfg.b // 2, 1) * cfg.b

    if cfg.trained_groups:
        assert meta["groups"], "--trained-groups needs a --groups run"
        labels = np.arange(m) // (m // meta["groups"])
        print(f"scoring the trained partition: {meta['groups']} groups "
              f"of {m // meta['groups']}")
    else:
        # pass 1 (discovery half): fire marginals + co-fire counts
        C = jnp.zeros((m, m), jnp.float32)
        n_fire = jnp.zeros(m, jnp.float32)
        for i in range(0, n1, cfg.b):
            X = jnp.asarray(np.asarray(mm[i:i + cfg.b], dtype=np.float32))
            F = (enc(X) > cfg.fire_thr).astype(jnp.float32)
            C = C + F.T @ F
            n_fire = n_fire + F.sum(0)
        ii, jj, vals = affinity_edges(np.asarray(C), np.asarray(n_fire), n1,
                                      cfg.min_fire, cfg.topn)
        labels = greedy_groups(ii, jj, vals, m, cfg.size)

    G = int(labels.max()) + 1
    grouped = labels >= 0
    assert G > 0, "no groups found (all latents below --min-fire?)"
    print(f"{G} groups covering {grouped.sum()}/{m} latents "
          f"(median size {int(np.median(np.bincount(labels[grouped])))})")

    # pass 2: head metrics for the real partition + size-matched nulls
    # (permute latent identities within the grouped pool, keeping sizes)
    rng = np.random.default_rng(cfg.seed)
    parts = [labels]
    pool = np.where(grouped)[0]
    for _ in range(cfg.nulls):
        null = np.full(m, -1, np.int64)
        null[rng.permutation(pool)] = labels[pool]
        parts.append(null)

    accs = [[np.zeros(G) for _ in range(4)] for _ in parts]
    n2 = 0
    for i in range(n1, n, cfg.b):
        X = jnp.asarray(np.asarray(mm[i:i + cfg.b], dtype=np.float32))
        z = enc(X)
        fire = (z > cfg.fire_thr).astype(jnp.float32)
        for acc, part in zip(accs, parts):
            for a, d in zip(acc, group_moments(z, fire, part, G)):
                a += np.asarray(d)
        n2 += cfg.b

    exh, exc, cv = metrics(*accs[0], n2)
    null_stats = np.array([np.concatenate([np.stack(metrics(*a, n2))], 0)
                           for a in accs[1:]])  # (nulls, 3, G)
    nm, ns = null_stats.mean((0, 2)), null_stats.mean(2).std(0) + 1e-12

    print(f"\n{'metric':<16} {'heads':>7} {'real':>8} "
          f"{'null':>8} {'z':>7}")
    for name, val, ideal, i in [("exhaustiveness", exh.mean(), 1.0, 0),
                                ("exclusivity", exc.mean(), 1.0, 1),
                                ("sum CV", cv.mean(), 0.0, 2)]:
        print(f"{name:<16} {ideal:>7.2f} {val:>8.4f} {nm[i]:>8.4f} "
              f"{(val - nm[i]) / ns[i]:>7.1f}")

    sizes = np.bincount(labels[grouped], minlength=G)
    with open(out / "groups.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "size", "exhaustiveness", "exclusivity",
                         "sum_cv"])
        for g in range(G):
            writer.writerow([g, sizes[g], exh[g], exc[g], cv[g]])
    np.save(out / "assignment.npy", labels)
    print(f"\n-> {out / 'groups.csv'}")


if __name__ == "__main__":
    main()
