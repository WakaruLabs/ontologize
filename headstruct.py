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

--onto scores a trained Ontologizer's heads the same way: its flattened
assignments (`autointerp.onto_acts_fn`, the l*h*k code at --temperature)
are the latents and each (layer, head) is a group of k, numbered
layer*h + head in groups.csv. The three head metrics hold by construction
there, so --onto is for --modularity. Each layer's heads are scored on the
graph of that layer's own input -- X for layer 0, the residual it
classifies above that (`onto_acts_fn(..., inputs=True)`) -- since an upper
layer partitions what the layers below left unexplained, not X. Nulls are
then permuted within each layer, so every null group has one layer and one
graph. Leave --fire-thr at 0 so the cells are the full soft assignment.

--prefix-layers is the SAE control for that per-layer reading. A
Matryoshka run (sae.py --prefixes P) trains each nested prefix of its P
latent blocks to reconstruct alone, the nearest SAE analogue of a
residual stack. Its blocks are treated as layers: discovery keeps only
pairs inside one block, nulls are permuted within blocks, and block L is
scored on the graph of its prefix residual, X - decode(code on blocks
< L) -- X - b_dec for block 0, which is the SAE encoder's own input. One
asymmetry is unavoidable: an Ontologizer layer classifies its residual,
while a Matryoshka block is encoded from X by the shared encoder and only
trained to reconstruct that residual. Under a global top-k the later
blocks fire on fewer samples, so each block row also reports its groups'
mean coverage.

For --group-fn softmax runs every latent fires every sample; pass a
--fire-thr (e.g. 2/group width, matching autointerp's tag threshold) so
"fires" means "meaningfully above its resting share".

--modularity adds a fourth, graph-side metric. The three above ask whether
a group behaves like one categorical variable; modularity asks whether the
partition it induces on the samples is a community structure of the input.
Each group splits the samples it fires on into soft cells,
P_ic = a_ic / sum_{c' in group} a_ic' over its members' activations, and
that partition is scored by soft modularity (Newman 2006; the fuzzy form of
Zhang et al. 2007) on each scoring batch's input affinity graph -- the
heat kernel of `ontologize.fns.pwak.affinity`, self-edges removed:

  Q = (1/2m) [ tr(P^T D P) - gamma * sum_c (k^T P_c)^2 / 2m ]

restricted to the subgraph of samples the group fires on (degrees k and
2m are taken within it), so a sparse group is not penalized for its
coverage, which exhaustiveness already measures. Q is 0 when every fired
sample lands in one cell and positive when cells hold more affinity than
the degree-preserving null model expects. It is not compared against an
ideal: any partition of points by firing regions is somewhat spatially
coherent, so the informative comparison is the size-matched null, i.e.
"more community-like than a random grouping of the same latents". gamma
is the resolution (Reichardt & Bornholdt 2006). On unit-norm embeddings
the dense heat kernel is nearly uniform -- all pairwise weights within a
small factor of each other -- and no partition of a near-uniform graph
has much modularity, so real and null both sit near 0. --knn K keeps each
sample's K strongest edges (symmetrized by union, weights kept), the kNN
graph of DEWAKSS and Leiden-on-kNN pipelines, which gives Q room to
separate partitions. What Q measures is alignment with the input's
dominant cluster structure, not quality as a variable: a head that
encodes one of many independent factors cuts across the joint kNN graph
(neighbours share most factors, not this one), so low Q is the expected
reading for a factorial code. Soft cells near uniform also force Q toward
0, so read Q alongside the assignment entropy. The graph is batch-local,
so Q is averaged over scoring batches. Cost is one (b, b) x (b, m) matmul
per batch per partition, real and null alike.

  uv run python headstruct.py --sae data/out/sonar/sae/m11264_k32/params.npz
  uv run python headstruct.py --sae .../m5120_g160top1/params.npz --trained-groups
  uv run python headstruct.py --sae .../m11264_k32/params.npz --modularity
  uv run python headstruct.py --onto data/out/sonar/multilingual/resid_nc \
      --modularity --knn 15
  uv run python headstruct.py --sae .../m5120_k32_p5/params.npz --prefix-layers \
      --modularity --knn 15

Writes <out>/groups.csv (per-group size + metrics; a trailing modularity
column with --modularity), assignment.npy, and summary.json (the printed
summary table: per metric, and per layer or block, the real value, the null
mean and spread, z, and a block's group count and coverage).
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
from jaxtyping import Array, Float
from pathlib import Path

import sae
from ontologize.fns.pwak import affinity


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sae", help="params.npz of a sae.py run")
    src.add_argument("--onto", help="Ontologizer checkpoint dir; scores its "
                                    "heads as the trained partition")
    p.add_argument("--step", type=int, default=0,
                   help="--onto checkpoint step (0 = latest)")
    p.add_argument("--temperature", type=float, default=0.03,
                   help="--onto classification temperature")
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
    p.add_argument("--modularity", action="store_true",
                   help="also score each group's sample partition by soft "
                        "modularity on the input affinity graph")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="modularity resolution")
    p.add_argument("--tau", type=float, default=0.2,
                   help="heat-kernel temperature of the input affinity")
    p.add_argument("--prefix-layers", action="store_true",
                   help="treat a Matryoshka (sae.py --prefixes) run's prefix "
                        "blocks as layers: discovery and nulls within blocks, "
                        "each block scored on its prefix residual's graph")
    p.add_argument("--knn", type=int, default=0,
                   help="sparsify the affinity to each sample's K nearest "
                        "neighbours (union, kernel weights kept); 0 = dense")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="output dir (default <run dir>/headstruct)")
    return p.parse_args()


def affinity_edges(C, n_fire, N, min_fire=1e-3, topn=64, block=0):
    """Candidate same-head pairs: high co-fire-profile similarity (fire with
    the same *other* latents) x anti-correlation (rarely fire together).
    C is the m x m co-fire count matrix, n_fire the per-latent fire counts.
    block > 0 keeps only pairs inside the same block of `block` contiguous
    latents, so discovered groups never straddle blocks.
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
    if block:
        blk = np.arange(A.shape[0]) // block
        A[blk[:, None] != blk[None, :]] = 0.0

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


def knn_graph(D: Float[Array, "b b"], k: int) -> Float[Array, "b b"]:
    """Keep each row's k largest affinities, symmetrized by union (an edge
    survives if either endpoint ranks the other among its k nearest), with
    the kernel weights kept. D must have a zero diagonal and k < b."""
    v = jax.lax.top_k(D, k)[0][:, -1:]
    keep = D >= v
    return jnp.where(keep | keep.T, D, 0.0)


def soft_modularity(D: Float[Array, "b b"], z: Float[Array, "b m"],
                    fire: Float[Array, "b m"], labels, n_groups: int,
                    gamma: float = 1.0) -> np.ndarray:
    """Per-group soft modularity of the sample partition each group induces
    on one batch, scored on the affinity graph D (zero diagonal) restricted
    to the samples the group fires on. Cell weights are the members'
    activations normalized within the group. Returns (G,) float64, NaN for
    a group with no fired edges in this batch. labels (m,), -1 = ungrouped."""
    onehot = jnp.asarray(
        (labels[:, None] == np.arange(n_groups)[None, :]).astype(np.float32))
    a = z * fire                  # (b, m) activation where it counts as firing
    tot = a @ onehot              # (b, G) group mass per sample
    den = tot @ onehot.T          # (b, m) each latent's own group mass
    P = jnp.where(den > 0, a / jnp.where(den > 0, den, 1.0), 0.0)
    f = (tot > 0).astype(jnp.float32)
    k = f * (D @ f)               # (b, G) degree within the fired subgraph
    two_m = k.sum(0)
    within = (P * (D @ P)).sum(0) @ onehot          # tr(P^T D P) per group
    kP = ((k @ onehot.T) * P).sum(0)                # (m,) k^T P_c
    null = (kP ** 2) @ onehot
    safe = jnp.where(two_m > 0, two_m, 1.0)
    Q = jnp.where(two_m > 0, (within - gamma * null / safe) / safe, jnp.nan)
    return np.asarray(Q, np.float64)


def block_modularity(Ds, z: Float[Array, "b m"], fire: Float[Array, "b m"],
                     labels, n_groups: int, bw: int,
                     gamma: float = 1.0) -> np.ndarray:
    """Per-group soft modularity with each block of `bw` contiguous latents
    scored on its own graph: Ds[L] is the affinity graph of block L's input
    (an Ontologizer layer's heads, a Matryoshka prefix block). Every group
    must lie inside one block, which holds for the heads, for discovery
    restricted to blocks, and for nulls permuted within a block. Returns
    (n_groups,), NaN where a group has no fired edges."""
    Q = np.full(n_groups, np.nan)
    for L, D in enumerate(Ds):
        sl = slice(L * bw, (L + 1) * bw)
        lab = labels[sl]
        present = np.unique(lab[lab >= 0])
        if not len(present):
            continue
        local = np.where(lab >= 0, np.searchsorted(present, lab), -1)
        Q[present] = soft_modularity(D, z[:, sl], fire[:, sl], local,
                                     len(present), gamma)
    return Q


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
    labels = layer_inputs = None
    assert not (cfg.onto and cfg.prefix_layers), \
        "--prefix-layers applies to --sae runs"
    if cfg.onto:
        # the flattened assignments, latent (l*h + head)*k + entry, so the
        # heads are the contiguous partition into groups of k
        from autointerp import onto_acts_fn
        enc, _, m, ometa, layer_inputs = onto_acts_fn(
            cfg.onto, cfg.step, cfg.temperature, inputs=True)
        labels = np.arange(m) // ometa["k"]
        out = Path(cfg.out) if cfg.out else Path(cfg.onto) / "headstruct"
        print(f"scoring Ontologizer heads (step {ometa['step']}): "
              f"{ometa['l']} layers x {ometa['h']} heads of {ometa['k']}")
    else:
        path = Path(cfg.sae)
        out = Path(cfg.out) if cfg.out else path.parent / "headstruct"
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
        if cfg.trained_groups:
            assert meta["groups"], "--trained-groups needs a --groups run"
            labels = np.arange(m) // (m // meta["groups"])
            print(f"scoring the trained partition: {meta['groups']} groups "
                  f"of {m // meta['groups']}")
    out.mkdir(parents=True, exist_ok=True)

    # blocks: contiguous latent ranges that each get their own graph, null
    # pool and summary row -- Ontologizer layers, Matryoshka prefix blocks,
    # or the whole dictionary as one block
    if cfg.onto:
        n_blocks, kind = ometa["l"], "layer"
    elif cfg.prefix_layers:
        n_blocks, kind = meta.get("prefixes", 1), "block"
        assert n_blocks > 1, "--prefix-layers needs a sae.py --prefixes run"
        print(f"scoring {n_blocks} prefix blocks of {m // n_blocks}")
    else:
        n_blocks, kind = 1, None
    bw = m // n_blocks
    lat_blk = np.arange(m) // bw

    mm = np.load(cfg.cache, mmap_mode="r")
    n = min(cfg.rows, mm.shape[0]) // cfg.b * cfg.b
    assert n >= 2 * cfg.b, "--rows must cover at least two batches"
    # split-half: discover groups on [0, n1), score them on [n1, n)
    n1 = max(n // cfg.b // 2, 1) * cfg.b

    if labels is None:
        # pass 1 (discovery half): fire marginals + co-fire counts
        C = jnp.zeros((m, m), jnp.float32)
        n_fire = jnp.zeros(m, jnp.float32)
        for i in range(0, n1, cfg.b):
            X = jnp.asarray(np.asarray(mm[i:i + cfg.b], dtype=np.float32))
            F = (enc(X) > cfg.fire_thr).astype(jnp.float32)
            C = C + F.T @ F
            n_fire = n_fire + F.sum(0)
        ii, jj, vals = affinity_edges(np.asarray(C), np.asarray(n_fire), n1,
                                      cfg.min_fire, cfg.topn,
                                      bw if n_blocks > 1 else 0)
        labels = greedy_groups(ii, jj, vals, m, cfg.size)

    G = int(labels.max()) + 1
    grouped = labels >= 0
    assert G > 0, "no groups found (all latents below --min-fire?)"
    grp_blk = np.full(G, -1)
    grp_blk[labels[grouped]] = lat_blk[grouped]
    assert all((lat_blk[labels == g] == grp_blk[g]).all() for g in range(G)), \
        "a group straddles blocks"
    print(f"{G} groups covering {grouped.sum()}/{m} latents "
          f"(median size {int(np.median(np.bincount(labels[grouped])))})")

    # pass 2: head metrics for the real partition + size-matched nulls
    # (permute latent identities within the grouped pool, keeping sizes;
    # within each block, so every null group has a block and hence a graph)
    rng = np.random.default_rng(cfg.seed)
    parts = [labels]
    pools = [np.where(grouped & (lat_blk == L))[0] for L in range(n_blocks)]
    for _ in range(cfg.nulls):
        null = np.full(m, -1, np.int64)
        for pool in pools:
            null[rng.permutation(pool)] = labels[pool]
        parts.append(null)

    def graph(E):
        D = affinity(E, cfg.tau)
        return knn_graph(D, cfg.knn) if cfg.knn else D

    accs = [[np.zeros(G) for _ in range(4)] for _ in parts]
    # modularity: per-partition sum and count of finite per-batch Q
    qacc = [[np.zeros(G), np.zeros(G)] for _ in parts]
    n2 = 0
    for i in range(n1, n, cfg.b):
        X = jnp.asarray(np.asarray(mm[i:i + cfg.b], dtype=np.float32))
        z = enc(X)
        fire = (z > cfg.fire_thr).astype(jnp.float32)
        if cfg.modularity:
            # one graph per block: for --onto X, then each layer's own
            # input; for --prefix-layers the residual of the prefix before
            # the block, X - decode(code on blocks < L) (block 0: X - b_dec,
            # the SAE encoder's own input); X otherwise
            if cfg.onto:
                Ds = [graph(X)] + [graph(U) for U in layer_inputs(X)]
            elif cfg.prefix_layers:
                Ds = [graph(X - sae.decode(params, z * (lat_blk < L)))
                      for L in range(n_blocks)]
            else:
                Ds = [graph(X)]
        for acc, q, part in zip(accs, qacc, parts):
            for a, d in zip(acc, group_moments(z, fire, part, G)):
                a += np.asarray(d)
            if cfg.modularity:
                Q = block_modularity(Ds, z, fire, part, G, bw, cfg.gamma)
                q[0] += np.nan_to_num(Q)
                q[1] += np.isfinite(Q)
        n2 += cfg.b

    exh, exc, cv = metrics(*accs[0], n2)
    null_stats = np.array([np.concatenate([np.stack(metrics(*a, n2))], 0)
                           for a in accs[1:]])  # (nulls, 3, G)
    nm, ns = null_stats.mean((0, 2)), null_stats.mean(2).std(0) + 1e-12

    # the printed table, also written to summary.json; null_sd is the spread
    # of the per-null means, the denominator of z
    table = []

    def entry(metric, real, null, null_sd, **extra):
        z = (real - null) / (null_sd + 1e-12)
        table.append(dict(metric=metric, real=float(real), null=float(null),
                          null_sd=float(null_sd), z=float(z), **extra))
        return z

    print(f"\n{'metric':<16} {'heads':>7} {'real':>8} "
          f"{'null':>8} {'z':>7}")
    for name, val, ideal, i in [("exhaustiveness", exh.mean(), 1.0, 0),
                                ("exclusivity", exc.mean(), 1.0, 1),
                                ("sum CV", cv.mean(), 0.0, 2)]:
        z = entry(name, val, nm[i], ns[i], ideal=ideal)
        print(f"{name:<16} {ideal:>7.2f} {val:>8.4f} {nm[i]:>8.4f} {z:>7.1f}")
    if cfg.modularity:
        # groups that never fired on a scoring batch have no Q
        mod = [s / np.where(c > 0, c, np.nan) for s, c in qacc]
        qn = np.array([np.nanmean(q) for q in mod[1:]])
        val = np.nanmean(mod[0])
        z = entry("modularity", val, qn.mean(), qn.std())
        print(f"{'modularity':<16} {'-':>7} {val:>8.4f} {qn.mean():>8.4f} "
              f"{z:>7.1f}")
        # each block against its own within-block nulls; cov is its groups'
        # mean exhaustiveness, how much of the batch the block's graphs see
        for L in range(n_blocks if kind else 0):
            b = grp_blk == L
            if not b.any():
                print(f"{f'  {kind} {L}':<16} no groups")
                continue
            qL = np.array([np.nanmean(q[b]) for q in mod[1:]])
            vL = np.nanmean(mod[0][b])
            z = entry("modularity", vL, qL.mean(), qL.std(), kind=kind,
                      block=L, groups=int(b.sum()),
                      coverage=float(exh[b].mean()))
            print(f"{f'  {kind} {L}':<16} {'-':>7} {vL:>8.4f} "
                  f"{qL.mean():>8.4f} {z:>7.1f}"
                  f"   {b.sum()} groups, cov {exh[b].mean():.3f}")

    sizes = np.bincount(labels[grouped], minlength=G)
    with open(out / "groups.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "size", "exhaustiveness", "exclusivity",
                         "sum_cv"] + (["modularity"] if cfg.modularity else []))
        for g in range(G):
            writer.writerow([g, sizes[g], exh[g], exc[g], cv[g]]
                            + ([mod[0][g]] if cfg.modularity else []))
    np.save(out / "assignment.npy", labels)
    (out / "summary.json").write_text(json.dumps(dict(
        source=cfg.onto or cfg.sae, rows=n2, nulls=cfg.nulls,
        trained_groups=bool(cfg.onto or cfg.trained_groups),
        prefix_layers=cfg.prefix_layers, modularity=cfg.modularity,
        knn=cfg.knn, tau=cfg.tau, gamma=cfg.gamma, fire_thr=cfg.fire_thr,
        table=table), indent=1))
    print(f"\n-> {out / 'groups.csv'}, summary.json")


if __name__ == "__main__":
    main()
