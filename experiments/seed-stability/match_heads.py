"""Partition-level seed stability for the Ontologizer.

splitting.py answers the entry-level question (do individual features
reproduce across seeds? SAE k32 vs k32_s43: 0.31/0.24/0.15/0.03 at tau
0.3/0.5/0.7/0.9) but the paper's frame note is explicit: for a steering
generator the unit that must reproduce is the HEAD's task (partition),
not the individual feature vector. This script measures that:

  1. head matching   each entry has an exact decode direction in SONAR
     output space (the row of G with Y = P_flat @ G,
     refit.onto_linear_model). Head-to-head similarity between runs =
     mean entry cosine under the optimal entry bijection (Hungarian on
     the k x k cosine matrix); heads are then matched across runs by a
     second Hungarian on that similarity matrix, within the same layer
     (layers are ordered residual stages; --cross-layer lifts this).
  2. entry matching   within each matched head pair, the entry bijection
     from step 1: matched-entry cosine fractions over a threshold sweep.
  3. activation cross-check (--rows > 0)   the same Hungarian-matched
     entry pairs scored in splitting.py's own frame: co-fire containment
     (P(a|b) and P(b|a) over shared cache rows, fire = p > 2/k) at the
     same tau sweep -- the number that lands directly next to the SAEs'
     0.31/0.24/0.15/0.03.

A shuffled null (entries randomly regrouped into heads within each
layer of run B, --nulls draws) gives the chance level of the head-sim
statistic. NOTE: Ontologizer checkpoints restore on GPU JAX only.

  uv run python experiments/seed-stability/match_heads.py \\
      --a data/out/sonar/multilingual/resid_nc_hm \\
      --b data/out/sonar/multilingual/resid_nc_hm_s43

Writes <out>/heads.csv, <out>/entries.csv, <out>/summary.json.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import numpy as np

SAE_REFERENCE = "SAE k32 vs k32_s43: 0.31/0.24/0.15/0.03"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default="data/out/sonar/multilingual/resid_nc_hm",
                   help="first Ontologizer checkpoint dir")
    p.add_argument("--b",
                   default="data/out/sonar/multilingual/resid_nc_hm_s43",
                   help="second-seed checkpoint dir (train_seed43.py)")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--metric", choices=["raw", "whitened"], default="raw",
                   help="cosine metric: raw SONAR space, or the training "
                        "objective's inverse-variance inner product "
                        "(headcoh convention: rows scaled by sqrt(w))")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--center", action="store_true",
                   help="subtract each layer's mean decode direction "
                        "before cosine (removes layer-common structure)")
    p.add_argument("--cross-layer", action="store_true",
                   help="match heads across all layers, not within-layer")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=262144,
                   help="cache rows for the activation cross-check "
                        "(0 = geometry only)")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--min-fires", type=int, default=50,
                   help="entry pairs where either side fires fewer times "
                        "are excluded from the containment fractions")
    p.add_argument("--taus", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9],
                   help="containment sweep (splitting.py's)")
    p.add_argument("--cos-thrs", type=float, nargs="+",
                   default=[0.3, 0.5, 0.7, 0.9],
                   help="decode-direction cosine sweep")
    p.add_argument("--nulls", type=int, default=5,
                   help="shuffled-regrouping null draws (0 = skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default experiments/seed-stability/out/<a>__<b>")
    return p.parse_args()


# ---------- assignment ----------

def _hungarian_py(cost):
    """Exact min-cost square assignment (potentials + shortest augmenting
    path, O(n^3)); pure-python fallback when scipy is unavailable."""
    n = len(cost)
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    ans = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            ans[p[j] - 1] = j - 1
    return np.array(ans, int)


try:  # scipy ships transitively with jax (see uv.lock); fallback is exact too
    from scipy.optimize import linear_sum_assignment as _lsa

    def assign_max(S):
        """Column index per row maximizing the total similarity."""
        return _lsa(-np.asarray(S))[1]
except ImportError:  # pragma: no cover
    def assign_max(S):
        return _hungarian_py((-np.asarray(S)).tolist())


# ---------- pure helpers ----------

def unit_rows(R, eps=1e-9):
    return R / (np.linalg.norm(R, axis=-1, keepdims=True) + eps)


def pool_match(A, B):
    """A, B: (h, k, d) unit entry directions of one pool (layer).
    Returns (head_perm (h,), head_sims (h,), entry_perms (h, k),
    entry_cos (h, k)): head a of A matches head head_perm[a] of B; the
    per-head entry bijection and its cosines come along."""
    h, k, _ = A.shape
    C = np.einsum("aid,bjd->abij", A, B)  # all head-pair entry cosines
    S = np.zeros((h, h))
    P = np.zeros((h, h, k), int)
    for a in range(h):
        for b in range(h):
            perm = assign_max(C[a, b])
            P[a, b] = perm
            S[a, b] = C[a, b][np.arange(k), perm].mean()
    hp = assign_max(S)
    sims = S[np.arange(h), hp]
    eperm = P[np.arange(h), hp]
    ecos = np.stack([C[a, hp[a]][np.arange(k), eperm[a]]
                     for a in range(h)])
    return hp, sims, eperm, ecos


def load_directions(ckpt, step, temperature):
    """acts fn + unit decode directions (l*h*k, d) + (l, h, k)."""
    from refit import onto_linear_model
    acts, G, meta = onto_linear_model(ckpt, step, temperature)
    return acts, np.asarray(G), meta["l"], meta["h"], meta["k"]


def main():
    cfg = parse_args()
    name_a, name_b = Path(cfg.a).name, Path(cfg.b).name
    out = Path(cfg.out) if cfg.out else \
        ROOT / "experiments/seed-stability/out" / f"{name_a}__{name_b}"
    out.mkdir(parents=True, exist_ok=True)

    acts_a, G_a, l, h, k = load_directions(
        cfg.a, cfg.step_a, cfg.temperature)
    acts_b, G_b, l2, h2, k2 = load_directions(
        cfg.b, cfg.step_b, cfg.temperature)
    assert (l, h, k) == (l2, h2, k2), \
        f"architecture mismatch: {(l, h, k)} vs {(l2, h2, k2)}"
    print(f"A: {name_a}  B: {name_b}  (l={l} h={h} k={k})")

    if cfg.metric == "whitened":
        w = np.load(cfg.mse_weights)
        assert G_a.shape[1] == len(w), "mse-weights dim != direction dim"
        G_a = G_a * np.sqrt(w)
        G_b = G_b * np.sqrt(w)
    La = G_a.reshape(l, h * k, -1)
    Lb = G_b.reshape(l, h * k, -1)
    if cfg.center:
        La = La - La.mean(1, keepdims=True)
        Lb = Lb - Lb.mean(1, keepdims=True)
    La = unit_rows(La)
    Lb = unit_rows(Lb)

    # pools: within-layer by default (layers are ordered residual stages)
    if cfg.cross_layer:
        pools = [(0, np.arange(l * h))]
        A_heads = La.reshape(l * h, k, -1)
        B_heads = Lb.reshape(l * h, k, -1)
    else:
        pools = [(li, np.arange(li * h, (li + 1) * h)) for li in range(l)]
        A_heads = La.reshape(l * h, k, -1)
        B_heads = Lb.reshape(l * h, k, -1)

    rng = np.random.default_rng(cfg.seed)
    head_rows = []
    head_b_of = np.zeros(l * h, int)     # global head id map A -> B
    entry_perm = np.zeros((l * h, k), int)
    entry_cos = np.zeros((l * h, k))
    null_sims = []
    for pid, hid in pools:
        A_p, B_p = A_heads[hid], B_heads[hid]
        hp, sims, eperm, ecos = pool_match(A_p, B_p)
        head_b_of[hid] = hid[hp]
        entry_perm[hid] = eperm
        entry_cos[hid] = ecos
        for pos, ga in enumerate(hid):
            head_rows.append({"pool": pid, "head_a": int(ga),
                              "head_b": int(hid[hp[pos]]),
                              "head_sim": float(sims[pos]),
                              "mean_entry_cos": float(ecos[pos].mean())})
        # null: entries regrouped into heads at random within the pool
        for _ in range(cfg.nulls):
            perm = rng.permutation(len(hid) * k)
            B_null = B_p.reshape(-1, B_p.shape[-1])[perm] \
                .reshape(len(hid), k, -1)
            _, nsims, _, _ = pool_match(A_p, B_null)
            null_sims.append(nsims)
        print(f"pool {pid}: mean head sim "
              f"{sims.mean():.3f} (min {sims.min():.3f} "
              f"max {sims.max():.3f})", flush=True)
    null_sims = np.concatenate(null_sims) if null_sims else np.zeros(0)

    all_sims = np.array([r["head_sim"] for r in head_rows])
    all_cos = entry_cos.reshape(-1)
    print(f"\nmatched-head decode-direction similarity "
          f"(mean {all_sims.mean():.3f}"
          + (f", null {null_sims.mean():.3f} "
             f"+- {null_sims.std():.3f}" if len(null_sims) else "") + ")")
    print(f"{'thr':>5} {'heads sim>=thr':>15} {'entries cos>=thr':>17}")
    geo = {}
    for thr in cfg.cos_thrs:
        fh = float((all_sims >= thr).mean())
        fe = float((all_cos >= thr).mean())
        geo[thr] = {"head_frac": fh, "entry_frac": fe}
        print(f"{thr:>5.2f} {fh:>15.3f} {fe:>17.3f}")

    # ---------- activation cross-check on the matched pairs ----------
    contain = {}
    na = nb = co = None
    if cfg.rows:
        import jax.numpy as jnp
        # flat entry ids under the composed head+entry assignment
        F = l * h * k
        ia = np.arange(F)
        ib = (head_b_of[:, None] * k + entry_perm).reshape(-1)
        mm = np.load(cfg.cache, mmap_mode="r")
        n = min(cfg.rows, mm.shape[0]) // cfg.batch * cfg.batch
        thr_fire = 2.0 / k
        na = np.zeros(F)
        nb = np.zeros(F)
        co = np.zeros(F)
        for i in range(0, n, cfg.batch):
            X = jnp.asarray(np.asarray(mm[i:i + cfg.batch],
                                       dtype=np.float32))
            Fa = np.asarray(acts_a(X)) > thr_fire
            Fb = np.asarray(acts_b(X)) > thr_fire
            Fb_m = Fb[:, ib]
            na += Fa.sum(0)
            nb += Fb_m.sum(0)
            co += (Fa & Fb_m).sum(0)
        pa = co / np.maximum(nb, 1.0)   # P(a fires | matched b fires)
        pb = co / np.maximum(na, 1.0)   # P(b fires | a fires)
        live = (na >= cfg.min_fires) & (nb >= cfg.min_fires)
        print(f"\nactivation cross-check: {n} rows, "
              f"{live.sum()}/{F} matched pairs live "
              f"(>= {cfg.min_fires} fires both sides)")
        print(f"{'tau':>5} {'matched entries':>16}   ({SAE_REFERENCE})")
        for tau in cfg.taus:
            frac = float(((pa >= tau) & (pb >= tau))[live].mean()) \
                if live.any() else float("nan")
            contain[tau] = frac
            print(f"{tau:>5.2f} {frac:>16.3f}")

    # ---------- artifacts ----------
    thr_detail = [0.5, 0.7]
    with open(out / "heads.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["pool", "head_a", "head_b", "head_sim",
                       "mean_entry_cos"]
                      + [f"frac_cos_ge_{t_}" for t_ in thr_detail]
                      + (["frac_contained_tau0.7", "live_entries"]
                         if cfg.rows else []))
        for r in head_rows:
            ga = r["head_a"]
            row = [r["pool"], ga, r["head_b"], f"{r['head_sim']:.4f}",
                   f"{r['mean_entry_cos']:.4f}"]
            row += [f"{(entry_cos[ga] >= t_).mean():.3f}"
                    for t_ in thr_detail]
            if cfg.rows:
                sl = slice(ga * k, (ga + 1) * k)
                lv = live[sl]
                frac = float(((pa[sl] >= 0.7) & (pb[sl] >= 0.7))[lv].mean()) \
                    if lv.any() else float("nan")
                row += [f"{frac:.3f}", int(lv.sum())]
            wcsv.writerow(row)

    with open(out / "entries.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["layer", "head_a", "head_b", "entry_a", "entry_b",
                       "cos"] + (["n_a", "n_b", "co", "p_a_given_b",
                                  "p_b_given_a"] if cfg.rows else []))
        for g in range(l * h):
            li = g // h
            for e in range(k):
                fa = g * k + e
                row = [li, g, int(head_b_of[g]), e, int(entry_perm[g, e]),
                       f"{entry_cos[g, e]:.4f}"]
                if cfg.rows:
                    row += [int(na[fa]), int(nb[fa]), int(co[fa]),
                            f"{pa[fa]:.3f}", f"{pb[fa]:.3f}"]
                wcsv.writerow(row)

    (out / "summary.json").write_text(json.dumps({
        "a": str(cfg.a), "b": str(cfg.b), "l": l, "h": h, "k": k,
        "metric": cfg.metric, "center": cfg.center,
        "cross_layer": cfg.cross_layer,
        "mean_head_sim": float(all_sims.mean()),
        "null_head_sim_mean": float(null_sims.mean())
            if len(null_sims) else None,
        "null_head_sim_std": float(null_sims.std())
            if len(null_sims) else None,
        "geometry_fractions": {str(t_): v for t_, v in geo.items()},
        "containment_fractions": {str(t_): v for t_, v in contain.items()},
        "containment_rows": int(cfg.rows), "min_fires": cfg.min_fires,
        "sae_reference": SAE_REFERENCE}, indent=2))
    print(f"\n-> {out / 'heads.csv'}")


if __name__ == "__main__":
    main()
