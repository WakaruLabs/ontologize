"""Seed stability of head partitions.

Two runs of the same config with different seeds each assign every held-out
token to one label per head (argmax). A head is a partition of the tokens.
Heads are matched by their partitions, never by index (head order is
arbitrary across seeds): a one-to-one Hungarian matching on the matrix of
normalized mutual information (NMI, arithmetic normalization, in [0, 1])
between all head pairs within a layer, plus each head's best match over all
heads of all layers. Reported against two nulls, both
Hungarian-matched like the statistic: the best one-to-one matching of run A's
heads to *other* heads of run A (how much heads overlap anyway) and to a
row-shuffled run B (chance).

For matched pairs, also report how one-to-one the label correspondence is:
the fraction of the joint mass captured by the best one-to-one label matching
(Hungarian assignment on the contingency table). 1.0 means run B's head is a
relabeling of run A's head; 1/k means labels are unrelated.

  uv run python experiments/gpt2/seedstab.py data/out/gpt2/V_ste_T1 data/out/gpt2/V_ste_T1_s43
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
from common import Cache  # noqa: E402
import diagnose  # noqa: E402


def assignments(run, X, bs=4096, step=None):
    """Hard label per (token, layer, head): (N, l, h) int array."""
    model, params, step = diagnose.load(run, step)
    rc = json.loads((Path(run) / "config.json").read_text())
    f = min(step / rc["anneal_steps"], 1.0) if rc["anneal_steps"] else 1.0
    T_end = rc.get("temperature_end") or rc["temperature"]
    T = rc["temperature"] * (T_end / rc["temperature"]) ** f

    @jax.jit
    def hard(p, x):
        _, layers = model.apply(p, x, T, method=diagnose.forward_probe)
        return jnp.stack([jnp.argmax(K, -1) for K, *_ in layers], 1)

    A = np.concatenate([np.asarray(hard(params, jnp.asarray(X[i:i + bs])))
                        for i in range(0, len(X), bs)])
    return A.astype(np.int32), model.k, step


def entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def nmi_matrix(A, B, k):
    """NMI between every column of A (N, m) and every column of B (N, n)."""
    N = A.shape[0]
    HA = np.array([entropy(np.bincount(a, minlength=k)) for a in A.T])
    HB = np.array([entropy(np.bincount(b, minlength=k)) for b in B.T])
    out = np.zeros((A.shape[1], B.shape[1]))
    for i, a in enumerate(A.T):
        joint = np.stack([np.bincount(a * k + b, minlength=k * k) for b in B.T])  # (n, k*k)
        Hj = np.array([entropy(row) for row in joint])
        I = HA[i] + HB - Hj
        out[i] = np.maximum(I, 0.0) / (0.5 * (HA[i] + HB) + 1e-12)
    return out


def one_to_one(a, b, k):
    """Fraction of tokens on the best one-to-one label matching a -> b."""
    C = np.bincount(a * k + b, minlength=k * k).reshape(k, k)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].sum() / C.sum())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--cache", default="data/gpt2_l8")
    p.add_argument("--rows", type=int, default=65536)
    p.add_argument("--out", default=None)
    cfg = p.parse_args()
    X = Cache(cfg.cache).eval_rows(cfg.rows)
    A, k, sa = assignments(cfg.run_a, X)
    B, _, sb = assignments(cfg.run_b, X)
    N, l, h = A.shape
    rng = np.random.default_rng(0)
    res = {"runs": [cfg.run_a, cfg.run_b], "steps": [sa, sb], "rows": N, "layers": []}
    # heads are matched by their partitions, never by index: Hungarian
    # (one-to-one) matching on the NMI matrix within a layer, plus the
    # unconstrained best match over all heads of all layers
    Ball = B.reshape(N, l * h)
    print(f"{'layer':>5} {'hungarian NMI':>13} {'median':>7} {'frac>0.5':>9} {'best any-layer':>14} "
          f"{'labels 1to1':>12} | {'within(hung)':>12} {'shuffle(hung)':>13} {'dead A/B':>9}")
    for li in range(l):
        a, b = A[:, li], B[:, li]                       # (N, h) each
        M = nmi_matrix(a, b, k)                         # (h, h) across seeds
        r, c = linear_sum_assignment(-M)                # one-to-one head matching
        hung = M[r, c]
        anyl = nmi_matrix(a, Ball, k).max(1)            # best match in any layer of B
        # nulls matched the same way as the statistic (Hungarian selects the best
        # permutation, which is biased upward relative to a grand mean): the best
        # one-to-one matching of run A's heads to *other* heads of run A, and to
        # a row-shuffled run B
        W = nmi_matrix(a, a, k); np.fill_diagonal(W, -1.0)
        wr, wc = linear_sum_assignment(-W); within_h = W[wr, wc]
        np.fill_diagonal(W, np.nan)
        o2o = float(np.mean([one_to_one(a[:, i], b[:, j], k) for i, j in zip(r, c)]))
        Ms = nmi_matrix(a, b[rng.permutation(N)], k)
        sr, sc = linear_sum_assignment(-Ms); shuf = Ms[sr, sc].mean()
        dead = [int((np.array([len(np.unique(a[:, i])) for i in range(h)]) == 1).sum()),
                int((np.array([len(np.unique(b[:, i])) for i in range(h)]) == 1).sum())]
        row = {"layer": li, "hungarian_nmi_mean": float(hung.mean()),
               "hungarian_nmi_median": float(np.median(hung)), "frac_gt_0.5": float((hung > 0.5).mean()),
               "best_any_layer_mean": float(anyl.mean()), "labels_one_to_one": o2o,
               "within_null": float(within_h.mean()), "within_null_mean_all": float(np.nanmean(W)),
               "within_null_max": float(np.nanmax(W)),
               "shuffle_null": float(shuf), "dead_heads": dead,
               "hungarian_pairs": [[int(i), int(j), float(M[i, j])] for i, j in zip(r, c)]}
        res["layers"].append(row)
        print(f"{li:5d} {row['hungarian_nmi_mean']:13.3f} {row['hungarian_nmi_median']:7.3f} "
              f"{row['frac_gt_0.5']:9.2f} {row['best_any_layer_mean']:14.3f} {o2o:12.3f} | "
              f"{row['within_null']:12.3f} {shuf:13.3f} {dead[0]:>5}/{dead[1]}")
    out = cfg.out or (Path(cfg.run_b) / "seedstab.json")
    Path(out).write_text(json.dumps(res, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
