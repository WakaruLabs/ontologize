"""Do two seeds' slow blocks agree mode by mode, and does a slow head's match
across seeds fall with its redundancy inside its own seed?

The slow heads reproduce as a block but not head by head (`seedmatch.py`).
Sharma et al.'s soft-mode reading of that is a stiff span with soft
heads inside it: the block's leading modes reproduce and later ones less,
and a head that its seed's other heads already predict is free to differ
between seeds.

Modes. A block's code is its heads' one-hot assignments over the held-out
tail (`partition.codes`), centered. Its modes are the eigenvectors of that
code's covariance, fitted on the first half of the rows, in eigenvalue
order; a mode's score is the code projected on it. For each of A's first
--modes modes, a ridge map from B's block code to the score is fitted on
the first half and scored (R^2) on the second. Two nulls take B's block's
place: B's next block of as many heads (an ordinary block of the same
model), and B's block with its rows shuffled. The same is done from B to
A.

Heads. For each of A's block heads:
  vif      held-out R^2 of the head's one-hot from the block's other heads
           (ridge), the multivariate analogue of a variance inflation factor
  nmi_in   its largest NMI with any other head of A
  matched  NMI with its partner in B's block (Hungarian on NMI, as
           seedmatch.py)
  margin   its best NMI into B's block minus its runner-up's
with Spearman correlations of vif and nmi_in against matched and margin.

  uv run python experiments/next-sentence/modeagree.py \\
      --a data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38 \\
      --b data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38_s43

Writes <out>/modes.csv, <out>/heads.csv and <out>/summary.json.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def onehot(A: Int[np.ndarray, "heads n"], k: int) -> Float[np.ndarray, "n hk"]:
    """Heads' assignments as one block of one-hot columns, head-major."""
    h, n = A.shape
    Z = np.zeros((n, h * k), np.float32)
    Z[np.arange(n)[None, :], A + (np.arange(h) * k)[:, None]] = 1.0
    return Z


def ridge(Z: Float[np.ndarray, "n p"], Y: Float[np.ndarray, "n q"],
          lam: float):
    """Ridge map fitted on centered Z and Y; returns predict(Z')."""
    mz, my = Z.mean(0), Y.mean(0)
    Zc = (Z - mz).astype(np.float64)
    G = Zc.T @ Zc
    G[np.diag_indices_from(G)] += lam * np.trace(G) / len(G)
    W = np.linalg.solve(G, Zc.T @ (Y - my))
    return lambda Zq: (Zq - mz) @ W + my


def r2(Y: Float[np.ndarray, "n q"], Yhat: Float[np.ndarray, "n q"]
       ) -> Float[np.ndarray, "q"]:
    """Per column, 1 - SSE / SST about the column's own mean."""
    return 1.0 - ((Y - Yhat) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def modes(Z: Float[np.ndarray, "n p"], m: int):
    """(mean, the top-m covariance eigenvectors as columns, their share of
    the total variance)."""
    mu = Z.mean(0)
    C = np.cov((Z - mu).astype(np.float64), rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    return mu, V[:, :m], lam[:m] / lam.sum()


def mode_agreement(ZA, ZB, ZB_other, half: int, m: int, lam: float, seed: int):
    """Held-out R^2 of A's first m mode scores from B's block, from B's
    other block, and from B's block with rows shuffled."""
    mu, V, share = modes(ZA[:half], m)
    S = (ZA - mu) @ V
    perm = np.random.default_rng(seed).permutation(len(ZB))
    out = {"share": share}
    for name, Zp in (("block", ZB), ("other", ZB_other), ("shuffled", ZB[perm])):
        f = ridge(Zp[:half], S[:half], lam)
        out[name] = r2(S[half:], f(Zp[half:]))
    return out


def head_vif(Z: Float[np.ndarray, "n hk"], h: int, k: int, half: int,
             lam: float) -> Float[np.ndarray, "h"]:
    """Per head, held-out R^2 of its one-hot from the other heads'."""
    out = np.empty(h)
    for i in range(h):
        cols = np.arange(i * k, (i + 1) * k)
        rest = np.setdiff1d(np.arange(h * k), cols)
        f = ridge(Z[:half][:, rest], Z[:half][:, cols], lam)
        Y, Yhat = Z[half:][:, cols], f(Z[half:][:, rest])
        out[i] = 1.0 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    return out


def best_margin(M: Float[np.ndarray, "a b"]) -> Float[np.ndarray, "a"]:
    """Each row's best entry minus its second best."""
    s = np.sort(M, axis=1)
    return s[:, -1] - s[:, -2]


def spearman(x: np.ndarray, y: np.ndarray, perms: int, seed: int):
    """Spearman's rho and a two-sided permutation p-value."""
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1]
                     for _ in range(perms)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--step-a", type=int, default=0, help="0 = latest")
    p.add_argument("--step-b", type=int, default=0, help="0 = latest")
    p.add_argument("--cache", default=str(ROOT / "data" / "sonar_embeddings"
                                          / "mc4_sent16.npy"))
    p.add_argument("--rows", type=int, default=32768,
                   help="the cache's last ROWS rows, held out from training")
    p.add_argument("--slow-heads", type=int, default=38)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--modes", type=int, default=64)
    p.add_argument("--lam", type=float, default=1e-3,
                   help="ridge, as a share of the Gram's mean diagonal")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--perms", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    cfg = p.parse_args()

    sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
    from partition import codes, nmi_matrix
    from scipy.optimize import linear_sum_assignment

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    A, la, k, sa = codes(cfg.a, cfg.step_a, cfg.temperature, X, cfg.batch)
    B, lb, kb, sb = codes(cfg.b, cfg.step_b, cfg.temperature, X, cfg.batch)
    assert k == kb and (la == 0).all() and (lb == 0).all(), "one-layer runs"
    s, half = cfg.slow_heads, len(X) // 2
    ZA, ZB = onehot(A[:s], k), onehot(B[:s], k)
    ZB_other = onehot(B[s:2 * s], k)
    print(f"A {Path(cfg.a).name} step {sa}; B {Path(cfg.b).name} step {sb}; "
          f"{len(X)} rows, fit on the first {half}")

    rows, summary = [], {"a": cfg.a, "b": cfg.b, "step_a": int(sa),
                         "step_b": int(sb), "rows": len(X), "slow_heads": s}
    for direction, (Z1, Z2, Z2o) in (("a_from_b", (ZA, ZB, ZB_other)),
                                     ("b_from_a", (ZB, ZA, onehot(A[s:2 * s], k)))):
        res = mode_agreement(Z1, Z2, Z2o, half, cfg.modes, cfg.lam, cfg.seed)
        for i in range(cfg.modes):
            rows.append([direction, i + 1, f"{res['share'][i]:.5f}",
                         f"{res['block'][i]:.4f}", f"{res['other'][i]:.4f}",
                         f"{res['shuffled'][i]:.4f}"])
        bands = {"1-4": slice(0, 4), "5-16": slice(4, 16),
                 "17-64": slice(16, cfg.modes)}
        summary[direction] = {
            name: {key: float(res[key][sl].mean())
                   for key in ("share", "block", "other", "shuffled")}
            for name, sl in bands.items()}
        print(f"{direction}: mean held-out R^2 of the mode scores")
        for name, v in summary[direction].items():
            print(f"  modes {name:>5}: block {v['block']:.3f}, other block "
                  f"{v['other']:.3f}, shuffled {v['shuffled']:.3f} "
                  f"(variance share per mode {v['share']:.4f})")
    with open(out / "modes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction", "mode", "share", "r2_block", "r2_other",
                    "r2_shuffled"])
        w.writerows(rows)

    vif = head_vif(ZA, s, k, half, cfg.lam)
    W = nmi_matrix(A, k, A, k)
    np.fill_diagonal(W, -np.inf)
    nmi_in = W[:s].max(1)
    M = nmi_matrix(A[:s], k, B[:s], k)
    r, c = linear_sum_assignment(-M)
    matched = np.empty(s)
    matched[r] = M[r, c]
    margin = best_margin(M)
    with open(out / "heads.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["head", "vif_r2", "nmi_in", "matched", "margin"])
        for i in range(s):
            wr.writerow([i, f"{vif[i]:.4f}", f"{nmi_in[i]:.4f}",
                         f"{matched[i]:.4f}", f"{margin[i]:.4f}"])
    corr = {}
    for xn, x in (("vif_r2", vif), ("nmi_in", nmi_in)):
        for yn, y in (("matched", matched), ("margin", margin)):
            corr[f"{xn}~{yn}"] = spearman(x, y, cfg.perms, cfg.seed)
    summary["heads"] = {
        "vif_r2": [float(vif.min()), float(np.median(vif)), float(vif.max())],
        "nmi_in": [float(nmi_in.min()), float(np.median(nmi_in)), float(nmi_in.max())],
        "matched": [float(matched.min()), float(np.median(matched)), float(matched.max())],
        "margin": [float(margin.min()), float(np.median(margin)), float(margin.max())],
        "spearman": {key: {"rho": v[0], "p": v[1]} for key, v in corr.items()}}
    print("heads (min / median / max):")
    for key in ("vif_r2", "nmi_in", "matched", "margin"):
        print(f"  {key:>8}: " + " / ".join(f"{v:.3f}" for v in summary["heads"][key]))
    for key, v in corr.items():
        print(f"  Spearman {key}: rho {v[0]:+.3f} (permutation p {v[1]:.3f})")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
