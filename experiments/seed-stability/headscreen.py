"""Which heads reproduce across seeds as heads? A multitrait-multimethod screen.

Two seeds are two "methods" measuring the same "traits". A head of seed A
whose partition reproduces in seed B should agree with its matched partner
in B (convergent) more than with anything else:
- `floor`: more than an unrelated head would, i.e. above zero once the
  plug-in NMI's bias is removed;
- `cross_a`, `cross_b`: more than either side agrees with its runner-up
  partner in the other seed (A's head with B's second-best match for it,
  and B's head with A's);
- `mono_a`, `mono_b`: more than either side agrees with its closest other
  head in its own seed (`headnmi.py`'s within-model NMI). A head whose
  partner is no closer than its own near-copy reproduces only as a block.

Heads are matched one to one over the whole models by maximizing total NMI
(Hungarian, as `partition.py` does), and the runner-ups and closest
within-seed heads are chosen, on the first third of the rows. Every NMI is
then scored on the other two thirds, so selection does not inflate it. Each
NMI is corrected by the Miller-Madow bias of the plug-in mutual information
between independent partitions, (r_a - 1)(r_b - 1) / 2n nats for r_a and r_b
occupied cells, divided by the same mean entropy NMI divides by;
`null_check` reports the corrected NMI of a rows-shuffled pairing, which
should sit at zero. A comparison's noise is read from how much the two
scoring thirds disagree on it: for each layer of A and each comparison, the
SD of the difference between the thirds over that layer's heads, halved, is
the SD of their mean. A head passes when every comparison's mean is above
`--z` such SDs.

    JAX_PLATFORMS=cpu uv run python experiments/seed-stability/headscreen.py \\
        --a <run> --b <run>_s43 --cache data/sonar_embeddings/mc4_sent16.npy \\
        --out <dir>
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

from partition import nmi_matrix  # noqa: E402

CHECKS = ("floor", "cross_a", "cross_b", "mono_a", "mono_b")


def entropy_cells(A: Int[np.ndarray, "h n"], k: int
                  ) -> tuple[Float[np.ndarray, "h"], Int[np.ndarray, "h"]]:
    """Each head's plug-in entropy (nats) and number of occupied cells."""
    cnt = np.stack([np.bincount(a, minlength=k) for a in A]).astype(np.float64)
    p = cnt / cnt.sum(1, keepdims=True)
    H = -np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(1)
    return H, (cnt > 0).sum(1)


def corrected_nmi(A: Int[np.ndarray, "ha n"], B: Int[np.ndarray, "hb n"], k: int
                  ) -> Float[np.ndarray, "ha hb"]:
    """`nmi_matrix` less the Miller-Madow bias of independent partitions,
    on the scale NMI divides by (the mean of the two entropies)."""
    n = A.shape[1]
    Ha, ra = entropy_cells(A, k)
    Hb, rb = entropy_cells(B, k)
    bias = np.outer(ra - 1, rb - 1) / (2.0 * n)
    denom = np.maximum(0.5 * (Ha[:, None] + Hb[None, :]), 1e-12)
    return nmi_matrix(A, k, B, k).astype(np.float64) - bias / denom


def screen(A: Int[np.ndarray, "ha n"], B: Int[np.ndarray, "hb n"], k: int,
           layer_a: Int[np.ndarray, "ha"], z: float = 2.0, seed: int = 0) -> dict:
    """The screen on rows split in thirds: match and choose on the first,
    score on the other two. Returns per matched pair its heads, the five
    comparisons' means (the matched NMI itself under `floor`), their
    z-scores and whether it passes, plus the null check."""
    n = A.shape[1]
    third = n // 3
    part = [slice(i * third, (i + 1) * third) for i in range(3)]
    M = [corrected_nmi(A[:, q], B[:, q], k) for q in part]
    WA = [corrected_nmi(A[:, q], A[:, q], k) for q in part]
    WB = [corrected_nmi(B[:, q], B[:, q], k) for q in part]

    ia, ib = linear_sum_assignment(-M[0])
    M1 = M[0].copy()
    M1[ia, ib] = -np.inf
    b2 = M1[ia].argmax(1)                       # A's head's runner-up in B
    a2 = M1[:, ib].argmax(0)                    # B's head's runner-up in A
    WA1, WB1 = WA[0].copy(), WB[0].copy()
    np.fill_diagonal(WA1, -np.inf)
    np.fill_diagonal(WB1, -np.inf)
    a3 = WA1[ia].argmax(1)                      # A's head's closest in A
    b3 = WB1[ib].argmax(1)                      # B's head's closest in B

    def values(q: int) -> Float[np.ndarray, "m 5"]:
        conv = M[q][ia, ib]
        return np.stack([conv, M[q][ia, b2], M[q][a2, ib],
                         WA[q][ia, a3], WB[q][ib, b3]], 1)

    v1, v2 = values(1), values(2)
    # floor compares the match with zero; the rest with the match
    d1 = np.concatenate([v1[:, :1], v1[:, :1] - v1[:, 1:]], 1)
    d2 = np.concatenate([v2[:, :1], v2[:, :1] - v2[:, 1:]], 1)
    mean = 0.5 * (d1 + d2)
    sd = np.zeros_like(mean)
    la = layer_a[ia]
    for l in np.unique(la):
        m = la == l
        sd[m] = np.maximum((d1[m] - d2[m]).std(0), 1e-12) / 2.0
    zs = mean / sd

    rng = np.random.default_rng(seed)
    q = part[1]
    Ms = corrected_nmi(A[:, q], B[:, q][:, rng.permutation(third)], k)
    return {"a": ia, "b": ib, "values": 0.5 * (v1 + v2), "z": zs,
            "pass": (zs > z).all(1),
            "null_check": {"mean": float(Ms.mean()), "sd": float(Ms.std()),
                           "matched_mean": float(Ms[linear_sum_assignment(-Ms)].mean())}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step-a", type=int, default=0, help="0 = latest")
    p.add_argument("--step-b", type=int, default=0, help="0 = latest")
    p.add_argument("--cache", default=str(ROOT / "data/sonar_embeddings/mc4_sent16.npy"))
    p.add_argument("--rows", type=int, default=98304,
                   help="the cache's last rows, split in thirds")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--z", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0, help="the null check's shuffle")
    p.add_argument("--out", required=True)
    cfg = p.parse_args()
    from partition import codes
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    rows = min(cfg.rows, mm.shape[0]) // 3 * 3
    X = np.asarray(mm[-rows:], np.float32)
    A, la, ka, sa = codes(cfg.a, cfg.step_a, cfg.temperature, X, cfg.batch)
    B, lb, kb, sb = codes(cfg.b, cfg.step_b, cfg.temperature, X, cfg.batch)
    assert ka == kb, "the screen compares heads of one k"
    h = int((la == 0).sum())
    print(f"A {Path(cfg.a).name} step {sa}, B {Path(cfg.b).name} step {sb}: "
          f"{A.shape[0]} and {B.shape[0]} heads, k={ka}, {rows} rows", flush=True)

    r = screen(A, B, ka, la, cfg.z, cfg.seed)
    ia, ib = r["a"], r["b"]
    cols = ["layer_a", "head_a", "layer_b", "head_b"] + \
        [f"{c}" for c in ("matched",) + CHECKS[1:]] + [f"z_{c}" for c in CHECKS] + ["pass"]
    with open(out / "heads.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for j in range(len(ia)):
            f.write(",".join(
                [str(int(la[ia[j]])), str(int(ia[j] % h)), str(int(lb[ib[j]])),
                 str(int(ib[j] % h))]
                + [f"{x:.6g}" for x in r["values"][j]]
                + [f"{x:.3f}" for x in r["z"][j]] + [str(int(r["pass"][j]))]) + "\n")

    summary = {"a": cfg.a, "b": cfg.b, "step_a": sa, "step_b": sb, "rows": rows,
               "k": ka, "z": cfg.z, "null_check": r["null_check"], "layers": {}}
    lay = la[ia]
    for l in np.unique(lay):
        m = lay == l
        fails = {c: int((r["z"][m, i] <= cfg.z).sum()) for i, c in enumerate(CHECKS)}
        passed = np.flatnonzero(r["pass"] & m)
        summary["layers"][str(int(l))] = {
            "heads": int(m.sum()), "pass": int(len(passed)),
            "same_layer_partner": float((lb[ib[m]] == l).mean()),
            "matched_median": float(np.median(r["values"][m, 0])),
            "fails": fails,
            "passing_heads": [int(ia[j] % h) for j in passed]}
        s = summary["layers"][str(int(l))]
        print(f"layer {int(l)}: {s['pass']} of {s['heads']} pass; matched NMI median "
              f"{s['matched_median']:.4f}; partner in the same layer "
              f"{s['same_layer_partner']:.0%}; failing "
              + ", ".join(f"{c} {v}" for c, v in fails.items()), flush=True)
    nc = r["null_check"]
    print(f"null check (rows shuffled, corrected): mean {nc['mean']:+.5f}, "
          f"SD {nc['sd']:.5f}, matched {nc['matched_mean']:+.5f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
