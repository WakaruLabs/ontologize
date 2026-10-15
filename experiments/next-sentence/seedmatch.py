"""Do two seeds' slow heads learn the same partitions?

`experiments/ste-arm/partition.py` matches every head of one model to one
of another by the NMI between their partitions of the same rows
(Hungarian), against a null with B's rows shuffled. This splits that match
by block:
- `slow`: A's slow heads (the first `--slow-heads` of each layer) against
  B's slow heads;
- `rest`: the other heads against the other heads;
- `all`: the whole model, as `partition.py` scores it;
- `slow_to_slow`: of A's slow heads, the share that the whole-model
  assignment pairs with one of B's slow heads. Chance is B's slow share.

A pair trained with the pair term should match better in `slow` than the
control's first `--slow-heads` heads, which nothing distinguishes.

`--labels` (a per-row label array aligned with the cache, such as
`mc4_sent16.langs.npy`) adds `slow_within_label`. For the `slow` block's
matched pairs it is the NMI computed inside each label's rows, averaged
weighted by rows, beside a null that shuffles B's rows within each label.
Two heads that agree only because both track the label score near that
null.

    uv run python experiments/next-sentence/seedmatch.py \\
        --a data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38 \\
        --b data/out/sonar/next-sentence/long/doc2_fano_sp1e-3_ph38_s43
"""
import os

# before jax is imported (partition.py sets the same): on demand, so the
# card stays shared
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Bool, Float, Int  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"


def slow_mask(n_heads: int, h: int, slow_heads: int) -> Bool[np.ndarray, "n"]:
    """Which of `n_heads` layer-major heads (index = layer * h + head) are
    slow: the first `slow_heads` of each layer."""
    return np.arange(n_heads) % h < slow_heads


def block_match(M: Float[np.ndarray, "na nb"], rows: Int[np.ndarray, "p"],
                cols: Int[np.ndarray, "q"]) -> Float[np.ndarray, "m"]:
    """Matched NMI of the one-to-one assignment (Hungarian, maximizing total
    NMI) between A's heads `rows` and B's heads `cols`."""
    sub = M[np.ix_(rows, cols)]
    r, c = linear_sum_assignment(-sub)
    return sub[r, c]


def slow_to_slow(M: Float[np.ndarray, "na nb"], slow_a: Bool[np.ndarray, "na"],
                 slow_b: Bool[np.ndarray, "nb"]) -> float:
    """Of A's slow heads, the share that the whole-model assignment pairs
    with one of B's slow heads."""
    r, c = linear_sum_assignment(-M)
    mine = slow_a[r]
    return float(slow_b[c[mine]].mean())


def nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"], k: int) -> float:
    """NMI between two partitions of the same rows into k cells each,
    normalized by the mean of the two entropies, as `partition.py` does."""
    J = np.zeros((k, k))
    np.add.at(J, (a, b), 1.0)
    J /= len(a)
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    mi = float((J[nz] * np.log(J[nz] / np.outer(pa, pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / max(0.5 * (ha + hb), 1e-12)


def within_label_nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"],
                     labels: np.ndarray, k: int, min_rows: int = 200) -> float:
    """NMI computed inside each label's rows, averaged weighted by rows,
    over labels with at least `min_rows` rows."""
    total, weight = 0.0, 0
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() >= min_rows:
            total += m.sum() * nmi(a[m], b[m], k)
            weight += int(m.sum())
    return total / max(weight, 1)


def shuffle_within(b: Int[np.ndarray, "n"], labels: np.ndarray,
                   rng: np.random.Generator) -> Int[np.ndarray, "n"]:
    """`b` with its rows permuted inside each label, which keeps every
    label's marginal of b and breaks its pairing with any other code."""
    out = b.copy()
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        out[idx] = b[rng.permutation(idx)]
    return out


def summarize(v: Float[np.ndarray, "m"], vn: Float[np.ndarray, "m"]
              ) -> Dict[str, float]:
    """Matched NMI and the row-shuffled null of one block."""
    return {"heads": int(len(v)), "matched": float(v.mean()),
            "median": float(np.median(v)), "max": float(v.max()),
            "null": float(vn.mean())}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step-a", type=int, default=0, help="0 = latest")
    p.add_argument("--step-b", type=int, default=0, help="0 = latest")
    p.add_argument("--slow-heads", type=int, default=38,
                   help="slow heads per layer, the first that many")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--rows", type=int, default=32768,
                   help="the cache's last ROWS rows, held out from training")
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="classifier temperature; under `ste` the code is "
                        "an argmax and this does not matter")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="the null's shuffle")
    p.add_argument("--labels", default=None,
                   help="per-row labels aligned with the cache (e.g. "
                        "mc4_sent16.langs.npy): adds the slow pairs' NMI "
                        "within each label")
    p.add_argument("--min-rows", type=int, default=200,
                   help="--labels: the fewest rows a label needs to count")
    p.add_argument("--label-rows", type=int, default=0,
                   help="--labels: score the slow pairs on the cache's last "
                        "LABEL_ROWS rows instead of --rows. NMI over k "
                        "entries is biased upward on a few hundred rows a "
                        "label, so this needs more rows than the matching "
                        "does; every row past the training prefix is "
                        "held out")
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args()

    sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
    from partition import codes, nmi_matrix

    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    A, la, ka, sa = codes(cfg.a, cfg.step_a, cfg.temperature, X, cfg.batch)
    B, lb, kb, sb = codes(cfg.b, cfg.step_b, cfg.temperature, X, cfg.batch)
    ha = len(A) // (int(la.max()) + 1)
    hb = len(B) // (int(lb.max()) + 1)
    slow_a = slow_mask(len(A), ha, cfg.slow_heads)
    slow_b = slow_mask(len(B), hb, cfg.slow_heads)

    M = nmi_matrix(A, ka, B, kb)
    perm = np.random.default_rng(cfg.seed).permutation(B.shape[1])
    Mn = nmi_matrix(A, ka, B[:, perm], kb)

    blocks = {"slow": (np.flatnonzero(slow_a), np.flatnonzero(slow_b)),
              "rest": (np.flatnonzero(~slow_a), np.flatnonzero(~slow_b)),
              "all": (np.arange(len(A)), np.arange(len(B)))}
    res = {"a": cfg.a, "b": cfg.b, "step_a": int(sa), "step_b": int(sb),
           "rows": cfg.rows, "slow_heads": cfg.slow_heads}
    print(f"A {Path(cfg.a).name} step {sa}; B {Path(cfg.b).name} step {sb}; "
          f"{cfg.rows} rows")
    for name, (ra, cb) in blocks.items():
        res[name] = summarize(block_match(M, ra, cb), block_match(Mn, ra, cb))
        s = res[name]
        print(f"  {name:<5} {s['heads']:>4} heads: matched NMI {s['matched']:.4f} "
              f"(median {s['median']:.4f}, max {s['max']:.4f}); "
              f"null {s['null']:.4f}")
    res["slow_to_slow"] = slow_to_slow(M, slow_a, slow_b)
    res["slow_to_slow_chance"] = float(slow_b.mean())
    print(f"  A's slow heads paired with B's slow heads: "
          f"{res['slow_to_slow']:.1%} (chance {res['slow_to_slow_chance']:.1%})")

    if cfg.labels:
        assert ka == kb, "within-label NMI needs one k"
        n = cfg.label_rows or cfg.rows
        labels = np.asarray(np.load(cfg.labels, mmap_mode="r")[-n:])
        if n != cfg.rows:
            XL = np.asarray(np.load(cfg.cache, mmap_mode="r")[-n:], np.float32)
            AL = codes(cfg.a, cfg.step_a, cfg.temperature, XL, cfg.batch)[0]
            BL = codes(cfg.b, cfg.step_b, cfg.temperature, XL, cfg.batch)[0]
        else:
            AL, BL = A, B
        ra, cb = blocks["slow"]
        r, c = linear_sum_assignment(-M[np.ix_(ra, cb)])
        rng = np.random.default_rng(cfg.seed)
        v = np.array([within_label_nmi(AL[i], BL[j], labels, ka, cfg.min_rows)
                      for i, j in zip(ra[r], cb[c])])
        vn = np.array([within_label_nmi(
            AL[i], shuffle_within(BL[j], labels, rng), labels, ka,
            cfg.min_rows) for i, j in zip(ra[r], cb[c])])
        kept = [lab for lab in np.unique(labels)
                if (labels == lab).sum() >= cfg.min_rows]
        res["slow_within_label"] = dict(summarize(v, vn), labels=len(kept),
                                        rows=int(np.isin(labels, kept).sum()))
        s = res["slow_within_label"]
        print(f"  slow pairs within label ({s['labels']} labels, {s['rows']} "
              f"rows): NMI {s['matched']:.4f} (median {s['median']:.4f}); "
              f"null {s['null']:.4f}")
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
