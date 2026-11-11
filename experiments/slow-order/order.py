"""Are two seeds' slow heads the same heads, one by one and in one order?

`experiments/next-sentence/seedmatch.py` scores the slow block of two
seeds by its matched NMI beside any pairing's and the NMI among one seed's
own slow heads (its `specificity`, reused here). This adds what an order
predicts and a decorrelation alone does not:

- `same_index`: the NMI between A's slow head i and B's slow head i, with
  no matching. Under an order that reproduces, head i is the same variable
  in both seeds and this equals the matched NMI; without one it equals
  any pairing's.
- `on_diagonal`: the share of the Hungarian match that pairs a head with
  its own index, and `best_is_own`, the share of A's heads whose best
  partner in B has their index. Chance for both is one over the block.
- `index_rank`: the rank correlation between a head's index and its
  matched partner's.
- `offset`: the mean NMI between two heads by how far apart their indices
  are, across the two seeds (`cross`) and inside one (`within`). If what
  heads share falls with the index, the match follows the index and
  `same_index` exceeds any pairing's with no head identified: then the
  other seed's head i (`cross` at 0) is no closer than a neighbouring
  head of the same seed (`within` at 1), which the term has pushed apart.
  A head that is identified is closer to its counterpart than to either.
- `neighbour_gap`: the same question head by head, with an error bar: a
  head's NMI with the other seed's head of its index minus the mean of the
  four entries one index away, averaged over heads.
- `kappa_trend`, `lang_trend`: the rank correlation of the head index with
  each head's Cohen's kappa between two sentences of a document `--d`
  segments apart (`transitions.kappa`) and with its NMI with the language
  label. An ordered term leaves head 0 unconstrained and gives each later
  head what the earlier ones left, so slowness should fall with the index.
- `prefix`: within one seed, the mean NMI of each head with the heads
  before it, for the first and the second half of the block.

Also the held-out whitened FVU of each run, since a term on the slow heads
is paid for there if anywhere.

    uv run python experiments/slow-order/order.py \\
        --a data/out/sonar/slow-order/ord_dec1_sp0.001_ph38_s42 \\
        --b data/out/sonar/slow-order/ord_dec1_sp0.001_ph38_s43 --step 15000
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
from jaxtyping import Float  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

from seedmatch import matching, specificity  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
WEIGHTS = ROOT / "data" / "out" / "sonar" / "sentcache" / "mse_weights_no837.npy"


def diagonal(cross: Float[np.ndarray, "s s"]) -> Dict[str, float]:
    """How far the match between two seeds' blocks follows the head index.
    `cross[i, j]` is the NMI between A's head i and B's head j."""
    s = len(cross)
    own = np.arange(s)
    m = matching(cross)
    return {"same_index": float(cross[own, own].mean()),
            "on_diagonal": float((m == own).mean()),
            "best_is_own": float((cross.argmax(1) == own).mean()),
            "index_rank": float(spearmanr(own, m)[0]),
            "chance": 1.0 / s}


def by_offset(M: Float[np.ndarray, "s s"], within: bool = False
              ) -> Dict[str, float]:
    """Mean NMI by how far apart two heads' indices are. With `within`
    (one seed against itself) offset 0 is a head with itself and is left
    out."""
    s = len(M)
    off = np.abs(np.arange(s)[:, None] - np.arange(s)[None, :])
    bands = {"0": off == 0, "1": off == 1, "2": off == 2,
             "3-5": (off >= 3) & (off <= 5), "6+": off >= 6}
    if within:
        del bands["0"]
    return {name: float(M[m].mean()) for name, m in bands.items()}


def neighbour_gap(cross: Float[np.ndarray, "s s"]) -> Dict[str, float]:
    """For each interior head, its NMI with the other seed's head of the
    same index minus the mean of the four entries one index away
    (`cross[i, i +- 1]`, `cross[i +- 1, i]`), as the mean over heads and
    its standard error. A smooth fall with the index cancels in the
    difference; an identified head leaves it positive."""
    s = len(cross)
    i = np.arange(1, s - 1)
    near = 0.25 * (cross[i, i - 1] + cross[i, i + 1]
                   + cross[i - 1, i] + cross[i + 1, i])
    gap = cross[i, i] - near
    return {"mean": float(gap.mean()),
            "se": float(gap.std(ddof=1) / np.sqrt(len(gap))),
            "positive": float((gap > 0).mean())}


def trend(v: Float[np.ndarray, "s"]) -> float:
    """Rank correlation of a per-head statistic with the head index, over
    the heads where it is defined."""
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(np.flatnonzero(ok), v[ok])[0])


def prefix(within: Float[np.ndarray, "s s"]) -> Dict[str, float]:
    """Mean NMI of each head with the heads before it, averaged over the
    first and over the second half of the block (head 0 has none)."""
    s = len(within)
    before = np.array([within[i, :i].mean() for i in range(1, s)])
    half = (s - 1) // 2
    return {"first_half": float(before[:half].mean()),
            "second_half": float(before[half:].mean())}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--slow-heads", type=int, default=38)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--mse-weights", default=str(WEIGHTS))
    p.add_argument("--holdout", type=int, default=32768,
                   help="the cache's last rows, held out from training")
    p.add_argument("--d", type=int, default=2,
                   help="segment distance of the pairs kappa is scored on "
                        "(the trained pairs are at least 2 apart)")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args()

    from eval_next import fvu, model_outputs
    from partition import codes, nmi_matrix
    from transitions import kappa, same_doc_pairs

    cache = Path(cfg.cache)
    stem = cache.with_name(cache.stem)
    arr = np.load(cache, mmap_mode="r")
    lo = len(arr) - cfg.holdout
    X = np.asarray(arr[lo:], np.float32)
    index = np.asarray(np.load(f"{stem}.index.npy", mmap_mode="r")[lo:])
    _, lang = np.unique(
        np.asarray(np.load(f"{stem}.langs.npy", mmap_mode="r")[lo:]),
        return_inverse=True)
    i, j = same_doc_pairs(index, cfg.d)
    w = np.load(cfg.mse_weights).astype(np.float64)

    res = {"a": cfg.a, "b": cfg.b, "slow_heads": cfg.slow_heads,
           "rows": cfg.holdout, "d": cfg.d, "pairs": int(len(i))}
    S, K = {}, {}
    for name, ckpt in (("a", cfg.a), ("b", cfg.b)):
        A, layer, k, step = codes(ckpt, cfg.step, cfg.temperature, X,
                                  cfg.batch)
        h = len(A) // (int(layer.max()) + 1)
        slow = A[np.arange(len(A)) % h < cfg.slow_heads]
        S[name], K[name] = slow, k
        kap = kappa(slow[:, i], slow[:, j], k)
        nl = nmi_matrix(slow, k, lang[None, :].astype(np.int64),
                        int(lang.max()) + 1)[:, 0]
        W = nmi_matrix(slow, k, slow, k)
        out = model_outputs(ckpt, step, X, 1.0, 1024)
        res[name] = {"step": int(step), "fvu": fvu(out.astype(np.float64),
                                                   X.astype(np.float64), w),
                     "kappa_mean": float(np.nanmean(kap)),
                     "kappa_trend": trend(kap), "lang_trend": trend(nl),
                     "nmi_lang_mean": float(nl.mean()),
                     "prefix": prefix(W), "kappa": kap.tolist(),
                     "nmi_lang": nl.tolist()}
        res[name + "_within"] = W.tolist()
        r = res[name]
        print(f"{Path(ckpt).name} step {step}: FVU {r['fvu']:.4f}; slow "
              f"heads' kappa_d{cfg.d} {r['kappa_mean']:.3f}, rank "
              f"correlation with the index {r['kappa_trend']:+.3f}; "
              f"NMI(lang) {r['nmi_lang_mean']:.3f}, with the index "
              f"{r['lang_trend']:+.3f}; NMI with earlier heads "
              f"{r['prefix']['first_half']:.4f} (first half) "
              f"{r['prefix']['second_half']:.4f} (second)")

    AB = nmi_matrix(S["a"], K["a"], S["b"], K["b"])
    WA = np.asarray(res.pop("a_within"))
    WB = np.asarray(res.pop("b_within"))
    res["specificity"] = specificity(WA, WB, AB)
    res["diagonal"] = diagonal(AB)
    res["offset"] = {"cross": by_offset(AB),
                     "within": by_offset(0.5 * (WA + WB), within=True)}
    res["neighbour_gap"] = neighbour_gap(AB)
    res["cross"] = AB.tolist()
    s, d = res["specificity"], res["diagonal"]
    print(f"slow block a-b: matched NMI {s['matched']:.4f}, any pairing "
          f"{s['all_pairs']:.4f}, within one seed {s['within_a']:.4f} / "
          f"{s['within_b']:.4f}; best over runner-up {s['margin']:.4f}")
    print(f"  same index {d['same_index']:.4f}; match on the diagonal "
          f"{d['on_diagonal']:.1%}, best partner is own index "
          f"{d['best_is_own']:.1%} (chance {d['chance']:.1%}); index rank "
          f"correlation {d['index_rank']:+.3f}")
    for name, v in res["offset"].items():
        print(f"  NMI by index offset, {name:>6}: " + "  ".join(
            f"{b} {x:.4f}" for b, x in v.items()))
    g = res["neighbour_gap"]
    print(f"  same index over its index neighbours: {g['mean']:+.4f} "
          f"(se {g['se']:.4f}; positive for {g['positive']:.0%} of heads)")
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
