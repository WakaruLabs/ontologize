"""Does a code of sentence differences reproduce across seeds?

Two difference models (`train_diff.py`, two seeds) are read on the
held-out differences `diffs.py` wrote, beside two sentence models of the
same shape and step read on the first sentence of each held-out pair
(`--sent-a`, `--sent-b`; the pair objective's `s_pair = 0` controls). For
each pair of seeds:

- `matched`: the NMI between two seeds' heads under the one-to-one match
  (Hungarian, `experiments/ste-arm/partition.py`'s score), beside `null`,
  the same match with B's rows shuffled, and `all_pairs`, the mean NMI of
  any head of A with any head of B. A code whose heads all tell the same
  thing matches well under any pairing, so the margin over `all_pairs` is
  what says the heads reproduce one by one.
- `within_lang`: the matched pairs' NMI inside each language's rows,
  beside the same with B's rows shuffled inside each language. Agreement
  that only restates the language scores at that null.

For each model: the mean and largest NMI of a head with the document's
language beside the mean with the labels shuffled (the NMI's bias at this
many rows), each head's entropy, and the held-out whitened FVU. The
sentence models' FVU of the difference is that of x_hat_a - x_hat_b, the
two sentences' reconstructions, under the differences' weights: what a
sentence code already gives for a difference, at twice the heads.

For each difference model, `antipodal`: how its code of x_b - x_a relates
to its code of x_a - x_b. `keep` is the share of heads with the same
entry on both, beside the chance its entry usage gives; `nmi` is the NMI
between the two codes head by head. A head that reads the difference's
direction keeps nothing and has high NMI (each entry has an opposite); a
head that reads only its size or its axis keeps everything.

    uv run python experiments/pair-diff/evaldiff.py \\
        --a data/out/sonar/pair-diff/diff_h380_l1_s42 \\
        --b data/out/sonar/pair-diff/diff_h380_l1_s43 --step 15000
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
from jaxtyping import Float, Int  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))

from seedmatch import shuffle_within, within_label_nmi  # noqa: E402

from diffs import CACHE, OUT, WEIGHTS  # noqa: E402

LONG = ROOT / "data" / "out" / "sonar" / "next-sentence" / "long"


def seed_match(M: Float[np.ndarray, "na nb"], Mn: Float[np.ndarray, "na nb"]
               ) -> Dict[str, float]:
    """Matched NMI of the one-to-one match of `M`, the same for the
    row-shuffled `Mn`, and the mean and the median best-over-runner-up
    margin of `M`."""
    r, c = linear_sum_assignment(-M)
    rn, cn = linear_sum_assignment(-Mn)
    top = np.sort(M, axis=1)[:, ::-1]
    return {"heads": int(len(r)), "matched": float(M[r, c].mean()),
            "median": float(np.median(M[r, c])),
            "null": float(Mn[rn, cn].mean()), "all_pairs": float(M.mean()),
            "margin": float(np.median(top[:, 0] - top[:, 1]))}


def chance_keep(A: Int[np.ndarray, "heads n"], B: Int[np.ndarray, "heads n"],
                k: int) -> Float[np.ndarray, "heads"]:
    """The agreement two codes' entry frequencies give by chance, per
    head."""
    heads, n = A.shape
    off = (np.arange(heads, dtype=np.int64) * k)[:, None]
    pa = np.bincount((A + off).ravel(), minlength=heads * k).reshape(heads, k)
    pb = np.bincount((B + off).ravel(), minlength=heads * k).reshape(heads, k)
    return (pa * pb).sum(1) / n ** 2


def entropy_bits(A: Int[np.ndarray, "heads n"], k: int
                 ) -> Float[np.ndarray, "heads"]:
    """Entropy of each head's entry usage, in bits."""
    heads, n = A.shape
    off = (np.arange(heads, dtype=np.int64) * k)[:, None]
    p = np.bincount((A + off).ravel(), minlength=heads * k).reshape(heads, k)
    p = p / n
    return -np.where(p > 0, p * np.log2(np.maximum(p, 1e-30)), 0.0).sum(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="difference model, one seed")
    p.add_argument("--b", required=True, help="difference model, another")
    p.add_argument("--sent-a", default=str(LONG / "doc2_sp0_ph38"),
                   help="sentence model of the same shape, one seed")
    p.add_argument("--sent-b", default=str(LONG / "doc2_sp0_ph38_s43"))
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--data", default=str(OUT),
                   help="where diffs.py wrote tail.npy and its sidecars")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--sent-weights", default=str(WEIGHTS))
    p.add_argument("--min-rows", type=int, default=200,
                   help="the fewest rows a language needs to count in "
                        "within_lang")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="the nulls' shuffles")
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args()

    from eval_next import fvu, model_outputs
    from partition import codes, nmi_matrix

    data = Path(cfg.data)
    D = np.load(data / "tail.npy")
    rows = np.load(data / "tail_rows.npy")
    names, lang = np.unique(np.load(data / "tail_langs.npy"),
                            return_inverse=True)
    arr = np.load(cfg.cache, mmap_mode="r")
    Xa = np.asarray(arr[np.sort(rows[:, 0])], np.float32)[
        np.argsort(np.argsort(rows[:, 0]))]
    Xb = np.asarray(arr[np.sort(rows[:, 1])], np.float32)[
        np.argsort(np.argsort(rows[:, 1]))]
    assert np.allclose(Xa - Xb, D, atol=1e-6), "tail.npy is not x_a - x_b"
    wd = np.load(data / "mse_weights_diff.npy").astype(np.float64)
    ws = np.load(cfg.sent_weights).astype(np.float64)
    lang_row = lang[None, :].astype(np.int64)
    n_lang = len(names)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(D))
    print(f"{len(D)} held-out pairs, {n_lang} languages")

    res = {"pairs": int(len(D)), "step": cfg.step}
    for kind, X, w, (ca, cb) in (
            ("diff", D, wd, (cfg.a, cfg.b)),
            ("sentence", Xa, ws, (cfg.sent_a, cfg.sent_b))):
        C = {}
        for name, ckpt in (("a", ca), ("b", cb)):
            A, _, k, step = codes(ckpt, cfg.step, cfg.temperature, X,
                                  cfg.batch)
            C[name] = A
            nl = nmi_matrix(A, k, lang_row, n_lang)[:, 0]
            nl0 = nmi_matrix(A, k, lang_row[:, perm], n_lang)[:, 0]
            ent = entropy_bits(A, k)
            out = model_outputs(ckpt, step, X, 1.0, 1024).astype(np.float64)
            m = {"ckpt": ckpt, "step": int(step), "k": int(k),
                 "fvu": fvu(out, X.astype(np.float64), w),
                 "nmi_lang_mean": float(nl.mean()),
                 "nmi_lang_max": float(nl.max()),
                 "nmi_lang_null": float(nl0.mean()),
                 "entropy_bits_mean": float(ent.mean()),
                 "heads_below_1_bit": int((ent < 1.0).sum())}
            if kind == "sentence":
                outb = model_outputs(ckpt, step, Xb, 1.0, 1024)
                m["fvu_of_difference"] = fvu(out - outb.astype(np.float64),
                                             D.astype(np.float64), wd)
            else:
                N = codes(ckpt, cfg.step, cfg.temperature, -X, cfg.batch)[0]
                na = np.array([float(nmi_matrix(A[r:r + 1], k, N[r:r + 1],
                                                k)[0, 0])
                               for r in range(len(A))])
                m["antipodal"] = {
                    "keep": float((A == N).mean()),
                    "keep_chance": float(chance_keep(A, N, k).mean()),
                    "nmi": float(na.mean()),
                    "nmi_null": float(nmi_matrix(
                        A[:32], k, N[:32][:, perm], k).diagonal().mean())}
            res[f"{kind}_{name}"] = m
            print(f"{kind} {Path(ckpt).name} step {step}: FVU {m['fvu']:.4f}"
                  + (f" (of the difference, from two reconstructions "
                     f"{m['fvu_of_difference']:.4f})"
                     if kind == "sentence" else "")
                  + f"; NMI(lang) mean {m['nmi_lang_mean']:.4f} max "
                  f"{m['nmi_lang_max']:.3f} (labels shuffled "
                  f"{m['nmi_lang_null']:.4f}); entropy "
                  f"{m['entropy_bits_mean']:.2f} bits, "
                  f"{m['heads_below_1_bit']} heads under 1 bit")
            if kind == "diff":
                a = m["antipodal"]
                print(f"  code of -x: same entry {a['keep']:.3f} (chance "
                      f"{a['keep_chance']:.3f}); NMI with the code of x "
                      f"{a['nmi']:.4f} (rows shuffled {a['nmi_null']:.4f})")
        A, B = C["a"], C["b"]
        M = nmi_matrix(A, k, B, k)
        Mn = nmi_matrix(A, k, B[:, perm], k)
        s = seed_match(M, Mn)
        r, c = linear_sum_assignment(-M)
        v = np.array([within_label_nmi(A[i], B[j], lang, k, cfg.min_rows)
                      for i, j in zip(r, c)])
        vn = np.array([within_label_nmi(A[i], shuffle_within(B[j], lang, rng),
                                        lang, k, cfg.min_rows)
                       for i, j in zip(r, c)])
        s["within_lang"] = float(v.mean())
        s["within_lang_null"] = float(vn.mean())
        res[f"{kind}_match"] = s
        print(f"{kind} seeds: matched NMI {s['matched']:.4f} (median "
              f"{s['median']:.4f}); rows shuffled {s['null']:.4f}; any "
              f"pairing {s['all_pairs']:.4f}; best over runner-up "
              f"{s['margin']:.4f}; within language {s['within_lang']:.4f} "
              f"(shuffled within {s['within_lang_null']:.4f})")
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
