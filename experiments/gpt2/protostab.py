"""Seed stability below the head level: prototypes and co-assignment.

`seedstab.py` asks whether the *head partitions* of two seeds correspond.
They mostly don't. This asks two weaker questions:

1. Prototype matching. Are the dictionary entries (steering directions, for
   the `direct` design) reproducible across seeds regardless of which head
   they sit in? Hungarian one-to-one matching of the l*h*k entries of a layer
   by cosine, against (a) entries of the other run's *other* layers and (b)
   random directions. For matched entries, "head purity": the largest
   fraction of a run-A head's 32 matches that land in one run-B head (about
   0.11 under a random matching, printed as the null; 1 = the head is
   reproduced entry for entry).

2. Co-assignment kernel. For a sample of tokens, K(x, y) = fraction of heads
   that give x and y the same label. Compare the two seeds' kernels
   (linear CKA, i.e. double-centered, so a token's overall co-assignment
   rate does not count as agreement) against each run's own split-half
   value (heads 0..h/2 vs h/2..h), a row-shuffled B (chance) and the same
   run at an earlier checkpoint (pass `run:step` as run_b) as the ceiling.

3. (--lens) Functional matching by logit lens: top-20 promoted vocabulary
   items of each entry (unembedding of the entry direction through ln_f's
   affine part), Jaccard between matched pairs vs random pairs. A crude proxy
   for "same feature" that ignores blocks 8..11.

  uv run python experiments/gpt2/protostab.py data/out/gpt2/V_direct_gain_ste data/out/gpt2/V_direct_gain_ste_s43 --lens
  # within-run ceiling (same run, earlier checkpoint):
  uv run python experiments/gpt2/protostab.py data/out/gpt2/V_direct_gain_ste data/out/gpt2/V_direct_gain_ste:30000 --out /tmp/x.json
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
from common import Cache  # noqa: E402
import diagnose  # noqa: E402
import seedstab  # noqa: E402


def split_step(run):
    """'path' or 'path:step' -> (path, step or None)."""
    if ":" in run and run.rsplit(":", 1)[1].isdigit():
        path, step = run.rsplit(":", 1)
        return path, int(step)
    return run, None


def entries(run):
    """(l, h, k, d) dictionary entries in activation space (direct designs only)."""
    model, params, step = diagnose.load(*split_step(run))
    assert model.direct and model.signed_dict and not model.private_heads, \
        "entries must live in activation space (direct, signed, not private)"
    W = np.stack([np.asarray(params["params"][f"dictencs_{i}"]["dict"]["weights"]) for i in range(model.l)])
    return W, model, params, step


def unit(W):
    return W / (np.linalg.norm(W, axis=-1, keepdims=True) + 1e-12)


def hungarian_cos(A, B):
    """Mean matched cosine and the matching for unit rows A (n, d), B (m, d)."""
    C = A @ B.T
    r, c = linear_sum_assignment(-C)
    return C[r, c], r, c, C


def head_purity(c, h, k):
    """For each run-A head, largest fraction of its k matches landing in one run-B head."""
    out = []
    for hh in range(h):
        heads_b = c[hh * k:(hh + 1) * k] // k
        out.append(np.bincount(heads_b, minlength=h).max() / k)
    return float(np.mean(out)), float(np.max(out))


def coassign_kernel(A, heads):
    """A: (N, h) labels. K[x, y] = fraction of `heads` assigning x and y the same label."""
    N = A.shape[0]
    K = np.zeros((N, N), np.float32)
    for j in heads:
        oh = np.eye(A[:, j].max() + 1, dtype=np.float32)[A[:, j]]
        K += oh @ oh.T
    return K / len(heads)


def kernel_corr(K1, K2):
    """Linear CKA (double-centered kernel alignment): removes each token's
    base co-assignment rate (frequent tokens co-assign everywhere), which a
    plain Pearson correlation over pairs would count as agreement."""
    N = K1.shape[0]
    K1, K2 = K1.copy(), K2.copy()
    np.fill_diagonal(K1, 0.0)          # the unit self-similarity survives any permutation
    np.fill_diagonal(K2, 0.0)          # of tokens and would inflate alignment at finite N
    H = np.eye(N, dtype=np.float32) - 1.0 / N
    A, B = H @ K1 @ H, H @ K2 @ H
    return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))


def pearson_corr(K1, K2):
    iu = np.triu_indices(K1.shape[0], 1)
    return float(np.corrcoef(K1[iu], K2[iu])[0, 1])


_LENS = {}


def logit_lens_topk(W, scale, topk=20):
    """Top-k promoted token ids per entry (n, d): unembed ln_f's affine part
    (centering and gamma; the RMS denominator is a positive scalar for a lone
    direction and does not change the ranking) of scale*W. Ignores blocks
    8..11 and the nonlinearity of ln_f around a real residual: a crude proxy."""
    if not _LENS:
        from transformers import GPT2LMHeadModel
        m = GPT2LMHeadModel.from_pretrained("gpt2")
        _LENS["gamma"] = m.transformer.ln_f.weight.detach().numpy().astype(np.float32)
        _LENS["WU"] = m.lm_head.weight.detach().numpy().astype(np.float32)      # (V, d)
    gamma, WU = _LENS["gamma"], _LENS["WU"]
    V = scale * W
    V = V - V.mean(-1, keepdims=True)
    logits = (V * gamma) @ WU.T
    return np.argsort(-logits, axis=-1)[:, :topk]


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--cache", default="data/gpt2_l8")
    p.add_argument("--rows", type=int, default=4096, help="tokens for the co-assignment kernel")
    p.add_argument("--lens", action="store_true")
    p.add_argument("--out", default=None)
    cfg = p.parse_args()
    rng = np.random.default_rng(0)
    cache = Cache(cfg.cache)

    WA, model, _, sa = entries(cfg.run_a)
    WB, _, _, sb = entries(cfg.run_b)
    l, h, k, d = WA.shape
    res = {"runs": [cfg.run_a, cfg.run_b], "steps": [sa, sb], "layers": []}

    print("== prototype matching (Hungarian on cosine, l*h*k entries per layer)")
    print(f"{'layer':>5} {'matched cos':>11} {'median':>7} {'frac>0.5':>9} {'frac>0.8':>9} {'best-match':>10} "
          f"{'purity mean/max/null':>20} | {'other-layer':>11} {'random':>7}")
    UA, UB = unit(WA.reshape(l, h * k, d)), unit(WB.reshape(l, h * k, d))
    matches = []
    for i in range(l):
        m, r, c, C = hungarian_cos(UA[i], UB[i])
        matches.append(c)
        best = C.max(1)
        pm, px = head_purity(c, h, k)
        pm0 = np.mean([head_purity(rng.permutation(h * k), h, k)[0] for _ in range(20)])
        other = np.mean([hungarian_cos(UA[i], UB[j])[0].mean() for j in range(l) if j != i])
        R = unit(rng.standard_normal((h * k, d)).astype(np.float32))
        rand = hungarian_cos(UA[i], R)[0].mean()
        row = {"layer": i, "matched_cos_mean": float(m.mean()), "matched_cos_median": float(np.median(m)),
               "frac_gt_0.5": float((m > 0.5).mean()), "frac_gt_0.8": float((m > 0.8).mean()),
               "best_match_mean": float(best.mean()), "head_purity_mean": pm, "head_purity_max": px,
               "head_purity_null": float(pm0),
               "other_layer_null": float(other), "random_null": float(rand)}
        res["layers"].append(row)
        print(f"{i:5d} {row['matched_cos_mean']:11.3f} {row['matched_cos_median']:7.3f} {row['frac_gt_0.5']:9.2f} "
              f"{row['frac_gt_0.8']:9.2f} {row['best_match_mean']:10.3f} {pm:8.2f}/{px:4.2f}/{pm0:4.2f}   | "
              f"{other:11.3f} {rand:7.3f}")

    print("\n== co-assignment kernel (corr over token pairs)")
    X = cache.eval_rows(65536)
    X = X[rng.choice(len(X), cfg.rows, replace=False)]
    ra, sa_ = split_step(cfg.run_a)
    rb, sb_ = split_step(cfg.run_b)
    A, _, _ = seedstab.assignments(ra, X, step=sa_)
    B, _, _ = seedstab.assignments(rb, X, step=sb_)
    # CKA between full kernels; split-half (16 vs 16 heads) within and across runs;
    # a row-shuffled B as the chance level. Within-run halves are complementary
    # rather than redundant (heads are trained to explain different parts of the
    # residual), so split-half is a floor, not a ceiling: the ceiling is the same
    # run at an earlier checkpoint (run:step).
    print(f"{'layer':>5} {'A vs B':>7} {'A split-half':>12} {'B split-half':>12} {'A1 vs B1':>9} "
          f"{'pearson AB':>10} {'shuffle':>8}")
    for i in list(range(l)) + [None]:
        MA = A[:, i] if i is not None else A.reshape(len(X), -1)
        MB = B[:, i] if i is not None else B.reshape(len(X), -1)
        nh = MA.shape[1]
        h1, h2 = list(range(nh // 2)), list(range(nh // 2, nh))
        KA, KB = coassign_kernel(MA, range(nh)), coassign_kernel(MB, range(nh))
        KA1, KA2 = coassign_kernel(MA, h1), coassign_kernel(MA, h2)
        KB1, KB2 = coassign_kernel(MB, h1), coassign_kernel(MB, h2)
        perm = rng.permutation(len(X))
        row = {"ab": kernel_corr(KA, KB), "a_split": kernel_corr(KA1, KA2), "b_split": kernel_corr(KB1, KB2),
               "a1_b1": kernel_corr(KA1, KB1), "ab_pearson": pearson_corr(KA, KB),
               "shuffle": kernel_corr(KA, KB[perm][:, perm])}
        if i is not None:
            res["layers"][i]["kernel"] = row
        else:
            res["kernel_all_layers"] = row
        lab = f"{i:5d}" if i is not None else "  all"
        print(f"{lab} {row['ab']:7.3f} {row['a_split']:12.3f} {row['b_split']:12.3f} {row['a1_b1']:9.3f} "
              f"{row['ab_pearson']:10.3f} {row['shuffle']:8.3f}")

    if cfg.lens:
        print("\n== logit-lens top-20 Jaccard, matched pairs vs random pairs")
        for i in range(l):
            TA = logit_lens_topk(WA[i].reshape(h * k, d), cache.scale)
            TB = logit_lens_topk(WB[i].reshape(h * k, d), cache.scale)
            c = matches[i]
            jm = np.mean([jaccard(TA[a], TB[c[a]]) for a in range(h * k)])
            perm = rng.permutation(h * k)
            jr = np.mean([jaccard(TA[a], TB[perm[a]]) for a in range(h * k)])
            res["layers"][i]["lens_jaccard_matched"] = float(jm)
            res["layers"][i]["lens_jaccard_random"] = float(jr)
            print(f"{i:5d} matched {jm:.3f}  random {jr:.3f}")

    out = cfg.out or (Path(split_step(cfg.run_b)[0]) / "protostab.json")
    Path(out).write_text(json.dumps(res, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
