"""Does a head hold SCRIPT, where it provably cannot hold LANGUAGE?

`headlang.py` asks whether one head is the language variable and finds not.
Part of that is capacity: 86 languages carry 6.43 bits and a k=32 head holds
log2(32) = 5.00, so no single head can represent the label and the NMI
ceiling is 0.875 rather than 1. Script is a coarser partition well inside
one head's budget, so it is the version of the question the architecture
can actually answer.

Also reports the finest-grained reading of the same claim: for each label,
the single best (head, entry) cell by F1. A tag that IS a language should
show up as one cell with high precision and recall for it.

Script comes from `MC4_TO_SONAR`'s NLLB suffix, except that the five
romanized mC4 splits (`bg-Latn`, `el-Latn`, `hi-Latn`, `ja-Latn`,
`ru-Latn`) map to their base language's NLLB code and so would be credited
to Cyrillic/Greek/Devanagari/Japanese; their text is Latin, and they are
relabelled Latn here.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from headlang import codes, nmi                              # noqa: E402
from ontologize.data.langs import MC4_TO_SONAR                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, required=True,
                   help="the arm's trained T (ste_h76 is 0.00015)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=65536,
                   help="cache tail rows to score")
    p.add_argument("--b", type=int, default=256)
    p.add_argument("--nulls", type=int, default=3)
    p.add_argument("--topm", type=int, default=0,
                   help="also report the top-m cells per label combined by "
                        "AND and by OR, for m up to this")
    p.add_argument("--out", default=None)
    return p.parse_args()


def script_of(code: str) -> str:
    """mC4 split name -> script. The `-Latn` splits are romanized text."""
    if code.endswith("-Latn"):
        return "Latn"
    nllb = MC4_TO_SONAR.get(code)
    return nllb.split("_")[1] if nllb else "???"


def ceiling(H_a: float, H_b: float) -> float:
    """Attainable NMI under arithmetic-mean normalization: I is capped by
    the smaller entropy, so the max is the min over the mean."""
    return 2 * min(H_a, H_b) / max(H_a + H_b, 1e-9)


def entropy(y, n):
    p = np.bincount(y, minlength=n) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def best_cells(A, y, n_cls, k):
    """For each label, the best (head, entry) cell by F1.

    Returns (f1, precision, recall, head, entry) per label. This is the
    per-tag version of the claim: one cell that fires for exactly one label
    and all of it.

    The maximum runs over `heads * k` cells, so it is biased upward, and
    precision is bounded below by the label's own share -- a label holding
    62% of the rows gets precision 0.62 from any cell at all. Both are why
    the caller scores this against a label-shuffled null.
    """
    n, heads = A.shape
    out = np.zeros((n_cls, 5))
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    for j in range(heads):
        # (k, n_cls) co-occurrence of this head's entry with the label
        C = np.zeros((k, n_cls))
        np.add.at(C, (A[:, j], y), 1.0)
        fired = C.sum(1, keepdims=True)                      # entry support
        prec = C / np.maximum(fired, 1e-9)
        rec = C / np.maximum(counts[None, :], 1e-9)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        e = f1.argmax(0)                                     # best entry
        better = f1[e, np.arange(n_cls)] > out[:, 0]
        idx = np.where(better)[0]
        out[idx] = np.stack([f1[e[idx], idx], prec[e[idx], idx],
                             rec[e[idx], idx], np.full(len(idx), j),
                             e[idx]], -1)
    return out


def cell_null(A, y, n_cls, k, nulls, rng):
    """Best-cell F1 per label under label shuffling: the same maximum over
    the same number of cells, with the association destroyed."""
    acc = np.zeros(n_cls)
    for _ in range(nulls):
        acc += best_cells(A, y[rng.permutation(len(y))], n_cls, k)[:, 0]
    return acc / nulls


def cooccur(A, y, n_cls, k):
    """C[j, e, c]: rows where head j emits entry e and the label is c."""
    C = np.zeros((A.shape[1], k, n_cls))
    for j in range(A.shape[1]):
        np.add.at(C[j], (A[:, j], y), 1.0)
    return C


def topm_curves(A, y, n_cls, k, M, rng, nulls=1):
    """Precision/recall for the top-m cells per label, combined two ways.

    Cells are ranked by their own F1 and taken from DISTINCT heads, since a
    head emits one entry per row and two entries of the same head never
    co-occur -- their conjunction is empty by construction.

    AND is the test that matters: each single cell is a high-recall,
    ~2%-FPR detector, so if those false positives were independent across
    heads a conjunction of two would cut FPR to ~4e-4 and precision would
    jump. Correlated false positives (the same hard rows) leave it flat.
    """
    C = cooccur(A, y, n_cls, k)
    fired = C.sum(-1)                                      # (heads, k)
    counts = np.bincount(y, minlength=n_cls).astype(np.float64)
    n = len(y)
    out = np.zeros((M, 4))                                 # prec/rec and/or
    picked = np.zeros((n_cls, M, 2), int)
    for c in range(n_cls):
        tp = C[:, :, c]
        prec = tp / np.maximum(fired, 1e-9)
        rec = tp / max(counts[c], 1e-9)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        order = np.argsort(-f1, axis=None)
        seen, cells = set(), []
        for flat in order:
            j, e = divmod(int(flat), k)
            if j in seen:
                continue
            seen.add(j)
            cells.append((j, e))
            if len(cells) == M:
                break
        yc = y == c
        m_and = np.ones(n, bool)
        m_or = np.zeros(n, bool)
        for i, (j, e) in enumerate(cells):
            picked[c, i] = (j, e)
            hit = A[:, j] == e
            m_and &= hit
            m_or |= hit
            for col, mask in ((0, m_and), (2, m_or)):
                f = mask.sum()
                out[i, col] += (yc & mask).sum() / max(f, 1)      # precision
                out[i, col + 1] += (yc & mask).sum() / max(yc.sum(), 1)
    return out / n_cls, picked


def main():
    cfg = parse_args()
    cache = Path(cfg.cache)
    langs_path = cache.with_name(cache.name.replace(".npy", ".langs.npy"))
    X = np.asarray(np.load(cache, mmap_mode="r")[-cfg.rows:], dtype=np.float32)
    raw = np.load(langs_path)[-cfg.rows:]

    lnames, y_lang = np.unique(raw, return_inverse=True)
    scripts = np.array([script_of(c) for c in lnames])
    snames, s_idx = np.unique(scripts, return_inverse=True)
    y_scr = s_idx[y_lang]

    H_l = entropy(y_lang, len(lnames))
    H_s = entropy(y_scr, len(snames))
    print(f"{cfg.rows} rows: {len(lnames)} languages H = {H_l:.2f} bits, "
          f"{len(snames)} scripts H = {H_s:.2f} bits")

    A, model, step = codes(cfg.model, cfg.step, cfg.temperature, X, cfg.b)
    k, h, l = model.k, model.h, model.l
    print(f"{Path(cfg.model).name} step {step}: {l}x{h} heads, k={k}, "
          f"a head holds log2(k) = {np.log2(k):.2f} bits\n")

    rng = np.random.default_rng(0)
    rows = {}
    for lab, y, n_cls, H in (("language", y_lang, len(lnames), H_l),
                             ("script", y_scr, len(snames), H_s)):
        S, Hh = [], []
        for j in range(A.shape[1]):
            s, _, hh = nmi(A[:, j], y, k, n_cls)
            S.append(s)
            Hh.append(hh)
        S, Hh = np.array(S), np.array(Hh)
        best = int(S.argmax())
        null = float(np.mean([
            nmi(A[:, j], y[rng.permutation(len(y))], k, n_cls)[0]
            for _ in range(cfg.nulls) for j in range(0, len(S), 37)]))
        ceil = ceiling(Hh[best], H)
        print(f"{lab:>9}: best-head NMI {S.max():.3f} "
              f"(layer {best // h} head {best % h}, H(head) {Hh[best]:.2f} "
              f"bits, ceiling {ceil:.3f}) "
              f"-> {S.max() / ceil:.1%} of attainable")
        print(f"{'':>9}  median head {np.median(S):.3f}   "
              f"shuffled null {null:.3f}   less null {S.max() - null:.3f}")
        rows[lab] = {"best_nmi": float(S.max()), "ceiling": ceil,
                     "null": null, "median": float(np.median(S)),
                     "H_label": H, "H_head_best": float(Hh[best])}

    for lab, y, names in (("script", y_scr, snames),
                          ("language", y_lang, lnames)):
        B = best_cells(A, y, len(names), k)
        null = cell_null(A, y, len(names), k, cfg.nulls, rng)
        freq = np.bincount(y, minlength=len(names)) / len(y)
        # enrichment of the cell's precision over the label's base rate: 1.0
        # means the cell says nothing the prior does not
        enr = B[:, 1] / np.maximum(freq, 1e-9)
        lift = B[:, 0] - null
        order = np.argsort(-lift)
        print(f"\nbest single (head, entry) cell per {lab}, ranked by F1 "
              f"over its shuffled null -- top 12 of {len(names)}")
        # precision is capped by the class imbalance (85:1 for languages), so
        # recall against the false-positive rate is the fair pair; `prec`
        # follows from them and the share
        fpr = np.where(B[:, 1] > 0,
                       B[:, 2] * freq * (1 / np.maximum(B[:, 1], 1e-9) - 1)
                       / np.maximum(1 - freq, 1e-9), 0.0)
        print(f"{lab:>10} {'share':>7} {'lift':>6} {'rec':>6} {'FPR':>6} "
              f"{'prec':>6} {'enr':>6}  cell")
        for c in order[:12]:
            print(f"{str(names[c]):>10} {freq[c]:>7.3f} {lift[c]:>+6.3f} "
                  f"{B[c,2]:>6.3f} {fpr[c]:>6.3f} {B[c,1]:>6.3f} "
                  f"{enr[c]:>5.1f}x  "
                  f"L{int(B[c,3])//h}h{int(B[c,3])%h}e{int(B[c,4])}")
        print(f"{'':>10} median FPR {np.median(fpr):.4f}; precision 0.8 at "
              f"this recall would need FPR "
              f"{float(np.median(B[:,2]*freq*0.25/np.maximum(1-freq,1e-9))):.4f}")
        print(f"{'':>10} median over all {lab}s: F1 {np.median(B[:,0]):.3f}, "
              f"null {np.median(null):.3f}, lift {np.median(lift):+.3f}, "
              f"enrichment {np.median(enr):.1f}x")
        print(f"{'':>10} lift>0.2: {(lift>0.2).sum()}/{len(names)}   "
              f"lift>0.4: {(lift>0.4).sum()}/{len(names)}   "
              f"recall>0.8: {(B[:,2]>0.8).sum()}/{len(names)}   "
              f"precision>0.8: {(B[:,1]>0.8).sum()}/{len(names)}")
        rows[f"{lab}_cells"] = {
            "median_f1": float(np.median(B[:, 0])),
            "median_null": float(np.median(null)),
            "median_lift": float(np.median(lift)),
            "median_enrichment": float(np.median(enr)),
            "n_lift_over_0.2": int((lift > 0.2).sum()),
            "n_recall_over_0.8": int((B[:, 2] > 0.8).sum()),
            "n_precision_over_0.8": int((B[:, 1] > 0.8).sum()),
            "n": len(names)}

    # precision near 1/m suggests a cell covers about m labels. Name them:
    # a coherent group is a finding about the ontology, an arbitrary mixture
    # is not.
    B = best_cells(A, y_lang, len(lnames), k)
    lift = B[:, 0] - cell_null(A, y_lang, len(lnames), k, cfg.nulls, rng)
    print("\nwhat else fires in the best cell for each of the top languages "
          "(>=5% of the cell)")
    for c in np.argsort(-lift)[:8]:
        j, e = int(B[c, 3]), int(B[c, 4])
        mask = A[:, j] == e
        share = np.bincount(y_lang[mask], minlength=len(lnames)) / mask.sum()
        named = [(lnames[i], share[i]) for i in np.argsort(-share)
                 if share[i] >= 0.05]
        body = "  ".join(f"{n} {s:.2f}" for n, s in named)
        print(f"  {str(lnames[c]):>8} L{j//h}h{j%h}e{e} "
              f"(n={int(mask.sum())}): {body}")

    if cfg.topm:
        for lab, y, names in (("language", y_lang, lnames),
                              ("script", y_scr, snames)):
            cur, picked = topm_curves(A, y, len(names), k, cfg.topm, rng)
            share = 1.0 / len(names)
            print(f"\ntop-m cells per {lab} (distinct heads, ranked by own "
                  f"F1), averaged over {len(names)} labels")
            print(f"{'m':>3} {'AND prec':>9} {'AND rec':>8} {'AND F1':>7}  "
                  f"{'OR prec':>8} {'OR rec':>7} {'OR F1':>7}")
            for i in range(cfg.topm):
                ap, ar, op, orr = cur[i]
                af = 2 * ap * ar / max(ap + ar, 1e-9)
                of = 2 * op * orr / max(op + orr, 1e-9)
                print(f"{i+1:>3} {ap:>9.3f} {ar:>8.3f} {af:>7.3f}  "
                      f"{op:>8.3f} {orr:>7.3f} {of:>7.3f}")
            rows[f"{lab}_topm"] = {
                "and_prec": cur[:, 0].tolist(), "and_rec": cur[:, 1].tolist(),
                "or_prec": cur[:, 2].tolist(), "or_rec": cur[:, 3].tolist()}

    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {cfg.out}")


if __name__ == "__main__":
    main()
