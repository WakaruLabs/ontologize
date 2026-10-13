"""Head transitions between sentences, and how close adjacent sentences are.

Two measurements on the held-out tail of a sentence cache (the last
`--holdout` rows, which no model trained on the cache has seen). Both come
before designing a pair objective (README.md).

1. Head transitions. For each (layer, head) of each checkpoint: how often the
   entry it assigns agrees between the two sentences of a pair, as Cohen's
   kappa. 0 is the agreement the head's own entry usage gives by chance; 1 is
   always the same entry. Pair types:
   - `d1`, `d2`, ...: one document, segments d apart (`d1` is adjacent);
   - `d1_sent`: adjacent pairs where both rows are full sentences, not a
     line or a piece the length cap cut out of one sentence;
   - `lang`: a sentence of another document in the same language;
   - `random`: a sentence of another document.
   `lang` and `random` partners are drawn for the `d1` anchors. Also reported:
   each head's NMI with the language label, and how many heads change
   between the two sentences of a pair.
2. Pair concentration, with no model. For the same pair types: the whitened
   squared distance (`dist`, the training loss's metric) and the cosine
   (`cos`, the direction gain-shape classifies). Then how well each
   separates adjacent pairs from `lang` and `random` pairs (AUC), and how
   often a sentence's successor is the nearest of it and `--negatives`
   other-document sentences. The AUCs also cover the whitened cosine
   (`wcos`) and the raw squared distance (`rawdist`), which separate the
   effect of whitening from that of the norm, and the correlation of the two
   sentences' log norms.

    uv run python experiments/next-sentence/transitions.py \\
        --ckpt data/out/sonar/sent16/ste_h76_init01_no837 \\
        --out data/out/sonar/next-sentence/transitions
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

# partition.py sets the XLA environment on import, as running it would
from partition import codes, nmi_matrix  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

from pairs import adjacent  # noqa: E402

CACHE = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.npy"
WEIGHTS = ROOT / "data" / "out" / "sonar" / "sentcache" / "mse_weights_no837.npy"
DISTANCES = (1, 2, 4, 8, 12)


def same_doc_pairs(index: Int[np.ndarray, "n 3"], d: int
                   ) -> Tuple[Int[np.ndarray, "p"], Int[np.ndarray, "p"]]:
    """Row pairs (i, j) of one document whose segments are `d` apart."""
    doc = index[:, 0].astype(np.int64)
    seg = index[:, 2].astype(np.int64)
    key = (doc << 32) | seg
    order = np.argsort(key, kind="stable")
    sk = key[order]
    want = (doc << 32) | (seg + d)
    at = np.minimum(np.searchsorted(sk, want), len(sk) - 1)
    hit = sk[at] == want
    return np.flatnonzero(hit), order[at[hit]]


def other_doc_partners(anchors: Int[np.ndarray, "p"],
                       doc: Int[np.ndarray, "n"], group: Int[np.ndarray, "n"],
                       rng: np.random.Generator, tries: int = 64
                       ) -> Int[np.ndarray, "p"]:
    """For each anchor row, a row drawn uniformly from the rows of other
    documents with the same `group` value (a constant group draws from every
    other document). -1 where the group has no other document."""
    out = np.full(len(anchors), -1, np.int64)
    ga = group[anchors]
    for g in np.unique(ga):
        cand = np.flatnonzero(group == g)
        if len(np.unique(doc[cand])) < 2:
            continue
        todo = np.flatnonzero(ga == g)
        for _ in range(tries):
            pick = cand[rng.integers(len(cand), size=len(todo))]
            ok = doc[pick] != doc[anchors[todo]]
            out[todo[ok]] = pick[ok]
            todo = todo[~ok]
            if not len(todo):
                break
    return out


def usage(A: Int[np.ndarray, "heads p"], k: int) -> Float[np.ndarray, "heads k"]:
    """Each head's entry frequencies over the rows of `A`."""
    heads, p = A.shape
    off = (np.arange(heads, dtype=np.int64) * k)[:, None]
    counts = np.bincount((A + off).ravel(), minlength=heads * k)
    return counts.reshape(heads, k) / p


def kappa(Ai: Int[np.ndarray, "heads p"], Aj: Int[np.ndarray, "heads p"],
          k: int) -> Float[np.ndarray, "heads"]:
    """Cohen's kappa per head between the two sides of each pair:
    (agreement - chance) / (1 - chance), where chance is the agreement of
    the two sides' entry frequencies. NaN for a head whose chance agreement
    is 1, i.e. one entry on every row."""
    agree = (Ai == Aj).mean(1)
    chance = (usage(Ai, k) * usage(Aj, k)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = (agree - chance) / (1.0 - chance)
    return np.where(chance < 1.0 - 1e-9, kap, np.nan)


def entropy_bits(P: Float[np.ndarray, "heads k"]) -> Float[np.ndarray, "heads"]:
    return -np.where(P > 0, P * np.log2(np.maximum(P, 1e-30)), 0.0).sum(1)


def auc(pos: Float[np.ndarray, "a"], neg: Float[np.ndarray, "b"]) -> float:
    """P(pos > neg), ties counted half (Mann-Whitney)."""
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def similarity(Z: Float[np.ndarray, "n d"], i: Int[np.ndarray, "p"],
               j: Int[np.ndarray, "p"], metric: str) -> Float[np.ndarray, "p"]:
    """Higher is closer: for a `*dist` metric the negative squared
    distance between rows of `Z` (already scaled by sqrt of any weights),
    otherwise their dot product (rows already unit-normalized)."""
    if metric.endswith("dist"):
        return -((Z[i] - Z[j]) ** 2).sum(1)
    return (Z[i] * Z[j]).sum(1)


def retrieval(Z: Float[np.ndarray, "n d"], i: Int[np.ndarray, "p"],
              j: Int[np.ndarray, "p"], negs: Int[np.ndarray, "m p"],
              metric: str) -> Tuple[float, float]:
    """Top-1 rate and median rank of `j` among itself and the negatives,
    by similarity to `i`. Rank 1 is the nearest; ties count against `j`."""
    s = similarity(Z, i, j, metric)
    beat = np.zeros(len(i), np.int64)
    for m in range(len(negs)):
        beat += similarity(Z, i, negs[m], metric) >= s
    return float((beat == 0).mean()), float(np.median(beat + 1))


def pair_sets(index: Int[np.ndarray, "n 3"], kind: np.ndarray,
              doc: Int[np.ndarray, "n"], lang: Int[np.ndarray, "n"],
              seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Every pair type of the module docstring, as local row indices."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in DISTANCES:
        out[f"d{d}"] = same_doc_pairs(index, d)
    i1, j1 = out["d1"]
    assert np.array_equal(i1, adjacent(index, 0, len(index))), \
        "d1 should be exactly the adjacent pairs"
    sent = (kind[i1] == "sentence") & (kind[j1] == "sentence")
    out["d1_sent"] = (i1[sent], j1[sent])
    for name, group in (("lang", lang), ("random", np.zeros_like(lang))):
        j = other_doc_partners(i1, doc, group, rng)
        out[name] = (i1[j >= 0], j[j >= 0])
    return out


def transitions(A: Int[np.ndarray, "heads n"], layer: Int[np.ndarray, "heads"],
                k: int, pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
                lang: Int[np.ndarray, "n"]) -> Tuple[Dict, List[Dict]]:
    """Per-head kappa and switch rates for every pair type, with each
    head's entropy and NMI with the language label."""
    kap = {t: kappa(A[:, i], A[:, j], k) for t, (i, j) in pairs.items()}
    switch = {t: (A[:, i] != A[:, j]).mean(1) for t, (i, j) in pairs.items()}
    nmi = nmi_matrix(A, k, lang[None, :].astype(np.int64),
                     int(lang.max()) + 1)[:, 0]
    ent = entropy_bits(usage(A, k))
    l = int(layer.max()) + 1
    h = len(layer) // l
    heads = [{"layer": int(layer[r]), "head": int(r % h),
              "entropy_bits": float(ent[r]), "nmi_lang": float(nmi[r]),
              **{f"kappa_{t}": float(kap[t][r]) for t in pairs},
              **{f"switch_{t}": float(switch[t][r]) for t in pairs}}
             for r in range(len(layer))]
    ok = np.isfinite(kap["d1"])
    summary = {
        "k": k, "heads": len(layer), "layers": l,
        "dead_heads": int((~ok).sum()),
        "pairs": {t: int(len(i)) for t, (i, _) in pairs.items()},
        "kappa_mean": {t: float(np.nanmean(v)) for t, v in kap.items()},
        "kappa_mean_by_layer": {
            t: [float(np.nanmean(v[layer == a])) for a in range(l)]
            for t, v in kap.items()},
        "switch_mean": {t: float(v.mean()) for t, v in switch.items()},
        "switch_mean_by_layer": {
            t: [float(v[layer == a].mean()) for a in range(l)]
            for t, v in switch.items()},
        "kappa_d1_quantiles": {
            q: float(np.nanquantile(kap["d1"], q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "kappa_d1_above": {f"{c}": int((kap["d1"][ok] > c).sum())
                           for c in (0.2, 0.5, 0.8)},
        "spearman_kappa_d1_nmi_lang": float(
            spearmanr(kap["d1"][ok], nmi[ok])[0]),
        "spearman_kappa_d1_minus_lang_nmi_lang": float(
            spearmanr((kap["d1"] - kap["lang"])[ok], nmi[ok])[0]),
    }
    # how many heads change per pair: the transition's sparsity
    for t in ("d1", "d1_sent", "lang", "random"):
        i, j = pairs[t]
        n_sw = (A[:, i] != A[:, j]).mean(0)
        summary.setdefault("heads_switched_frac", {})[t] = {
            q: float(np.quantile(n_sw, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    return summary, heads


def concentration(X: Float[np.ndarray, "n d"], w: Float[np.ndarray, "d"],
                  pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
                  doc: Int[np.ndarray, "n"], lang: Int[np.ndarray, "n"],
                  negatives: int, seed: int) -> Dict:
    """Distances, AUCs and successor retrieval for every pair type."""
    rng = np.random.default_rng(seed + 1)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    Xw = X * np.sqrt(w)
    Z = {"dist": Xw.astype(np.float32),
         "cos": (X / norm).astype(np.float32),
         "wcos": (Xw / np.linalg.norm(Xw, axis=1, keepdims=True)).astype(
             np.float32),
         "rawdist": X}
    sims = {m: {t: similarity(Z[m], i, j, m) for t, (i, j) in pairs.items()}
            for m in Z}
    rand_d2 = -sims["dist"]["random"].mean()
    log_norm = np.log(norm[:, 0])
    out = {"mean_dist_over_random": {
               t: float(-s.mean() / rand_d2) for t, s in sims["dist"].items()},
           "mean_cos": {t: float(s.mean()) for t, s in sims["cos"].items()},
           "log_norm_corr": {
               t: float(np.corrcoef(log_norm[i], log_norm[j])[0, 1])
               for t, (i, j) in pairs.items()},
           "auc": {}, "retrieval": {}}
    for m in Z:
        for pos in ("d1", "d1_sent", "d12"):
            for neg in ("lang", "random"):
                out["auc"][f"{m}: {pos} vs {neg}"] = auc(sims[m][pos],
                                                         sims[m][neg])
    i1, j1 = pairs["d1"]
    negs = {}
    for name, group in (("lang", lang), ("random", np.zeros_like(lang))):
        draws = np.stack([other_doc_partners(i1, doc, group, rng)
                          for _ in range(negatives)])
        keep = (draws >= 0).all(0)
        negs[name] = (keep, draws[:, keep])
    for m in ("dist", "cos"):
        for name, (keep, draws) in negs.items():
            top1, rank = retrieval(Z[m], i1[keep], j1[keep], draws, m)
            out["retrieval"][f"{m}: successor among {negatives} {name}"] = {
                "anchors": int(keep.sum()), "top1": top1,
                "median_rank": rank, "chance_top1": 1.0 / (negatives + 1)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[],
                   help="checkpoint dirs whose head transitions to measure")
    p.add_argument("--step", type=int, default=0,
                   help="checkpoint step (0 = latest)")
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--mse-weights", default=str(WEIGHTS),
                   help="per-coordinate weights of the whitened distance")
    p.add_argument("--holdout", type=int, default=32768,
                   help="the held-out tail rows, as the models trained")
    p.add_argument("--negatives", type=int, default=255)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="directory for concentration.json and, per "
                        "checkpoint, <run>_heads.csv and <run>_summary.json")
    cfg = p.parse_args()

    cache = Path(cfg.cache)
    stem = cache.with_name(cache.stem)
    arr = np.load(cache, mmap_mode="r")
    n = len(arr)
    lo = n - cfg.holdout
    X = np.asarray(arr[lo:], np.float32)
    index = np.asarray(np.load(f"{stem}.index.npy", mmap_mode="r")[lo:])
    kind = np.asarray(np.load(f"{stem}.kind.npy", mmap_mode="r")[lo:])
    names, lang = np.unique(
        np.asarray(np.load(f"{stem}.langs.npy", mmap_mode="r")[lo:]),
        return_inverse=True)
    doc = index[:, 0].astype(np.int64)
    pairs = pair_sets(index, kind, doc, lang, cfg.seed)
    print(f"{cfg.holdout} held-out rows, {len(np.unique(doc))} documents, "
          f"{len(names)} languages; pairs: "
          + ", ".join(f"{t} {len(i)}" for t, (i, _) in pairs.items()))
    out = Path(cfg.out) if cfg.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    conc = concentration(X, np.load(cfg.mse_weights).astype(np.float32),
                         pairs, doc, lang, cfg.negatives, cfg.seed)
    print("\npair concentration (no model)")
    for t, v in conc["mean_dist_over_random"].items():
        print(f"  {t:>8}: whitened squared distance {v:.3f} x random, "
              f"cosine {conc['mean_cos'][t]:.3f}, "
              f"log-norm correlation {conc['log_norm_corr'][t]:.3f}")
    for name, v in conc["auc"].items():
        print(f"  AUC {name}: {v:.3f}")
    for name, v in conc["retrieval"].items():
        print(f"  {name}: top-1 {v['top1']:.3f} (chance {v['chance_top1']:.4f}),"
              f" median rank {v['median_rank']:.0f}, {v['anchors']} anchors")
    if out:
        (out / "concentration.json").write_text(json.dumps(conc, indent=1))

    for ckpt in cfg.ckpt:
        A, layer, k, step = codes(ckpt, cfg.step, cfg.temperature, X,
                                  cfg.batch)
        summary, heads = transitions(A, layer, k, pairs, lang)
        summary.update({"ckpt": ckpt, "step": step})
        name = Path(ckpt).name
        print(f"\n{name} (step {step}; {summary['layers']} layers x "
              f"{summary['heads'] // summary['layers']} heads, k {k}; "
              f"{summary['dead_heads']} single-entry heads)")
        print("  mean kappa by layer:")
        for t, v in summary["kappa_mean_by_layer"].items():
            print(f"    {t:>8}: " + "  ".join(f"{x:6.3f}" for x in v)
                  + f"   all {summary['kappa_mean'][t]:.3f}")
        print("  share of heads that switch, by layer:")
        for t, v in summary["switch_mean_by_layer"].items():
            print(f"    {t:>8}: " + "  ".join(f"{x:6.3f}" for x in v))
        q = summary["kappa_d1_quantiles"]
        print("  kappa_d1 over heads: " + ", ".join(
            f"q{int(a * 100)} {b:.3f}" for a, b in q.items())
            + "; heads above 0.2 / 0.5 / 0.8: "
            + " / ".join(str(v) for v in summary["kappa_d1_above"].values()))
        print("  heads switched per pair (quantiles 5/25/50/75/95%):")
        for t, v in summary["heads_switched_frac"].items():
            print(f"    {t:>8}: " + "  ".join(f"{x:.3f}" for x in v.values()))
        print(f"  Spearman over heads: kappa_d1 vs NMI(lang) "
              f"{summary['spearman_kappa_d1_nmi_lang']:.3f}; "
              f"kappa_d1 - kappa_lang vs NMI(lang) "
              f"{summary['spearman_kappa_d1_minus_lang_nmi_lang']:.3f}")
        top = sorted((r for r in heads if np.isfinite(r["kappa_d1"])),
                     key=lambda r: -r["kappa_d1"])[:12]
        print("  slowest heads: layer head  kappa_d1 d2 d8  lang  random "
              " NMI(lang)  entropy")
        for r in top:
            print(f"    {r['layer']:>5} {r['head']:>4}  {r['kappa_d1']:.3f} "
                  f"{r['kappa_d2']:.3f} {r['kappa_d8']:.3f}  "
                  f"{r['kappa_lang']:.3f} {r['kappa_random']:+.3f}  "
                  f"{r['nmi_lang']:.3f}  {r['entropy_bits']:.2f}")
        if out:
            with open(out / f"{name}_heads.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(heads[0]))
                wr.writeheader()
                wr.writerows(heads)
            (out / f"{name}_summary.json").write_text(
                json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
