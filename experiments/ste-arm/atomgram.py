"""Do two checkpoints' heads hold the same atoms, whatever order each
stores its entries in?

A head's atom Gram, the cosines between its k atoms, describes the head's
internal geometry without reference to a basis, so it can be compared
across runs whose decoders differ. What the Gram cannot say alone is
which entry of one head corresponds to which entry of the other. Entry
order is a gauge (relabeling a head's entries, together with its
classifier rows, changes nothing the model computes), and how the
comparison fixes it decides what it measures. Three ways, all reported:

  canonical  each head's entries sorted by their mean cosine to the rest
             of the head. Needs no correspondence, but where a head's
             Gram is nearly one-parameter every entry's mean cosine is
             nearly the same, so the sort is set by small differences
             that need not survive from one checkpoint to the next.
  stored     entries in stored order: the correspondence within one run,
             since training never relabels entries (with dead-entry
             revival off). Undefined across runs, and not reported there.
  matched    entries paired by linear assignment on the cosine of their
             centered decoded atoms, which fixes the gauge exactly,
             because both runs' decoded atoms live in the one output
             space the objective scores.

For each order, the off-diagonal RMS difference between two heads'
Grams. Heads pair by index within a run, and across runs by linear
assignment on that distance, the most favorable pairing. The ratio
reported is the mean distance to the layer's other heads over the
distance to the paired head, so 1 is chance against a null that keeps
the layer: layers differ in collinearity far more than heads within a
layer do, so a whole-model null reads the layer signal as agreement.
Within a run, `recovers` is the share of heads that the assignment on
the distance pairs with themselves.

Grams come in three variants: the dictionary's own atoms (e_dec, as
`dicts()` returns them), uncentered; and the atoms decoded to output
space in the objective's whitened frame, uncentered and centered on the
head's usage-weighted mean atom. In e_dec
part of each atom lies in the decoder's null space, which decoding
removes, and a non-negative dictionary gives every head a shared offset,
which centering removes.

The atoms themselves are compared too: per pair of heads, the mean
cosine of matched centered decoded atoms over live entries (selected on
at least one of `--rows` tail rows), for the paired head against the
layer's other heads, and within a run in stored order.

Two checkpoints are one run when `--a` and `--b` name the same
directory. Atoms are decoded as `headcontrib.atoms_and_codes` decodes
them, so router (`scaled`) and fiber models are refused.

  # one run against its own earlier checkpoint
  uv run python experiments/ste-arm/atomgram.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \\
      --b data/out/sonar/multilingual/ste_h76_init01 --step-b 350000

  # two seeds
  uv run python experiments/ste-arm/atomgram.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 --step-a 369500 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --step-b 369500

Writes <out>/gram.csv (space, centered, layer, the three ratios and the
two recovery shares), <out>/atoms.csv (layer, paired, other, stored) and
<out>/summary.json; default out directory
data/out/sonar/atomgram/<a>_<step>_vs_<b>_<step>.
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from jaxtyping import Bool, Float, Int
from scipy.optimize import linear_sum_assignment

from headcontrib import atoms_and_codes
from pareto import load_onto

# (space, centered)
VARIANTS = (("e", False), ("decoded", False), ("decoded", True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir")
    p.add_argument("--b", required=True, help="checkpoint dir")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015,
                   help="classification temperature for the usage counts; "
                        "irrelevant under select='ste'")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy",
                   help="per-dimension weights defining the whitened frame")
    p.add_argument("--rows", type=int, default=32768,
                   help="cache tail rows that set usage (centering, live "
                        "entries)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", default=None)
    return p.parse_args()


def unit(A: Float[np.ndarray, "... d"]) -> Float[np.ndarray, "... d"]:
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-30)


def gram(A: Float[np.ndarray, "h k d"]) -> Float[np.ndarray, "h k k"]:
    """Each head's atom-cosine Gram."""
    U = unit(A)
    return U @ U.transpose(0, 2, 1)


def centered(A: Float[np.ndarray, "h k d"], usage: Float[np.ndarray, "h k"]
             ) -> Float[np.ndarray, "h k d"]:
    """Each head's atoms less its usage-weighted mean atom."""
    p = usage / np.maximum(usage.sum(-1, keepdims=True), 1e-30)
    return A - (p[..., None] * A).sum(1, keepdims=True)


def canonical(G: Float[np.ndarray, "h k k"]) -> Int[np.ndarray, "h k"]:
    """Each head's entries by descending mean cosine to the rest of the
    head. A stable sort, so tied entries keep their stored order."""
    k = G.shape[-1]
    mean = (G.sum(-1) - np.einsum("hkk->hk", G)) / (k - 1)
    return np.argsort(-mean, axis=-1, kind="stable")


def reorder(G: Float[np.ndarray, "h k k"], perm: Int[np.ndarray, "h k"]
            ) -> Float[np.ndarray, "h k k"]:
    """Each head's Gram with its entries in the order `perm` gives."""
    return np.take_along_axis(
        np.take_along_axis(G, perm[:, :, None], 1), perm[:, None, :], 2)


def offdiag_rms(G1: Float[np.ndarray, "ha k k"],
                G2: Float[np.ndarray, "hb k k"]
                ) -> Float[np.ndarray, "ha hb"]:
    """RMS difference over the off-diagonal entries, for every pair of a
    head of G1 with a head of G2."""
    k = G1.shape[-1]
    off = ~np.eye(k, dtype=bool)
    a, b = G1[:, off], G2[:, off]                       # (h, k(k-1))
    sq = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :]
          - 2.0 * a @ b.T)
    return np.sqrt(np.maximum(sq, 0.0) / off.sum())


def entry_match(A1: Float[np.ndarray, "ha k d"],
                A2: Float[np.ndarray, "hb k d"]) -> Int[np.ndarray, "ha hb k"]:
    """For every pair of heads (i, j), the order of j's entries that the
    linear assignment on atom cosine pairs with i's entries 0..k-1, so
    A2[j][perm[i, j]] lines up with A1[i]."""
    (ha, k, d), hb = A1.shape, A2.shape[0]
    C = (unit(A1).reshape(ha * k, d) @ unit(A2).reshape(hb * k, d).T
         ).reshape(ha, k, hb, k).transpose(0, 2, 1, 3)  # (ha, hb, k, k)
    perm = np.empty(C.shape[:3], np.int64)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            r, c = linear_sum_assignment(-C[i, j])
            perm[i, j, r] = c
    return perm


def matched_rms(G1: Float[np.ndarray, "ha k k"],
                G2: Float[np.ndarray, "hb k k"],
                perm: Int[np.ndarray, "ha hb k"]
                ) -> Float[np.ndarray, "ha hb"]:
    """`offdiag_rms` with each pair's entries aligned by `perm`."""
    k = G1.shape[-1]
    off = ~np.eye(k, dtype=bool)
    out = np.empty(perm.shape[:2])
    for j in range(G2.shape[0]):
        P = perm[:, j]                                  # (ha, k)
        G2j = G2[j][P[:, :, None], P[:, None, :]]       # (ha, k, k)
        out[:, j] = np.sqrt(((G1 - G2j)[:, off] ** 2).mean(1))
    return out


def paired_mask(D: Float[np.ndarray, "h h"], same_run: bool
                ) -> Bool[np.ndarray, "h h"]:
    """Which (head of A, head of B) pairs are paired: the identity within a
    run, the linear assignment minimizing D across runs."""
    h = D.shape[0]
    pair = np.arange(h) if same_run else linear_sum_assignment(D)[1]
    mask = np.zeros(D.shape, dtype=bool)
    mask[np.arange(h), pair] = True
    return mask


def pair_ratio(D: Float[np.ndarray, "h h"], same_run: bool) -> float:
    """Mean distance to the layer's other heads over mean distance to the
    paired head."""
    paired = paired_mask(D, same_run)
    return float(D[~paired].mean() / D[paired].mean())


def recovers(D: Float[np.ndarray, "h h"]) -> float:
    """Share of heads the linear assignment on D pairs with themselves."""
    return float((linear_sum_assignment(D)[1] == np.arange(D.shape[0])).mean())


def mean_matched_cos(A1: Float[np.ndarray, "k1 d"],
                     A2: Float[np.ndarray, "k2 d"]) -> float:
    """Mean cosine of the atoms the linear assignment pairs; the smaller
    set is matched in full."""
    C = unit(A1) @ unit(A2).T
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def atom_agreement(A1: Float[np.ndarray, "h k d"],
                   A2: Float[np.ndarray, "h k d"],
                   live1: Bool[np.ndarray, "h k"],
                   live2: Bool[np.ndarray, "h k"], same_run: bool
                   ) -> Tuple[float, float, float]:
    """Mean matched atom cosine over live entries for the paired heads and
    for the layer's other heads; within a run also the cosine in stored
    order (NaN across runs). Heads pair as in `paired_mask`, maximizing
    the cosine across runs."""
    h = A1.shape[0]
    S = np.array([[mean_matched_cos(A1[i][live1[i]], A2[j][live2[j]])
                   for j in range(h)] for i in range(h)])
    paired = paired_mask(-S, same_run)
    stored = np.nan
    if same_run:
        both = live1 & live2
        stored = float(np.mean([(unit(A1[i][both[i]])
                                 * unit(A2[i][both[i]])).sum(-1).mean()
                                for i in range(h)]))
    return float(S[paired].mean()), float(S[~paired].mean()), stored


def head_atoms(ckpt: str, step: int, T: float,
               X: Float[np.ndarray, "n d_in"],
               w_sqrt: Float[np.ndarray, "d_out"], batch: int
               ) -> Tuple[Dict[str, List[np.ndarray]],
                          List[Float[np.ndarray, "h k"]], int]:
    """Per layer, each head's atoms in both spaces and its usage on the
    rows: ({"e": [(h, k, e)], "decoded": [(h, k, d_out)]}, [(h, k)],
    step loaded)."""
    D, I, _, layer, used = atoms_and_codes(ckpt, step, T, X, w_sqrt, batch)
    model, raw, _ = load_onto(ckpt, used)
    E = model.apply({"params": raw},
                    method=lambda m: [de.dict.dicts() for de in m.dictencs])
    k = D.shape[1]
    U = np.stack([np.bincount(row, minlength=k) for row in I]) / I.shape[1]
    atoms = {"e": [np.asarray(e, np.float64) for e in E],
             "decoded": [D[layer == i].astype(np.float64)
                         for i in range(model.l)]}
    return atoms, [U[layer == i] for i in range(model.l)], used


def compare(atoms_a: Dict[str, List[np.ndarray]],
            atoms_b: Dict[str, List[np.ndarray]],
            usage_a: List[np.ndarray], usage_b: List[np.ndarray],
            same_run: bool) -> Tuple[List[dict], List[dict]]:
    """The Gram rows (one per variant and layer) and the atom rows (one
    per layer)."""
    grams, agree = [], []
    for i, (ua, ub) in enumerate(zip(usage_a, usage_b)):
        dec_a = centered(atoms_a["decoded"][i], ua)
        dec_b = centered(atoms_b["decoded"][i], ub)
        perm = entry_match(dec_a, dec_b)
        for space, center in VARIANTS:
            Aa, Ab = atoms_a[space][i], atoms_b[space][i]
            if center:
                Aa, Ab = centered(Aa, ua), centered(Ab, ub)
            Ga, Gb = gram(Aa), gram(Ab)
            D = {"canonical": offdiag_rms(reorder(Ga, canonical(Ga)),
                                          reorder(Gb, canonical(Gb))),
                 "matched": matched_rms(Ga, Gb, perm)}
            if same_run:
                D["stored"] = offdiag_rms(Ga, Gb)
            row = {"space": space, "centered": center, "layer": i}
            for order in ("canonical", "stored", "matched"):
                row[order] = (pair_ratio(D[order], same_run)
                              if order in D else np.nan)
            for order in ("canonical", "matched"):
                row[f"recovers_{order}"] = (recovers(D[order])
                                            if same_run else np.nan)
            grams.append(row)
        paired, other, stored = atom_agreement(dec_a, dec_b, ua > 0, ub > 0,
                                               same_run)
        agree.append({"layer": i, "paired": paired, "other": other,
                      "stored": stored})
    return grams, agree


def main() -> None:
    cfg = parse_args()
    same_run = Path(cfg.a).resolve() == Path(cfg.b).resolve()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    w_sqrt = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    atoms_a, usage_a, sa = head_atoms(cfg.a, cfg.step_a, cfg.temperature, X,
                                      w_sqrt, cfg.batch)
    atoms_b, usage_b, sb = head_atoms(cfg.b, cfg.step_b, cfg.temperature, X,
                                      w_sqrt, cfg.batch)
    if [u.shape for u in usage_a] != [u.shape for u in usage_b]:
        raise SystemExit(f"{cfg.a} and {cfg.b} differ in l, h or k")
    na, nb = Path(cfg.a).name, Path(cfg.b).name
    print(f"A {na} step {sa}, B {nb} step {sb}: "
          f"{'one run' if same_run else 'two runs'}, {len(X)} rows")
    grams, agree = compare(atoms_a, atoms_b, usage_a, usage_b, same_run)

    print("\nGram ratio: mean distance to the layer's other heads / "
          "distance to the paired head (1 = chance)")
    for space, center in VARIANTS:
        print(f"\n  {space} atoms, {'centered' if center else 'uncentered'}")
        print(f"  {'layer':>5} {'canonical':>10} {'stored':>8} "
              f"{'matched':>8}" + (f" {'recovers (canon, matched)':>27}"
                                   if same_run else ""))
        for r in grams:
            if r["space"] != space or r["centered"] != center:
                continue
            rec = (f" {r['recovers_canonical']:13.2f} "
                   f"{r['recovers_matched']:13.2f}" if same_run else "")
            print(f"  {r['layer']:5d} {r['canonical']:10.2f} "
                  f"{r['stored']:8.2f} {r['matched']:8.2f}{rec}")
    print("\nmean matched cosine of centered decoded atoms, live entries")
    print(f"  {'layer':>5} {'paired':>8} {'other':>8} {'stored':>8}")
    for r in agree:
        print(f"  {r['layer']:5d} {r['paired']:8.3f} {r['other']:8.3f} "
              f"{r['stored']:8.3f}")

    out = Path(cfg.out or f"data/out/sonar/atomgram/{na}_{sa}_vs_{nb}_{sb}")
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("gram.csv", grams), ("atoms.csv", agree)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "a": cfg.a, "step_a": sa, "b": cfg.b, "step_b": sb,
        "same_run": same_run, "rows": len(X), "cache": cfg.cache,
        "mse_weights": cfg.mse_weights, "temperature": cfg.temperature},
        indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
