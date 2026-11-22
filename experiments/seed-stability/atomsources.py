"""Do a stack's atoms point along FastICA's sources?

FastICA's sources of the training rows are of two kinds
(`notes.md`, 2026-11-22). The skewed ones reproduce on a disjoint
sample. The symmetric ones are mostly one scale shared by every source,
and do not. A source's direction in the data is its row of the mixing
matrix (`kmeans_seeds.mixing`, in the loss's whitened frame). A head's
entry writes its decoded atom there (`headcontrib.atoms_and_codes`). Each
atom is centered on its head's usage-weighted mean atom, so what is
compared is how the head's entries differ from one another. Only entries
selected on at least one of the rows count.

Per run and layer:
  - for every source, its best |cosine| with any of the layer's atoms;
  - for every atom, its best |cosine| with any source.
Beside the sources are two references in the same frame: the sources
rotated at random (the same spectrum and no ICA) and the principal
directions. With a second run (`--b`) it also gives, per source, whether
both runs have an atom along it, beside what independence of the two
would give.

Sources are classed from `ica_seeds.py`'s `sources.csv`:
  - skewed: excess kurtosis >= 5 and |skew| >= 1;
  - symmetric: kurtosis < 3 and |skew| < 0.3;
  - other: the rest.
The file's matched correlation across samples is read beside each
source's best cosine.

    JAX_PLATFORMS=cpu uv run python experiments/seed-stability/atomsources.py \\
        --frame data/out/sonar/kmeans_seeds_ica/frame_ica.iter800.npz \\
        --sources data/out/sonar/ica_seeds/sources.csv \\
        --a <run> --b <run>_s43 --out <dir>
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))
sys.path.insert(0, str(HERE))

from kmeans_seeds import mixing  # noqa: E402

CLASSES = ("skewed", "symmetric", "other")
THRESHOLDS = (0.3, 0.5, 0.7)


def unit(A: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
    return A / np.maximum(np.linalg.norm(A, axis=-1, keepdims=True), 1e-30)


def best_abs_cos(A: Float[np.ndarray, "a d"], B: Float[np.ndarray, "b d"]
                 ) -> tuple[Float[np.ndarray, "b"], Float[np.ndarray, "a"]]:
    """For each row of B its best |cosine| with any row of A, and for each
    row of A its best with any row of B."""
    C = np.abs(unit(A) @ unit(B).T)
    return C.max(0), C.max(1)


def source_classes(kurtosis: Float[np.ndarray, "p"], skew: Float[np.ndarray, "p"]
                   ) -> np.ndarray:
    """Each source's class: skewed, symmetric or other."""
    out = np.full(len(kurtosis), "other", dtype=object)
    out[(kurtosis >= 5) & (np.abs(skew) >= 1)] = "skewed"
    out[(kurtosis < 3) & (np.abs(skew) < 0.3)] = "symmetric"
    return out


def rotated_mixing(fit: dict, seed: int = 0) -> Float[np.ndarray, "p d"]:
    """The mixing rows of a random rotation of the sources in the sphered
    frame: directions with the rows' spectrum and nothing ICA found."""
    p = len(fit["eigenvalues"])
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((p, p)))[0]
    return Q @ (fit["eigenvectors"] * np.sqrt(fit["eigenvalues"])).T


def centered_live(D: Float[np.ndarray, "h k d"], usage: Float[np.ndarray, "h k"]
                  ) -> Float[np.ndarray, "n d"]:
    """Each head's atoms less its usage-weighted mean atom, the entries
    used on at least one row only."""
    p = usage / np.maximum(usage.sum(-1, keepdims=True), 1e-30)
    C = D - (p[..., None] * D).sum(1, keepdims=True)
    return C[usage > 0]


def coverage(best: Float[np.ndarray, "p"], mask: Bool[np.ndarray, "p"]) -> dict:
    """The median best |cosine| over the masked directions and the share
    at or above each threshold."""
    b = best[mask]
    return {"directions": int(mask.sum()), "median": float(np.median(b)),
            **{f"share_{t:g}": float((b >= t).mean()) for t in THRESHOLDS}}


def spearman(x: Float[np.ndarray, "n"], y: Float[np.ndarray, "n"]) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def layer_table(atoms: Float[np.ndarray, "n d"], refs: dict, cls: np.ndarray,
                corr: Float[np.ndarray, "p"], var: Float[np.ndarray, "p"]
                ) -> tuple[dict, Float[np.ndarray, "p"]]:
    """One layer's coverage of each reference's directions and its atoms'
    best cosines; returns the table and each source's best cosine."""
    out, best_src = {"atoms": int(len(atoms))}, None
    for name, M in refs.items():
        per_dir, per_atom = best_abs_cos(atoms, M)
        if name == "sources":
            best_src = per_dir
            for c in CLASSES:
                out[c] = coverage(per_dir, cls == c)
            out["atoms_best_with_sources"] = float(np.median(per_atom))
            nearest = np.abs(unit(atoms) @ unit(M).T).argmax(1)
            out["atoms_nearest_class"] = {c: float((cls[nearest] == c).mean())
                                          for c in CLASSES}
            out["spearman_with_reproduction"] = spearman(per_dir, corr)
            out["spearman_with_variance"] = spearman(per_dir, var)
        else:
            out[name] = coverage(per_dir, np.ones(len(M), bool))
            out[f"atoms_best_with_{name}"] = float(np.median(per_atom))
    return out, best_src


def both(best_a: Float[np.ndarray, "p"], best_b: Float[np.ndarray, "p"],
         mask: Bool[np.ndarray, "p"], t: float) -> dict:
    """The share of masked sources both runs have an atom along (best
    |cosine| >= t), beside the product of each run's share."""
    a, b = best_a[mask] >= t, best_b[mask] >= t
    return {"both": float((a & b).mean()), "independent": float(a.mean() * b.mean())}


def run_atoms(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d"],
              sw: Float[np.ndarray, "d"], batch: int) -> tuple[list, int]:
    """Per layer, the run's live centered atoms in the whitened frame."""
    from headcontrib import atoms_and_codes
    D, I, _, layer, used = atoms_and_codes(ckpt, step, T, X, sw, batch)
    k = D.shape[1]
    U = np.stack([np.bincount(row, minlength=k) for row in I]) / I.shape[1]
    return [centered_live(D[layer == i].astype(np.float64), U[layer == i])
            for i in range(int(layer.max()) + 1)], used


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frame", required=True, help="kmeans_seeds.py's frame_ica.npz")
    p.add_argument("--sources", required=True, help="ica_seeds.py's sources.csv")
    p.add_argument("--a", required=True)
    p.add_argument("--b", default=None)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default=str(ROOT / "data/sonar_embeddings/mc4_sent16.npy"))
    p.add_argument("--mse-weights",
                   default=str(ROOT / "data/out/sonar/sentcache/mse_weights_med837.npy"),
                   help="the weights the frame was fitted under and the runs trained on")
    p.add_argument("--rows", type=int, default=32768,
                   help="tail rows that decide which entries are used")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", required=True)
    cfg = p.parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    fit = dict(np.load(cfg.frame))
    src = np.genfromtxt(cfg.sources, delimiter=",", names=True)
    cls = source_classes(src["kurtosis"], src["skew"])
    refs = {"sources": mixing(fit), "rotated": rotated_mixing(fit),
            "principal": fit["eigenvectors"].T}
    sw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)

    summary = {"classes": {c: int((cls == c).sum()) for c in CLASSES}, "runs": {}}
    best = {}
    for name, ckpt in (("a", cfg.a), ("b", cfg.b)):
        if ckpt is None:
            continue
        layers, used = run_atoms(ckpt, cfg.step, cfg.temperature, X, sw, cfg.batch)
        summary["runs"][name] = {"ckpt": ckpt, "step": used, "layers": {}}
        best[name] = []
        for i, A in enumerate(layers):
            t, b = layer_table(A, refs, cls, src["corr"], src["variance_share"])
            summary["runs"][name]["layers"][i] = t
            best[name].append(b)
            print(f"{name} layer {i}: {t['atoms']} atoms; sources with an atom at "
                  f"|cos| >= 0.5: skewed {t['skewed']['share_0.5']:.3f}, symmetric "
                  f"{t['symmetric']['share_0.5']:.3f}, rotated "
                  f"{t['rotated']['share_0.5']:.3f}, principal "
                  f"{t['principal']['share_0.5']:.3f}; median best: skewed "
                  f"{t['skewed']['median']:.3f}, symmetric {t['symmetric']['median']:.3f}, "
                  f"rotated {t['rotated']['median']:.3f}, principal "
                  f"{t['principal']['median']:.3f}", flush=True)
    if "b" in best:
        summary["both"] = {}
        for i, (ba, bb) in enumerate(zip(best["a"], best["b"])):
            summary["both"][i] = {
                "spearman": spearman(ba, bb),
                **{c: {f"{t:g}": both(ba, bb, cls == c, t) for t in THRESHOLDS}
                   for c in CLASSES}}
            s = summary["both"][i]
            print(f"layer {i}: both runs at |cos| >= 0.5, skewed {s['skewed']['0.5']['both']:.3f} "
                  f"(independent {s['skewed']['0.5']['independent']:.3f}); Spearman of "
                  f"the two runs' best cosines over sources {s['spearman']:.3f}", flush=True)
    with open(out / "sources.csv", "w") as f:
        cols = [f"{n}_l{i}" for n in best for i in range(len(best[n]))]
        f.write("source,class,kurtosis,skew,variance_share,corr," + ",".join(cols) + "\n")
        for j in range(len(cls)):
            f.write(f"{j},{cls[j]},{src['kurtosis'][j]:.6g},{src['skew'][j]:.6g},"
                    f"{src['variance_share'][j]:.6g},{src['corr'][j]:.6g},"
                    + ",".join(f"{best[n][i][j]:.6g}" for n in best
                               for i in range(len(best[n]))) + "\n")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
