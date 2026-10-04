"""Verification mode for make_mse_weights.py: recompute the per-dim
inverse-variance MSE weights from a cache and compare them against the
weights a run actually trained/evaluated with.

Why this exists: make_mse_weights.py's own docstring warns that "a cache
trained with one weight vector must be evaluated with the same one", and
sonar.py records that the live data/out/sonar/mse_weights.npy was
computed from the *step-97600 census corpus embeddings* -- which is not
necessarily byte-identical to the current data/sonar_embeddings/
mc4_4M.npy. Before regenerating weights on top of an existing run, this
script answers: does the recipe over THIS cache reproduce THOSE weights?

The fresh vector is produced by shelling out to the repo-root
make_mse_weights.py itself (so the recipe cannot drift from the one the
tree actually ships); pass --fresh to compare two existing .npy files
instead and skip the recompute.

Comparison (float64):
  cosine        cosine similarity of the two weight vectors
  max_rel_dev   max over dims of |fresh - orig| / |orig|
  rel-dev p50 / p99, mean ratio fresh/orig
  scatter CSV   one row per dim: dim, w_orig, w_fresh, rel_dev

Verdict:
  PASS      cosine >= --pass-cos and max_rel_dev <= --pass-maxrel:
            the recipe over this cache reproduces the original weights;
            regenerating is safe.
  MARGINAL  cosine passes but a few dims deviate: same corpus, likely
            float32 accumulation-order or a handful of shifted rows;
            inspect the worst dims before regenerating.
  FAIL      the vectors materially differ. Most likely the original was
            computed from a different corpus snapshot (see above), in
            which case the ORIGINAL remains the correct vector for
            every existing run and cached eval -- do NOT overwrite it.

Run (from the repo root):

  uv run python experiments/verify-mse-weights/verify_mse_weights.py
  uv run python experiments/verify-mse-weights/verify_mse_weights.py \\
      --fresh /path/to/candidate.npy          # compare, no recompute
"""
import argparse
import csv
import json
import subprocess
import sys
import numpy as np
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parents[1]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy",
                   help="embedding cache to recompute weights from "
                        "(relative to the repo root)")
    p.add_argument("--original", default="data/out/sonar/mse_weights.npy",
                   help="the weight vector runs actually used")
    p.add_argument("--fresh", default=None,
                   help="existing candidate .npy; skips the recompute")
    p.add_argument("--out-dir", default=str(EXP_DIR),
                   help="where the fresh vector, scatter CSV and summary "
                        "are written")
    p.add_argument("--pass-cos", type=float, default=0.9999,
                   help="cosine threshold for PASS/MARGINAL")
    p.add_argument("--pass-maxrel", type=float, default=0.01,
                   help="max relative deviation threshold for PASS")
    p.add_argument("--worst", type=int, default=10,
                   help="how many worst dims to print")
    return p.parse_args()


def recompute(cache, out_path):
    """Run the repo's own make_mse_weights.py so the recipe cannot drift."""
    script = REPO_ROOT / "make_mse_weights.py"
    if not script.exists():
        raise SystemExit(f"make_mse_weights.py not found at {script}")
    cmd = [sys.executable, str(script), str(cache), str(out_path)]
    print(f"recomputing: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"make_mse_weights.py failed (exit {r.returncode})")
    if not Path(out_path).exists():
        raise SystemExit(f"recompute produced no file at {out_path}")


def compare(orig, fresh):
    """All comparison statistics, in float64."""
    o = np.asarray(orig, np.float64)
    f = np.asarray(fresh, np.float64)
    tiny = np.finfo(np.float64).tiny
    cos = float(o @ f / (np.linalg.norm(o) * np.linalg.norm(f) + tiny))
    rel = np.abs(f - o) / np.maximum(np.abs(o), tiny)
    return {
        "cosine": cos,
        "max_rel_dev": float(rel.max()),
        "p50_rel_dev": float(np.percentile(rel, 50)),
        "p99_rel_dev": float(np.percentile(rel, 99)),
        "mean_ratio": float((f / np.maximum(o, tiny)).mean()),
        "d": int(len(o)),
    }, rel


def verdict(stats, pass_cos, pass_maxrel):
    if stats["cosine"] >= pass_cos and stats["max_rel_dev"] <= pass_maxrel:
        return ("PASS", "the recipe over this cache reproduces the "
                "original weights; regenerating is safe.")
    if stats["cosine"] >= pass_cos:
        return ("MARGINAL", "globally aligned but individual dims deviate "
                "beyond tolerance; inspect the worst dims (float32 "
                "accumulation order, or a few changed cache rows) before "
                "regenerating.")
    return ("FAIL", "the vectors materially differ. The shipped weights "
            "were computed from the step-97600 census corpus embeddings "
            "(sonar.py comment), so a FAIL against a different cache "
            "snapshot likely means corpus mismatch, not a bug -- the "
            "ORIGINAL stays authoritative for every existing run; do not "
            "overwrite it.")


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_path = Path(cfg.original)
    if not orig_path.is_absolute():
        orig_path = REPO_ROOT / orig_path
    if not orig_path.exists():
        raise SystemExit(f"original weights not found: {orig_path}")

    if cfg.fresh:
        fresh_path = Path(cfg.fresh)
    else:
        cache = Path(cfg.cache)
        if not cache.is_absolute():
            cache = REPO_ROOT / cache
        if not cache.exists():
            raise SystemExit(f"cache not found: {cache}")
        fresh_path = out_dir / "mse_weights_fresh.npy"
        recompute(cache, fresh_path)

    orig = np.load(orig_path)
    fresh = np.load(fresh_path)
    if orig.shape != fresh.shape:
        print(f"VERDICT: FAIL -- shape mismatch: original {orig.shape} vs "
              f"fresh {fresh.shape}")
        raise SystemExit(1)

    stats, rel = compare(orig, fresh)
    v, why = verdict(stats, cfg.pass_cos, cfg.pass_maxrel)

    scatter = out_dir / "verify_scatter.csv"
    with open(scatter, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dim", "w_orig", "w_fresh", "rel_dev"])
        for i in range(len(orig)):
            w.writerow([i, f"{orig[i]:.8g}", f"{fresh[i]:.8g}",
                        f"{rel[i]:.6g}"])

    summary = dict(stats, original=str(orig_path), fresh=str(fresh_path),
                   cache=(None if cfg.fresh else cfg.cache),
                   pass_cos=cfg.pass_cos, pass_maxrel=cfg.pass_maxrel,
                   verdict=v)
    (out_dir / "verify_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"\noriginal: {orig_path}")
    print(f"fresh:    {fresh_path}")
    print(f"d={stats['d']}  cosine={stats['cosine']:.8f}  "
          f"max_rel_dev={stats['max_rel_dev']:.4g}  "
          f"p50={stats['p50_rel_dev']:.4g}  p99={stats['p99_rel_dev']:.4g}  "
          f"mean_ratio={stats['mean_ratio']:.6f}")
    worst = np.argsort(rel)[::-1][:cfg.worst]
    print(f"worst {cfg.worst} dims (dim: orig -> fresh, rel_dev):")
    for i in worst:
        print(f"  {int(i):4d}: {orig[i]:.6g} -> {fresh[i]:.6g}  "
              f"({rel[i]:.3g})")
    print(f"\nVERDICT: {v} -- {why}")
    print(f"-> {scatter}")
    print(f"-> {out_dir / 'verify_summary.json'}")
    if v == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
