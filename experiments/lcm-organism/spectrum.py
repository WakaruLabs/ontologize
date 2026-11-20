"""The covariance spectrum of each arm's input, in the frame its loss weighs.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/spectrum.py

A dictionary trained on a source of low effective dimension reconstructs it
better, and two seeds' codes may agree more for that reason alone. This reads
each arm's rows scaled by the square root of the MSE weights its dictionary
was trained under (so the eigenvalues are what FVU_w divides by), on the same
mc4_sent16 rows in every arm:
- **prefix**: `--rows` rows drawn from the cache rows a harvest keeps under
  their own numbers, which the dictionaries train on;
- **tail**: the last `--tail` rows, the held-out documents seedmatch.py and
  eval_next.py score.

Per arm and row set, of the total covariance and of the pooled covariance
within each language (labels with at least `--min-rows` rows in the set, as
seedmatch.py's within-label score takes them):
- the participation ratio (sum lam)^2 / sum lam^2 and the entropy effective
  rank exp H(lam / sum lam);
- the directions needed for 50, 80, 90, 95 and 99% of the variance, and the
  share in the top 1, 10 and 100;
- pareto.gaussian_reference's FVU at each `--bits` rate: what reverse
  water-filling over a Gaussian with this spectrum leaves, a reference for
  how compressible the spectrum alone makes each source;
- the share of the variance between languages, 1 - tr(within) / tr(total).

An arm is NAME CACHE WEIGHTS. A harvest (harvest_lcm.py's meta.json with
`rows`) maps cache row r to its row r below its prefix end P and to P + r - T
from its tail start T on; any other cache is read at the cache row itself.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from jaxtyping import Float, Int, Shaped

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pareto import gaussian_reference  # noqa: E402

ARMS = [
    ("lcm_b4", "data/activations/lcm-organism/lcm_b4.npy",
     "data/activations/lcm-organism/lcm_b4.mse_weights_matched.npy"),
    ("init_b4", "data/activations/lcm-organism/init_b4.npy",
     "data/activations/lcm-organism/init_b4.mse_weights_matched.npy"),
    ("sonar", "data/sonar_embeddings/mc4_sent16_f16.npy",
     "data/out/sonar/sentcache/mse_weights_no837.npy"),
]
SHARES = (0.5, 0.8, 0.9, 0.95, 0.99)
TOPS = (1, 10, 100)


def stem(path: Path) -> Path:
    return path.with_suffix("")


def row_map(path: Path) -> Optional[Tuple[int, int]]:
    """A harvest's (P, T): its prefix end and its tail's first cache row."""
    meta = Path(f"{stem(path)}.meta.json")
    if not meta.exists():
        return None
    rows = json.loads(meta.read_text()).get("rows")
    if not rows:
        return None
    return int(rows["prefix"][1]), int(rows["tail"][0])


def file_rows(cache_rows: Int[np.ndarray, "n"],
              mapping: Optional[Tuple[int, int]]) -> Int[np.ndarray, "n"]:
    if mapping is None:
        return cache_rows
    P, T = mapping
    if ((cache_rows >= P) & (cache_rows < T)).any():
        raise SystemExit("rows between a harvest's prefix and tail are not in it")
    return np.where(cache_rows < P, cache_rows, P + cache_rows - T)


def moments(x: Shaped[np.ndarray, "n d"], w: Float[np.ndarray, "d"],
            labels: Shaped[np.ndarray, "n"], rows: Int[np.ndarray, "m"],
            min_rows: int, chunk: int = 16384,
            ) -> Tuple[Float[np.ndarray, "d d"], Float[np.ndarray, "d d"], Dict[str, int]]:
    """The total covariance and the pooled within-label covariance of
    sqrt(w) * x over `rows` (sorted), in float64. Labels with fewer than
    `min_rows` of the rows are left out of the within-label covariance."""
    names, lab = np.unique(np.asarray(labels[rows]), return_inverse=True)
    counts = np.bincount(lab, minlength=len(names))
    kept = counts >= min_rows
    d = x.shape[1]
    sw = np.sqrt(np.asarray(w, np.float64))
    S = np.zeros((d, d))
    S_rare = np.zeros((d, d))
    s_lab = np.zeros((len(names), d))
    for lo in range(0, len(rows), chunk):
        r = rows[lo:lo + chunk]
        if r[-1] - r[0] + 1 == len(r):
            a = np.asarray(x[r[0]:r[-1] + 1], np.float64) * sw
        else:
            a = np.asarray(x[r], np.float64) * sw
        lb = lab[lo:lo + chunk]
        S += a.T @ a
        s_lab += (lb[:, None] == np.arange(len(names))).T.astype(np.float64) @ a
        rare = ~kept[lb]
        if rare.any():
            S_rare += a[rare].T @ a[rare]
    n = len(rows)
    mu = s_lab.sum(0) / n
    total = S / n - np.outer(mu, mu)
    n_kept = int(counts[kept].sum())
    s_k = s_lab[kept]
    within = (S - S_rare - (s_k.T / counts[kept]) @ s_k) / n_kept
    return total, within, {"rows": n, "labels": int(kept.sum()), "label_rows": n_kept}


def stats(lam: Float[np.ndarray, "d"], bits: List[float]) -> Dict[str, object]:
    lam = np.sort(np.clip(lam, 0, None))[::-1]
    p = lam / lam.sum()
    cum = np.cumsum(p)
    nz = p[p > 0]
    out = {
        "trace": float(lam.sum()),
        "participation": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "erank": float(np.exp(-(nz * np.log(nz)).sum())),
        "k_share": {str(q): int(np.searchsorted(cum, q) + 1) for q in SHARES},
        "top_share": {str(t): float(cum[min(t, len(cum)) - 1]) for t in TOPS},
        "gaussian_fvu": {str(b): float(f) for b, f in
                         zip(bits, gaussian_reference(lam, np.asarray(bits)))},
    }
    return out


def measure(x: Shaped[np.ndarray, "n d"], w: Float[np.ndarray, "d"],
            labels: Shaped[np.ndarray, "n"], rows: Int[np.ndarray, "m"],
            min_rows: int, bits: List[float],
            ) -> Tuple[Dict[str, object], Float[np.ndarray, "d"], Float[np.ndarray, "d"]]:
    total, within, counts = moments(x, w, labels, rows, min_rows)
    lam_t = np.linalg.eigvalsh(total)[::-1]
    lam_w = np.linalg.eigvalsh(within)[::-1]
    res = {**counts, "total": stats(lam_t, bits), "within": stats(lam_w, bits),
           "language_share": float(1 - np.trace(within) / np.trace(total))}
    return res, lam_t, lam_w


def table(results: Dict[str, Dict[str, Dict]], bits: List[float]) -> str:
    head = ("arm", "PR", "erank", *(f"k{int(q * 100)}" for q in SHARES),
            *(f"top{t}" for t in TOPS), *(f"D({b:g})" for b in bits))
    lines = []
    for where in ("prefix", "tail"):
        for kind in ("total", "within"):
            lines.append(f"\n{where}, {kind}" + ("" if kind == "total" else
                         " language (pooled)"))
            lines.append("  ".join(f"{h:>8}" for h in head))
            for arm, res in results.items():
                r = res[where]
                s = r[kind]
                cells = (arm, f"{s['participation']:.1f}", f"{s['erank']:.1f}",
                         *(str(s["k_share"][str(q)]) for q in SHARES),
                         *(f"{s['top_share'][str(t)]:.3f}" for t in TOPS),
                         *(f"{s['gaussian_fvu'][str(b)]:.3f}" for b in bits))
                lines.append("  ".join(f"{c:>8}" for c in cells))
        lines.append("  language share of the variance: " + ", ".join(
            f"{arm} {res[where]['language_share']:.3f}" for arm, res in results.items()))
    return "\n".join(lines)


def main(argv=None) -> Dict[str, Dict[str, Dict]]:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "CACHE", "WEIGHTS"),
                   help="repeatable; default: the Phase 2 arms (lcm_b4, init_b4, sonar)")
    p.add_argument("--rows", type=int, default=262144, help="prefix rows drawn")
    p.add_argument("--tail", type=int, default=262144, help="last rows read")
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument("--bits", type=float, nargs="+", default=[64, 256, 1024, 1900],
                   help="rates for the Gaussian reference, bits per row (1900 = "
                        "380 heads of 32 entries, the most one such layer sends)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/lcm-organism/spectrum",
                   help="directory for spectrum.json and the eigenvalues (spectrum.npz)")
    cfg = p.parse_args(argv)
    arms = cfg.arm or ARMS
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s
    caches = {name: (resolve(c), resolve(wt)) for name, c, wt in arms}
    maps = {name: row_map(c) for name, (c, _) in caches.items()}
    sizes = {name: np.load(c, mmap_mode="r").shape[0] for name, (c, _) in caches.items()}
    harvests = {m for m in maps.values() if m is not None}
    if len(harvests) > 1:
        raise SystemExit(f"the harvests keep different rows: {harvests}")
    # cache rows: the prefix every harvest keeps, and the cache's last rows
    if harvests:
        P, T = harvests.pop()
        n_cache = {T + sizes[k] - maps[k][0] for k in maps if maps[k] is not None}
        n_cache |= {sizes[k] for k in maps if maps[k] is None}
        if len(n_cache) != 1:
            raise SystemExit(f"the arms end at different cache rows: {n_cache}")
        n_cache = n_cache.pop()
    else:
        P = n_cache = min(sizes.values())
    rng = np.random.default_rng(cfg.seed)
    sets = {"prefix": np.sort(rng.choice(P, size=min(cfg.rows, P), replace=False)),
            "tail": np.arange(n_cache - cfg.tail, n_cache)}
    results, eig = {}, {}
    for name, (c, wt) in caches.items():
        x = np.load(c, mmap_mode="r")
        labels = np.load(f"{stem(c)}.langs.npy", mmap_mode="r")
        w = np.load(wt)
        results[name] = {"cache": str(c), "weights": str(wt)}
        for where, cache_rows in sets.items():
            rows = file_rows(cache_rows, maps[name])
            res, lam_t, lam_w = measure(x, w, labels, rows, cfg.min_rows, cfg.bits)
            results[name][where] = res
            eig[f"{name}_{where}_total"] = lam_t
            eig[f"{name}_{where}_within"] = lam_w
            print(f"{name} {where}: {res['rows']} rows, PR {res['total']['participation']:.1f}",
                  flush=True)
    print(table(results, cfg.bits))
    out = resolve(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"prefix_end": int(P), "cache_rows": int(n_cache), "rows": cfg.rows,
            "tail": cfg.tail, "min_rows": cfg.min_rows, "bits": cfg.bits, "seed": cfg.seed}
    (out / "spectrum.json").write_text(json.dumps({"meta": meta, "arms": results},
                                                  indent=2) + "\n")
    np.savez(out / "spectrum.npz", **eig)
    return results


if __name__ == "__main__":
    main()
