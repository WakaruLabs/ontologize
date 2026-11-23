"""What each arm's coordinate axes carry that a rotation would remove, in the frame its loss weighs.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/axes.py

A dictionary trained by Adam can depend on the basis its rows are written in, so a random
rotation of a cache can change how well two seeds' codes agree. This reads each arm's rows
scaled by the square root of the MSE weights its dictionary was trained under and centered, on
the same `--rows` prefix rows in every arm (drawn as spectrum.py draws them), and asks three
things of its axes, each beside what a rotation would give:
- **marginals**: per coordinate, skew, excess kurtosis, Sarle's bimodality coefficient and the
  share of rows beyond 4 standard deviations, beside the same along `--dirs` random directions.
  An axis is outlying when its kurtosis or |skew| is above every random direction's;
- **energy**: per pair of coordinates, the Spearman correlation of their magnitudes less the same
  statistic on Gaussian rows with the arm's covariance. A Gaussian reads zero excess in every
  basis, and one random scale per row reads the same excess on the axes as along random
  directions. Per-coordinate scaling leaves the ranks as they are;
- **covariance**: the tail of the correlations between coordinates and how many coordinates the
  principal directions of the correlation matrix spread over (1/IPR), beside the correlations
  of `--rotations` random rotations of the arm's covariance, which keep its spectrum.
And from the covariance:
- **group**: the coordinates with a partner above `--threshold` in |correlation|, their pairs
  and the strongest pairs, with the group's size at each of `--thresholds`. `--reference` names
  the arm whose group and pairs every other arm's are compared with, by coordinate number, which
  means something only where the arms share coordinates (block 4 and its readouts). `--disjoint`
  reads every group again on as many prefix rows as the first sample, drawn from those it leaves
  out;
- **draw**: with `--draw K`, K coordinates of `--draw-arm` drawn at `--draw-seed` from its live
  coordinates with no partner above `--draw-below`, for weightframe.py's `--rotate-except`.

The nulls are drawn in arm order: the random directions once, and the Gaussian copies and the
rotations from one stream each across the arms, so a run of the same arms in the same order
repeats. An arm is NAME CACHE WEIGHTS; harvests are read at the cache rows they keep
(spectrum.py's row map).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from jaxtyping import Bool, Float, Int, Shaped
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectrum import file_rows, row_map  # noqa: E402

A = "data/activations/lcm-organism"
ARMS = [(name, f"{A}/{name}.npy", f"{A}/{name}.mse_weights_matched.npy") for name in (
    "lcm_b4", "lcm_readout_b4", "lcm_readout_x_b4", "lcm_readout_x_rot_b4", "sonar_ctxmean_b4",
    "sonar_ctxmean_recol_b4", "sonar_recol_b4")]
ARMS += [("sonar", "data/sonar_embeddings/mc4_sent16.npy",
          "data/out/sonar/sentcache/mse_weights_no837.npy")]
ARMS += [(name, f"{A}/{name}.npy", f"{A}/{name}.mse_weights_matched.npy") for name in (
    "init_b4", "lcm_readout_x_keep19_b4", "lcm_readout_x_rand19_b4")]
CHECKS = ("marginals", "energy", "covariance")


def weighted(x: Shaped[np.ndarray, "N d"], w: Float[np.ndarray, "d"],
             rows: Int[np.ndarray, "n"]) -> Tuple[Float[np.ndarray, "n l"], Bool[np.ndarray, "d"]]:
    """The live coordinates of the rows, scaled by the root of their weights and centered."""
    live = w > 0
    X = np.asarray(x[rows], np.float64)[:, live] * np.sqrt(w[live].astype(np.float64))
    return X - X.mean(0), live


def shape(Y: Float[np.ndarray, "n m"]) -> Dict[str, Float[np.ndarray, "m"]]:
    n = len(Y)
    Z = (Y - Y.mean(0)) / Y.std(0)
    skew = (Z ** 3).mean(0)
    kurt = (Z ** 4).mean(0) - 3.0
    bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return dict(skew=skew, kurt=kurt, bc=bc, tail4=(np.abs(Z) > 4).mean(0))


def shape_summary(s: Dict[str, np.ndarray]) -> Dict[str, float]:
    k = s["kurt"]
    q = {f"q{p}_kurt".replace(".", ""): float(np.quantile(k, p)) for p in (0.5, 0.9, 0.99, 0.999)}
    return dict(median_kurt=float(np.median(k)), mean_kurt=float(k.mean()), max_kurt=float(k.max()),
                **q, median_abs_skew=float(np.median(np.abs(s["skew"]))),
                max_abs_skew=float(np.abs(s["skew"]).max()), bimodal=int((s["bc"] > 5 / 9).sum()),
                tail4=float(s["tail4"].mean()))


def outlying(axes: Dict[str, np.ndarray], rand: Dict[str, np.ndarray]) -> Int[np.ndarray, "o"]:
    """Axes whose kurtosis or |skew| is above every random direction's."""
    return np.flatnonzero((axes["kurt"] > rand["kurt"].max())
                          | (np.abs(axes["skew"]) > np.abs(rand["skew"]).max()))


def magcorr(Y: Float[np.ndarray, "n m"]) -> Float[np.ndarray, "m m"]:
    """Spearman correlations between the columns' magnitudes."""
    R = rankdata(np.abs(Y - Y.mean(0)), axis=0).astype(np.float32)
    R -= R.mean(0)
    R /= R.std(0)
    return (R.T @ R / len(Y)).astype(np.float64)


def offdiag(M: Float[np.ndarray, "m m"]) -> Float[np.ndarray, "p"]:
    return M[np.triu_indices(len(M), 1)]


def excess(Y: Float[np.ndarray, "n m"], G: Float[np.ndarray, "n m"]) -> Float[np.ndarray, "p"]:
    """Per pair of columns, Y's magnitude correlation less the Gaussian copy G's."""
    return offdiag(magcorr(Y)) - offdiag(magcorr(G))


def gaussian_copy(X: Float[np.ndarray, "n l"], rng: np.random.Generator) -> Float[np.ndarray, "n l"]:
    lam, V = np.linalg.eigh(X.T @ X / len(X))
    return (rng.standard_normal(X.shape) * np.sqrt(np.clip(lam, 0, None))) @ V.T


def excess_summary(e: Float[np.ndarray, "p"]) -> Dict[str, float]:
    return dict(mean=float(e.mean()), median=float(np.median(e)), q99=float(np.quantile(e, 0.99)),
                q999=float(np.quantile(e, 0.999)), max=float(e.max()), min=float(e.min()),
                over_005=int((e > 0.05).sum()), over_01=int((e > 0.1).sum()))


def haar(d: int, rng: np.random.Generator) -> Float[np.ndarray, "d d"]:
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def correlations(C: Float[np.ndarray, "l l"]) -> Float[np.ndarray, "l l"]:
    sd = np.sqrt(np.diag(C))
    return C / np.outer(sd, sd)


def partners(C: Float[np.ndarray, "l l"]) -> Float[np.ndarray, "l"]:
    """Each coordinate's largest |correlation| with another."""
    return np.abs(correlations(C) - np.eye(len(C))).max(1)


def corr_stats(C: Float[np.ndarray, "l l"], threshold: float) -> Dict[str, float]:
    rho = correlations(C)
    off = np.abs(offdiag(rho))
    lam, V = np.linalg.eigh(rho)  # each coordinate at unit variance
    lam, V = lam[::-1], V[:, ::-1]
    ipr = (V ** 4).sum(0)
    sd2 = np.diag(C)
    return dict(diag_span=float(sd2.max() / sd2.min()), rho2_mean=float((off ** 2).mean()),
                rho_median=float(np.median(off)), rho_q99=float(np.quantile(off, 0.99)),
                rho_q999=float(np.quantile(off, 0.999)), rho_max=float(off.max()),
                pairs_over_03=int((off > 0.3).sum()), pairs_over_threshold=int((off > threshold).sum()),
                coords_with_partner=int((partners(C) > threshold).sum()),
                coords_top1=float(1 / ipr[0]), coords_top10=float(np.mean(1 / ipr[:10])),
                coords_top100=float(np.mean(1 / ipr[:100])),
                coords_weighted=float(1 / ((lam / lam.sum()) @ ipr)))


def group(C: Float[np.ndarray, "l l"], coords: Int[np.ndarray, "l"], threshold: float,
          top: int = 10) -> Dict[str, list]:
    """The coordinates with a partner above the threshold, in the cache's numbering, their pairs
    above it, and the strongest pairs with their signed correlations."""
    rho = correlations(C)
    i, j = np.triu_indices(len(C), 1)
    r = rho[i, j]
    strong = np.abs(r) > threshold
    best = np.argsort(-np.abs(r))[:top]
    return dict(coords=coords[partners(C) > threshold].tolist(),
                pairs=[[int(coords[a]), int(coords[b])] for a, b in zip(i[strong], j[strong])],
                strongest=[[int(coords[i[t]]), int(coords[j[t]]), float(r[t])] for t in best])


def overlap(g: Dict[str, list], ref: Dict[str, list]) -> Dict[str, int]:
    pairs = {tuple(p) for p in g["pairs"]}
    return dict(coords=len(set(g["coords"]) & set(ref["coords"])), of_coords=len(ref["coords"]),
                pairs=len(pairs & {tuple(p) for p in ref["pairs"]}), of_pairs=len(ref["pairs"]))


def draw(C: Float[np.ndarray, "l l"], coords: Int[np.ndarray, "l"], k: int, below: float,
         seed: int) -> Tuple[List[int], Dict[str, float]]:
    """k coordinates drawn from those with no partner above `below`, and how they correlate."""
    p = partners(C)
    pool = np.flatnonzero(p <= below)
    if len(pool) < k:
        raise SystemExit(f"only {len(pool)} coordinates have no partner above {below}")
    pick = np.sort(np.random.default_rng(seed).choice(pool, k, replace=False))
    sub = np.abs(correlations(C)[np.ix_(pick, pick)])[~np.eye(k, dtype=bool)]
    return coords[pick].tolist(), dict(
        pool=len(pool), excluded=int(len(p) - len(pool)), pick_partner_median=float(np.median(p[pick])),
        pool_partner_median=float(np.median(p[pool])), within_median=float(np.median(sub)),
        within_max=float(sub.max()))


def main(argv: Optional[List[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "CACHE", "WEIGHTS"),
                   help="repeatable; default: the Phase 2 and readout arms")
    p.add_argument("--checks", nargs="+", choices=CHECKS, default=list(CHECKS))
    p.add_argument("--rows", type=int, default=200_000, help="prefix rows drawn")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dirs", type=int, default=1024, help="random directions for the marginals")
    p.add_argument("--rotations", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.45, 0.5, 0.55])
    p.add_argument("--reference", default="lcm_b4")
    p.add_argument("--disjoint", action="store_true")
    p.add_argument("--draw", type=int, default=0, metavar="K")
    p.add_argument("--draw-arm", default="lcm_readout_x_b4")
    p.add_argument("--draw-below", type=float, default=0.45)
    p.add_argument("--draw-seed", type=int, default=19)
    p.add_argument("--out", default="data/out/lcm-organism/axes")
    cfg = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s
    arms = [(name, resolve(c), resolve(w)) for name, c, w in (cfg.arm or ARMS)]
    maps = {name: row_map(c) for name, c, _ in arms}
    harvests = {m for m in maps.values() if m is not None}
    if len(harvests) > 1:
        raise SystemExit(f"the harvests keep different rows: {harvests}")
    P = harvests.pop()[0] if harvests else min(np.load(c, mmap_mode="r").shape[0] for _, c, _ in arms)
    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(rng.choice(P, size=min(cfg.rows, P), replace=False))
    widths = {np.load(c, mmap_mode="r").shape[1] for _, c, _ in arms}
    if len(widths) > 1:
        raise SystemExit(f"the arms have different widths: {widths}")
    D = rng.standard_normal((widths.pop(), cfg.dirs))
    gauss = np.random.default_rng(cfg.seed + 1)
    rot = np.random.default_rng(cfg.seed + 2)
    second = None
    if cfg.disjoint:
        rest = np.setdiff1d(np.arange(P), rows)
        second = np.sort(np.random.default_rng(cfg.seed + 3).choice(rest, len(rows), replace=False))
    out = resolve(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    results, covs, coord_of = {}, {}, {}
    for name, c, wt in arms:
        x = np.load(c, mmap_mode="r")
        X, live = weighted(x, np.load(wt), file_rows(rows, maps[name]))
        coords = coord_of[name] = np.flatnonzero(live)
        res = results[name] = {"cache": str(c), "weights": str(wt), "live": int(live.sum())}
        arrays = {}
        if "marginals" in cfg.checks:
            Dl = D[live] / np.linalg.norm(D[live], axis=0)
            axes, rand = shape(X), shape(X @ Dl)
            o = outlying(axes, rand)
            res["marginals"] = dict(axes=shape_summary(axes), random=shape_summary(rand),
                                    outlying=[[int(coords[i]), float(axes["kurt"][i]),
                                               float(axes["skew"][i])] for i in o])
            arrays |= {f"axes_{k}": v for k, v in axes.items()} | {f"random_{k}": v for k, v in rand.items()}
        if "energy" in cfg.checks:
            Dl = D[live] / np.linalg.norm(D[live], axis=0)
            G = gaussian_copy(X, gauss)
            ea, er = excess(X, G), excess(X @ Dl, G @ Dl)
            res["energy"] = dict(axes=excess_summary(ea), random=excess_summary(er),
                                 axes_abs_rho_median=float(np.median(np.abs(offdiag(np.corrcoef(X.T))))))
            arrays |= dict(energy_axes=ea, energy_random=er)
        if "covariance" in cfg.checks:
            C = covs[name] = X.T @ X / len(X)
            rots = [corr_stats((Q := haar(len(C), rot)).T @ C @ Q, cfg.threshold)
                    for _ in range(cfg.rotations)]
            res["covariance"] = dict(axes=corr_stats(C, cfg.threshold), rotations=rots)
            res["group"] = group(C, coords, cfg.threshold)
            res["group"]["sizes"] = {str(t): int((partners(C) > t).sum()) for t in cfg.thresholds}
            if second is not None:
                X2, _ = weighted(x, np.load(wt), file_rows(second, maps[name]))
                g2 = group(X2.T @ X2 / len(X2), coords, cfg.threshold)["coords"]
                res["group"]["disjoint"] = dict(coords=g2, same=g2 == res["group"]["coords"])
            arrays |= dict(cov=C, coords=coords)
        np.savez(out / f"{name}.npz", **arrays)
        print(f"{name}: {res['live']} live coordinates read", flush=True)
    report = {"meta": {"prefix_end": int(P), "rows": len(rows), "seed": cfg.seed, "dirs": cfg.dirs,
                       "rotations": cfg.rotations, "threshold": cfg.threshold}, "arms": results}
    if "covariance" in cfg.checks and cfg.reference in results:
        ref = results[cfg.reference]["group"]
        for name, r in results.items():
            if name != cfg.reference:
                r["group"]["overlap_with_reference"] = overlap(r["group"], ref)
    if cfg.draw:
        if "covariance" not in cfg.checks or cfg.draw_arm not in covs:
            raise SystemExit("--draw needs the covariance check on --draw-arm")
        pick, about = draw(covs[cfg.draw_arm], coord_of[cfg.draw_arm], cfg.draw, cfg.draw_below,
                           cfg.draw_seed)
        report["draw"] = dict(arm=cfg.draw_arm, k=cfg.draw, below=cfg.draw_below, seed=cfg.draw_seed,
                              coords=pick, **about)
    print(table(results))
    if "draw" in report:
        print(f"draw from {cfg.draw_arm}: {','.join(map(str, report['draw']['coords']))}")
    (out / "axes.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def table(results: Dict[str, Dict]) -> str:
    lines = []
    for name, r in results.items():
        bits = [f"{name:26s}"]
        if "marginals" in r:
            a, b = r["marginals"]["axes"], r["marginals"]["random"]
            bits.append(f"kurt median {a['median_kurt']:.2f}/{b['median_kurt']:.2f} "
                        f"q99.9 {a['q0999_kurt']:.2f}/{b['q0999_kurt']:.2f} "
                        f"outlying {len(r['marginals']['outlying'])}")
        if "energy" in r:
            a, b = r["energy"]["axes"], r["energy"]["random"]
            bits.append(f"excess mean {a['mean']:.3f}/{b['mean']:.3f} max {a['max']:.3f}/{b['max']:.3f} "
                        f">0.1 {a['over_01']}/{b['over_01']}")
        if "covariance" in r:
            a, rots = r["covariance"]["axes"], r["covariance"]["rotations"]
            m = lambda k: np.mean([s[k] for s in rots])
            bits.append(f"|rho| max {a['rho_max']:.2f}/{m('rho_max'):.2f} "
                        f"partnered {a['coords_with_partner']}/{m('coords_with_partner'):.1f} "
                        f"top 1/IPR {a['coords_top1']:.0f}/{m('coords_top1'):.0f}")
        lines.append(" | ".join(bits))
    return "axes/random (marginals, energy) or axes/rotated mean (covariance):\n" + "\n".join(lines)


if __name__ == "__main__":
    main()
