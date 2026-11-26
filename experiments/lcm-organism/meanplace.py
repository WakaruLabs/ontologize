"""Where each arm's mean lies, and whether centering changes what the heads can cut.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/meanplace.py \\
        --arms lcm_wf sonar_wf lcm_readout_x_wf

Each arm is a cache laid out as a harvest, `<harvest dir>/<arm>_b4.npy`, read
as the trainer reads it: the gain-shape layer normalizes each row as written,
about the origin, so an arm's mean is part of every unit shape its classifier
sees. On the same rows of every arm (prefix rows for the moments, rows of the
scored tail for the k-means readings), with m the rows' mean and v = y - m:

- **Size.** The mean's share of a row's squared norm, and the spread (CV) of
  the rows' norms about the origin and about the mean.
- **Against the covariance.** The variance along the mean's direction over the
  average direction's; the mean's share in the leading and trailing principal
  directions; its Mahalanobis distance from the origin, over the directions
  whose variance is at least 1e-3 of the average (the rest would only measure
  rounding).
- **Against the coordinate axes.** How many coordinates the mean spreads over,
  the participation ratio of its squared components (1 on one coordinate, n in
  equal shares over n); the share on its largest coordinate (and which it is)
  and on its largest 19; how far the coordinates sit from zero, |m_j| / sd_j,
  at most and how many beyond 1.
- **What a row's norm says about its shape.** The share of the unit shape's
  variance that a linear function of the log norm explains, uncentered and
  centered.
- **Whether centering changes the cells.** The held-out FVU of the
  least-squares affine map from the centered unit shapes v / |v| to the
  uncentered y / |y|, and back. A classifier that is quadratic in its input
  with a constant coordinate draws the same cells under an affine change of
  that input, so near zero means centering changes the parametrization rather
  than what can be drawn. Beside it, spherical k-means (k-means++ seeding on
  cosine distance) on each, fit on prefix rows at several seeds and read on
  tail rows: the NMI between seeds (mutual information over the mean of the two
  entropies), over all rows and within each language with enough rows, each
  beside a null that shuffles one seed's cells.
- **Languages.** Each language's own mean as a share of its rows' squared norm,
  uncentered and after the global mean is removed, and the share of the
  variation between languages.

The output is one JSON object per arm, written after each arm, so a rerun
skips the arms it has.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from jaxtyping import Float, Int, Shaped

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from spectrum import row_map  # noqa: E402


def r2_on(U: Float[np.ndarray, "n d"], z: Float[np.ndarray, "n"]) -> float:
    """The pooled share of U's variance that a linear function of the scalar z explains."""
    zc = z - z.mean()
    Uc = U - U.mean(0)
    cov = zc @ Uc / len(z)
    return float((cov ** 2).sum() / zc.var() / Uc.var(0).sum())


def affine_fvu(X: Float[np.ndarray, "n d"], T: Float[np.ndarray, "n e"],
               frac: float = 2 / 3, eps: float = 1e-6) -> float:
    """The pooled FVU, on the last third of the rows, of the least-squares affine map X -> T
    fitted on the first two thirds (a ridge of eps times the mean eigenvalue, for rank)."""
    n = int(len(X) * frac)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    G = Xa[:n].T @ Xa[:n]
    G += eps * np.trace(G) / len(G) * np.eye(len(G))
    B = np.linalg.solve(G, Xa[:n].T @ T[:n])
    R = T[n:] - Xa[n:] @ B
    return float((R ** 2).sum() / ((T[n:] - T[:n].mean(0)) ** 2).sum())


def nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"], k: int) -> float:
    """Mutual information over the mean of the two entropies, in nats."""
    c = np.zeros((k, k))
    np.add.at(c, (a, b), 1)
    p = c / c.sum()
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum()

    def h(q: Float[np.ndarray, "k"]) -> float:
        return float(-(q[q > 0] * np.log(q[q > 0])).sum())

    return float(mi / ((h(pa) + h(pb)) / 2))


def skmeans(X: Float[np.ndarray, "n d"], k: int, seed: int, iters: int = 50
            ) -> Float[np.ndarray, "k d"]:
    """Spherical k-means on unit rows X: k-means++ seeding on cosine distance, then
    alternating nearest-centre assignment and renormalized means; an emptied centre restarts at
    the row its nearest centre fits worst."""
    rng = np.random.default_rng(seed)
    n = len(X)
    C = [X[rng.integers(n)]]
    dist = 1 - X @ C[0]
    for _ in range(1, k):
        p = np.maximum(dist, 0)
        C.append(X[rng.choice(n, p=p / p.sum())])
        dist = np.minimum(dist, 1 - X @ C[-1])
    C = np.stack(C)
    lab = None
    for _ in range(iters):
        S = X @ C.T
        new = S.argmax(1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        C = np.eye(k, dtype=X.dtype)[lab].T @ X
        for j in np.flatnonzero(np.linalg.norm(C, axis=1) == 0):
            C[j] = X[S.max(1).argmin()]
        C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C


def kmeans_agreement(fit: Float[np.ndarray, "n d"], tail: Float[np.ndarray, "t d"],
                     langs: Shaped[np.ndarray, "t"], k: int = 32, seeds: int = 4,
                     min_rows: int = 300) -> Dict[str, object]:
    """Spherical k-means fit on unit rows `fit` at each seed and assigned on unit rows `tail`:
    the mean NMI over pairs of seeds on every tail row, and the row-weighted mean of it within
    each language with at least `min_rows` rows, each beside a null that shuffles the second
    seed's cells."""
    labs = [np.argmax(tail @ skmeans(fit, k, s).T, 1) for s in range(seeds)]
    pairs = [(i, j) for i in range(seeds) for j in range(i + 1, seeds)]
    rng = np.random.default_rng(0)
    out: Dict[str, object] = {
        "all": float(np.mean([nmi(labs[i], labs[j], k) for i, j in pairs])),
        "null": float(np.mean([nmi(labs[i], rng.permutation(labs[j]), k) for i, j in pairs]))}
    within, null_w, wts = [], [], []
    for lang in np.unique(langs):
        r = langs == lang
        if r.sum() < min_rows:
            continue
        within.append(np.mean([nmi(labs[i][r], labs[j][r], k) for i, j in pairs]))
        null_w.append(np.mean([nmi(labs[i][r], rng.permutation(labs[j][r]), k)
                               for i, j in pairs]))
        wts.append(int(r.sum()))
    avg = (lambda v: float(np.average(v, weights=wts)) if wts else None)  # noqa: E731
    out.update(within=avg(within), within_null=avg(null_w), within_langs=len(wts))
    return out


def geometry(Y: Float[np.ndarray, "n d"], langs: Shaped[np.ndarray, "n"], min_rows: int = 300
             ) -> Tuple[Dict[str, object], Float[np.ndarray, "d"]]:
    """The mean's size and place, and the shapes' relation to the norms (the module docstring),
    with the mean itself."""
    m = Y.mean(0)
    V = Y - m
    ny, nv = np.linalg.norm(Y, axis=1), np.linalg.norm(V, axis=1)
    mn = float(np.linalg.norm(m))
    mh = m / mn
    o: Dict[str, object] = {
        "mean_norm": mn, "rms_row": float(np.sqrt((ny ** 2).mean())),
        "share_mean": mn ** 2 / float((ny ** 2).mean()),
        "cv_norm_origin": float(ny.std() / ny.mean()),
        "cv_norm_mean": float(nv.std() / nv.mean())}
    cos = (Y @ mh) / ny
    o["cos_to_mean"] = [float(cos.mean()), float(cos.std())]
    along = V @ mh
    across = np.linalg.norm(V - np.outer(along, mh), axis=1)
    o["corr_along_vs_across"] = float(np.corrcoef(along, across)[0, 1])
    Cv = V.T @ V / len(V)
    lam, E = np.linalg.eigh(Cv)
    lam, E = lam[::-1], E[:, ::-1]
    live = lam > 1e-9 * lam[0]
    nl = int(live.sum())
    o["d_live"] = nl
    o["participation_ratio"] = float(lam[live].sum() ** 2 / (lam[live] ** 2).sum())
    o["var_along_mean_over_avg"] = float(mh @ Cv @ mh / lam[live].mean())
    proj = (E.T @ m) ** 2 / mn ** 2
    for kk in (1, 10, 100):
        o[f"mean_in_top{kk}_pcs"] = float(proj[:kk].sum())
    o["mean_in_bottom_half_pcs"] = float(proj[nl // 2:nl].sum())
    keep = lam > 1e-3 * lam[live].mean()
    o["mahalanobis_of_mean"] = float(np.sqrt(((E[:, keep].T @ m) ** 2 / lam[keep]).sum()))
    m2 = m ** 2
    order = np.argsort(m2)[::-1]
    s = m2[order] / m2.sum()
    o["mean_axis_pr"] = float(m2.sum() ** 2 / (m2 ** 2).sum())
    o["mean_axis_top1"], o["mean_axis_top19"] = float(s[0]), float(s[:19].sum())
    o["mean_axis_top1_coord"] = int(order[0])
    sd = np.sqrt(np.diag(Cv))
    off = np.where(sd > 0, np.abs(m) / np.where(sd > 0, sd, 1), 0)
    o["coord_mean_over_sd_max"] = float(off.max())
    o["coords_mean_over_sd_gt1"] = int((off > 1).sum())
    U, Uc = Y / ny[:, None], V / nv[:, None]
    o["shape_from_log_norm"] = {
        "uncentered_by_origin_norm": r2_on(U, np.log(ny)),
        "uncentered_by_mean_dist": r2_on(U, np.log(nv)),
        "centered_by_mean_dist": r2_on(Uc, np.log(nv)),
        "centered_by_origin_norm": r2_on(Uc, np.log(ny))}
    o["affine_fvu"] = {"centered_to_uncentered": affine_fvu(Uc, U),
                       "uncentered_to_centered": affine_fvu(U, Uc)}
    su, sc, wts = [], [], []
    between = 0.0
    for lang in np.unique(langs):
        r = langs == lang
        ml = Y[r].mean(0)
        between += r.sum() * float(((ml - m) ** 2).sum())
        if r.sum() < min_rows:
            continue
        su.append(float((ml ** 2).sum() / (ny[r] ** 2).mean()))
        sc.append(float(((ml - m) ** 2).sum() / (nv[r] ** 2).mean()))
        wts.append(int(r.sum()))
    avg = (lambda v: float(np.average(v, weights=wts)) if wts else None)  # noqa: E731
    o["language"] = {"own_mean_share_uncentered": avg(su), "own_mean_share_centered": avg(sc),
                     "between_share": between / float((nv ** 2).sum()), "langs": len(wts)}
    return o, m


def unit(X: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
    """Rows scaled to unit norm, in float32 for the k-means products."""
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def sample_rows(P: int, n: int, fit: int, tail: int, scored: int, seed: int
                ) -> Tuple[Int[np.ndarray, "f"], Int[np.ndarray, "t"]]:
    """Sorted prefix rows for the moments and sorted rows of the scored tail, the last `scored`
    of the n harvest rows."""
    if fit > P or tail > scored or scored > n - P:
        raise SystemExit(f"cannot draw {fit} of {P} prefix rows and {tail} of the last "
                         f"{scored} of {n - P} tail rows")
    rng = np.random.default_rng(seed)
    fit_rows = np.sort(rng.choice(P, fit, replace=False))
    tail_rows = np.sort(rng.choice(np.arange(n - scored, n), tail, replace=False))
    return fit_rows, tail_rows


def main(argv: Optional[List[str]] = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["lcm_wf", "sonar_wf", "lcm_readout_x_wf"],
                   help="caches by name: <dir>/<arm>_b4.npy")
    p.add_argument("--dir", default="data/activations/lcm-organism")
    p.add_argument("--fit", type=int, default=60_000, help="prefix rows for the moments")
    p.add_argument("--tail", type=int, default=40_000, help="scored-tail rows for k-means")
    p.add_argument("--kfit", type=int, default=40_000, help="of the prefix rows, k-means' fit")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--seeds", type=int, default=4, help="k-means seeds")
    p.add_argument("--min-rows", type=int, default=300, help="rows a language needs")
    p.add_argument("--scored", type=int, default=262_144,
                   help="the scored tail, the harvest's last rows (phase2.sh's --label-rows)")
    p.add_argument("--seed", type=int, default=0, help="the rows' draw")
    p.add_argument("--out", default="data/out/lcm-organism/meanplace/meanplace.json")
    cfg = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s  # noqa: E731
    src, dst = resolve(cfg.dir), resolve(cfg.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(dst.read_text()) if dst.exists() else {}
    rows = None
    for arm in cfg.arms:
        if arm in results:
            continue
        t0 = time.time()
        path = src / f"{arm}_b4.npy"
        mapping = row_map(path)
        if mapping is None:
            raise SystemExit(f"{path} has no harvest meta.json with row ranges")
        x = np.load(path, mmap_mode="r")
        drawn = sample_rows(mapping[0], len(x), cfg.fit, cfg.tail, cfg.scored, cfg.seed)
        if rows is not None and not all(np.array_equal(a, b) for a, b in zip(rows, drawn)):
            raise SystemExit(f"{arm} is not laid out as {cfg.arms[0]}, so its rows differ")
        rows = drawn
        langs = np.load(src / f"{arm}_b4.langs.npy", mmap_mode="r")
        Y = np.asarray(x[rows[0]], np.float64)
        T = np.asarray(x[rows[1]], np.float64)
        g, m = geometry(Y, np.asarray(langs[rows[0]]), cfg.min_rows)
        lt = np.asarray(langs[rows[1]])
        g["kmeans"] = {
            "k": cfg.k,
            "uncentered": kmeans_agreement(unit(Y[:cfg.kfit]), unit(T), lt, cfg.k, cfg.seeds,
                                           cfg.min_rows),
            "centered": kmeans_agreement(unit(Y[:cfg.kfit] - m), unit(T - m), lt, cfg.k,
                                         cfg.seeds, cfg.min_rows)}
        g["rows"] = {"fit": cfg.fit, "tail": cfg.tail, "kfit": cfg.kfit, "scored": cfg.scored,
                     "seed": cfg.seed}
        results[arm] = g
        dst.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{arm}: share {g['share_mean']:.3f}, axes {g['mean_axis_pr']:.1f} "
              f"(top {g['mean_axis_top1']:.2f} on {g['mean_axis_top1_coord']}, "
              f"{g['coord_mean_over_sd_max']:.1f} SD), along the variance "
              f"{g['var_along_mean_over_avg']:.2f}, affine FVU "
              f"{g['affine_fvu']['centered_to_uncentered']:.4f}, k-means "
              f"{g['kmeans']['uncentered']['all']:.3f} / {g['kmeans']['centered']['all']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return dst


if __name__ == "__main__":
    main()
