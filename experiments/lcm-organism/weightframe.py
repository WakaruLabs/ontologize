"""A cache's rows in the frame its loss weighs, unrotated: the control that splits a rotation.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/weightframe.py \\
        --scale-like data/activations/lcm-organism/lcm_readout_x_rot_b4.npy

The trainer's loss weighs coordinate j of a cache by w_j, its inverse
variance (harvest_lcm.py's matched weights), so the loss is plain squared
error on sqrt(w) * y. recolour.py's map of a cache to its own spectrum is

    y_rot = c R U^T (sqrt(w) * y),

a rotation of those rows: the loss measures errors in both caches alike, and
the model's input changes in two steps, each coordinate scaled by sqrt(w_j)
and then the coordinates mixed. This writes the first step alone,

    y' = c (sqrt(w) * y),

with c taken from the recoloured cache's map (`--scale-like`), so that y_rot
is y' rotated, or else chosen to bring each coordinate's variance to about 1.
A dictionary on y' differs from one on y only in its input's scale per
coordinate, and from one on y_rot only in the rotation.

`--rotate-except J` rotates part of the frame: the live coordinates outside
the list J are mixed among themselves by a random rotation (drawn as
recolour.py draws R, from `--rotation-seed`), and those in J keep their axes.
That is still a rotation of sqrt(w) * y, but the output's weights are its own
inverse variances, and the mixed coordinates' variances are no longer even,
so its loss is only nearly the source's. A dictionary on it tells whether the
coordinates in J carry what the basis gives.

Three options change the coordinates J keeps, so that what about them matters
can be told apart; with `--rotate-except` each list must lie in J:
- `--rotate-block K` mixes the coordinates K among themselves by a rotation of
  their own (`--block-seed`). They stay a block apart from the rest, but no
  longer on their own axes. With `--block-basis principal` the rotation is
  the one onto the block's principal axes over the prefix rows, in the frame
  the loss weighs, largest first: the block's coordinates are then
  uncorrelated there, and their variances are its covariance's eigenvalues.
- `--whiten-block K` maps the block, centered on its prefix mean (which is
  added back), by the symmetric whitening of its prefix covariance (ZCA),
  scaled to keep the block's total variance. Its coordinates are then
  uncorrelated with equal variances, as near their own axes as a whitening
  leaves them, and the block's spectrum is flat. Within a block, uncorrelated
  coordinates can keep equal variances or the spectrum, not both, so this and
  the principal rotation change one each.
- `--gaussianize K` maps each coordinate in K, before the scaling, onto a
  Gaussian with its own mean and standard deviation, through its empirical
  distribution over the prefix rows: a value goes to the Gaussian quantile at
  the middle of its step in that distribution, so equal values stay equal,
  the order of the rows is kept, and values beyond the prefix's range take
  its end quantiles. Each row keeps what that coordinate says about it, and
  the coordinate loses its marginal's shape.
Each map is fitted on the prefix alone, so the held-out rows inform none of it.
`--gaussianize` applies before the block's map, whose fit sees the
Gaussianized values.

A rotated block's coordinates no longer have even variances, so the
output's own inverse-variance weights would change the loss along with the
basis. `--weights-like NPY` takes the weights of another cache laid out
alike instead (its `.mse_weights.npy` and `.mse_weights_matched.npy`). It is
refused unless they are equal across the block, where the loss is then that
cache's, turned within the block.

The output is laid out as the source, with its `.index`, `.kind`, `.langs`
and `.docstart` sidecars linked, and its `.meta.json` the source's with the
scale recorded. Its `.mse_weights.npy` is its own inverse variance over the
prefix rows, accumulated from the rows as written: 1 on every coordinate when
the source's weights are its inverse variances and nothing is rotated, and 0
wherever the source's weights are 0. `.mse_weights_matched.npy` is that times one constant, so the
whitened error of predicting the mean equals the source's under its weights,
as harvest_lcm.py matches a harvest. Rows are written through
`<out>.partial` and renamed when complete.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from typing import Dict, List, Optional, Tuple

import numpy as np
from jaxtyping import Float
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from f16cache import link, to_f16  # noqa: E402
from harvest_lcm import whitened_base  # noqa: E402
from readout import LINKED  # noqa: E402
from spectrum import row_map, stem  # noqa: E402


def coordinates(spec: Optional[str], d: int, flag: str) -> Optional[List[int]]:
    """A comma-separated list of coordinates, sorted, each checked to lie in 0..d-1."""
    if spec is None:
        return None
    js = sorted({int(j) for j in spec.split(",") if j.strip()})
    if any(j < 0 or j >= d for j in js):
        raise SystemExit(f"{flag} names a coordinate outside 0..{d - 1}")
    return js


def rotation(n: int, seed: int) -> Float[np.ndarray, "n n"]:
    """A random rotation, drawn as recolour.py draws R."""
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))[0].astype(np.float32)


def gaussian_fit(v: Float[np.ndarray, "p"]) -> Tuple[Float[np.ndarray, "p"], float, float]:
    """A coordinate's prefix values, sorted, with their mean and standard deviation."""
    v = np.asarray(v, np.float64)
    return np.sort(v), float(v.mean()), float(v.std())


def gaussianize(v: Float[np.ndarray, "n"], fit: Tuple[np.ndarray, float, float]
                ) -> Float[np.ndarray, "n"]:
    """Each value sent to the Gaussian quantile at the middle of its step in the fitted values'
    distribution, at the fitted mean and standard deviation."""
    s, mu, sd = fit
    n = len(s)
    q = (np.searchsorted(s, v, "left") + np.searchsorted(s, v, "right")) / (2 * n)
    return (mu + sd * ndtri(np.clip(q, 0.5 / n, 1 - 0.5 / n))).astype(np.float32)


def principal_axes(C: Float[np.ndarray, "k k"]) -> Float[np.ndarray, "k k"]:
    """A covariance's eigenvectors as columns, largest eigenvalue first, each signed so that its
    largest entry is positive: y @ V puts rows y on the principal axes."""
    V = np.linalg.eigh(C)[1][:, ::-1]
    return V * np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])


def zca(C: Float[np.ndarray, "k k"]) -> Float[np.ndarray, "k k"]:
    """The symmetric whitening C^(-1/2), times the root of C's mean eigenvalue, so that rows of
    covariance C mapped by it have covariance (tr C / k) I."""
    lam, V = np.linalg.eigh(C)
    if lam.min() <= 1e-12 * lam.max():
        raise SystemExit("the block's covariance on the prefix is singular")
    return (V * np.sqrt(lam.mean() / lam)) @ V.T


def block_moments(Y: Float[np.ndarray, "p k"]) -> Tuple[Float[np.ndarray, "k"], Float[np.ndarray, "k k"]]:
    """The mean and covariance of a block's prefix rows, in float64."""
    Y = np.asarray(Y, np.float64)
    mu = Y.mean(0)
    return mu, (Y - mu).T @ (Y - mu) / len(Y)


def prefix_columns(x: np.ndarray, cols: List[int], P: int, chunk: int
                   ) -> Dict[int, Float[np.ndarray, "p"]]:
    """The listed columns of the first P rows, read in one pass of row chunks."""
    parts = [np.asarray(x[lo:min(lo + chunk, P)][:, cols], np.float32) for lo in range(0, P, chunk)]
    V = np.concatenate(parts)
    return {j: V[:, i] for i, j in enumerate(cols)}


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/activations/lcm-organism/lcm_readout_x_b4.npy")
    p.add_argument("--source-weights", default=None,
                   help="default: the source's .mse_weights_matched.npy")
    p.add_argument("--scale-like", default=None, metavar="NPY",
                   help="a recolour.py output whose map's scale c this uses")
    p.add_argument("--rotate-except", default=None, metavar="J",
                   help="comma-separated coordinates to keep on their axes while "
                        "the other live coordinates are rotated among themselves")
    p.add_argument("--rotation-seed", type=int, default=0)
    p.add_argument("--rotate-block", default=None, metavar="K",
                   help="comma-separated coordinates to rotate among themselves, apart from "
                        "the rest")
    p.add_argument("--block-seed", type=int, default=1)
    p.add_argument("--block-basis", choices=("random", "principal"), default="random",
                   help="--rotate-block's rotation: random, from --block-seed, or onto the block's "
                        "principal axes over the prefix")
    p.add_argument("--whiten-block", default=None, metavar="K",
                   help="comma-separated coordinates to whiten among themselves (ZCA on the "
                        "prefix, the block's total variance kept), apart from the rest")
    p.add_argument("--gaussianize", default=None, metavar="K",
                   help="comma-separated coordinates to map monotonically onto a Gaussian of "
                        "their own mean and standard deviation, fitted on the prefix")
    p.add_argument("--weights-like", default=None, metavar="NPY",
                   help="a cache whose loss weights the output takes, in place of its own")
    p.add_argument("--out", default="data/activations/lcm-organism/lcm_readout_x_wf_b4.npy")
    p.add_argument("--chunk", type=int, default=65536)
    cfg = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s  # noqa: E731
    src, dst = resolve(cfg.source), resolve(cfg.out)
    src_w = (resolve(cfg.source_weights) if cfg.source_weights
             else Path(f"{stem(src)}.mse_weights_matched.npy"))
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    mapping = row_map(src)
    if mapping is None:
        raise SystemExit(f"{src} has no harvest meta.json with row ranges")
    P, _ = mapping

    x = np.load(src, mmap_mode="r")
    m, d = x.shape
    w = np.load(src_w).astype(np.float64)
    if w.shape != (d,):
        raise SystemExit(f"{src_w} has shape {w.shape}; {src} has {d} coordinates")
    live = w > 0
    if cfg.scale_like:
        like = Path(f"{stem(resolve(cfg.scale_like))}.meta.json")
        c = float(json.loads(like.read_text())["map"]["scale"])
    else:
        c = 1.0 / np.sqrt(whitened_base(x, w) * d / live.sum())
    sw = (c * np.sqrt(w)).astype(np.float32)
    keep = coordinates(cfg.rotate_except, d, "--rotate-except")
    block = coordinates(cfg.rotate_block, d, "--rotate-block")
    whiten = coordinates(cfg.whiten_block, d, "--whiten-block")
    gauss = coordinates(cfg.gaussianize, d, "--gaussianize")
    if block is not None and whiten is not None:
        raise SystemExit("--rotate-block and --whiten-block each map the block; give one")
    for flag, js in (("--rotate-block", block), ("--whiten-block", whiten),
                     ("--gaussianize", gauss)):
        if js is None:
            continue
        if keep is not None and not set(js) <= set(keep):
            raise SystemExit(f"{flag} must list only coordinates --rotate-except keeps")
        if not live[js].all():
            raise SystemExit(f"{flag} names a coordinate the weights zero")
    rot = R = B = W = mu = lam = None
    if keep is not None:
        rot = np.flatnonzero(live & ~np.isin(np.arange(d), keep))
        R = rotation(len(rot), cfg.rotation_seed)
    principal = block is not None and cfg.block_basis == "principal"
    fitted = block if principal else whiten
    cols = sorted(set(gauss or []) | set(fitted or []))
    V = prefix_columns(x, cols, P, cfg.chunk) if cols else {}
    fits = {j: gaussian_fit(V[j]) for j in gauss} if gauss is not None else {}
    if fitted is not None:
        Y = np.stack([gaussianize(V[j], fits[j]) if j in fits else V[j] for j in fitted], 1)
        mu, C = block_moments(Y * sw[fitted])
        lam = np.linalg.eigvalsh(C)[::-1]
    del V
    if block is not None:
        B = (principal_axes(C) if principal else rotation(len(block), cfg.block_seed)
             ).astype(np.float32)
    if whiten is not None:
        W, mu = zca(C).astype(np.float32), mu.astype(np.float32)
    mapped = block if block is not None else whiten
    like_w = None
    if cfg.weights_like:
        like = stem(resolve(cfg.weights_like))
        like_w = (np.load(f"{like}.mse_weights.npy"), np.load(f"{like}.mse_weights_matched.npy"))
        if like_w[0].shape != (d,):
            raise SystemExit(f"{cfg.weights_like}'s weights don't have {d} coordinates")
        if mapped is not None:
            wb = like_w[0][mapped].astype(np.float64)
            if wb.max() > wb.min() * (1 + 1e-3):
                raise SystemExit(f"{cfg.weights_like}'s weights differ across the block, so its "
                                 "loss would change with the block's map")

    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(m, d))
    conv = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    s1, s2 = np.zeros(d), np.zeros(d)
    for lo in range(0, m, cfg.chunk):
        hi = min(lo + cfg.chunk, m)
        xs = np.asarray(x[lo:hi], np.float32)
        for j, fit in fits.items():
            xs[:, j] = gaussianize(xs[:, j], fit)
        y32 = xs * sw
        if B is not None:
            y32[:, block] = y32[:, block] @ B
        if W is not None:
            y32[:, whiten] = (y32[:, whiten] - mu) @ W + mu
        if R is not None:
            y32[:, rot] = y32[:, rot] @ R
        y16, st = to_f16(y32)
        conv["max_abs"] = max(conv["max_abs"], st["max_abs"])
        for k in ("subnormal", "to_zero", "nonfinite"):
            conv[k] += st[k]
        out[lo:hi] = y16
        if lo < P:
            y = y16[:P - lo].astype(np.float64)
            s1 += y.sum(0)
            s2 += (y * y).sum(0)
    out.flush()
    del out
    if conv["nonfinite"]:
        raise SystemExit(f"{conv['nonfinite']} values are not finite")

    var = s2 / P - (s1 / P) ** 2
    own = np.zeros(d)
    own[live] = 1.0 / var[live]
    own[live] /= own[live].mean()
    y = np.load(part, mmap_mode="r")
    matched = whitened_base(x, w) / whitened_base(y, own)
    del y
    side = lambda suf: dst.with_name(dst.stem + suf)  # noqa: E731
    if like_w is None:
        np.save(side(".mse_weights.npy"), own.astype(np.float32))
        np.save(side(".mse_weights_matched.npy"), (own * matched).astype(np.float32))
    else:
        own = like_w[0].astype(np.float64)
        np.save(side(".mse_weights.npy"), like_w[0])
        np.save(side(".mse_weights_matched.npy"), like_w[1])
        matched = float(like_w[1][live].sum() / like_w[0][live].sum())
    for suf in LINKED:
        link(Path(f"{stem(src)}{suf}"), side(suf))
    os.replace(part, dst)

    src_meta = json.loads(Path(f"{stem(src)}.meta.json").read_text())
    span = float(own[live].max() / own[live].min())
    meta = {**src_meta, "rows_from": src.name, "conversion": conv,
            "frame": {"weights": src_w.name, "scale": c,
                      "scale_like": Path(cfg.scale_like).name if cfg.scale_like else None,
                      "rotate_except": keep,
                      "rotation_seed": cfg.rotation_seed if keep is not None else None,
                      "rotated": int(len(rot)) if rot is not None else 0,
                      "rotate_block": block,
                      "block_basis": cfg.block_basis if block is not None else None,
                      "block_seed": (cfg.block_seed if block is not None and not principal
                                     else None),
                      "whiten_block": whiten,
                      "block_spectrum": (None if lam is None
                                         else [float(v) for v in lam / lam.mean()]),
                      "gaussianize": gauss,
                      "weights_like": Path(cfg.weights_like).name if cfg.weights_like else None},
            "mse_weights": {"rows": P, "live": int(live.sum()), "span": span},
            "mse_weights_matched_scale": matched}
    side(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), "scale": c, "weights_span": span,
                      "matched_scale": matched, "conversion": conv}, indent=2))
    return dst


if __name__ == "__main__":
    main()
