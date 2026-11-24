"""SONAR's rows recoloured to a harvest's spectrum: the control for effective dimension.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/recolour.py

A dictionary's seeds may agree more on a source of low effective dimension
whatever the source's content. This writes the SONAR cache's rows, on the
target harvest's rows, through one linear map that gives them the target's
covariance spectrum in the frame each one's loss weighs (spectrum.py's frame):

    y = c R diag(s) U^T (sqrt(w_s) * x)

- U, lam_s: the eigenvectors and eigenvalues of sqrt(w_s) * x's covariance,
  fitted on `--fit-rows` prefix rows, with w_s the source's MSE weights;
- s = sqrt(lam_t / lam_s), rank for rank, lam_t the target harvest's
  spectrum under its own weights on the same rows; a direction the source's
  weights zero (lam_s = 0) stays zero, so the target's smallest eigenvalue
  goes unmatched;
- R: a random rotation, so that every coordinate has about the same
  variance and the inverse-variance weights the trainer applies stay near
  uniform, which keeps the spectrum the map gives;
- c: a constant that brings each coordinate's variance to about 1, inside
  float16's precision.

The map is linear and has no offset, so the rows keep everything SONAR's
embedding carries (up to the zeroed direction) and no dynamics. The output is
laid out as the target harvest (harvest_lcm.py): the same rows in the same
order, with the target's `.index`, `.kind`, `.langs` and `.docstart` sidecars
linked, its own `.mse_weights.npy` and `.mse_weights_matched.npy` computed as
harvest_lcm.py computes them, and a `.meta.json` with the target's row ranges.
Rows are written through `<out>.partial` and renamed when complete.

A source already laid out as the target (its meta.json's row ranges are the
target's, as ctxmean.py writes them) is read row for row: the recoloured
fixed mix,

    recolour.py --source <ctxmean>.npy --source-weights <ctxmean>.mse_weights_matched.npy \\
        --out data/activations/lcm-organism/sonar_ctxmean_recol_b4.npy

`--about-mean` maps each row's variation about the source's mean and leaves the
mean where the source's weighted frame has it, scaled by c and unrotated:

    y = c m + c R diag(s) U^T (sqrt(w_s) * x - m),   m the mean of sqrt(w_s) * x

over the fit rows. The rows' covariance, and so their weights, are what they
are without it; only the mean's direction changes. A cache recoloured to its
own spectrum about its mean is weightframe.py's rows, c sqrt(w) * y, with
everything but the mean rotated.

`--center` maps the same variation and leaves the mean out:

    y = c R diag(s) U^T (sqrt(w_s) * x - m).

A cache recoloured to its own spectrum this way is weightframe.py's rows
written with its `--center`, rotated, up to where each fits the mean (the fit
rows here, the whole prefix there).
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from jaxtyping import Float, Int, Shaped

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from f16cache import link, to_f16  # noqa: E402
from harvest_lcm import (MATCH_ROWS, SLICED, harvest_rows, inverse_variance,  # noqa: E402
                         whitened_base)
from ontologize.data.loaders import doc_holdout  # noqa: E402
from spectrum import file_rows, moments, row_map, stem, stats  # noqa: E402

LINKED = (*SLICED, ".docstart.npy")


def recolour_map(C_s: Float[np.ndarray, "d d"], C_t: Float[np.ndarray, "d d"],
                 seed: int, eps: float = 1e-9
                 ) -> Tuple[Float[np.ndarray, "d d"], Dict[str, object]]:
    """The map A with A C_s A^T = c^2 R diag(lam_t) R^T, rank for rank, and
    each diagonal entry about 1 (see the module docstring)."""
    lam_s, U = np.linalg.eigh(C_s)
    lam_s, U = lam_s[::-1], U[:, ::-1]
    lam_t = np.clip(np.linalg.eigvalsh(C_t)[::-1], 0, None)
    live = lam_s > eps * lam_s[0]
    s = np.where(live, np.sqrt(lam_t / np.where(live, lam_s, 1.0)), 0.0)
    d = len(s)
    R = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))[0]
    target = (s ** 2 * lam_s)
    c = np.sqrt(d / target.sum())
    A = c * (R * s) @ U.T
    return A, {"scale": float(c), "rotation_seed": seed, "zeroed": int((~live).sum()),
               "unmatched_target_share": float(lam_t[~live].sum() / lam_t.sum())}


def weighted_mean(x: Shaped[np.ndarray, "n d"], w: Float[np.ndarray, "d"],
                  rows: Int[np.ndarray, "m"], chunk: int = 16384) -> Float[np.ndarray, "d"]:
    """The mean of sqrt(w) * x over `rows` (sorted), in float64."""
    sw = np.sqrt(np.asarray(w, np.float64))
    total = np.zeros(x.shape[1])
    for lo in range(0, len(rows), chunk):
        total += (np.asarray(x[rows[lo:lo + chunk]], np.float64) * sw).sum(0)
    return total / len(rows)


def mean_share(total: Float[np.ndarray, "d d"], mu: Float[np.ndarray, "d"]) -> float:
    """|mean|^2 over the total variance: how much of a row's norm is shared."""
    return float(mu @ mu / np.trace(total))


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/sonar_embeddings/mc4_sent16_f16.npy")
    p.add_argument("--source-weights", default="data/out/sonar/sentcache/mse_weights_no837.npy")
    p.add_argument("--target", default="data/activations/lcm-organism/lcm_b4.npy",
                   help="a harvest: its rows and its spectrum")
    p.add_argument("--target-weights", default=None,
                   help="default: the target's .mse_weights_matched.npy")
    p.add_argument("--out", default="data/activations/lcm-organism/sonar_recol_b4.npy")
    p.add_argument("--fit-rows", type=int, default=262144, help="prefix rows drawn for the fit")
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="the fit rows' and the rotation's")
    p.add_argument("--chunk", type=int, default=65536)
    p.add_argument("--about-mean", action="store_true",
                   help="map the rows about the source's mean and keep the mean on its own axes")
    p.add_argument("--center", action="store_true",
                   help="map the rows about the source's mean and leave the mean out")
    cfg = p.parse_args(argv)
    if cfg.about_mean and cfg.center:
        raise SystemExit("--about-mean keeps the mean and --center drops it; give one")
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s  # noqa: E731
    src, src_w = resolve(cfg.source), resolve(cfg.source_weights)
    tgt = resolve(cfg.target)
    tgt_w = resolve(cfg.target_weights) if cfg.target_weights else \
        Path(f"{stem(tgt)}.mse_weights_matched.npy")
    dst = resolve(cfg.out)
    if dst.exists():
        raise SystemExit(f"{dst} exists")

    mapping = row_map(tgt)
    if mapping is None:
        raise SystemExit(f"{tgt} has no harvest meta.json with row ranges")
    P, T = mapping
    x_s = np.load(src, mmap_mode="r")
    x_t = np.load(tgt, mmap_mode="r")
    n = T + x_t.shape[0] - P
    idx_s = np.load(f"{stem(src)}.index.npy", mmap_mode="r")
    idx_t = np.load(f"{stem(tgt)}.index.npy", mmap_mode="r")
    laid_out = row_map(src) == mapping
    if laid_out:
        # the source is already laid out as the target: row i is row i
        if x_s.shape != x_t.shape:
            raise SystemExit(f"{src} is {x_s.shape}, laid out as {tgt}, which is {x_t.shape}")
        rows = np.arange(x_s.shape[0])
        if not np.array_equal(idx_t, idx_s):
            raise SystemExit(f"{tgt}'s index is not {src}'s")
    else:
        if x_s.shape[0] != n or x_s.shape[1] != x_t.shape[1]:
            raise SystemExit(f"{src} is {x_s.shape}; {tgt}'s rows end at cache row {n}")
        rows = harvest_rows(P, T, n)
        if not (np.array_equal(idx_t[:P], idx_s[:P]) and np.array_equal(idx_t[P:], idx_s[T:n])):
            raise SystemExit(f"{tgt}'s index is not {src}'s on the harvest's rows")

    # the fit, on the same prefix rows of both
    fit = np.sort(np.random.default_rng(cfg.seed).choice(P, size=min(cfg.fit_rows, P),
                                                         replace=False))
    w_s, w_t = np.load(src_w), np.load(tgt_w)
    lab_s = np.load(f"{stem(src)}.langs.npy", mmap_mode="r")
    lab_t = np.load(f"{stem(tgt)}.langs.npy", mmap_mode="r")
    C_s, _, _ = moments(x_s, w_s, lab_s, fit, cfg.min_rows)
    C_t, _, _ = moments(x_t, w_t, lab_t, file_rows(fit, mapping), cfg.min_rows)
    A, info = recolour_map(C_s, C_t, cfg.seed)
    # the offset each mapped row takes: c m - A m about the mean, -A m centered, else none
    shift = np.zeros(A.shape[0])
    if cfg.about_mean or cfg.center:
        m = weighted_mean(x_s, w_s, fit)
        shift = (info["scale"] * m if cfg.about_mean else 0.0) - A @ m
    info["about_mean"] = bool(cfg.about_mean)
    info["center"] = bool(cfg.center)
    print(f"map: {json.dumps(info)}", flush=True)

    A32 = A.T.astype(np.float32)
    shift32 = shift.astype(np.float32)
    sw = np.sqrt(w_s).astype(np.float32)
    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(len(rows), A.shape[0]))
    conv = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    for lo in range(0, len(rows), cfg.chunk):
        r = rows[lo:lo + cfg.chunk]
        a = (np.asarray(x_s[r[0]:r[-1] + 1], np.float32) if r[-1] - r[0] + 1 == len(r)
             else np.asarray(x_s[r], np.float32))
        h, st = to_f16((a * sw) @ A32 + shift32)
        conv["max_abs"] = max(conv["max_abs"], st["max_abs"])
        for k in ("subnormal", "to_zero", "nonfinite"):
            conv[k] += st[k]
        out[lo:lo + len(h)] = h
    out.flush()
    del out
    if conv["nonfinite"]:
        raise SystemExit(f"{conv['nonfinite']} values are not finite")

    h = np.load(part, mmap_mode="r")
    w = inverse_variance(h, P)
    matched = (whitened_base(x_s, w_s) / whitened_base(h, w))
    side = lambda suf: dst.with_name(dst.stem + suf)  # noqa: E731
    np.save(side(".mse_weights.npy"), w.astype(np.float32))
    np.save(side(".mse_weights_matched.npy"), (w * matched).astype(np.float32))
    for suf in LINKED:
        link(Path(f"{stem(tgt)}{suf}"), side(suf))
    # what the trainer will see: the output under its own matched weights
    C_o, _, _ = moments(h, w * matched, lab_t, fit, cfg.min_rows)
    del h
    os.replace(part, dst)
    held = {"harvest": doc_holdout(dst, 32768), "target": doc_holdout(tgt, 32768)}
    if held["harvest"] != held["target"]:
        raise SystemExit(f"the scored tail is {held['harvest']} rows here, {held['target']} "
                         f"in {tgt}")

    mu = lambda x, wt, rr: (np.asarray(x[rr], np.float64) * np.sqrt(wt)).mean(0)  # noqa: E731
    some = fit[:MATCH_ROWS]
    shape = lambda C: {k: v for k, v in stats(np.linalg.eigvalsh(C), []).items()  # noqa: E731
                       if k in ("participation", "erank")}
    spectra = {k: {**shape(C), "mean_share": m}
               for k, C, m in (
                   ("source", C_s, mean_share(C_s, mu(x_s, w_s, some))),
                   ("target", C_t, mean_share(C_t, mu(x_t, w_t, file_rows(some, mapping)))),
                   ("out", C_o, mean_share(C_o, mu(np.load(dst, mmap_mode="r"), w * matched,
                                                   some))))}
    meta = {"source": src.name, "source_laid_out": bool(laid_out), "target": tgt.name,
            "n": int(len(rows)), "d": int(A.shape[0]), "dtype": "float16",
            "rows": {"prefix": [0, P], "tail": [T, n],
                     "note": "harvest row i is cache row i below P, cache row T + i - P after"},
            "holdout": held,
            "fit": {"rows": int(len(fit)), "seed": cfg.seed, "min_rows": cfg.min_rows},
            "map": info, "spectra": spectra, "conversion": conv,
            "mse_weights": {"rows": P, "max": float(w.max()), "median": float(np.median(w))},
            "mse_weights_matched_scale": matched}
    side(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), "map": info, "spectra": spectra, "conversion": conv},
                     indent=2))
    return dst


if __name__ == "__main__":
    main()
