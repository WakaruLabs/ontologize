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

The output is laid out as the source, with its `.index`, `.kind`, `.langs`
and `.docstart` sidecars linked, and its `.meta.json` the source's with the
scale recorded. Its `.mse_weights.npy` is its own inverse variance over the
prefix rows, accumulated from the rows as written: 1 on every coordinate when
the source's weights are its inverse variances, and 0 wherever the source's
weights are 0. `.mse_weights_matched.npy` is that times one constant, so the
whitened error of predicting the mean equals the source's under its weights,
as harvest_lcm.py matches a harvest. Rows are written through
`<out>.partial` and renamed when complete.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from f16cache import link, to_f16  # noqa: E402
from harvest_lcm import whitened_base  # noqa: E402
from readout import LINKED  # noqa: E402
from spectrum import row_map, stem  # noqa: E402


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/activations/lcm-organism/lcm_readout_x_b4.npy")
    p.add_argument("--source-weights", default=None,
                   help="default: the source's .mse_weights_matched.npy")
    p.add_argument("--scale-like", default=None, metavar="NPY",
                   help="a recolour.py output whose map's scale c this uses")
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

    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(m, d))
    conv = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    s1, s2 = np.zeros(d), np.zeros(d)
    for lo in range(0, m, cfg.chunk):
        hi = min(lo + cfg.chunk, m)
        y16, st = to_f16(np.asarray(x[lo:hi], np.float32) * sw)
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
    np.save(side(".mse_weights.npy"), own.astype(np.float32))
    np.save(side(".mse_weights_matched.npy"), (own * matched).astype(np.float32))
    for suf in LINKED:
        link(Path(f"{stem(src)}{suf}"), side(suf))
    os.replace(part, dst)

    src_meta = json.loads(Path(f"{stem(src)}.meta.json").read_text())
    span = float(own[live].max() / own[live].min())
    meta = {**src_meta, "rows_from": src.name, "conversion": conv,
            "frame": {"weights": src_w.name, "scale": c,
                      "scale_like": Path(cfg.scale_like).name if cfg.scale_like else None},
            "mse_weights": {"rows": P, "live": int(live.sum()), "span": span},
            "mse_weights_matched_scale": matched}
    side(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), "scale": c, "weights_span": span,
                      "matched_scale": matched, "conversion": conv}, indent=2))
    return dst


if __name__ == "__main__":
    main()
