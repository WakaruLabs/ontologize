"""SONAR's row plus a fixed share of its document's earlier mean: the fixed-mix control.

    JAX_PLATFORMS=cpu uv run python experiments/lcm-organism/ctxmean.py --weight 1

readout.py's arm reads LCM block 4's linear readout from x_n and the mean of
the document's earlier rows, with the readout's matrices fitted to the LCM's
states. This arm keeps the context and drops the LCM:

    y_n = x_n + weight * mean(x_0 .. x_{n-1}),   0 at n = 0,

in SONAR's own coordinates, on the target harvest's rows (readout.py's
`context_features`). If a dictionary's seeds agree on it as they do on the
readout, averaging in the context is enough; if they agree as on SONAR, the
LCM's own linear map is what helps.

Its loss weights are SONAR's (`--source-weights`, the SONAR arm's no837),
scaled so the whitened error of predicting the mean equals SONAR's under
them, as harvest_lcm.py matches a harvest's: the coordinates weigh as in the
SONAR arm, and the recipe's loss weights transfer. `.mse_weights.npy` holds
them unscaled, `.mse_weights_matched.npy` scaled. The output is laid out as
the target harvest, as readout.py lays out its rows, with the target's
sidecars linked and a `.meta.json` with its row ranges. Rows are written
through `<out>.partial` and renamed when complete.
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
from ontologize.data.loaders import doc_holdout  # noqa: E402
from readout import LINKED, context_features, doc_chunks, source_span  # noqa: E402
from spectrum import row_map, stem  # noqa: E402


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/sonar_embeddings/mc4_sent16_f16.npy")
    p.add_argument("--source-weights", default="data/out/sonar/sentcache/mse_weights_no837.npy")
    p.add_argument("--target", default="data/activations/lcm-organism/lcm_b4.npy",
                   help="a harvest: the rows to write, and its layout")
    p.add_argument("--weight", type=float, default=1.0, help="the earlier mean's weight")
    p.add_argument("--out", default="data/activations/lcm-organism/sonar_ctxmean_b4.npy")
    p.add_argument("--holdout", type=int, default=32768, help="the scored tail, snapped")
    p.add_argument("--chunk", type=int, default=65536)
    cfg = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s  # noqa: E731
    src, src_w, tgt, dst = (resolve(s) for s in (cfg.source, cfg.source_weights, cfg.target,
                                                 cfg.out))
    if dst.exists():
        raise SystemExit(f"{dst} exists")

    mapping = row_map(tgt)
    if mapping is None:
        raise SystemExit(f"{tgt} has no harvest meta.json with row ranges")
    P, T = mapping
    x_s = np.load(src, mmap_mode="r")
    m = np.load(tgt, mmap_mode="r").shape[0]
    n = T + m - P
    d = x_s.shape[1]
    if x_s.shape[0] != n:
        raise SystemExit(f"{src} has {x_s.shape[0]} rows; {tgt}'s rows end at cache row {n}")
    idx_s = np.load(f"{stem(src)}.index.npy", mmap_mode="r")
    idx_t = np.load(f"{stem(tgt)}.index.npy", mmap_mode="r")
    if not (np.array_equal(idx_t[:P], idx_s[:P]) and np.array_equal(idx_t[P:], idx_s[T:n])):
        raise SystemExit(f"{tgt}'s index is not {src}'s on the harvest's rows")
    ds = np.load(f"{stem(tgt)}.docstart.npy").astype(np.int64)
    if P not in ds:
        raise SystemExit(f"the prefix end {P} is not a document boundary of {tgt}")

    held = doc_holdout(tgt, cfg.holdout)
    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(m, d))
    conv = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    for lo, hi in doc_chunks(ds, [P, m - held], m, cfg.chunk):
        first = np.unique(ds[(ds >= lo) & (ds < hi)] - lo)
        F = context_features(np.asarray(x_s[source_span(lo, hi, P, T)], np.float32), first)
        y16, st = to_f16(F[:, :d] + np.float32(cfg.weight) * F[:, d:2 * d])
        conv["max_abs"] = max(conv["max_abs"], st["max_abs"])
        for k in ("subnormal", "to_zero", "nonfinite"):
            conv[k] += st[k]
        out[lo:hi] = y16
    out.flush()
    del out
    if conv["nonfinite"]:
        raise SystemExit(f"{conv['nonfinite']} values are not finite")

    w = np.load(src_w).astype(np.float64)
    y = np.load(part, mmap_mode="r")
    matched = whitened_base(x_s, w) / whitened_base(y, w)
    del y
    side = lambda suf: dst.with_name(dst.stem + suf)  # noqa: E731
    np.save(side(".mse_weights.npy"), w.astype(np.float32))
    np.save(side(".mse_weights_matched.npy"), (w * matched).astype(np.float32))
    for suf in LINKED:
        link(Path(f"{stem(tgt)}{suf}"), side(suf))
    os.replace(part, dst)
    if doc_holdout(dst, cfg.holdout) != held:
        raise SystemExit(f"the scored tail is {doc_holdout(dst, cfg.holdout)} rows here, "
                         f"{held} in {tgt}")

    meta = {"source": src.name, "target": tgt.name, "n": int(m), "d": int(d),
            "dtype": "float16", "weight": cfg.weight,
            "rows": {"prefix": [0, P], "tail": [T, n],
                     "note": "harvest row i is cache row i below P, cache row T + i - P after"},
            "holdout": {"rows": cfg.holdout, "harvest": held}, "conversion": conv,
            "mse_weights": {"from": src_w.name}, "mse_weights_matched_scale": matched}
    side(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), "weight": cfg.weight, "matched_scale": matched,
                      "conversion": conv}, indent=2))
    return dst


if __name__ == "__main__":
    main()
