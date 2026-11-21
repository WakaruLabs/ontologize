"""Plain quantizers at the stack's rates: what does the classifier cost
against a codec that searches for each row's code?

FAISS's quantizers in the loss's whitened frame (each row times the square
root of the MSE weights, so a weight of 0 drops a coordinate, as
`ratecurve.py` does), fit on rows drawn from the cache's training prefix
and scored on the same tail rows as `ratecurve.py`: FVU_w against the bits
each code spends, a fixed length per row.

- PQ: the 1,024 coordinates in `M` blocks, each quantized with 2**nbits
  centroids, bits M * nbits.
- OPQ: the same after a learned rotation (`--opq-iters` updates).
- RQ: residual quantization, `--rq-stages` codebooks of 2**nbits entries
  each fit by one k-means to what the earlier ones leave, encoded by beam
  search (`--beam`; 1 is greedy). Its first m stages are a code of m * nbits bits, so one fit
  gives the whole curve; with 380 stages of 32 it has the stack's shape
  laid end to end, one head per stage, and the stack's prefixes of 76
  heads are read at stages 76, 152, 228, 304 and 380.

FAISS is not a project dependency: `uv run --with faiss-cpu`.

    JAX_PLATFORMS=cpu uv run --with faiss-cpu python experiments/rate-penalty/plainquant.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from jaxtyping import Float

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "seed-stability"))


def fvu(Z: Float[np.ndarray, "n d"], Y: Float[np.ndarray, "n d"]) -> float:
    """Squared error over the rows' total variance, in the whitened frame."""
    Z64 = Z.astype(np.float64)
    return float(((Z64 - Y) ** 2).sum() / ((Z64 - Z64.mean(0)) ** 2).sum())


def rq_prefix(codes, books: Float[np.ndarray, "m k d"], nbits: int,
              stages: list) -> dict:
    """FVU numerator pieces for each prefix of an RQ code: the partial
    sums of its first m codebooks' entries."""
    from kmeans_structure import unpack
    lab = unpack(codes, len(books), nbits)                  # (m, n)
    Y = np.zeros((lab.shape[1], books.shape[2]), np.float64)
    out, m0 = {}, 0
    for m in sorted(stages):
        for s in range(m0, m):
            Y += books[s][lab[s]]
        out[m] = Y.copy()
        m0 = m
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(ROOT / "data/sonar_embeddings/mc4_sent16.npy"))
    p.add_argument("--mse-weights",
                   default=str(ROOT / "data/out/sonar/sentcache/mse_weights_no837.npy"))
    p.add_argument("--prefix", type=int, default=12_800_000)
    p.add_argument("--train-rows", type=int, default=65536)
    p.add_argument("--rows", type=int, default=32765, help="tail rows, as ratecurve.py")
    p.add_argument("--pq", nargs="*", default=["256x7", "512x4"],
                   help="PQ settings as MxNBITS")
    p.add_argument("--opq", nargs="*", default=[],
                   help="as --pq; off by default, since learning the rotation at "
                        "d = 1024 took 400 s on 8,192 rows")
    p.add_argument("--opq-iters", type=int, default=10,
                   help="rotation updates, and PQ iterations before the first")
    p.add_argument("--rq-stages", type=int, default=380)
    p.add_argument("--rq-nbits", type=int, default=5)
    p.add_argument("--beam", type=int, nargs="*", default=[1],
                   help="RQ encoding beam widths; 1 is greedy")
    p.add_argument("--out", default=str(ROOT / "data/out/sonar/rate_penalty/plainquant"))
    cfg = p.parse_args()
    import faiss
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    sw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    X = np.load(cfg.cache, mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(cfg.prefix, cfg.train_rows,
                                                  replace=False))
    Z = np.ascontiguousarray(np.asarray(X[idx], np.float32) * sw)
    T = np.ascontiguousarray(np.asarray(X[-cfg.rows:], np.float32) * sw)
    d = Z.shape[1]
    rows = []

    def record(kind, setting, bits, f, secs):
        rows.append({"code": kind, "setting": setting, "bits": bits, "fvu": f,
                     "seconds": secs})
        print(f"{kind:<4} {setting:<14} bits {bits:>5}  FVU {f:.4f}  ({secs:.0f}s)",
              flush=True)

    for s in cfg.pq:
        M, nb = (int(v) for v in s.split("x"))
        t0 = time.time()
        pq = faiss.ProductQuantizer(d, M, nb)
        pq.train(Z)
        record("pq", s, M * nb, fvu(T, pq.decode(pq.compute_codes(T))), time.time() - t0)
    for s in cfg.opq:
        M, nb = (int(v) for v in s.split("x"))
        t0 = time.time()
        # OPQMatrix otherwise learns its rotation with an internal 8-bit PQ
        # over 50 iterations, which at this width runs for hours
        opq = faiss.OPQMatrix(d, M)
        inner = faiss.ProductQuantizer(d, M, nb)
        opq.pq = inner
        opq.niter, opq.niter_pq_0 = cfg.opq_iters, cfg.opq_iters
        opq.train(Z)
        Zr, Tr = opq.apply(Z), opq.apply(T)
        pq = faiss.ProductQuantizer(d, M, nb)
        pq.train(Zr)
        Y = opq.reverse_transform(pq.decode(pq.compute_codes(Tr)))
        record("opq", s, M * nb, fvu(T, Y), time.time() - t0)
    stages = sorted({cfg.rq_stages * i // 5 for i in range(1, 6)})
    for b in cfg.beam:
        t0 = time.time()
        rq = faiss.ResidualQuantizer(d, cfg.rq_stages, cfg.rq_nbits)
        rq.max_beam_size = b
        # one k-means per stage; FAISS's default (progressive dimensions with
        # codebook refinement) took 45x as long at d = 1024
        rq.train_type = faiss.ResidualQuantizer.Train_default
        rq.train(Z)
        codes = rq.compute_codes(T)
        books = faiss.vector_to_array(rq.codebooks).reshape(
            cfg.rq_stages, 1 << cfg.rq_nbits, d)
        secs = time.time() - t0
        for m, Y in rq_prefix(codes, books, cfg.rq_nbits, stages).items():
            record("rq", f"beam{b}_{m}st", m * cfg.rq_nbits, fvu(T, Y), secs)
    (out / "summary.json").write_text(json.dumps(
        {"config": vars(cfg), "rows": rows}, indent=1))
    print(f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
