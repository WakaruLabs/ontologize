"""A float16 copy of a float32 embedding cache, with its sidecars, for upload.

    uv run python experiments/lcm-organism/f16cache.py \\
        --cache data/sonar_embeddings/mc4_sent16.npy [--out <dir>/mc4_sent16_f16.npy]

The rows are converted in chunks read with pread and written with pwrite,
into `<out>.partial`, which is renamed only once every row is written and
checked. The page cache the copy fills is released as it goes, so it does
not push other users' files out of memory.

Readers find a cache's sidecars by its stem (`<stem>.index.npy`,
`<stem>.kind.npy`, `<stem>.mse_weights.npy`, ...), so each `<stem>.*` file
beside the cache is linked under the copy's stem: a hard link on the same
filesystem, a symlink otherwise. `<out stem>.meta.json` is the source's
meta.json with `dtype` set to float16 and a `float16` record of the
conversion: what rounding cost on the held-out tail and on rows spread
across the cache (each row's relative error, and the whitened FVU of the
rounding under each MSE weighting given).

With `--bulk DIR` the array itself is written in DIR, on another disk, and
`--out` is a symlink to it, while the sidecars and meta.json stay beside
`--out` (meta.json's `float16.array` says where the array is).

It refuses a value float16 cannot hold, and a copy that would leave less
than `--min-free-gb` free on the array's filesystem.
"""
import argparse
import datetime
import json
import os
import shutil
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int

F16_MAX = float(np.finfo(np.float16).max)
F16_TINY = float(np.finfo(np.float16).smallest_normal)


def to_f16(x: Float[np.ndarray, "n d"]) -> tuple[Float[np.ndarray, "n d"], dict]:
    """`x` rounded to float16, with the chunk's largest magnitude, how many
    nonzero values land in float16's subnormal range, how many round to
    zero, and how many are not finite. Raises on a finite value beyond
    float16's range, which would otherwise become inf."""
    a = np.abs(x)
    finite: Bool[np.ndarray, "n d"] = np.isfinite(x)
    if (a[finite] > F16_MAX).any():
        raise ValueError(f"a value of magnitude {a[finite].max():.6g} is beyond "
                         f"float16's range ({F16_MAX})")
    y = x.astype(np.float16)
    return y, {"max_abs": float(a[finite].max(initial=0.0)),
               "subnormal": int(((a > 0) & (a < F16_TINY)).sum()),
               "to_zero": int(((x != 0) & (y == 0)).sum()),
               "nonfinite": int((~finite).sum())}


def rounding_cost(x: Float[np.ndarray, "n d"],
                  weights: dict[str, Float[np.ndarray, "d"]]) -> dict:
    """What rounding `x` to float16 costs: the median and largest relative
    row error |x - q| / |x|, and per weighting w the whitened FVU of the
    rounding, mean(w e^2) / mean(w var(x)), with var over these rows."""
    x = np.asarray(x, np.float32)
    e = x - x.astype(np.float16).astype(np.float32)
    rel = np.linalg.norm(e, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-30)
    var = x.var(0)
    return {"rows": int(len(x)),
            "rel_err_median": float(np.median(rel)),
            "rel_err_max": float(rel.max()),
            "fvu_w": {k: float((w * e ** 2).mean() / (w * var).mean())
                      for k, w in weights.items()}}


def sidecars(cache: Path) -> list[Path]:
    """The files beside `cache` that readers find by its stem, other than
    the cache itself and its meta.json."""
    meta = cache.with_name(cache.stem + ".meta.json")
    return sorted(p for p in cache.parent.glob(cache.stem + ".*")
                  if p != cache and p != meta and p.is_file())


def link(src: Path, dst: Path) -> str:
    """`dst` made the same file as `src`: a hard link where both are on one
    filesystem, else a symlink. A `dst` that already is `src` is left."""
    if dst.exists() or dst.is_symlink():
        if dst.exists() and os.path.samefile(src, dst):
            return "kept"
        raise FileExistsError(f"{dst} exists and is not {src}")
    try:
        os.link(src, dst)
        return "hard"
    except OSError:
        dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
        return "symlink"


def _pread(fd: int, buf: np.ndarray, pos: int) -> None:
    mv, got = memoryview(buf).cast("B"), 0
    while got < len(mv):
        k = os.preadv(fd, [mv[got:]], pos + got)
        if k == 0:
            raise EOFError(f"the cache ends {len(mv) - got} bytes early")
        got += k


def _pwrite(fd: int, buf: np.ndarray, pos: int) -> None:
    mv, put = memoryview(buf).cast("B"), 0
    while put < len(mv):
        put += os.pwrite(fd, mv[put:], pos + put)


def _drop(fd: int, pos: int, size: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, pos, size, os.POSIX_FADV_DONTNEED)


def copy_f16(src: Path, dst: Path, chunk: int = 262144,
             sync_every: int = 16, log=print) -> dict:
    """Write `src` (a 2-D float32 .npy) to `dst` as float16, through
    `dst`.partial, and return the conversion's statistics."""
    a = np.load(src, mmap_mode="r")
    if a.dtype != np.float32 or a.ndim != 2 or not a.flags.c_contiguous:
        raise ValueError(f"{src}: expected a C-ordered 2-D float32 array, "
                         f"got {a.dtype} {a.shape}")
    n, d = a.shape
    off_in = a.offset
    del a
    part = dst.with_name(dst.name + ".partial")
    out = np.lib.format.open_memmap(part, mode="w+", dtype=np.float16, shape=(n, d))
    off_out = out.offset
    del out
    stats = {"max_abs": 0.0, "subnormal": 0, "to_zero": 0, "nonfinite": 0}
    t0 = datetime.datetime.now()
    fi = os.open(src, os.O_RDONLY)
    fo = os.open(part, os.O_WRONLY)
    try:
        synced = 0
        for i, lo in enumerate(range(0, n, chunk)):
            hi = min(lo + chunk, n)
            x = np.empty((hi - lo, d), np.float32)
            _pread(fi, x, off_in + lo * d * 4)
            _drop(fi, off_in + lo * d * 4, x.nbytes)
            y, s = to_f16(x)
            _pwrite(fo, y, off_out + lo * d * 2)
            stats["max_abs"] = max(stats["max_abs"], s["max_abs"])
            for k in ("subnormal", "to_zero", "nonfinite"):
                stats[k] += s[k]
            if (i + 1) % sync_every == 0 or hi == n:
                os.fdatasync(fo)
                _drop(fo, off_out + synced * d * 2, (hi - synced) * d * 2)
                synced = hi
                dt = (datetime.datetime.now() - t0).total_seconds()
                log(f"{hi:,} / {n:,} rows, {hi * d * 4 / dt / 2 ** 30:.2f} GiB/s read")
    finally:
        os.close(fi)
        os.close(fo)
    return {"n": n, "d": d, "values": n * d, **stats}


def check_rows(src: Path, part: Path,
               rows: Int[np.ndarray, "m"]) -> None:
    """Raise unless the copy's `rows` are the source's rounded to float16."""
    a = np.load(src, mmap_mode="r")
    b = np.load(part, mmap_mode="r")
    if b.dtype != np.float16 or b.shape != a.shape:
        raise ValueError(f"the copy is {b.dtype} {b.shape}, the source {a.shape}")
    if not np.array_equal(b[rows], a[rows].astype(np.float16), equal_nan=True):
        raise ValueError("the copy's rows differ from the source's rounded to float16")


def spread(n: int, holdout: int, m: int) -> Int[np.ndarray, "m"]:
    """`m` rows evenly spaced over the training prefix (rows before the
    held-out tail)."""
    return np.unique(np.linspace(0, n - holdout - 1, m).astype(np.int64))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("data/sonar_embeddings/mc4_sent16.npy"))
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <cache stem>_f16.npy beside the cache")
    ap.add_argument("--bulk", type=Path, default=None,
                    help="write the array in this directory instead, with --out a "
                         "symlink to it; the sidecars and meta.json stay beside --out")
    ap.add_argument("--weights", type=Path, action="append", default=None,
                    help="MSE weights for the rounding's FVU (repeatable); "
                         "default: the cache's <stem>.mse_weights.npy")
    ap.add_argument("--holdout", type=int, default=None,
                    help="the held-out tail's rows; default: meta.json's holdout")
    ap.add_argument("--sample", type=int, default=32768,
                    help="rows checked across the training prefix")
    ap.add_argument("--chunk", type=int, default=262144, help="rows per read")
    ap.add_argument("--min-free-gb", type=float, default=50.0,
                    help="space that must stay free on the destination's filesystem")
    args = ap.parse_args(argv)

    src = args.cache
    dst = args.out or src.with_name(src.stem + "_f16.npy")
    arr = args.bulk / dst.name if args.bulk else dst
    if src.resolve() in (dst.resolve(), arr.resolve()):
        raise SystemExit("--out is the cache itself")
    for p in {dst, arr}:
        if p.exists() or p.is_symlink():
            raise SystemExit(f"{p} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr.parent.mkdir(parents=True, exist_ok=True)
    meta_src = src.with_name(src.stem + ".meta.json")
    meta = json.loads(meta_src.read_text()) if meta_src.exists() else {}
    a = np.load(src, mmap_mode="r")
    n, d = a.shape
    del a
    holdout = args.holdout if args.holdout is not None else int(meta.get("holdout", 0))
    need = n * d * 2
    free = shutil.disk_usage(arr.parent).free
    if free - need < args.min_free_gb * 2 ** 30:
        raise SystemExit(f"the copy needs {need / 2 ** 30:.1f} GiB and {arr.parent} has "
                         f"{free / 2 ** 30:.1f} GiB free; that would leave less than "
                         f"--min-free-gb {args.min_free_gb:g}")
    weights = {p.stem: np.load(p) for p in
               (args.weights or [src.with_name(src.stem + ".mse_weights.npy")])}
    for k, w in weights.items():
        if w.shape != (d,):
            raise SystemExit(f"weights {k} have shape {w.shape}, not ({d},)")

    stats = copy_f16(src, arr, args.chunk)
    part = arr.with_name(arr.name + ".partial")
    tail = np.arange(n - holdout, n) if holdout else np.zeros(0, np.int64)
    body = spread(n, holdout, args.sample)
    check_rows(src, part, np.concatenate([body, tail]))
    a = np.load(src, mmap_mode="r")
    cost = {"spread": rounding_cost(a[body], weights)}
    if holdout:
        cost["tail"] = rounding_cost(a[-holdout:], weights)
    del a
    os.replace(part, arr)
    if arr != dst:
        dst.symlink_to(arr.resolve())

    links = {}
    for p in sidecars(src):
        q = dst.with_name(dst.stem + p.name[len(src.stem):])
        links[q.name] = link(p, q)
    meta.update(dtype="float16", float16={
        "source": src.name, "source_dtype": "float32",
        "written": datetime.date.today().isoformat(),
        "array": str(arr.resolve()) if arr != dst else dst.name,
        "conversion": stats, "rounding": cost, "sidecars": links})
    dst.with_name(dst.stem + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"out": str(dst), "array": str(arr), **stats, "rounding": cost}, indent=2))


if __name__ == "__main__":
    main()
