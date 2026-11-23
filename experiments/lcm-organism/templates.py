"""Which held-out rows of mc4_sent16 are templates: repeated text, or a near-twin in another document.

    uv run python experiments/lcm-organism/templates.py

Every arm scores seed agreement on the cache's last `--tail` rows, which are the same sentences
in mc4_sent16 and in every harvest (a harvest's tail follows its prefix). Each of those rows gets
two flags:
- **text**: its sentence, casefolded with every digit replaced by 0 and whitespace collapsed, is
  also a sentence of another document anywhere in the cache. `<cache>.sentences.jsonl` holds each
  document's usable sentences, and `<cache>.index.npy` maps a row to its document and segment;
- **twin**: its SONAR embedding, centered on the tail's mean, has cosine at least `--cos` with a
  row of another document in the tail. This catches templates whose slots vary in more than
  digits.

Writes, as masks over the tail for seedmatch.py's `--row-mask` (True keeps a row):
keep_text.npy, keep_twin.npy and keep_neither.npy, and only_text.npy, only_twin.npy and
only_either.npy, which keep the flagged rows instead; twin_cos.npy, each row's largest cosine
with another document's row; and flags.json, the flagged shares overall, by language (all rows,
and the languages with at least `--min-rows` rows, as seedmatch.py's within-language score takes
them) and by the cache's kind, with the twin cosine's quantiles.
"""
import os

# before jax is imported: on demand, so the card stays shared
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Bool, Float, Int  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPACE = re.compile(r"\s+")
DIGIT = re.compile(r"\d")
SEGMENTS = 1 << 16  # a key is doc * SEGMENTS + segment


def normalize(text: str) -> str:
    return SPACE.sub(" ", DIGIT.sub("0", text.casefold())).strip()


def digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(normalize(text).encode(), digest_size=8).digest(), "little")


def read_span(args: Tuple[str, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each sentence's (hash, document, segment) for the lines that start in [start, end)."""
    path, start, end = args
    h: List[int] = []
    d: List[int] = []
    s: List[int] = []
    with open(path, "rb") as f:
        f.seek(start)
        if start:
            f.readline()  # the line under way at `start` belongs to the span before
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            rec = json.loads(line)
            for sent in rec["sents"]:
                h.append(digest(sent[3]))
                d.append(rec["doc"])
                s.append(sent[0])
    return (np.array(h, np.uint64), np.array(d, np.int64), np.array(s, np.int64))


def sentence_hashes(path: Path, workers: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = path.stat().st_size
    cuts = np.linspace(0, size, workers * 8 + 1).astype(np.int64)
    with get_context("spawn").Pool(workers) as pool:  # no fork under a threaded caller
        parts = pool.map(read_span, [(str(path), int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])])
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def documents_per_hash(h: Int[np.ndarray, "m"], d: Int[np.ndarray, "m"]
                       ) -> Tuple[Int[np.ndarray, "u"], Int[np.ndarray, "u"]]:
    """The distinct hashes, sorted, and how many distinct documents each occurs in."""
    order = np.lexsort((d, h))
    hs, ds = h[order], d[order]
    first = np.r_[True, (hs[1:] != hs[:-1]) | (ds[1:] != ds[:-1])]
    return np.unique(hs[first], return_counts=True)


def text_flags(row_doc: Int[np.ndarray, "n"], row_seg: Int[np.ndarray, "n"], h: Int[np.ndarray, "m"],
               d: Int[np.ndarray, "m"], s: Int[np.ndarray, "m"]) -> Bool[np.ndarray, "n"]:
    """Whether each row's sentence also occurs in another document."""
    keys = d * SEGMENTS + s
    order = np.argsort(keys)
    want = row_doc * SEGMENTS + row_seg
    at = np.searchsorted(keys[order], want)
    if (at >= len(keys)).any() or (keys[order][np.minimum(at, len(keys) - 1)] != want).any():
        raise SystemExit("some rows' sentences are not in the sentences file")
    uh, docs = documents_per_hash(h, d)
    row_h = h[order][at]
    return docs[np.searchsorted(uh, row_h)] >= 2


@jax.jit
def chunk_twin(Xc: Float[jax.Array, "c e"], dc: Int[jax.Array, "c"], X: Float[jax.Array, "n e"],
               d: Int[jax.Array, "n"]) -> Float[jax.Array, "c"]:
    """Each of a chunk's rows' largest inner product with a row of another document."""
    S = jnp.matmul(Xc, X.T, precision=jax.lax.Precision.HIGHEST)
    return jnp.where(dc[:, None] == d[None, :], -2.0, S).max(1)


def twin_cos(E: Float[np.ndarray, "n e"], doc: Int[np.ndarray, "n"],
             chunk: int = 2048) -> Float[np.ndarray, "n"]:
    """Each row's largest cosine, after centering, with a row of another document. The last chunk
    is padded to the others' size (padding's document is -1), so every chunk has one shape."""
    X = E.astype(np.float32) - E.astype(np.float64).mean(0).astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    n, pad = len(X), -len(X) % chunk
    Xq = jnp.asarray(np.concatenate([X, np.zeros((pad, X.shape[1]), np.float32)]))
    dq = jnp.asarray(np.concatenate([doc, np.full(pad, -1)]).astype(np.int32))
    Xa, da = jnp.asarray(X), jnp.asarray(doc.astype(np.int32))
    out = [np.asarray(chunk_twin(Xq[a:a + chunk], dq[a:a + chunk], Xa, da))
           for a in range(0, n + pad, chunk)]
    return np.concatenate(out)[:n]


def shares(flag: Bool[np.ndarray, "n"], groups: np.ndarray, min_rows: int = 0) -> Dict[str, Dict]:
    out = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= min_rows:
            out[str(g)] = dict(rows=int(m.sum()), flagged=float(flag[m].mean()))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["flagged"]))


def main(argv: Optional[List[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_sent16_f16.npy")
    p.add_argument("--tail", type=int, default=262144)
    p.add_argument("--cos", type=float, default=0.95)
    p.add_argument("--min-rows", type=int, default=1000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--out", default="data/out/lcm-organism/templates")
    cfg = p.parse_args(argv)
    resolve = lambda s: Path(s) if Path(s).is_absolute() else ROOT / s
    cache = resolve(cfg.cache)
    stem = cache.with_suffix("")
    out = resolve(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(np.load(f"{stem}.index.npy", mmap_mode="r")[-cfg.tail:])
    langs = np.asarray(np.load(f"{stem}.langs.npy", mmap_mode="r")[-cfg.tail:])
    kinds = np.asarray(np.load(f"{stem}.kind.npy", mmap_mode="r")[-cfg.tail:])
    doc, seg = idx[:, 0].astype(np.int64), idx[:, 2].astype(np.int64)

    h, d, s = sentence_hashes(Path(f"{stem}.sentences.jsonl"), cfg.workers)
    text = text_flags(doc, seg, h, d, s)
    del h, d, s
    cos = twin_cos(np.asarray(np.load(cache, mmap_mode="r")[-cfg.tail:]), doc)
    twin = cos >= cfg.cos

    masks = {"keep_text": ~text, "keep_twin": ~twin, "keep_neither": ~(text | twin),
             "only_text": text, "only_twin": twin, "only_either": text | twin}
    for name, m in masks.items():
        np.save(out / f"{name}.npy", m)
    np.save(out / "twin_cos.npy", cos)
    scored = np.isin(langs, [g for g in np.unique(langs) if (langs == g).sum() >= cfg.min_rows])
    report = {
        "meta": dict(cache=str(cache), tail=cfg.tail, cos=cfg.cos, min_rows=cfg.min_rows,
                     documents=int(len(np.unique(doc)))),
        "flagged": {name: dict(all=float(f.mean()), scored_languages=float(f[scored].mean()))
                    for name, f in (("text", text), ("twin", twin), ("either", text | twin),
                                    ("both", text & twin))},
        "twin_cos_quantiles": {str(q): float(np.quantile(cos, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "twin_share_at": {str(c): float((cos >= c).mean()) for c in (0.9, 0.95, 0.98, 0.99)},
        "by_kind": {name: shares(f, kinds) for name, f in (("text", text), ("twin", twin))},
        "by_language": {name: shares(f, langs, cfg.min_rows) for name, f in
                        (("text", text), ("twin", twin), ("either", text | twin))},
    }
    (out / "flags.json").write_text(json.dumps(report, indent=1) + "\n")
    f = report["flagged"]
    print(f"{cfg.tail} rows, {report['meta']['documents']} documents: text {f['text']['all']:.1%}, "
          f"twin (cos >= {cfg.cos}) {f['twin']['all']:.1%}, either {f['either']['all']:.1%}; "
          f"in the scored languages {f['either']['scored_languages']:.1%}")
    return report


if __name__ == "__main__":
    main()
