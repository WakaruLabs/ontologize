"""Which held-out rows of mc4_sent16 are templates: repeated text, a repeated skeleton, or a
near-twin in another document.

    uv run python experiments/lcm-organism/templates.py

Every arm scores seed agreement on the cache's last `--tail` rows, which are the same sentences
in mc4_sent16 and in every harvest (a harvest's tail follows its prefix). Each of those rows gets
four flags:
- **text**: its sentence, casefolded with every digit replaced by 0 and whitespace collapsed, is
  also a sentence of another document anywhere in the cache. `<cache>.sentences.jsonl` holds each
  document's usable sentences, and `<cache>.index.npy` maps a row to its document and segment;
- **twin**: its SONAR embedding, centered on the tail's mean, has cosine at least `--cos` with a
  row of another document in the tail;
- **skeleton**: its words with their slots masked (`masked_tokens`: URLs, words with a digit, and
  names, a word after the first that starts with a capital or any word with a capital after its
  first letter, a run of one mask counting as one) are also another document's, anywhere in the
  cache. Only a skeleton with at least two unmasked words, at least half its tokens, counts, so two
  sentences don't match on their masks alone;
- **shingle**: the masked words' 3-grams (for a sentence of at least four tokens whose skeleton
  counts) share a Jaccard index of at least `--jaccard` with a row of another document in the tail.
  MinHash signatures (`--hashes` of them, in `--bands` bands) propose the pairs, and each pair is
  checked exactly.
The skeleton and the shingles catch templates whose slots vary in words, which the text misses.

Writes, as masks over the tail for seedmatch.py's `--row-mask` (True keeps a row): keep_<flag>.npy
and only_<flag>.npy for each flag; keep_neither.npy and only_either.npy (text or twin);
keep_none.npy and only_any.npy (any of the four); and keep_random.npy, which drops as many rows of
each language as keep_none does, at random (`--seed`), the control for which rows were dropped.
Also twin_cos.npy, each row's largest cosine with another document's row, and flags.json, the
flagged shares overall, by language (all rows, and the languages with at least `--min-rows` rows,
as seedmatch.py's within-language score takes them) and by the cache's kind, with the twin
cosine's quantiles.
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
WORD = re.compile(r"(?:https?://|www\.)\S+|\w+")
NAME, NUMBER, LINK = "<name>", "<number>", "<url>"
MASKS = (NAME, NUMBER, LINK)
SEGMENTS = 1 << 16  # a key is doc * SEGMENTS + segment


def normalize(text: str) -> str:
    return SPACE.sub(" ", DIGIT.sub("0", text.casefold())).strip()


def hash_of(text: str, size: int = 8) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=size).digest(), "little")


def digest(text: str) -> int:
    return hash_of(normalize(text))


def masked_tokens(text: str) -> List[str]:
    """The sentence's words, casefolded, with its slots masked: a URL, a word with a digit, and a
    name (a word after the first that starts with a capital, or a word with a capital after its
    first letter). A run of one mask is one mask."""
    out: List[str] = []
    for i, m in enumerate(WORD.finditer(text)):
        w = m.group()
        if w.startswith(("http://", "https://", "www.")):
            t = LINK
        elif any(c.isdigit() for c in w):
            t = NUMBER
        elif (i > 0 and w[0].isupper()) or any(c.isupper() for c in w[1:]):
            t = NAME
        else:
            t = w.casefold()
        if not (out and t in MASKS and out[-1] == t):
            out.append(t)
    return out


def skeleton_counts(tokens: List[str]) -> bool:
    """Whether a skeleton says enough to match on: at least two unmasked words, at least half its
    tokens."""
    words = sum(t not in MASKS for t in tokens)
    return words >= 2 and 2 * words >= len(tokens)


def shingles(tokens: List[str]) -> Int[np.ndarray, "s"]:
    """The distinct hashes (32 bits) of a token list's 3-grams."""
    return np.unique(np.array([hash_of("\x1f".join(tokens[i:i + 3]), 4)
                               for i in range(len(tokens) - 2)], np.uint64))


def read_span(args: Tuple[str, int, int, int]) -> Tuple[np.ndarray, ...]:
    """For the lines that start in [start, end): each sentence's text hash, skeleton hash, whether
    its skeleton counts, document and segment, and the (document, segment, text) of each sentence
    of a document numbered `tail_first` or later."""
    path, start, end, tail_first = args
    h: List[int] = []
    k: List[int] = []
    ok: List[bool] = []
    d: List[int] = []
    s: List[int] = []
    tail: List[Tuple[int, int, str]] = []
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
                tokens = masked_tokens(sent[3])
                h.append(digest(sent[3]))
                k.append(hash_of(" ".join(tokens)))
                ok.append(skeleton_counts(tokens))
                d.append(rec["doc"])
                s.append(sent[0])
                if rec["doc"] >= tail_first:
                    tail.append((rec["doc"], sent[0], sent[3]))
    return (np.array(h, np.uint64), np.array(k, np.uint64), np.array(ok, bool),
            np.array(d, np.int64), np.array(s, np.int64), tail)


def sentence_hashes(path: Path, workers: int, tail_first: int = 1 << 62
                    ) -> Tuple[np.ndarray, ...]:
    """`read_span`'s arrays over the whole file, and its tail sentences as a dict keyed by
    (document, segment)."""
    size = path.stat().st_size
    cuts = np.linspace(0, size, workers * 8 + 1).astype(np.int64)
    with get_context("spawn").Pool(workers) as pool:  # no fork under a threaded caller
        parts = pool.map(read_span, [(str(path), int(a), int(b), tail_first)
                                     for a, b in zip(cuts[:-1], cuts[1:])])
    arrays = tuple(np.concatenate([p[i] for p in parts]) for i in range(5))
    return arrays + ({(d, s): t for p in parts for d, s, t in p[5]},)


def documents_per_hash(h: Int[np.ndarray, "m"], d: Int[np.ndarray, "m"]
                       ) -> Tuple[Int[np.ndarray, "u"], Int[np.ndarray, "u"]]:
    """The distinct hashes, sorted, and how many distinct documents each occurs in."""
    order = np.lexsort((d, h))
    hs, ds = h[order], d[order]
    first = np.r_[True, (hs[1:] != hs[:-1]) | (ds[1:] != ds[:-1])]
    return np.unique(hs[first], return_counts=True)


def row_index(row_doc: Int[np.ndarray, "n"], row_seg: Int[np.ndarray, "n"],
              d: Int[np.ndarray, "m"], s: Int[np.ndarray, "m"]) -> Int[np.ndarray, "n"]:
    """Where each row's sentence sits in the per-sentence arrays."""
    keys = d * SEGMENTS + s
    order = np.argsort(keys)
    sorted_keys = keys[order]
    want = row_doc * SEGMENTS + row_seg
    at = np.searchsorted(sorted_keys, want)
    if (at >= len(keys)).any() or (sorted_keys[np.minimum(at, len(keys) - 1)] != want).any():
        raise SystemExit("some rows' sentences are not in the sentences file")
    return order[at]


def repeated(h: Int[np.ndarray, "m"], d: Int[np.ndarray, "m"], at: Int[np.ndarray, "n"]
             ) -> Bool[np.ndarray, "n"]:
    """Whether the hash of each sentence `at` points to is also another document's."""
    uh, docs = documents_per_hash(h, d)
    return docs[np.searchsorted(uh, h[at])] >= 2


def text_flags(row_doc: Int[np.ndarray, "n"], row_seg: Int[np.ndarray, "n"], h: Int[np.ndarray, "m"],
               d: Int[np.ndarray, "m"], s: Int[np.ndarray, "m"]) -> Bool[np.ndarray, "n"]:
    """Whether each row's sentence (by its hash `h`) also occurs in another document."""
    return repeated(h, d, row_index(row_doc, row_seg, d, s))


def mix64(z: Int[np.ndarray, "m"]) -> Int[np.ndarray, "m"]:
    """splitmix64's finalizer: a bijection of 64-bit words that scatters nearby inputs."""
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def minhash(sets: List[Int[np.ndarray, "s"]], n_hash: int, seed: int) -> Int[np.ndarray, "n h"]:
    """Each set's MinHash signature: for each of `n_hash` seeds, the least `mix64(x ^ seed)` over
    the set, so two sets agree on a hash about as often as their Jaccard index."""
    lens = np.array([len(x) for x in sets])
    if lens.min() == 0:
        raise ValueError("an empty set has no signature")
    X = np.concatenate(sets).astype(np.uint64)
    starts = np.r_[0, np.cumsum(lens)[:-1]]
    seeds = np.random.default_rng(seed).integers(0, 1 << 63, n_hash, dtype=np.uint64)
    sig = np.empty((len(sets), n_hash), np.uint64)
    for j in range(n_hash):
        sig[:, j] = np.minimum.reduceat(mix64(X ^ seeds[j]), starts)
    return sig


def jaccard(a: Int[np.ndarray, "p"], b: Int[np.ndarray, "q"]) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def near_twins(sets: List[Int[np.ndarray, "s"]], doc: Int[np.ndarray, "n"], sig: Int[np.ndarray, "n h"],
               bands: int, threshold: float, tries: int = 8) -> Bool[np.ndarray, "n"]:
    """Whether each set has a Jaccard index of at least `threshold` with a set of another
    document. Sets that agree on every hash of a band are candidates; each candidate is compared
    exactly with up to `tries` of its band-mates from other documents."""
    n, H = sig.shape
    r = H // bands
    mult = np.random.default_rng(0).integers(1, 1 << 62, r, dtype=np.uint64) | np.uint64(1)
    flag = np.zeros(n, bool)
    for band in range(bands):
        key = (sig[:, band * r:(band + 1) * r] * mult).sum(1)  # wraps mod 2^64, as a hash may
        _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
        order = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.cumsum(counts)]
        for g in np.flatnonzero(counts > 1):
            members = order[bounds[g]:bounds[g + 1]]
            if (doc[members] == doc[members[0]]).all():
                continue
            for i in members:
                if flag[i]:
                    continue
                for j in members[doc[members] != doc[i]][:tries]:
                    if jaccard(sets[i], sets[j]) >= threshold:
                        flag[i] = flag[j] = True
                        break
    return flag


def random_like(drop: Bool[np.ndarray, "n"], groups: np.ndarray, seed: int) -> Bool[np.ndarray, "n"]:
    """A mask that keeps all but as many rows of each group as `drop` marks, drawn at random."""
    keep = np.ones(len(drop), bool)
    rng = np.random.default_rng(seed)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        keep[rng.choice(idx, int(drop[idx].sum()), replace=False)] = False
    return keep


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
    p.add_argument("--jaccard", type=float, default=0.5)
    p.add_argument("--hashes", type=int, default=60)
    p.add_argument("--bands", type=int, default=20)
    p.add_argument("--seed", type=int, default=0, help="the MinHash family and the random control")
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

    h, k, ok, d, s, tail_text = sentence_hashes(Path(f"{stem}.sentences.jsonl"), cfg.workers,
                                                int(doc.min()))
    at = row_index(doc, seg, d, s)
    text = repeated(h, d, at)
    skeleton = repeated(k, d, at) & ok[at]
    del h, k, d, s
    cos = twin_cos(np.asarray(np.load(cache, mmap_mode="r")[-cfg.tail:]), doc)
    twin = cos >= cfg.cos
    tokens = [masked_tokens(tail_text[(int(a), int(b))]) for a, b in zip(doc, seg)]
    eligible = np.array([len(t) >= 4 and skeleton_counts(t) for t in tokens])
    rows = np.flatnonzero(eligible)
    sets = [shingles(tokens[i]) for i in rows]
    shingle = np.zeros(len(doc), bool)
    shingle[rows] = near_twins(sets, doc[rows], minhash(sets, cfg.hashes, cfg.seed), cfg.bands,
                               cfg.jaccard)

    flags = {"text": text, "twin": twin, "skeleton": skeleton, "shingle": shingle,
             "either": text | twin, "any": text | twin | skeleton | shingle}
    masks = {f"{w}_{name}": (~f if w == "keep" else f) for name, f in flags.items()
             for w in ("keep", "only") if name not in ("either", "any")}
    masks |= {"keep_neither": ~flags["either"], "only_either": flags["either"],
              "keep_none": ~flags["any"], "only_any": flags["any"],
              "keep_random": random_like(flags["any"], langs, cfg.seed)}
    for name, m in masks.items():
        np.save(out / f"{name}.npy", m)
    np.save(out / "twin_cos.npy", cos)
    scored = np.isin(langs, [g for g in np.unique(langs) if (langs == g).sum() >= cfg.min_rows])
    report = {
        "meta": dict(cache=str(cache), tail=cfg.tail, cos=cfg.cos, jaccard=cfg.jaccard,
                     hashes=cfg.hashes, bands=cfg.bands, seed=cfg.seed, min_rows=cfg.min_rows,
                     documents=int(len(np.unique(doc))), shingle_eligible=float(eligible.mean())),
        "flagged": {name: dict(all=float(f.mean()), scored_languages=float(f[scored].mean()))
                    for name, f in flags.items()},
        "added_by": {name: float((flags[name] & ~flags["either"]).mean())
                     for name in ("skeleton", "shingle")},
        "twin_cos_quantiles": {str(q): float(np.quantile(cos, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        "twin_share_at": {str(c): float((cos >= c).mean()) for c in (0.9, 0.95, 0.98, 0.99)},
        "by_kind": {name: shares(flags[name], kinds) for name in ("text", "twin", "skeleton",
                                                                  "shingle", "any")},
        "by_language": {name: shares(flags[name], langs, cfg.min_rows) for name in
                        ("text", "twin", "either", "skeleton", "shingle", "any")},
    }
    (out / "flags.json").write_text(json.dumps(report, indent=1) + "\n")
    f = report["flagged"]
    print(f"{cfg.tail} rows, {report['meta']['documents']} documents: text {f['text']['all']:.1%}, "
          f"twin (cos >= {cfg.cos}) {f['twin']['all']:.1%}, skeleton {f['skeleton']['all']:.1%}, "
          f"shingle (J >= {cfg.jaccard}) {f['shingle']['all']:.1%}; any {f['any']['all']:.1%}, "
          f"in the scored languages {f['any']['scored_languages']:.1%}")
    return report


if __name__ == "__main__":
    main()
