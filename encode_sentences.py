"""Split the mC4 corpus into sentences and encode them with SONAR: the
sentence-level counterpart of `encode_corpus.py`'s document cache.

SONAR is a sentence encoder, and its auto-encoding degrades past about 250
characters (LCM team 2024), while `mc4_4M`'s rows are 512-token document
prefixes. Here each of `mc4_4M`'s documents contributes up to `--k` (16)
consecutive usable sentences, starting at a random usable one, so a
document's rows are a passage in reading order: rows for dictionary
learning, and sequences for next-sentence prediction.

Two stages, so the selection can be checked before the encoding and the
sentences re-encoded without replaying the stream:

  split   replay the stream in `mc4_4M`'s order (a reader process),
          checking each document's language against `mc4_4M.langs.npy`;
          segment each document's first --chars characters with SaT Capped
          (`sat-3l-sm` on the GPU, threshold 0.25, then every segment over
          200 characters broken at its most probable boundaries,
          `ontologize.data.segment`); keep segments of 20-200 characters
          whose letters are mostly in the config's script; choose each
          document's run (`segment.run`), dropping any segment of it GlotLID
          calls non-linguistic (`zxx_*`) and choosing again (worker
          processes); write <name>.sentences.jsonl, one line per document
  encode  encode every chosen sentence under its config's SONAR tag
          (`MC4_TO_SONAR`) and without the L2 step (`encode(normalize=
          False)`, SONAR's own embedding) into <name>.npy, float32, in
          batches mixed across tags (the tag is only the first token) and
          capped by padded tokens

The settings and their grounds are in `experiments/sentence-split/notes.md`.

Sidecars keyed by cache row, all written by `encode`:

  <name>.index.npy        (n, 3) int32 (doc, pos, segment): the document's
                          `mc4_4M` row, the sentence's place in the run, and
                          its ordinal among the document's segments
  <name>.docstart.npy     (n_docs,) int64 first row of each document; a
                          document with no usable sentence has no rows, and
                          its entry equals the next one's
  <name>.langs.npy        (n,) mC4 config
  <name>.lid.npy          (n,) GlotLID's top label, <name>.lid_p.npy its
                          probability
  <name>.mse_weights.npy  per-dimension inverse variance over the train rows
  <name>.meta.json

`split` writes <name>.sentences.jsonl: per document its config, segment
counts and chosen sentences as [segment, start, end, text, label, p], with
start and end character offsets in the NFC-normalized text. Its counts of
segments dropped as non-linguistic cover only the segments GlotLID saw,
those in or tried for a run.

Both stages are crash-safe and resumable on `encode_acts.py`'s pattern: the
JSONL is cut back to the byte count of the last checkpoint, and arrays grow
in `.part` files renamed only when complete.

SaT and GlotLID are not project dependencies; layer them on:

    uv run --with wtpsplit --with fasttext==0.9.3 python encode_sentences.py split
    uv run python encode_sentences.py encode

(fastText 0.9.3 builds from source on Python 3.13; its Python wrapper's
`predict` breaks under numpy 2, so the C++ binding is called directly.)
"""
# disable preallocation so jax and torch can share VRAM (as encode_corpus.py)
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import traceback
import unicodedata
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ontologize.data.segment import capped_spans, dominant_script, expected_scripts, run

ENCODER_ID = "cointegrated/SONAR_200_text_encoder"
OUT = Path("data") / "sonar_embeddings"
NAME = "mc4_sent16"
REF = OUT / "mc4_4M.langs.npy"   # the document order every row indexes
D = 1024
SPLIT_SETTINGS = ("chars", "k", "min_chars", "max_chars", "threshold", "model", "lid", "seed")


def usable(text: str, a: int, b: int, config: str, tag: str,
           min_chars: int, max_chars: int) -> str | None:
    """None if the segment `text[a:b]` may be chosen, else why not:
    `length`, `script` (its letters are mostly another script, or it has
    none). The language identifier's check comes after, on the run."""
    if not min_chars <= b - a <= max_chars:
        return "length"
    if dominant_script(text[a:b]) not in expected_scripts(config, tag):
        return "script"
    return None


def lid_top(model, text: str) -> tuple[str, float]:
    """fastText's top label (without __label__) and probability, through its
    C++ binding: the Python wrapper's `predict` breaks under numpy 2."""
    out = model.f.predict(text.replace("\n", " "), 1, 0.0, "replace")
    if not out:
        return "", 0.0
    p, label = out[0]
    return label.removeprefix("__label__"), float(p)


def choose(text: str, probs, config: str, tag: str, cut: bool, cfg, lid,
           seed) -> tuple[list, Counter, int]:
    """One document's run: [[segment, start, end, text, label, p], ...], the
    reasons segments were not usable, and the number of segments.

    The language identifier sees only the chosen run; a segment it calls
    non-linguistic is dropped and the run chosen again from the same seed,
    so the result is that of filtering first, at a fraction of the calls."""
    spans = capped_spans(text, probs, cfg.threshold, cfg.max_chars)
    if cut and spans:
        spans = spans[:-1]  # the last segment of a cut document is a fragment
    why, ok = Counter(), []
    for a, b, _ in spans:
        reason = usable(text, a, b, config, tag, cfg.min_chars, cfg.max_chars)
        if reason:
            why[reason] += 1
        ok.append(reason is None)
    labels = {}
    while True:
        chosen = run(ok, cfg.k, np.random.default_rng(seed))
        fresh = [i for i in chosen if i not in labels]
        for i in fresh:
            labels[i] = lid_top(lid, text[spans[i][0]:spans[i][1]])
        bad = [i for i in fresh if labels[i][0].startswith("zxx_")]
        if not bad:
            break
        for i in bad:
            ok[i] = False
        why["nonlinguistic"] += len(bad)
    sents = [[i, spans[i][0], spans[i][1], text[spans[i][0]:spans[i][1]],
              labels[i][0], round(labels[i][1], 4)] for i in chosen]
    return sents, why, len(spans)


def token_batches(lengths: np.ndarray, max_tokens: int, max_n: int) -> list[list[int]]:
    """Indices into `lengths` in ascending order of length, cut into batches
    that each pad to at most `max_tokens` tokens and hold at most `max_n`."""
    out, cur = [], []
    for i in np.argsort(lengths, kind="stable"):
        if cur and ((len(cur) + 1) * int(lengths[i]) > max_tokens or len(cur) == max_n):
            out.append(cur)
            cur = []
        cur.append(int(i))
    if cur:
        out.append(cur)
    return out


# split's worker processes: each holds the language identifier
_W = {}


def _init_worker(settings: dict) -> None:
    import fasttext
    from huggingface_hub import hf_hub_download
    from ontologize.data.langs import MC4_TO_SONAR
    _W["cfg"] = SimpleNamespace(**settings)
    _W["lid"] = fasttext.load_model(hf_hub_download(settings["lid"], "model.bin"))
    _W["tags"] = MC4_TO_SONAR


def _work(args) -> tuple[str, dict, int, int]:
    doc, lang, text, probs, cut = args
    cfg = _W["cfg"]
    sents, why, nseg = choose(text, probs, lang, _W["tags"][lang], cut, cfg, _W["lid"],
                              [cfg.seed, doc])
    line = json.dumps({"doc": doc, "lang": lang, "segments": nseg,
                       "usable": nseg - sum(why.values()), "sents": sents},
                      ensure_ascii=False)
    return line, dict(why), nseg, len(sents)


def _reader(q, skip: int, n_docs: int, b: int) -> None:
    """Put batches of (config, text) from the stream on `q`, from document
    `skip` to `n_docs`, then None; an error goes on `q` as ("__error__", tb)."""
    try:
        from ontologize.data.loaders import HFDataSource
        from ontologize.data.multilingual import mc4_data
        it = iter(HFDataSource(mc4_data("allenai/c4", split="train", streaming=True),
                               text_key="text"))
        for _ in range(skip):
            next(it)
        doc = skip
        while doc < n_docs:
            batch = []
            while len(batch) < b and doc + len(batch) < n_docs:
                try:
                    x = next(it)
                except StopIteration:
                    break
                batch.append((x["lang"], x["text"]))
            if not batch:
                break
            q.put(batch)
            doc += len(batch)
        q.put(None)
    except BaseException:
        q.put(("__error__", traceback.format_exc()))
        raise


def split(cfg: argparse.Namespace) -> None:
    import multiprocessing as mp
    from tqdm import tqdm
    from wtpsplit import SaT

    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = outdir / f"{cfg.name}.sentences.jsonl"
    prog = outdir / f"{cfg.name}.split.json"
    ref = np.load(cfg.ref, mmap_mode="r")
    n_docs = min(cfg.n_docs or len(ref), len(ref))
    settings = {k: getattr(cfg, k) for k in SPLIT_SETTINGS}
    settings.update(ref=str(cfg.ref), n_docs=n_docs)

    st = {"docs": 0, "bytes": 0, "sentences": 0, "segments": 0, "why": {}}
    if prog.exists():
        st = json.loads(prog.read_text())
        if st.get("complete"):
            print(f"error: {jpath} is already complete. exiting.")
            return
        if st["settings"] != settings:
            print(f"error: {jpath} was begun with settings {st['settings']}, "
                  f"not {settings}; delete it and {prog} to start over.")
            return
        with open(jpath, "r+b") as f:
            f.truncate(st["bytes"])
        print(f"Resuming {cfg.name} split from document {st['docs']}/{n_docs}")
    else:
        jpath.write_text("")
    st["settings"] = settings
    why = Counter(st["why"])

    sat = SaT(cfg.model)
    sat.half().to("cuda")
    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=8)
    reader = ctx.Process(target=_reader, args=(q, st["docs"], n_docs, cfg.b), daemon=True)
    reader.start()
    pool = ctx.Pool(cfg.workers, initializer=_init_worker,
                    initargs=({k: getattr(cfg, k) for k in SPLIT_SETTINGS},))

    pending = deque()
    written = doc = st["docs"]
    batches = 0
    with open(jpath, "a", encoding="utf-8") as f, \
            tqdm(total=n_docs, initial=doc, desc="Splitting", unit="doc") as bar:

        def checkpoint() -> None:
            f.flush()
            os.fsync(f.fileno())
            st.update(docs=written, bytes=f.tell(), why=dict(why))
            prog.write_text(json.dumps(st))

        def drain(keep: int) -> None:
            """Write finished batches in order, waiting while more than
            `keep` are in flight."""
            nonlocal written, batches
            while pending and (len(pending) > keep or pending[0].ready()):
                for line, w, nseg, nsent in pending.popleft().get():
                    f.write(line + "\n")
                    why.update(w)
                    st["segments"] += nseg
                    st["sentences"] += nsent
                    written += 1
                    bar.update(1)
                batches += 1
                if batches % cfg.flush_each == 0:
                    checkpoint()

        while True:
            batch = q.get()
            if batch is None:
                break
            if isinstance(batch, tuple):
                raise RuntimeError(f"stream reader failed:\n{batch[1]}")
            for j, (lang, _) in enumerate(batch):
                if lang != ref[doc + j]:
                    raise RuntimeError(
                        f"stream drift at document {doc + j}: the stream gives "
                        f"{lang!r}, {cfg.ref} has {ref[doc + j]!r}")
            full = [unicodedata.normalize("NFC", text) for _, text in batch]
            texts = [x[:cfg.chars] for x in full]
            probs = sat.predict_proba(texts, stride=64, batch_size=cfg.sat_batch)
            pending.append(pool.map_async(
                _work, [(doc + j, lang, text, p, len(full[j]) > cfg.chars)
                        for j, ((lang, _), text, p) in enumerate(zip(batch, texts, probs))],
                chunksize=16))
            doc += len(batch)
            drain(keep=cfg.in_flight)
        drain(keep=0)
        checkpoint()
    pool.close()
    pool.join()
    reader.join(timeout=60)
    st["complete"] = written == n_docs
    prog.write_text(json.dumps(st))
    if not st["complete"]:
        raise SystemExit(f"stream ended at document {written} of {n_docs}")
    print(f"✓ {jpath}: {written} documents, {st['sentences']} sentences "
          f"({st['sentences'] / max(written, 1):.1f} per document) of {st['segments']} "
          f"segments; not usable: {dict(why)}")


def encode_stage(cfg: argparse.Namespace) -> None:
    import torch as t
    from tqdm import tqdm
    from ontologize.data.langs import MC4_TO_SONAR
    from ontologize.data.loaders import doc_boundaries
    from ontologize.data.pretrained import encode, pretrained_transformer

    outdir = Path(cfg.out)
    name = cfg.name
    jpath = outdir / f"{name}.sentences.jsonl"
    sp = json.loads((outdir / f"{name}.split.json").read_text())
    if not sp.get("complete") and not cfg.partial:
        print(f"error: the split of {name} is not complete (--partial encodes it as is)")
        return
    n, n_docs = sp["sentences"], sp["docs"]
    final = outdir / f"{name}.npy"
    if final.exists():
        print(f"error: {final} already exists. exiting.")
        return
    paths = {k: outdir / f"{name}{suffix}" for k, suffix in
             (("E", ".npy.part"), ("I", ".index.npy.part"), ("S", ".docstart.npy.part"),
              ("L", ".langs.npy.part"), ("G", ".lid.npy.part"), ("P", ".lid_p.npy.part"))}
    prog = outdir / f"{name}.encode.json"
    shapes = {"E": ((n, D), np.float32), "I": ((n, 3), np.int32), "S": ((n_docs,), np.int64),
              "L": ((n,), "U8"), "G": ((n,), "U16"), "P": ((n,), np.float32)}
    st = {"rows": 0, "docs": 0, "bytes": 0, "tf32": cfg.tf32, "maxlen": cfg.maxlen}
    if prog.exists() and paths["E"].exists():
        st = json.loads(prog.read_text())
        if (st["tf32"], st["maxlen"]) != (cfg.tf32, cfg.maxlen):
            print(f"error: {name} was begun with tf32={st['tf32']}, maxlen={st['maxlen']}")
            return
        A = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in paths.items()}
        assert A["E"].shape == (n, D), f"{paths['E']} has shape {A['E'].shape}, expected {(n, D)}"
        print(f"Resuming {name} encode from row {st['rows']}/{n} ({st['docs']} documents)")
    else:
        A = {k: np.lib.format.open_memmap(paths[k], mode="w+", dtype=dt, shape=sh)
             for k, (sh, dt) in shapes.items()}

    dev = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    t.backends.cuda.matmul.allow_tf32 = cfg.tf32
    t.backends.cudnn.allow_tf32 = cfg.tf32
    model, tok = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    model.eval()
    pad = tok.pad_token_id

    def checkpoint(nbytes: int) -> None:
        for a in A.values():
            a.flush()
        st["bytes"] = nbytes
        prog.write_text(json.dumps(st))

    def encode_window(window: list) -> None:
        """Encode (row, config, text) triples: tokenized per tag, then
        batched across tags by length."""
        by_tag = {}
        for j, (_, lang, _) in enumerate(window):
            by_tag.setdefault(MC4_TO_SONAR[lang], []).append(j)
        ids = [None] * len(window)
        for tag, js in by_tag.items():
            tok.src_lang = tag
            enc = tok([window[j][2] for j in js], truncation=True, max_length=cfg.maxlen)
            for j, e in zip(js, enc["input_ids"]):
                ids[j] = e
        lengths = np.array([len(e) for e in ids])
        for b in token_batches(lengths, cfg.max_tokens, cfg.b):
            x = np.full((len(b), lengths[b].max()), pad, np.int64)
            m = np.zeros_like(x)
            for r, j in enumerate(b):
                x[r, :lengths[j]] = ids[j]
                m[r, :lengths[j]] = 1
            emb = encode(model, {"input_ids": x, "attention_mask": m}, dev, normalize=False)
            A["E"][[window[j][0] for j in b]] = np.asarray(emb)

    rows, docs = st["rows"], st["docs"]
    with open(jpath, "rb") as f, tqdm(total=n, initial=rows, desc="Encoding", unit="sent") as bar:
        f.seek(st["bytes"])
        while docs < n_docs:
            window, line = [], b""
            while len(window) < cfg.window and docs < n_docs:
                line = f.readline()
                if not line:
                    break
                r = json.loads(line)
                assert r["doc"] == docs, f"{jpath} line {docs} is document {r['doc']}"
                A["S"][docs] = rows
                for pos, (seg, _a, _b, text, label, p) in enumerate(r["sents"]):
                    A["I"][rows] = (r["doc"], pos, seg)
                    A["L"][rows], A["G"][rows], A["P"][rows] = r["lang"], label, p
                    window.append((rows, r["lang"], text))
                    rows += 1
                docs += 1
            if not line:
                break
            encode_window(window)
            st.update(rows=rows, docs=docs)
            checkpoint(f.tell())
            bar.update(len(window))
    assert rows == n and docs == n_docs, f"encoded {rows}/{n} rows, {docs}/{n_docs} documents"

    # per-dimension inverse-variance MSE weights, mean-1 normalized, over the
    # rows before the earliest document boundary the holdout could snap to,
    # as encode_acts.py computes them
    _, wide = doc_boundaries(np.asarray(A["S"]), n, cfg.holdout)
    train = n - wide
    mean, var, seen = np.zeros(D), np.zeros(D), 0
    for i in range(0, train, 1 << 16):
        c = np.asarray(A["E"][i:min(i + (1 << 16), train)], np.float64)
        mu, v, k = c.mean(0), c.var(0), len(c)
        delta = mu - mean
        var = (seen * var + k * v + delta ** 2 * seen * k / (seen + k)) / (seen + k)
        mean += delta * k / (seen + k)
        seen += k
    w = 1.0 / np.maximum(var, np.finfo(np.float64).tiny)
    np.save(outdir / f"{name}.mse_weights.npy", (w / w.mean()).astype(np.float32))

    for a in A.values():
        a.flush()
    del A
    norms = np.linalg.norm(np.load(paths["E"], mmap_mode="r")[:min(n, 1 << 16)], axis=1)
    (outdir / f"{name}.meta.json").write_text(json.dumps({
        "n": n, "d": D, "dtype": "float32", "n_docs": n_docs, "order": "document",
        "encoder": ENCODER_ID, "encoder_dtype": "float32",
        "encoder_matmul": "tf32" if cfg.tf32 else "float32", "maxlen": cfg.maxlen,
        "pooling": "masked mean", "src_tags": "MC4_TO_SONAR",
        "source": "allenai/c4 train, interleaved over MC4_TO_SONAR, in mc4_4M's document order",
        "split": sp["settings"], "segments": sp["segments"],
        "not_usable": sp["why"],
        "not_usable_note": "nonlinguistic counts only segments GlotLID saw (in or tried for a run)",
        "index": f"{name}.index.npy (doc, pos, segment)", "docstart": f"{name}.docstart.npy",
        "sentences": f"{name}.sentences.jsonl",
        "mse_weights": f"{name}.mse_weights.npy", "mse_weights_train_rows": train,
        "holdout": cfg.holdout, "norm_median_head": float(np.median(norms)),
    }, indent=2))
    for k, suffix in (("E", ".npy"), ("I", ".index.npy"), ("S", ".docstart.npy"),
                      ("L", ".langs.npy"), ("G", ".lid.npy"), ("P", ".lid_p.npy")):
        os.replace(paths[k], outdir / f"{name}{suffix}")
    st["complete"] = True
    prog.write_text(json.dumps(st))
    lo, hi = doc_boundaries(np.load(outdir / f"{name}.docstart.npy"), n, cfg.holdout)
    print(f"✓ {final}: {n} x {D} float32, {n_docs} documents, {n / n_docs:.1f} rows/doc; "
          f"whole-document tails nearest {cfg.holdout} rows: {lo}, {hi}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    for stage in ("split", "encode"):
        s = sub.add_parser(stage)
        s.add_argument("--name", default=NAME)
        s.add_argument("-o", "--out", default=str(OUT))
    s = sub.choices["split"]
    s.add_argument("--ref", default=str(REF), help="langs sidecar of the document cache to follow")
    s.add_argument("--n-docs", type=int, default=0, help="0 = every document of --ref")
    s.add_argument("--chars", type=int, default=8000, help="segment each document's first CHARS characters")
    s.add_argument("--k", type=int, default=16, help="sentences per document, at most")
    s.add_argument("--min-chars", type=int, default=20)
    s.add_argument("--max-chars", type=int, default=200, help="the cap")
    s.add_argument("--threshold", type=float, default=0.25, help="SaT's default for the -sm models")
    s.add_argument("--model", default="sat-3l-sm")
    s.add_argument("--lid", default="cis-lmu/glotlid", help="fastText language identifier (HF repo)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-b", "--b", type=int, default=256, help="documents per SaT batch")
    s.add_argument("--sat-batch", type=int, default=64, help="SaT's own batch, in 512-token blocks")
    s.add_argument("--workers", type=int, default=6, help="processes choosing runs (each loads GlotLID)")
    s.add_argument("--in-flight", type=int, default=4, help="SaT batches queued to the workers, at most")
    s.add_argument("--flush-each", type=int, default=50, help="batches between checkpoints")
    s = sub.choices["encode"]
    s.add_argument("-b", "--b", type=int, default=2048, help="sentences per encoder batch, at most")
    s.add_argument("--max-tokens", type=int, default=65536, help="padded tokens per encoder batch, at most")
    s.add_argument("--window", type=int, default=65536, help="sentences per checkpoint")
    s.add_argument("--maxlen", type=int, default=256, help="SONAR tokens per sentence, at most")
    s.add_argument("--tf32", action="store_true",
                   help="TF32 matmuls: ~1.8x faster, cosine to float32 >= 0.99998")
    s.add_argument("--holdout", type=int, default=32768,
                   help="rows the trainers hold out; snaps to a document boundary")
    s.add_argument("--partial", action="store_true", help="encode an incomplete split")
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    if cfg.stage == "split":
        split(cfg)
    else:
        encode_stage(cfg)


if __name__ == "__main__":
    main()
