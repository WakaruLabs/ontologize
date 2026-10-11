"""Can SaT split the mC4 corpus into SONAR-sized sentences?

A test of Segment any Text (SaT; Frohmann et al. 2024) on the documents
`experiments/lang-tags/retag.py texts` already replayed (400 per mC4
config), before building a sentence-level cache. It reproduces the LCM
team's "SaT Capped": split at SaT's sentence-boundary threshold and at
the input's own newlines, then break every segment longer than the cap
(200 characters) at the fewest, most probable further boundaries.

wtpsplit's own `max_length` is not that: with a uniform prior its Viterbi
maximizes the summed log boundary probability, so it makes as few splits
as the cap allows and packs several sentences into each segment. Here the
constrained search runs only inside segments over the cap (`cap_spans`).

Per language it reports how many segments a document yields, their
lengths in characters and SONAR tokens, how many the cap had to break,
how many boundaries fall inside a word, the share of documents with a
segment in the eligible window, and the agreement of the 3-layer model
with the 12-layer one on a subset. It also writes worked examples.

Needs wtpsplit, which the project does not depend on:

    uv run --with wtpsplit python experiments/sentence-split/segtest.py

Outputs under --out (default data/out/sonar/sentsplit): langs.csv,
summary.json, segments.jsonl (each document's spans), examples.md.
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

from ontologize.data.segment import capped_spans, threshold_spans  # noqa: E402

SENTENCE_FINAL = set(".!?。！？؟۔।॥။።፧…។៕։")
# scripts that do not separate words with spaces: a boundary between two
# letters is not evidence of a mid-word split there
UNSPACED = {"zh", "ja", "th", "lo", "km", "my"}


def midword(text: str, cut: int) -> bool:
    """Whether a cut before character `cut` separates two letters or digits."""
    return 0 < cut < len(text) and text[cut - 1].isalnum() and text[cut].isalnum()


def boundary_f1(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def load_texts(path: Path, per_lang: int, chars: int) -> list[dict]:
    """The first `per_lang` documents of each language, each cut to its first
    `chars` characters; `cut` records whether it was."""
    seen, rows = {}, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if seen.get(r["lang"], 0) >= per_lang:
                continue
            seen[r["lang"]] = seen.get(r["lang"], 0) + 1
            text = unicodedata.normalize("NFC", r["text"])
            rows.append({"row": r["row"], "lang": r["lang"], "text": text[:chars],
                         "cut": len(text) > chars})
    return rows


def segment(sat, texts: list[str], stride: int, batch_size: int) -> list[np.ndarray]:
    return list(sat.predict_proba(texts, stride=stride, batch_size=batch_size))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default="data/out/sonar/langtags/texts.jsonl")
    ap.add_argument("--out", default="data/out/sonar/sentsplit")
    ap.add_argument("--per-lang", type=int, default=200)
    ap.add_argument("--chars", type=int, default=4000, help="segment each document's first CHARS characters")
    ap.add_argument("--model", default="sat-3l-sm")
    ap.add_argument("--compare-model", default="sat-12l-sm")
    ap.add_argument("--compare-per-lang", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.25, help="SaT's default for the -sm models")
    ap.add_argument("--max-chars", type=int, default=200, help="the cap")
    ap.add_argument("--min-chars", type=int, default=20, help="lower edge of the eligible window")
    ap.add_argument("--stride", type=int, default=64, help="SaT.split's default")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--examples", nargs="*", default=[
        "en", "de", "fr", "ru", "zh", "ja", "ko", "ar", "hi", "th", "my", "km",
        "lo", "ht", "lb", "mi", "ny", "sd", "sm", "sn", "so", "st", "su",
        "bg-Latn", "el-Latn", "hi-Latn", "ja-Latn", "ru-Latn"])
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from wtpsplit import SaT
    from ontologize.data.langs import MC4_TO_SONAR

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = load_texts(Path(args.texts), args.per_lang, args.chars)
    texts = [d["text"] for d in docs]
    print(f"{len(docs)} documents, {sum(map(len, texts)):,} characters")

    sat = SaT(args.model)
    sat.half().to("cuda")
    segment(sat, texts[:64], args.stride, args.batch_size)  # warm up
    t0 = time.time()
    probs = segment(sat, texts, args.stride, args.batch_size)
    secs = time.time() - t0

    tok = AutoTokenizer.from_pretrained("cointegrated/SONAR_200_text_encoder")
    per_lang, records = {}, []
    for d, p in zip(docs, probs):
        spans = capped_spans(d["text"], p, args.threshold, args.max_chars)
        base = threshold_spans(d["text"], p, args.threshold)
        if d["cut"] and spans:
            # the last segment of a truncated document is a fragment
            spans, base = spans[:-1], base[:-1]
        segs = [d["text"][a:b] for a, b, _ in spans]
        tok.src_lang = MC4_TO_SONAR[d["lang"]]
        ntok = [len(x) for x in tok(segs)["input_ids"]] if segs else []
        cuts = [a for a, _, _ in spans[1:]]
        model_cuts = [c for c in cuts if d["text"][c - 1] != "\n"]
        L = per_lang.setdefault(d["lang"], {"docs": 0, "chars": 0, "base": [], "len": [],
                                            "tok": [], "capped": 0, "midword": 0, "cuts": 0,
                                            "final": 0, "eligible_docs": 0})
        L["docs"] += 1
        L["chars"] += len(d["text"])
        L["base"] += [b - a for a, b in base]
        L["len"] += [len(s) for s in segs]
        L["tok"] += ntok
        L["capped"] += sum(c for _, _, c in spans)
        L["cuts"] += len(model_cuts)
        if d["lang"].split("-")[0] not in UNSPACED or d["lang"].endswith("-Latn"):
            L["midword"] += sum(midword(d["text"], c) for c in model_cuts)
        L["final"] += sum(s[-1] in SENTENCE_FINAL for s in segs)
        L["eligible_docs"] += any(args.min_chars <= len(s) <= args.max_chars for s in segs)
        records.append({"row": d["row"], "lang": d["lang"], "cut": d["cut"],
                        "spans": [[a, b, int(c)] for a, b, c in spans]})

    # agreement with the larger model on a subset, uncapped boundaries
    agree = {}
    if args.compare_per_lang:
        sub, count = [], {}
        for i, d in enumerate(docs):
            count[d["lang"]] = count.get(d["lang"], 0) + 1
            if count[d["lang"]] <= args.compare_per_lang:
                sub.append(i)
        big = SaT(args.compare_model)
        big.half().to("cuda")
        t1 = time.time()
        bprobs = segment(big, [texts[i] for i in sub], args.stride, args.batch_size)
        big_secs = time.time() - t1
        for i, bp in zip(sub, bprobs):
            d = docs[i]
            a = {s for s, _ in threshold_spans(d["text"], probs[i], args.threshold)[1:]}
            b = {s for s, _ in threshold_spans(d["text"], bp, args.threshold)[1:]}
            agree.setdefault(d["lang"], []).append(boundary_f1(a, b))
        sub_chars = sum(len(texts[i]) for i in sub)

    rows = []
    for lang, L in sorted(per_lang.items()):
        base, ln, tk = map(np.asarray, (L["base"], L["len"], L["tok"]))
        rows.append({
            "lang": lang,
            "docs": L["docs"],
            "segments": len(ln),
            "seg_per_kchar": round(1000 * len(ln) / max(L["chars"], 1), 2),
            "chars_median": float(np.median(ln)) if len(ln) else np.nan,
            "chars_p90": float(np.percentile(ln, 90)) if len(ln) else np.nan,
            "over_cap_before": round(float((base > args.max_chars).mean()), 3) if len(base) else np.nan,
            "capped_share": round(L["capped"] / max(len(ln), 1), 3),
            "under_min": round(float((ln < args.min_chars).mean()), 3) if len(ln) else np.nan,
            "tok_median": float(np.median(tk)) if len(tk) else np.nan,
            "tok_p99": float(np.percentile(tk, 99)) if len(tk) else np.nan,
            "tok_max": int(tk.max()) if len(tk) else 0,
            "midword": round(L["midword"] / max(L["cuts"], 1), 3)
                       if lang.split("-")[0] not in UNSPACED or lang.endswith("-Latn") else np.nan,
            "ends_final_punct": round(L["final"] / max(len(ln), 1), 3),
            "eligible_docs": round(L["eligible_docs"] / L["docs"], 3),
            "f1_vs_12l": round(float(np.mean(agree[lang])), 3) if lang in agree else np.nan,
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "langs.csv", index=False)
    allp = np.concatenate([np.asarray(L["tok"]) for L in per_lang.values()])
    alll = np.concatenate([np.asarray(L["len"]) for L in per_lang.values()])
    summary = {
        "model": args.model, "threshold": args.threshold, "max_chars": args.max_chars,
        "docs": len(docs), "chars": int(sum(map(len, texts))),
        "seconds": round(secs, 1), "docs_per_s": round(len(docs) / secs, 1),
        "chars_per_s": round(sum(map(len, texts)) / secs),
        "segments": int(len(alll)), "chars_median": float(np.median(alll)),
        "tok_median": float(np.median(allp)), "tok_p99": float(np.percentile(allp, 99)),
        "tok_max": int(allp.max()),
        "capped_share": round(sum(L["capped"] for L in per_lang.values()) / len(alll), 3),
        "eligible_docs": round(sum(L["eligible_docs"] for L in per_lang.values()) / len(docs), 3),
    }
    if agree:
        summary.update({"compare_model": args.compare_model,
                        "compare_docs": len(sub),
                        "compare_chars_per_s": round(sub_chars / big_secs),
                        "f1_vs_compare_mean": round(float(np.mean([v for x in agree.values() for v in x])), 3)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with open(out / "segments.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(out / "examples.md", "w") as f:
        f.write(f"# SaT Capped examples ({args.model}, threshold {args.threshold}, "
                f"cap {args.max_chars} characters)\n\nEach line is one segment: "
                "length in characters, `C` if the cap made it.\n")
        for lang in args.examples:
            ds = [(d, r) for d, r in zip(docs, records) if d["lang"] == lang][:2]
            if not ds:
                continue
            f.write(f"\n## {lang}\n")
            for d, r in ds:
                f.write(f"\nrow {d['row']}\n\n")
                for a, b, c in r["spans"][:12]:
                    seg = d["text"][a:b].replace("\n", " ⏎ ")
                    f.write(f"- {b - a:4d}{' C' if c else '  '} | {seg}\n")

    print(json.dumps(summary, indent=1))
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
