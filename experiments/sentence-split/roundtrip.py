"""Which SaT model gives SONAR better sentences?

There is no gold segmentation of mC4, so `segtest.py` can only measure how
often two SaT models agree. This scores segmentations by the LCM team's
criterion for comparing segmenters: encode each segment with SONAR, decode
it back, and compare the decode with the segment (chrF2). A segment that is
one sentence should survive the round trip better than one that straddles a
boundary or stops mid-sentence.

Shorter segments survive better whatever their boundaries, so a model that
splits more would win on the mean alone. Scores are therefore also compared
within bins of segment length, where a difference reflects where the
boundaries fall.

Every segment is encoded under its config's own tag (`MC4_TO_SONAR`) without
the L2 step, the embedding the sentence cache would hold, and decoded greedy
into the same language at that embedding's own norm. Segments both models
produce are decoded once.

    uv run --with wtpsplit python experiments/sentence-split/roundtrip.py

Writes roundtrip.json and roundtrip_bins.csv under --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

UNCOVERED = {"ht", "lb", "lo", "mi", "ny", "sd", "sm", "sn", "so", "st", "su"}
BINS = [0, 50, 100, 150, 201]


def group(lang: str) -> str:
    if lang in UNCOVERED:
        return "uncovered"
    return "romanized" if lang.endswith("-Latn") else "covered"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default="data/out/sonar/langtags/texts.jsonl")
    ap.add_argument("--out", default="data/out/sonar/sentsplit")
    ap.add_argument("--per-lang", type=int, default=25)
    ap.add_argument("--chars", type=int, default=4000)
    ap.add_argument("--models", nargs=2, default=["sat-3l-sm", "sat-12l-sm"])
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--b", type=int, default=128, help="SONAR batch size")
    args = ap.parse_args()

    import torch as t
    from transformers import AutoTokenizer, M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from wtpsplit import SaT
    from ontologize.data.langs import MC4_TO_SONAR
    from ontologize.data.pretrained import pretrained_transformer, encode
    from segtest import capped_spans, load_texts
    from textfid import ENCODER_ID, DECODER_ID, chrf

    out = Path(args.out)
    docs = load_texts(Path(args.texts), args.per_lang, args.chars)

    # segments of each document under each model
    segs = {m: [] for m in args.models}
    for m in args.models:
        sat = SaT(m)
        sat.half().to("cuda")
        probs = list(sat.predict_proba([d["text"] for d in docs], stride=64, batch_size=64))
        for d, p in zip(docs, probs):
            spans = capped_spans(d["text"], p, args.threshold, args.max_chars)
            if d["cut"] and spans:
                spans = spans[:-1]
            segs[m].append([d["text"][a:b] for a, b, _ in spans])
        del sat
        t.cuda.empty_cache()

    # each distinct (language, segment) once
    uniq = sorted({(d["lang"], s) for m in args.models for d, ss in zip(docs, segs[m]) for s in ss},
                  key=lambda x: len(x[1]))
    print(f"{len(docs)} documents; segments: "
          + ", ".join(f"{m} {sum(map(len, segs[m]))}" for m in args.models)
          + f"; {len(uniq)} distinct")

    dev = t.device("cuda")
    enc, tok = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    enc.eval()
    dec = M2M100ForConditionalGeneration.from_pretrained(DECODER_ID).to(dev)
    dec.eval()
    score = {}
    for i in range(0, len(uniq), args.b):
        chunk = uniq[i:i + args.b]
        # one language per tokenizer call: the tag is the first token
        embs, ntoks = [None] * len(chunk), [0] * len(chunk)
        for lang in {l for l, _ in chunk}:
            idx = [j for j, (l, _) in enumerate(chunk) if l == lang]
            tok.src_lang = MC4_TO_SONAR[lang]
            batch = tok([chunk[j][1] for j in idx], padding=True, truncation=True,
                        max_length=256, return_tensors="np")
            E = np.asarray(encode(enc, {k: batch[k] for k in ("input_ids", "attention_mask")},
                                  dev, normalize=False))
            for r, j in enumerate(idx):
                embs[j] = E[r]
                ntoks[j] = int(batch["attention_mask"][r].sum())
        Y = t.from_numpy(np.stack(embs)).to(dev, t.float32).unsqueeze(1)
        bos = t.tensor([tok.convert_tokens_to_ids(MC4_TO_SONAR[l]) for l, _ in chunk], device=dev)
        with t.no_grad():
            # every row starts with its own language tag after the
            # decoder-start token, as forced_bos does for a single language
            start = t.full((len(chunk), 1), dec.config.decoder_start_token_id, device=dev)
            gen = dec.generate(encoder_outputs=BaseModelOutput(last_hidden_state=Y),
                               decoder_input_ids=t.cat([start, bos[:, None]], 1),
                               max_new_tokens=2 * max(ntoks) + 10, num_beams=1)
        for (lang, s), g in zip(chunk, gen):
            score[(lang, s)] = chrf(s, tok.decode(g, skip_special_tokens=True).strip())
        if (i // args.b) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(uniq)}", flush=True)

    # per-document character-weighted round-trip chrF, and per-segment rows
    res = {"models": args.models, "docs": len(docs), "threshold": args.threshold,
           "max_chars": args.max_chars, "distinct_segments": len(uniq)}
    rows = []
    for m in args.models:
        for d, ss in zip(docs, segs[m]):
            for s in ss:
                rows.append((m, d["row"], d["lang"], group(d["lang"]), len(s), score[(d["lang"], s)]))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["model", "row", "lang", "group", "chars", "chrf"])
    df["w"] = df.chars * df.chrf
    doc = df.groupby(["model", "row", "lang", "group"]).agg(w=("w", "sum"), chars=("chars", "sum"),
                                                           n=("chrf", "size")).reset_index()
    doc["chrf"] = doc.w / doc.chars
    wide = doc.pivot(index=["row", "lang", "group"], columns="model", values="chrf").dropna()
    a, b = args.models
    diff = wide[b] - wide[a]
    res["doc_chrf"] = {m: round(float(wide[m].mean()), 4) for m in args.models}
    res["doc_diff"] = {"mean": round(float(diff.mean()), 4),
                       "se": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 4),
                       "share_better": round(float((diff > 0).mean()), 3),
                       "share_worse": round(float((diff < 0).mean()), 3)}
    res["by_group"] = {g: {"diff": round(float(diff[wide.index.get_level_values("group") == g].mean()), 4),
                           "se": round(float(diff[wide.index.get_level_values("group") == g].std(ddof=1)
                                             / np.sqrt((wide.index.get_level_values("group") == g).sum())), 4)}
                       for g in ("covered", "uncovered", "romanized")}
    res["segments"] = {m: {"n": int((df.model == m).sum()),
                           "chars_median": float(df[df.model == m].chars.median())} for m in args.models}
    # within length bins: is a segment of a given length a better unit?
    df["bin"] = pd.cut(df.chars, BINS, right=False)
    bins = df.groupby(["bin", "model"], observed=True).chrf.agg(["mean", "count"]).unstack("model")
    bins.columns = [f"{s}_{m}" for s, m in bins.columns]
    bins.to_csv(out / "roundtrip_bins.csv")
    res["bins"] = {str(k): {c: (round(float(v), 4) if "mean" in c else int(v)) for c, v in r.items()}
                   for k, r in bins.iterrows()}
    (out / "roundtrip.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
