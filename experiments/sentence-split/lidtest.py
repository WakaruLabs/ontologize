"""Are the segments in the language their tag says, and does retagging help?

Every segment inherits its mC4 config's label, which picks the SONAR source
tag it is encoded under (`MC4_TO_SONAR`). This runs two language
identifiers over the segments `segtest.py` wrote: NLLB's fastText model
(`facebook/fasttext-language-identification`, whose labels are SONAR's
FLORES-200 codes) and GlotLID (`cis-lmu/glotlid`, many more languages and
scripts). Per config it reports how many segments match the config's tag,
how many differ only in script, how many confidently name another language
and which, how many are uncertain, how that varies with length, and how
many documents keep an eligible segment if mismatches are filtered out.

Then the retag check: for a sample of segments NLLB confidently places in
another SONAR language, encode each under its config's tag and under the
predicted one (no L2 step), decode both greedy into the predicted language,
and compare chrF2 against the segment. If retagging helps, the predicted
tag's round trip is better.

    uv run --with fasttext==0.9.3 python experiments/sentence-split/lidtest.py

(fastText's Python wrapper calls numpy in a way numpy 2 rejects; this calls
its C++ binding directly.) Writes lid_langs.csv, lid_summary.json,
lid_examples.md and lid_retag.json under --out.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BINS = [0, 20, 50, 100, 201]


def predict(model, text: str) -> tuple[str, float]:
    """Top label (without __label__) and its probability."""
    out = model.f.predict(text.replace("\n", " "), 1, 0.0, "replace")
    if not out:
        return "", 0.0
    p, label = out[0]
    return label.replace("__label__", ""), float(p)


def kind(pred: str, p: float, expected: str, sure: float) -> str:
    """match: the config's tag; script: its language in another script;
    other: another language at probability >= `sure`; unsure: the rest."""
    if pred == expected:
        return "match"
    if pred.split("_")[0] == expected.split("_")[0]:
        return "script"
    return "other" if p >= sure else "unsure"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default="data/out/sonar/langtags/texts.jsonl")
    ap.add_argument("--segments", default="data/out/sonar/sentsplit/segments.jsonl")
    ap.add_argument("--out", default="data/out/sonar/sentsplit")
    ap.add_argument("--per-lang", type=int, default=200, help="as segtest.py ran")
    ap.add_argument("--chars", type=int, default=4000, help="as segtest.py ran")
    ap.add_argument("--sure", type=float, default=0.9, help="probability for a confident other-language call")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=200)
    ap.add_argument("--retag-per-lang", type=int, default=30)
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import fasttext
    from huggingface_hub import hf_hub_download
    from ontologize.data.langs import MC4_TO_SONAR
    from segtest import load_texts
    from textfid import ENCODER_ID

    out = Path(args.out)
    docs = {d["row"]: d for d in load_texts(Path(args.texts), args.per_lang, args.chars)}
    segs = []
    with open(args.segments) as f:
        for line in f:
            r = json.loads(line)
            text = docs[r["row"]]["text"]
            for a, b, c in r["spans"]:
                segs.append({"row": r["row"], "lang": r["lang"], "text": text[a:b], "capped": bool(c)})
    print(f"{len(segs)} segments from {len(docs)} documents")

    nllb = fasttext.load_model(hf_hub_download("facebook/fasttext-language-identification", "model.bin"))
    glot = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
    for s in segs:
        s["nllb"], s["nllb_p"] = predict(nllb, s["text"])
        s["glot"], s["glot_p"] = predict(glot, s["text"])
        exp = MC4_TO_SONAR[s["lang"]]
        s["nllb_kind"] = kind(s["nllb"], s["nllb_p"], exp, args.sure)
        s["glot_kind"] = kind(s["glot"], s["glot_p"], exp, args.sure)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER_ID)
    # SONAR takes a tag only if its tokenizer has it as one token
    labels = {s["nllb"] for s in segs} | set(MC4_TO_SONAR.values())
    sonar_codes = {l for l in labels if tok.convert_tokens_to_ids(l) != tok.unk_token_id}

    import pandas as pd
    df = pd.DataFrame(segs)
    df["chars"] = df.text.str.len()
    df["eligible"] = df.chars.between(args.min_chars, args.max_chars)
    rows = []
    for lang, g in df.groupby("lang"):
        n = len(g)
        other = g[g.nllb_kind == "other"]
        top = Counter(other.nllb).most_common(3)
        docs_ok = g[g.eligible].groupby("row").size().index
        docs_match = g[g.eligible & g.nllb_kind.isin(["match", "script"])].groupby("row").size().index
        docs_match_glot = g[g.eligible & g.glot_kind.isin(["match", "script"])].groupby("row").size().index
        rows.append({
            "lang": lang, "tag": MC4_TO_SONAR[lang], "segments": n,
            "match": round((g.nllb_kind == "match").mean(), 3),
            "script": round((g.nllb_kind == "script").mean(), 3),
            "other": round(len(other) / n, 3),
            "unsure": round((g.nllb_kind == "unsure").mean(), 3),
            "eng_in_other": round((other.nllb == "eng_Latn").sum() / max(len(other), 1), 3),
            "top_other": "; ".join(f"{l} {c}" for l, c in top),
            "glot_match_or_script": round(g.glot_kind.isin(["match", "script"]).mean(), 3),
            "glot_top": "; ".join(f"{l} {c}" for l, c in Counter(g.glot).most_common(2)),
            "docs_eligible": len(docs_ok),
            "docs_eligible_matching": len(docs_match),
            "docs_eligible_matching_glot": len(docs_match_glot),
        })
    langs = pd.DataFrame(rows)
    langs.to_csv(out / "lid_langs.csv", index=False)

    df["bin"] = pd.cut(df.chars, BINS, right=False)
    by_len = (df.groupby("bin", observed=True).nllb_kind.value_counts(normalize=True)
              .unstack().round(3))
    w = df.chars / df.chars.sum()
    summary = {
        "segments": len(df), "docs": len(docs), "sure": args.sure,
        "nllb": {k: round(float((df.nllb_kind == k).mean()), 4) for k in ("match", "script", "other", "unsure")},
        "nllb_by_chars": {k: round(float(w[df.nllb_kind == k].sum()), 4) for k in ("match", "script", "other", "unsure")},
        "glot": {k: round(float((df.glot_kind == k).mean()), 4) for k in ("match", "script", "other", "unsure")},
        "other_is_english": round(float((df[df.nllb_kind == "other"].nllb == "eng_Latn").mean()), 3),
        "other_in_sonar": round(float(df[df.nllb_kind == "other"].nllb.isin(sonar_codes).mean()), 3),
        "by_length": {str(k): v.dropna().to_dict() for k, v in by_len.iterrows()},
        "docs_with_eligible": int(langs.docs_eligible.sum()),
        "docs_with_eligible_matching_nllb": int(langs.docs_eligible_matching.sum()),
        "docs_with_eligible_matching_glot": int(langs.docs_eligible_matching_glot.sum()),
    }

    rng = random.Random(args.seed)
    with open(out / "lid_examples.md", "w") as f:
        f.write(f"# Segments NLLB's identifier places in another language (p >= {args.sure})\n\n"
                "Up to 6 per config: predicted label, probability, GlotLID's label, text.\n")
        for lang, g in df[df.nllb_kind == "other"].groupby("lang"):
            f.write(f"\n## {lang} ({MC4_TO_SONAR[lang]})\n\n")
            for _, s in g.sample(min(6, len(g)), random_state=args.seed).iterrows():
                f.write(f"- {s.nllb} {s.nllb_p:.2f} / {s.glot} | {s.text[:160]}\n")

    # retag check: config tag against predicted tag, both decoded into the
    # predicted language
    cand = df[(df.nllb_kind == "other") & df.eligible & df.nllb.isin(sonar_codes)]
    sample = (cand.sample(frac=1, random_state=args.seed)
              .groupby("lang").head(args.retag_per_lang))
    import torch as t
    from transformers import M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from ontologize.data.pretrained import pretrained_transformer, encode
    from textfid import DECODER_ID, chrf
    dev = t.device("cuda")
    enc, tok = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    enc.eval()
    dec = M2M100ForConditionalGeneration.from_pretrained(DECODER_ID).to(dev)
    dec.eval()

    def embed(texts, tags):
        E = np.zeros((len(texts), 1024), np.float32)
        for tag in set(tags):
            idx = [i for i, x in enumerate(tags) if x == tag]
            tok.src_lang = tag
            bt = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=256,
                     return_tensors="np")
            E[idx] = np.asarray(encode(enc, {k: bt[k] for k in ("input_ids", "attention_mask")},
                                       dev, normalize=False))
        return E

    def roundtrip(E, targets, texts):
        Y = t.from_numpy(E).to(dev).unsqueeze(1)
        start = t.full((len(E), 1), dec.config.decoder_start_token_id, device=dev)
        bos = t.tensor([tok.convert_tokens_to_ids(x) for x in targets], device=dev)[:, None]
        with t.no_grad():
            gen = dec.generate(encoder_outputs=BaseModelOutput(last_hidden_state=Y),
                               decoder_input_ids=t.cat([start, bos], 1),
                               max_new_tokens=160, num_beams=1)
        return [chrf(x, tok.decode(g, skip_special_tokens=True).strip()) for x, g in zip(texts, gen)]

    sample = sample.sort_values("chars")
    texts = sample.text.tolist()
    cfg_tags = [MC4_TO_SONAR[l] for l in sample.lang]
    pred_tags = sample.nllb.tolist()
    s_cfg, s_pred = [], []
    for i in range(0, len(texts), args.b):
        sl = slice(i, i + args.b)
        s_cfg += roundtrip(embed(texts[sl], cfg_tags[sl]), pred_tags[sl], texts[sl])
        s_pred += roundtrip(embed(texts[sl], pred_tags[sl]), pred_tags[sl], texts[sl])
    sample = sample.assign(chrf_config_tag=s_cfg, chrf_predicted_tag=s_pred)
    d = sample.chrf_predicted_tag - sample.chrf_config_tag
    english = sample.nllb == "eng_Latn"
    retag = {
        "segments": len(sample), "configs": int(sample.lang.nunique()),
        "chrf_config_tag": round(float(sample.chrf_config_tag.mean()), 4),
        "chrf_predicted_tag": round(float(sample.chrf_predicted_tag.mean()), 4),
        "diff": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
        "share_better": round(float((d > 0).mean()), 3),
        "predicted_english": {"n": int(english.sum()), "diff": round(float(d[english].mean()), 4),
                              "se": round(float(d[english].std(ddof=1) / np.sqrt(english.sum())), 4)},
        "predicted_other": {"n": int((~english).sum()), "diff": round(float(d[~english].mean()), 4),
                            "se": round(float(d[~english].std(ddof=1) / np.sqrt((~english).sum())), 4)},
    }
    sample[["lang", "nllb", "nllb_p", "chars", "chrf_config_tag", "chrf_predicted_tag", "text"]].to_csv(
        out / "lid_retag.csv", index=False)
    (out / "lid_retag.json").write_text(json.dumps(retag, indent=1))
    (out / "lid_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(json.dumps(retag, indent=1))


if __name__ == "__main__":
    main()
