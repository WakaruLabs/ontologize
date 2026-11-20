"""Force one entry of one head over held-out rows and read what the decodes
become: how much of each decode is in a target script, and how much of the
source row's meaning survives, beside every other entry of the same head.

For each of the head's k entries j (the target and its k - 1 siblings,
which are the null), every row is run through `withArgs` with head
(--layer, --head) set to entry j, and the output is decoded through M2M100
(`textfid.SonarDecoder`: English forced, greedy, rescaled to SONAR's
norm). Two references are decoded too: the input row itself (`x`) and the
model's unforced reconstruction (`recon`). Per decode:

  share   the fraction of its letters in --script (Unicode blocks below)
  cos     cosine between the decode's SONAR re-encoding (under the English
          tag, as the cache was encoded) and the source row
  cos0    the same against another row (a seeded derangement): what any
          decode shares with an unrelated row

The rows are drawn from the cache's held-out tail, excluding --exclude-lang
(the target script's own language), so a high share means the entry moved
text into the script. The script also records which tail rows choose the
entry on their own, per language (usage.csv), from the model's argmax.

  JAX_PLATFORMS=cpu uv run python experiments/entry-force/force.py \\
      --model data/out/sonar/multilingual/resid_nc --layer 1 --head 8 \\
      --entry 17 --device cuda

Outputs under --out: decodes.jsonl (one line per row and condition),
summary.csv (per condition), usage.csv, summary.json (the target entry's
values and its rank among the head's entries).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Unicode blocks per script, and the language token M2M100 starts its
# decodes in that script with
SCRIPTS = {
    "mya": [(0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)],
}
LANG_TOKENS = {"mya": "mya_Mymr"}


def script_share(text: str, ranges) -> float:
    """Fraction of the letters in `text` that fall in `ranges` (0 for a
    text without letters). Myanmar vowel signs are combining marks, not
    letters, so only its consonants and independent vowels count."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hit = sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters)
    return hit / len(letters)


def derangement(n: int, rng) -> np.ndarray:
    """A random permutation with no fixed point (n >= 2)."""
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def rank_of(values: np.ndarray, j: int) -> int:
    """1-based rank of entry j among `values`, largest first (ties count
    against j: rank 1 means strictly above every other entry)."""
    return int((values >= values[j]).sum())


def pick_rows(langs: np.ndarray, start: int, n: int, exclude: str,
              seed: int, only: str = None) -> np.ndarray:
    """`n` tail rows (absolute indices >= start), seeded, none in
    `exclude`, and all in `only` when it is given."""
    idx = np.arange(start, len(langs))
    keep = langs[start:] != exclude
    if only is not None:
        keep &= langs[start:] == only
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx[keep], n, replace=False))


def load_rows(mm, idx, unit: bool) -> np.ndarray:
    """Cache rows as float32, scaled to unit norm when `unit`: a model
    trained on mc4_4M (unit rows) reading SONAR's own embeddings
    (mc4_sent16) needs them normalized; a model trained on mc4_sent16 reads
    them as they are."""
    X = np.asarray(mm[idx], np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True) if unit else X


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Ontologizer run directory")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--entry", type=int, required=True,
                   help="the entry under test; the head's others are the null")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="held-out tail rows of the cache")
    p.add_argument("--rows", type=int, default=256,
                   help="tail rows forced and decoded per entry")
    p.add_argument("--script", default="mya", choices=sorted(SCRIPTS))
    p.add_argument("--exclude-lang", default="my",
                   help="mC4 code of the script's own language, left out")
    p.add_argument("--lang", default=None,
                   help="draw the rows from this mC4 language only")
    p.add_argument("--own-rows", type=int, default=64,
                   help="rows of --exclude-lang decoded as the positive control")
    p.add_argument("--controls-only", action="store_true",
                   help="write usage.csv and controls.json, then stop")
    p.add_argument("--unit", action="store_true",
                   help="scale rows to unit norm (a model trained on mc4_4M "
                        "reading mc4_sent16)")
    p.add_argument("--free-lang", action="store_true",
                   help="let the decoder choose its output language rather "
                        "than forcing English")
    p.add_argument("--device", default="cpu", help="torch device (M2M100)")
    p.add_argument("--enc-device", default=None,
                   help="torch device for the re-encoding (default --device)")
    p.add_argument("--b-decode", type=int, default=64)
    p.add_argument("--b", type=int, default=1024, help="JAX batch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default data/out/sonar/entry_force/<run>_<step>_L<l>H<h>")
    return p.parse_args()


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import jax.numpy as jnp
    import torch as t
    from pareto import load_onto
    from steerfid import load_steerable
    from textfid import SonarDecoder
    from ontologize.ontologizer import Ontologizer, DictIntervention
    from ontologize.data.loaders import TokenizeTransform
    from ontologize.data.pretrained import pretrained_transformer

    cfg = parse_args()
    model, params, step = load_onto(cfg.model, cfg.step)
    l, k = model.l, model.k
    name = f"{Path(cfg.model).name}_{step}_L{cfg.layer}H{cfg.head}"
    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/entry_force") / name
    out.mkdir(parents=True, exist_ok=True)
    T = cfg.temperature

    mm = np.load(cfg.cache, mmap_mode="r")
    langs = np.load(Path(cfg.cache).with_suffix(".langs.npy"), mmap_mode="r")
    start = len(mm) - cfg.eval_rows

    def run(X, arglist):
        Y, _, _, _ = model.apply({"params": params}, jnp.asarray(X), arglist,
                                 temperature=T, method=Ontologizer.withArgs)
        return np.asarray(Y)

    # every layer's assignment probabilities, as the forward computes them
    acts, _, _, _, _ = load_steerable(
        cfg.model, argparse.Namespace(mode="dec", step=step, temperature=T))

    def assign(X):
        """The head's entry per row, as the model's forward picks it."""
        P = np.asarray(acts(jnp.asarray(X))).reshape(len(X), l, model.h, k)
        return P[:, cfg.layer, cfg.head].argmax(-1)

    # natural usage: which tail rows choose the entry, per language
    tail_lang = np.asarray(langs[start:])
    choice = np.concatenate([
        assign(load_rows(mm, slice(i, min(i + cfg.b, len(mm))), cfg.unit))
        for i in range(start, len(mm), cfg.b)])
    with open(out / "usage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lang", "rows", "chose_entry", "share"])
        for lg in np.unique(tail_lang):
            m = tail_lang == lg
            w.writerow([lg, int(m.sum()), int((choice[m] == cfg.entry).sum()),
                        f"{(choice[m] == cfg.entry).mean():.4f}"])
    usage = {"own_lang": float((choice[tail_lang == cfg.exclude_lang]
                                == cfg.entry).mean()),
             "other_langs": float((choice[tail_lang != cfg.exclude_lang]
                                   == cfg.entry).mean()),
             "uniform": 1.0 / k}

    rows = pick_rows(np.asarray(langs), start, cfg.rows, cfg.exclude_lang,
                     cfg.seed, cfg.lang)
    X = load_rows(mm, rows, cfg.unit)
    perm = derangement(len(rows), np.random.default_rng(cfg.seed + 1))

    dec = SonarDecoder(cfg.device, cfg.b_decode)
    ranges = SCRIPTS[cfg.script]
    lang_tok = dec.tokenizer.convert_tokens_to_ids(LANG_TOKENS[cfg.script])

    def decode(Y):
        """(texts, the language token each decode starts with). English is
        forced, as in `SonarDecoder`, unless --free-lang lets the decoder
        choose the language from the embedding itself."""
        if not cfg.free_lang:
            seqs = dec.generate(Y)
        else:
            seqs = []
            with t.no_grad():
                for i in range(0, len(Y), dec.b):
                    seqs += list(dec.dec.generate(
                        encoder_outputs=dec._enc(Y[i:i + dec.b]),
                        max_length=dec.max_length, num_beams=1,
                        repetition_penalty=1.2).cpu())
        return [dec.text(s) for s in seqs], np.array([int(s[1]) for s in seqs])

    # positive control: the script's own language's rows through the same
    # decoder, input and reconstruction. If these come out without the
    # script, the decoder cannot show it and a zero share means nothing
    own = pick_rows(np.asarray(langs), start, min(cfg.rows, cfg.own_rows),
                    "", cfg.seed, cfg.exclude_lang)
    X_own = load_rows(mm, own, cfg.unit)
    controls = {"lang": cfg.exclude_lang, "rows": len(own)}
    with open(out / "controls.jsonl", "w") as fh:
        for c, Y in (("own_x", X_own),
                     ("own_recon", run(X_own, [DictIntervention()
                                               for _ in range(l)]))):
            texts, first = decode(Y)
            share = np.array([script_share(s, ranges) for s in texts])
            controls[c] = {"share": float(share.mean()),
                           "majority": float((share >= 0.5).mean()),
                           "lang_tok": float((first == lang_tok).mean())}
            for r, s, sh in zip(own, texts, share):
                fh.write(json.dumps({"row": int(r), "cond": c, "text": s,
                                     "share": round(float(sh), 4)},
                                    ensure_ascii=False) + "\n")
            print(f"{c:9s} share {share.mean():.3f}  majority "
                  f"{(share >= 0.5).mean():.3f}  lang_tok "
                  f"{(first == lang_tok).mean():.3f}", flush=True)
    (out / "controls.json").write_text(json.dumps(
        {"usage": usage, "controls": controls}, indent=1))
    if cfg.controls_only:
        return

    conds = {"x": X, "recon": run(X, [DictIntervention() for _ in range(l)])}
    for j in range(k):
        args = [DictIntervention() for _ in range(l)]
        args[cfg.layer] = DictIntervention(h_set=jnp.array([cfg.head]),
                                           k_set=jnp.array([j]))
        conds[f"force{j}"] = run(X, args)

    enc_dev = t.device(cfg.enc_device or cfg.device)
    pt_enc, tokenizer = pretrained_transformer(
        "cointegrated/SONAR_200_text_encoder", "float32", dev=enc_dev)
    tok = TokenizeTransform(tokenizer, maxlen=512)

    def reencode(texts):
        """SONAR's embedding (the masked mean `pretrained.l2_pooling`
        takes), pooled in torch: this process's JAX may lack the GPU."""
        E = []
        with t.no_grad():
            for i in range(0, len(texts), cfg.b_decode):
                batch = [tok.map({"text": s, "lang": "en"})
                         for s in texts[i:i + cfg.b_decode]]
                ids, mask = (t.as_tensor(np.stack([b[f] for b in batch])).to(enc_dev)
                             for f in ("input_ids", "attention_mask"))
                H = pt_enc(input_ids=ids, attention_mask=mask).last_hidden_state
                m = mask.unsqueeze(-1).to(H.dtype)
                E.append(((H * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().numpy())
        return np.concatenate(E)

    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    summary, log = [], open(out / "decodes.jsonl", "w")
    for c, Y in conds.items():
        texts, first = decode(Y)
        E = reencode(texts)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        share = np.array([script_share(s, ranges) for s in texts])
        cos = (E * Xn).sum(1)
        cos0 = (E * Xn[perm]).sum(1)
        for r, lg, s, sh, a, b, f0 in zip(rows, langs[rows], texts, share, cos,
                                          cos0, first):
            log.write(json.dumps({"row": int(r), "lang": str(lg), "cond": c,
                                  "text": s, "share": round(float(sh), 4),
                                  "cos": round(float(a), 4),
                                  "cos0": round(float(b), 4),
                                  "lang_tok": dec.tokenizer.convert_ids_to_tokens(int(f0))},
                                 ensure_ascii=False) + "\n")
        summary.append({"cond": c, "share": share.mean(),
                        "majority": (share >= 0.5).mean(),
                        "lang_tok": (first == lang_tok).mean(),
                        "cos": cos.mean(), "cos0": cos0.mean()})
        print(f"{c:8s} share {share.mean():.3f}  majority "
              f"{(share >= 0.5).mean():.3f}  lang_tok "
              f"{(first == lang_tok).mean():.3f}  cos {cos.mean():.3f}  "
              f"cos0 {cos0.mean():.3f}", flush=True)
    log.close()
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows([{f: (f"{v:.4f}" if isinstance(v, float) else v)
                      for f, v in r.items()} for r in summary])

    forced = [r for r in summary if r["cond"].startswith("force")]
    res = {}
    for f in ("share", "majority", "lang_tok", "cos"):
        v = np.array([r[f] for r in forced])
        others = np.delete(v, cfg.entry)
        res[f] = {"entry": float(v[cfg.entry]), "rank": rank_of(v, cfg.entry),
                  "others_max": float(others.max()),
                  "others_median": float(np.median(others))}
    ref = {r["cond"]: {f: float(r[f]) for f in ("share", "majority", "lang_tok",
                                                 "cos", "cos0")}
           for r in summary if not r["cond"].startswith("force")}
    meta = {"model": cfg.model, "step": int(step), "temperature": T,
            "layer": cfg.layer, "head": cfg.head, "entry": cfg.entry, "k": k,
            "rows": len(rows), "exclude_lang": cfg.exclude_lang,
            "lang": cfg.lang, "cache": cfg.cache, "unit": cfg.unit,
            "free_lang": cfg.free_lang, "controls": controls,
            "script": cfg.script, "seed": cfg.seed, "usage": usage,
            "target": res, "references": ref,
            "cos0_forced_mean": float(np.mean([r["cos0"] for r in forced]))}
    (out / "summary.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({"usage": usage, "target": res}, indent=1))


if __name__ == "__main__":
    main()
