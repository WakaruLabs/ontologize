"""Task naturalness: fluency + content preservation of steered decodes.

steerfid.py (and the steering-overlay experiment) log every base/steered
decode pair to steers.jsonl but score only effect and collateral. This
harness turns those pairs into a naturalness readout:

  automatic  chrF(base, steered)   content preservation (textfid.chrf;
                                   1 - the collateral steerfid reports)
             len_ratio             steered/base character length --
                                   degenerate repetition shows up here
             NLL_generic           mean per-token M2M100 NLL of each text
                                   under the CORPUS-MEAN embedding (the
                                   generic conditioning textfid.py already
                                   uses as its floor): a fluency proxy
                                   that needs no reference text. Reported
                                   for base and steered; the delta is how
                                   much less plausible the steered text is
                                   as decoder output.
  human      a blinded rating sheet: steered and unsteered decodes as
             sides A/B in seed-randomized order, rating columns left
             blank, with the unblinding key in a separate file.

Only the steers.jsonl is needed (base/steered are read from it); the
cache is read once for the corpus-mean embedding. --skip-nll drops the
decoder pass entirely (chrF/length/blind sheet are CPU-only, seconds).

  uv run python experiments/task-naturalness/naturalness.py \\
      --steers data/out/sonar/steerfid/<run>/steers.jsonl --device cuda
  uv run python experiments/task-naturalness/naturalness.py \\
      --steers experiments/steering-overlay/out/<run>/steers.jsonl --skip-nll

Writes <out>/naturalness.csv, <out>/rating.csv (blind), <out>/key.csv
(do not open before rating), <out>/summary.json.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import numpy as np

from textfid import chrf, ENCODER_ID, DECODER_ID


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steers", required=True, nargs="+",
                   help="steers.jsonl path(s) from steerfid.py or the "
                        "steering-overlay experiment")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy",
                   help="embedding cache; only the corpus-mean embedding "
                        "is read (for the generic-conditioning NLL)")
    p.add_argument("--skip-nll", action="store_true",
                   help="skip the M2M100 likelihood pass (CPU-only run)")
    p.add_argument("--device", default="cpu",
                   help="torch device for the NLL pass")
    p.add_argument("--b-decode", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=64,
                   help="token cap per text in the NLL pass")
    p.add_argument("--n-pairs", type=int, default=120,
                   help="pairs on the blind rating sheet (0 = all "
                        "non-identical pairs), sampled round-robin over "
                        "(kind, mag) strata")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default experiments/task-naturalness/out/<run name>")
    return p.parse_args()


def load_pairs(paths):
    """Rows from one or more steers.jsonl files. Random-control rows are
    kept (they calibrate what collateral-without-effect reads like)."""
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "src": Path(path).parent.name,
                    "kind": r.get("kind", "?"),
                    "lang": r.get("lang", ""),
                    "feature": r.get("feature", ""),
                    "mag": r.get("mag", float("nan")),
                    "sample": r.get("sample", -1),
                    "base": r["base"],
                    "steered": r["steered"],
                })
    return rows


def pick_blind(rows, n_pairs, seed):
    """Round-robin sample over (kind, mag) strata among non-identical
    pairs, so the sheet is not dominated by whichever arm has most rows."""
    idx = [i for i, r in enumerate(rows) if r["steered"] != r["base"]]
    rng = np.random.default_rng(seed)
    strata = {}
    for i in idx:
        strata.setdefault((rows[i]["kind"], rows[i]["mag"]), []).append(i)
    for pool in strata.values():
        rng.shuffle(pool)
    if not n_pairs or n_pairs >= len(idx):
        picked = idx
    else:
        picked = []
        keys = sorted(strata)
        while len(picked) < n_pairs and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(picked) < n_pairs:
                    picked.append(strata[k].pop())
    order = rng.permutation(len(picked))
    return [picked[i] for i in order], rng


def main():
    cfg = parse_args()
    rows = load_pairs(cfg.steers)
    assert rows, "no rows in the given steers.jsonl"
    run_name = Path(cfg.steers[0]).parent.name
    out = Path(cfg.out) if cfg.out else \
        ROOT / "experiments/task-naturalness/out" / run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{run_name}: {len(rows)} decode pairs from {len(cfg.steers)} file(s)")

    # ---- automatic readout: chrF + length ----
    for r in rows:
        r["chrf"] = chrf(r["base"], r["steered"])
        r["exact"] = float(r["base"] == r["steered"])
        r["len_ratio"] = len(r["steered"]) / max(len(r["base"]), 1)
        r["nll_base"] = r["nll_steered"] = float("nan")

    # ---- automatic readout: M2M100 NLL under generic conditioning ----
    nll_ok = not cfg.skip_nll
    if nll_ok:
        try:
            import torch as t
            from transformers import M2M100ForConditionalGeneration
            from transformers.modeling_outputs import BaseModelOutput
            from ontologize.data.pretrained import pretrained_transformer
        except ImportError as e:
            print(f"NLL pass unavailable ({e}); continuing without it")
            nll_ok = False
    if nll_ok:
        mm = np.load(cfg.cache, mmap_mode="r")
        x_mean = np.asarray(mm[:1 << 17], dtype=np.float32).mean(0)
        dev = t.device(cfg.device)
        # reference SONAR norm, exactly as textfid.py derives it
        pt_enc, tokenizer = pretrained_transformer(ENCODER_ID, "float32",
                                                   dev=dev)
        refs = tokenizer(
            ["The weather is nice today.",
             "She walked to the store to buy some bread.",
             "Scientists discovered a new species in the rainforest."],
            return_tensors="pt", padding=True).to(dev)
        with t.no_grad():
            h = pt_enc(**refs).last_hidden_state
            mask = refs["attention_mask"].unsqueeze(-1).float()
            ref_norm = t.norm((h * mask).sum(1) / mask.sum(1),
                              dim=-1).mean().item()
        del pt_enc
        dec = M2M100ForConditionalGeneration.from_pretrained(
            DECODER_ID).to(dev)
        dec.eval()
        ENG = tokenizer.convert_tokens_to_ids("eng_Latn")
        EOS = tokenizer.eos_token_id
        Y_mean = t.nn.functional.normalize(
            t.from_numpy(x_mean[None]).to(dev, t.float32), dim=-1) * ref_norm

        def label_seq(text):
            """Target labels in the layout textfid scores generated
            sequences in: [eng_Latn, tokens..., eos] (the decoder-start
            token is added by the model's internal shift)."""
            ids = tokenizer(text, add_special_tokens=False, truncation=True,
                            max_length=cfg.max_tokens)["input_ids"]
            return [ENG] + ids + [EOS]

        def nll_generic(texts):
            outs = np.empty(len(texts))
            with t.no_grad():
                for i in range(0, len(texts), cfg.b_decode):
                    batch = [label_seq(s) for s in texts[i:i + cfg.b_decode]]
                    width = max(len(s) for s in batch)
                    lab = t.full((len(batch), width), -100, dtype=t.long)
                    for j, s in enumerate(batch):
                        lab[j, :len(s)] = t.tensor(s)
                    labd = lab.to(dev)
                    logits = dec(
                        encoder_outputs=BaseModelOutput(
                            last_hidden_state=Y_mean.expand(
                                len(batch), -1).unsqueeze(1)),
                        labels=labd).logits
                    lp = t.log_softmax(logits, -1)
                    tok_lp = lp.gather(
                        -1, labd.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                    m = (labd != -100).float()
                    outs[i:i + len(batch)] = \
                        (-(tok_lp * m).sum(-1)
                         / m.sum(-1).clamp(min=1)).cpu().numpy()
            return outs

        print(f"NLL pass: {2 * len(rows)} texts on {cfg.device}")
        nb = nll_generic([r["base"] for r in rows])
        ns = nll_generic([r["steered"] for r in rows])
        for r, a, b in zip(rows, nb, ns):
            r["nll_base"], r["nll_steered"] = float(a), float(b)

    # ---- per-pair CSV ----
    cols = ["src", "kind", "lang", "feature", "mag", "sample", "chrf",
            "exact", "len_ratio", "nll_base", "nll_steered", "nll_delta"]
    with open(out / "naturalness.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["src"], r["kind"], r["lang"], r["feature"],
                        r["mag"], r["sample"], f"{r['chrf']:.4f}",
                        r["exact"], f"{r['len_ratio']:.3f}",
                        r["nll_base"], r["nll_steered"],
                        r["nll_steered"] - r["nll_base"]])

    # ---- blinded rating sheet ----
    picked, rng = pick_blind(rows, cfg.n_pairs, cfg.seed)
    with open(out / "rating.csv", "w", newline="") as fr, \
            open(out / "key.csv", "w", newline="") as fk:
        wr = csv.writer(fr)
        wk = csv.writer(fk)
        wr.writerow(["pair_id", "text_A", "text_B",
                     "fluency_A_1to5", "fluency_B_1to5",
                     "same_meaning_0to2", "notes"])
        wk.writerow(["pair_id", "steered_side", "src", "kind", "lang",
                     "feature", "mag", "sample", "chrf"])
        for pid, i in enumerate(picked):
            r = rows[i]
            steered_side = "A" if rng.random() < 0.5 else "B"
            a, b = ((r["steered"], r["base"]) if steered_side == "A"
                    else (r["base"], r["steered"]))
            wr.writerow([pid, a, b, "", "", "", ""])
            wk.writerow([pid, steered_side, r["src"], r["kind"], r["lang"],
                         r["feature"], r["mag"], r["sample"],
                         f"{r['chrf']:.4f}"])

    # ---- summary ----
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["mag"]), []).append(r)
    summary = {"run": run_name, "n_pairs": len(rows),
               "n_rated": len(picked), "nll": nll_ok, "by_kind_mag": []}
    print(f"\n{'kind':<8} {'mag':>5} {'n':>5} {'chrF':>7} {'exact':>6} "
          f"{'len_r':>6} {'NLL base':>9} {'NLL steer':>10} {'delta':>7}")
    for (kind, mag), g in sorted(groups.items(), key=lambda x: (
            str(x[0][0]), float(x[0][1]))):
        c = float(np.mean([r["chrf"] for r in g]))
        e = float(np.mean([r["exact"] for r in g]))
        lr = float(np.mean([r["len_ratio"] for r in g]))
        na = float(np.nanmean([r["nll_base"] for r in g]))
        ns_ = float(np.nanmean([r["nll_steered"] for r in g]))
        summary["by_kind_mag"].append(
            {"kind": kind, "mag": mag, "n": len(g), "chrf": c, "exact": e,
             "len_ratio": lr, "nll_base": na, "nll_steered": ns_})
        print(f"{kind:<8} {mag:>5.2f} {len(g):>5} {c:>7.3f} {e:>6.2f} "
              f"{lr:>6.2f} {na:>9.3f} {ns_:>10.3f} {ns_ - na:>+7.3f}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out / 'naturalness.csv'}")
    print(f"-> {out / 'rating.csv'} (blind; key in key.csv -- "
          "do not open the key before rating)")


if __name__ == "__main__":
    main()
