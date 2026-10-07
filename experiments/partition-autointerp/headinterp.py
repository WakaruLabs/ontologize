"""Head-level (partition) auto-interp: does describing HEADS instead of
latents recover the interpretability that per-latent detection misses?

The paper claims per-latent evaluation lenses systematically mis-measure
classifier codes because a head's content lives in the PARTITION it induces
over the corpus, not in any single entry's magnitude. autointerp.py scores
each latent (tag) as if it were an SAE feature: one description, binary
detection. This harness makes the head the unit instead:

  description  a head is described as ONE categorical dimension plus a
               label per entry value. Two modes, mirroring autointerp's
               params/acts split:
                 entries  each kept entry's parameter decode (generic-
                          origin code, entry forced one-hot, through the
                          model's own decoder + M2M100) is the value
                          label; an LLM names the shared dimension.
                 acts     the judge sees the kept entries' top-activating
                          texts side by side, grouped by entry, and must
                          name the dimension and label each group.
  detection    the blinded judge gets the dimension + value labels and a
               shuffled set of held-out snippets (positives from each
               entry's top list, distractors from the head's OTHER
               entries) and must assign each snippet to a value (or 0 =
               none). Scored per head: assignment accuracy, macro-F1 over
               values, binary F1 (any-value vs none; the column
               comparable to autointerp's per-latent F1), and distractor
               rejection. --null re-grades the same items against another
               head's same-mode description, exactly autointerp's floor.

Pipeline (stages resumable, artifacts in --out):

  1. harvest   activations over a cache slice -> per-entry top rows for a
               stratified sample of heads, per-head entry usage, distractor
               rows, and the entry decode embeddings. Needs the checkpoint;
               GPU JAX.
  2. texts     NOT reimplemented: point the existing autointerp.py texts
               stage at this harvest's heads.npz (it only reads top_i and
               neg_i, which this file provides):
                 uv run python autointerp.py texts \\
                     --features experiments/partition-autointerp/out/heads.npz \\
                     --out data/out/sonar/autointerp
  3. describe  writes descriptions.jsonl ({head, mode, dimension, values}).
               entries mode: torch M2M100 decode + 1 LLM call per head.
               acts mode: 1 LLM call per head.
  4. score     head-level assignment detection -> scores.csv; --null adds
               the donor-description floor.
  5. compare   joins scores.csv with an existing per-latent autointerp
               scores.csv (features -> head via f // k) -> comparison.csv,
               the direct test of the mis-measurement claim.

  uv run python experiments/partition-autointerp/headinterp.py harvest \\
      --ckpt data/out/sonar/multilingual/resid_nc
  uv run python experiments/partition-autointerp/headinterp.py describe --mode both
  uv run python experiments/partition-autointerp/headinterp.py score --null
  uv run python experiments/partition-autointerp/headinterp.py compare \\
      --latent-scores data/out/sonar/autointerp/onto/scores.csv
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import re
import sys
import numpy as np
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]  # repository root; root eval scripts live here
sys.path.insert(0, str(ROOT))

from autointerp import (merge_topk, make_llm, prf, snippet, null_pairing,
                        cap_pairing, load_texts, ENCODER_ID, DECODER_ID)

DIMENSION_PROMPT = """\
The following short texts are the decoded meanings of the ALTERNATIVE VALUES
of one categorical feature of a text encoder (each value excludes the
others):

{values}

Reply with ONE short English phrase naming the dimension along which these
values vary. No preamble."""

ACTS_PROMPT = """\
The text snippets below (possibly in different languages) are grouped by
which value of ONE categorical feature of a text encoder they most strongly
express. Snippets in different groups express different, mutually exclusive
values of the same underlying dimension.

{groups}

Name the single dimension along which the groups differ, then give a short
label for each group's value. Reply in exactly this format, nothing else:
DIMENSION: <one short phrase>
1: <label for group 1>
2: <label for group 2>
(continue for every group)"""

ASSIGN_PROMPT = """\
A categorical feature of a text encoder varies along the dimension:
"{dimension}"
Its possible values are:
{values}

Which value does each of the following numbered text snippets (possibly in
different languages) express?

{samples}

Reply with ONLY a JSON object mapping each snippet number to a value number,
using 0 when the snippet expresses none of the listed values, e.g.
{{"1": 2, "2": 0, "3": 1}}."""


# ---------- pure helpers ----------

def pick_heads(l, h, n, seed):
    """Stratified head sample: n heads spread evenly over layers, random
    within a layer. Returns global head ids (layer * h + head)."""
    rng = np.random.default_rng(seed)
    per = [n // l + (1 if i < n % l else 0) for i in range(l)]
    out = []
    for li in range(l):
        pick = rng.choice(h, min(per[li], h), replace=False)
        out += [li * h + int(x) for x in np.sort(pick)]
    return np.array(out, np.int64)


def head_distractors(sub_rows, sub_acts, head, k, kept, excl, n_neg, rng):
    """Rows whose argmax entry for this head is NOT one of the kept
    (described) entries: in-partition negatives, so rejecting them requires
    knowing the described values, not just the head's broad topic."""
    cols = sub_acts[:, head * k:(head + 1) * k].astype(np.float32)
    am = cols.argmax(1)
    pool = np.setdiff1d(sub_rows[~np.isin(am, kept)], excl)
    if not len(pool):
        pool = np.setdiff1d(sub_rows, excl)
    return rng.choice(pool, n_neg, replace=len(pool) < n_neg)


def head_items(top_e, neg_rows, n_desc, n_test, n_dis, seed, head):
    """Blind-detection items for one head. top_e: (E, n_top) per kept
    entry. Positives are held-out ranks n_desc..n_desc+n_test per entry
    (disjoint from the ranks the description saw), truth = 1-based entry
    position; distractors get truth 0. Shuffle is seeded per head so a
    resume rebuilds the same items."""
    rows, truth = [], []
    for e in range(top_e.shape[0]):
        pos = top_e[e, n_desc:n_desc + n_test]
        rows += [int(r) for r in pos]
        truth += [e + 1] * len(pos)
    n_dis = min(n_dis, len(neg_rows))
    rows += [int(r) for r in neg_rows[:n_dis]]
    truth += [0] * n_dis
    order = np.random.default_rng(seed + head).permutation(len(rows))
    return [rows[o] for o in order], [truth[o] for o in order]


def parse_labeling(ans, n_groups):
    """Parse the acts-mode describe reply: (dimension, [labels] or None)."""
    m = re.search(r"DIMENSION\s*[:\-]\s*(.+)", ans, re.I)
    dim = m.group(1).strip() if m else ""
    vals = {}
    for num, lab in re.findall(r"^\s*(\d{1,2})\s*[:.)]\s*(.+?)\s*$", ans, re.M):
        i = int(num)
        if 1 <= i <= n_groups:
            vals.setdefault(i, lab.strip())
    if len(vals) != n_groups:
        return dim, None
    return dim, [vals[i + 1] for i in range(n_groups)]


def parse_assignment(ans, n_items, n_values):
    """Parse the score reply into {item -> value}; omitted items count as
    0 (none) so a lazy judge is penalized, not silently skipped."""
    m = re.search(r"\{.*\}", ans, re.S)
    body = m.group(0) if m else ans
    pred = {}
    for a, b in re.findall(r'"?(\d{1,3})"?\s*[:=]\s*"?(\d{1,3})"?', body):
        i, v = int(a), int(b)
        if 1 <= i <= n_items and 0 <= v <= n_values:
            pred.setdefault(i, v)
    return {i: pred.get(i, 0) for i in range(1, n_items + 1)}


def assignment_metrics(pred, truth, n_values):
    """(accuracy over positives, macro-F1 over values, binary F1
    any-value-vs-none, distractor rejection rate)."""
    pos = [i for i, t in enumerate(truth, 1) if t > 0]
    neg = [i for i, t in enumerate(truth, 1) if t == 0]
    acc = float(np.mean([pred[i] == truth[i - 1] for i in pos])) if pos else 0.0
    rej = float(np.mean([pred[i] == 0 for i in neg])) if neg else float("nan")
    f1s = []
    for v in range(1, n_values + 1):
        p_v = {i for i in pred if pred[i] == v}
        t_v = {i for i, t in enumerate(truth, 1) if t == v}
        if t_v:
            f1s.append(prf(p_v, t_v)[2])
    macro = float(np.mean(f1s)) if f1s else 0.0
    bin_f1 = prf({i for i in pred if pred[i] > 0}, set(pos))[2]
    return acc, macro, bin_f1, rej


def desc_text(rec):
    """Canonical text of a head description (used for the duplicate-donor
    check in null pairing and for the score prompt)."""
    return rec["dimension"] + " | " + " / ".join(rec["values"])


SCORE_COLS = ["head", "mode", "kind", "n_items", "acc", "macro_f1",
              "bin_f1", "dis_rej"]


def read_scores(path):
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def spearman(a, b):
    """Spearman rank correlation without scipy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------- stage: harvest ----------

def harvest(cfg):
    import jax.numpy as jnp
    from tqdm import tqdm
    from autointerp import onto_acts_fn

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    acts_fn, embed_fn, F, meta = onto_acts_fn(cfg.ckpt, cfg.step,
                                              cfg.temperature)
    l, h, k = meta["l"], meta["h"], meta["k"]

    mm = np.load(cfg.cache, mmap_mode="r")
    n = min(cfg.rows, mm.shape[0])
    b = cfg.b
    top_v = np.full((F, cfg.n_top), -np.inf, np.float32)
    top_i = np.zeros((F, cfg.n_top), np.int64)
    sub_rows, sub_acts = [], []
    stride = max(1, (n // b) // cfg.sub_batches)
    for bi, i in enumerate(tqdm(range(0, n - b + 1, b), desc="harvest")):
        A = np.asarray(acts_fn(jnp.asarray(
            np.asarray(mm[i:i + b], dtype=np.float32))))
        top_v, top_i = merge_topk(
            top_v, top_i, A.T,
            np.broadcast_to(np.arange(i, i + b), (F, b)), cfg.n_top)
        if bi % stride == 0:
            sub_rows.append(np.arange(i, i + b))
            sub_acts.append(A.astype(np.float16))
    sub_rows = np.concatenate(sub_rows)
    sub_acts = np.concatenate(sub_acts)

    heads = pick_heads(l, h, cfg.n_heads, cfg.seed)
    E = min(cfg.entries_per_head, k)
    rng = np.random.default_rng(cfg.seed)
    usage = np.zeros((len(heads), k), np.float32)
    entries = np.zeros((len(heads), E), np.int64)
    h_top_i = np.zeros((len(heads), E, cfg.n_top), np.int64)
    h_top_v = np.zeros((len(heads), E, cfg.n_top), np.float32)
    neg_i = np.zeros((len(heads), cfg.n_neg), np.int64)
    origin = sub_acts.astype(np.float32).mean(0).reshape(l, h, k)
    origin /= origin.sum(-1, keepdims=True)
    codes = np.broadcast_to(origin, (len(heads) * E, l, h, k)).copy()
    for j, g in enumerate(heads):
        cols = sub_acts[:, g * k:(g + 1) * k].astype(np.float32)
        usage[j] = cols.mean(0)
        kept = np.argsort(-usage[j])[:E]  # most-used entries get described
        entries[j] = kept
        h_top_i[j] = top_i[g * k + kept]
        h_top_v[j] = top_v[g * k + kept]
        excl = h_top_i[j].ravel()
        neg_i[j] = head_distractors(sub_rows, sub_acts, g, k, kept, excl,
                                    cfg.n_neg, rng)
        li, hi = g // h, g % h
        for e, ki in enumerate(kept):
            codes[j * E + e, li, hi] = np.eye(k, dtype=np.float32)[ki]
    emb = embed_fn(codes).reshape(len(heads), E, -1)

    np.savez(out / "heads.npz", heads=heads, entries=entries, usage=usage,
             top_i=h_top_i, top_v=h_top_v, neg_i=neg_i, emb=emb)
    (out / "meta.json").write_text(json.dumps(
        {**meta, "ckpt": str(cfg.ckpt), "rows": n, "n_top": cfg.n_top,
         "n_desc": cfg.n_desc, "n_test": cfg.n_test, "n_neg": cfg.n_neg,
         "entries_per_head": E, "seed": cfg.seed}, indent=2))
    print(f"harvest: {len(heads)} heads x {E} entries -> {out/'heads.npz'}")


# ---------- stage: describe ----------

def decode_values(cfg, emb):
    """Decode entry embeddings (N, d) to text through M2M100, exactly as
    autointerp.describe_params does (forced eng_Latn, SONAR_NORM rescale)."""
    import torch as t
    from transformers import AutoTokenizer, M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from textfid import SONAR_NORM

    dev = t.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
    ref_norm = SONAR_NORM
    dec = M2M100ForConditionalGeneration.from_pretrained(DECODER_ID).to(dev)
    dec.eval()
    eng = tokenizer.convert_tokens_to_ids("eng_Latn")
    texts = []
    with t.no_grad():
        for i in range(0, len(emb), cfg.b_decode):
            Y = t.from_numpy(np.asarray(emb[i:i + cfg.b_decode],
                                        np.float32)).to(dev)
            Y = t.nn.functional.normalize(Y, dim=-1) * ref_norm
            gen = dec.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=Y.unsqueeze(1)),
                forced_bos_token_id=eng, max_length=48, num_beams=1,
                repetition_penalty=1.2)
            texts += [tokenizer.decode(g, skip_special_tokens=True).strip()
                      for g in gen]
    return texts


def describe_entries(cfg, dat, meta, done):
    todo = [j for j, g in enumerate(dat["heads"])
            if (int(g), "entries") not in done]
    if not todo:
        return
    E = dat["emb"].shape[1]
    vals = decode_values(cfg, dat["emb"][todo].reshape(len(todo) * E, -1))
    vals = [vals[i * E:(i + 1) * E] for i in range(len(todo))]
    llm_map = make_llm(cfg.judge, cfg.dry_dir(), cfg.jobs, cfg.rate)
    items = [(DIMENSION_PROMPT.format(values="\n".join(
        f"{e+1}. {v}" for e, v in enumerate(vs))),
        f"dim_{int(dat['heads'][j])}") for j, vs in zip(todo, vals)]
    for j, vs, dim in zip(todo, vals, llm_map(items)):
        if cfg.dry_dir():
            continue
        yield {"head": int(dat["heads"][j]), "mode": "entries",
               "dimension": dim.strip(), "values": vs}


def describe_acts(cfg, dat, meta, done):
    txt = load_texts(cfg.texts)
    llm_map = make_llm(cfg.judge, cfg.dry_dir(), cfg.jobs, cfg.rate)
    items, todo = [], []
    for j, g in enumerate(dat["heads"]):
        if (int(g), "acts") in done:
            continue
        groups = []
        for e in range(dat["top_i"].shape[1]):
            rows = dat["top_i"][j, e, :meta["n_desc"]]
            body = "\n".join(f"- {snippet(txt[int(r)]['text'])}" for r in rows)
            groups.append(f"Group {e + 1}:\n{body}")
        items.append((ACTS_PROMPT.format(groups="\n\n".join(groups)),
                      f"acts_{int(g)}"))
        todo.append((j, int(g)))
    n_groups = dat["top_i"].shape[1]
    for (j, g), ans in zip(todo, llm_map(items)):
        if not ans:
            continue
        dim, vals = parse_labeling(ans, n_groups)
        if vals is None:
            print(f"describe[acts]: unparseable labels for head {g}; "
                  "skipping (resume retries)")
            continue
        yield {"head": g, "mode": "acts", "dimension": dim, "values": vals}


def describe(cfg):
    d = Path(cfg.out)
    dat = np.load(d / "heads.npz")
    meta = json.loads((d / "meta.json").read_text())
    out = d / "descriptions.jsonl"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["head"], r["mode"]) for r in map(json.loads, f)}

    def flush(recs):
        n = 0
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        return n

    n = 0
    if cfg.mode in ("entries", "both"):
        n += flush(describe_entries(cfg, dat, meta, done))
    if cfg.mode in ("acts", "both"):
        n += flush(describe_acts(cfg, dat, meta, done))
    print(f"describe: +{n} -> {out}")


# ---------- stage: score ----------

def score(cfg):
    d = Path(cfg.out)
    dat = np.load(d / "heads.npz")
    meta = json.loads((d / "meta.json").read_text())
    txt = load_texts(cfg.texts)
    with open(d / "descriptions.jsonl") as f:
        descs = [json.loads(line) for line in f]
    llm_map = make_llm(cfg.judge, cfg.dry_dir(), cfg.jobs, cfg.rate)
    pos_of = {int(g): j for j, g in enumerate(dat["heads"])}
    E = dat["top_i"].shape[1]

    path = d / "scores.csv"
    done = {(int(r["head"]), r["mode"], r["kind"]) for r in read_scores(path)}

    nulls = {}
    if cfg.null:
        by_mode = {}
        for rec in descs:
            by_mode.setdefault(rec["mode"], {})[rec["head"]] = rec
        for mode, m in by_mode.items():
            texts = {g: desc_text(r) for g, r in m.items()}
            pair = null_pairing(sorted(m), meta["seed"], texts=texts)
            pair = cap_pairing(pair, cfg.null_n, meta["seed"])
            nulls[mode] = {g: m[donor] for g, donor in pair.items()}

    items, keys, truths = [], [], []
    for rec in descs:
        g = rec["head"]
        j = pos_of[g]
        rows, truth = head_items(dat["top_i"][j], dat["neg_i"][j],
                                 meta["n_desc"], cfg.n_test, cfg.n_dis,
                                 meta["seed"], g)
        samples = "\n".join(f"{i+1}. {snippet(txt[int(r)]['text'])}"
                            for i, r in enumerate(rows))
        for kind in ("self", "null"):
            src = rec if kind == "self" else nulls.get(rec["mode"], {}).get(g)
            if src is None or (g, rec["mode"], kind) in done:
                continue
            values = "\n".join(f"{e+1}. {v}"
                               for e, v in enumerate(src["values"]))
            items.append((ASSIGN_PROMPT.format(
                dimension=src["dimension"] or "(unnamed)", values=values,
                samples=samples), f"assign_{rec['mode']}_{kind}_{g}"))
            keys.append((g, rec["mode"], kind))
            truths.append(truth)

    replies = llm_map(items)
    if cfg.dry_dir():
        for _ in replies:
            pass
        print(f"score: dry run, {len(items)} prompts written")
        return
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SCORE_COLS)
        for (g, mode, kind), truth, ans in zip(keys, truths, replies):
            if not ans:
                continue
            pred = parse_assignment(ans, len(truth), E)
            acc, macro, bin_f1, rej = assignment_metrics(pred, truth, E)
            w.writerow([g, mode, kind, len(truth), f"{acc:.3f}",
                        f"{macro:.3f}", f"{bin_f1:.3f}", f"{rej:.3f}"])
    summarize(path, E)


def summarize(path, n_values):
    rows = read_scores(path)
    by = {}
    for r in rows:
        by.setdefault(r["mode"], {}).setdefault(r["kind"], []).append(
            (float(r["acc"]), float(r["macro_f1"]), float(r["bin_f1"])))
    chance = 1.0 / (n_values + 1)
    for mode, kinds in sorted(by.items()):
        s = np.array(kinds.get("self", []))
        nul = np.array(kinds.get("null", []))
        if not len(s):
            continue
        line = (f"score[{mode}]: acc {s[:, 0].mean():.3f} macro_f1 "
                f"{s[:, 1].mean():.3f} bin_f1 {s[:, 2].mean():.3f} "
                f"(n={len(s)}, guess-one-value acc ~{chance:.3f})")
        if len(nul):
            line += (f" | null acc {nul[:, 0].mean():.3f} "
                     f"| acc delta {s[:, 0].mean() - nul[:, 0].mean():+.3f}")
        print(f"{line} -> {path}")


# ---------- stage: compare ----------

def compare(cfg):
    d = Path(cfg.out)
    meta = json.loads((d / "meta.json").read_text())
    k = meta["k"]
    ours = read_scores(d / "scores.csv")
    if not ours:
        raise SystemExit(f"no {d/'scores.csv'}; run score first")
    with open(cfg.latent_scores) as fh:
        lat = [r for r in csv.DictReader(fh)
               if r["mode"] == cfg.latent_mode]
    lat_by_head = {}
    for r in lat:
        g = int(r["feature"]) // k
        lat_by_head.setdefault(g, {}).setdefault(
            r.get("kind") or "self", []).append(float(r["f1"]))

    rows = []
    for mode in sorted({r["mode"] for r in ours}):
        by_head = {}
        for r in ours:
            if r["mode"] == mode:
                by_head.setdefault(int(r["head"]), {})[r["kind"]] = r
        for g, kinds in sorted(by_head.items()):
            s = kinds.get("self")
            if s is None:
                continue
            nul = kinds.get("null")
            lat_self = lat_by_head.get(g, {}).get("self", [])
            lat_null = lat_by_head.get(g, {}).get("null", [])
            rows.append({
                "head": g, "mode": mode,
                "head_acc": float(s["acc"]),
                "head_bin_f1": float(s["bin_f1"]),
                "head_acc_null": float(nul["acc"]) if nul else float("nan"),
                "head_delta": float(s["acc"]) - float(nul["acc"])
                              if nul else float("nan"),
                "latent_f1_mean": float(np.mean(lat_self))
                                  if lat_self else float("nan"),
                "latent_f1_null": float(np.mean(lat_null))
                                  if lat_null else float("nan"),
                "n_latents_scored": len(lat_self)})
    out = d / "comparison.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    both = [r for r in rows if r["n_latents_scored"]
            and not np.isnan(r["head_acc"])]
    if len(both) >= 4:
        rho = spearman([r["head_acc"] for r in both],
                       [r["latent_f1_mean"] for r in both])
        print(f"compare: {len(both)} heads with both lenses | spearman("
              f"head acc, mean latent F1) = {rho:+.3f}")
    print(f"-> {out}  (latent coverage is the autointerp campaign's "
          "stratified sample; heads with 0 scored latents carry NaN)")


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["harvest", "describe", "score",
                                     "compare"])
    p.add_argument("--out", default=str(EXP / "out"))
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc",
                   help="Ontologizer checkpoint dir (resid_nc matches the "
                        "existing autointerp campaign; resid_nc_hm is the "
                        "other live run)")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=262144,
                   help="cache rows to harvest activations over")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--sub-batches", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=20,
                   help="heads sampled, stratified over layers")
    p.add_argument("--entries-per-head", type=int, default=8,
                   help="most-used entries described per head (k=32 whole "
                        "heads make the judge prompt unwieldy)")
    p.add_argument("--n-top", type=int, default=12,
                   help="top rows kept per entry; >= n_desc + n_test")
    p.add_argument("--n-desc", type=int, default=6,
                   help="top texts per entry used to WRITE the description")
    p.add_argument("--n-test", type=int, default=3,
                   help="held-out positives per entry used to SCORE it")
    p.add_argument("--n-neg", type=int, default=12,
                   help="distractor rows harvested per head")
    p.add_argument("--n-dis", type=int, default=6,
                   help="distractors shown per score prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--texts", default="data/out/sonar/autointerp",
                   help="dir containing texts.jsonl (autointerp.py texts "
                        "output; reuse the existing campaign dir)")
    p.add_argument("--mode", choices=["entries", "acts", "both"],
                   default="both")
    p.add_argument("--device", default="cpu",
                   help="torch device for the M2M100 decode")
    p.add_argument("--b-decode", type=int, default=16)
    p.add_argument("--judge", default="haiku")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rate", type=int, default=0)
    p.add_argument("--null", action="store_true",
                   help="also grade against another head's description")
    p.add_argument("--null-n", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--latent-scores",
                   default="data/out/sonar/autointerp/onto/scores.csv",
                   help="per-latent autointerp scores.csv (compare stage)")
    p.add_argument("--latent-mode", default="cacts",
                   help="which per-latent description mode to compare "
                        "against")
    cfg = p.parse_args()

    if cfg.n_top < cfg.n_desc + cfg.n_test:
        raise SystemExit("--n-top must be >= --n-desc + --n-test")

    def dry_dir():
        if not cfg.dry_run:
            return None
        dd = Path(cfg.out) / "prompts"
        dd.mkdir(parents=True, exist_ok=True)
        return dd
    cfg.dry_dir = dry_dir

    {"harvest": harvest, "describe": describe,
     "score": score, "compare": compare}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
