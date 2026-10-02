# Systematic ontofeature annotation pipeline for a trained Ontologizer.
#
# Stages (cached in the output dir, resumable):
#   census    sample a held-out multilingual corpus, classify every
#             (layer, head, entry) as dead / constant / rare / live
#   evidence  per live entry, build a JSON evidence card:
#             max-activating examples, rev archetype decode, steering
#             fingerprint (Ontologizer.intervene), ablation fingerprint,
#             decoder-space neighbors / duplicate families
#   label     LLM annotator (claude -p) turns each card into a structured
#             label: {label, description, type, not_determined, confidence}
#   score     validate labels: detection (judge picks activating sentences),
#             generation (LLM writes should/shouldn't-fire sentences, checked
#             through the real encoder+model), causal consistency (judge
#             rates steering diffs against the label)
#
# Usage:
#   uv run python scratch/annotate.py <ckpt_dir> <step> --stage all
#   uv run python scratch/annotate.py data/out/sonar/multilingual 85000 \
#       --stage evidence --layers 1 --limit 12
#
# Outputs in scratch/annotate_<ckpt>_<step>/:
#   corpus.npz census.json cards.jsonl annotations.jsonl scores.jsonl report.md

import os
import sys

if "CUDNN_INJECTED" not in os.environ:
    try:
        import nvidia.cudnn
        cudnn_lib = list(nvidia.cudnn.__path__)[0] + "/lib"
        old_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{old_path}"
        os.environ["CUDNN_INJECTED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except ImportError:
        pass

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
jax.random.PRNGKey(0)

import numpy as np
import torch as t
import torch.nn.functional as tF
import orbax.checkpoint as ocp

from ontologize.ontologizer import Ontologizer

# ---------------------------------------------------------------- config

TEMP = 0.1
N_CORPUS = 4096
BATCH_ENC = 32
BATCH_ONTO = 512
MAXLEN = 512
N_MAXACT = 8            # max-activating examples per card
STEER_STRENGTH = 0.15
FAMILY_COS = 0.97       # duplicate-family threshold in decoder space
MIN_LIVE_WINS = 5       # fewer wins than this -> status "rare"
CONSTANT_SHARE = 0.98   # win share above this -> status "constant"
REFINE_STEPS = 60
REFINE_LR = 0.05

PANEL = [
    "The weather is nice today.",
    "The government announced new economic policies yesterday.",
    "I bought a new phone last week.",
    "The team won the championship game.",
    "Researchers published a study on ocean temperatures.",
    "Please enter your email address to subscribe to our newsletter.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "He said the meeting would be postponed until Friday.",
    "This hotel is located five minutes from the train station.",
    "My daughter started school in September.",
]

ANNOTATOR_CMD = ["claude", "-p", "--model", "sonnet"]

def log(msg):
    print(msg, flush=True)

def ekey(layer, head, entry):
    return f"{layer}:{head}:{entry}"

# ---------------------------------------------------------------- loading

def load_model(ckpt, step):
    manager = ocp.CheckpointManager(
        str(Path(ckpt).resolve()),
        checkpointers={'state': ocp.PyTreeCheckpointer(),
                       'spec': ocp.PyTreeCheckpointer()})
    spec = manager.restore(step, items={'spec': None})['spec']
    model = Ontologizer(**spec)
    # restore as plain numpy: the checkpoint carries GPU sharding metadata
    # that the CPU backend cannot deserialize
    meta = manager.item_metadata(step)['state']
    ra = jax.tree.map(lambda m: ocp.RestoreArgs(restore_type=np.ndarray),
                      meta)
    sd = manager.restore(step, items={'state': None},
                         restore_kwargs={'state': {'restore_args': ra}}
                         )['state']
    params = sd['params'] if 'opt_state' in sd else sd
    if 'params' not in params:
        params = {'params': params}
    while 'params' in params and 'params' in params['params']:
        params = params['params']
    return model, params

_pt = {}

def load_encoder():
    if 'enc' not in _pt:
        from ontologize.data.pretrained import pretrained_transformer
        _pt['dev'] = t.device("cuda" if t.cuda.is_available() else "cpu")
        _pt['enc'], _pt['tok'] = pretrained_transformer(
            "cointegrated/SONAR_200_text_encoder", dtype_str="float32",
            dev=_pt['dev'])
    return _pt['enc'], _pt['tok'], _pt['dev']

def load_decoder():
    if 'dec' not in _pt:
        from transformers import M2M100ForConditionalGeneration
        _pt['dec'] = M2M100ForConditionalGeneration.from_pretrained(
            "raxtemur/SONAR_200_text_decoder", torch_dtype=t.float16)
    return _pt['dec']

def gpu_swap(active):
    """Encoder (~5 GB fp32) and decoder cannot coexist in leftover VRAM while
    training runs; park the inactive one on CPU."""
    other = 'dec' if active == 'enc' else 'enc'
    if other in _pt and next(_pt[other].parameters()).is_cuda:
        _pt[other].to('cpu')
        t.cuda.empty_cache()
    if not next(_pt[active].parameters()).is_cuda:
        _pt[active].to(_pt['dev'])

def sonar_encode(texts, src_lang="eng_Latn"):
    """Returns (unit-norm embeddings (n, d) np, raw mean-pooled norms (n,))."""
    enc, tok, dev = load_encoder()
    gpu_swap('enc')
    tok.src_lang = src_lang
    X, norms = [], []
    with t.no_grad():
        for i in range(0, len(texts), BATCH_ENC):
            chunk = texts[i:i + BATCH_ENC]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAXLEN)
            out = enc(**{k: v.to(dev) for k, v in inputs.items()})
            mask = inputs['attention_mask'].to(dev).unsqueeze(-1).float()
            E = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            norms.append(t.norm(E, p=2, dim=-1).cpu().numpy())
            X.append(tF.normalize(E, p=2, dim=-1).cpu().numpy())
    return np.concatenate(X), np.concatenate(norms)

def sonar_decode(embs, norms):
    """Decode (n, d) unit-norm embeddings (rescaled to norms) to English."""
    from transformers.modeling_outputs import BaseModelOutput
    dec = load_decoder()
    _, tok, dev = load_encoder()
    gpu_swap('dec')
    E = t.from_numpy(np.asarray(embs, np.float32)).to(dev)
    E = tF.normalize(E, p=2, dim=-1) * t.tensor(
        np.asarray(norms, np.float32), device=dev)[:, None]
    E = E.to(dtype=next(dec.parameters()).dtype)
    bos = tok.convert_tokens_to_ids("eng_Latn")
    outs = []
    with t.no_grad():
        for i in range(0, len(E), 16):
            gen = dec.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=E[i:i + 16].unsqueeze(1)),
                forced_bos_token_id=bos, max_length=64, num_beams=4,
                repetition_penalty=1.2)
            outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs

# ---------------------------------------------------------------- model fns

def fwd_probs(module, X, temperature):
    """Forward pass returning reconstruction and per-layer classifications."""
    E, _ = module.encode(X, 0.0, None)
    R, Ps = module.classify(E, temperature=temperature)
    return module.decode(R).astype(jnp.float32), Ps.astype(jnp.float32)

def probs_at(module, X, layer, temperature):
    """Classification probabilities (b, h, k) at `layer`."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    for i in range(layer):
        R, E = module.dictencs[i].withClusts(R, E, temperature=temperature)
    d = module.dictencs[layer]
    return d.dict.cluster(d.classifier(E), temperature).astype(jnp.float32)

def adjoint_inputs(module, P, layer):
    """Adjoint chain from classifications (b, h, k) at `layer` to unit inputs."""
    D = module.dictencs[layer].classifier.rev(P)
    for i in reversed(range(layer)):
        D = module.dictencs[i].classifier.rev(
            D.reshape(D.shape[:-1] + (module.h, module.k)))
    if module.encoded:
        D = module.encoder.rev(D)
    return D / (jnp.linalg.norm(D, axis=-1, keepdims=True) + 1e-9)

def tag_vectors(module, layer):
    """Decoder-space vectors (h*k, d_out) for a layer's dictionary."""
    return module.decode(module.dictencs[layer].tags()).astype(jnp.float32)

# ---------------------------------------------------------------- stage: census

def stage_census(args, model, params, out):
    from ontologize.data.multilingual import mc4_data
    from ontologize.data.loaders import HFDataSource
    from ontologize.data.langs import MC4_TO_SONAR

    log("census: streaming C4 validation split...")
    ds = mc4_data("allenai/c4", split="validation", streaming=True)
    src = HFDataSource(ds, text_key="text")
    texts, langs = [], []
    for item in iter(src):
        txt = item["text"].strip()
        if txt:
            texts.append(txt)
            langs.append(item["lang"])
        if len(texts) >= args.n_corpus:
            break
    log(f"census: {len(texts)} sentences, {len(set(langs))} languages")

    by_lang = defaultdict(list)
    for i, lg in enumerate(langs):
        by_lang[lg].append(i)
    X = np.zeros((len(texts), model.d_in), np.float32)
    norms = np.zeros(len(texts), np.float32)
    _, tok, _ = load_encoder()
    done = 0
    for lg, idxs in by_lang.items():
        Xl, nl = sonar_encode([texts[i] for i in idxs],
                              MC4_TO_SONAR.get(lg, "eng_Latn"))
        X[idxs], norms[idxs] = Xl, nl
        done += len(idxs)
        log(f"census: encoded {done}/{len(texts)}")

    log("census: Ontologizer forward pass...")
    Ps = []
    for i in range(0, len(X), BATCH_ONTO):
        _, P = model.apply(params, jnp.asarray(X[i:i + BATCH_ONTO]), TEMP,
                           method=fwd_probs)
        Ps.append(np.asarray(P))
    P = np.concatenate(Ps, axis=1).reshape(model.l, len(X), model.h, model.k)

    counts = np.zeros((model.l, model.h, model.k), np.int64)
    am = P.argmax(-1)
    for li in range(model.l):
        for hi in range(model.h):
            counts[li, hi] = np.bincount(am[li, :, hi], minlength=model.k)

    np.savez_compressed(
        out / "corpus.npz", X=X.astype(np.float16), norms=norms,
        P=P.astype(np.float16), counts=counts,
        texts=np.array(texts, dtype=object), langs=np.array(langs))

    census = {}
    n = len(X)
    for li in range(model.l):
        for hi in range(model.h):
            order = np.argsort(-counts[li, hi])
            for ki in range(model.k):
                wins = int(counts[li, hi, ki])
                share = wins / n
                won = am[li, :, hi] == ki
                if wins == 0:
                    status = "dead"
                elif share >= CONSTANT_SHARE:
                    status = "constant"
                elif wins < MIN_LIVE_WINS:
                    status = "rare"
                else:
                    status = "live"
                lg_top = []
                if wins:
                    lg, ct = np.unique(np.array(langs)[won], return_counts=True)
                    top = np.argsort(-ct)[:5]
                    lg_top = [[str(lg[j]), int(ct[j])] for j in top]
                census[ekey(li, hi, ki)] = {
                    "layer": li, "head": hi, "entry": ki, "status": status,
                    "wins": wins, "win_share": round(share, 5),
                    "mean_p_won": round(float(P[li, won, hi, ki].mean()), 4) if wins else 0.0,
                    "runner_up": int(order[1] if order[0] == ki else order[0]),
                    "top_langs": lg_top,
                }
    with open(out / "census.json", "w") as f:
        json.dump(census, f)

    by_status = defaultdict(int)
    for v in census.values():
        by_status[v["status"]] += 1
    log(f"census: {dict(by_status)}")
    return census

# ---------------------------------------------------------------- stage: evidence

def select_entries(census, args):
    sel = [v for v in census.values() if v["status"] == "live"]
    if args.layers:
        sel = [v for v in sel if v["layer"] in args.layers]
    sel.sort(key=lambda v: -v["wins"])
    if args.limit:
        sel = sel[:args.limit]
    return sel

def stage_evidence(args, model, params, out):
    census = json.load(open(out / "census.json"))
    dat = np.load(out / "corpus.npz", allow_pickle=True)
    X = dat["X"].astype(np.float32)
    P = dat["P"].astype(np.float32)
    texts, langs, norms = dat["texts"], dat["langs"], dat["norms"]
    entries = select_entries(census, args)
    if (out / "cards.jsonl").exists():
        done = {(c["layer"], c["head"], c["entry"])
                for c in map(json.loads, open(out / "cards.jsonl"))}
        entries = [v for v in entries
                   if (v["layer"], v["head"], v["entry"]) not in done]
    log(f"evidence: {len(entries)} live entries selected")

    ref_norm = float(np.median(norms))
    layers = sorted({v["layer"] for v in entries})

    # duplicate families per layer (decoder-space cosine)
    family = {}
    neighbors = {}
    for li in layers:
        V = np.asarray(model.apply(params, li, method=tag_vectors))
        V = V / (np.linalg.norm(V, axis=-1, keepdims=True) + 1e-9)
        C = V @ V.T
        live_flat = [v["head"] * model.k + v["entry"]
                     for v in census.values()
                     if v["layer"] == li and v["status"] in ("live", "constant")]
        parent = {i: i for i in live_flat}
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a in live_flat:
            for b in live_flat:
                if a < b and C[a, b] >= FAMILY_COS:
                    parent[find(a)] = find(b)
        for i in live_flat:
            family[(li, i // model.k, i % model.k)] = f"{li}:f{find(i)}"
        for v in census.values():
            if v["layer"] != li or v["status"] != "live":
                continue
            fl = v["head"] * model.k + v["entry"]
            order = np.argsort(-C[fl])
            nb = [(int(j), float(C[fl, j])) for j in order
                  if j != fl and (li, j // model.k, j % model.k) in family][:5]
            neighbors[(li, v["head"], v["entry"])] = [
                [ekey(li, j // model.k, j % model.k), round(c, 3)]
                for j, c in nb]

    # archetypes: batched adjoint + gradient refinement
    log("evidence: computing rev archetypes...")
    hs = jnp.asarray([v["head"] for v in entries])
    ks = jnp.asarray([v["entry"] for v in entries])
    arch, arch_ref, arch_p = {}, {}, {}
    for li in layers:
        m = [i for i, v in enumerate(entries) if v["layer"] == li]
        if not m:
            continue
        onehot = jnp.zeros((len(m), model.h, model.k))
        onehot = onehot.at[jnp.arange(len(m)), hs[jnp.array(m)], ks[jnp.array(m)]].set(1.0)
        X0 = model.apply(params, onehot, li, method=adjoint_inputs)

        h_i, k_i = hs[jnp.array(m)], ks[jnp.array(m)]
        def loss_fn(Xc):
            Pc = model.apply(params, Xc, li, TEMP, method=probs_at)
            return -jnp.log(Pc[jnp.arange(len(m)), h_i, k_i] + 1e-9).mean()
        gfn = jax.jit(jax.grad(loss_fn))
        Xr = X0
        for _ in range(REFINE_STEPS):
            Xr = Xr - REFINE_LR * gfn(Xr)
            Xr = Xr / (jnp.linalg.norm(Xr, axis=-1, keepdims=True) + 1e-9)
        Pr = np.asarray(model.apply(params, Xr, li, TEMP, method=probs_at))
        for j, i in enumerate(m):
            v = entries[i]
            arch[i] = np.asarray(X0[j])
            arch_ref[i] = np.asarray(Xr[j])
            arch_p[i] = float(Pr[j, v["head"], v["entry"]])

    # natural reconstructions of the probe panel (shared)
    log("evidence: encoding probe panel...")
    Xp, norm_p = sonar_encode(PANEL)
    Yp, _ = model.apply(params, jnp.asarray(Xp), TEMP, method=fwd_probs)
    Yp = np.asarray(Yp)

    # steering + ablation passes, decode queue
    log("evidence: steering / ablation passes...")
    queue_emb, queue_norm, queue_tag = [], [], []

    def enqueue(tag, emb, norm):
        queue_tag.append(tag)
        queue_emb.append(np.asarray(emb, np.float32))
        queue_norm.append(float(norm))

    for i, v in enumerate(entries):
        li, hi, ki = v["layer"], v["head"], v["entry"]
        enqueue((i, "arch", 0), arch[i], ref_norm)
        enqueue((i, "arch_ref", 0), arch_ref[i], ref_norm)

        Ys, _, Psr = model.apply(
            params, jnp.asarray(Xp), li, temperature=TEMP,
            strength=STEER_STRENGTH,
            h_set=jnp.asarray([hi]), k_set=jnp.asarray([ki]),
            method=Ontologizer.intervene)
        Psr = np.asarray(Psr[li]).reshape(len(PANEL), model.h, model.k)
        v["steer_p"] = round(float(Psr[:, hi, ki].mean()), 3)
        for j in range(len(PANEL)):
            enqueue((i, "steer", j), np.asarray(Ys[j]), norm_p[j])

        top = np.argsort(-P[li, :, hi, ki])[:N_MAXACT]
        v["_top"] = top
        for rank, jx in enumerate(top[:2]):
            Ya, _, _ = model.apply(
                params, jnp.asarray(X[jx][None]), li, temperature=TEMP,
                strength=STEER_STRENGTH,
                h_set=jnp.asarray([hi]), k_set=jnp.asarray([v["runner_up"]]),
                method=Ontologizer.intervene)
            enqueue((i, "ablate", rank), np.asarray(Ya[0]), norms[jx])
            Yn, _ = model.apply(params, jnp.asarray(X[jx][None]), TEMP,
                                method=fwd_probs)
            enqueue((i, "ablate_nat", rank), np.asarray(Yn[0]), norms[jx])
        if (i + 1) % 8 == 0:
            log(f"evidence: interventions {i + 1}/{len(entries)}")

    log(f"evidence: decoding {len(queue_emb) + len(PANEL)} embeddings...")
    panel_nat = sonar_decode(Yp, norm_p)
    decoded = sonar_decode(np.stack(queue_emb), np.array(queue_norm))
    dec = {}
    for tag, txt in zip(queue_tag, decoded):
        dec[tag] = txt

    def clean(s, n=180):
        return str(s).replace("\n", " ")[:n]

    with open(out / "cards.jsonl", "a") as f:
        for i, v in enumerate(entries):
            li, hi, ki = v["layer"], v["head"], v["entry"]
            steer = []
            for j in range(len(PANEL)):
                after = dec[(i, "steer", j)]
                if after.strip() != panel_nat[j].strip():
                    steer.append({"before": panel_nat[j], "after": after})
            ablate = []
            for rank in (0, 1):
                if (i, "ablate", rank) in dec:
                    ablate.append({"natural": dec[(i, "ablate_nat", rank)],
                                   "ablated": dec[(i, "ablate", rank)]})
            card = {
                "key": ekey(li, hi, ki),
                "layer": li, "head": hi, "entry": ki,
                "census": {k2: v[k2] for k2 in
                           ("wins", "win_share", "mean_p_won", "top_langs")},
                "family": family.get((li, hi, ki)),
                "neighbors": neighbors.get((li, hi, ki), []),
                "max_activating": [
                    {"p": round(float(P[li, jx, hi, ki]), 3),
                     "lang": str(langs[jx]), "text": clean(texts[jx])}
                    for jx in v["_top"]],
                "archetype": dec[(i, "arch", 0)],
                "archetype_refined": dec[(i, "arch_ref", 0)],
                "archetype_p": round(arch_p[i], 3),
                "steering": {"strength": STEER_STRENGTH,
                             "realized_p": v["steer_p"],
                             "changed": steer[:6],
                             "n_changed": len(steer),
                             "n_panel": len(PANEL)},
                "ablation": ablate,
            }
            f.write(json.dumps(card) + "\n")
    log(f"evidence: wrote {len(entries)} cards to cards.jsonl")

# ---------------------------------------------------------------- LLM helpers

def ask_llm(prompt, out, tag, timeout=180, retries=3):
    """Run the annotator; returns parsed JSON or None (prompt saved on failure)."""
    err = ""
    for attempt in range(retries):
        if attempt:
            time.sleep(10 * attempt)
        try:
            r = subprocess.run(ANNOTATOR_CMD, input=prompt, capture_output=True,
                               text=True, timeout=timeout)
            if r.returncode == 0:
                m = re.search(r"\{.*\}", r.stdout, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
            err = (r.stderr or r.stdout)[:500]
        except Exception as e:
            err = str(e)
    pdir = out / "prompts"
    pdir.mkdir(exist_ok=True)
    (pdir / f"{tag}.txt").write_text(prompt)
    log(f"  annotator failed for {tag} ({err[:120]}); prompt saved")
    return None

LABEL_PROMPT = """\
You are annotating features of a sparse dictionary-learning model trained on
SONAR multilingual sentence embeddings (web text). Each feature is a dictionary
entry that competes with 31 others inside one classifier head; for every input
sentence exactly one entry per head wins. Tag identity tends to encode
register/frame/style of web text; the soft mixture weights (not the tag
identity) carry instance specifics, so DO NOT label a feature by the topic of
individual example sentences unless the evidence is overwhelming.

Evidence for feature {key} (JSON):
- census: corpus win statistics and language distribution
- max_activating: held-out sentences this entry wins, with confidence
- archetype / archetype_refined: text decoded from the input that maximally
  activates this entry (a synthetic "pure" example of the feature)
- steering.changed: before/after decodes when this feature is softly injected
  into unrelated probe sentences (what it CHANGES when added)
- ablation: decodes of top activating sentences with this entry swapped for
  the head's runner-up (what is LOST without it)
- neighbors: most similar entries in decoder space (>= {fam} cosine means
  near-duplicate)

{card}

Respond with ONLY a JSON object, no other text:
{{"label": "<2-6 word name>",
  "description": "<1-3 sentences: what this feature represents and what the causal evidence shows>",
  "type": "register|frame|topic|language|formatting|mixed|unclear",
  "not_determined": "<1 sentence: what this feature does NOT control>",
  "confidence": "high|medium|low"}}
"""

def stage_label(args, model, params, out):
    cards = [json.loads(l) for l in open(out / "cards.jsonl")]
    done = {}
    ann_path = out / "annotations.jsonl"
    if ann_path.exists():
        for l in open(ann_path):
            a = json.loads(l)
            done[a["key"]] = a
    log(f"label: {len(cards)} cards, {len(done)} already annotated")
    with open(ann_path, "a") as f:
        for c in cards:
            if c["key"] in done:
                continue
            slim = {k: v for k, v in c.items() if k != "key"}
            prompt = LABEL_PROMPT.format(
                key=c["key"], fam=FAMILY_COS,
                card=json.dumps(slim, ensure_ascii=False, indent=1))
            ans = ask_llm(prompt, out, f"label_{c['key'].replace(':', '_')}")
            if ans is None:
                continue
            rec = {"key": c["key"], **{k: ans.get(k) for k in
                   ("label", "description", "type", "not_determined",
                    "confidence")}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            log(f"  {c['key']}: [{rec['type']}/{rec['confidence']}] {rec['label']}")

# ---------------------------------------------------------------- stage: score

DETECT_PROMPT = """\
A feature of a text model has been given this annotation:
label: {label}
description: {description}

Below are {n} numbered sentences (various languages). Decide for which
sentences this feature should be ACTIVE (win its classifier head). Typically
roughly {npos} of them are positives.

{sents}

Respond with ONLY a JSON object: {{"active": [<sentence numbers>]}}
"""

GEN_PROMPT = """\
A feature of a text model has been given this annotation:
label: {label}
description: {description}
type: {type}

Write 20 short, realistic web-text sentences in English that SHOULD strongly
express this feature, and 20 that should NOT express it at all (but are
otherwise similar web text). Respond with ONLY a JSON object:
{{"positive": ["...", ...], "negative": ["...", ...]}}
"""

CAUSAL_PROMPT = """\
A feature of a text model has been given this annotation:
label: {label}
description: {description}

The feature was softly injected into unrelated sentences. Numbered
before/after decoded text pairs:

{pairs}

Question: for EACH pair, is the change consistent with the annotation (does
the "after" text move toward what the label describes)? Respond with ONLY a
JSON object:
{{"scores": [<one score per pair, in order: 0 = contradicts,
             1 = unrelated/unclear, 2 = consistent>],
  "why": "<one sentence overall>"}}
"""

def stage_score(args, model, params, out):
    census = json.load(open(out / "census.json"))
    anns = [json.loads(l) for l in open(out / "annotations.jsonl")]
    cards = {json.loads(l)["key"]: json.loads(l)
             for l in open(out / "cards.jsonl")}
    dat = np.load(out / "corpus.npz", allow_pickle=True)
    P, texts = dat["P"].astype(np.float32), dat["texts"]
    rng = np.random.default_rng(0)

    done = set()
    sc_path = out / "scores.jsonl"
    if sc_path.exists():
        done = {json.loads(l)["key"] for l in open(sc_path)}

    with open(sc_path, "a") as f:
        for a in anns:
            if a["key"] in done or a["key"] not in cards:
                continue
            c = cards[a["key"]]
            li, hi, ki = c["layer"], c["head"], c["entry"]
            rec = {"key": a["key"], "label": a["label"]}

            # -- detection: 32 held-out positives + 64 negatives
            am = P[li, :, hi].argmax(-1)
            pos_pool = np.where(am == ki)[0]
            pos_pool = pos_pool[np.argsort(-P[li, pos_pool, hi, ki])]
            pos = pos_pool[N_MAXACT:N_MAXACT + 32]    # skip evidence examples
            if len(pos) < 8:
                pos = pos_pool[:32]
            neg = rng.choice(np.where(am != ki)[0],
                             size=min(64, (am != ki).sum()), replace=False)
            idxs = np.concatenate([pos, neg])
            order = rng.permutation(len(idxs))
            idxs = idxs[order]
            truth = {int(n_): bool(am[j] == ki)
                     for n_, j in enumerate(idxs, 1)}
            sents = "\n".join(
                f"{n_}. {str(texts[j]).strip()[:160]}"
                for n_, j in enumerate(idxs, 1))
            ans = ask_llm(DETECT_PROMPT.format(
                label=a["label"], description=a["description"],
                n=len(idxs), npos=len(pos), sents=sents),
                out, f"detect_{a['key'].replace(':', '_')}")
            if ans and isinstance(ans.get("active"), list):
                pred = {int(x) for x in ans["active"] if str(x).isdigit()
                        or isinstance(x, int)}
                tp = sum(1 for n_ in pred if truth.get(n_))
                fp = len(pred) - tp
                fn = sum(truth.values()) - tp
                prec = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                rec["detection"] = {"precision": round(prec, 3),
                                    "recall": round(recall, 3),
                                    "n_pos": int(sum(truth.values()))}

            # -- generation: LLM sentences through the real encoder+model
            ans = ask_llm(GEN_PROMPT.format(
                label=a["label"], description=a["description"],
                type=a["type"]), out, f"gen_{a['key'].replace(':', '_')}")
            if ans and ans.get("positive") and ans.get("negative"):
                gp, gn = list(ans["positive"])[:20], list(ans["negative"])[:20]
                Xg, _ = sonar_encode(gp + gn)
                Pg = np.asarray(model.apply(
                    params, jnp.asarray(Xg), li, TEMP, method=probs_at))
                wins = Pg[:, hi].argmax(-1) == ki
                rec["generation"] = {
                    "pos_hit": round(float(wins[:len(gp)].mean()), 3),
                    "neg_fp": round(float(wins[len(gp):].mean()), 3)}

            # -- causal consistency
            if c["steering"]["changed"]:
                ch = c["steering"]["changed"][:6]
                pairs = "\n".join(
                    f"{n_}. BEFORE: {p['before']}\n   AFTER:  {p['after']}"
                    for n_, p in enumerate(ch, 1))
                ans = ask_llm(CAUSAL_PROMPT.format(
                    label=a["label"], description=a["description"],
                    pairs=pairs), out, f"causal_{a['key'].replace(':', '_')}")
                if ans and isinstance(ans.get("scores"), list) and ans["scores"]:
                    ss = [int(x) for x in ans["scores"][:len(ch)]]
                    rec["causal"] = {"score": round(sum(ss) / len(ss), 2),
                                     "scores": ss,
                                     "why": str(ans.get("why", ""))[:200]}

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            d = rec.get("detection", {})
            g = rec.get("generation", {})
            log(f"  {a['key']} '{a['label']}': det P={d.get('precision')} "
                f"R={d.get('recall')}  gen +{g.get('pos_hit')} -{g.get('neg_fp')}  "
                f"causal {rec.get('causal', {}).get('score')}")

# ---------------------------------------------------------------- report

def stage_report(args, model, params, out):
    census = json.load(open(out / "census.json"))
    by_status = defaultdict(int)
    for v in census.values():
        by_status[v["status"]] += 1
    anns = {json.loads(l)["key"]: json.loads(l)
            for l in open(out / "annotations.jsonl")} \
        if (out / "annotations.jsonl").exists() else {}
    scores = {json.loads(l)["key"]: json.loads(l)
              for l in open(out / "scores.jsonl")} \
        if (out / "scores.jsonl").exists() else {}
    cards = {json.loads(l)["key"]: json.loads(l)
             for l in open(out / "cards.jsonl")} \
        if (out / "cards.jsonl").exists() else {}

    with open(out / "report.md", "w") as f:
        f.write(f"# Ontofeature annotations — {args.ckpt} @ step {args.step}\n\n")
        f.write(f"Census: {dict(by_status)} of "
                f"{model.l * model.h * model.k} entries\n\n")
        f.write("| entry | fam | wins | label | type | conf "
                "| det P/R | gen +/− | causal |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        rows = sorted(anns.values(),
                      key=lambda a: -census[a["key"]]["wins"])
        for a in rows:
            k = a["key"]
            s = scores.get(k, {})
            d, g = s.get("detection", {}), s.get("generation", {})
            fam = cards.get(k, {}).get("family", "")
            f.write(f"| {k} | {fam} | {census[k]['wins']} | {a['label']} "
                    f"| {a['type']} | {a['confidence']} "
                    f"| {d.get('precision', '')}/{d.get('recall', '')} "
                    f"| {g.get('pos_hit', '')}/{g.get('neg_fp', '')} "
                    f"| {s.get('causal', {}).get('score', '')} |\n")
        f.write("\nFull evidence: `cards.jsonl`; labels: `annotations.jsonl`; "
                "scores: `scores.jsonl`.\n")
    log(f"report: {out}/report.md")

# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("step", type=int)
    ap.add_argument("--stage", default="all",
                    choices=["census", "evidence", "label", "score",
                             "report", "all"])
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-corpus", type=int, default=N_CORPUS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    out = Path(args.out) if args.out else \
        Path(f"scratch/annotate_{ckpt.name}_{args.step}")
    out.mkdir(parents=True, exist_ok=True)

    model, params = load_model(ckpt, args.step)
    log(f"model: l={model.l} h={model.h} k={model.k} @ step {args.step}")

    stages = ([args.stage] if args.stage != "all"
              else ["census", "evidence", "label", "score", "report"])
    for st in stages:
        if st == "census" and (out / "census.json").exists() \
                and args.stage == "all":
            log("census: cached, skipping")
            continue
        log(f"=== stage: {st}")
        globals()[f"stage_{st}"](args, model, params, out)

if __name__ == "__main__":
    main()
