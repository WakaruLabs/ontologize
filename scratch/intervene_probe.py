# Ontologizer.intervene probe on real sentences — layer-1 ontofeatures.
#
# For each carrier sentence and each donor dictionary entry (h,k): force the
# entry with (a) withArgs, which only affects layers after layer 1, and
# (b) Ontologizer.intervene, which revs the classification delta back to input
# space and re-runs a clean forward pass, at several steering strengths.
# Decodes every result with the SONAR text decoder for side-by-side reading.
#
# Usage: uv run python scratch/intervene_probe.py

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

from pathlib import Path

import jax
import jax.numpy as jnp
jax.random.PRNGKey(0)

import numpy as np
import torch as t
import torch.nn.functional as tF
import orbax.checkpoint as ocp

from transformers import M2M100ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from ontologize.ontologizer import Ontologizer, DictIntervention
from ontologize.data.pretrained import pretrained_transformer

CKPT = Path("data/out/sonar/multilingual").resolve()
STEP = 85000
OUT = Path("scratch/intervene_85000")
OUT.mkdir(parents=True, exist_ok=True)

LAYER = 1
TEMP = 0.1
STRENGTHS = [0.1, 0.35, 1.0]

SENTENCES = {
    "weather":    "The weather is nice today.",
    "government": "The government announced new economic policies yesterday.",
    "phone":      "I bought a new phone last week.",
    "team":       "The team won the championship game.",
}
# donor entries picked from rev_85000: distinct decoded archetypes
ENTRIES = {
    "breadcrumb (You are here: Home > ...)":            (20, 13),
    "cta (find out more by clicking here)":             (5, 1),
    "fun (We're going to have a lot of fun)":           (22, 0),
}

def log(msg):
    print(msg, flush=True)

# ---------------------------------------------------------------- checkpoint
log(f"Loading checkpoint {CKPT} @ step {STEP}...")
manager = ocp.CheckpointManager(
    CKPT,
    checkpointers={'state': ocp.PyTreeCheckpointer(),
                   'spec': ocp.PyTreeCheckpointer()})
spec = manager.restore(STEP, items={'spec': None})['spec']
model = Ontologizer(**spec)
sd = manager.restore(STEP, items={'state': None})['state']
params = sd['params'] if 'opt_state' in sd else sd
if 'params' not in params:
    params = {'params': params}
while isinstance(params.get('params'), dict) and 'params' in params['params']:
    params = params['params']

# ---------------------------------------------------------------- encode
log("Loading SONAR encoder/decoder...")
device = t.device("cuda")
pt_encoder, tokenizer = pretrained_transformer(
    "cointegrated/SONAR_200_text_encoder", dtype_str="float32", dev=device)
pt_decoder = M2M100ForConditionalGeneration.from_pretrained(
    "raxtemur/SONAR_200_text_decoder").to(device)

names = list(SENTENCES)
X, norm = {}, {}
tokenizer.src_lang = "eng_Latn"
with t.no_grad():
    for name in names:
        inputs = tokenizer([SENTENCES[name]], return_tensors="pt", padding=True)
        outputs = pt_encoder(**{k: v.to(device) for k, v in inputs.items()})
        mask = inputs['attention_mask'].to(device).unsqueeze(-1).float()
        E_mean = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        norm[name] = float(t.norm(E_mean, p=2, dim=-1))
        X[name] = jnp.asarray(tF.normalize(E_mean, p=2, dim=-1).cpu().numpy()[0])
log(f"Encoded {len(names)} sentences.")

# ---------------------------------------------------------------- helpers
def natural(Xb):
    arglist = [DictIntervention() for _ in range(model.l)]
    Y, _, _, _ = model.apply(params, Xb[None], arglist=arglist, temperature=TEMP,
                             method=Ontologizer.withArgs)
    return Y[0].astype(jnp.float32)

def with_args(Xb, h_idx, k_idx):
    arglist = [DictIntervention() for _ in range(model.l)]
    arglist[LAYER] = DictIntervention(h_set=jnp.asarray([h_idx]),
                                      k_set=jnp.asarray([k_idx]))
    Y, _, _, _ = model.apply(params, Xb[None], arglist=arglist, temperature=TEMP,
                             method=Ontologizer.withArgs)
    return Y[0].astype(jnp.float32)

def intervene(Xb, h_idx, k_idx, strength):
    Y, X_int, Ps = model.apply(
        params, Xb[None], LAYER, temperature=TEMP, strength=strength,
        h_set=jnp.asarray([h_idx]), k_set=jnp.asarray([k_idx]),
        method=Ontologizer.intervene)
    P1 = np.asarray(Ps[LAYER][0], np.float32).reshape(model.h, model.k)
    return (Y[0].astype(jnp.float32), X_int[0].astype(jnp.float32),
            float(P1[h_idx, k_idx]), int(P1[h_idx].argmax()))

def cos(a, b):
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-9))

# ---------------------------------------------------------------- run
log("Running interventions...")
Y_nat = {n: natural(X[n]) for n in names}
rows = []            # dicts with all metrics
embs, keys = [], []  # embeddings to decode

for n in names:
    for ename, (h_idx, k_idx) in ENTRIES.items():
        Yw = with_args(X[n], h_idx, k_idx)
        rows.append({"sent": n, "entry": ename, "mode": "withArgs", "s": None,
                     "p": None, "hit": None,
                     "cos_nat": cos(Yw, Y_nat[n]), "cos_in": None})
        embs.append(np.asarray(Yw)); keys.append(len(rows) - 1)
        for s in STRENGTHS:
            Yi, Xi, p, argm = intervene(X[n], h_idx, k_idx, s)
            rows.append({"sent": n, "entry": ename, "mode": "intervene", "s": s,
                         "p": p, "hit": argm == k_idx,
                         "cos_nat": cos(Yi, Y_nat[n]), "cos_in": cos(Xi, X[n])})
            embs.append(np.asarray(Yi)); keys.append(len(rows) - 1)
    log(f"  {n} done")

# ---------------------------------------------------------------- decode
log("Decoding...")
all_embs = [np.asarray(Y_nat[n]) for n in names] + embs
all_norms = [norm[n] for n in names] + [norm[rows[i]["sent"]] for i in keys]
pt_embs = t.from_numpy(np.stack(all_embs).copy()).to(device)
pt_embs = tF.normalize(pt_embs, p=2, dim=-1) * t.tensor(all_norms, device=device)[:, None]
forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn")
decoded = []
with t.no_grad():
    for i in range(0, len(pt_embs), 16):
        batch = pt_embs[i:i + 16]
        gen = pt_decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=batch.unsqueeze(1)),
            forced_bos_token_id=forced_bos,
            max_length=64, num_beams=4, repetition_penalty=1.2)
        decoded.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
nat_texts = dict(zip(names, decoded[:len(names)]))
for i, txt in zip(keys, decoded[len(names):]):
    rows[i]["text"] = txt

# ---------------------------------------------------------------- report
with open(OUT / "intervene_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Ontologizer.intervene on real sentences — layer {LAYER}, "
            f"step {STEP}, temperature {TEMP}\n")
    f.write("withArgs forces the entry downstream-only; intervene revs the "
            "classification delta\nback to input space (strength s) and "
            "re-runs a clean forward pass.\n")
    f.write("p = realized probability of forced tag; cos_in = cos(steered "
            "input, original input).\n\n")
    f.write("Natural reconstructions:\n")
    for n in names:
        f.write(f"  {n:11s} IN : {SENTENCES[n]}\n")
        f.write(f"  {'':11s} OUT: {nat_texts[n]}\n")
    for n in names:
        for ename, (h_idx, k_idx) in ENTRIES.items():
            f.write(f"\n{'='*100}\n")
            f.write(f"{n}  <-  head {h_idx} entry {k_idx}: {ename}\n")
            sel = [r for r in rows if r["sent"] == n and r["entry"] == ename]
            for r in sel:
                if r["mode"] == "withArgs":
                    f.write(f"  withArgs        cos_nat {r['cos_nat']:.3f}"
                            f"{'':22s}  {r['text']}\n")
                else:
                    mark = "HIT " if r["hit"] else "miss"
                    f.write(f"  intervene s={r['s']:<4} cos_nat {r['cos_nat']:.3f} "
                            f"cos_in {r['cos_in']:.3f} [{mark} p={r['p']:.2f}]"
                            f"  {r['text']}\n")

log(f"Done. Report: {OUT}/intervene_results.txt")
