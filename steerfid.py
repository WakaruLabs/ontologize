"""Steering fidelity: effect vs collateral at matched intervention magnitude.

For each selected feature, steer held-out embeddings along the feature's
native direction, decode through M2M100, and measure both sides of the
trade every steering claim rests on:

  effect      cycle consistency: re-encode the steered decode with the
              SONAR encoder and measure the target feature's activation
              gain, normalized by the feature's q99 activation over a
              reference subsample (1.0 = steered text now activates the
              feature like a top-activating corpus text). "hit" = the
              cycled activation clears the feature's q90.
  collateral  1 - chrF(base decode, steered decode): how much of the
              original text the intervention destroyed.

Steering directions are unit-normalized and applied at matched embedding
magnitudes (--mags, relative to the unit-norm SONAR inputs), so
architectures compare at equal intervention budget:

  sae params.npz     x' = x + m * W_dec[j]/|W_dec[j]|        (mode dec)
                     x' = x + m * eigenfeature_j              (mode eig,
                     bilinear runs -- the closed-form input direction)
  onto checkpoint    the model's own intervention interface: Y with head
                     (li,hi) forced to entry ki via withArgs minus the
                     unintervened Y, per sample, unit-normalized
  random             --n-random random unit directions: the control curve
                     (collateral without intended effect)

Per-steer decodes go to <out>/steers.jsonl for qualitative inspection;
per-(mode, feature, magnitude) means to <out>/steer.csv. NOTE: Ontologizer
checkpoints restore on GPU JAX only; M2M100 runs on --device.

  uv run python steerfid.py --model data/out/sonar/sae/m5120_k32/params.npz
  uv run python steerfid.py --model data/out/sonar/sae/m5120_k32_bl/params.npz --mode eig
  uv run python steerfid.py --model data/out/sonar/multilingual/resid_nc --device cuda
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import json
import numpy as np
from pathlib import Path

import sae

ENCODER_ID = "cointegrated/SONAR_200_text_encoder"
DECODER_ID = "raxtemur/SONAR_200_text_decoder"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="sae params.npz or onto checkpoint dir")
    p.add_argument("--mode", choices=["dec", "eig"], default="dec",
                   help="sae steering direction (eig needs a bilinear run)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--n-samples", type=int, default=8,
                   help="cache-tail rows steered per feature")
    p.add_argument("--mags", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--n-random", type=int, default=8,
                   help="random-direction controls")
    p.add_argument("--sub-rows", type=int, default=16384,
                   help="reference rows for firing rates and quantiles")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--b-decode", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default data/out/sonar/steerfid/<model name>")
    return p.parse_args()


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-9)


def pick_features(fire_rate, n, seed, min_rate=1e-3):
    """Seeded choice among live features."""
    live = np.where(fire_rate > min_rate)[0]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(live, min(n, len(live)), replace=False))


def effect_scores(a_base, a_steer, q_hi, q_hit):
    """(normalized effect, hit) per sample for one feature."""
    eff = (a_steer - a_base) / (q_hi + 1e-9)
    return eff, (a_steer >= q_hit).astype(np.float32)


def firing_quantiles(sub, thr, q_hi=0.9, q_hit=0.5):
    """Per-feature quantiles of the FIRING activations. Sparse features
    are mostly zeros, so unconditional quantiles are 0 and would blow up
    the effect normalization; conditioning on act > thr gives 'how strong
    is a typical/strong firing'."""
    n = sub.shape[1]
    hi = np.ones(n)
    hit = np.ones(n)
    for i in range(n):
        pos = sub[sub[:, i] > thr, i]
        if len(pos):
            hi[i] = np.quantile(pos, q_hi)
            hit[i] = np.quantile(pos, q_hit)
    return hi, hit


def load_steerable(path, cfg):
    """-> (acts_fn, deltas_fn(X, f) -> unit (S, d), F, fire_thr, name).
    deltas_fn returns the model's native steering direction per sample."""
    import jax
    import jax.numpy as jnp
    p = Path(path)
    if p.suffix == ".npz":
        params = {k: jnp.asarray(v) for k, v in np.load(p).items()}
        meta = json.loads((p.parent / "meta.json").read_text()) \
            if (p.parent / "meta.json").exists() else \
            {"topk": 32, "groups": 0, "group_fn": "top1"}
        acts = jax.jit(lambda X: sae.encode(
            params, X, meta["topk"], meta["groups"], meta["group_fn"]))
        if cfg.mode == "eig":
            assert "W_enc2" in params, "--mode eig needs a bilinear run"
            dirs = np.asarray(sae.eigenfeatures(params))
        else:
            dirs = unit(np.asarray(params["W_dec"]))

        def deltas(X, f):
            return np.broadcast_to(dirs[f], X.shape)

        return acts, deltas, dirs.shape[0], 0.0, p.parent.name

    from pareto import load_onto
    from ontologize.ontologizer import Ontologizer, DictIntervention
    model, mparams, step = load_onto(str(p), cfg.step)
    l, h, k = model.l, model.h, model.k
    T = cfg.temperature

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = E
        Ps = []
        for i, de in enumerate(module.dictencs):
            P = de.dict.cluster(de.classifier(Ein), T)
            Ps.append(P)
            R = R + de.dict.combine(de.dict.hfwd(P))
            if i < module.l - 1:
                Ein = module.nextinput(X, R, None)
        return jnp.stack(Ps, 1).reshape(X.shape[0], -1)

    acts = jax.jit(lambda X: model.apply({"params": mparams}, X, method=probe))

    def run_args(X, arglist):
        Y, _, _, _ = model.apply({"params": mparams}, jnp.asarray(X), arglist,
                                 temperature=T, method=Ontologizer.withArgs)
        return np.asarray(Y)

    def deltas(X, f):
        li, hi, ki = f // (h * k), (f // k) % h, f % k
        noop = [DictIntervention() for _ in range(l)]
        arglist = [DictIntervention() for _ in range(l)]
        arglist[li] = DictIntervention(h_set=jnp.array([hi]),
                                       k_set=jnp.array([ki]))
        return unit(run_args(X, arglist) - run_args(X, noop))

    return acts, deltas, l * h * k, 2.0 / k, f"{p.name}_{step}"


def main():
    import jax.numpy as jnp
    import torch as t
    from transformers import M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from ontologize.data.pretrained import pretrained_transformer, encode
    from ontologize.data.loaders import TokenizeTransform
    from textfid import chrf

    cfg = parse_args()
    acts_fn, deltas_fn, F, thr, name = load_steerable(cfg.model, cfg)
    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/steerfid") / name
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    # reference pass: firing rates -> feature selection -> quantiles
    n_sub = min(cfg.sub_rows, mm.shape[0]) // cfg.b * cfg.b
    rate = np.zeros(F)
    for i in range(0, n_sub, cfg.b):
        A = np.asarray(acts_fn(jnp.asarray(
            np.asarray(mm[i:i + cfg.b], dtype=np.float32))))
        rate += (A > thr).mean(0)
    rate /= n_sub // cfg.b
    feats = pick_features(rate, cfg.n_features, cfg.seed)
    sub = np.concatenate([np.asarray(acts_fn(jnp.asarray(
        np.asarray(mm[i:i + cfg.b], dtype=np.float32))))[:, feats]
        for i in range(0, n_sub, cfg.b)])
    q_hi, q_hit = firing_quantiles(sub, thr)

    X_s = np.asarray(mm[-cfg.n_samples:], dtype=np.float32)

    # torch: decoder for generation, encoder for the cycle
    dev = t.device(cfg.device)
    pt_enc, tokenizer = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    refs = tokenizer(["The weather is nice today.",
                      "She walked to the store to buy some bread.",
                      "Scientists discovered a new species in the rainforest."],
                     return_tensors="pt", padding=True).to(dev)
    with t.no_grad():
        hh = pt_enc(**refs).last_hidden_state
        mask = refs["attention_mask"].unsqueeze(-1).float()
        ref_norm = t.norm((hh * mask).sum(1) / mask.sum(1), dim=-1).mean().item()
    dec = M2M100ForConditionalGeneration.from_pretrained(DECODER_ID).to(dev)
    dec.eval()
    ENG = tokenizer.convert_tokens_to_ids("eng_Latn")
    tok = TokenizeTransform(tokenizer, maxlen=512)

    def generate(Y):
        texts = []
        with t.no_grad():
            for i in range(0, len(Y), cfg.b_decode):
                Yb = t.from_numpy(Y[i:i + cfg.b_decode]).to(dev, t.float32)
                Yb = t.nn.functional.normalize(Yb, dim=-1) * ref_norm
                gen = dec.generate(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=Yb.unsqueeze(1)),
                    forced_bos_token_id=ENG, max_length=48, num_beams=1,
                    repetition_penalty=1.2)
                texts += [tokenizer.decode(g, skip_special_tokens=True).strip()
                          for g in gen]
        return texts

    def cycle_acts(texts):
        batch = [tok.map({"text": s, "lang": "en"}) for s in texts]
        batch = {k_: np.stack([b[k_] for b in batch])
                 for k_ in ("input_ids", "attention_mask")}
        E = np.asarray(encode(pt_enc, batch, dev))
        return np.asarray(acts_fn(jnp.asarray(E)))

    base_txt = generate(X_s)
    # baseline activations from the CYCLED base decode, so effect compares
    # steered and unsteered text through the same decode->encode roundtrip
    a_base_cyc = cycle_acts(base_txt)
    rng = np.random.default_rng(cfg.seed)
    rand_dirs = unit(rng.normal(size=(cfg.n_random, X_s.shape[1]))
                     .astype(np.float32))

    rows, log = [], open(out / "steers.jsonl", "w")
    jobs = [("feat", int(f)) for f in feats] + \
           [("random", r) for r in range(cfg.n_random)]
    for kind, j in jobs:
        D = deltas_fn(X_s, j) if kind == "feat" else \
            np.broadcast_to(rand_dirs[j], X_s.shape)
        for m in cfg.mags:
            Xp = (X_s + m * D).astype(np.float32)
            steer_txt = generate(Xp)
            coll = np.array([1 - chrf(a, b)
                             for a, b in zip(base_txt, steer_txt)])
            if kind == "feat":
                fi = int(np.where(feats == j)[0][0])
                a_cyc = cycle_acts(steer_txt)[:, j]
                eff, hit = effect_scores(a_base_cyc[:, j], a_cyc,
                                         q_hi[fi], q_hit[fi])
                rows.append([kind, j, m, float(eff.mean()),
                             float(hit.mean()), coll.mean()])
            else:
                eff = np.full(len(X_s), np.nan)
                rows.append([kind, j, m, float("nan"), float("nan"),
                             coll.mean()])
            for s in range(len(X_s)):
                log.write(json.dumps(
                    {"kind": kind, "feature": j, "mag": m, "sample": s,
                     "base": base_txt[s], "steered": steer_txt[s],
                     "collateral": float(coll[s]),
                     "effect": None if kind != "feat" else float(eff[s])},
                    ensure_ascii=False) + "\n")
        print(f"{kind} {j}: done", flush=True)
    log.close()

    with open(out / "steer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "feature", "mag", "effect", "hit_rate",
                    "collateral"])
        w.writerows(rows)
    print(f"\n{'mag':>6} {'effect':>8} {'hit rate':>9} {'collateral':>11} "
          f"{'rand coll':>10}")
    arr = np.array([r[2:] for r in rows], dtype=np.float64)
    kinds = np.array([r[0] for r in rows])
    for m in cfg.mags:
        sel = (arr[:, 0] == m) & (kinds == "feat")
        rnd = (arr[:, 0] == m) & (kinds == "random")
        print(f"{m:>6.2f} {np.nanmean(arr[sel, 1]):>8.3f} "
              f"{np.nanmean(arr[sel, 2]):>9.3f} {arr[sel, 3].mean():>11.3f} "
              f"{arr[rnd, 3].mean() if rnd.any() else float('nan'):>10.3f}")
    print(f"-> {out / 'steer.csv'}")


if __name__ == "__main__":
    main()
