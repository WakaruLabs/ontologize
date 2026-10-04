"""RL for classifications: can a small policy trained to push a chosen
head's classification high (or low) beat the model's own paired-steering
vector at matched intervention magnitude?

Keira's note (paper/description introduction, "Steering could push it too
far off manifold"): "can we RL a model for high/low scores on a
classification and have it converge faster than paired steering or PCA
vectors?" This is the minimal first experiment:

  reward   a chosen head/entry's classification probability under the
           FROZEN Ontologizer (the same soft-forward tag-probability probe
           steerfid.py and autointerp.py use), normalized by the entry's
           firing quantile q_hi so 1.0 = "activates like a top corpus
           row"; minus a collateral penalty.
  policy   a small MLP adapter on the SONAR embedding: x' is x plus a
           unit direction (the MLP's output, normalized) scaled to a
           FIXED magnitude --mag, then rescaled to |x| -- the exact
           matched-magnitude protocol steerfid.py scores steering under,
           so policy and paired steering compare at equal budget.
  loop     two reward channels:
             embed  (runnable sanity tier) reward = tag-probability gain
                    of x' itself, collateral = 1 - cos(x, x'). Fully
                    differentiable through the frozen jax model, so this
                    tier trains by direct gradient ascent -- it exists to
                    debug the policy/normalization and to give REINFORCE
                    a ceiling, not as the claim test.
             cycle  (the actual experiment) reward = cycle-consistency:
                    decode x' through M2M100, re-encode with SONAR, take
                    the entry's activation gain on the re-encoded
                    embedding; collateral = 1 - chrF(base decode, steered
                    decode). Generation is non-differentiable, so this
                    tier is REINFORCE: a ~ N(mu_theta(x), sigma^2 I),
                    x' = renorm(x + mag * a/|a|), per-state mean-reward
                    baseline, surrogate grad through log N(a; mu).
  compare  eval stage scores policy vs the model's own paired steering
           (withArgs set-intervention delta, as steerfid.py's onto mode)
           vs random directions, on held-out cache-tail rows, reporting
           steerfid's effect / hit-rate / collateral columns.

"Converge faster than paired steering" is operationalized honestly: paired
steering needs zero training, so the questions are (a) how many reward
queries the policy needs to REACH the paired-steering effect at matched
collateral, and (b) whether it EXCEEDS it with more. compare.csv +
rl_log.csv contain both curves.

  # sanity tier (GPU jax only, minutes)
  uv run python experiments/rl-classifications/rl_reinforce.py train \\
      --ckpt data/out/sonar/multilingual/resid_nc --layer 0 --head 3 \\
      --entry 5 --reward embed --steps 500
  # the real loop (adds torch M2M100 decode in the loop; slow)
  uv run python experiments/rl-classifications/rl_reinforce.py train \\
      --ckpt data/out/sonar/multilingual/resid_nc --layer 0 --head 3 \\
      --entry 5 --reward cycle --steps 300 --device cuda
  uv run python experiments/rl-classifications/rl_reinforce.py eval \\
      --ckpt data/out/sonar/multilingual/resid_nc --layer 0 --head 3 \\
      --entry 5 --device cuda

See README.md for which parts are design vs runnable.
"""
# disable preallocation so jax and torch share the GPU (as steerfid.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))

from steerfid import unit, firing_quantiles, effect_scores
from textfid import chrf

ENCODER_ID = "cointegrated/SONAR_200_text_encoder"
DECODER_ID = "raxtemur/SONAR_200_text_decoder"
DIR_EPS = 1e-9  # normalization floor, matches steerfid.unit


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["train", "eval"])
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc",
                   help="frozen Ontologizer checkpoint dir (resid_nc_hm is "
                        "the other live run); restores on GPU JAX only")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, default=0)
    p.add_argument("--entry", type=int, default=0,
                   help="(layer, head, entry) select the classification; "
                        "pick a live entry (autointerp freq or steerfid "
                        "fire rates)")
    p.add_argument("--direction", choices=["high", "low"], default="high")
    p.add_argument("--reward", choices=["embed", "cycle"], default="embed")
    p.add_argument("--mag", type=float, default=0.5,
                   help="fixed steering magnitude relative to the unit-norm "
                        "SONAR inputs (steerfid's matched-budget knob)")
    p.add_argument("--lam", type=float, default=1.0,
                   help="collateral penalty weight in the reward")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--b", type=int, default=32,
                   help="states per update (cycle mode decodes b*(n_samples"
                        "+1) texts per step; keep small)")
    p.add_argument("--n-samples", type=int, default=4,
                   help="REINFORCE action samples per state")
    p.add_argument("--sigma", type=float, default=0.3,
                   help="policy exploration std around mu")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--sub-rows", type=int, default=16384,
                   help="reference rows for the entry's firing quantiles")
    p.add_argument("--eval-rows", type=int, default=32768,
                   help="cache tail excluded from training (sae.py split)")
    p.add_argument("--n-eval", type=int, default=16,
                   help="held-out rows scored per arm in the eval stage")
    p.add_argument("--rows-file", default=None,
                   help="npy of training row indices (e.g. the entry's "
                        "top-activating rows: required for --direction low "
                        "to have anything to push down)")
    p.add_argument("--b-probe", type=int, default=4096)
    p.add_argument("--b-decode", type=int, default=32)
    p.add_argument("--save-each", type=int, default=50)
    p.add_argument("--device", default="cpu",
                   help="torch device for M2M100/SONAR (cycle + eval)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default experiments/rl-classifications/out/"
                        "<run>_l<layer>h<head>k<entry>_<reward>")
    return p.parse_args()


# ---------- frozen reward model (jax) ----------

def load_frozen(cfg):
    """Frozen Ontologizer: (acts_col, deltas_fn, meta). acts_col maps a
    batch of embeddings to the target entry's tag probability
    (differentiable); deltas_fn is the paired-steering direction
    (withArgs set-intervention decode delta, steerfid's onto mode)."""
    import jax
    import jax.numpy as jnp
    from pareto import load_onto
    from ontologize.ontologizer import Ontologizer, DictIntervention

    model, mparams, step = load_onto(cfg.ckpt, cfg.step)
    l, h, k = model.l, model.h, model.k
    if not (0 <= cfg.layer < l and 0 <= cfg.head < h and 0 <= cfg.entry < k):
        raise SystemExit(f"(layer, head, entry) out of range for "
                         f"l={l} h={h} k={k}")
    f = (cfg.layer * h + cfg.head) * k + cfg.entry
    T = cfg.temperature

    def probe(module, X):  # steerfid's soft-forward tag-probability probe
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

    acts = jax.jit(lambda X: model.apply({"params": mparams}, X,
                                         method=probe))

    def run_args(X, arglist):
        Y, _, _, _ = model.apply({"params": mparams}, jnp.asarray(X),
                                 arglist, temperature=T,
                                 method=Ontologizer.withArgs)
        return np.asarray(Y)

    def deltas(X):
        noop = [DictIntervention() for _ in range(l)]
        arglist = [DictIntervention() for _ in range(l)]
        arglist[cfg.layer] = DictIntervention(
            h_set=jnp.array([cfg.head]), k_set=jnp.array([cfg.entry]))
        return unit(run_args(X, arglist) - run_args(X, noop))

    meta = {"l": l, "h": h, "k": k, "f": f, "step": int(step),
            "run": Path(cfg.ckpt).name}
    return acts, deltas, meta


def reference_quantiles(acts, mm, f, sub_rows, b, k):
    """(fire rate, q_hi, q_hit) of the target entry over a cache-head
    reference slice; thr = 2/k, autointerp's tag firing threshold."""
    import jax.numpy as jnp
    thr = 2.0 / k
    n = min(sub_rows, mm.shape[0]) // b * b
    col = np.concatenate([
        np.asarray(acts(jnp.asarray(
            np.asarray(mm[i:i + b], np.float32))))[:, f]
        for i in range(0, n, b)])
    rate = float((col > thr).mean())
    q_hi, q_hit = firing_quantiles(col[:, None], thr)
    return rate, float(q_hi[0]), float(q_hit[0])


# ---------- policy (raw jax MLP + optax) ----------

def init_policy(seed, d, hidden):
    rng = np.random.default_rng(seed)
    s1 = (1.0 / d) ** 0.5
    s2 = 1e-2 * (1.0 / hidden) ** 0.5  # near-zero delta at init
    return {"W1": rng.normal(0, s1, (d, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, np.float32),
            "W2": rng.normal(0, s2, (hidden, d)).astype(np.float32),
            "b2": np.zeros(d, np.float32)}


def policy_mu(theta, X):
    import jax.numpy as jnp
    Hd = jnp.tanh(X @ theta["W1"] + theta["b1"])
    return Hd @ theta["W2"] + theta["b2"]


def steer(X, direction, mag):
    """Apply a direction at fixed magnitude, then rescale to |x| (the
    Ontologizer.intervene / steerfid convention: SONAR inputs are
    unit-norm, so effects are direction changes, not norm changes)."""
    import jax.numpy as jnp
    D = direction / (jnp.linalg.norm(direction, axis=-1, keepdims=True)
                     + DIR_EPS)
    Xp = X + mag * D
    nX = jnp.linalg.norm(X, axis=-1, keepdims=True)
    return Xp * nX / (jnp.linalg.norm(Xp, axis=-1, keepdims=True) + DIR_EPS)


def make_embed_update(acts, f, sign, mag, lam, q_hi, tx):
    """Differentiable sanity tier: direct gradient ascent on the in-embed
    reward. Returns jitted (theta, opt_state, X) -> (theta, opt_state,
    reward, eff, coll)."""
    import jax
    import jax.numpy as jnp
    import optax

    def loss_fn(theta, X):
        Xp = steer(X, policy_mu(theta, X), mag)
        eff = sign * (acts(Xp)[:, f] - acts(X)[:, f]) / (q_hi + DIR_EPS)
        cos = (X * Xp).sum(-1) / (
            jnp.linalg.norm(X, axis=-1) * jnp.linalg.norm(Xp, axis=-1)
            + DIR_EPS)
        coll = 1.0 - cos
        r = eff - lam * coll
        return -r.mean(), (r.mean(), eff.mean(), coll.mean())

    @jax.jit
    def update(theta, opt_state, X):
        (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(theta, X)
        upd, opt_state = tx.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, upd)
        return theta, opt_state, *aux

    return update


def make_reinforce_update(sigma, tx):
    """REINFORCE surrogate: given sampled actions A (S, b, d) and centered
    advantages ADV (S, b), ascend E[ADV * log N(A; mu_theta(x), sigma^2)].
    Returns jitted (theta, opt_state, X, A, ADV) -> (theta, opt_state)."""
    import jax
    import jax.numpy as jnp
    import optax

    def surrogate(theta, X, A, ADV):
        mu = policy_mu(theta, X)                      # (b, d)
        lp = -((A - mu[None]) ** 2).sum(-1) / (2 * sigma ** 2)
        return -(ADV * lp).mean()

    @jax.jit
    def update(theta, opt_state, X, A, ADV):
        g = jax.grad(surrogate)(theta, X, A, ADV)
        upd, opt_state = tx.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, upd)
        return theta, opt_state

    return update


# ---------- torch generation stack (cycle reward + eval) ----------

def torch_stack(cfg):
    """(generate(Y) -> texts, encode_texts(texts) -> embeddings); the
    decode/encode pair from steerfid.py's main, unchanged."""
    import torch as t
    from transformers import M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from ontologize.data.pretrained import pretrained_transformer, encode
    from ontologize.data.loaders import TokenizeTransform

    dev = t.device(cfg.device)
    pt_enc, tokenizer = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    refs = tokenizer(["The weather is nice today.",
                      "She walked to the store to buy some bread.",
                      "Scientists discovered a new species in the rainforest."],
                     return_tensors="pt", padding=True).to(dev)
    with t.no_grad():
        hh = pt_enc(**refs).last_hidden_state
        mask = refs["attention_mask"].unsqueeze(-1).float()
        ref_norm = t.norm((hh * mask).sum(1) / mask.sum(1),
                          dim=-1).mean().item()
    dec = M2M100ForConditionalGeneration.from_pretrained(DECODER_ID).to(dev)
    dec.eval()
    eng = tokenizer.convert_tokens_to_ids("eng_Latn")
    tok = TokenizeTransform(tokenizer, maxlen=512)

    def generate(Y):
        texts = []
        with t.no_grad():
            for i in range(0, len(Y), cfg.b_decode):
                Yb = t.from_numpy(np.asarray(Y[i:i + cfg.b_decode],
                                             np.float32)).to(dev)
                Yb = t.nn.functional.normalize(Yb, dim=-1) * ref_norm
                gen = dec.generate(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=Yb.unsqueeze(1)),
                    forced_bos_token_id=eng, max_length=48, num_beams=1,
                    repetition_penalty=1.2)
                texts += [tokenizer.decode(g, skip_special_tokens=True)
                          .strip() for g in gen]
        return texts

    def encode_texts(texts):
        E = []
        for i in range(0, len(texts), cfg.b_decode):
            batch = [tok.map({"text": s, "lang": "en"})
                     for s in texts[i:i + cfg.b_decode]]
            batch = {k_: np.stack([x[k_] for x in batch])
                     for k_ in ("input_ids", "attention_mask")}
            E.append(np.asarray(encode(pt_enc, batch, dev)))
        return np.concatenate(E)

    return generate, encode_texts


# ---------- training ----------

def out_dir(cfg, meta):
    if cfg.out:
        return Path(cfg.out)
    return EXP / "out" / (f"{meta['run']}_l{cfg.layer}h{cfg.head}"
                          f"k{cfg.entry}_{cfg.reward}")


def train_rows(cfg, n_total):
    if cfg.rows_file:
        rows = np.load(cfg.rows_file).astype(np.int64)
        return rows[rows < n_total - cfg.eval_rows]
    if cfg.direction == "low":
        raise SystemExit("--direction low needs --rows-file (rows where "
                         "the entry actually fires; random corpus rows "
                         "give a zero gradient floor)")
    return np.arange(n_total - cfg.eval_rows)


def log_row(path, row, cols):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(cols)
        w.writerow(row)


def cycle_step(cfg, X, theta, acts, meta, gen, enc, base, rng):
    """One REINFORCE data collection: sample actions, decode, re-encode,
    score. Returns (A (S,b,d), ADV (S,b), stats dict)."""
    import jax.numpy as jnp
    sign = 1.0 if cfg.direction == "high" else -1.0
    b, d = X.shape
    S = cfg.n_samples
    mu = np.asarray(policy_mu(theta, jnp.asarray(X)))
    A = mu[None] + cfg.sigma * rng.standard_normal((S, b, d)).astype(
        np.float32)
    Xp = np.asarray(steer(jnp.asarray(np.repeat(X[None], S, 0)
                                      .reshape(S * b, d)),
                          jnp.asarray(A.reshape(S * b, d)), cfg.mag))
    steer_txt = gen(Xp)
    a_cyc = np.asarray(acts(jnp.asarray(enc(steer_txt))))[:, meta["f"]]
    eff = sign * (a_cyc - np.tile(base["a_cyc"], S)) / (base["q_hi"]
                                                        + DIR_EPS)
    coll = np.array([1 - chrf(base["txt"][i % b], s)
                     for i, s in enumerate(steer_txt)])
    r = (eff - cfg.lam * coll).reshape(S, b)
    adv = r - r.mean(0, keepdims=True)  # per-state baseline
    adv = adv / (adv.std() + DIR_EPS)
    hit = (a_cyc >= base["q_hit"]).mean() if cfg.direction == "high" \
        else (a_cyc < base["q_hit"]).mean()
    stats = {"reward": float(r.mean()), "eff": float(eff.mean()),
             "coll": float(coll.mean()), "hit": float(hit)}
    return A, adv.astype(np.float32), stats


def train(cfg):
    import jax.numpy as jnp
    import optax

    acts, _, meta = load_frozen(cfg)
    out = out_dir(cfg, meta)
    out.mkdir(parents=True, exist_ok=True)
    mm = np.load(cfg.cache, mmap_mode="r")
    rate, q_hi, q_hit = reference_quantiles(
        acts, mm, meta["f"], cfg.sub_rows, cfg.b_probe, meta["k"])
    print(f"target f={meta['f']} (l{cfg.layer} h{cfg.head} k{cfg.entry}): "
          f"fire rate {rate:.4f} q_hi {q_hi:.3f} q_hit {q_hit:.3f}")
    if rate == 0.0:
        raise SystemExit("target entry never fires on the reference slice; "
                         "pick a live entry")
    (out / "meta.json").write_text(json.dumps(
        {**meta, "layer": cfg.layer, "head": cfg.head, "entry": cfg.entry,
         "direction": cfg.direction, "reward": cfg.reward, "mag": cfg.mag,
         "lam": cfg.lam, "sigma": cfg.sigma, "fire_rate": rate,
         "q_hi": q_hi, "q_hit": q_hit, "seed": cfg.seed}, indent=2))

    d = mm.shape[1]
    theta = init_policy(cfg.seed, d, cfg.hidden)
    start = 0
    pol_path = out / "policy.npz"
    if pol_path.exists():
        dat = np.load(pol_path)
        theta = {k_: dat[k_] for k_ in ("W1", "b1", "W2", "b2")}
        start = int(dat["step"])
        print(f"resumed policy at step {start} (fresh optimizer state)")
    theta = {k_: jnp.asarray(v) for k_, v in theta.items()}
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(theta)

    rows = train_rows(cfg, mm.shape[0])
    rng = np.random.default_rng(cfg.seed + start)
    sign = 1.0 if cfg.direction == "high" else -1.0
    log = out / "rl_log.csv"
    cols = ["step", "reward", "eff", "coll", "hit"]

    if cfg.reward == "embed":
        update = make_embed_update(acts, meta["f"], sign, cfg.mag, cfg.lam,
                                   q_hi, tx)
        for s in range(start, cfg.steps):
            idx = np.sort(rng.choice(rows, cfg.b, replace=False))
            X = jnp.asarray(np.asarray(mm[idx], np.float32))
            theta, opt_state, r, eff, coll = update(theta, opt_state, X)
            log_row(log, [s + 1, f"{float(r):.4f}", f"{float(eff):.4f}",
                          f"{float(coll):.4f}", ""], cols)
            if (s + 1) % cfg.save_each == 0 or s + 1 == cfg.steps:
                np.savez(pol_path, step=s + 1,
                         **{k_: np.asarray(v) for k_, v in theta.items()})
                print(f"step {s+1}: reward {float(r):+.4f} "
                      f"eff {float(eff):+.4f} coll {float(coll):.4f}",
                      flush=True)
    else:
        gen, enc = torch_stack(cfg)
        reinforce = make_reinforce_update(cfg.sigma, tx)
        for s in range(start, cfg.steps):
            idx = np.sort(rng.choice(rows, cfg.b, replace=False))
            X = np.asarray(mm[idx], np.float32)
            base_txt = gen(X)
            a_base = np.asarray(acts(jnp.asarray(enc(base_txt))
                                     ))[:, meta["f"]]
            base = {"txt": base_txt, "a_cyc": a_base,
                    "q_hi": q_hi, "q_hit": q_hit}
            A, adv, st = cycle_step(cfg, X, theta, acts, meta, gen, enc,
                                    base, rng)
            theta, opt_state = reinforce(theta, opt_state, jnp.asarray(X),
                                         jnp.asarray(A), jnp.asarray(adv))
            log_row(log, [s + 1] + [f"{st[c]:.4f}" for c in cols[1:]], cols)
            if (s + 1) % cfg.save_each == 0 or s + 1 == cfg.steps:
                np.savez(pol_path, step=s + 1,
                         **{k_: np.asarray(v) for k_, v in theta.items()})
            print(f"step {s+1}: reward {st['reward']:+.4f} eff "
                  f"{st['eff']:+.4f} coll {st['coll']:.4f} hit "
                  f"{st['hit']:.3f}", flush=True)
    print(f"-> {log}\n-> {pol_path}")


# ---------- eval: policy vs paired steering vs random ----------

def evaluate(cfg):
    import jax.numpy as jnp

    acts, deltas, meta = load_frozen(cfg)
    out = out_dir(cfg, meta)
    pol_path = out / "policy.npz"
    if not pol_path.exists():
        raise SystemExit(f"no trained policy at {pol_path}; run train first")
    dat = np.load(pol_path)
    theta = {k_: jnp.asarray(dat[k_]) for k_ in ("W1", "b1", "W2", "b2")}
    # score under the TRAINING run's settings, not whatever the eval
    # invocation happened to pass; a magnitude/direction mismatch here
    # would silently unmatch the budgets the comparison rests on
    run_meta = json.loads((out / "meta.json").read_text())
    q_hi, q_hit = run_meta["q_hi"], run_meta["q_hit"]
    cfg.mag = run_meta["mag"]
    sign = 1.0 if run_meta["direction"] == "high" else -1.0

    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.n_eval:], np.float32)  # sae.py eval-tail rows
    gen, enc = torch_stack(cfg)
    base_txt = gen(X)
    a_base = np.asarray(acts(jnp.asarray(enc(base_txt))))[:, meta["f"]]

    rng = np.random.default_rng(cfg.seed)
    arms = {
        "policy": np.asarray(policy_mu(theta, jnp.asarray(X))),
        "paired_steering": deltas(X),
        "random": np.broadcast_to(
            unit(rng.normal(size=(1, X.shape[1])).astype(np.float32)),
            X.shape),
    }
    rows, log = [], open(out / "eval_texts.jsonl", "w")
    for arm, D in arms.items():
        Xp = np.asarray(steer(jnp.asarray(X), jnp.asarray(
            np.asarray(D, np.float32)), cfg.mag))
        steer_txt = gen(Xp)
        a_cyc = np.asarray(acts(jnp.asarray(enc(steer_txt))))[:, meta["f"]]
        eff, hit = effect_scores(a_base, a_cyc, q_hi, q_hit)
        eff = sign * eff
        coll = np.array([1 - chrf(a, b_) for a, b_ in
                         zip(base_txt, steer_txt)])
        rows.append([arm, cfg.mag, float(eff.mean()), float(hit.mean()),
                     float(coll.mean()), len(X)])
        for i in range(len(X)):
            log.write(json.dumps(
                {"arm": arm, "mag": cfg.mag, "base": base_txt[i],
                 "steered": steer_txt[i], "effect": float(eff[i]),
                 "collateral": float(coll[i])}, ensure_ascii=False) + "\n")
        print(f"{arm:>16}: eff {eff.mean():+.3f} hit {hit.mean():.3f} "
              f"coll {coll.mean():.3f}")
    log.close()
    with open(out / "compare.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "mag", "effect", "hit_rate", "collateral", "n"])
        w.writerows(rows)
    print(f"-> {out / 'compare.csv'}")


def main():
    cfg = parse_args()
    {"train": train, "eval": evaluate}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
