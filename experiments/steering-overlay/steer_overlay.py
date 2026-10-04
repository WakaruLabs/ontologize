"""Steering-quality overlay: Ontologizer classification-forcing vs
supervised steering baselines, scored in steerfid.py's frame.

steerfid.py compares each model's native steering directions against
random directions. This harness adds the supervised upper-baseline the
steering claim actually has to beat: directions computed WITH labels.
The one free label set in the pipeline is language identity (the
.langs.npy sidecar encode_corpus.py writes next to the cache), so the
shared steering task is "move this embedding toward language L":

  onto    the model's own intervention interface: the tag (li,hi,ki)
          most associated with L (largest two-sample t of its activation
          vs the L label over a labeled cache slice, langprobe's
          statistic) is forced via withArgs, and the per-sample delta
          (forced minus unintervened decode, unit-normalized) is the
          steering direction -- exactly steerfid.load_steerable's deltas.
  dm      difference of means: unit(mean(X | L) - mean(X | not L)) on
          the same labeled slice. The classic supervised steering vector.
  probe   unit weight vector of a class-balanced logistic probe on the
          raw embeddings (langprobe.fit_logistic with the direction
          exposed), the discriminative counterpart of dm.
  random  unit random directions: steerfid's collateral-only control.

All arms steer the same held-out cache-tail rows (rows labeled L are
excluded per target) at the same matched magnitudes, decode through
M2M100 (forced eng_Latn, steerfid's convention), and are scored on:

  eff_native / hit_native   each arm's own activation frame, normalized
          exactly as steerfid does (onto: firing-conditional q90/q50 of
          the tag; dm/probe: the score's q90/q50 over rows OF language L,
          scaled by q90(L) - median(not L) since dense scores have no
          firing threshold).
  eff_dm / hit_dm           the COMMON frame: every arm's cycled decode
          scored on the dm direction for L, so arms are comparable on
          one yardstick regardless of what they steered along.
  collateral                1 - chrF(base decode, steered decode).
  langid (--langid)         an optional second, generation-grounded
          readout: decode WITHOUT forcing eng_Latn and count how often
          the decoder's own first language token is L's SONAR code.

NOTE: Ontologizer checkpoints restore on GPU JAX only; M2M100 runs on
--device. The cycle re-encodes forced-English decodes, so language
signal must survive translation to be seen by eff_* -- that attenuation
hits every arm identically (it is the shared task), and --langid is the
readout that does not route through forced English.

  uv run python experiments/steering-overlay/steer_overlay.py \\
      --model data/out/sonar/multilingual/resid_nc_hm --device cuda
  uv run python experiments/steering-overlay/steer_overlay.py \\
      --model data/out/sonar/sae/m5120_k32/params.npz --device cuda

Writes <out>/steer_overlay.csv, <out>/steers.jsonl, <out>/meta.json.
"""
# disable preallocation so this can share the GPU (same as sonar.py)
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

import steerfid                      # load_steerable, unit, quantile helpers
from textfid import chrf
from langprobe import t_stats
from ontologize.data.langs import MC4_TO_SONAR


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",
                   default="data/out/sonar/multilingual/resid_nc_hm",
                   help="onto checkpoint dir (resid_nc_hm / resid_nc) or "
                        "sae params.npz")
    p.add_argument("--mode", choices=["dec", "eig"], default="dec",
                   help="sae steering direction (passed to steerfid)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--langs", default=None,
                   help="default: <cache>.langs.npy sidecar")
    p.add_argument("--languages", nargs="*", default=None,
                   help="explicit mC4 codes (e.g. en fr de ru); default: "
                        "the --n-langs most frequent in the labeled slice")
    p.add_argument("--n-langs", type=int, default=8)
    p.add_argument("--train-rows", type=int, default=65536,
                   help="labeled cache-head rows for t-stats, means, probes")
    p.add_argument("--n-pool", type=int, default=16,
                   help="cache-tail rows decoded once as the steering pool")
    p.add_argument("--n-samples", type=int, default=8,
                   help="pool rows steered per (language, arm); rows labeled "
                        "with the target language are excluded first")
    p.add_argument("--mags", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--n-random", type=int, default=8)
    p.add_argument("--no-probe", action="store_true",
                   help="skip the logistic-probe arm")
    p.add_argument("--probe-steps", type=int, default=300)
    p.add_argument("--langid", action="store_true",
                   help="add the unforced-decode language-token readout "
                        "(one extra generation pass per steer batch)")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--b-decode", type=int, default=32)
    p.add_argument("--device", default="cpu",
                   help="torch device for SONAR/M2M100 (cpu protects training)")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-rate", type=float, default=1e-3,
                   help="dead-feature floor for the onto tag selection")
    p.add_argument("--out", default=None,
                   help="default experiments/steering-overlay/out/<model name>")
    return p.parse_args()


def logistic_direction(X, y, steps=300, lr=0.1):
    """langprobe.fit_logistic (class-balanced full-batch logistic
    regression, standardized inputs) with the weight vector exposed and
    folded back into RAW embedding space; returns the unit direction."""
    import jax
    import jax.numpy as jnp
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = jnp.asarray((X - mu) / sd)
    yj = jnp.asarray(y.astype(np.float32))
    pos = float(y.mean())
    wgt = jnp.where(yj > 0, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    w = jnp.zeros(X.shape[1])
    b = jnp.array(0.0)

    def loss(wb):
        w_, b_ = wb
        z = Xn @ w_ + b_
        return (wgt * (jnp.logaddexp(0.0, z) - yj * z)).mean()

    grad = jax.jit(jax.grad(loss))
    m = (jnp.zeros_like(w), jnp.array(0.0))
    v = (jnp.zeros_like(w), jnp.array(0.0))
    wb = (w, b)
    for i in range(steps):  # plain adam, matching langprobe.fit_logistic
        g = grad(wb)
        m = jax.tree_util.tree_map(lambda a, b_: 0.9 * a + 0.1 * b_, m, g)
        v = jax.tree_util.tree_map(lambda a, b_: 0.999 * a + 0.001 * b_ ** 2,
                                   v, g)
        t_ = i + 1
        wb = jax.tree_util.tree_map(
            lambda p, mi, vi: p - lr * (mi / (1 - 0.9 ** t_))
            / (jnp.sqrt(vi / (1 - 0.999 ** t_)) + 1e-8), wb, m, v)
    w_raw = np.asarray(wb[0]) / np.asarray(sd)
    return steerfid.unit(w_raw[None])[0]


def dense_frame(scores_pos, scores_neg):
    """Effect frame for a dense (non-firing) score: 1.0 = moved from a
    typical non-L row to a strong (q90) L row; hit = clears the median
    L-row score. The dense analogue of steerfid.firing_quantiles."""
    q_hi = float(np.quantile(scores_pos, 0.9))
    q_hit = float(np.quantile(scores_pos, 0.5))
    med_neg = float(np.median(scores_neg))
    return {"scale": max(q_hi - med_neg, 1e-9), "q_hit": q_hit}


def main():
    import jax.numpy as jnp
    import torch as t
    from transformers import M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from ontologize.data.pretrained import pretrained_transformer, encode
    from ontologize.data.loaders import TokenizeTransform

    cfg = parse_args()
    acts_fn, deltas_fn, F, thr, name = steerfid.load_steerable(cfg.model, cfg)
    out = Path(cfg.out) if cfg.out else \
        ROOT / "experiments/steering-overlay/out" / name
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    d = mm.shape[1]
    langs_path = cfg.langs or str(Path(cfg.cache).with_name(
        Path(cfg.cache).name.replace(".npy", ".langs.npy")))
    all_langs = np.load(langs_path, mmap_mode="r")
    assert len(all_langs) == len(mm), "langs sidecar length != cache rows"

    # ---- labeled head slice: supervised statistics ----
    n_tr = min(cfg.train_rows, mm.shape[0]) // cfg.b * cfg.b
    y_tr = np.asarray(all_langs[:n_tr])
    codes, y_idx = np.unique(y_tr, return_inverse=True)
    counts = np.bincount(y_idx)
    if cfg.languages:
        known = set(codes.tolist())
        targets = [c for c in cfg.languages if c in known]
        skipped = [c for c in cfg.languages if c not in known]
        if skipped:
            print(f"skipping languages absent from the slice: {skipped}")
    else:
        targets = [str(codes[i]) for i in np.argsort(-counts)[:cfg.n_langs]]
    assert targets, "no target languages"
    print(f"{name}: {F} features; targets {targets}")

    X_tr = np.asarray(mm[:n_tr], dtype=np.float32)
    L = len(codes)
    onehot = np.eye(L, dtype=np.float32)[y_idx]

    # one activation pass: per-language moments (t-stats), firing rates,
    # and an activation subsample for the onto quantile frame
    S = np.zeros((L, F))
    S2 = np.zeros((L, F))
    rate = np.zeros(F)
    subs = []
    stride = max(1, (n_tr // cfg.b) // 16)
    for bi, i in enumerate(range(0, n_tr, cfg.b)):
        A = np.asarray(acts_fn(jnp.asarray(X_tr[i:i + cfg.b])))
        O = onehot[i:i + cfg.b]
        S += np.asarray(O.T @ A, np.float64)
        S2 += np.asarray(O.T @ (A * A), np.float64)
        rate += (A > thr).mean(0)
        if bi % stride == 0:
            subs.append(A.astype(np.float16))
    rate /= n_tr // cfg.b
    sub = np.concatenate(subs).astype(np.float32)
    T = t_stats(S, S2, counts, n_tr)

    live = rate > cfg.min_rate
    assert live.any(), "no live features above --min-rate"
    lang_pos = {str(c): i for i, c in enumerate(codes)}
    feat_of, t_of = {}, {}
    for lang in targets:
        row = T[lang_pos[lang]].copy()
        row[~live] = -np.inf
        feat_of[lang] = int(np.argmax(row))
        t_of[lang] = float(row[feat_of[lang]])
    feats = np.array([feat_of[lang] for lang in targets])
    q_hi_o, q_hit_o = steerfid.firing_quantiles(sub[:, feats], thr)

    # supervised directions + dense score frames per target
    dirs_dm, dirs_pr, frames = {}, {}, {}
    for lang in targets:
        yb = y_tr == lang
        assert yb.sum() >= 32, f"language {lang}: too few labeled rows"
        mu1, mu0 = X_tr[yb].mean(0), X_tr[~yb].mean(0)
        dirs_dm[lang] = steerfid.unit((mu1 - mu0)[None])[0]
        s = X_tr @ dirs_dm[lang]
        frames[lang] = {"dm": dense_frame(s[yb], s[~yb])}
        if not cfg.no_probe:
            dirs_pr[lang] = logistic_direction(
                X_tr, yb.astype(np.float32), cfg.probe_steps)
            sp = X_tr @ dirs_pr[lang]
            frames[lang]["probe"] = dense_frame(sp[yb], sp[~yb])
    del X_tr, sub, subs

    # ---- torch: decoder for generation, encoder for the cycle ----
    dev = t.device(cfg.device)
    pt_enc, tokenizer = pretrained_transformer(
        steerfid.ENCODER_ID, "float32", dev=dev)
    refs = tokenizer(["The weather is nice today.",
                      "She walked to the store to buy some bread.",
                      "Scientists discovered a new species in the rainforest."],
                     return_tensors="pt", padding=True).to(dev)
    with t.no_grad():
        hh = pt_enc(**refs).last_hidden_state
        mask = refs["attention_mask"].unsqueeze(-1).float()
        ref_norm = t.norm((hh * mask).sum(1) / mask.sum(1), dim=-1).mean().item()
    dec = M2M100ForConditionalGeneration.from_pretrained(
        steerfid.DECODER_ID).to(dev)
    dec.eval()
    ENG = tokenizer.convert_tokens_to_ids("eng_Latn")
    tok = TokenizeTransform(tokenizer, maxlen=512)

    def generate(Y, forced=True):
        """Decode embeddings; forced=False lets the decoder pick its own
        language token (the --langid readout). Returns (texts, lang_tokens);
        position 1 of the generated sequence is the language token
        (position 0 is the decoder-start token)."""
        texts, lang_toks = [], []
        with t.no_grad():
            old = dec.generation_config.forced_bos_token_id
            dec.generation_config.forced_bos_token_id = ENG if forced else None
            try:
                for i in range(0, len(Y), cfg.b_decode):
                    Yb = t.from_numpy(np.ascontiguousarray(
                        Y[i:i + cfg.b_decode])).to(dev, t.float32)
                    Yb = t.nn.functional.normalize(Yb, dim=-1) * ref_norm
                    gen = dec.generate(
                        encoder_outputs=BaseModelOutput(
                            last_hidden_state=Yb.unsqueeze(1)),
                        max_length=48, num_beams=1, repetition_penalty=1.2)
                    for g in gen.cpu():
                        texts.append(tokenizer.decode(
                            g, skip_special_tokens=True).strip())
                        lang_toks.append(tokenizer.convert_ids_to_tokens(
                            int(g[1])) if len(g) > 1 else "")
            finally:
                dec.generation_config.forced_bos_token_id = old
        return texts, lang_toks

    def cycle_embed(texts):
        """Re-encode decodes with the SONAR encoder (steerfid's cycle)."""
        batch = [tok.map({"text": s, "lang": "en"}) for s in texts]
        batch = {k_: np.stack([b[k_] for b in batch])
                 for k_ in ("input_ids", "attention_mask")}
        return np.asarray(encode(pt_enc, batch, dev))

    # ---- steering pool: held-out cache tail ----
    pool_X = np.asarray(mm[-cfg.n_pool:], dtype=np.float32)
    pool_langs = np.asarray(all_langs[-cfg.n_pool:])
    base_txt, _ = generate(pool_X)
    E_base_cyc = cycle_embed(base_txt)
    A_base_cyc = np.asarray(acts_fn(jnp.asarray(E_base_cyc)))
    if cfg.langid:
        _, base_lang_toks = generate(pool_X, forced=False)
    rng = np.random.default_rng(cfg.seed)
    rand_dirs = steerfid.unit(
        rng.normal(size=(cfg.n_random, d)).astype(np.float32))

    rows = []
    log = open(out / "steers.jsonl", "w")

    def score_and_log(kind, lang, feature, m, sel, base_sel, steer_txt,
                      E_cyc, langid_hits):
        coll = np.array([1 - chrf(a, b)
                         for a, b in zip(base_sel, steer_txt)])
        fr = frames[lang]["dm"]
        s_st = E_cyc @ dirs_dm[lang]
        s_ba = E_base_cyc[sel] @ dirs_dm[lang]
        eff_dm = (s_st - s_ba) / fr["scale"]
        hit_dm = (s_st >= fr["q_hit"]).astype(np.float32)
        if kind == "onto":
            fi = int(np.where(feats == feature)[0][0])
            a_st = np.asarray(acts_fn(jnp.asarray(E_cyc)))[:, feature]
            eff_n, hit_n = steerfid.effect_scores(
                A_base_cyc[sel, feature], a_st, q_hi_o[fi], q_hit_o[fi])
        elif kind == "dm":
            eff_n, hit_n = eff_dm, hit_dm
        else:  # probe
            fp = frames[lang]["probe"]
            p_st = E_cyc @ dirs_pr[lang]
            p_ba = E_base_cyc[sel] @ dirs_pr[lang]
            eff_n = (p_st - p_ba) / fp["scale"]
            hit_n = (p_st >= fp["q_hit"]).astype(np.float32)
        lid_b = lid_s = float("nan")
        if cfg.langid:
            want = MC4_TO_SONAR.get(lang)
            lid_b = float(np.mean([base_lang_toks[i] == want for i in sel]))
            lid_s = float(np.mean([tk == want for tk in langid_hits]))
        rows.append([kind, lang, feature, m, float(np.mean(eff_n)),
                     float(np.mean(hit_n)), float(np.mean(eff_dm)),
                     float(np.mean(hit_dm)), float(coll.mean()),
                     lid_b, lid_s, len(sel)])
        for j, i in enumerate(sel):
            log.write(json.dumps(
                {"kind": kind, "lang": lang, "feature": feature, "mag": m,
                 "sample": int(i), "base": base_sel[j],
                 "steered": steer_txt[j], "collateral": float(coll[j]),
                 "eff_native": float(np.asarray(eff_n)[j]),
                 "eff_dm": float(eff_dm[j])}, ensure_ascii=False) + "\n")

    for lang in targets:
        sel = np.where(pool_langs != lang)[0][:cfg.n_samples]
        assert len(sel), f"pool has no non-{lang} rows; raise --n-pool"
        Xs = pool_X[sel]
        base_sel = [base_txt[i] for i in sel]
        arms = [("onto", feat_of[lang])]
        arms.append(("dm", -1))
        if not cfg.no_probe:
            arms.append(("probe", -1))
        for kind, f in arms:
            if kind == "onto":
                D = np.asarray(deltas_fn(Xs, f), dtype=np.float32)
            elif kind == "dm":
                D = np.broadcast_to(dirs_dm[lang], Xs.shape)
            else:
                D = np.broadcast_to(dirs_pr[lang], Xs.shape)
            for m in cfg.mags:
                Xp = (Xs + m * D).astype(np.float32)
                steer_txt, _ = generate(Xp)
                E_cyc = cycle_embed(steer_txt)
                lid = []
                if cfg.langid:
                    _, lid = generate(Xp, forced=False)
                score_and_log(kind, lang, f, m, sel, base_sel, steer_txt,
                              E_cyc, lid)
        print(f"{lang}: done (onto tag {feat_of[lang]}, t={t_of[lang]:.1f})",
              flush=True)

    # random-direction control: collateral only, over the whole pool
    for r in range(cfg.n_random):
        for m in cfg.mags:
            Xp = (pool_X + m * rand_dirs[r]).astype(np.float32)
            steer_txt, _ = generate(Xp)
            coll = np.array([1 - chrf(a, b)
                             for a, b in zip(base_txt, steer_txt)])
            rows.append(["random", "", r, m, float("nan"), float("nan"),
                         float("nan"), float("nan"), float(coll.mean()),
                         float("nan"), float("nan"), len(pool_X)])
            for j in range(len(pool_X)):
                log.write(json.dumps(
                    {"kind": "random", "lang": "", "feature": r, "mag": m,
                     "sample": j, "base": base_txt[j],
                     "steered": steer_txt[j], "collateral": float(coll[j])},
                    ensure_ascii=False) + "\n")
    log.close()

    cols = ["kind", "lang", "feature", "mag", "eff_native", "hit_native",
            "eff_dm", "hit_dm", "collateral", "langid_base", "langid_steer",
            "n"]
    with open(out / "steer_overlay.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    (out / "meta.json").write_text(json.dumps(
        {"model": str(cfg.model), "name": name, "targets": targets,
         "onto_tags": feat_of, "onto_t": t_of, "train_rows": n_tr,
         "mags": cfg.mags, "n_samples": cfg.n_samples,
         "langid": cfg.langid, "seed": cfg.seed}, indent=2))

    print(f"\n{'arm':<7} {'mag':>5} {'eff_nat':>8} {'hit_nat':>8} "
          f"{'eff_dm':>8} {'hit_dm':>8} {'collateral':>11}")
    arr = np.array([r[3:9] for r in rows], dtype=np.float64)
    kinds = np.array([r[0] for r in rows])
    for kind in ("onto", "dm", "probe", "random"):
        for m in cfg.mags:
            s = (kinds == kind) & (arr[:, 0] == m)
            if not s.any():
                continue
            print(f"{kind:<7} {m:>5.2f} "
                  f"{np.nanmean(arr[s, 1]):>8.3f} "
                  f"{np.nanmean(arr[s, 2]):>8.3f} "
                  f"{np.nanmean(arr[s, 3]):>8.3f} "
                  f"{np.nanmean(arr[s, 4]):>8.3f} "
                  f"{np.nanmean(arr[s, 5]):>11.3f}")
    print(f"-> {out / 'steer_overlay.csv'}")


if __name__ == "__main__":
    main()
