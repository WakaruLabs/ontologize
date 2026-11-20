"""Which entries of a model move its outputs toward one language: a scan of
every (layer, head, entry), read by a language probe on the embedding, with
the strongest entries then decoded.

1. Probe. A class-balanced logistic probe (`langprobe.fit_logistic`) for
   --lang against every other language, fitted on the unit directions of
   --probe-rows cache rows just before the held-out tail. It is checked on
   the tail, by its AUROC on the inputs and on the model's reconstructions,
   and it is affine in the unit direction, so it runs inside the jitted
   forward.
2. Scan. Every entry is forced (`withArgs`) over the same --rows tail rows
   of other languages, split into halves A and B. An entry's effect is the
   mean probe logit of its outputs minus that of the unforced
   reconstruction of the same rows, and the share of rows the probe then
   calls --lang. Entries are chosen on A and read on B, so the largest of
   many effects is not mistaken for one: B's effect of A's top entries
   against B's effect of random entries, and the A-B correlation over all
   entries.
3. Input side. One forward over the tail gives each entry's usage among
   --lang rows and among the rest: an entry that --lang inputs choose need
   not be one that writes --lang.
4. Decodes. A's top --decode-top entries, the --decode-top entries --lang
   inputs choose most, and as many random entries are forced on B's rows
   and decoded with the decoder choosing its own output language (M2M100
   with no forced first token), beside the positive control (--lang tail
   rows, input and reconstruction) and B's unforced reconstruction: the
   share of decodes that start with the language's token and of letters in
   its script.

  uv run python experiments/entry-force/scan.py \\
      --model data/out/sonar/recipe_ladder/ste_h76_init01_med837_ctrl \\
      --cache data/sonar_embeddings/mc4_sent16.npy --lang my --device cuda

Outputs under --out: scan.csv (every entry), usage.csv (every entry's usage
by --lang rows and the rest), decodes.jsonl, summary.json.
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from force import LANG_TOKENS, SCRIPTS, load_rows, pick_rows, script_share  # noqa: E402

SCRIPT_OF = {"my": "mya"}


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Probability that a positive scores above a negative (Mann-Whitney)."""
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def affine(score, d: int):
    """(w, c) with score(U) = U @ w + c for an affine `score`, read off the
    function itself: c at the origin, w from the unit vectors."""
    c = float(np.asarray(score(np.zeros((1, d), np.float32)))[0])
    w = np.asarray(score(np.eye(d, dtype=np.float32))) - c
    return w.astype(np.float32), c


def log_odds(a: np.ndarray, n_a: int, b: np.ndarray, n_b: int) -> np.ndarray:
    """log2 of the usage ratio between two groups, Haldane-corrected."""
    return np.log2((a + 0.5) / (n_a + 1)) - np.log2((b + 0.5) / (n_b + 1))


def run_temperature(run: Path) -> float:
    """The temperature a run ended training at, from its log.jsonl."""
    last = None
    for line in open(run / "log.jsonl"):
        if "env_config" in line:
            rec = json.loads(line)
            last = rec["env_config"] if "env_config" in rec else rec
    hp = last.get("hyper", last)
    end = hp.get("temperature_end")
    return float(end if end is not None else hp["temperature"])


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Ontologizer run directory")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the run's final training temperature")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_sent16.npy")
    p.add_argument("--unit", action="store_true",
                   help="scale rows to unit norm (see force.load_rows)")
    p.add_argument("--eval-rows", type=int, default=32768)
    p.add_argument("--probe-rows", type=int, default=200_000,
                   help="rows before the tail the probe is fitted on")
    p.add_argument("--lang", default="my", help="mC4 code of the language")
    p.add_argument("--rows", type=int, default=256,
                   help="tail rows of other languages forced (halves A, B)")
    p.add_argument("--own-rows", type=int, default=64)
    p.add_argument("--decode-top", type=int, default=8)
    p.add_argument("--b", type=int, default=1024)
    p.add_argument("--b-decode", type=int, default=64)
    p.add_argument("--device", default="cpu", help="torch device (M2M100)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="default data/out/sonar/entry_force/scan_<run>_<step>_<lang>")
    return p.parse_args()


def main():
    import jax
    import jax.numpy as jnp
    import torch as t
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from langprobe import fit_logistic
    from pareto import load_onto
    from steerfid import load_steerable
    from textfid import SonarDecoder
    from ontologize.ontologizer import Ontologizer

    cfg = parse_args()
    run = Path(cfg.model)
    model, params, step = load_onto(str(run), cfg.step)
    l, h, k = model.l, model.h, model.k
    T = cfg.temperature if cfg.temperature is not None else run_temperature(run)
    out = Path(cfg.out) if cfg.out else \
        Path("data/out/sonar/entry_force") / f"scan_{run.name}_{step}_{cfg.lang}"
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    langs = np.load(Path(cfg.cache).with_suffix(".langs.npy"), mmap_mode="r")
    start = len(mm) - cfg.eval_rows
    d = mm.shape[1]

    def unit(A):
        return A / jnp.linalg.norm(A, axis=-1, keepdims=True)

    # 1. the probe, on unit directions
    lo = start - cfg.probe_rows
    Xp = load_rows(mm, slice(lo, start), True)
    yp = np.asarray(langs[lo:start]) == cfg.lang
    w, c = affine(fit_logistic(Xp, yp, seed=cfg.seed), d)
    del Xp
    wj = jnp.asarray(w)
    print(f"probe: {int(yp.sum())} {cfg.lang} rows of {len(yp)}", flush=True)

    def probe(Y):
        return unit(Y) @ wj + c

    def args_for(L=None, hi=None, ki=None):
        a = [{} for _ in range(l)]
        if L is not None:
            a[L] = {"h_set": hi[None], "k_set": ki[None]}
        return a

    def forward(X, a):
        Y, _, _, _ = model.apply({"params": params}, X, a, temperature=T,
                                 method=Ontologizer.withArgs)
        return Y

    unforced = jax.jit(lambda X: forward(X, args_for()))
    unforced_logit = jax.jit(lambda X: probe(forward(X, args_for())))
    forced = {L: jax.jit(lambda X, hi, ki, L=L: forward(X, args_for(L, hi, ki)))
              for L in range(l)}
    forced_logit = {L: jax.jit(lambda X, hi, ki, L=L:
                               probe(forward(X, args_for(L, hi, ki))))
                    for L in range(l)}
    acts, _, _, _, _ = load_steerable(
        str(run), argparse.Namespace(mode="dec", step=step, temperature=T))

    # 3 (and 1's check): one pass over the tail
    tail_lang = np.asarray(langs[start:])
    is_lang = tail_lang == cfg.lang
    cnt_lang = np.zeros((l, h, k))
    cnt_other = np.zeros((l, h, k))
    x_logit, r_logit = [], []
    eye = np.eye(k)
    for i in range(start, len(mm), cfg.b):
        X = jnp.asarray(load_rows(mm, slice(i, min(i + cfg.b, len(mm))), cfg.unit))
        x_logit.append(np.asarray(probe(X)))
        r_logit.append(np.asarray(unforced_logit(X)))
        A = np.asarray(acts(X)).reshape(len(X), l, h, k).argmax(-1)
        m = is_lang[i - start:i - start + len(X)]
        cnt_lang += eye[A[m]].sum(0)
        cnt_other += eye[A[~m]].sum(0)
    x_logit, r_logit = np.concatenate(x_logit), np.concatenate(r_logit)
    n_lang, n_other = int(is_lang.sum()), int((~is_lang).sum())
    lor = log_odds(cnt_lang, n_lang, cnt_other, n_other)
    check = {"auroc_input": auroc(x_logit, is_lang),
             "auroc_recon": auroc(r_logit, is_lang),
             "lang_recon_called": float((r_logit[is_lang] > 0).mean()),
             "other_recon_called": float((r_logit[~is_lang] > 0).mean()),
             "lang_recon_logit": float(r_logit[is_lang].mean()),
             "other_recon_logit": float(r_logit[~is_lang].mean())}
    print("probe check:", json.dumps(check), flush=True)
    with open(out / "usage.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["layer", "head", "entry", f"use_{cfg.lang}", "use_other",
                     "log2_ratio"])
        for L, H, K in np.ndindex(l, h, k):
            wr.writerow([L, H, K, f"{cnt_lang[L, H, K] / n_lang:.4f}",
                         f"{cnt_other[L, H, K] / n_other:.4f}",
                         f"{lor[L, H, K]:.3f}"])

    # 2. the scan
    rows = pick_rows(np.asarray(langs), start, cfg.rows, cfg.lang, cfg.seed)
    X = jnp.asarray(load_rows(mm, rows, cfg.unit))
    half = len(rows) // 2
    base = np.asarray(unforced_logit(X))
    dA = np.zeros((l, h, k))
    dB = np.zeros((l, h, k))
    callA = np.zeros((l, h, k))
    callB = np.zeros((l, h, k))
    for L in range(l):
        for H in range(h):
            for K in range(k):
                z = np.asarray(forced_logit[L](X, jnp.asarray(H), jnp.asarray(K)))
                dA[L, H, K] = (z[:half] - base[:half]).mean()
                dB[L, H, K] = (z[half:] - base[half:]).mean()
                callA[L, H, K] = (z[:half] > 0).mean()
                callB[L, H, K] = (z[half:] > 0).mean()
        print(f"scan: layer {L} done; max dA {dA[L].max():.3f}", flush=True)
    with open(out / "scan.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["layer", "head", "entry", "dA", "dB", "callA", "callB"])
        for L, H, K in np.ndindex(l, h, k):
            wr.writerow([L, H, K, f"{dA[L, H, K]:.4f}", f"{dB[L, H, K]:.4f}",
                         f"{callA[L, H, K]:.4f}", f"{callB[L, H, K]:.4f}"])

    flat = lambda a: a.reshape(-1)
    n_ent = l * h * k
    top = np.argsort(flat(dA))[::-1][:cfg.decode_top]
    rng = np.random.default_rng(cfg.seed)
    rand = rng.choice(np.setdiff1d(np.arange(n_ent), top), cfg.decode_top,
                      replace=False)
    chosen = np.argsort(flat(lor))[::-1][:cfg.decode_top]
    rand_all = rng.choice(n_ent, 1000, replace=False)
    sel = {"top_dB": float(flat(dB)[top].mean()),
           "random_dB_mean": float(flat(dB)[rand_all].mean()),
           "random_dB_sd": float(flat(dB)[rand_all].std()),
           "top_callB": float(flat(callB)[top].mean()),
           "random_callB": float(flat(callB)[rand_all].mean()),
           "base_callB": float((base[half:] > 0).mean()),
           "corr_AB": float(np.corrcoef(flat(dA), flat(dB))[0, 1]),
           "max_dA": float(flat(dA).max()), "max_dB": float(flat(dB).max())}
    print("selection:", json.dumps(sel), flush=True)

    # 4. decodes, the decoder choosing the language
    dec = SonarDecoder(cfg.device, cfg.b_decode)
    lang_tok = dec.tokenizer.convert_tokens_to_ids(LANG_TOKENS[SCRIPT_OF[cfg.lang]])
    ranges = SCRIPTS[SCRIPT_OF[cfg.lang]]

    def decode(Y):
        Y = np.asarray(Y)
        seqs = []
        with t.no_grad():
            for i in range(0, len(Y), dec.b):
                seqs += list(dec.dec.generate(
                    encoder_outputs=dec._enc(Y[i:i + dec.b]),
                    max_length=dec.max_length, num_beams=1,
                    repetition_penalty=1.2).cpu())
        return [dec.text(s) for s in seqs], np.array([int(s[1]) for s in seqs])

    own = pick_rows(np.asarray(langs), start, cfg.own_rows, "", cfg.seed,
                    cfg.lang)
    X_own = jnp.asarray(load_rows(mm, own, cfg.unit))
    XB = X[half:]
    jobs = [("own_x", None, X_own), ("own_recon", None, unforced(X_own)),
            ("recon", None, unforced(XB))]
    for group, idx in (("top", top), ("chosen", chosen), ("random", rand)):
        for f in idx:
            L, H, K = np.unravel_index(f, (l, h, k))
            jobs.append((group, (int(L), int(H), int(K)),
                         forced[L](XB, jnp.asarray(H), jnp.asarray(K))))
    decoded, log = [], open(out / "decodes.jsonl", "w")
    for group, ent, Y in jobs:
        texts, first = decode(Y)
        share = np.array([script_share(s, ranges) for s in texts])
        r = {"group": group, "entry": ent, "lang_tok": float((first == lang_tok).mean()),
             "share": float(share.mean()), "majority": float((share >= 0.5).mean())}
        if ent is not None:
            r.update(dA=float(dA[ent]), dB=float(dB[ent]),
                     log2_ratio=float(lor[ent]))
        decoded.append(r)
        print(json.dumps(r), flush=True)
        for s, f0, sh in zip(texts, first, share):
            log.write(json.dumps({"group": group, "entry": ent, "text": s,
                                  "lang_tok": dec.tokenizer.convert_ids_to_tokens(int(f0)),
                                  "share": round(float(sh), 4)},
                                 ensure_ascii=False) + "\n")
    log.close()

    meta = {"model": str(run), "step": int(step), "temperature": T,
            "cache": cfg.cache, "unit": cfg.unit, "lang": cfg.lang,
            "rows": len(rows), "probe_rows": cfg.probe_rows,
            "probe_lang_rows": int(yp.sum()), "tail_lang_rows": n_lang,
            "probe_check": check, "selection": sel, "decodes": decoded}
    (out / "summary.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
