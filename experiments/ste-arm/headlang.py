"""Is a head a variable? Head-level localization of language identity.

The architecture's distinguishing commitment is that a head IS a
categorical variable: k entries, exhaustive and exclusive by
construction. An SAE has no such object and a plain VQ has one codebook,
so "a labeled variable lands in one head" is the claim neither baseline
can make. The embedding cache's `.langs.npy` sidecar gives one labeled
variable with ground truth, so the claim is directly testable.

Three readings, each against its own null:

  per-head NMI     normalized mutual information between a head's argmax
                   and the language label. A head that IS the language
                   variable scores near its ceiling. `nmi` normalizes by
                   the arithmetic mean, so with `I <= min(H(head),
                   H(lang))` that ceiling is
                   `2 min(H_h, H_l) / (H_h + H_l)`, below 1 whenever the
                   two entropies differ in EITHER direction: a k=32 head
                   caps at 0.875 against 86 languages because it holds
                   5.00 bits and the label carries 6.43. Scattered
                   language scores near the label-shuffled null.
  concentration    the share of the total head/language information held
                   by the best head. 1.0 means one head owns it.
  k-head probe     multinomial ridge on the concatenated one-hot codes of
                   the top-m heads by NMI, scored on held-out rows, next
                   to a ridge on the raw embedding as the ceiling any
                   code is trying to reach. `langprobe.py` asks this of
                   individual LATENTS; the point here is that the unit
                   is the head.

A lookup table over the joint argmax is reported for m <= 2 only: at
k=32 a three-head table has more cells than training rows, so its
accuracy falls for cell-starvation reasons rather than informational
ones, which is misleading next to the probe.

  uv run python experiments/ste-arm/headlang.py \\
      --model data/out/sonar/multilingual/ste_h76 --temperature 0.00015

Several --model paths are scored in one pass and printed together.
NOTE: Ontologizer checkpoints restore on GPU JAX only.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from pareto import load_onto
from ontologize.ontologizer import Ontologizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", required=True,
                   help="one or more Ontologizer checkpoint dirs")
    p.add_argument("--temperature", type=float, nargs="+", default=[0.00015],
                   help="per model, or one value for all")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--langs", default=None,
                   help="default: <cache>'s .langs.npy sidecar")
    p.add_argument("--train-rows", type=int, default=65536)
    p.add_argument("--test-rows", type=int, default=32768,
                   help="the cache tail sae.py holds out")
    p.add_argument("--ms", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--b", type=int, default=2048)
    p.add_argument("--ridge", type=float, default=1e-2)
    p.add_argument("--nulls", type=int, default=3,
                   help="label-shuffled draws for the NMI chance level")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/headlang")
    return p.parse_args()


# ---------- information ----------

def nmi(a: Int[np.ndarray, "n"], b: Int[np.ndarray, "n"], ka: int, kb: int
        ) -> Tuple[float, float, float]:
    """Symmetric normalized mutual information of two labelings.

    Returns (NMI, mutual information in bits, H(a) in bits)."""
    C = np.zeros((ka, kb))
    np.add.at(C, (a, b), 1.0)
    C /= C.sum()
    pa, pb = C.sum(1), C.sum(0)

    def H(p: Float[np.ndarray, "c"]) -> float:
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    nz = C > 0
    I = float((C[nz] * np.log2(C[nz] / np.outer(pa, pb)[nz])).sum())
    den = 0.5 * (H(pa) + H(pb))
    return I / den if den > 0 else 0.0, I, H(pa)


# ---------- probes ----------

def ridge_probe(Ftr: Float[np.ndarray, "n_tr f"], ytr: Int[np.ndarray, "n_tr"],
                Fte: Float[np.ndarray, "n_te f"], yte: Int[np.ndarray, "n_te"],
                n_cls: int, lam: float) -> float:
    """Closed-form multinomial ridge on one-hot targets; argmax accuracy.
    No sklearn in the environment, and the ranking is what matters."""
    Ftr = np.concatenate([Ftr, np.ones((len(Ftr), 1), Ftr.dtype)], 1)
    Fte = np.concatenate([Fte, np.ones((len(Fte), 1), Fte.dtype)], 1)
    Y = np.zeros((len(ytr), n_cls), np.float64)
    Y[np.arange(len(ytr)), ytr] = 1.0
    A = Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1])
    W = np.linalg.solve(A, Ftr.T @ Y)
    return float(((Fte @ W).argmax(1) == yte).mean())


def onehot_heads(A: Int[np.ndarray, "n heads_all"], heads: Sequence[int],
                 k: int) -> Float[np.ndarray, "n hk"]:
    """(n, len(heads) * k) one-hot code of the selected heads."""
    out = np.zeros((len(A), len(heads) * k), np.float64)
    for j, h in enumerate(heads):
        out[np.arange(len(A)), j * k + A[:, h]] = 1.0
    return out


def lookup_acc(Atr: Int[np.ndarray, "n_tr heads_all"],
               ytr: Int[np.ndarray, "n_tr"],
               Ate: Int[np.ndarray, "n_te heads_all"],
               yte: Int[np.ndarray, "n_te"],
               heads: Sequence[int], k: int) -> float:
    """Majority-label table over the joint argmax; only honest for the
    few-head case (a 3-head table has k^3 cells)."""
    key = lambda M: np.ravel_multi_index([M[:, h] for h in heads], [k] * len(heads))
    tab = {}
    for c, lab in zip(key(Atr), ytr):
        tab.setdefault(c, []).append(lab)
    tab = {c: np.bincount(v).argmax() for c, v in tab.items()}
    pred = np.array([tab.get(c, -1) for c in key(Ate)])
    return float((pred == yte).mean())


# ---------- model ----------

def codes(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d_in"],
          b: int) -> Tuple[Int[np.ndarray, "n heads_all"], Ontologizer, int]:
    """Per-sample argmax entry of every head, (n, l*h), the model it came
    from, and the step actually loaded."""
    model, params, step = load_onto(ckpt, step)

    def probe(module, X: Float[Array, "b d_in"]) -> Int[Array, "b heads_all"]:
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        A = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            A.append(jnp.argmax(P, -1))
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        return jnp.concatenate(A, -1)

    f = jax.jit(lambda p, x: model.apply(p, x, method=probe))
    out = [np.asarray(f({"params": params}, jnp.asarray(X[i:i + b])))
           for i in range(0, len(X), b)]
    return np.concatenate(out), model, step


def main() -> None:
    cfg = parse_args()
    Ts = (cfg.temperature * len(cfg.model))[:len(cfg.model)] \
        if len(cfg.temperature) == 1 else cfg.temperature
    assert len(Ts) == len(cfg.model), "give one temperature, or one per model"
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = Path(cfg.cache)
    langs_path = cfg.langs or str(cache.with_name(
        cache.name.replace(".npy", ".langs.npy")))
    n_tr, n_te = cfg.train_rows, cfg.test_rows
    mm = np.load(cache, mmap_mode="r")
    sl = slice(len(mm) - n_te - n_tr, len(mm))
    X = np.asarray(mm[sl], np.float32)
    langs, y = np.unique(np.load(langs_path)[sl], return_inverse=True)
    n_cls = len(langs)
    ytr, yte = y[:n_tr], y[n_tr:]
    rng = np.random.default_rng(cfg.seed)

    H_lang = -sum(p * np.log2(p) for p in np.bincount(yte) / len(yte) if p)
    print(f"{n_cls} languages, H(lang) = {H_lang:.2f} bits on {n_te} "
          f"held-out rows; {n_tr} probe-training rows")
    dense = ridge_probe(X[:n_tr].astype(np.float64), ytr,
                        X[n_tr:].astype(np.float64), yte, n_cls, cfg.ridge)
    print(f"ridge on the raw 1024-d embedding (the ceiling): {dense:.3f}   "
          f"chance {1/n_cls:.3f}\n")

    rows, summary = [], {"dense": dense, "chance": 1 / n_cls,
                         "H_lang": float(H_lang), "models": {}}
    for ckpt, T in zip(cfg.model, Ts):
        A, model, step = codes(ckpt, cfg.step, T, X, cfg.b)
        k, h, l = model.k, model.h, model.l
        Atr, Ate = A[:n_tr], A[n_tr:]
        S, I, Hh = [], [], []
        for j in range(A.shape[1]):
            s, i_, hh = nmi(Ate[:, j], yte, k, n_cls)
            S.append(s); I.append(i_); Hh.append(hh)
        S, I, Hh = np.array(S), np.array(I), np.array(Hh)
        null = np.mean([nmi(Ate[:, j], yte[rng.permutation(n_te)], k, n_cls)[0]
                        for _ in range(cfg.nulls) for j in range(0, len(S), 37)])
        order = np.argsort(-S)
        best = order[0]
        # NMI divides by the arithmetic mean of the two entropies and I is
        # capped by the smaller, so the attainable maximum is the ratio of
        # the min to the mean -- under 1 whichever entropy is larger
        ceil = 2 * min(Hh[best], H_lang) / max(Hh[best] + H_lang, 1e-9)
        name = Path(ckpt).name
        print(f"=== {name}  step {step}  ({l}x{h} heads, k={k}, T={T:g})")
        print(f"  per-head NMI: best {S.max():.3f} (layer {best // h} head "
              f"{best % h}, ceiling {ceil:.2f})  median {np.median(S):.3f}  "
              f"null {null:.3f}")
        print(f"  heads over 0.5/0.3/0.1: {(S > .5).sum()}/{(S > .3).sum()}"
              f"/{(S > .1).sum()}   best head holds "
              f"{I.max() / I.sum():.1%} of the total head/language information")
        print(f"  {'m':>3} {'probe acc':>10} {'lookup':>8}")
        accs = {}
        for m in cfg.ms:
            if m > A.shape[1]:
                continue
            hs = order[:m]
            a = ridge_probe(onehot_heads(Atr, hs, k), ytr,
                            onehot_heads(Ate, hs, k), yte, n_cls, cfg.ridge)
            accs[m] = a
            lk = f"{lookup_acc(Atr, ytr, Ate, yte, hs, k):8.3f}" if m <= 2 else "       -"
            print(f"  {m:>3} {a:10.3f} {lk}")
        summary["models"][name] = {
            "step": int(step), "T": T, "l": l, "h": h, "k": k,
            "nmi_best": float(S.max()), "nmi_median": float(np.median(S)),
            "nmi_null": float(null), "nmi_ceiling": float(ceil),
            "best_head": [int(best // h), int(best % h)],
            "info_share_best": float(I.max() / I.sum()),
            "probe_acc": accs}
        for j in np.argsort(-S)[:h]:
            rows.append({"model": name, "layer": int(j // h), "head": int(j % h),
                         "nmi": round(float(S[j]), 4),
                         "bits": round(float(I[j]), 4),
                         "head_entropy": round(float(Hh[j]), 4)})
        print()

    with open(out / "heads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
