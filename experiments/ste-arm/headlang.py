"""Is a head a variable? Head-level localization of language identity.

The architecture's distinguishing commitment is that a head IS a
categorical variable: k entries, exhaustive and exclusive by
construction. An SAE has no such object and a plain VQ has one codebook,
so "a labeled variable lands in one head" is the claim neither baseline
can make. The embedding cache's `.langs.npy` sidecar gives one labeled
variable with ground truth, so the claim is directly testable.

Four readings, each against its own null:

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
  completeness     the same probe on sets built around C, the top
                   `--complete` heads: the next as many, the rest of
                   layer 0, as many random deeper heads, all deeper
                   heads, every head but C, and every head. With probe
                   accuracy in place of the model's behavior, C is
                   complete in IOI's sense (Wang et al. 2023, Sec. 4.1)
                   when every head but C reads little. A C that reads
                   most of what every head does, while the heads outside
                   it still read much of it, is faithful but not
                   complete: the variable is spread, not held by C.

Heads are ranked by NMI on the probe-training rows and every probe and
reported NMI is scored on the held-out rows, so the selection never
sees the rows it is scored on. The ridge penalty is chosen per probe
from `--ridge`: each value is fit on the first `1 - --val-share` of the
training rows and scored on the rest, and the best is refit on all of
them; a single value is used as given. A fixed small penalty overfits
once a probe reads hundreds of heads, which would make the large sets
score below the small ones. The probes solve the normal equations from
co-occurrence counts of the codes (`code_gram`), so a probe on every
head never forms its `n x (l*h*k)` design.

A lookup table over the joint argmax is reported for m <= 2 only: at
k=32 a three-head table has more cells than training rows, so its
accuracy falls for cell-starvation reasons rather than informational
ones, which is misleading next to the probe.

  uv run python experiments/ste-arm/headlang.py \\
      --model data/out/sonar/multilingual/ste_h76 --temperature 0.00015

Several --model paths are scored in one pass and printed together.
Writes heads.csv (each model's top-h heads by training-row NMI, with
their held-out NMI), completeness.csv and summary.json.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.linalg
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
    p.add_argument("--complete", type=int, default=32,
                   help="size of the top set C for the completeness rows "
                        "(0 skips them)")
    p.add_argument("--b", type=int, default=2048)
    p.add_argument("--ridge", type=float, nargs="+",
                   default=[1e-2, 1.0, 10.0, 100.0, 1000.0],
                   help="probe penalties to choose from on the validation "
                        "rows; one value is used as given")
    p.add_argument("--val-share", type=float, default=0.25,
                   help="share of the probe-training rows (the last ones) "
                        "that scores the penalties")
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


def code_gram(A: Int[np.ndarray, "n heads_all"], y: Int[np.ndarray, "n"],
              k: int, n_cls: int
              ) -> Tuple[Float[np.ndarray, "f f"], Float[np.ndarray, "f c"]]:
    """`F^T F` and `F^T Y` for `F` the one-hot code of every head with a
    constant column last (`f = heads_all * k + 1`, the column order of
    `onehot_heads`) and `Y` the one-hot label. Counted from the codes one
    head against the later ones at a time, so `F` is never formed; a
    probe on a subset of heads reads its rows and columns
    (`head_features`)."""
    A = np.asarray(A, np.int64)
    n, H = A.shape
    f = H * k + 1
    col = A + k * np.arange(H)
    G = np.zeros((f, f))
    for i in range(H):
        span = (H - i) * k
        c = np.bincount((A[:, i, None] * span + col[:, i:] - i * k).ravel(),
                        minlength=k * span).reshape(k, span)
        G[i * k:(i + 1) * k, i * k:H * k] = c
        G[i * k:H * k, i * k:(i + 1) * k] = c.T
    G[:-1, -1] = G[-1, :-1] = np.bincount(col.ravel(), minlength=H * k)
    G[-1, -1] = n
    B = np.zeros((f, n_cls))
    B[:-1] = np.bincount((col * n_cls + y[:, None]).ravel(),
                         minlength=H * k * n_cls).reshape(H * k, n_cls)
    B[-1] = np.bincount(y, minlength=n_cls)
    return G, B


def head_features(heads: Sequence[int], k: int, f: int) -> Int[np.ndarray, "g"]:
    """`code_gram` columns of `heads`, in their order, then the constant."""
    return np.concatenate([np.arange(h * k, (h + 1) * k) for h in heads]
                          + [np.array([f - 1])])


def ridge_solve(G: Float[np.ndarray, "f f"], B: Float[np.ndarray, "f c"],
                lam: float) -> Float[np.ndarray, "f c"]:
    """`(G + lam I)^{-1} B` by Cholesky: `G` is a Gram matrix, so
    `G + lam I` is positive definite for `lam > 0`."""
    return scipy.linalg.cho_solve(
        scipy.linalg.cho_factor(G + lam * np.eye(len(G)), overwrite_a=True,
                                check_finite=False), B, check_finite=False)


def code_scores(A: Int[np.ndarray, "n heads_all"], heads: Sequence[int],
                k: int, W: Float[np.ndarray, "g c"]) -> Float[np.ndarray, "n c"]:
    """`F W` for `F` the one-hot code of `heads` with the constant last,
    as a sum of gathered rows of `W`."""
    S = np.repeat(W[-1:], len(A), 0)
    for j, h in enumerate(heads):
        S += W[j * k + A[:, h]]
    return S


def choose_ridge(G: Float[np.ndarray, "f f"], B: Float[np.ndarray, "f c"],
                 val_acc: Callable[[Float[np.ndarray, "f c"]], float],
                 lams: Sequence[float]) -> float:
    """The penalty whose fit on `(G, B)` scores best under `val_acc`, the
    first on ties; a single penalty is returned without fitting."""
    if len(lams) == 1:
        return lams[0]
    acc = {lam: val_acc(ridge_solve(G, B, lam)) for lam in lams}
    return max(acc, key=acc.get)


def code_probe(heads: Sequence[int], k: int,
               fit: Tuple[Float[np.ndarray, "f f"], Float[np.ndarray, "f c"]],
               full: Tuple[Float[np.ndarray, "f f"], Float[np.ndarray, "f c"]],
               Aval: Int[np.ndarray, "n_val heads_all"],
               yval: Int[np.ndarray, "n_val"],
               Ate: Int[np.ndarray, "n_te heads_all"],
               yte: Int[np.ndarray, "n_te"],
               lams: Sequence[float]) -> Tuple[float, float]:
    """Held-out accuracy of a multinomial ridge on the one-hot codes of
    `heads`, and its penalty: chosen on the validation rows from a fit on
    `fit`'s rows (`code_gram` of the training rows before them), then
    refit on `full`'s (all training rows). `fit` is unused for a single
    penalty."""
    cols = head_features(heads, k, len(full[0]))
    acc = lambda W, A, y: float((code_scores(A, heads, k, W).argmax(1)
                                 == y).mean())
    lam = lams[0] if len(lams) == 1 else choose_ridge(
        fit[0][np.ix_(cols, cols)], fit[1][cols],
        lambda W: acc(W, Aval, yval), lams)
    W = ridge_solve(full[0][np.ix_(cols, cols)], full[1][cols], lam)
    return acc(W, Ate, yte), lam


def dense_probe(Xtr: Float[np.ndarray, "n_tr d"], ytr: Int[np.ndarray, "n_tr"],
                Xte: Float[np.ndarray, "n_te d"], yte: Int[np.ndarray, "n_te"],
                n_cls: int, lams: Sequence[float], val_share: float
                ) -> Tuple[float, float]:
    """`ridge_probe` with the penalty chosen as `code_probe` chooses it,
    on the last `val_share` of the training rows; returns (accuracy,
    penalty)."""
    def gram(X: Float[np.ndarray, "n d"], y: Int[np.ndarray, "n"]
             ) -> Tuple[Float[np.ndarray, "f f"], Float[np.ndarray, "f c"]]:
        F = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
        return F.T @ F, F.T @ np.eye(n_cls)[y]

    def acc(W: Float[np.ndarray, "f c"], X: Float[np.ndarray, "n d"],
            y: Int[np.ndarray, "n"]) -> float:
        return float(((X @ W[:-1] + W[-1]).argmax(1) == y).mean())

    cut = len(Xtr) - int(round(val_share * len(Xtr)))
    lam = lams[0] if len(lams) == 1 else choose_ridge(
        *gram(Xtr[:cut], ytr[:cut]), lambda W: acc(W, Xtr[cut:], ytr[cut:]),
        lams)
    return acc(ridge_solve(*gram(Xtr, ytr), lam), Xte, yte), lam


def completeness_sets(order: Int[np.ndarray, "heads_all"],
                      layer: Int[np.ndarray, "heads_all"], m: int,
                      rng: np.random.Generator) -> Dict[str, Int[np.ndarray, "_"]]:
    """The completeness rows' head sets around C, the first `m` heads of
    `order` (all heads, best first): the next `m`, the rest of layer 0,
    `m` random deeper heads, all deeper heads, every head but C, every
    head. Empty sets, and sets equal to an earlier one (the rest of layer
    0 is every head but C in a one-layer model), are left out."""
    C, rest = order[:m], order[m:]
    deep = np.flatnonzero(layer > 0)
    l_max = int(layer.max())
    n_rand = min(m, len(deep))
    cand = {
        f"C: top {m} by NMI": C,
        f"ranks {m + 1}-{2 * m}": order[m:2 * m],
        "layer 0 minus C": rest[layer[rest] == 0],
        f"{n_rand} random deeper heads":
            rng.choice(deep, n_rand, replace=False) if n_rand else deep,
        (f"layer {l_max}" if l_max == 1 else f"layers 1-{l_max}"): deep,
        "all heads minus C": rest,
        "all heads": np.arange(len(order)),
    }
    sets, seen = {}, set()
    for label, hs in cand.items():
        key = tuple(sorted(hs.tolist()))
        if len(hs) and key not in seen:
            sets[label] = hs
            seen.add(key)
    return sets


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
    dense, dense_lam = dense_probe(
        X[:n_tr].astype(np.float64), ytr, X[n_tr:].astype(np.float64), yte,
        n_cls, cfg.ridge, cfg.val_share)
    print(f"ridge on the raw {X.shape[1]}-d embedding (the ceiling): "
          f"{dense:.3f} (ridge {dense_lam:g})   chance {1/n_cls:.3f}\n")

    rows, comp_rows = [], []
    summary = {"dense": dense, "dense_ridge": dense_lam, "chance": 1 / n_cls,
               "H_lang": float(H_lang), "models": {}}
    cut = n_tr - int(round(cfg.val_share * n_tr))
    for ckpt, T in zip(cfg.model, Ts):
        A, model, step = codes(ckpt, cfg.step, T, X, cfg.b)
        k, h, l = model.k, model.h, model.l
        Atr, Ate = A[:n_tr], A[n_tr:]
        layer = np.arange(A.shape[1]) // h
        S_tr = np.array([nmi(Atr[:, j], ytr, k, n_cls)[0]
                         for j in range(A.shape[1])])
        S, I, Hh = [], [], []
        for j in range(A.shape[1]):
            s, i_, hh = nmi(Ate[:, j], yte, k, n_cls)
            S.append(s); I.append(i_); Hh.append(hh)
        S, I, Hh = np.array(S), np.array(I), np.array(Hh)
        null = np.mean([nmi(Ate[:, j], yte[rng.permutation(n_te)], k, n_cls)[0]
                        for _ in range(cfg.nulls) for j in range(0, len(S), 37)])
        order = np.argsort(-S_tr, kind="stable")
        best = order[0]
        # NMI divides by the arithmetic mean of the two entropies and I is
        # capped by the smaller, so the attainable maximum is the ratio of
        # the min to the mean -- under 1 whichever entropy is larger
        ceil = 2 * min(Hh[best], H_lang) / max(Hh[best] + H_lang, 1e-9)
        name = Path(ckpt).name
        print(f"=== {name}  step {step}  ({l}x{h} heads, k={k}, T={T:g})")
        print(f"  per-head NMI: best {S[best]:.3f} (layer {best // h} head "
              f"{best % h}, ceiling {ceil:.2f})  median {np.median(S):.3f}  "
              f"null {null:.3f}")
        print(f"  heads over 0.5/0.3/0.1: {(S > .5).sum()}/{(S > .3).sum()}"
              f"/{(S > .1).sum()}   best head holds "
              f"{I[best] / I.sum():.1%} of the total head/language information")
        by_layer, first_deep = np.bincount(layer[order[:h]], minlength=l), None
        if l > 1:
            first_deep = int(np.argmax(layer[order] > 0)) + 1
            print(f"  top {h} heads by layer {by_layer.tolist()}, first head "
                  f"past layer 0 at rank {first_deep}; held-out NMI of "
                  f"layer 0 >= {S[layer == 0].min():.3f}, deeper heads <= "
                  f"{S[layer > 0].max():.3f}")

        if len(cfg.ridge) > 1:
            fit = code_gram(Atr[:cut], ytr[:cut], k, n_cls)
            G, B = code_gram(Atr[cut:], ytr[cut:], k, n_cls)
            G += fit[0]
            B += fit[1]
            full = (G, B)
        else:
            fit, full = None, code_gram(Atr, ytr, k, n_cls)

        def probe(hs: Sequence[int]) -> Tuple[float, float]:
            return code_probe(hs, k, fit, full, Atr[cut:], ytr[cut:], Ate, yte,
                              cfg.ridge)

        print(f"  {'m':>3} {'probe acc':>10} {'ridge':>7} {'lookup':>8}")
        accs, lams = {}, {}
        for m in cfg.ms:
            if m > A.shape[1]:
                continue
            hs = order[:m]
            accs[m], lams[m] = probe(hs)
            lk = f"{lookup_acc(Atr, ytr, Ate, yte, hs, k):8.3f}" if m <= 2 else "       -"
            print(f"  {m:>3} {accs[m]:10.3f} {lams[m]:7g} {lk}")
        comp = {}
        if 0 < cfg.complete < A.shape[1]:
            C = order[:cfg.complete]
            print(f"  completeness, C = top {cfg.complete} by NMI (by layer "
                  f"{np.bincount(layer[C], minlength=l).tolist()}):")
            for label, hs in completeness_sets(order, layer, cfg.complete,
                                               rng).items():
                a, lam = probe(hs)
                comp[label] = {"heads": int(len(hs)), "acc": a, "ridge": lam}
                comp_rows.append({"model": name, "set": label,
                                  "heads": int(len(hs)), "acc": round(a, 4),
                                  "ridge": lam})
                print(f"    {label:<26} {len(hs):>4} heads   probe acc "
                      f"{a:.3f}  (ridge {lam:g})")
        summary["models"][name] = {
            "step": int(step), "T": T, "l": l, "h": h, "k": k,
            "nmi_best": float(S[best]), "nmi_best_train": float(S_tr[best]),
            "nmi_median": float(np.median(S)),
            "nmi_null": float(null), "nmi_ceiling": float(ceil),
            "best_head": [int(best // h), int(best % h)],
            "info_share_best": float(I[best] / I.sum()),
            "top_h_by_layer": by_layer.tolist(),
            "first_rank_past_layer0": first_deep,
            "probe_acc": accs, "probe_ridge": lams, "completeness": comp}
        for j in order[:h]:
            rows.append({"model": name, "layer": int(j // h), "head": int(j % h),
                         "nmi": round(float(S[j]), 4),
                         "bits": round(float(I[j]), 4),
                         "head_entropy": round(float(Hh[j]), 4),
                         "nmi_train": round(float(S_tr[j]), 4)})
        fit = full = None
        print()

    with open(out / "heads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    if comp_rows:
        with open(out / "completeness.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comp_rows[0]))
            w.writeheader(); w.writerows(comp_rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
