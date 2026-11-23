"""Is one seed's code a re-factoring of another's?

`partition.py` matches heads one to one across seeds and finds almost
nothing past layer 0. That leaves open a weaker kind of agreement: seed
B's code could carry the same information as seed A's, split between
heads differently, so that no head of B matches a head of A while B's
whole layer still determines it. The test is a translation: predict each
head of A from B's code and score it on rows neither model trained on.

A translation score means nothing alone. Both codes are functions of the
same input, so any code with enough capacity predicts A's heads to a
degree. As with a probe and its control task, the score is read as a
margin over the same probe on a code that training did not shape:

  seed       B's code
  lsh        random-projection heads with the shape of B's code (each
             head the argmax of k random directions on the centered
             input)
  untrained  B's architecture at its initialization
  earlier    A's own code at an earlier checkpoint (--earlier): a code
             that is known to be the same model, as a ceiling
  input      the input itself, by a linear probe: how much of a head a
             linear map can read at all

Each code predicts every head of A by one multinomial ridge on its
one-hot features (closed form; the penalty is chosen on validation rows),
from the same layer's heads alone (`same`) and from the whole stack's
(`all`). A head's score is its held-out accuracy; `majority` is the rate
of its most used entry.

  uv run python experiments/seed-probes/translate.py \\
      --a data/out/sonar/multilingual/ste_h76_init01 \\
      --b data/out/sonar/multilingual/ste_h76_i01_s43 --earlier 350000
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import functools
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "ste-arm"))

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import optax
from jaxtyping import Array, Float, Int

HIGHEST = jax.lax.Precision.HIGHEST

from pareto import load_onto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="checkpoint dir: the targets")
    p.add_argument("--b", required=True, help="checkpoint dir: the source")
    p.add_argument("--step-a", type=int, default=0)
    p.add_argument("--step-b", type=int, default=0)
    p.add_argument("--earlier", type=int, default=0,
                   help="an earlier step of A, as a ceiling (0 = skip)")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_fresh.npy",
                   help="rows no model trained on")
    p.add_argument("--val", type=int, default=16384)
    p.add_argument("--test", type=int, default=32768,
                   help="the last TEST rows; VAL rows before them choose "
                        "the penalty; the rest fit")
    p.add_argument("--lams", type=float, nargs="+",
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                   help="ridge penalties, as shares of the feature Gram's "
                        "mean diagonal")
    p.add_argument("--mlp", action="store_true",
                   help="also a one-hidden-layer probe from each whole "
                        "code, which can read a head that is a joint "
                        "function of several source heads")
    p.add_argument("--mlp-hidden", type=int, default=2048)
    p.add_argument("--mlp-epochs", type=int, default=12)
    p.add_argument("--dict-init-scale", type=float, default=0.1,
                   help="the untrained model's initialization "
                        "(train_ste.py's flag)")
    p.add_argument("--init-cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ---------- codes ----------

def codes_of(model, params, X: Float[np.ndarray, "n d_in"], T: float,
             b: int) -> Int[np.ndarray, "n heads"]:
    """Every head's selected entry, (n, l*h), layer by layer."""
    def probe(module, Xb: Float[Array, "b d_in"]) -> Int[Array, "b heads"]:
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        A = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            P = de.dict.cluster(de.classifier(U), T)
            A.append(jnp.argmax(P, -1))
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, P.reshape(Xb.shape[0], -1))
        return jnp.concatenate(A, -1)

    f = jax.jit(lambda p, x: model.apply(p, x, method=probe))
    return np.concatenate([np.asarray(f(params, jnp.asarray(X[i:i + b])))
                           for i in range(0, len(X), b)])


def lsh_codes(X: Float[np.ndarray, "n d"], heads: int, k: int, seed: int,
              b: int) -> Int[np.ndarray, "n heads"]:
    """Random-projection heads: each the argmax of `k` random unit
    directions on the input, centered on `X`'s mean."""
    rng = np.random.default_rng([seed, 7])
    W = rng.standard_normal((heads, k, X.shape[1])).astype(np.float32)
    W /= np.linalg.norm(W, axis=-1, keepdims=True)
    Wj, mu = jnp.asarray(W), jnp.asarray(X.mean(0, keepdims=True))
    f = jax.jit(lambda x: jnp.argmax(
        jnp.einsum("bd,hkd->bhk", x - mu, Wj), -1))
    return np.concatenate([np.asarray(f(jnp.asarray(X[i:i + b])))
                           for i in range(0, len(X), b)])


# ---------- the probe (pure) ----------

def x64(f: Callable) -> Callable:
    """Runs `f` with JAX's 64-bit types on, for its float64 sums and
    solves; outside them a float64 array would turn float32 again. Only
    these functions turn them on, so the models that make the codes run as
    they were trained."""
    @functools.wraps(f)
    def run(*args, **kwargs):
        with jax.enable_x64(True):
            return f(*args, **kwargs)
    return run


def padded(A: Array, rows: int, fill: float = 0) -> Array:
    """`A` with `fill` rows appended up to `rows`, so that a loop's last,
    shorter chunk has the others' shape."""
    short = rows - A.shape[0]
    if short == 0:
        return jnp.asarray(A)
    return jnp.concatenate([jnp.asarray(A),
                            jnp.full((short,) + A.shape[1:], fill, A.dtype)])


def onehot(S: Int[Array, "n heads"], k: int, const: bool
           ) -> Float[Array, "n f"]:
    """Dense one-hot code, head by head, with a constant column last when
    `const`."""
    n, H = S.shape
    F = jax.nn.one_hot(S, k, dtype=jnp.float32).reshape(n, H * k)
    if const:
        F = jnp.concatenate([F, jnp.ones((n, 1), jnp.float32)], 1)
    return F


@jax.jit
def products(F: Float[Array, "c f"], Y: Float[Array, "c t"]
             ) -> Tuple[Float[Array, "f f"], Float[Array, "f t"]]:
    """A chunk's `F'F` and `F'Y`, summed in float32 and returned in
    float64."""
    return (jnp.matmul(F.T, F, precision=HIGHEST).astype(jnp.float64),
            jnp.matmul(F.T, Y, precision=HIGHEST).astype(jnp.float64))


@x64
def gram(F_of, Y_of, n: int, chunk: int
         ) -> Tuple[Float[Array, "f f"], Float[Array, "f c"]]:
    """`F'F` and `F'Y` in float64, accumulated over row chunks; `F_of` and
    `Y_of` return a chunk's feature and target matrices. The last chunk is
    padded with zero rows, which add nothing."""
    G = B = None
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        g, b = products(padded(F_of(i, j), chunk), padded(Y_of(i, j), chunk))
        G, B = (g, b) if G is None else (G + g, B + b)
    return G, B


@x64
def ridge(G: Float[Array, "f f"], B: Float[Array, "f c"],
          lam: float) -> Float[Array, "f c"]:
    """`(G + lam s I)^{-1} B` by Cholesky, in float64, with `s` the mean
    diagonal of `G`: the penalty is relative, so one grid serves one-hot
    codes and the dense input alike."""
    G, B = jnp.asarray(G, jnp.float64), jnp.asarray(B, jnp.float64)
    A = G + lam * jnp.diagonal(G).mean() * jnp.eye(len(G), dtype=G.dtype)
    return jsl.cho_solve((jnp.linalg.cholesky(A), True), B)


@jax.jit
def hits(F: Float[Array, "c f"], W: Float[Array, "f hk"],
         T: Int[Array, "c heads"]) -> Int[Array, "heads"]:
    """Per head, the chunk's rows whose highest-scoring entry is `T`'s."""
    S = jnp.matmul(F, W, precision=HIGHEST).reshape(F.shape[0], T.shape[1], -1)
    return (S.argmax(-1) == T).sum(0)


@x64
def head_accuracy(F_of, W: Float[Array, "f c"],
                  T: Int[Array, "n heads"], k: int, lo: int, hi: int,
                  chunk: int) -> Float[np.ndarray, "heads"]:
    """Per target head, the share of rows `lo:hi` whose predicted entry
    (the argmax of the head's `k` score columns) is the one selected. The
    last chunk's padding rows target entry -1, which no argmax is."""
    W = jnp.asarray(W, jnp.float32)
    hit = np.zeros(T.shape[1], np.int64)
    for i in range(lo, hi, chunk):
        j = min(i + chunk, hi)
        hit += np.asarray(hits(padded(F_of(i, j), chunk), W,
                               padded(T[i:j], chunk, -1)))
    return hit / (hi - lo)


@x64
def probe(F_of, T: Int[Array, "n heads"], k: int, n_fit: int,
          n_val: int, n: int, lams: Sequence[float], chunk: int
          ) -> Tuple[Float[np.ndarray, "heads"], float]:
    """Held-out accuracy on every target head of a ridge from the features
    `F_of` returns, and its penalty: chosen on rows `n_fit:n_fit+n_val`
    by mean accuracy from a fit on the rows before them, then refit on
    both. Rows from `n_fit + n_val` on are the test rows."""
    Y_of = lambda i, j: onehot(T[i:j], k, False)
    G, B = gram(F_of, Y_of, n_fit, chunk)
    lam = lams[0]
    if len(lams) > 1:
        acc = {lam: float(head_accuracy(F_of, ridge(G, B, lam), T, k, n_fit,
                                        n_fit + n_val, chunk).mean())
               for lam in lams}
        lam = max(acc, key=acc.get)
    Gv, Bv = gram(lambda i, j: F_of(n_fit + i, n_fit + j),
                  lambda i, j: Y_of(n_fit + i, n_fit + j), n_val, chunk)
    W = ridge(G + Gv, B + Bv, lam)
    return head_accuracy(F_of, W, T, k, n_fit + n_val, n, chunk), lam


Layers = List[Tuple[Float[Array, "a b"], Float[Array, "b"]]]


def mlp_init(key: Array, sizes: Sequence[int]) -> Layers:
    """Each layer's weights and biases, uniform on +-1/sqrt(fan_in)."""
    layers = []
    for key, a, b in zip(jax.random.split(key, len(sizes) - 1), sizes[:-1],
                         sizes[1:]):
        kw, kb = jax.random.split(key)
        s = 1.0 / np.sqrt(a)
        layers.append((jax.random.uniform(kw, (a, b), jnp.float32, -s, s),
                       jax.random.uniform(kb, (b,), jnp.float32, -s, s)))
    return layers


def mlp(layers: Layers, x: Float[Array, "b f"]) -> Float[Array, "b o"]:
    """ReLU between the layers, none after the last."""
    for i, (W, c) in enumerate(layers):
        x = jnp.matmul(x, W, precision=HIGHEST) + c
        if i < len(layers) - 1:
            x = jax.nn.relu(x)
    return x


def mlp_probe(F_of, T: Int[Array, "n heads"], k: int, n_fit: int,
              n_val: int, n: int, hidden: int, epochs: int, seed: int,
              batch: int = 1024) -> Tuple[Float[np.ndarray, "heads"], int]:
    """Held-out accuracy on every target head of one network with a
    hidden layer, trained by Adam on the mean cross-entropy over heads, and
    the epoch it was read at: the one with the best mean validation
    accuracy. Unlike `probe`, it can read a head that is a joint function
    of several source heads. Rows are split as in `probe`. Batches are
    padded to `batch` rows, which the loss weighs at zero."""
    H = T.shape[1]
    layers = mlp_init(jax.random.PRNGKey(seed), [F_of(0, 1).shape[1], hidden, H * k])
    opt = optax.adam(1e-3)
    state = opt.init(layers)

    def loss(p: Layers, x, t, m) -> Float[Array, ""]:
        ce = optax.softmax_cross_entropy_with_integer_labels(
            mlp(p, x).reshape(x.shape[0], H, k), t)
        return (ce.mean(1) * m).sum() / m.sum()

    @jax.jit
    def step(p: Layers, s, x, t, m):
        u, s = opt.update(jax.grad(loss)(p, x, t, m), s, p)
        return optax.apply_updates(p, u), s

    @jax.jit
    def hit(p: Layers, x, t) -> Int[Array, "heads"]:
        return (mlp(p, x).reshape(x.shape[0], H, k).argmax(-1) == t).sum(0)

    def acc(p: Layers, lo: int, hi: int) -> Float[np.ndarray, "heads"]:
        out = np.zeros(H, np.int64)
        for i in range(lo, hi, 4 * batch):
            j = min(i + 4 * batch, hi)
            out += np.asarray(hit(p, padded(F_of(i, j), 4 * batch),
                                  padded(T[i:j], 4 * batch, -1)))
        return out / (hi - lo)

    best, at, test = -1.0, 0, None
    starts = np.arange(0, n_fit, batch)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        for i in rng.permutation(starts):
            i = int(i)
            j = min(i + batch, n_fit)
            m = padded(jnp.ones(j - i, jnp.float32), batch)
            layers, state = step(layers, state, padded(F_of(i, j), batch),
                                 padded(T[i:j], batch, 0), m)
        v = float(acc(layers, n_fit, n_fit + n_val).mean())
        if v > best:
            best, at, test = v, ep + 1, acc(layers, n_fit + n_val, n)
    return test, at


def majority(T: Int[np.ndarray, "n heads"], k: int
             ) -> Float[np.ndarray, "heads"]:
    """Each head's most used entry's share of the rows."""
    return np.array([np.bincount(c, minlength=k).max() / len(c)
                     for c in T.T])


# ---------- the experiment ----------

def main() -> None:
    cfg = parse_args()
    X = np.asarray(np.load(cfg.cache, mmap_mode="r"), np.float32)
    n = len(X)
    n_fit = n - cfg.val - cfg.test
    T = cfg.temperature

    ma, pa, sa = load_onto(cfg.a, cfg.step_a)
    mb, pb, sb = load_onto(cfg.b, cfg.step_b)
    l, h, k = ma.l, ma.h, ma.k
    if (mb.l, mb.h, mb.k) != (l, h, k):
        raise ValueError("A and B differ in shape")
    A = codes_of(ma, {"params": pa}, X, T, cfg.batch)
    src: Dict[str, Int[np.ndarray, "n heads"]] = {
        "seed": codes_of(mb, {"params": pb}, X, T, cfg.batch),
        "lsh": lsh_codes(X, l * h, k, cfg.seed, cfg.batch)}
    from steerembed import init_onto
    mi, pi = init_onto(cfg.b, cfg.seed + 1000, T, cfg.dict_init_scale,
                       cfg.init_cache)
    src["untrained"] = codes_of(mi, pi, X, T, cfg.batch)
    if cfg.earlier:
        me, pe, se = load_onto(cfg.a, cfg.earlier)
        src["earlier"] = codes_of(me, {"params": pe}, X, T, cfg.batch)

    out = Path(cfg.out or f"data/out/sonar/seedprobes/translate_"
                          f"{Path(cfg.a).name}__{Path(cfg.b).name}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"A {Path(cfg.a).name} step {sa}; B {Path(cfg.b).name} step {sb}: "
          f"{l}x{h} heads, k={k}")
    print(f"{n} rows of {Path(cfg.cache).name}: fit {n_fit}, validate "
          f"{cfg.val}, test {cfg.test}\n")

    At = jnp.asarray(A)
    layer = np.repeat(np.arange(l), h)
    cols: Dict[str, Float[np.ndarray, "heads"]] = {"majority": majority(A, k)}
    lam_used: Dict[str, Any] = {}
    chunk = 4096

    Xt = jnp.asarray(X)
    ones = jnp.ones((n, 1), jnp.float32)
    acc, lam = probe(lambda i, j: jnp.concatenate([Xt[i:j], ones[i:j]], 1),
                     At, k, n_fit, cfg.val, n, cfg.lams, chunk)
    cols["input"], lam_used["input"] = acc, lam
    if cfg.mlp:
        cols["input/mlp"], lam_used["input/mlp"] = mlp_probe(
            lambda i, j: Xt[i:j] * X.shape[1] ** 0.5, At, k, n_fit, cfg.val,
            n, cfg.mlp_hidden, cfg.mlp_epochs, cfg.seed)

    for name, S in src.items():
        St = jnp.asarray(S)
        acc, lam = probe(lambda i, j: onehot(St[i:j], k, True), At, k,
                         n_fit, cfg.val, n, cfg.lams, chunk)
        cols[f"{name}/all"], lam_used[f"{name}/all"] = acc, lam
        if cfg.mlp:
            cols[f"{name}/mlp"], lam_used[f"{name}/mlp"] = mlp_probe(
                lambda i, j: onehot(St[i:j], k, False), At, k, n_fit,
                cfg.val, n, cfg.mlp_hidden, cfg.mlp_epochs, cfg.seed)
        if l > 1:
            same = np.empty(l * h)
            lam_used[f"{name}/same"] = []
            for i in range(l):
                sl = slice(i * h, (i + 1) * h)
                acc, lam = probe(
                    lambda a, b: onehot(St[a:b, sl], k, True), At[:, sl], k,
                    n_fit, cfg.val, n, cfg.lams, chunk)
                same[sl] = acc
                lam_used[f"{name}/same"].append(lam)
            cols[f"{name}/same"] = same
        print(f"  {name}: done", flush=True)

    with open(out / "heads.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["layer", "head"] + list(cols))
        for j in range(l * h):
            wr.writerow([int(layer[j]), j % h] + [f"{cols[c][j]:.5f}"
                                                  for c in cols])

    summary = {c: {"all": float(v.mean()),
                   **{f"layer{i}": float(v[layer == i].mean())
                      for i in range(l)}} for c, v in cols.items()}
    print(f"\nmean held-out accuracy over A's heads (chance {1 / k:.3f})")
    print(f"  {'source':<16}{'all':>8}" + "".join(
        f"{'L' + str(i):>8}" for i in range(l)))
    for c, v in summary.items():
        print(f"  {c:<16}{v['all']:>8.4f}" + "".join(
            f"{v[f'layer{i}']:>8.4f}" for i in range(l)))
    scopes = (["all"] + (["same"] if l > 1 else [])
              + (["mlp"] if cfg.mlp else []))
    margin = {}
    for sc in scopes:
        ctl = np.maximum(cols[f"lsh/{sc}"], cols[f"untrained/{sc}"])
        d = cols[f"seed/{sc}"] - ctl
        margin[sc] = {"all": float(d.mean()),
                      "heads_above": float((d > 0).mean()),
                      **{f"layer{i}": float(d[layer == i].mean())
                         for i in range(l)}}
        print(f"\nseed minus the better control ({sc}): "
              f"{margin[sc]['all']:+.4f}; by layer "
              + " ".join(f"{margin[sc][f'layer{i}']:+.4f}" for i in range(l))
              + f"; above it for {margin[sc]['heads_above']:.0%} of heads")
    (out / "summary.json").write_text(json.dumps(
        {"a": str(cfg.a), "step_a": int(sa), "b": str(cfg.b),
         "step_b": int(sb), "earlier": cfg.earlier, "rows": n,
         "fit": n_fit, "val": cfg.val, "test": cfg.test, "chance": 1 / k,
         "accuracy": summary, "margin": margin, "lams": lam_used}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
