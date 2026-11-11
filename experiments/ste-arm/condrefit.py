"""Would a layer's atoms gain from depending on an earlier layer's choice?

A residual stack gives layer l one dictionary for every input. Huijben
et al. (2024, QINCo) argue that residual quantizers lose by this: the
residual a stage sees is distributed differently in each cell of the
stages before it, so each cell wants its own codebook. Atoms that
depend on earlier choices would no longer be one dictionary to read, so
before building such an arm this asks whether there is anything to gain.

With every assignment and gain of a trained stack held fixed, and the
other layers' contributions held as they are, layer l's decoded atoms are
refit by least squares in the whitened frame, three ways:

  global       one set of atoms: the layer's own optimum given its
               assignments
  conditional  a separate set within each cell of one head of an earlier
               layer (`--cond-layer`), for the heads whose cells explain
               most of the layer's target and for as many drawn at random
  random       a separate set within each cell of a random partition of
               the rows with the same cell sizes. This is the null: a
               partition that knows nothing about the input still gets k
               sets of atoms

Each set is fit toward the global one with a small ridge, so an entry a
cell never selects keeps its global atom. Fit on training rows of the
cache and scored on a cache the model never saw; the figure is the whole
model's whitened FVU with layer l's atoms replaced.

  uv run python experiments/ste-arm/condrefit.py \\
      --ckpt data/out/sonar/multilingual/ste_h76_init01 --layer 1
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jaxtyping import Array, Float, Int  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "next-sentence"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def design(I: Int[np.ndarray, "h n"], g: Float[np.ndarray, "n"], k: int
           ) -> Float[np.ndarray, "n hk"]:
    """The layer's output as a linear function of its atoms: row n holds
    the gain g_n at the entry each head selects, so `design @ atoms` is
    the layer's contribution."""
    h, n = I.shape
    Phi = np.zeros((n, h * k), np.float32)
    Phi[np.arange(n)[:, None], (I + np.arange(h)[:, None] * k).T] = \
        g[:, None]
    return Phi


@jax.jit
def normal_eq(Phi: Float[Array, "n a"], T: Float[Array, "n d"],
              onehot: Float[Array, "n c"]
              ) -> Tuple[Float[Array, "c a a"], Float[Array, "c a d"]]:
    """Per cell of a partition of the rows, the normal equations of
    T ~ Phi A."""
    def one(m):
        Pm = Phi * m[:, None]
        return Pm.T @ Phi, Pm.T @ T
    return jax.lax.map(one, onehot.T)


@jax.jit
def solve_toward(AtA: Float[Array, "c a a"], AtT: Float[Array, "c a d"],
                 A_0: Float[Array, "a d"], ridge: float
                 ) -> Float[Array, "c a d"]:
    """Per cell, the atoms minimizing |T - Phi A|^2 + lam |A - A_0|^2,
    with lam = `ridge` times the cell's mean diagonal. An entry the cell
    never selects has no data and stays at `A_0`."""
    def one(args):
        G, B = args
        lam = ridge * jnp.trace(G) / G.shape[0] + 1e-12
        return A_0 + jnp.linalg.solve(G + lam * jnp.eye(G.shape[0]),
                                      B - G @ A_0)
    return jax.lax.map(one, (AtA, AtT))


@jax.jit
def predict(A: Float[Array, "c a d"], cols: Int[Array, "n h"],
            g: Float[Array, "n"], cell: Int[Array, "n"]
            ) -> Float[Array, "n d"]:
    """The layer's contribution under per-cell atoms: each row sums the
    atoms its heads select, from its own cell's set, times its gain."""
    return A[cell[:, None], cols].sum(1) * g[:, None]


def eta2(T: Float[np.ndarray, "n d"], cell: Int[np.ndarray, "n"]) -> float:
    """Share of the target's variance that lies between a partition's
    cells."""
    tot = ((T - T.mean(0)) ** 2).sum()
    within = sum(((T[cell == c] - T[cell == c].mean(0)) ** 2).sum()
                 for c in np.unique(cell))
    return float(1.0 - within / tot)


def accumulate(I: Int[np.ndarray, "h n"], g: Float[np.ndarray, "n"], k: int,
               T: Float[np.ndarray, "n d"], cell: Int[np.ndarray, "n"],
               chunk: int = 8192
               ) -> Tuple[Float[Array, "c a a"], Float[Array, "c a d"]]:
    """The normal equations of the layer's target on its atoms, per cell
    of one partition of the rows. `I` and `g` are the layer's selections
    and gains."""
    c, a = int(cell.max()) + 1, I.shape[0] * k
    G = jnp.zeros((c, a, a), jnp.float32)
    B = jnp.zeros((c, a, T.shape[1]), jnp.float32)
    for i in range(0, T.shape[0], chunk):
        Phi = design(I[:, i:i + chunk], g[i:i + chunk], k)
        oh = jax.nn.one_hot(jnp.asarray(cell[i:i + chunk]), c)
        gg, b = normal_eq(jnp.asarray(Phi), jnp.asarray(T[i:i + chunk]), oh)
        G, B = G + gg, B + b
    return G, B


def refit(I: Int[np.ndarray, "h n"], g: Float[np.ndarray, "n"], k: int,
          T: Float[np.ndarray, "n d"], cells: Dict[str, Int[np.ndarray, "n"]],
          A_0: Float[np.ndarray, "a d"], ridge: float
          ) -> Dict[str, Float[Array, "c a d"]]:
    """Atoms for each named partition of the rows, each fit toward
    `A_0`."""
    A0 = jnp.asarray(A_0)
    return {name: solve_toward(*accumulate(I, g, k, T, cell), A0, ridge)
            for name, cell in cells.items()}


def score(A: Float[Array, "c a d"], I: Int[np.ndarray, "h n"],
          g: Float[np.ndarray, "n"], cell: Int[np.ndarray, "n"],
          T: Float[np.ndarray, "n d"], k: int, chunk: int = 2048) -> float:
    """Sum of squared error of T against the layer's contribution under
    the atoms `A`."""
    cols = (I + np.arange(len(I))[:, None] * k).T
    sse = 0.0
    for i in range(0, len(T), chunk):
        P = predict(A, jnp.asarray(cols[i:i + chunk]),
                    jnp.asarray(g[i:i + chunk]),
                    jnp.asarray(cell[i:i + chunk]))
        sse += float(((jnp.asarray(T[i:i + chunk]) - P) ** 2).sum())
    return sse


def like(cell: Int[np.ndarray, "n"], n: int, rng: np.random.Generator
         ) -> Int[np.ndarray, "n"]:
    """`n` random cell labels with the frequencies of `cell`."""
    return rng.choice(cell, n, replace=True)


def layer_view(ckpt: str, step: int, T: float, X: Float[np.ndarray, "n d"],
               w_sqrt: Float[np.ndarray, "d"], batch: int, layer: int):
    """For layer `layer` on rows `X`, in the whitened frame: its trained
    decoded atoms (a, d), every head's selections (H, n) with each head's
    layer, the layer's gains (n,), and the layer's target: the input less
    everything the rest of the model writes."""
    from eval_next import model_outputs
    from headcontrib import atoms_and_codes
    D, I, G, lay, used = atoms_and_codes(ckpt, step, T, X, w_sqrt, batch)
    Y = model_outputs(ckpt, used, X, T, 1024)
    k = D.shape[1]
    mine = lay == layer
    A_0 = D[mine].reshape(-1, D.shape[2]).astype(np.float32)
    g = G[layer].astype(np.float32)
    own = np.zeros_like(Y)
    cols = (I[mine] + np.arange(int(mine.sum()))[:, None] * k).T
    for i in range(0, len(X), 4096):
        own[i:i + 4096] = A_0[cols[i:i + 4096]].sum(1) * g[i:i + 4096, None]
    target = (X - Y) * w_sqrt + own
    return A_0, I, lay, g, target, k, used


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--layer", type=int, default=1,
                   help="the layer whose atoms are refit")
    p.add_argument("--cond-layer", type=int, default=0,
                   help="the earlier layer whose heads' cells condition it")
    p.add_argument("--heads", type=int, default=3,
                   help="conditioning heads taken from the top by eta^2 of "
                        "the target, and as many at random")
    p.add_argument("--nulls", type=int, default=2,
                   help="random partitions")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--fit-rows", type=int, default=1 << 20,
                   help="the cache's first rows, which the model trained on")
    p.add_argument("--eval-cache",
                   default="data/sonar_embeddings/mc4_fresh.npy")
    p.add_argument("--eval-rows", type=int, default=1 << 17)
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--ridge", type=float, nargs="+", default=[1e-4],
                   help="pull toward the prior atoms, as a multiple of each "
                        "cell's mean diagonal; several values are scored "
                        "from one pass. A per-cell set has k times the "
                        "parameters of the global one, so the comparison "
                        "that matters is with the random partition at the "
                        "same value")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON of the results")
    cfg = p.parse_args()
    assert cfg.cond_layer < cfg.layer

    w_sqrt = np.sqrt(np.load(cfg.mse_weights).astype(np.float64)
                     ).astype(np.float32)
    Xf = np.asarray(np.load(cfg.cache, mmap_mode="r")[:cfg.fit_rows],
                    np.float32)
    Xe = np.asarray(np.load(cfg.eval_cache, mmap_mode="r")[:cfg.eval_rows],
                    np.float32)
    rng = np.random.default_rng(cfg.seed)

    A_0, If, lay, gf, Tf, k, step = layer_view(
        cfg.ckpt, cfg.step, cfg.temperature, Xf, w_sqrt, cfg.batch, cfg.layer)
    _, Ie, _, ge, Te, _, _ = layer_view(
        cfg.ckpt, step, cfg.temperature, Xe, w_sqrt, cfg.batch, cfg.layer)
    mine = lay == cfg.layer
    cond = np.flatnonzero(lay == cfg.cond_layer)

    sub = slice(0, min(len(Xf), 65536))
    e2 = np.array([eta2(Tf[sub], If[c, sub]) for c in cond])
    top = cond[np.argsort(-e2)[:cfg.heads]]
    rest = np.setdiff1d(cond, top)
    drawn = rng.choice(rest, min(cfg.heads, len(rest)), replace=False)
    fit_cells = {"global": np.zeros(len(Xf), np.int64)}
    eval_cells = {"global": np.zeros(len(Xe), np.int64)}
    kinds = {"global": "global"}
    for kind, heads in (("top", top), ("drawn", drawn)):
        for c in heads:
            name = f"head {c - cond[0]} ({kind})"
            fit_cells[name], eval_cells[name] = If[c], Ie[c]
            kinds[name] = kind
    for j in range(cfg.nulls):
        name = f"random {j}"
        fit_cells[name] = rng.permutation(If[top[0]])
        eval_cells[name] = like(If[top[0]], len(Xe), rng)
        kinds[name] = "random"

    var = float((((Xe - Xe.mean(0)) * w_sqrt) ** 2).sum())
    A0 = jnp.asarray(A_0)
    trained = score(A0[None], Ie[mine], ge, eval_cells["global"], Te, k) / var
    res = {"ckpt": cfg.ckpt, "step": int(step), "layer": cfg.layer,
           "cond_layer": cfg.cond_layer, "fit_rows": len(Xf),
           "eval_rows": len(Xe), "ridge": cfg.ridge,
           "fvu_trained": trained, "eta2": {}, "fvu": {}, "kind": kinds}
    print(f"{Path(cfg.ckpt).name} step {step}: layer {cfg.layer} refit, "
          f"conditioned on layer {cfg.cond_layer}; fit on {len(Xf)} rows, "
          f"scored on {len(Xe)}")
    print(f"  trained atoms          FVU_w {trained:.5f}")
    print(f"  {'ridge':<22} " + "  ".join(f"{r:>8g}" for r in cfg.ridge))
    for name, cell in fit_cells.items():
        G, B = accumulate(If[mine], gf, k, Tf, cell)
        fs = [score(solve_toward(G, B, A0, r), Ie[mine], ge,
                    eval_cells[name], Te, k) / var for r in cfg.ridge]
        del G, B
        res["fvu"][name] = fs
        note = ""
        if name.startswith("head"):
            j = int(name.split()[1])
            res["eta2"][name] = float(e2[j])
            note = f"  (eta^2 of the target {e2[j]:.4f})"
        print(f"  {name:<22} " + "  ".join(f"{f:.5f}" for f in fs) + note,
              flush=True)
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
