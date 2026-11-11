"""Does a residual stack put a planted parent in layer 0 and its children
in layer 1?

`experiments/dictenc-toy` plants independent factors for one layer. This
plants a two-level tree, after the synthetic benchmark of Costa et al.
(2025): `--parents` parent concepts, the first `--with-children` of which
each have `--children` children. A sample draws one parent uniformly; if
the parent has children, one of them is added with probability
`--p-child`:

    x = P[z_p] + child_scale * C[z_c] + sigma * eps

with one atom per parent and per child, N(0, 1/2) per coordinate, and
z_c = 0 meaning no child (a zero atom). The child is a second
categorical variable whose support depends on the first.

Three things can decide which layer a factor lands in, and the options
separate them:

- the hierarchy: `--shuffle-tree` permutes z_c across samples, keeping
  both marginals and removing the dependence;
- variance: `--child-scale` above 1 gives the child component more
  variance than the parent's;
- depth itself: `--l 1 --h 2` is the same capacity with no order between
  the heads, and `--no-deepsup` is the stack without a loss on the
  layer-0 prefix.

Scored on held-out samples, per head: the share of each factor's entropy
the head's entry carries (mutual information over the factor's entropy;
1 is a head that determines the factor), and the share of what the child
adds once the parent is known, since in a tree the parent alone says
whether a child is possible; each beside the same with rows permuted.
Per layer: the FVU of the prefix through that layer, beside the noise
floor and the best reconstruction from the parent alone and from the
child alone.

    uv run python experiments/planted-hierarchy/tree.py --seeds 5
    uv run python experiments/planted-hierarchy/tree.py --seeds 5 --shuffle-tree
    uv run python experiments/planted-hierarchy/tree.py --seeds 5 --l 1 --h 2
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float, Int  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("planted tree")
    g.add_argument("--parents", type=int, default=11)
    g.add_argument("--with-children", type=int, default=3)
    g.add_argument("--children", type=int, default=3)
    g.add_argument("--p-child", type=float, default=0.75,
                   help="chance a parent that has children shows one")
    g.add_argument("--child-scale", type=float, default=1.0)
    g.add_argument("--shuffle-tree", action="store_true",
                   help="permute the child across samples: the same "
                        "marginals with no dependence on the parent")
    g.add_argument("--d", type=int, default=64)
    g.add_argument("--sigma", type=float, default=0.3)
    g.add_argument("--n-train", type=int, default=32768)
    g.add_argument("--n-eval", type=int, default=8192)
    g.add_argument("--data-seed", type=int, default=0)
    m = p.add_argument_group("model")
    m.add_argument("--l", type=int, default=2)
    m.add_argument("--h", type=int, default=1, help="heads per layer")
    m.add_argument("--k", type=int, default=16)
    m.add_argument("--e-dec", type=int, default=128)
    m.add_argument("--select", default="ste")
    m.add_argument("--no-deepsup", action="store_true")
    t = p.add_argument_group("training (dictenc-toy's schedule)")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--anneal-steps", type=int, default=3000)
    t.add_argument("--b", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--temperature-end", type=float, default=0.03)
    t.add_argument("--p-drop", type=float, default=0.1)
    t.add_argument("--sd-k", type=float, default=0.02)
    t.add_argument("--sd-f", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, default=1,
                   help="replicates; model and data seeds advance together")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None,
                   help="default experiments/planted-hierarchy/out/<name>")
    cfg = p.parse_args()
    if cfg.name is None:
        cfg.name = (f"l{cfg.l}h{cfg.h}k{cfg.k}"
                    + ("_nodeepsup" if cfg.no_deepsup else "")
                    + ("_shuffled" if cfg.shuffle_tree else "_tree")
                    + f"_cs{cfg.child_scale:g}"
                    + (f"_x{cfg.seeds}" if cfg.seeds > 1 else ""))
    return cfg


# ---------- planted tree ----------

def plant(rng: np.random.Generator, parents: int, with_children: int,
          children: int, d: int, child_scale: float
          ) -> Tuple[Float[np.ndarray, "p d"], Float[np.ndarray, "c1 d"],
                     Int[np.ndarray, "c1"]]:
    """Parent atoms, child atoms (row 0 is "no child", a zero atom) and
    each child's parent (-1 for row 0). Child i + 1 belongs to parent
    i // children."""
    P = rng.normal(0.0, np.sqrt(0.5), (parents, d))
    C = np.zeros((1 + with_children * children, d))
    C[1:] = child_scale * rng.normal(0.0, np.sqrt(0.5), (len(C) - 1, d))
    owner = np.concatenate([[-1], np.repeat(np.arange(with_children),
                                            children)])
    return P.astype(np.float32), C.astype(np.float32), owner


def sample(rng: np.random.Generator, P: Float[np.ndarray, "p d"],
           C: Float[np.ndarray, "c1 d"], owner: Int[np.ndarray, "c1"],
           n: int, sigma: float, p_child: float, shuffle: bool
           ) -> Tuple[Float[np.ndarray, "n d"], Int[np.ndarray, "n"],
                      Int[np.ndarray, "n"], Float[np.ndarray, "n d"]]:
    """(X, z_p, z_c, X_clean). A parent with children shows one of them,
    uniformly, with probability `p_child`; `shuffle` then permutes z_c."""
    zp = rng.integers(0, len(P), n)
    children = int((owner == 0).sum())
    has = (zp <= owner.max()) & (rng.random(n) < p_child)
    zc = np.where(has, 1 + zp * children + rng.integers(0, children, n), 0)
    if shuffle:
        zc = zc[rng.permutation(n)]
    clean = P[zp] + C[zc]
    X = clean + sigma * rng.normal(size=clean.shape)
    return X.astype(np.float32), zp, zc, clean.astype(np.float32)


# ---------- scoring ----------

def explained(a: Int[np.ndarray, "n"], z: Int[np.ndarray, "n"]) -> float:
    """I(a; z) / H(z): the share of the factor's entropy that the head's
    entry carries. 0 for a constant factor."""
    ka, kz = int(a.max()) + 1, int(z.max()) + 1
    J = np.zeros((ka, kz))
    np.add.at(J, (a, z), 1.0)
    J /= J.sum()
    pa, pz = J.sum(1), J.sum(0)
    nz = J > 0
    mi = (J[nz] * np.log(J[nz] / np.outer(pa, pz)[nz])).sum()
    hz = -(pz[pz > 0] * np.log(pz[pz > 0])).sum()
    return float(mi / hz) if hz > 0 else 0.0


def explained_given(a: Int[np.ndarray, "n"], z: Int[np.ndarray, "n"],
                    w: Int[np.ndarray, "n"]) -> float:
    """I(a; z | w) / H(z | w): of what `z` adds once `w` is known, the
    share the head's entry carries. In the tree the parent alone fixes
    much of the child (whether one is possible), so a head that carries
    the parent scores on `explained(a, child)` without telling children
    apart; this is the score that does not credit it."""
    joint = a * (int(w.max()) + 1) + w
    hz = -sum(p * np.log(p) for p in np.bincount(z) / len(z) if p > 0)
    left = hz * (1.0 - explained(w, z))                   # H(z | w)
    if left <= 0:
        return 0.0
    return float(hz * (explained(joint, z) - explained(w, z)) / left)


def head_table(A: Int[np.ndarray, "heads n"], zp: Int[np.ndarray, "n"],
               zc: Int[np.ndarray, "n"], rng: np.random.Generator
               ) -> List[Dict[str, float]]:
    """Per head: the share of the parent's entropy it carries, of the
    child's, and of the child's beyond the parent; and each against
    permuted rows (the chance level at this sample size)."""
    perm = rng.permutation(len(zp))
    return [{"parent": explained(a, zp), "child": explained(a, zc),
             "child_new": explained_given(a, zc, zp),
             "parent_null": explained(a, zp[perm]),
             "child_null": explained(a, zc[perm]),
             "child_new_null": explained_given(a[perm], zc, zp)} for a in A]


def placement(table: List[Dict[str, float]], layer: Int[np.ndarray, "heads"]
              ) -> Dict[str, int]:
    """Which head carries most of the parent, which most of what the
    child adds beyond it, and their layers."""
    hp = int(np.argmax([t["parent"] for t in table]))
    hc = int(np.argmax([t["child_new"] for t in table]))
    return {"parent_head": hp, "parent_layer": int(layer[hp]),
            "child_head": hc, "child_layer": int(layer[hc])}


def fvu(Y: Float[np.ndarray, "n d"], X: Float[np.ndarray, "n d"]) -> float:
    return float(((Y - X) ** 2).mean() / ((X - X.mean(0)) ** 2).mean())


def fvu_given(z: Int[np.ndarray, "n"], X: Float[np.ndarray, "n d"]) -> float:
    """FVU of the best reconstruction that knows `z` alone: each sample
    replaced by the mean of the samples sharing its value."""
    Y = np.zeros_like(X)
    for v in np.unique(z):
        Y[z == v] = X[z == v].mean(0)
    return fvu(Y, X)


# ---------- model ----------

def build(cfg: argparse.Namespace, seed: int):
    from ontologize.training.config import Hyperparams
    hyper = Hyperparams(
        cfg.d, cfg.d, cfg.b, epochs=1, lr=cfg.lr, wd=0.0,
        temperature=cfg.temperature,
        noise_in="batchnorm", noise_K="normal", noise_F="featvar",
        sd_in=0.0, sd_K=cfg.sd_k, sd_F=cfg.sd_f, p_drop=cfg.p_drop,
        seed=seed, ghost=False, temperature_end=cfg.temperature_end,
        anneal_steps=cfg.anneal_steps, p_drop_start=0.0,
        sd_K_end=0.2 * cfg.temperature_end)
    model = hyper.ontologizer(
        cfg.d, cfg.d, cfg.e_dec, cfg.k, cfg.h, cfg.l,
        n=2, gate="none", select=cfg.select, forward="resid",
        deepsup=not cfg.no_deepsup and cfg.l > 1,
        resid_norm=False, resid_const=True, resid_gain=False,
        dtype_str="float32", dtype_p_str="float32")
    return hyper, model, hyper.init(model, save_each=cfg.steps)


def run_model(model, params, X: Float[np.ndarray, "n d"], T: float
              ) -> Tuple[Float[np.ndarray, "l n d"], Int[np.ndarray, "heads n"]]:
    """The decode of every prefix and each head's entry, layer by layer."""
    import jax.numpy as jnp

    def probe(module, Xb):
        E, _ = module.encode(Xb, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        Ys, As = [], []
        for i, de in enumerate(module.dictencs):
            R, K = de.withClusts(R, Ein, temperature=T)
            Ys.append(module.decode(R))
            As.append(jnp.argmax(K.reshape(Xb.shape[0], module.h, module.k),
                                 -1))
            if i < module.l - 1:
                Ein = module.nextinput(Xb, R, K)
        return jnp.stack(Ys), jnp.stack(As)

    Y, A = model.apply(params, jnp.asarray(X), method=probe)
    A = np.asarray(A)                                      # (l, n, h)
    return np.asarray(Y), A.transpose(0, 2, 1).reshape(-1, A.shape[1])


def train(cfg: argparse.Namespace, hyper, state, X_tr: Float[np.ndarray, "n d"],
          seed: int):
    import jax
    import jax.numpy as jnp
    from ontologize.training.ontostate import schedules, update
    rng = hyper.rng()
    order = np.random.default_rng(seed)
    n_batches = len(X_tr) // cfg.b
    perm = order.permutation(len(X_tr))
    for step in range(cfg.steps):
        if step and step % n_batches == 0:
            perm = order.permutation(len(X_tr))
        i = (step % n_batches) * cfg.b
        Xb = jnp.asarray(X_tr[perm[i:i + cfg.b]])
        T, pd, sk, _ = schedules(
            step, hyper.anneal_steps, hyper.temperature,
            hyper.temperature_end, hyper.p_drop, hyper.p_drop_start,
            hyper.sd_K, hyper.sd_K_end)
        rng, r = jax.random.split(rng)
        state, _, _ = update(
            state, hyper.loss, r, Xb, Xb, temperature=T, p_drop=pd,
            sd_K=sk, sd_in=0.0, sd_F=hyper.sd_F, grad_clip=hyper.grad_clip,
            p_revive=hyper.p_revive, revive_frac=hyper.revive_frac)
    return state


def replicate(cfg: argparse.Namespace, seed: int, data_seed: int
              ) -> Dict[str, object]:
    data = np.random.default_rng(data_seed)
    P, C, owner = plant(data, cfg.parents, cfg.with_children, cfg.children,
                        cfg.d, cfg.child_scale)
    X_tr, _, _, _ = sample(data, P, C, owner, cfg.n_train, cfg.sigma,
                           cfg.p_child, cfg.shuffle_tree)
    X, zp, zc, clean = sample(data, P, C, owner, cfg.n_eval, cfg.sigma,
                              cfg.p_child, cfg.shuffle_tree)
    hyper, model, state = build(cfg, seed)
    state = train(cfg, hyper, state, X_tr, seed)
    Y, A = run_model(model, state.params, X, cfg.temperature_end)
    layer = np.repeat(np.arange(cfg.l), cfg.h)
    table = head_table(A, zp, zc, np.random.default_rng(seed + 1))
    return {"seed": seed, "data_seed": data_seed, "heads": table,
            **placement(table, layer),
            "fvu_prefix": [fvu(y, X) for y in Y],
            "fvu_oracle": fvu(clean, X),
            "fvu_parent_only": fvu_given(zp, X),
            "fvu_child_only": fvu_given(zc, X)}


def report(cfg: argparse.Namespace, r: Dict[str, object]) -> None:
    layer = np.repeat(np.arange(cfg.l), cfg.h)
    heads = "  ".join(
        f"L{layer[i]}h{i % cfg.h}: parent {t['parent']:.2f} child "
        f"{t['child']:.2f} (beyond parent {t['child_new']:.2f})"
        for i, t in enumerate(r["heads"]))
    fv = " ".join(f"{v:.3f}" for v in r["fvu_prefix"])
    print(f"seed {r['seed']}: {heads} | prefix FVU {fv} (oracle "
          f"{r['fvu_oracle']:.3f}, parent alone {r['fvu_parent_only']:.3f}, "
          f"child alone {r['fvu_child_only']:.3f})", flush=True)


def tally(cfg: argparse.Namespace, results: List[Dict[str, object]]
          ) -> Dict[str, object]:
    """Over replicates: per head, the mean share of each factor it
    carries; how often the parent's head is in layer 0 and the child's in
    the last layer; the mean prefix FVUs."""
    H = cfg.l * cfg.h
    out = {"n": len(results),
           "parent": [float(np.mean([r["heads"][i]["parent"]
                                     for r in results])) for i in range(H)],
           "child": [float(np.mean([r["heads"][i]["child"]
                                    for r in results])) for i in range(H)],
           "child_new": [float(np.mean([r["heads"][i]["child_new"]
                                        for r in results]))
                         for i in range(H)],
           "null": float(np.mean([max(t["parent_null"], t["child_null"])
                                  for r in results for t in r["heads"]])),
           "child_new_null": float(np.mean([t["child_new_null"]
                                            for r in results
                                            for t in r["heads"]])),
           "parent_in_layer0": int(sum(r["parent_layer"] == 0
                                       for r in results)),
           "child_in_last": int(sum(r["child_layer"] == cfg.l - 1
                                    for r in results)),
           "parent_head0": int(sum(r["parent_head"] == 0 for r in results)),
           "fvu_prefix": np.mean([r["fvu_prefix"] for r in results], 0
                                 ).tolist(),
           "fvu_oracle": float(np.mean([r["fvu_oracle"] for r in results])),
           "fvu_parent_only": float(np.mean([r["fvu_parent_only"]
                                             for r in results])),
           "fvu_child_only": float(np.mean([r["fvu_child_only"]
                                            for r in results]))}
    return out


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.out) if cfg.out else \
        ROOT / "experiments/planted-hierarchy/out" / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"{cfg.name}: {cfg.parents} parents, {cfg.with_children} with "
          f"{cfg.children} children each (p {cfg.p_child}, scale "
          f"{cfg.child_scale:g}{', shuffled' if cfg.shuffle_tree else ''}); "
          f"model l={cfg.l} h={cfg.h} k={cfg.k} select={cfg.select}"
          f"{' no deepsup' if cfg.no_deepsup else ''}; {cfg.steps} steps x "
          f"{cfg.seeds} seed(s)")
    t0 = time.time()
    results = []
    for i in range(cfg.seeds):
        results.append(replicate(cfg, cfg.seed + i, cfg.data_seed + i))
        report(cfg, results[-1])
    tl = tally(cfg, results)
    where = (f"parent's head in layer 0 {tl['parent_in_layer0']}/{tl['n']}, "
             f"child's head in the last layer {tl['child_in_last']}/{tl['n']}"
             if cfg.l > 1 else
             f"parent's head is head 0 {tl['parent_head0']}/{tl['n']}")
    print(f"tally over {tl['n']} seeds: {where}; share of parent by head "
          f"{np.round(tl['parent'], 3).tolist()}, of child "
          f"{np.round(tl['child'], 3).tolist()} (permuted rows "
          f"{tl['null']:.3f}), of child beyond the parent "
          f"{np.round(tl['child_new'], 3).tolist()} (permuted rows "
          f"{tl['child_new_null']:.3f}); prefix FVU "
          f"{np.round(tl['fvu_prefix'], 3).tolist()} (oracle "
          f"{tl['fvu_oracle']:.3f}); {time.time() - t0:.0f}s")
    (out / "summary.json").write_text(json.dumps(
        {"config": vars(cfg), "seeds": results, "tally": tl}, indent=1))


if __name__ == "__main__":
    main()
