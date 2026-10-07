"""Between-head disjoint support, measured on trained checkpoints.

The disjoint-support argument (writeup, Section "Disjoint support") says
that under non-negative atoms, heads that are orthogonal own disjoint
blocks of the dictionary space, as a property of the weights. The penalty
the reported models trained with, `cossim_h`, stood in for that; the
support overlap `sigma` (`DictBlock.support_overlap`) measures it
directly, but every reported model predates it. This computes it from
their checkpoints, two ways:

  dict     sigma in the dictionary space, where the partition is defined:
           mean off-diagonal cosine between heads' usage-weighted
           coordinate profiles s_h = sum_k pbar_hk |W_hk|
  decoded  the same with each atom replaced by its decoded vector,
           |dec(W_hk) - dec(0)|, i.e. in the output space steering acts
           in. The shared decoder mixes the blocks, so this need not be
           small even when `dict` is 0

`pbar` is each head's mean assignment over the evaluation tail, at the
run's final scheduled temperature, from the model's own forward
(`Ontologizer.classify`, so gain-shape and the router are applied).

Non-negative profiles overlap heavily by chance, so neither number reads
on its own. The null permutes each head's profile coordinates
independently (`--draws` times), keeping every head's own distribution
and destroying only the alignment between heads; sigma below the null
means heads avoid each other's coordinates, near it means no partition.
Under `ConcatDictBlock` `dict` is 0 by construction and its null is
skipped.

An `sae.py` params.npz (`--sae`) has no dictionary space: its decoder rows
are already output-space directions, so only `decoded` is defined, with a
head's profile `s_g = sum_{j in g} E[z_j] |W_dec,j|` (`z` the latent
activation, the analogue of the assignment). Heads are the trained groups
of an `sae.py --groups` run, or the post hoc groups `headstruct.py`
discovered (`--assignment`, latents labelled -1 left out).

  uv run python experiments/ste-arm/headsupport.py \\
      --model data/out/sonar/multilingual/ste_h76_init01 \\
      --model data/out/gpt2_l8/ste_h76_sgn --out headsupport.json
  uv run python experiments/ste-arm/headsupport.py \\
      --sae data/out/sonar/sae_conv/m5120_g160top1/params.npz \\
      --assignment trained \\
      --sae data/out/sonar/sae_conv/m11264_k32/params.npz \\
      --assignment data/out/sonar/sae_conv/m11264_k32/headstruct/assignment.npy
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ontologize.data.loaders import doc_holdout
from ontologize.training.ontostate import schedules
from pareto import load_onto

CACHE = {"sonar": ROOT / "data" / "sonar_embeddings" / "mc4_4M.npy",
         "gpt2": ROOT / "data" / "activations" / "gpt2_l8.npy"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[],
                   help="Ontologizer checkpoint dir; repeat for several")
    p.add_argument("--sae", action="append", default=[],
                   help="sae.py params.npz; repeat for several")
    p.add_argument("--assignment", action="append", default=[],
                   help="headstruct.py assignment.npy for the --sae at the "
                        "same position; 'trained' uses the run's own groups")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--eval-rows", type=int, default=32768)
    p.add_argument("--b", type=int, default=1024)
    p.add_argument("--draws", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    return p.parse_args()


def hyper(run: Path) -> dict:
    """The run's recorded hyperparameters: the last `env_config`."""
    last = None
    with open(run / "log.jsonl") as f:
        for line in f:
            if '"env_config"' in line:
                last = line
    return json.loads(last)["hyper"]


def temperature(h: dict, step: int) -> float:
    return schedules(step, h.get("anneal_steps", 0), h["temperature"],
                     h.get("temperature_end"), h.get("p_drop", 0.0),
                     h.get("p_drop_start"), h.get("sd_K", 0.0),
                     h.get("sd_K_end"))[0]


def eval_tail(run: Path, rows: int) -> Float[np.ndarray, "n d"]:
    cache = CACHE["gpt2" if "gpt2_l8" in run.parts else "sonar"]
    mm = np.load(cache, mmap_mode="r")
    n = doc_holdout(cache, rows)
    return np.asarray(mm[len(mm) - n:], dtype=np.float32)


def sigma(S: Float[np.ndarray, "h d"]) -> float:
    """Mean off-diagonal cosine between the rows of `S`."""
    U = S / (np.linalg.norm(S, axis=-1, keepdims=True) + 1e-12)
    C = U @ U.T
    h = len(S)
    return float((C.sum() - np.trace(C)) / (h * (h - 1)))


def spread(S: Float[np.ndarray, "h d"]) -> float:
    """Mean over heads of the profile's participation ratio as a fraction
    of the dimension: 1 when a head uses every coordinate equally, 1/d when
    it uses one. A partition into h blocks needs it near 1/h."""
    pr = S.sum(-1) ** 2 / (S ** 2).sum(-1)
    return float((pr / S.shape[-1]).mean())


def null(S: Float[np.ndarray, "h d"], draws: int,
         rng: np.random.Generator) -> Float[np.ndarray, "draws"]:
    """sigma with each head's coordinates permuted independently."""
    return np.array([sigma(rng.permuted(S, axis=1)) for _ in range(draws)])


def mean_assignment(model, params, X: Float[np.ndarray, "n d"], T: float,
                    b: int) -> Float[np.ndarray, "l h k"]:
    """Each head's assignment, averaged over `X` (`Ontologizer.classify`,
    so gain-shape and the router apply)."""
    def assign(module, X: Float[Array, "b d"]) -> Float[Array, "l b hk"]:
        E, _ = module.encode(X, 0.0, None)
        _, Ps = module.classify(E, X_ref=X, temperature=T)
        return Ps
    run = jax.jit(lambda X: model.apply({"params": params}, X, method=assign))
    acc, nb = 0.0, 0
    for i in range(0, len(X) - b + 1, b):
        acc = acc + np.asarray(run(jnp.asarray(X[i:i + b]))).mean(1)
        nb += 1
    return (acc / nb).reshape(model.l, model.h, model.k)


def atoms(model, params) -> tuple:
    """Per layer, the atoms in the dictionary space (h, k, e) and decoded
    into the output space (h, k, d), the latter less `dec(0)`."""
    def get(module):
        W = [d.dict.tags().reshape(module.h, module.k, -1)
             for d in module.dictencs]
        Z = module.decode(module.dictencs[0].place(jnp.zeros_like(W[0][0, 0])))
        return W, [module.decode(d.place(w)) - Z
                   for d, w in zip(module.dictencs, W)]
    W, D = model.apply({"params": params}, method=get)
    return [np.asarray(w) for w in W], [np.asarray(d) for d in D]


def measure(path: str, cfg: argparse.Namespace,
            rng: np.random.Generator) -> dict:
    run = Path(path).resolve()
    model, params, step = load_onto(run, cfg.step)
    h = hyper(run)
    T = temperature(h, step)
    concat = bool(getattr(model, "concat", False))
    P = mean_assignment(model, params, eval_tail(run, cfg.eval_rows), T, cfg.b)
    W, D = atoms(model, params)
    layers = []
    for i in range(model.l):
        row = {}
        for name, A in (("dict", W[i]), ("decoded", D[i])):
            S = np.einsum("hk,hkd->hd", P[i], np.abs(A))
            s = sigma(S)
            if name == "dict" and concat:
                row[name] = {"sigma": s, "null": None, "null_sd": None,
                             "spread": None}
                continue
            n = null(S, cfg.draws, rng)
            row[name] = {"sigma": s, "null": float(n.mean()),
                         "null_sd": float(n.std()), "spread": spread(S)}
        layers.append(row)
    return {"model": run.name, "path": str(run), "step": int(step),
            "T": T, "s_hcossim": h.get("s_hcossim"),
            "signed": bool(getattr(model, "signed", False)),
            "direct": bool(getattr(model, "direct", False)),
            "concat": concat, "l": model.l, "h": model.h, "k": model.k,
            "layers": layers}


def measure_sae(path: str, assignment: str, cfg: argparse.Namespace,
                rng: np.random.Generator) -> dict:
    """`decoded` sigma for an sae.py run, over its trained groups or a
    discovered assignment."""
    import sae
    p = Path(path).resolve()
    if (p.parent / "meta.json").exists():
        meta = json.loads((p.parent / "meta.json").read_text())
    else:  # pre-meta.json run: recover the encode rule from the name
        from pareto import parse_sae_name
        m, topk = parse_sae_name(p)
        meta = {"m": m, "topk": topk, "groups": 0, "group_fn": "top1"}
    params = {k: jnp.asarray(v) for k, v in np.load(p).items()}
    m, groups = meta["m"], meta["groups"]
    if assignment == "trained":
        assert groups, f"{p.parent.name} has no trained groups"
        labels = np.arange(m) // (m // groups)
    else:
        labels = np.load(assignment).astype(int)
    enc = jax.jit(lambda X: sae.encode(params, X, meta["topk"], groups,
                                       meta["group_fn"]))
    X = np.asarray(np.load(CACHE["sonar"], mmap_mode="r")[-cfg.eval_rows:],
                   np.float32)
    zbar, nb = 0.0, 0
    for i in range(0, len(X) - cfg.b + 1, cfg.b):
        zbar = zbar + np.asarray(enc(jnp.asarray(X[i:i + cfg.b]))).mean(0)
        nb += 1
    zbar = zbar / nb
    R = np.abs(np.asarray(params["W_dec"]))                     # (m, d)
    ids = [g for g in np.unique(labels) if g >= 0]
    S = np.stack([(zbar[labels == g, None] * R[labels == g]).sum(0)
                  for g in ids])
    n = null(S, cfg.draws, rng)
    return {"model": p.parent.name, "path": str(p),
            "groups": "trained" if assignment == "trained" else "discovered",
            "n_groups": len(ids), "assigned": int((labels >= 0).sum()),
            "m": m, "layers": [{"dict": None, "decoded": {
                "sigma": sigma(S), "null": float(n.mean()),
                "null_sd": float(n.std()), "spread": spread(S)}}]}


def summary(r: dict) -> str:
    out = []
    for name in ("dict", "decoded"):
        if r["layers"][0][name] is None:
            continue
        s = np.mean([L[name]["sigma"] for L in r["layers"]])
        if r["layers"][0][name]["null"] is None:
            out.append(f"{name} {s:.4f} (0 by construction)")
            continue
        n = np.mean([L[name]["null"] for L in r["layers"]])
        sd = np.mean([L[name]["null_sd"] for L in r["layers"]])
        sp = np.mean([L[name]["spread"] for L in r["layers"]])
        out.append(f"{name} {s:.4f} null {n:.4f}±{sd:.4f} ratio {s / n:.2f} "
                   f"spread {sp:.2f}")
    tag = f"step {r['step']:>7}" if "step" in r else \
        f"{r['n_groups']} {r['groups']} groups, {r['assigned']}/{r['m']} latents"
    return f"{r['model']:>24} {tag}  " + "  |  ".join(out)


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)
    results = []
    for path in cfg.model:
        r = measure(path, cfg, rng)
        results.append(r)
        print(summary(r), flush=True)
        for i, L in enumerate(r["layers"]):
            print(f"{'':>26}layer {i}: " + "  ".join(
                f"{n} {L[n]['sigma']:.4f}"
                + (f" (null {L[n]['null']:.4f})" if L[n]["null"] is not None
                   else "") for n in ("dict", "decoded")), flush=True)
        jax.clear_caches()
    assert len(cfg.assignment) in (0, len(cfg.sae)), \
        "pass one --assignment per --sae, or none"
    for path, assignment in zip(cfg.sae, cfg.assignment or
                                ["trained"] * len(cfg.sae)):
        r = measure_sae(path, assignment, cfg, rng)
        results.append(r)
        print(summary(r), flush=True)
    if cfg.out:
        Path(cfg.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
