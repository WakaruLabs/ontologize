"""How much of each layer's input is training noise, in a residual stack.

Feature noise (sd_F) is added to a layer's output, so it enters the
accumulator and is part of every later layer's input: the residual the
next layer classifies is X minus the noisy prefix's decode. `featvar` noise
is scaled by each output feature's batch std, so an upper layer, whose own
input residual has shrunk, may receive noise sized by the earlier layers'
outputs. The hypothesis this tests is that the noise is why upper heads
do not reproduce and the last layer does little.

The training forward is replayed over held-out rows, in batches of the
run's training size (featvar reads batch statistics), with each training
noise alone at the run's own value (or the flags'), then all together,
beside the clean forward on the same rows. Per layer:

  share   E||y_noisy - y_clean||^2 / E||y_clean||^2 over the input's
          residual coordinates (layer 0's input carries no noise: sd_in is
          0 in these runs, so its share is 0 under every noise)
  shape   mean 1 - cos(u_noisy, u_clean), u the unit direction (with the
          constant coordinate) that the classifier reads
  flip    the share of the layer's heads whose entry differs from the
          clean forward's

  JAX_PLATFORMS=cpu uv run python experiments/recipe-ladder/noiseshare.py \\
      --ckpt data/out/sonar/recipe_ladder/ste_h76_init01_med837_ctrl

Writes <out>/noiseshare.csv (run, noise, layer, share, shape, flip).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def trace(module, X, rng, sd_K, sd_F, p_drop, temperature):
    """The training forward (`Ontologizer.withStats`, sd_in 0), returning
    each layer's input residual, its classifier input and its assignments,
    stacked over layers."""
    import jax.numpy as jnp
    d = X.shape[-1]
    E, rng_K = module.encode(X, 0.0, rng)
    R = module.resid(E)
    E = module.constinput(E)
    Ys, Us, Ks = [], [], []
    for i, de in enumerate(module.dictencs):
        Ys.append(E[..., :d])
        Us.append(de.gainshape_in(E)[0])
        R, K, _, rng_K = de.withStats(R, E, sd_K=sd_K, sd_F=sd_F, rng=rng_K,
                                      p_drop=p_drop, temperature=temperature)
        Ks.append(K)
        if i < module.l - 1:
            E = module.nextinput(X, R, K)
    return jnp.stack(Ys), jnp.stack(Us), jnp.stack(Ks)


def share(Y_noisy: np.ndarray, Y_clean: np.ndarray) -> np.ndarray:
    """Per layer: E||y_noisy - y_clean||^2 / E||y_clean||^2, over (l, b, d)."""
    num = ((Y_noisy - Y_clean) ** 2).sum(-1).mean(-1)
    return num / (Y_clean ** 2).sum(-1).mean(-1)


def shape_dist(U_noisy: np.ndarray, U_clean: np.ndarray) -> np.ndarray:
    """Per layer: mean 1 - cos between the two forwards' classifier inputs."""
    n = np.linalg.norm(U_noisy, axis=-1) * np.linalg.norm(U_clean, axis=-1)
    return (1.0 - (U_noisy * U_clean).sum(-1) / np.maximum(n, 1e-12)).mean(-1)


def flip_rate(K_noisy: np.ndarray, K_clean: np.ndarray, k: int) -> np.ndarray:
    """Per layer: the share of (row, head) pairs whose entry differs, from
    flattened (l, b, h*k) assignments."""
    a = K_noisy.reshape(*K_noisy.shape[:-1], -1, k).argmax(-1)
    c = K_clean.reshape(*K_clean.shape[:-1], -1, k).argmax(-1)
    return (a != c).mean((-1, -2))


def run_hyper(run: Path) -> dict:
    """The run's last recorded training configuration (log.jsonl)."""
    last = None
    for line in open(run / "log.jsonl"):
        if "env_config" in line:
            rec = json.loads(line)
            last = rec.get("env_config", rec)
    return last.get("hyper", last)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="run directories, each at its latest step, or dir@step")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_sent16.npy")
    p.add_argument("--eval-rows", type=int, default=32768)
    p.add_argument("--rows", type=int, default=8192,
                   help="tail rows replayed, in batches of the training size")
    p.add_argument("--sd-k", type=float, default=None,
                   help="logit noise (default: the run's own)")
    p.add_argument("--sd-f", type=float, default=None,
                   help="feature noise (default: the run's own)")
    p.add_argument("--p-drop", type=float, default=None,
                   help="winner dropout (default: the run's own)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/out/sonar/recipe_ladder/noiseshare")
    return p.parse_args()


def main():
    import functools
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import jax
    from pareto import load_onto

    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    mm = np.load(cfg.cache, mmap_mode="r")
    start = len(mm) - cfg.eval_rows
    rows_out = []
    for spec in cfg.ckpt:
        path, _, step = spec.partition("@")
        run = Path(path)
        model, params, step = load_onto(str(run), int(step or 0))
        hp = run_hyper(run)
        T = float(hp["temperature"] if hp.get("temperature_end") is None
                  else hp["temperature_end"])
        b = int(hp["b"])
        noise = {"sd_K": cfg.sd_k if cfg.sd_k is not None else float(hp["sd_K"]),
                 "sd_F": cfg.sd_f if cfg.sd_f is not None else float(hp["sd_F"]),
                 "p_drop": cfg.p_drop if cfg.p_drop is not None else float(hp["p_drop"])}
        conds = {"clean": (0.0, 0.0, 0.0),
                 "sd_F": (0.0, noise["sd_F"], 0.0),
                 "sd_K": (noise["sd_K"], 0.0, 0.0),
                 "p_drop": (0.0, 0.0, noise["p_drop"]),
                 "all": (noise["sd_K"], noise["sd_F"], noise["p_drop"])}
        fns = {c: jax.jit(functools.partial(
                   model.apply, {"params": params}, sd_K=v[0], sd_F=v[1],
                   p_drop=v[2], temperature=T, method=trace))
               for c, v in conds.items()}
        acc = {c: np.zeros((3, model.l)) for c in conds if c != "clean"}
        n = 0
        key = jax.random.PRNGKey(cfg.seed)
        for i in range(start, start + cfg.rows - b + 1, b):
            X = np.asarray(mm[i:i + b], np.float32)
            key, sub = jax.random.split(key)
            Yc, Uc, Kc = (np.asarray(a) for a in fns["clean"](X, sub))
            for c in acc:
                Y, U, K = (np.asarray(a) for a in fns[c](X, sub))
                acc[c] += np.stack([share(Y, Yc), shape_dist(U, Uc),
                                    flip_rate(K, Kc, model.k)])
            n += 1
        print(f"{run.name} @ {step}: t={T}, b={b}, noise {noise}, {n * b} rows")
        print(f"  {'noise':7s} " + " ".join(f"{'L' + str(L) + ' share/shape/flip':>24s}"
                                           for L in range(model.l)))
        for c, a in acc.items():
            a = a / n
            print(f"  {c:7s} " + " ".join(f"{a[0, L]:8.4f} {a[1, L]:7.4f} {a[2, L]:7.4f}"
                                         for L in range(model.l)))
            for L in range(model.l):
                rows_out.append([run.name, step, c, L, f"{a[0, L]:.5f}",
                                 f"{a[1, L]:.5f}", f"{a[2, L]:.5f}"])
        with open(out / "noiseshare.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "step", "noise", "layer", "share", "shape", "flip"])
            w.writerows(rows_out)


if __name__ == "__main__":
    main()
