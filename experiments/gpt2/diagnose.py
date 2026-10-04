"""Diagnostics for an Ontologizer trained on the GPT-2 cache.

Per layer (on held-out eval rows, at the evaluation temperature):
  entropy     mean per-sample, per-head entropy of p = softmax(K/T), in bits
              (log2 k = uniform; 0 = one-hot)
  spread      mean per-head std over k of K/T (softmax is near-linear when << 1)
  maxp        mean per-head max probability; hard = fraction of heads with maxp > 0.9
  kl_mean     KL(E[p] || uniform) per head, bits (usage imbalance)
  n_norm      mean norm of the residual n_i = X - decode(R_{i-1}) the layer corrects
  c_norm      mean norm of the layer's contribution c_i = decode(R_i) - decode(R_{i-1})
  corr_nc     correlation of |n_i| and |c_i| across samples (does output size track
              the residual's size?)
  alpha_cv    coefficient of variation of the per-sample optimal gain
              alpha* = <n_i, c_i> / |c_i|^2 (0 = the layer already gets magnitudes right)
  corr_alpha_n correlation of alpha* with |n_i|
Plus FVU per prefix (soft), FVU of the end-to-end argmax code (hard), and
optionally splice CE (clean / mean-ablated / soft / hard, and loss recovered).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# JAX never returns its pool to the device; cap it so two concurrent runs
# plus GPT-2 (torch, for the splice eval) fit in 12 GB
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
# not "platform": it cudaMalloc/cudaFree-s every buffer each step (~1.7x slower here)

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
from common import Cache, Splicer, fvu  # noqa: E402


def forward_probe(module, X, T, hard=False, no_fiber=False):
    """Replicates `Ontologizer.classify` (no noise, no dropout), optionally
    with end-to-end argmax selection, and returns per-layer internals."""
    E, _ = module.encode(X)
    R = module.resid(E)
    E = module.first_input(X, E, R)
    layers = []
    for i, de in enumerate(module.dictencs):
        R0 = R
        g = module.gain(X, R0, i)
        n = X - module.decode(R0)
        K = de.logits(E)
        if hard:
            P = jax.nn.one_hot(jnp.argmax(K, -1), module.k, dtype=K.dtype)
        else:
            P = de.dict.cluster(K, T)
        S = de.scale(E) if de.scaled else None
        C = None if no_fiber else de.fiber_coords(E)
        Fs = de.dict.hfwd(P, S, C=C)
        R = module.apply_gain(R0, R0 + de.dict.combine(Fs), g)
        c = module.decode(R) - module.decode(R0)
        # per-head fiber norms (in the layer's unit frame), for the question
        # "do only a few heads per token receive a correction?"
        Fb = de.dict.fiber_out(P, C, per_head=True) if C is not None else None
        fn = jnp.linalg.norm(Fb, axis=-1) if Fb is not None else jnp.zeros(P.shape[:-1], K.dtype)
        layers.append((K, P, n, c, fn))
        if i < module.l - 1:
            E = module.nextinput(X, R, P.reshape(*P.shape[:-2], -1))
    return module.decode(R), layers


def layer_stats(K, P, n, c, T, fn=None):
    H = -(P * jnp.log2(P + 1e-12)).sum(-1)                    # (N, h)
    nn_ = jnp.linalg.norm(n, axis=-1)
    cn = jnp.linalg.norm(c, axis=-1)
    alpha = (n * c).sum(-1) / (cn ** 2 + 1e-12)
    out = {"entropy": H.mean(1), "spread": (K / T).std(-1).mean(1),
           "maxp": P.max(-1).mean(1), "hard": (P.max(-1) > 0.9).mean(1),
           "n_norm": nn_, "c_norm": cn, "alpha": alpha, "Pmean": P.mean(0)}
    if fn is not None:
        # how concentrated the correction is across heads, per token: the
        # share of the total fiber norm carried by the top-4 heads
        # (4/h = 0.125 if flat, 1.0 if only four heads are ever corrected)
        srt = jnp.sort(fn, axis=-1)[..., ::-1]
        out["fiber_top4_share"] = srt[..., :4].sum(-1) / (fn.sum(-1) + 1e-12)
        out["fiber_norm"] = fn.mean(1)
    return out


def make_fns(model, params, T):
    """Jitted probes. Params are passed as arguments, not closed over: closing
    over them makes XLA constant-fold the weights (slow compiles)."""
    @jax.jit
    def _stats(p, X):
        Y, layers = model.apply(p, X, T, method=forward_probe)
        return Y, [layer_stats(K, P, n, c, T, fn) for K, P, n, c, fn in layers]

    @jax.jit
    def _prefixes(p, X):
        Y, layers = model.apply(p, X, T, method=forward_probe)
        # decode(R_i) = X - n_{i+1}; the final prefix is Y
        return jnp.stack([X - n for _, _, n, _, _ in layers[1:]] + [Y])

    @jax.jit
    def _hard(p, X):
        return model.apply(p, X, T, True, method=forward_probe)[0]

    @jax.jit
    def _soft(p, X):
        return model.apply(p, X, T, method=forward_probe)[0]

    @jax.jit
    def _nofiber(p, X):
        return model.apply(p, X, T, False, True, method=forward_probe)[0]

    return (lambda X: _stats(params, X), lambda X: _prefixes(params, X),
            lambda X: _hard(params, X), lambda X: _soft(params, X),
            lambda X: _nofiber(params, X))


def summarize(model, params, X, T, bs=1024):
    soft_stats, prefixes, hard, _, nofiber = make_fns(model, params, T)
    acc = None
    pre, hrd, nof = [], [], []
    for i in range(0, len(X), bs):
        xb = jnp.asarray(X[i:i + bs])
        _, stats = soft_stats(xb)
        stats = jax.tree_util.tree_map(np.asarray, stats)
        acc = stats if acc is None else [
            {k: (np.concatenate([a[k], s[k]]) if k != "Pmean" else a[k] + s[k])
             for k in a} for a, s in zip(acc, stats)]
        pre.append(np.asarray(prefixes(xb)))
        hrd.append(np.asarray(hard(xb)))
        if model.fiber_rank:
            nof.append(np.asarray(nofiber(xb)))
    nb = -(-len(X) // bs)
    pre = np.concatenate(pre, 1)
    hrd = np.concatenate(hrd)
    out = {"T": T, "fvu_prefix": [fvu(X, p) for p in pre],
           "fvu_soft": fvu(X, pre[-1]), "fvu_hard": fvu(X, hrd), "layers": []}
    if nof:  # the discrete base alone (fiber zeroed)
        out["fvu_base_only"] = fvu(X, np.concatenate(nof))
    k = model.k
    for i, a in enumerate(acc):
        Pm = a["Pmean"] / nb
        kl = float((np.log2(k) - (-(Pm * np.log2(Pm + 1e-12)).sum(-1))).mean())
        al = a["alpha"]
        row = {"layer": i, "entropy": float(a["entropy"].mean()),
               "spread": float(a["spread"].mean()), "maxp": float(a["maxp"].mean()),
               "hard": float(a["hard"].mean()), "kl_mean": kl,
               "n_norm": float(a["n_norm"].mean()), "c_norm": float(a["c_norm"].mean()),
               "corr_nc": float(np.corrcoef(a["n_norm"], a["c_norm"])[0, 1]),
               "alpha_mean": float(al.mean()), "alpha_cv": float(al.std() / abs(al.mean())),
               "corr_alpha_n": float(np.corrcoef(al, a["n_norm"])[0, 1])}
        if "fiber_top4_share" in a and model.fiber_rank:
            row["fiber_top4_share"] = float(a["fiber_top4_share"].mean())
            row["fiber_norm"] = float(a["fiber_norm"].mean())
        out["layers"].append(row)
    return out


def chunked(fn, bs):
    """Apply fn to rows in chunks of bs (wide encoders don't fit 4096 rows at once)."""
    return lambda x: np.concatenate([np.asarray(fn(jnp.asarray(x[i:i + bs]))) for i in range(0, len(x), bs)])


def splice_ce(model, params, cache, T, n_windows=512, bs=1024, splice_batch=32):
    _, _, hard, soft, _ = make_fns(model, params, T)
    cpu = jax.default_backend() == "cpu"   # keep GPT-2 beside the model
    sp = Splicer(cache, n_windows=n_windows, batch=8 if cpu else splice_batch, device="cpu" if cpu else "cuda")
    return sp.evaluate({"soft": chunked(soft, bs), "hard": chunked(hard, bs)})


def print_summary(s, ce=None):
    print(f"T={s['T']:.3g}  FVU soft {s['fvu_soft']:.4f}  hard {s['fvu_hard']:.4f}  "
          + (f"base-only {s['fvu_base_only']:.4f}  " if "fvu_base_only" in s else "")
          + f"prefixes {[round(f, 4) for f in s['fvu_prefix']]}")
    cols = ["entropy", "spread", "maxp", "hard", "kl_mean", "n_norm", "c_norm",
            "corr_nc", "alpha_mean", "alpha_cv", "corr_alpha_n"]
    if "fiber_top4_share" in s["layers"][0]:
        cols += ["fiber_norm", "fiber_top4_share"]
    print("layer " + " ".join(f"{c:>11}" for c in cols))
    for r in s["layers"]:
        print(f"{r['layer']:5d} " + " ".join(f"{r[c]:11.4f}" for c in cols))
    if ce:
        print("CE " + "  ".join(f"{k} {v:.4f}" for k, v in ce.items()))


def load(run_dir, step=None):
    import orbax.checkpoint as ocp
    from ontologize.ontologizer import Ontologizer
    from ontologize.training.ontostate import state_init
    import optax
    mgr = ocp.CheckpointManager(os.path.abspath(run_dir), checkpointers={
        "state": ocp.PyTreeCheckpointer(), "spec": ocp.PyTreeCheckpointer()})
    step = mgr.latest_step() if step is None else step
    spec = mgr.restore(step, items={"spec": None})["spec"]
    model = Ontologizer(**spec)
    # init and restore on the host with explicit targets: the checkpoints carry
    # GPU sharding metadata (unusable from a CPU process), and a wide model's
    # optimizer state doesn't fit on the GPU beside a training run
    with jax.default_device(jax.devices("cpu")[0]):
        state = state_init(model, 8, optax.adam(1e-3), jax.random.PRNGKey(0), ghost=False)
        meta = mgr.item_metadata(step)["state"]
        if "opt_state" in meta:
            # params only: the optimizer state may have a different structure
            # (e.g. multi_transform for --train-only) and is not needed here
            target = {"params": state.params}
            ra = ocp.checkpoint_utils.construct_restore_args(target)
            params = mgr.restore(step, items={"state": target},
                                 restore_kwargs={"state": {"restore_args": ra, "partial_restore": True}}
                                 )["state"]["params"]
        else:  # old format: params only
            ra = ocp.checkpoint_utils.construct_restore_args(state.params)
            params = mgr.restore(step, items={"state": state.params},
                                 restore_kwargs={"state": {"restore_args": ra}})["state"]
    params = jax.device_put(params, jax.devices()[0])
    return model, params, step


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--cache", default="data/gpt2_l8")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--T", type=float, default=None,
                   help="evaluation temperature (default: the schedule's T at this step)")
    p.add_argument("--rows", type=int, default=65536)
    p.add_argument("--ce", action="store_true", help="also compute splice CE")
    p.add_argument("--bs", type=int, default=1024, help="eval batch (rows); lower it for wide encoders")
    p.add_argument("--splice-batch", type=int, default=32, help="GPT-2 windows per splice batch")
    cfg = p.parse_args()
    cache = Cache(cfg.cache)
    model, params, step = load(cfg.run, cfg.step)
    T = cfg.T
    if T is None:  # the training schedule's temperature at this checkpoint
        rc = json.loads((Path(cfg.run) / "config.json").read_text())
        f = min(step / rc["anneal_steps"], 1.0) if rc["anneal_steps"] else 1.0
        T = rc["temperature"] * (rc["temperature_end"] / rc["temperature"]) ** f
    X = cache.eval_rows(cfg.rows)
    s = summarize(model, params, X, T, bs=cfg.bs)
    s["step"] = step
    ce = splice_ce(model, params, cache, T, bs=cfg.bs, splice_batch=cfg.splice_batch) if cfg.ce else None
    if ce:
        s["ce"] = ce
    print_summary(s, ce)
    suffix = "" if cfg.T is None else f"_T{cfg.T:g}"   # explicit T: don't clobber the run's record
    (Path(cfg.run) / f"diag_{step}{suffix}.json").write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
