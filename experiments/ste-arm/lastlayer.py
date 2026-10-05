"""Does the collapsed final layer contribute anything?

Three reads on the cache tail pareto.py scores, all weighted FVU:

  prefix i   deepsup per-prefix decode after layers 0..i, so prefix 3 vs
             prefix 4 is exactly what the last layer adds
  unif       layer 4's heads uniform-ablated: its classification replaced
             by 1/k, so the layer emits its MEAN atom
  zero       layer 4's heads zero-ablated: the layer emits nothing

The discriminating comparison is unif against zero. A layer whose rows
are collinear AND equal-norm already emits a near-constant vector, so
forcing the mixture uniform changes little while removing it costs the
constant. A layer carrying real information loses under both.
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from ontologize.ontologizer import DictIntervention, Ontologizer
from ontologize.training.serialize import restore_spec

ROOT = Path("/home/keira/flock/ontologize/data/out/sonar/multilingual")
CACHE = "/home/keira/flock/ontologize/data/sonar_embeddings/mc4_4M.npy"
WEIGHTS = "/home/keira/flock/ontologize/data/out/sonar/mse_weights.npy"
ROWS, B = 32768, 256


def load(arm):
    p = ROOT / arm
    man = ocp.CheckpointManager(
        p.resolve(), checkpointers={"state": ocp.PyTreeCheckpointer(),
                                    "spec": ocp.PyTreeCheckpointer()})
    step = man.latest_step()
    spec = restore_spec(man, step)
    state = man.restore(step, items={"state": None})["state"]
    params = state["params"] if "opt_state" in state else state
    while "params" in params:
        params = params["params"]
    return Ontologizer(**spec), {"params": params}, spec, step


def main():
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    # these arms anneal to temperature_end over anneal_steps; evaluating at
    # the T=1.0 default instead puts a softmax arm at FVU_w 0.91
    T = next((float(a.split("=")[1]) for a in sys.argv[1:]
              if a.startswith("--temperature=")), 1.0)
    print(f"temperature {T}")
    mm = np.load(CACHE, mmap_mode="r")
    X = np.asarray(mm[-ROWS:], dtype=np.float32)
    w = np.load(WEIGHTS) if Path(WEIGHTS).exists() else np.ones(X.shape[1])
    base = float((X.var(0) * w).mean())
    wj = jnp.asarray(w, jnp.float32)
    print(f"cache tail {ROWS} rows, weighted base {base:.6g}\n")

    for arm in arms:
        model, params, spec, step = load(arm)
        h, l = spec["h"], spec["l"]
        allh = jnp.arange(h, dtype=jnp.uint32)
        none = [DictIntervention() for _ in range(l)]
        unif = [DictIntervention() for _ in range(l)]
        unif[l - 1] = DictIntervention(h_unif=allh)
        zero = [DictIntervention() for _ in range(l)]
        zero[l - 1] = DictIntervention(h_zero=allh)

        @jax.jit
        def runs(Xb):
            Y, _, _ = model.apply(params, Xb, temperature=T,
                                  method=Ontologizer.withStats)
            Yu, _, _, _ = model.apply(params, Xb, unif, temperature=T,
                                      method=Ontologizer.withArgs)
            Yz, _, _, _ = model.apply(params, Xb, zero, temperature=T,
                                      method=Ontologizer.withArgs)
            Yn, _, _, _ = model.apply(params, Xb, none, temperature=T,
                                      method=Ontologizer.withArgs)
            return Y, Yu, Yz, Yn

        acc = None
        n = 0
        for i in range(0, len(X) - B + 1, B):
            Xb = jnp.asarray(X[i:i + B])
            Y, Yu, Yz, Yn = runs(Xb)
            # Y is (l, b, d) under deepsup: the per-prefix decodes
            e = [float((((Y[j] - Xb) ** 2) * wj).mean()) for j in range(l)]
            e += [float((((q - Xb) ** 2) * wj).mean()) for q in (Yu, Yz, Yn)]
            acc = np.array(e) if acc is None else acc + np.array(e)
            n += 1
        e = acc / n / base
        pre = "  ".join(f"p{j}={e[j]:.4f}" for j in range(l))
        print(f"{arm}  select={spec['select']}  step={step}")
        print(f"  prefixes: {pre}")
        print(f"  layer {l-1} adds: {e[l-2] - e[l-1]:+.4f} FVU_w "
              f"({100*(e[l-2]-e[l-1])/e[l-2]:+.1f}% of prefix {l-2})")
        print(f"  full={e[l+2]:.4f}   L{l-1} uniform-ablated={e[l]:.4f} "
              f"({100*(e[l]/e[l+2]-1):+.1f}%)   zero-ablated={e[l+1]:.4f} "
              f"({100*(e[l+1]/e[l+2]-1):+.1f}%)\n")


if __name__ == "__main__":
    main()
