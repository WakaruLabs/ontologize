"""How much does the gain carry, outside the discrete code?

`gainshape_in` splits a layer's input into a unit direction and its
norm. The classifier only ever sees the direction, so the code is
discrete. But `gained` multiplies the layer's OUTPUT by that norm, so
one real number per layer per sample reaches the decoder without
passing through the bottleneck. A "1900-bit code" accounting misses it.

It is not uniform across arms. SONAR embeddings are exactly unit-norm,
so with no encoder a layer-0 gain is identically 1 and carries nothing.
Residual layers see a varying norm, and a whitened encoder makes
layer 0's vary too.

Ablation: replace a layer's gain with its held-out mean, a constant,
and measure the held-out FVU_w that costs. Per layer and all at once.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from pareto import load_onto
from ontologize.ontologizer import Ontologizer

CACHE = "data/sonar_embeddings/mc4_4M.npy"
W_PATH = "data/out/sonar/mse_weights.npy"
ROWS, B = 16384, 128


def build(model: Ontologizer, T: float, const: Sequence[Optional[float]]
          ) -> Callable[..., Tuple[Float[Array, "b d_out"],
                                   Float[Array, "l b 1"]]]:
    """Forward with each layer's gain optionally pinned to a constant.
    `const[i]` is None to keep the measured gain."""
    def fwd(module, X: Float[Array, "b d_in"]
            ) -> Tuple[Float[Array, "b d_out"], Float[Array, "l b 1"]]:
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = module.constinput(E)
        gains = []
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(Ein)
            gains.append(G)
            P = de.dict.cluster(de.classifier(U), T)
            g = G if const[i] is None else jnp.full_like(G, const[i])
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), g)
            if i < module.l - 1:
                Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
        return module.decode(R), jnp.stack(gains)
    return fwd


def run(model: Ontologizer, params: PyTree, X: Float[np.ndarray, "n d_in"],
        rw: Float[np.ndarray, "d_out"], T: float,
        const: Sequence[Optional[float]]
        ) -> Tuple[float, Float[np.ndarray, "l n 1"]]:
    """(held-out FVU_w, the measured gains) for one pinning."""
    f = jax.jit(lambda p, x: build(model, T, const)(model.bind(p), x))
    Xw = X * rw
    den = float(((Xw - Xw.mean(0)) ** 2).sum())
    num, gs = 0.0, []
    for i in range(0, len(X), B):
        Y, G = f(params, jnp.asarray(X[i:i + B]))
        num += float((((np.asarray(Y) * rw) - Xw[i:i + B]) ** 2).sum())
        gs.append(np.asarray(G))
    return num / den, np.concatenate(gs, axis=1)


def main(ckpt: str, T: float) -> None:
    model, raw, step = load_onto(ckpt, 0)
    params = {"params": raw}
    X = np.asarray(np.load(CACHE, mmap_mode="r")[-ROWS:], np.float32)
    rw = np.sqrt(np.load(W_PATH)).astype(np.float32)
    l = model.l

    base, G = run(model, params, X, rw, T, [None] * l)
    Gm = G.reshape(l, -1).mean(1)
    Gs = G.reshape(l, -1).std(1)
    print(f"{Path(ckpt).name} step {step}  l={l}  baseline FVU_w {base:.4f}")
    print(f"{'layer':>6}{'gain mean':>11}{'gain sd':>10}{'sd/mean':>9}"
          f"{'FVU pinned':>12}{'cost':>9}")
    for i in range(l):
        c = [None] * l
        c[i] = float(Gm[i])
        f, _ = run(model, params, X, rw, T, c)
        print(f"{i:>6}{Gm[i]:11.4f}{Gs[i]:10.4f}{Gs[i] / Gm[i]:9.1%}"
              f"{f:12.4f}{(f - base) / base:+9.1%}")
    if l > 1:
        f, _ = run(model, params, X, rw, T, [float(x) for x in Gm])
        print(f"{'all':>6}{'':11}{'':10}{'':9}{f:12.4f}{(f - base) / base:+9.1%}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.5e-4)
