"""Where the code capacity actually goes: per-prefix error and per-head
utilization.

Nominal capacity is `l * h * log2(k)` bits, and that is what a
capacity-matched comparison between architectures usually means. It is
not what the model carries. A head that only ever selects two of its 32
entries contributes one bit, not five, so the realized capacity is the
sum of the per-head argmax entropies over held-out rows. Comparing
`ste_h76` against `ste_l1_h380` on nominal bits reads as 1900 against
1900; on realized bits it is 1714 against 484.

The per-prefix FVU alongside it says how the cascade divides the error,
which is the other half of the same question: whether a layer that
fills its code is also a layer that explains much.

`heads.csv` from `headlang.py` cannot answer this. It stores each
model's top-`h` heads ranked by language NMI, which is 76 of 380 for a
stacked arm and all 380 for a flat one, so its entropy column compares
a language-selected subsample against a whole population.

For the utilization half, `loss.csv` already has the answer at every
step and this script is only a check on it. `hmean_kl` averages
`log2(k) - entropy(E_batch[p])` over a layer's heads and the loss sums
over layers, and with one-hot `ste` codes `E_batch[p]` *is* the usage
distribution, so

    realized bits = h * (l * log2(k) - KL_m)

On `ste_h76` that reads 1694 against the 1714 measured here, and on
`ste_l1_h380` 483 against 484; the residue is the training-time noise
and winner dropout the logged statistic sees and a clean forward does
not. The per-prefix FVU and the per-head distribution still need the
forward pass.

Utilization is measured on the argmax, so it is temperature-free and is
exactly the code a `select="ste"` model emits. Under `deepsup` the
prefix decodes come from the model's own forward, so the FVU column is
the same quantity `pareto.py` reports for the final prefix.

  uv run python experiments/ste-arm/codeuse.py \\
      --model data/out/sonar/multilingual/ste_h76 --temperature 0.00015
"""
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import sys
from pathlib import Path
from typing import Tuple

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
    p.add_argument("--model", required=True, help="checkpoint dir")
    p.add_argument("--step", type=int, default=0, help="0 = latest")
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=32768,
                   help="cache tail rows; the held-out split")
    p.add_argument("--b", type=int, default=512,
                   help="rows per forward; hfwd is (b, h, e_dec)")
    return p.parse_args()


def argmax_codes(module: Ontologizer, X: Float[Array, "b d_in"], T: float
                 ) -> Int[Array, "l b h"]:
    """Per-layer argmax of a clean forward, (l, b, h)."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    A = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        P = de.dict.cluster(de.classifier(U), T)
        A.append(jnp.argmax(P, -1))
        R = R + de.gained(de.dict.combine(de.dict.hfwd(P)), G)
        if i < module.l - 1:
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
    return jnp.stack(A)


def usage(counts: Int[np.ndarray, "l h k"]
          ) -> Tuple[Float[np.ndarray, "l h"], Int[np.ndarray, "l h"]]:
    """Per-head argmax entropy in bits and live-entry count."""
    p = counts / np.maximum(counts.sum(-1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(p > 0, p * np.log2(p), 0.0).sum(-1)
    return H, (p >= 0.005).sum(-1)


def main() -> None:
    cfg = parse_args()
    model, raw, step = load_onto(cfg.model, cfg.step)
    params = {"params": raw}
    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.rows:], np.float32)
    rw = (np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
          if cfg.mse_weights else np.float32(1.0))

    f_code = jax.jit(lambda p, x: argmax_codes(
        model.bind(p), x, cfg.temperature))
    f_dec = jax.jit(lambda p, x: model.apply(
        p, x, 0.0, None, temperature=cfg.temperature,
        method=Ontologizer.withStats)[0])

    l, h, k = model.l, model.h, model.k
    counts = np.zeros((l, h, k), np.int64)
    Xw = X * rw
    den = float(((Xw - Xw.mean(0)) ** 2).sum())
    num = None
    for i in range(0, cfg.rows, cfg.b):
        xb = jnp.asarray(X[i:i + cfg.b])
        A = np.asarray(f_code(params, xb))
        for li in range(l):
            for hi in range(h):
                counts[li, hi] += np.bincount(A[li, :, hi], minlength=k)
        Y = np.asarray(f_dec(params, xb))
        if Y.ndim == 2:
            Y = Y[None]
        e = ((Y * rw) - Xw[i:i + cfg.b][None]) ** 2
        num = e.sum((1, 2)) if num is None else num + e.sum((1, 2))

    H, live = usage(counts)
    nominal = l * h * np.log2(k)
    print(f"{Path(cfg.model).name} step {step}  {l}x{h} heads, k={k}, "
          f"T={cfg.temperature:g}")
    print(f"  per-head entropy  mean {H.mean():.3f}  median "
          f"{np.median(H):.3f}  of max {np.log2(k):.0f}")
    print(f"  live entries/head mean {live.mean():.1f}  median "
          f"{np.median(live):.0f}  of {k}")
    print(f"  heads under 1.5 bits: {int((H < 1.5).sum())} of {l * h}")
    print(f"  realized {H.sum():.0f} bits of {nominal:.0f} nominal "
          f"({H.sum() / nominal:.1%})")

    fvu = num / den
    print(f"\n{'layer':>6}{'prefix FVU_w':>14}{'added':>9}{'entropy':>9}"
          f"{'live':>7}{'bits':>8}")
    prev = 1.0
    for i in range(l):
        v = float(fvu[i]) if len(fvu) > i else float("nan")
        print(f"{i:>6}{v:14.4f}{prev - v:9.4f}{H[i].mean():9.3f}"
              f"{live[i].mean():7.1f}{H[i].sum():8.0f}")
        prev = v


if __name__ == "__main__":
    main()
