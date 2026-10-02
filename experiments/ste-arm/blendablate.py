"""What does the residual cascade contribute, beyond a second look?

Depth is worth 1.7x reconstruction at matched parameters, matched
nominal capacity, matched per-head entropy pressure and MORE realized
bits on the flat side. Everything that could have explained it has been
eliminated, which leaves "sequential dependence between stages" -- a
label rather than a measurement.

This measures it. Layer i+1 classifies `X - decode(R_i)`, so its input
is paired with what this sample's earlier layers already explained.
Blend that subtracted term toward a reference that carries the same
marginal distribution and none of the pairing, and sweep the blend:

  shuffle   decode(R_i) from a different row of the batch. Same
            marginals by construction, zero pairing at a=1.
  mean      the batch mean of decode(R_i). Removes the pairing and the
            spread at once, so it is the weaker control; reported
            alongside because the two disagreeing would be informative.
  noise     Gaussian, rescaled per sample so the perturbation has
            exactly the shuffle's magnitude. This is the control that
            makes the other two readable: feeding a classifier the
            wrong input produces wrong corrections whatever the wrong
            input is, so "breaking the pairing hurts" means nothing
            until it is measured against an equally large perturbation
            that carries no other sample's structure.

The weights never change, so at a=0 this is the trained model. The
classifier input's direction distribution moves as the blend rises --
a true residual is small and nearly isotropic precisely BECAUSE it is
this sample's own error -- so the geometry is reported next to the
error rather than pretended away. The endpoint is expected to overshoot
a flat arm, which was trained for its condition where this model is
being taken out of distribution; the shape of the curve is the result,
not the endpoint.
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
from jaxtyping import Array, Float, Int, PRNGKeyArray

from pareto import load_onto


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.00015)
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--mse-weights", default="data/out/sonar/mse_weights.npy")
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--b", type=int, default=512)
    p.add_argument("--blends", type=float, nargs="+",
                   default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def forward(module, X: Float[Array, "b d_in"], T: float,
            a: Float[Array, ""], mode: str, perm: Int[Array, "b"],
            rng: PRNGKeyArray
            ) -> Tuple[Float[Array, "b d_out"], Float[Array, "l b d"]]:
    """Trained forward with the cascade's pairing blended away.

    `a` is traced, so it is a scalar array rather than a float; `mode`
    is static, since the three references are different graphs.

    Mirrors `nextinput` for forward='resid' with resid_gain and
    resid_const on, which is every live arm: the residual is passed raw
    and the constant coordinate appended, and `gainshape_in` inside the
    DictEnc does the normalizing.
    """
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    Ein = module.constinput(E)
    Us = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        Us.append(U)
        P = de.dict.cluster(de.classifier(U), T)
        R = R + de.gained(de.dict.combine(de.dict.hfwd(P)), G)
        if i < module.l - 1:
            Y = module.decode(R)
            # `a` is traced, so the blend is unconditional; a=0 is the
            # identity arithmetically and leaves the trained model intact
            if mode == "shuffle":
                ref = Y[perm]
            elif mode == "mean":
                ref = jnp.broadcast_to(Y.mean(0, keepdims=True), Y.shape)
            else:
                g = jax.random.normal(jax.random.fold_in(rng, i), Y.shape,
                                      Y.dtype)
                shuf = jnp.linalg.norm(Y[perm] - Y, axis=-1, keepdims=True)
                g = g * shuf / (jnp.linalg.norm(g, axis=-1, keepdims=True)
                                + 1e-9)
                ref = Y + g
            Y = (1.0 - a) * Y + a * ref
            D = jax.lax.stop_gradient(X.astype(module.dtype) - Y)
            ones = jnp.ones(D.shape[:-1] + (1,), D.dtype)
            Ein = jnp.concatenate([D, ones], axis=-1)
    return module.decode(R), jnp.stack(Us)


def geom(U: Float[np.ndarray, "n d"]) -> float:
    """Effective dimension of a classifier input's direction cloud."""
    A = U[:, U.std(0) > 1e-8]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    ev = np.linalg.eigvalsh(np.cov(A.T)).clip(0)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))


def main() -> None:
    cfg = parse_args()
    model, raw, step = load_onto(cfg.model, cfg.step)
    params = {"params": raw}
    X = np.asarray(np.load(cfg.cache, mmap_mode="r")[-cfg.rows:], np.float32)
    rw = np.sqrt(np.load(cfg.mse_weights)).astype(np.float32)
    Xw = X * rw
    den = float(((Xw - Xw.mean(0)) ** 2).sum())
    rng = np.random.default_rng(cfg.seed)
    perm = jnp.asarray(rng.permutation(cfg.b))

    key = jax.random.PRNGKey(cfg.seed)
    f = jax.jit(lambda p, x, a, mode: forward(
        model.bind(p), x, cfg.temperature, a, mode, perm, key),
        static_argnums=(3,))

    print(f"{Path(cfg.model).name} step {step}  l={model.l} h={model.h}")
    print(f"{'blend':>7}{'shuffle':>10}{'mean':>10}{'noise':>10}"
          f"{'eff dim L1':>12}{'(noise)':>10}")
    for a in cfg.blends:
        out, dims, dims_n = [], None, None
        for mode in ("shuffle", "mean", "noise"):
            num, Us = 0.0, []
            for i in range(0, cfg.rows, cfg.b):
                xb = jnp.asarray(X[i:i + cfg.b])
                if xb.shape[0] != cfg.b:
                    continue
                Y, U = f(params, xb, float(a), mode)
                num += float((((np.asarray(Y) * rw)
                               - Xw[i:i + xb.shape[0]]) ** 2).sum())
                if len(Us) < 4:
                    Us.append(np.asarray(U))
            out.append(num / den)
            if mode in ("shuffle", "noise"):
                Uc = np.concatenate(Us, axis=1)
                d1 = geom(Uc[1]) if model.l > 1 else geom(Uc[0])
                if mode == "shuffle":
                    dims = d1
                else:
                    dims_n = d1
        print(f"{a:>7.3f}{out[0]:10.4f}{out[1]:10.4f}{out[2]:10.4f}"
              f"{dims:12.1f}{dims_n:10.1f}")


if __name__ == "__main__":
    main()
