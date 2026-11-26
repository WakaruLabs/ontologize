"""Reference fixtures for readers of a categorical dictionary.

Each fixture is a small trained model, a few input rows, and what this
repository's JAX forward computes on them at every layer. They are written
as plain files that a reader in any framework can load (README.md). Every
fixture directory holds:

  spec.json     the model's fields, as a checkpoint stores them
  params.npz    the parameter tree, keys joined by "/"
  inputs.npz    X, the raw input rows
  expected.npz  per layer, under "L<l>/", what the forward computes there,
                and "Y", the model's output
  meta.json     the temperature, the tolerances, which rows are which, and
                how the fixture was made

`trace` is the forward, layer by layer, as DictStack.classify computes it;
`build` checks that its last reconstruction is the model's own output.
Regenerate with

  uv run python tests/reference/build.py

and review the diff. tests/test_reference.py fails whenever the forward no
longer reproduces these files.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from jaxtyping import Float  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

D = 16
FIXTURES = {
    "legacy_const0": dict(
        about="a checkpoint from before `const0` existed, shaped like "
              "resid_nc: soft selection, no gain-shape, and no constant "
              "coordinate at layer 0. Its spec lacks the field, and the "
              "fields added after it, so migrate_spec must infer const0 "
              "from the classifier's width",
        model=dict(d_in=D, d_out=D, e_dec=32, k=8, h=4, l=2, n=2,
                   gate="none", select="softmax", forward="resid",
                   deepsup=True, resid_norm=True, resid_const=True,
                   const0=False, resid_gain=False),
        temperature=0.03, anneal=True, steps=1500,
        drop_spec=("const0", "resid_first", "gain_clip", "logit_norm")),
    "direct_gain_ste": dict(
        about="headline's V_direct_gain_ste design, small: signed entries "
              "in the output space with no decoder, gain-shape coding of "
              "every layer's residual including layer 0's (resid_first), "
              "each gain clipped at the row's norm (gain_clip), hard "
              "selection. The model is positively homogeneous: f(cX) = "
              "c f(X), with the same codes",
        model=dict(d_in=D, d_out=D, e_dec=D, k=8, h=4, l=3, n=2, gate="none",
                   select="ste", forward="resid", deepsup=True,
                   resid_norm=True, resid_const=True, resid_gain=True,
                   resid_first=True, gain_clip=True, direct=True,
                   signed=True),
        temperature=0.03, anneal=True, steps=1500, drop_spec=()),
}
# a float32 reader must reproduce every array to these tolerances; atol is
# relative to the array's largest magnitude
RTOL, ATOL_REL = 1e-4, 1e-5
# the near tie's top-two logit gap, over the head's largest logit
# magnitude: far above float32's error, below float16's resolution
NEAR_TIE_REL = 2e-4


# ---- the forward ------------------------------------------------------------

def trace(module, X, T: float):
    """Per layer: the classifier's input U and the gain G (None without
    gain-shape), the logits K before the temperature, the selection P at T,
    the router gains S and the fiber coordinates C (None without them), and
    the reconstruction after the layer, with and without the fibers. The
    same steps as DictStack.classify, with nothing applied that training
    alone applies."""
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    R_b = R
    Ein = module.constinput(E)
    out = []
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(Ein)
        K = de.classifier(U)
        P = de.dict.cluster(K, T)
        S = de.scale(U) if de.scaled else None
        C = de.fiber_coords(U)
        R = R + de.gained(de.dict.combine(de.dict.hfwd(P, S, C=C)), G)
        R_b = R_b + de.gained(de.dict.combine(de.dict.hfwd(P, S)), G)
        out.append(dict(U=U, G=G, K=K, P=P, S=S, C=C, Y=module.decode(R),
                        Y_base=module.decode(R_b)))
        if i < module.l - 1:
            Ein = module.nextinput(X, R, P.reshape(X.shape[0], -1))
    return out


def run(model, params: dict, X: Float[np.ndarray, "n d"], T: float
        ) -> Dict[str, np.ndarray]:
    """`trace` as the arrays expected.npz holds."""
    import jax.numpy as jnp
    layers = model.apply({"params": params}, jnp.asarray(X), T,
                         method=trace)
    res = {}
    for l, o in enumerate(layers):
        K = np.asarray(o["K"], np.float32)
        top = np.argsort(-K, axis=-1, kind="stable")[..., :2]
        res[f"L{l}/top2_index"] = top.astype(np.int32)
        res[f"L{l}/top2_logit"] = np.take_along_axis(K, top, -1)
        for key in ("U", "G", "K", "P", "S", "C", "Y", "Y_base"):
            if o[key] is None or (key == "Y_base" and o["C"] is None):
                continue
            res[f"L{l}/{key}"] = np.asarray(o[key], np.float32)
    res["Y"] = res[f"L{len(layers) - 1}/Y"]
    return res


# ---- files --------------------------------------------------------------------

def flatten(tree, prefix: str = "") -> Dict[str, np.ndarray]:
    out = {}
    for k, v in tree.items():
        key = f"{prefix}/{k}" if prefix else k
        if hasattr(v, "items"):
            out.update(flatten(v, key))
        else:
            out[key] = np.asarray(v)
    return out


def unflatten(flat: Dict[str, np.ndarray]) -> dict:
    tree = {}
    for key, v in flat.items():
        node = tree
        *path, leaf = key.split("/")
        for p in path:
            node = node.setdefault(p, {})
        node[leaf] = v
    return tree


def load(name: str) -> Tuple[object, dict, np.ndarray, dict, dict]:
    """A fixture's model (built as a loader builds one: from the spec,
    migrated with the stored parameters), parameters, inputs, expected
    arrays and metadata."""
    import jax.numpy as jnp
    from ontologize.ontologizer import Ontologizer
    from ontologize.training.serialize import migrate_spec
    d = HERE / name
    spec = json.loads((d / "spec.json").read_text())
    with np.load(d / "params.npz") as f:
        params = unflatten({k: jnp.asarray(f[k]) for k in f.files})
    model = Ontologizer(**migrate_spec(spec, params))
    with np.load(d / "inputs.npz") as f:
        X = f["X"]
    with np.load(d / "expected.npz") as f:
        expected = {k: f[k] for k in f.files}
    meta = json.loads((d / "meta.json").read_text())
    return model, params, X, expected, meta


# ---- building -----------------------------------------------------------------

def planted(rng: np.random.Generator, n: int, A, B) -> Float[np.ndarray, "n d"]:
    """Rows that sum one atom of A and one of B, plus noise, at gains from
    0.5 to 1.5: two factors and a scale for the heads and the gain to read."""
    X = A[rng.integers(0, len(A), n)] + B[rng.integers(0, len(B), n)] \
        + 0.1 * rng.normal(size=(n, A.shape[1]))
    return (X * rng.uniform(0.5, 1.5, (n, 1))).astype(np.float32)


def train(fx: dict, X: Float[np.ndarray, "n d"], seed: int):
    import jax
    import jax.numpy as jnp
    from ontologize.training.config import Hyperparams
    from ontologize.training.ontostate import schedules, update
    T, steps, b = fx["temperature"], fx["steps"], 256
    anneal = fx["anneal"]
    hyper = Hyperparams(
        D, D, b, epochs=1, lr=1e-3, wd=0.0, temperature=1.0 if anneal else T,
        noise_in="batchnorm", noise_K="normal", noise_F="featvar",
        sd_in=0.0, sd_K=0.02, sd_F=0.1, p_drop=0.1, p_revive=0.0,
        s_L1F=0.0, s_bcossim=0.0, s_hcossim=0.0, s_H=0.0, s_kcossim=0.0,
        s_Hm=0.0, seed=seed, ghost=False,
        temperature_end=T if anneal else None,
        anneal_steps=int(0.8 * steps) if anneal else 0,
        p_drop_start=0.0 if anneal else None, sd_K_end=None)
    model = hyper.ontologizer(**fx["model"], dtype_str="float32",
                              dtype_p_str="float32")
    state = hyper.init(model, save_each=steps)
    rng = hyper.rng()
    order = np.random.default_rng(seed)
    perm, n_b = order.permutation(len(X)), len(X) // b
    for step in range(steps):
        if step and step % n_b == 0:
            perm = order.permutation(len(X))
        i = (step % n_b) * b
        t, pd, sk, _ = schedules(
            step, hyper.anneal_steps, hyper.temperature,
            hyper.temperature_end, hyper.p_drop, hyper.p_drop_start,
            hyper.sd_K, hyper.sd_K_end)
        rng, r = jax.random.split(rng)
        state, L, _ = update(
            state, hyper.loss, r, jnp.asarray(X[perm[i:i + b]]),
            jnp.asarray(X[perm[i:i + b]]), temperature=t, p_drop=pd,
            sd_K=sk, sd_in=0.0, sd_F=hyper.sd_F, grad_clip=hyper.grad_clip,
            p_revive=0.0, revive_frac=hyper.revive_frac)
    return model, state.params["params"], float(L)


def gaps(K: Float[np.ndarray, "n h k"]) -> Float[np.ndarray, "n h"]:
    """Each head's top-two logit gap over its largest logit magnitude."""
    s = np.sort(K, axis=-1)
    return (s[..., -1] - s[..., -2]) / np.abs(K).max(-1)


def near_tie(model, params, T: float, pool: Float[np.ndarray, "n d"],
             head: int = 0) -> Tuple[np.ndarray, float]:
    """A row on the segment between two pool rows that layer 0's `head`
    gives different entries, placed where that head's top-two gap is
    NEAR_TIE_REL of its largest logit: bisect for the boundary, then for
    the gap on the first row's side of it."""
    def K0(x):
        return run(model, params, np.atleast_2d(x).astype(np.float32),
                   T)["L0/K"][:, head]
    win = K0(pool).argmax(-1)
    a, b = 0, int(np.flatnonzero(win != win[0])[0])
    seg = lambda t: (1 - t) * pool[a] + t * pool[b]  # noqa: E731
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if K0(seg(mid))[0].argmax() == win[a] else (lo, mid)
    top = 0.0
    for _ in range(60):                 # the gap shrinks toward the boundary
        mid = (top + lo) / 2
        k = K0(seg(mid))[0]
        g = np.diff(np.sort(k)[-2:])[0] / np.abs(k).max()
        top, lo = (mid, lo) if g > NEAR_TIE_REL else (top, mid)
    x = seg(top).astype(np.float32)
    k = K0(x)[0]
    return x, float(np.diff(np.sort(k)[-2:])[0] / np.abs(k).max())


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(name: str, fx: dict, seed: int = 0) -> None:
    import flax
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    A, B = rng.normal(size=(8, D)), rng.normal(size=(8, D)) * 0.5
    model, params, L = train(fx, planted(rng, 8192, A, B), seed)
    T = fx["temperature"]
    pool = planted(rng, 512, A, B)
    K0 = run(model, params, pool, T)["L0/K"]
    clear = int(gaps(K0).min(-1).argmax())
    tie, _ = near_tie(model, params, T, pool)
    X = np.stack([pool[1], pool[2], pool[3], pool[4], pool[clear], tie,
                  0.5 * pool[1], 3.0 * pool[1]]).astype(np.float32)
    expected = run(model, params, X, T)
    # the gap as the stored batch has it: float32's order of summation moves
    # it by about a thousandth of itself between a row alone and in a batch
    tie_gap = float(gaps(expected["L0/K"][5:6])[0, 0])
    own = np.asarray(model.apply({"params": params}, jnp.asarray(X),
                                 temperature=T))
    if np.abs(own - expected["Y"]).max() > 1e-5 * max(1.0, np.abs(own).max()):
        raise RuntimeError(f"{name}: trace does not reproduce the model's "
                           "own output")
    spec = {k: v for k, v in dataclasses.asdict(model).items()
            if k not in fx["drop_spec"]}
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1) + "\n")
    np.savez(out / "params.npz", **flatten(jax.device_get(params)))
    np.savez(out / "inputs.npz", X=X)
    np.savez(out / "expected.npz", **expected)
    meta = {
        "about": fx["about"], "temperature": T,
        "rtol": RTOL, "atol_rel": ATOL_REL,
        "rows": {"held_out": [0, 1, 2, 3], "clear_winner": 4,
                 "near_tie": 5, "rescaled_0.5": 6, "rescaled_3": 7},
        "rescaled_of": 0,
        "near_tie": {"layer": 0, "head": 0, "gap_over_max_logit": tie_gap},
        "clear_winner_min_gap_over_max_logit": float(gaps(K0[clear]).min()),
        "spec_without": list(fx["drop_spec"]),
        "made": {"commit": commit(), "seed": seed, "steps": fx["steps"],
                 "final_loss": L, "jax": jax.__version__,
                 "flax": flax.__version__,
                 "data": "planted: one of 8 atoms N(0, 1) plus one of 8 "
                         "N(0, 0.25), noise SD 0.1, gains U(0.5, 1.5), d 16"},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{name}: loss {L:.4g}; near tie {tie_gap:.2e}; clear winner's "
          f"least gap {meta['clear_winner_min_gap_over_max_logit']:.3f}")


def main() -> None:
    names = sys.argv[1:] or list(FIXTURES)
    for name in names:
        build(name, FIXTURES[name])


if __name__ == "__main__":
    main()
