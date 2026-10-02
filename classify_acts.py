# Run `Ontologizer.classify` over a precomputed embedding cache and write the
# per-layer classification probabilities to a .npy.
#
# Output row i is the flattened stack of all `l` layers' classifications for
# row i of the input cache, laid out so that feature index
#
#     f = (l_i * h + h_i) * k + k_i
#
# matches the convention autointerp.py uses for Ontologizer tags. Values are
# the post-softmax probabilities `DictBlock.cluster` produces at `--temperature`
# (default 0.03, the annealed end temperature these runs train to), so each
# contiguous block of `k` sums to 1.
#
# The forward pass runs in float64 (`--compute-dtype`) even though the run
# trained in float32. This is not fussiness: with `forward="resid"` and
# `resid_norm`, layer i's input is (X - decode(R)) / ||X - decode(R)||, and the
# residual shrinks with depth, so float32 error in R is amplified by the
# normalization. In float32 the result is *batch-shape dependent* -- recomputing
# the same rows with a different batch size flips the argmax of 5.9% of layer-4
# head-slots (0.0 / 0.0002 / 0.004 / 0.019 / 0.059 by layer) because different
# matmul kernels round differently. In float64 that disagreement is exactly 0 at
# every layer. Compute cost is ~1.3 min for 780k rows, so there is no reason not
# to. Storage stays float32; only the intermediates need the headroom.
#
# Crash-safe and resumable at chunk granularity: rows are written into
# <out>.npy.part with a progress sidecar and renamed to <out>.npy only when
# complete, so a half-built file can never be mistaken for a finished one.

# disable preallocation so this can share the GPU with other jobs
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import json
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_CKPT = "data/out/sonar/multilingual/resid_nc_hm"
DEFAULT_EMB = "data/sonar_embeddings/act_i.npy"


def load_model(ckpt: Path, step):
    import orbax.checkpoint as ocp
    from ontologize.ontologizer import Ontologizer

    manager = ocp.CheckpointManager(
        ckpt.resolve(),
        checkpointers={'state': ocp.PyTreeCheckpointer(),
                       'spec': ocp.PyTreeCheckpointer()})
    step = step or manager.latest_step()
    spec = manager.restore(step, items={'spec': None})['spec']
    model = Ontologizer(**spec)
    state = manager.restore(step, items={'state': None})['state']
    params = state['params'] if 'opt_state' in state else state
    while 'params' in params:
        params = params['params']
    return model, {'params': params}, int(step), spec


def main():
    p = argparse.ArgumentParser(
        description="Write Ontologizer.classify outputs for an embedding cache.")
    p.add_argument("--ckpt", type=Path, default=Path(DEFAULT_CKPT),
                   help="checkpoint directory (default: %(default)s)")
    p.add_argument("--step", type=int, default=None,
                   help="checkpoint step (default: latest)")
    p.add_argument("--emb", type=Path, default=Path(DEFAULT_EMB),
                   help="input .npy embedding cache (default: %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="output .npy (default: <ckpt>/<emb stem>_classify.npy)")
    p.add_argument("--temperature", type=float, default=0.03,
                   help="softmax temperature (default: %(default)s)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="output dtype (default: %(default)s)")
    p.add_argument("--compute-dtype", default="float64",
                   choices=["float64", "float32"],
                   help="forward-pass dtype; float32 is batch-shape dependent "
                        "at depth, see the module docstring (default: %(default)s)")
    p.add_argument("-b", "--batch", type=int, default=1024)
    p.add_argument("-c", "--chunk", type=int, default=65536,
                   help="rows between flush + progress write")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="only process the first N rows (smoke test)")
    args = p.parse_args()

    import jax
    if args.compute_dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from ontologize.ontologizer import Ontologizer

    model, params, step, spec = load_model(args.ckpt, args.step)
    l, h, k = model.l, model.h, model.k
    T = args.temperature
    print(f"model l={l} h={h} k={k} step={step} -> {l * h * k} features/row",
          flush=True)

    # rebuild the model at the compute dtype and lift the restored params to
    # match; the checkpoint itself is untouched.
    cdt = jnp.float64 if args.compute_dtype == "float64" else jnp.float32
    if args.compute_dtype != spec.get("dtype_str"):
        spec_c = dict(spec)
        spec_c["dtype_str"] = args.compute_dtype
        spec_c["dtype_p_str"] = args.compute_dtype
        model = Ontologizer(**spec_c)
        params = jax.tree.map(lambda a: jnp.asarray(a, cdt), params)

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        _, Ps = module.classify(E, temperature=T, X_ref=X)      # (l, b, h*k)
        return jnp.transpose(Ps, (1, 0, 2))          # (b, l, h*k)

    @jax.jit
    def acts(X):
        return model.apply(params, X, method=probe).reshape(X.shape[0], -1)

    E_in = np.load(args.emb, mmap_mode="r")
    n = E_in.shape[0] if args.limit is None else min(args.limit, E_in.shape[0])
    d_feat = l * h * k
    dtype = np.dtype(args.dtype)

    out = args.out or (args.ckpt / f"{args.emb.stem}_classify.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".npy.part")
    prog = out.with_suffix(".progress.json")

    done = 0
    if part.exists() and prog.exists():
        prev = json.loads(prog.read_text())
        if prev.get("shape") == [n, d_feat] and prev.get("dtype") == args.dtype:
            done = int(prev.get("count", 0))
            print(f"resuming at row {done}", flush=True)
        else:
            print("progress sidecar does not match this run; starting over",
                  flush=True)
            part.unlink()

    mode = "r+" if part.exists() and done else "w+"
    P = np.lib.format.open_memmap(part, mode=mode, dtype=dtype,
                                  shape=(n, d_feat))

    with tqdm(total=n, initial=done, desc="classify", unit="row") as bar:
        for c0 in range(done, n, args.chunk):
            c1 = min(c0 + args.chunk, n)
            for j in range(c0, c1, args.batch):
                j1 = min(j + args.batch, c1)
                X = jnp.asarray(np.asarray(E_in[j:j1], dtype=np.float32), cdt)
                P[j:j1] = np.asarray(acts(X), dtype=dtype)
                bar.update(j1 - j)
            P.flush()
            prog.write_text(json.dumps(
                {"count": c1, "shape": [n, d_feat], "dtype": args.dtype}))

    P.flush()
    del P
    part.rename(out)
    meta = {"ckpt": str(args.ckpt.resolve()), "step": step,
            "emb": str(args.emb.resolve()), "rows": n,
            "l": l, "h": h, "k": k, "d_feat": d_feat,
            "temperature": T, "dtype": args.dtype,
            "compute_dtype": args.compute_dtype,
            "layout": "f = (l_i * h + h_i) * k + k_i",
            "spec": spec}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1,
                                                        default=str))
    prog.write_text(json.dumps(
        {"count": n, "shape": [n, d_feat], "dtype": args.dtype,
         "complete": True}))
    print(f"✓ {out}: {n} x {d_feat} {args.dtype}", flush=True)


if __name__ == "__main__":
    main()
