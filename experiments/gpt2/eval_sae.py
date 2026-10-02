"""FVU, L0 and splice CE for SAEs trained by sae.py on the GPT-2 cache.

  uv run python sae.py --cache data/gpt2_l8/acts.npy --eval-rows 520192 \\
      --mse-weights "" --m 5120 --topk 32 --out data/out/gpt2/sae/m5120_k32
  uv run python experiments/gpt2/eval_sae.py data/out/gpt2/sae/m5120_k32
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
import sae  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--cache", default="data/gpt2_l8")
    p.add_argument("--rows", type=int, default=65536)
    cfg = p.parse_args()
    run = Path(cfg.run)
    meta = json.loads((run / "meta.json").read_text())
    params = {k: jnp.asarray(v) for k, v in np.load(run / "params.npz").items()}
    topk, groups, group_fn = meta["topk"], meta["groups"], meta["group_fn"]

    @jax.jit
    def recon(p, X):
        z = sae.encode(p, X, topk, groups, group_fn)
        return sae.decode(p, z), (z > 0).sum(-1)

    cache = Cache(cfg.cache)
    X = cache.eval_rows(cfg.rows)
    Y, L0 = [], []
    for i in range(0, len(X), 8192):
        y, l0 = recon(params, jnp.asarray(X[i:i + 8192]))
        Y.append(np.asarray(y))
        L0.append(np.asarray(l0))
    Y, L0 = np.concatenate(Y), np.concatenate(L0)
    out = {"run": str(run), "meta": meta, "fvu": fvu(X, Y), "l0": float(L0.mean())}
    sp = Splicer(cache)
    out["ce"] = sp.evaluate({"sae": lambda x: np.asarray(recon(params, jnp.asarray(x))[0])})
    print(json.dumps(out, indent=2))
    (run / "eval_gpt2.json").write_text(json.dumps(out, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
