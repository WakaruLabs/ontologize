"""Time one training step of the GPT-2 Ontologizer config under variations,
to find where the step time goes."""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# not "platform": it cudaMalloc/cudaFree-s every buffer each step (~1.7x slower here)
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from ontologize.training.config import Hyperparams  # noqa: E402
from ontologize.training.ontostate import update  # noqa: E402


def bench(label, b=256, noise_F="featvar", sd_F=0.1, l=5, deepsup=True, steps=30, **mk):
    d = 768
    hyper = Hyperparams(d, d, b, 1, 5e-5, 0.0, 1.0, "none", "normal", noise_F,
                        0.0, 0.02, sd_F, s_L1F=1e-9, s_bcossim=1e-5,
                        s_hcossim=1e-6, s_Hm=1e-6, p_drop=0.1, ghost=False)
    model = hyper.ontologizer(d, d, 2 * d, 32, 32, l, n=2, gate="none",
                              forward="resid", deepsup=deepsup, resid_norm=True,
                              resid_const=True, dtype_str="float32",
                              dtype_p_str="float32", **mk)
    state = hyper.init(model, save_each=1000)
    X = jax.random.normal(jax.random.PRNGKey(0), (b, d)) / np.sqrt(d)
    rng = jax.random.PRNGKey(1)
    kw = dict(temperature=0.5, sd_in=0.0, sd_K=0.02, sd_F=sd_F, p_drop=0.1,
              grad_clip=1.0)
    for _ in range(3):
        state, L, rng = update(state, hyper.loss, rng, X, X, **kw)
    L.block_until_ready()
    t = time.time()
    for _ in range(steps):
        state, L, rng = update(state, hyper.loss, rng, X, X, **kw)
    L.block_until_ready()
    ms = (time.time() - t) / steps * 1e3
    print(f"{label:38s} b={b:5d}  {ms:7.2f} ms/step  {b / ms * 1e3:9,.0f} rows/s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ref"):
        bench("reference stats")
    bench("fast_stats", fast_stats=True)
    bench("fast_stats", b=1024, fast_stats=True)
    bench("fast_stats", b=4096, steps=10, fast_stats=True)
    bench("fast_stats l=1", l=1, fast_stats=True)
