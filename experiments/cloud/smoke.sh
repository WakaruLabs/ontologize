#!/bin/bash
# VM side, after bootstrap.sh: does the environment work on this VM?
#   - JAX sees the TPU;
#   - on a multi-chip VM, one process per chip works (chips.py);
#   - how far a float32 matmul at each JAX precision lands from `highest`
#     (on a TPU the default is a single bfloat16 pass);
#   - the test suite, which runs on the CPU (tests/conftest.py); NO_TESTS=1 skips it.
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/cloud/env.sh
python -I -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"
N=$(python -I -c "import jax; print(jax.device_count())")
if [ "$N" -gt 1 ]; then python -I experiments/cloud/chips.py "$N"; fi
python -I - <<'PY'
import jax
import jax.numpy as jnp
import numpy as np
a = jax.random.normal(jax.random.PRNGKey(0), (4096, 1024), jnp.float32)
b = jax.random.normal(jax.random.PRNGKey(1), (1024, 1024), jnp.float32)
ref = np.asarray(jnp.matmul(a, b, precision="highest"))
for p in ("default", "high", "highest"):
    with jax.default_matmul_precision(p):
        y = np.asarray(jax.jit(jnp.matmul)(a, b))
    err = np.abs(y - ref).max() / np.abs(ref).max()
    print(f"matmul precision {p:8s} max error / max entry {err:.2e}")
PY
[ -n "${NO_TESTS:-}" ] || python -m pytest -q -x tests
