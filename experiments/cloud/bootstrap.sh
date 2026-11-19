#!/bin/bash
# VM side: Python 3.13 and the repository's locked environment, installed into
# <repo>/.venv for a TPU (or a CPU-only box). restart.sh runs it once; it is
# idempotent, so a rerun installs only what is missing.
#   ONTO_BACKEND=tpu|cpu   (default: tpu when the VM has TPU devices)
# The packages are uv.lock's, minus the CUDA and torch stack (tpureqs.py), plus
# jax[tpu] and a CPU torch at the locked versions, plus this package (editable).
# Never `uv run` or `uv sync` on the VM: they would install the lock's CUDA stack.
# Writes experiments/cloud/env.sh, which puts .venv/bin first on PATH, so a job
# command runs `python ...`.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
BACKEND=${ONTO_BACKEND:-auto}
if [ "$BACKEND" = auto ]; then
  if compgen -G "/dev/accel*" >/dev/null || compgen -G "/dev/vfio/[0-9]*" >/dev/null; then
    BACKEND=tpu
  else
    BACKEND=cpu
  fi
fi
echo "bootstrap: backend $BACKEND, repo $REPO"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH
uv python install 3.13
[ -x .venv/bin/python ] || uv venv --python 3.13 .venv
PY=$REPO/.venv/bin/python
REQ=$(mktemp -d)
uv export --frozen --no-hashes --no-dev --no-emit-project -o "$REQ/locked.txt" >/dev/null
$PY -I experiments/cloud/tpureqs.py requirements "$REQ/locked.txt" > "$REQ/tpu.txt"
JAX=$($PY -I experiments/cloud/tpureqs.py version uv.lock jax)
TORCH=$($PY -I experiments/cloud/tpureqs.py version uv.lock torch)
uv pip install --python "$PY" -r "$REQ/tpu.txt"
if [ "$BACKEND" = tpu ]; then
  uv pip install --python "$PY" "jax[tpu]==$JAX" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi
uv pip install --python "$PY" "torch==$TORCH" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" --no-deps -e .
rm -rf "$REQ"
cat > experiments/cloud/env.sh <<ENV
export ONTO_PY=$PY
export ONTO_BACKEND=$BACKEND
export VIRTUAL_ENV=$REPO/.venv
export PATH=$REPO/.venv/bin:\$PATH
ENV
$PY -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"
free -g | head -2; nproc; df -h . | tail -1
