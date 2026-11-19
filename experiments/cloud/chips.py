"""Check that independent processes can share a multi-chip TPU VM: start N
processes at once, each confined to one chip by `runq.chip_env`, and check
that each sees exactly one device and can run a matmul. If this fails on a
VM type, give it one queue (`restart.sh` with ONTO_CHIPS=0).

    python -I experiments/cloud/chips.py N
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runq import chip_env  # noqa: E402

CODE = ("import jax, jax.numpy as jnp; x = jnp.ones((1024, 1024)); "
        "y = float((x @ x).sum()); d = jax.devices(); "
        "print('CHIP', len(d), d[0].platform, y, flush=True)")


def main(n: int) -> int:
    procs = [subprocess.Popen([sys.executable, "-c", CODE],
                              env=dict(os.environ, **chip_env(i)),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) for i in range(n)]
    ok = True
    for i, p in enumerate(procs):
        out, _ = p.communicate(timeout=600)
        lines = [s for s in out.splitlines() if s.startswith("CHIP")]
        fields = lines[-1].split() if lines else []
        good = p.returncode == 0 and fields[1:3] == ["1", "tpu"]
        print(f"chip {i}: rc {p.returncode} "
              f"{lines[-1] if lines else out[-500:]}")
        ok &= good
    print("CHIPS", "PASS" if ok else "FAIL: run one queue (ONTO_CHIPS=0)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1])))
