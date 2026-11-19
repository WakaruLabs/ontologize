"""A TPU VM's requirements: the repository's locked versions, minus the CUDA
and torch stack.

`bootstrap.sh` exports `uv.lock` with `uv export` and passes the export
through `requirements`, which drops the CUDA packages, the jax CUDA plugins,
torch and torch-xla. It then installs `jax[tpu]` and a CPU build of torch,
at the versions `version` reads from the same lock. torch is only imported
by shared code on a TPU VM; it never computes there.

    python tpureqs.py requirements <uv export output>    # filtered, to stdout
    python tpureqs.py version <uv.lock> <package>        # its locked version
"""
import re
import sys
import tomllib
from pathlib import Path

DROP = re.compile(r"(nvidia-.*|cuda-.*|jax-cuda.*|torch|torch-xla|triton)")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def requirements(text: str) -> str:
    """`uv export` output without the packages a TPU VM replaces, along with
    the "# via" comment lines that follow each dropped package."""
    out, dropping = [], False
    for line in text.splitlines():
        if line[:1].isspace():               # a comment under the last package
            if not dropping:
                out.append(line)
            continue
        m = NAME.match(line)
        dropping = bool(m) and DROP.fullmatch(m.group(0).lower()) is not None
        if not dropping:
            out.append(line)
    return "\n".join(out) + "\n"


def version(lock: str, package: str) -> str:
    """The version `uv.lock` pins for `package`."""
    for p in tomllib.loads(lock)["package"]:
        if p["name"] == package:
            return p["version"]
    raise KeyError(f"{package} is not in the lock")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "requirements":
        sys.stdout.write(requirements(Path(sys.argv[2]).read_text()))
    elif len(sys.argv) == 4 and sys.argv[1] == "version":
        print(version(Path(sys.argv[2]).read_text(), sys.argv[3]))
    else:
        sys.exit(__doc__)
