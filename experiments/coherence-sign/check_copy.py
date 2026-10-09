"""Check that the code in `ontologize/` here is commit 4d32e33's.

Each copied module is parsed with its docstrings removed, and the SHA-256 of
the resulting syntax tree is compared with the one recorded below for the
same file at that commit. Comments are not part of a syntax tree, so the
check passes exactly when the copy differs from the commit only in comments
and docstrings, which is the only way the copy may be edited. It also checks
that the copy holds exactly these files plus its own `__init__.py`.

The fingerprints were computed with Python 3.13's `ast.dump`; another
version of `ast` can change the dump and fail the check without any change
to the code.

  uv run python experiments/coherence-sign/check_copy.py
"""

import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# syntax-tree fingerprint of each file at the commit, docstrings removed
FINGERPRINTS = {
    "ontologize/data/__init__.py":
        "ad2e13b69c4fc1fda46413f740f426fc1793cfc6d6da8d226ba619f1aef48be7",
    "ontologize/data/loaders.py":
        "a7101e185bdf4cec5a25a380accc7c33e28ec3ae40fc9c4ce4189e76ac49a457",
    "ontologize/data/pretrained.py":
        "b205b4ad17d19d8ba1cacbca47722035410dd72bc7d59b5958aa9e9345ece75e",
    "ontologize/fns/classify.py":
        "adf62f9e9c5e06ee4e34c0846d4c16d9963e88d21807c8098298d3e218e764b2",
    "ontologize/fns/keys.py":
        "27b85e5b2112e46500c28f7c92ba9a631953e4d3608f80372eed8bd767f32234",
    "ontologize/fns/loss.py":
        "de68012840d7f324d32de04632841558f98728356e456edf010fea5bb02b943f",
    "ontologize/fns/pwak.py":
        "a980fe08c8a0859833cd562a226f879895fd8da421902409dbe4f3c722b90e44",
    "ontologize/layers/__init__.py":
        "ad2e13b69c4fc1fda46413f740f426fc1793cfc6d6da8d226ba619f1aef48be7",
    "ontologize/layers/dictblock.py":
        "53dbf870bcad0d5810178da60eff8f14e076056383ca03f78fbec55672c07802",
    "ontologize/layers/dictenc.py":
        "612ebe52c7cb3b4d70842c3ad334fbb35d2b593ef0a560972c9bdda35e0256e3",
    "ontologize/layers/linear.py":
        "fbbb05cd8be21551f44c8c4d6ff6b0264f83c68b96fd2f5151d7a8904e797487",
    "ontologize/layers/nlinear.py":
        "b0b394c548a0f04006f759e4e3cbba642c4b819c5356d568a74cf906baa6044a",
    "ontologize/layers/sparse.py":
        "5f71b88c0bb84e41a3afa82270e3a3d0b4c97708f03b8228b6c89c5a166223d7",
    "ontologize/ontologizer.py":
        "52073b547b056b71f7de45a7ac649c83c600fb0f09c8befbbe6a18b8e3410ba7",
    "ontologize/training/config.py":
        "5ba21693eb87bccc79e267c23d84f3628fc6197a2c8773a47a6ce36ec44ea372",
    "ontologize/training/ontostate.py":
        "67398e6fafd007901026b3a94a850514b56a978564d27c0024bda37ae2a1b0de",
    "ontologize/visualize/loss.py":
        "fec82aaa0f27fbc2a339792ebe66d99c211bb1093c6fe9d274b0126713fa4fe8",
}


def code(source: str) -> str:
    """The syntax tree of `source` with every docstring removed, as text."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree)


def fingerprint(source: str) -> str:
    return hashlib.sha256(code(source).encode()).hexdigest()


def main() -> int:
    found = sorted(str(p.relative_to(HERE))
                   for p in (HERE / "ontologize").rglob("*.py"))
    expected = sorted([*FINGERPRINTS, "ontologize/__init__.py"])
    bad = []
    if found != expected:
        bad.append(f"file set differs: extra {sorted(set(found) - set(expected))}, "
                   f"missing {sorted(set(expected) - set(found))}")
    for f, want in FINGERPRINTS.items():
        path = HERE / f
        if path.exists() and fingerprint(path.read_text()) != want:
            bad.append(f"{f}: code differs from 4d32e33")
    for line in bad:
        print(line)
    if bad:
        print(f"FAIL: {len(bad)} problem(s)")
        return 1
    print(f"ok: code identical to 4d32e33 in {len(FINGERPRINTS)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
