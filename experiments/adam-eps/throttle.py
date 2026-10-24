"""Share of a full Adam step each layer's classifier and dictionary take.

Adam steps by m_hat / (sqrt(v_hat) + eps). This reads the second moments
(`nu`) a checkpoint saves with its optimizer state, bias-corrects them with
optax's b2 = 0.999, and prints per layer the median of
sqrt(v_hat) / (sqrt(v_hat) + eps): 1 is a full step, 0 a parameter that eps
has frozen. eps is the run's own, read from its `log.jsonl` (optax's 1e-8
when the run did not set one). Arrays are restored as numpy, so this runs
under JAX_PLATFORMS=cpu beside a GPU job.

    JAX_PLATFORMS=cpu uv run python experiments/adam-eps/throttle.py RUN:STEP [RUN:STEP ...]
"""
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import jax
import numpy as np
import orbax.checkpoint as ocp
from jaxtyping import Float

B2 = 0.999


def throttle(nu: Float[np.ndarray, "n"], count: int, eps: float,
             b2: float = B2) -> Float[np.ndarray, "n"]:
    """sqrt(v_hat) / (sqrt(v_hat) + eps) per entry, v_hat = nu / (1 - b2^count)."""
    s = np.sqrt(np.asarray(nu, np.float64) / (1.0 - b2 ** count))
    return s / (s + eps)


def run_eps(run: Path) -> float:
    """The run's Adam eps, from the hyperparameters `log.jsonl` records."""
    log = run / "log.jsonl"
    if log.exists():
        with open(log) as f:
            hyper = json.loads(f.readline()).get("hyper", {})
        return float(hyper.get("adam_eps", 1e-8))
    return 1e-8


def _plain(tree):
    if hasattr(tree, "keys"):
        return {k: _plain(tree[k]) for k in tree.keys()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_plain(v) for v in tree)
    return tree


def _walk(tree, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], object]]:
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            yield from _walk(v, path + (str(i),))
    else:
        yield path, tree


def layer_medians(run: Path, step: int) -> Tuple[float, Dict[Tuple[int, str], float]]:
    """eps and {(layer, "classifier" | "dict"): median throttle} at `step`."""
    manager = ocp.CheckpointManager(run.resolve(), checkpointers={"state": ocp.PyTreeCheckpointer()})
    args = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
                                  _plain(manager.item_metadata(step)["state"]))
    state = manager.restore(step, items={"state": None},
                            restore_kwargs={"state": {"restore_args": args}})["state"]
    opt = state["opt_state"]
    count = max(int(np.asarray(v)) for p, v in _walk(opt) if p[-1] == "count")
    eps = run_eps(run)
    parts: Dict[Tuple[int, str], List[np.ndarray]] = {}
    for p, v in _walk(opt):
        if "nu" not in p:
            continue
        names = [x for x in p[p.index("nu") + 1:] if x != "params"]
        if len(names) < 2 or not names[0].startswith("dictencs_"):
            continue
        key = (int(names[0].split("_")[1]), names[1])
        parts.setdefault(key, []).append(throttle(np.ravel(v), count, eps))
    return eps, {k: float(np.median(np.concatenate(v))) for k, v in parts.items()}


def main() -> None:
    print(f"{'run':28s} {'eps':>7s} {'step':>7s}  classifier by layer             dictionary by layer")
    for arg in sys.argv[1:]:
        run, step = arg.rsplit(":", 1)
        eps, med = layer_medians(Path(run), int(step))
        layers = sorted({k[0] for k in med})
        row = lambda kind: " ".join(f"{med.get((l, kind), float('nan')):.3f}" for l in layers)
        print(f"{Path(run).name:28s} {eps:7.0e} {int(step):7d}  {row('classifier'):32s} {row('dict')}")


if __name__ == "__main__":
    main()
