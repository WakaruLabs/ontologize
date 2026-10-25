"""Share of a full Adam step each layer's classifier and dictionary take.

Adam steps by m_hat / (sqrt(v_hat) + eps). This reads the second moments
(`nu`) a checkpoint saves with its optimizer state, bias-corrects them with
optax's b2 = 0.999, and prints per layer the median of
sqrt(v_hat) / (sqrt(v_hat) + eps): 1 is a full step, 0 a parameter that eps
has frozen. eps is the run's own, read from its `log.jsonl` (optax's 1e-8
when the run did not set one). Arrays are restored as numpy, so this runs
under JAX_PLATFORMS=cpu beside a GPU job.

An sae.py run directory (one holding `state.npz`) is read the same way, per
parameter (W_dec, W_enc, ...), at the step its state was last saved, under
the eps its `meta.json` records.

    JAX_PLATFORMS=cpu uv run python experiments/adam-eps/throttle.py RUN:STEP [RUN:STEP ...] [SAE_RUN ...]
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


def sae_medians(run: Path) -> Tuple[float, int, int, Dict[str, float]]:
    """eps, the saved step, Adam's count and {parameter: median throttle}
    for an sae.py run. Its `state.npz` holds the leaves of (params,
    optax.adam's state) in tree order: the parameters, the count, then mu
    and nu, each in the parameters' order, which is sorted by name. The
    names come from `params.npz`, which the run writes when it finishes;
    before then the parameters are named by leaf."""
    z = np.load(run / "state.npz")
    n_leaves = sum(k.startswith("leaf_") for k in z.files)
    n = (n_leaves - 1) // 3
    leaf = lambda i: z[f"leaf_{i}"]
    if 3 * n + 1 != n_leaves or any(
            not leaf(i).shape == leaf(n + 1 + i).shape == leaf(2 * n + 1 + i).shape
            for i in range(n)):
        raise ValueError(f"{run / 'state.npz'} is not (params, count, mu, nu)")
    count = int(leaf(n))
    meta = run / "meta.json"
    eps = float(json.loads(meta.read_text()).get("adam_eps", 1e-8)) if meta.exists() else 1e-8
    p = run / "params.npz"
    names = sorted(np.load(p).files) if p.exists() else [f"leaf_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"{p} has {len(names)} arrays; state.npz has {n} parameters")
    return eps, int(z["step"]), count, {
        k: float(np.median(throttle(np.ravel(leaf(2 * n + 1 + i)), count, eps)))
        for i, k in enumerate(names)}


def main() -> None:
    saes = [a for a in sys.argv[1:] if (Path(a) / "state.npz").exists()]
    ontos = [a for a in sys.argv[1:] if a not in saes]
    if ontos:
        print(f"{'run':28s} {'eps':>7s} {'step':>7s}  classifier by layer             dictionary by layer")
    for arg in ontos:
        run, step = arg.rsplit(":", 1)
        eps, med = layer_medians(Path(run), int(step))
        layers = sorted({k[0] for k in med})
        row = lambda kind: " ".join(f"{med.get((l, kind), float('nan')):.3f}" for l in layers)
        print(f"{Path(run).name:28s} {eps:7.0e} {int(step):7d}  {row('classifier'):32s} {row('dict')}")
    if saes:
        print(f"{'sae.py run':45s} {'eps':>7s} {'step':>7s} {'count':>7s}  by parameter")
    for arg in saes:
        eps, step, count, med = sae_medians(Path(arg))
        print(f"{Path(arg).name:45s} {eps:7.0e} {step:7d} {count:7d}  "
              + " ".join(f"{k} {v:.3f}" for k, v in med.items()))


if __name__ == "__main__":
    main()
