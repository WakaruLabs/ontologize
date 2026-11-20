"""Share of a full Adam step each layer's classifier and dictionary take,
and the step each takes relative to its own size.

Adam steps by m_hat / (sqrt(v_hat) + eps). This reads the second moments
(`nu`) a checkpoint saves with its optimizer state, bias-corrects them with
optax's b2 = 0.999, and prints per layer the median of
sqrt(v_hat) / (sqrt(v_hat) + eps): 1 is a full step, 0 a parameter that eps
has frozen. eps is the run's own, read from its `log.jsonl` (optax's 1e-8
when the run did not set one). Arrays are restored as numpy, so this runs
under JAX_PLATFORMS=cpu beside a GPU job.

That share is one of two factors in how fast a block moves. The other is
the step's size against the block's norm, the effective learning rate of a
normalized block. Their product is the realized relative step
lr * |m_hat / (sqrt(v_hat) + eps)| / |theta|, over the block's parameters
together, from the first moments (`mu`, bias-corrected with b1 = 0.9) and
the parameters the checkpoint holds beside them, at the run's own learning
rate. It is the fraction by which one Adam step changes the block, and puts
runs with different eps or parameter norms on one scale. Weight decay's
own step (AdamW) is not included.

An sae.py run directory (one holding `state.npz`) is read the same way, per
parameter (W_dec, W_enc, ...), at the step its state was last saved, under
the eps and learning rate its `meta.json` records. A run whose meta.json
has no learning rate (sae.py wrote none before it recorded `lr`) gets its
relative step per unit learning rate, marked "/lr".

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

B1 = 0.9
B2 = 0.999


def throttle(nu: Float[np.ndarray, "n"], count: int, eps: float,
             b2: float = B2) -> Float[np.ndarray, "n"]:
    """sqrt(v_hat) / (sqrt(v_hat) + eps) per entry, v_hat = nu / (1 - b2^count)."""
    s = np.sqrt(np.asarray(nu, np.float64) / (1.0 - b2 ** count))
    return s / (s + eps)


def relative_step(theta: Float[np.ndarray, "n"], mu: Float[np.ndarray, "n"],
                  nu: Float[np.ndarray, "n"], count: int, eps: float, lr: float,
                  b1: float = B1, b2: float = B2) -> float:
    """lr * |m_hat / (sqrt(v_hat) + eps)| / |theta| over one block: the
    fraction by which Adam's next step changes it."""
    m = np.asarray(mu, np.float64) / (1.0 - b1 ** count)
    s = np.sqrt(np.asarray(nu, np.float64) / (1.0 - b2 ** count))
    norm = np.linalg.norm(np.asarray(theta, np.float64))
    if norm == 0.0:  # a block still at zero has no size to step against
        return float("nan")
    return float(lr * np.linalg.norm(m / (s + eps)) / norm)


def run_hyper(run: Path) -> dict:
    """The hyperparameters a run's `log.jsonl` records ({} without one)."""
    log = run / "log.jsonl"
    if log.exists():
        with open(log) as f:
            return json.loads(f.readline()).get("hyper", {})
    return {}


def run_eps(run: Path) -> float:
    """The run's Adam eps, from the hyperparameters `log.jsonl` records."""
    return float(run_hyper(run).get("adam_eps", 1e-8))


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


Block = Tuple[int, str]


def layer_medians(run: Path, step: int
                  ) -> Tuple[float, Dict[Block, float], Dict[Block, float]]:
    """eps, {(layer, "classifier" | "dict"): median throttle} and
    {(layer, kind): realized relative step} at `step`. The relative step is
    at the run's learning rate, or per unit learning rate when its
    `log.jsonl` records none."""
    manager = ocp.CheckpointManager(run.resolve(), checkpointers={"state": ocp.PyTreeCheckpointer()})
    args = jax.tree_util.tree_map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
                                  _plain(manager.item_metadata(step)["state"]))
    state = manager.restore(step, items={"state": None},
                            restore_kwargs={"state": {"restore_args": args}})["state"]
    opt = state["opt_state"]
    count = max(int(np.asarray(v)) for p, v in _walk(opt) if p[-1] == "count")
    eps = run_eps(run)
    lr = float(run_hyper(run).get("lr", 1.0))
    # a parameter's moments sit under the same path below "mu" and "nu"
    # that the parameter has under the state's "params"
    moments: Dict[str, Dict[Tuple[str, ...], np.ndarray]] = {"mu": {}, "nu": {}}
    for p, v in _walk(opt):
        for k in moments:
            if k in p:
                moments[k][p[p.index(k) + 1:]] = np.ravel(v)
    theta = {p: np.ravel(v) for p, v in _walk(state["params"])}
    blocks: Dict[Block, List[Tuple[str, ...]]] = {}
    for p in moments["nu"]:
        names = [x for x in p if x != "params"]
        if len(names) < 2 or not names[0].startswith("dictencs_"):
            continue
        blocks.setdefault((int(names[0].split("_")[1]), names[1]), []).append(p)
    cat = lambda d, ps: np.concatenate([d[p] for p in ps])
    shares = {k: float(np.median(throttle(cat(moments["nu"], ps), count, eps)))
              for k, ps in blocks.items()}
    rel = {k: relative_step(cat(theta, ps), cat(moments["mu"], ps),
                            cat(moments["nu"], ps), count, eps, lr)
           for k, ps in blocks.items()}
    return eps, shares, rel


def sae_medians(run: Path) -> Tuple[float, int, int, Dict[str, float],
                                    Dict[str, float], bool]:
    """eps, the saved step, Adam's count, {parameter: median throttle},
    {parameter: realized relative step} and whether that step is at the
    run's learning rate (False: per unit learning rate, for a meta.json
    with no `lr`) for an sae.py run. Its `state.npz` holds the leaves of
    (params, optax.adam's state) in tree order: the parameters, the count,
    then mu and nu, each in the parameters' order, which is sorted by name.
    The names come from `params.npz`, which the run writes when it
    finishes; before then the parameters are named by leaf."""
    z = np.load(run / "state.npz")
    n_leaves = sum(k.startswith("leaf_") for k in z.files)
    n = (n_leaves - 1) // 3
    leaf = lambda i: z[f"leaf_{i}"]
    if 3 * n + 1 != n_leaves or any(
            not leaf(i).shape == leaf(n + 1 + i).shape == leaf(2 * n + 1 + i).shape
            for i in range(n)):
        raise ValueError(f"{run / 'state.npz'} is not (params, count, mu, nu)")
    count = int(leaf(n))
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    eps = float(meta.get("adam_eps", 1e-8))
    lr = meta.get("lr")
    p = run / "params.npz"
    names = sorted(np.load(p).files) if p.exists() else [f"leaf_{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"{p} has {len(names)} arrays; state.npz has {n} parameters")
    shares = {k: float(np.median(throttle(np.ravel(leaf(2 * n + 1 + i)), count, eps)))
              for i, k in enumerate(names)}
    rel = {k: relative_step(leaf(i).ravel(), leaf(n + 1 + i).ravel(),
                            leaf(2 * n + 1 + i).ravel(), count, eps,
                            1.0 if lr is None else float(lr))
           for i, k in enumerate(names)}
    return eps, int(z["step"]), count, shares, rel, lr is not None


def main() -> None:
    saes = [a for a in sys.argv[1:] if (Path(a) / "state.npz").exists()]
    ontos = [a for a in sys.argv[1:] if a not in saes]
    if ontos:
        print(f"{'run':28s} {'eps':>7s} {'step':>7s}  {'':5s} classifier by layer"
              f"{'':21s}dictionary by layer")
    for arg in ontos:
        run, step = arg.rsplit(":", 1)
        eps, med, rel = layer_medians(Path(run), int(step))
        layers = sorted({k[0] for k in med})
        share = lambda kind: " ".join(f"{med.get((l, kind), float('nan')):.3f}" for l in layers)
        step_ = lambda kind: " ".join(f"{rel.get((l, kind), float('nan')):.1e}" for l in layers)
        print(f"{Path(run).name:28s} {eps:7.0e} {int(step):7d}  share {share('classifier'):39s} "
              f"{share('dict')}")
        print(f"{'':44s} step  {step_('classifier'):39s} {step_('dict')}")
    if saes:
        print(f"{'sae.py run':45s} {'eps':>7s} {'step':>7s} {'count':>7s}  by parameter: "
              "share, relative step")
    for arg in saes:
        eps, step, count, med, rel, at_lr = sae_medians(Path(arg))
        unit = "" if at_lr else "/lr"
        print(f"{Path(arg).name:45s} {eps:7.0e} {step:7d} {count:7d}  "
              + " ".join(f"{k} {v:.3f}, {rel[k]:.1e}{unit}" for k, v in med.items()))


if __name__ == "__main__":
    main()
