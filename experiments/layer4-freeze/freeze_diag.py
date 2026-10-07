"""Checkpoint-sweep frozen-head diagnostic for the layer-4 freeze.

Context (sonar.py, s_Hm calibration note): the resid_nc run froze 22/32
layer-4 heads after step ~211k -- deep in the hard-temperature phase,
after the anneal (50k steps) had long finished. A frozen head emits the
same argmax entry for (nearly) every sample and stops moving between
checkpoints: its softmax has saturated, so gradient through the
classifier has vanished and nothing short of winner dropout or logit
noise can reach the runner-ups.

This script walks every checkpoint retained in a run directory and, for
a FIXED batch of cache rows (the cache tail, i.e. the sae.py eval-split
convention), measures per (layer, head):

  top_share    fraction of samples won by the head's modal entry
               (argmax of the classifier logits; temperature-free)
  n_live       number of dict entries that win at least one sample
  usage_bits   entropy (bits) of the hard-assignment histogram
  churn        fraction of samples whose argmax CHANGED since the
               previous retained checkpoint (NaN at the first one)
  H_mean       mean per-sample soft entropy (bits) of p at --temperature
  KL_m         KL(batch-mean p || uniform) in bits -- the per-head
               version of the s_Hm stat (dictblock.hmean_kl)
  frozen       top_share >= --freeze-share and churn <= --freeze-churn
               (share-only at the first checkpoint)

Freezing is detected on ARGMAX statistics, which are invariant to the
evaluation temperature, so the sweep does not need to reconstruct the
training-time temperature schedule; --temperature only affects the two
soft-entropy columns (default 0.03 = the anneal floor, matching the
autointerp/pareto convention).

Outputs (written into --out, default this experiment dir):
  heads.csv          long format, one row per (checkpoint, layer, head)
  summary.csv        per (checkpoint, layer): frozen count + means
  freeze_report.md   final-state table, per-head onset steps (the first
                     retained checkpoint from which a head stays frozen
                     through the end), layer-4 detail

Run (from the repo root, GPU JAX for checkpoint restore -- same caveat
as refit.py; CPU works but is slower):

  uv run python experiments/layer4-freeze/freeze_diag.py \\
      --ckpt data/out/sonar/multilingual/resid_nc
  uv run python experiments/layer4-freeze/freeze_diag.py \\
      --ckpt data/out/sonar/multilingual/resid_nc_hm \\
      --out experiments/layer4-freeze/diag_resid_nc_hm
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import functools
import json
import sys
import numpy as np
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))  # repo root: autointerp
EPS = 1e-12
LOG2E = 1.4426950408889634


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="data/out/sonar/multilingual/resid_nc",
                   help="Ontologizer run dir (orbax CheckpointManager root); "
                        "known runs: data/out/sonar/multilingual/resid_nc "
                        "(the frozen reference), .../resid_nc_hm (s_Hm run)")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy",
                   help="embedding cache; the tail --rows are used so the "
                        "batch matches the sae.py eval split")
    p.add_argument("--rows", type=int, default=4096,
                   help="fixed eval rows (same rows at every checkpoint, "
                        "so churn is well-defined)")
    p.add_argument("--b", type=int, default=1024, help="forward chunk size")
    p.add_argument("--temperature", type=float, default=0.03,
                   help="softmax temperature for the soft-entropy columns "
                        "(argmax stats are temperature-free)")
    p.add_argument("--min-step", type=int, default=0,
                   help="skip checkpoints below this step")
    p.add_argument("--max-checkpoints", type=int, default=0,
                   help="if >0, subsample the retained steps evenly down "
                        "to this many (first and last always kept)")
    p.add_argument("--freeze-share", type=float, default=0.95,
                   help="top_share threshold for the frozen flag")
    p.add_argument("--freeze-churn", type=float, default=0.02,
                   help="churn threshold for the frozen flag")
    p.add_argument("--out", default=str(EXP_DIR / "diag"),
                   help="output directory for heads.csv / summary.csv / "
                        "freeze_report.md")
    return p.parse_args()


def load_run(ckpt):
    """CheckpointManager + model spec, following autointerp.onto_acts_fn."""
    import orbax.checkpoint as ocp
    from ontologize.ontologizer import Ontologizer
    from ontologize.training.serialize import restore_spec

    manager = ocp.CheckpointManager(
        Path(ckpt).resolve(),
        checkpointers={'state': ocp.PyTreeCheckpointer(),
                       'spec': ocp.PyTreeCheckpointer()})
    steps = sorted(int(s) for s in manager.all_steps())
    if not steps:
        raise SystemExit(f"no checkpoints under {ckpt}")
    model = Ontologizer(**restore_spec(manager, steps[-1]))
    return manager, model, steps


def restore_params(manager, step):
    """Normalize both checkpoint formats (same as autointerp.onto_acts_fn)."""
    state = manager.restore(step, items={'state': None})['state']
    params = state['params'] if 'opt_state' in state else state
    while 'params' in params:
        params = params['params']
    return {'params': params}


def make_acts(model, temperature):
    """Jitted (params, X) -> per-layer classifications (b, l, h, k).

    The probe is `autointerp.onto_probe`: a clean forward (no noise, no
    winner dropout) with the constant coordinate, the gain-shape split,
    the layer gain, the router and fibers applied as the model applies
    them, and the flattened classification forwarded, so labels-mode
    checkpoints work too. Traced once; params vary across checkpoints.
    """
    import jax
    from autointerp import onto_probe

    @jax.jit
    def acts(params, X):
        return model.apply(params, X, temperature, method=onto_probe)

    return acts


def head_stats(A_lhr, sumP, sumH, rows, k):
    """Per-(layer, head) hard/soft usage statistics.

    A_lhr: (l, h, rows) int argmax per sample. sumP: (l, h, k) summed soft
    classifications. sumH: (l, h) summed per-sample entropies (bits)."""
    l, h, _ = A_lhr.shape
    top_entry = np.zeros((l, h), int)
    top_share = np.zeros((l, h))
    n_live = np.zeros((l, h), int)
    usage_bits = np.zeros((l, h))
    for li in range(l):
        for hi in range(h):
            counts = np.bincount(A_lhr[li, hi], minlength=k)
            top_entry[li, hi] = int(counts.argmax())
            top_share[li, hi] = counts.max() / rows
            n_live[li, hi] = int((counts > 0).sum())
            q = counts / rows
            q = q[q > 0]
            usage_bits[li, hi] = float(-(q * np.log2(q)).sum())
    meanP = sumP / rows
    meanP = meanP / np.maximum(meanP.sum(-1, keepdims=True), EPS)
    H_meanP = -(meanP * np.log(meanP + EPS)).sum(-1) * LOG2E
    KL_m = np.log2(k) - H_meanP                       # (l, h) bits
    H_mean = sumH / rows
    return top_entry, top_share, n_live, usage_bits, H_mean, KL_m


def subsample(steps, n):
    if n <= 0 or len(steps) <= n:
        return steps
    idx = np.unique(np.round(np.linspace(0, len(steps) - 1, n)).astype(int))
    return [steps[i] for i in idx]


def sweep(cfg):
    import jax.numpy as jnp

    manager, model, steps = load_run(cfg.ckpt)
    steps = [s for s in steps if s >= cfg.min_step]
    steps = subsample(steps, cfg.max_checkpoints)
    l, h, k = model.l, model.h, model.k
    print(f"{cfg.ckpt}: l={l} h={h} k={k}; {len(steps)} checkpoints "
          f"({steps[0]}..{steps[-1]})")

    mm = np.load(cfg.cache, mmap_mode="r")
    rows = min(cfg.rows, mm.shape[0]) // cfg.b * cfg.b
    X_all = np.asarray(mm[-rows:], dtype=np.float32)  # eval-split tail
    acts = make_acts(model, cfg.temperature)

    records = []          # per (step, layer, head) dicts
    A_prev = None
    frozen_seq = []       # (n_steps, l, h) bool
    for step in steps:
        params = restore_params(manager, step)
        A = np.empty((l, h, rows), np.int16)
        sumP = np.zeros((l, h, k), np.float64)
        sumH = np.zeros((l, h), np.float64)
        for i in range(0, rows, cfg.b):
            P = np.asarray(acts(params, jnp.asarray(X_all[i:i + cfg.b])),
                           dtype=np.float32)          # (b, l, h, k)
            A[:, :, i:i + cfg.b] = P.argmax(-1).transpose(1, 2, 0)
            sumP += P.sum(0)
            Hs = -(P * np.log(P + EPS)).sum(-1) * LOG2E   # (b, l, h) bits
            sumH += Hs.sum(0)

        top_entry, top_share, n_live, usage_bits, H_mean, KL_m = \
            head_stats(A, sumP, sumH, rows, k)
        churn = (np.full((l, h), np.nan) if A_prev is None
                 else (A != A_prev).mean(-1))
        # share-only at the first checkpoint (churn undefined there)
        frozen = (top_share >= cfg.freeze_share) & \
                 (np.isnan(churn) | (churn <= cfg.freeze_churn))
        frozen_seq.append(frozen)
        A_prev = A

        for li in range(l):
            for hi in range(h):
                records.append({
                    "step": step, "layer": li, "head": hi,
                    "top_entry": int(top_entry[li, hi]),
                    "top_share": round(float(top_share[li, hi]), 6),
                    "n_live": int(n_live[li, hi]),
                    "usage_bits": round(float(usage_bits[li, hi]), 5),
                    "churn": ("" if np.isnan(churn[li, hi])
                              else round(float(churn[li, hi]), 6)),
                    "H_mean": round(float(H_mean[li, hi]), 5),
                    "KL_m": round(float(KL_m[li, hi]), 5),
                    "frozen": int(frozen[li, hi]),
                })
        nf = frozen.sum(-1)
        churn_str = ("n/a" if np.all(np.isnan(churn))
                     else f"{np.nanmean(churn):.4f}")
        print(f"step {step:>8}: frozen per layer "
              f"{' '.join(f'{int(x):2d}' for x in nf)} (of {h}); "
              f"mean churn {churn_str}")
    return model, steps, records, np.stack(frozen_seq)


def onset_steps(steps, frozen_seq):
    """First retained step from which a head stays frozen through the last
    checkpoint; -1 if not frozen at the end."""
    n, l, h = frozen_seq.shape
    onset = np.full((l, h), -1, int)
    # suffix-AND: frozen at every checkpoint from i to the end
    suffix = np.flip(np.cumprod(np.flip(frozen_seq.astype(bool), 0), 0), 0)
    for li in range(l):
        for hi in range(h):
            if frozen_seq[-1, li, hi]:
                first = int(np.argmax(suffix[:, li, hi]))
                onset[li, hi] = steps[first]
    return onset


def write_outputs(cfg, model, steps, records, frozen_seq):
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    l, h, k = model.l, model.h, model.k

    fields = list(records[0].keys())
    with open(out / "heads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    by_key = {(r["step"], r["layer"]): [] for r in records}
    for r in records:
        by_key[(r["step"], r["layer"])].append(r)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "layer", "n_frozen", "mean_top_share",
                    "mean_usage_bits", "mean_churn", "mean_KL_m"])
        for (step, layer), rs in sorted(by_key.items()):
            churns = [r["churn"] for r in rs if r["churn"] != ""]
            w.writerow([
                step, layer, sum(r["frozen"] for r in rs),
                round(float(np.mean([r["top_share"] for r in rs])), 6),
                round(float(np.mean([r["usage_bits"] for r in rs])), 5),
                round(float(np.mean(churns)), 6) if churns else "",
                round(float(np.mean([r["KL_m"] for r in rs])), 5)])

    onset = onset_steps(steps, frozen_seq)
    final = frozen_seq[-1]
    lines = [
        "# Frozen-head report", "",
        f"run: `{cfg.ckpt}`  |  checkpoints: {len(steps)} "
        f"({steps[0]}..{steps[-1]})  |  eval rows: {cfg.rows} (cache tail)",
        f"frozen := top_share >= {cfg.freeze_share} and churn <= "
        f"{cfg.freeze_churn} (argmax over classifier logits, "
        f"temperature-free)", "",
        "## Final checkpoint", "",
        "| layer | frozen heads / total |", "|---|---|"]
    for li in range(l):
        lines.append(f"| {li} | {int(final[li].sum())} / {h} |")
    lines += ["", "## Onset (first retained step from which the head stays "
              "frozen through the end)", "",
              "| layer | head | onset step |", "|---|---|---|"]
    for li in range(l):
        for hi in range(h):
            if onset[li, hi] >= 0:
                lines.append(f"| {li} | {hi} | {onset[li, hi]} |")
    l4 = int(final[-1].sum()) if l > 0 else 0
    lines += ["", "## Layer-4 note", "",
              f"Layer {l - 1} (the top layer) has {l4}/{h} frozen heads at "
              f"the final checkpoint. The resid_nc reference measured 22/32 "
              f"frozen after step ~211k (sonar.py, s_Hm calibration note); "
              f"compare the onset table above against the temperature / "
              f"p_drop / sd_K schedule endpoints (anneal_steps in the run's "
              f"log.jsonl) to attribute the freeze to the hard phase.", ""]
    (out / "freeze_report.md").write_text("\n".join(lines))
    print(f"-> {out}/heads.csv, summary.csv, freeze_report.md")


def main():
    cfg = parse_args()
    model, steps, records, frozen_seq = sweep(cfg)
    write_outputs(cfg, model, steps, records, frozen_seq)


if __name__ == "__main__":
    main()
