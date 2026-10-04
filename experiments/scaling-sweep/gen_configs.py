"""Config-grid generator for the (h, k, l) scaling sweep.

The paper's polysemanticity claim ("polysemanticity decreases with
parameter count and we have more directions to increase parameters ...
increasing width [h] ... head size [k] ... more layers [l]") is testable
with the machinery already in the repo: train the live sonar.py
configuration at a grid of (h, k, l) around the live values (32, 32, 5)
and measure within-head semantic coherence (headcoh.py z-scores; a MORE
coherent head is a LESS polysemantic unit) and usage balance (KL_m) as a
function of parameter count.

This script writes:
  configs/<name>.json   one JSON per grid point: the full sonar.py
                        parameter set with h/k/l (and out dir) overridden.
                        train_run.py consumes these; every field mirrors a
                        sonar.py module-level constant by name.
  queue.sh              sequential queue: train each config (resumable --
                        TrainingEnv resumes from the latest checkpoint in
                        the out dir), then run headcoh.py on the result.

Default grid is the one-axis-at-a-time star around the center (7 runs):
h in {16,32,64} x k=32 x l=5, k in {16,64}, l in {3,7}. --full-cross
generates the full 27-point cross instead (weeks of GPU; don't).

  uv run python experiments/scaling-sweep/gen_configs.py
  uv run python experiments/scaling-sweep/gen_configs.py --epochs 4 --seed 43
  bash experiments/scaling-sweep/queue.sh
"""
import argparse
import itertools
import json
import stat
from pathlib import Path

EXP = Path(__file__).resolve().parent

# Mirror of sonar.py's live resid_nc_hm configuration, field for field.
# Two deliberate deviations, both surfaced as CLI flags:
#   epochs 4 (not 24): 4 epochs = ~62.5k steps at b=256 on the 4M cache,
#     past the 50k anneal horizon; the live 24-epoch budget times 7 runs
#     is not a defensible sweep cost, and overtraining exacerbates mode
#     collapse anyway (sonar.py's own comment).
#   out under scaling/<name>: fresh dirs, never the live run dirs.
BASE = {
    "d": 1024, "e_enc": 2048, "e_dec": 2048,
    "k": 32, "h": 32, "l": 5,
    "n": 2, "gate": "none",
    "fwd_mode": "resid", "deepsup": True, "deepsup_sg": False,
    "resid_norm": True, "resid_const": True,
    "scaled": False, "encoded": False,
    "b": 256, "epochs": 4,
    "lr": 5e-5, "wd": 0.0,
    "temperature": 1.0, "temperature_end": 0.03, "anneal_steps": 50000,
    "noise_in": "batchnorm", "sd_in": 0.0,
    "noise_K": "normal", "noise_F": "featvar",
    "sd_K": 0.02, "sd_F": 0.1,
    "p_drop": 0.1, "p_drop_start": 0.0,
    "sd_K_end": 0.2 * 0.03,
    "ghost": False,
    "s_g": 1e-4, "s_L1K": 0.0, "s_L1F": 1e-9, "s_H": 0.0,
    "s_bcossim": 1e-5, "s_hcossim": 1e-6, "s_Hm": 1e-6,
    "mse_weights": "data/out/sonar/mse_weights.npy",
    "cache": "data/sonar_embeddings/mc4_4M.npy",
    "shuffle": True,
    "save_each": 100, "checkpoint_each": 10000,
    "dtype_str": "float32", "dtype_p_str": "float32",
    "encoder": "cointegrated/SONAR_200_text_encoder",
    "seed": 42,
}
CENTER = (32, 32, 5)  # the live (h, k, l)


def grid_points(hs, ks, ls, full_cross):
    """Grid as (h, k, l) tuples: full cross, or the star of one-axis
    variations around CENTER (center included once)."""
    if full_cross:
        return sorted(set(itertools.product(hs, ks, ls)))
    pts = {CENTER}
    for h in hs:
        pts.add((h, CENTER[1], CENTER[2]))
    for k in ks:
        pts.add((CENTER[0], k, CENTER[2]))
    for l in ls:
        pts.add((CENTER[0], CENTER[1], l))
    return sorted(pts)


def run_name(h, k, l, seed):
    name = f"h{h}_k{k}_l{l}"
    if seed != 42:
        name += f"_s{seed}"  # replica convention matches sae.py
    return name


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--k", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--l", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--full-cross", action="store_true",
                   help="full h x k x l cross instead of the axis star")
    p.add_argument("--epochs", type=int, default=BASE["epochs"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-base", default="data/out/sonar/multilingual/scaling",
                   help="run dirs land at <out-base>/<name>")
    p.add_argument("--cache", default=BASE["cache"])
    p.add_argument("--mse-weights", default=BASE["mse_weights"])
    p.add_argument("--headcoh-temperature", type=float, default=0.03)
    args = p.parse_args()

    cfg_dir = EXP / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    pts = grid_points(args.h, args.k, args.l, args.full_cross)
    runs = []
    for h, k, l in pts:
        name = run_name(h, k, l, args.seed)
        cfg = dict(BASE, h=h, k=k, l=l, epochs=args.epochs, seed=args.seed,
                   cache=args.cache, mse_weights=args.mse_weights,
                   out=f"{args.out_base}/{name}", name=name)
        (cfg_dir / f"{name}.json").write_text(json.dumps(cfg, indent=2))
        runs.append((name, cfg["out"]))

    q = EXP / "queue.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# generated by gen_configs.py: sequential train + headcoh queue.",
        "# Re-running is safe: training resumes from the latest checkpoint",
        "# in each run dir (TrainingEnv resume_from = manager.latest_step).",
        "set -uo pipefail",
        'cd "$(dirname "$0")/../.."',
        "",
        "run() {",
        '  name="$1"; out="$2"',
        '  echo "=== $name: train ==="',
        "  uv run python experiments/scaling-sweep/train_run.py \\",
        '      "experiments/scaling-sweep/configs/$name.json" || exit 1',
        '  echo "=== $name: headcoh ==="',
        "  uv run python headcoh.py --model onto --ckpt \"$out\" \\",
        f"      --temperature {args.headcoh_temperature} || exit 1",
        "}",
        "",
    ]
    lines += [f'run {name} "{out}"' for name, out in runs]
    lines.append("")
    lines.append('echo "queue done; now: uv run python '
                 'experiments/scaling-sweep/analyze.py"')
    q.write_text("\n".join(lines) + "\n")
    q.chmod(q.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"{len(runs)} configs -> {cfg_dir}")
    for name, out in runs:
        print(f"  {name:<16} -> {out}")
    print(f"queue -> {q}")


if __name__ == "__main__":
    main()
