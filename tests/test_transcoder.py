# experiments/transcoder: the labels-forwarding transcoder builds from its
# own default flags and trains end to end on a pair of row-aligned caches.
# Its defaults once asked for gain-shape coding under labels forwarding and
# for the retired s_hcossim, and it unpacked the schedules' older arity;
# each failed before the first step.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "transcoder"))
from transcode_train import PairedNpySource, build, parse_args, train  # noqa: E402

D_X, D_Y = 6, 5
SMALL = ["--e-dec", "8", "--k", "4", "--h", "2", "--l", "2", "--b", "4",
         "--epochs", "1", "--anneal-steps", "4", "--save-each", "2",
         "--checkpoint-each", "4"]


def caches(tmp_path, n=16):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, D_X)).astype(np.float32)
    y = (x @ rng.normal(size=(D_X, D_Y))).astype(np.float32)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    return str(tmp_path / "x.npy"), str(tmp_path / "y.npy")


def test_default_flags_build_a_labels_model():
    cfg = parse_args(["--cache-x", "x", "--cache-y", "y"] + SMALL)
    hyper, model = build(cfg, D_X, D_Y)
    assert model.forward == "labels" and not model.resid_gain
    assert hyper.s_hcossim == 0.0 and hyper.s_support == cfg.s_support


def test_trains_on_a_cache_pair(tmp_path):
    cx, cy = caches(tmp_path)
    cfg = parse_args(["--cache-x", cx, "--cache-y", cy,
                      "--out", str(tmp_path / "run")] + SMALL)
    model, state = train(cfg, PairedNpySource(cx, cy))
    assert int(state.step) == 4
    Y = model.apply(state.params, np.load(cx)[:4])
    assert Y.shape == (4, D_Y) and np.isfinite(np.asarray(Y)).all()
