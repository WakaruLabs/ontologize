# experiments/ste-arm/train_ste.py: the recipe flags reach what setup
# builds. A flag left at None takes sonar.py's value, so each override is
# checked against the default as well as against its own value.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
import train_ste  # noqa: E402


def built(tmp_path, *argv):
    np.save(tmp_path / "c.npy", np.zeros((8, 4), np.float32))
    cfg = train_ste.parser().parse_args(
        ["--cache", str(tmp_path / "c.npy"), "--out", str(tmp_path / "run"), *argv])
    hyper, model, meta, out, src = train_ste.setup(train_ste.finish(cfg), src=list(range(4)))
    return hyper, meta


def test_noise_and_l1f_flags_reach_the_hyperparams(tmp_path):
    import sonar as base
    hyper, meta = built(tmp_path)
    assert (hyper.sd_F, hyper.s_L1F) == (base.sd_F, base.s_L1F)
    assert (meta.max_to_keep, meta.checkpoint_each) == (10, base.checkpoint_each)
    hyper, meta = built(tmp_path, "--sd-f", "0", "--s-l1f", "0", "--sd-k", "0",
                        "--p-drop", "0", "--max-to-keep", "1",
                        "--checkpoint-each", "50000")
    assert (hyper.sd_F, hyper.s_L1F, hyper.sd_K, hyper.p_drop) == (0, 0, 0, 0)
    assert (meta.max_to_keep, meta.checkpoint_each) == (1, 50000)
