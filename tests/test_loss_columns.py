"""`loss.csv`'s column layout, pinned against the code that writes it.

The layout is assembled in three places (`Hyperparams.loss`,
`DictEnc.withPWAK`, `DictBlock.withStats`) and consumed by readers that
index it by position, so a drifting copy of the names mislabels data
without failing. `experiments/devinterp-tracking/track.py` carried a
9-wide copy that put `KL_m` at index 8 -- `cossim_k` in the real layout --
and recorded that instead for every tracked run.
"""
import numpy as np

from ontologize.training.config import Hyperparams
from ontologize.visualize.loss import COLUMNS


def test_width_matches_n_stats():
    """`n_stats` sizes `OntoState`'s buffer; a mismatch truncates or pads."""
    assert len(COLUMNS) == Hyperparams(8, 8, 4).n_stats


def test_names_are_unique():
    assert len(set(COLUMNS)) == len(COLUMNS)


def test_documented_positions():
    """These indices are what positional readers rely on."""
    assert COLUMNS[:3] == ["loss", "MSE", "MSE_ghost"]
    # the three reductions of cossim_k are distinct columns, easily confused
    assert COLUMNS.index("cossim_k") == 8
    assert COLUMNS.index("cossim_k_hmax") == 9
    assert COLUMNS.index("cossim_k_max") == 15
    assert COLUMNS.index("KL_m") == 10
    # the controlled multipliers sit between the summed row and the tail
    assert COLUMNS[13:15] == ["s_L1F", "s_kcossim"]
    # appended later, so they trail rather than sit with the per-layer row
    assert COLUMNS[-2:] == ["cossim_flat", "L1_S"]


def test_s_loss_pairs_against_ghost_plus_layer_row():
    """`Hyperparams.loss` prepends `MSE_ghost` to the 12-wide per-layer row
    and dots that with `s_loss`, so the weight vector is 13 wide and its
    indices are NOT loss.csv's. `loss` overrides s[2] with the controlled
    `s_L1F` and s[6] with `s_kcossim`, which fixes those two positions."""
    s, isloss = Hyperparams(8, 8, 4).s_loss()
    assert len(s) == 13, "one weight per [MSE_ghost] + 12 layer stats"
    row = ["MSE_ghost"] + COLUMNS[3:13] + ["cossim_flat", "L1_S"]
    assert len(row) == len(s)
    assert row[2] == "L1_F", "s[2] is overridden by the s_L1F controller"
    assert row[6] == "cossim_k", "s[6] is overridden by the s_kcossim one"


def test_tracking_script_uses_the_shared_list():
    """It used to keep its own copy; the bug was invisible because a short
    list silently truncates instead of raising."""
    import importlib.util
    from pathlib import Path

    p = (Path(__file__).resolve().parents[1] / "experiments"
         / "devinterp-tracking" / "track.py")
    spec = importlib.util.spec_from_file_location("track_mod", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.LOSS_COLS is COLUMNS
    assert mod.LOSS_COLS.index("KL_m") == 10


def test_reader_tolerates_a_short_row():
    """Append-only means an older, narrower row still aligns, with the
    missing tail read as NaN."""
    import pandas as pd
    short = pd.read_csv(__import__("io").StringIO("0.1,0.2,0.0\n"),
                        names=COLUMNS)
    assert short["loss"][0] == 0.1 and short["MSE"][0] == 0.2
    assert np.isnan(short["L1_S"][0])
