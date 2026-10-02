import matplotlib.pyplot as plt
import pandas as pd
import json

from math import log
from typing import List
from pathlib import Path

# The authoritative `loss.csv` column layout. Import this rather than
# copying it: divergent copies have silently mislabelled columns before.
#
# One row per step, written by `OntoState.stats` from `Hyperparams.loss`.
# Assembled in three pieces, which is why the order looks arbitrary:
#
#   0-2    `Hyperparams.loss` itself -- total loss, the MSE term, and the
#          ghost-path MSE (0 when the ghost path is off)
#   3-12   the per-layer `DictEnc.withStats` row SUMMED OVER LAYERS. That
#          row is 12 wide (`DictEnc.withPWAK` builds it: `L1_K` in front
#          of `DictBlock.withStats`'s first seven, the two pwak stats
#          next, then its eighth, then `L1_S`), so only its first ten land
#          here; the last two are appended at 16-17.
#   13-15  the two setpoint-controlled multipliers AS APPLIED this step,
#          and the max of `cossim_k` over heads *and* layers
#   16+    later additions, appended so a shorter row from an older run
#          still aligns column for column (`read_loss` reads the missing
#          tail as NaN)
#
# Two columns are easy to confuse, and the names are the only thing
# separating them:
#   `cossim_k`       mean over a layer's heads, summed over layers
#   `cossim_k_hmax`  per-head MAX within a layer, summed over layers
#   `cossim_k_max`   max over heads AND layers (a single scalar, not a sum)
# The last exists because a summed `cossim_k` cannot see one layer's head
# collapse: the others improve faster than it degrades, so the sum falls
# while the max rises. See `Hyperparams.loss`.
#
# NOT universal: `experiments/hsic-bottleneck/hsic_hyperparams.py`
# overrides `loss` to write its own 9-wide row (`[L, L2, pen_raw]` plus a
# 6-wide layer-summed tail), so those runs' loss.csv does not match these
# names at all.
COLUMNS = ["loss", "MSE", "MSE_ghost", "L1_K", "L1_F", "entropy",
           "cossim_b", "cossim_h", "cossim_k", "cossim_k_hmax",
           "KL_m", "KL_pwak", "L2_pwak", "s_L1F", "s_kcossim",
           "cossim_k_max", "cossim_flat", "L1_S"]


def read_loss(path):
    file = Path(path) / "loss.csv"
    loss = pd.read_csv(file, names=COLUMNS)
    return loss

def plot_stat(y, path, stat: str, fmt='pdf', base=None, *args, **kwargs):
    x = range(len(y))
    if base:
        # Apply log transformation element-wise to the list
        y = [log(val, base) if val > 0 else 0 for val in y]
        ylab = f"log{base}({stat})"
    else:
        ylab = stat

    plt.scatter(x, y, alpha=0.25, *args, **kwargs)
    plt.xlabel('batch')
    plt.ylabel(ylab)

    plt.savefig(path / (stat + '.' + fmt))
    plt.close()  # Close the figure to free memory

def plot_loss(path, fmt="pdf", base=None, *args, **kwargs):
    loss = read_loss(path)
    for s in loss.columns:
        plot_stat(loss[s], path, s, fmt, base, *args, **kwargs)
