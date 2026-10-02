"""Training metric visualization and loss log parsing utilities.

This module provides functions to parse training metric logs (`loss.csv`) generated
by `ontologize.training.ontostate.OntoState.writestats` and generate scatter plots
of individual loss components (MSE, L1, entropy, cossim, KL divergence) across training steps.
"""
import matplotlib.pyplot as plt
import pandas as pd
import json

from math import log
from typing import List
from pathlib import Path

def read_loss(path):
    """Read training loss history from a checkpoint directory.

    Parses `loss.csv` located within `path` into a pandas DataFrame with columns:
    `loss`, `MSE`, `MSE_ghost`, `L1_K`, `L1_F`, `entropy`, `cossim_b`, `cossim_h`, `KL_m`, `aux`
    (runs before 2026-09-30 have 9 columns, before `KL_m` 8).

    Args:
        path: Path to the directory containing `loss.csv` (str or Path).

    Returns:
        pandas DataFrame containing recorded training metrics per logged batch.
    """
    file = Path(path) / "loss.csv"
    # row layout written by OntoState.stats via Hyperparams.loss; runs from
    # before the KL_m stat have 8 columns (KL_m reads as NaN and plots flat)
    colnames = ["loss", "MSE", "MSE_ghost", "L1_K", "L1_F", "entropy",
                "cossim_b", "cossim_h", "KL_m", "aux"]
    loss = pd.read_csv(file, names=colnames)
    return loss

def plot_stat(y, path, stat: str, fmt='pdf', base=None, *args, **kwargs):
    """Generate and save a scatter plot for a single training metric against batch index.

    Args:
        y: Sequence or Series of metric values over training batches.
        path: Destination directory (Path) where plot image will be saved.
        stat: Name of the metric (used as filename and plot label).
        fmt: File format extension (default 'pdf').
        base: Optional logarithmic base (e.g. 10 or 2). If provided, values > 0 are
            transformed via `log(val, base)`.
        *args: Additional positional arguments passed to `plt.scatter`.
        **kwargs: Additional keyword arguments passed to `plt.scatter`.
    """
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
    """Plot and save all training metrics from `loss.csv` in a directory.

    Reads `loss.csv` from `path` and calls `plot_stat` for each metric column.

    Args:
        path: Directory containing `loss.csv` (str or Path).
        fmt: Output file format for plots (default 'pdf').
        base: Optional logarithmic base for metric scaling.
        *args: Additional positional arguments passed to `plot_stat`.
        **kwargs: Additional keyword arguments passed to `plot_stat`.
    """
    loss = read_loss(path)
    for s in loss.columns:
        plot_stat(loss[s], path, s, fmt, base, *args, **kwargs)
