import matplotlib.pyplot as plt
import pandas as pd
import json

from math import log
from typing import List
from pathlib import Path

def read_loss(path):
    file = Path(path) / "loss.csv"
    # row layout written by OntoState.stats via Hyperparams.loss; runs from
    # before the KL_m stat have 8 columns and runs from before KL_pwak have 9
    # (missing trailing stats read as NaN and plot flat)
    colnames = ["loss", "MSE", "MSE_ghost", "L1_K", "L1_F", "entropy",
                "cossim_b", "cossim_h", "KL_m", "KL_pwak"]
    loss = pd.read_csv(file, names=colnames)
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
