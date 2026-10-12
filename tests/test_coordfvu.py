# experiments/ste-arm/coordfvu.py: per-coordinate error of a model. The
# tests pin that each coordinate's FVU is its own error over its own
# variance (so noise of a fixed size counts most on the smallest
# coordinates), that the shares and the whitened FVU follow the weights,
# that drift is scaled the same way, and that checkpoint steps sort
# numerically.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from coordfvu import (coord_stats, coord_sums, kept_steps,  # noqa: E402
                      output_params, param_moves, report_coords)


def test_fixed_noise_counts_most_on_the_smallest_coordinate():
    rng = np.random.default_rng(0)
    sd = np.array([1.0, 1.0, 1.0, 0.01])
    X = rng.normal(size=(20000, 4)) * sd
    Y = X + 0.005 * rng.normal(size=X.shape)  # the same noise everywhere
    err = ((Y - X) ** 2).mean(0)
    var = X.var(0)
    w = (1 / var) / (1 / var).mean()
    s = coord_stats(err, var, w, [3, 0], drift=err)
    assert np.isclose(s["coords"]["3"]["fvu"], 0.25, rtol=0.05)
    assert s["coords"]["0"]["fvu"] < 1e-4
    assert s["coords"]["3"]["fvu_over_median"] > 1000
    # with inverse-variance weights the whitened FVU is the mean of the
    # coordinates' FVUs, and the shares are their FVUs over the sum
    assert np.isclose(s["fvu"], s["fvu_d_mean"])
    assert np.isclose(s["coords"]["3"]["share"], err[3] / var[3] / (err / var).sum())
    assert np.isclose(s["coords"]["3"]["drift"], s["coords"]["3"]["fvu"])
    assert s["spearman_fvu_d_weight"] > 0.7


def test_coord_sums_takes_the_last_prefix():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 3)).astype(np.float32)

    def recon(x):  # two prefixes; only the last is the reconstruction
        return np.stack([np.zeros_like(x), x + 1.0]), None

    def prev(x):
        return (x + 0.5)[None], None

    err, drift = coord_sums(recon, X, 32, prev)
    np.testing.assert_allclose(err, 1.0, rtol=1e-6)
    np.testing.assert_allclose(drift, 0.25, rtol=1e-6)
    assert coord_sums(recon, X, 32)[1] is None


def test_raw_ratios_match_the_spread_when_error_scales_with_it():
    sd = np.array([1.0, 2.0, 4.0, 0.1])
    var = sd ** 2
    err = 0.04 * var  # 20% of each coordinate's spread
    s = coord_stats(err, var, 1 / var, [3], drift=0.01 * var)
    r = s["coords"]["3"]
    assert np.isclose(r["fvu_over_median"], 1.0)
    assert np.isclose(r["err_raw_over_median"], sd[3] / np.median(sd))
    assert np.isclose(r["drift_raw_over_median"], sd[3] / np.median(sd))


def test_output_params_reads_decoder_rows_or_atoms():
    W = np.arange(12.0).reshape(3, 4)  # (out = 3, in = 4)
    np.testing.assert_array_equal(output_params({"decoder": {"weights": W}}, 3), W.T)
    atoms = {f"dictencs_{i}": {"dict": {"weights": np.full((2, 5, 3), float(i))}}
             for i in range(2)}
    got = output_params(atoms, 3)
    assert got.shape == (20, 3) and set(got[:10, 0]) == {0.0} and set(got[10:, 0]) == {1.0}


def test_param_moves_scales_by_the_median():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(500, 5))
    step = np.array([1.0, 1.0, 1.0, 1.0, 3.0]) * 1e-3
    B = A + step * rng.choice([-1, 1], size=A.shape)
    r = param_moves(A, B, [4, 0])
    assert np.isclose(r["change_median"], 1e-3)
    assert np.isclose(r["coords"]["4"]["change_over_median"], 3.0)
    assert np.isclose(r["coords"]["0"]["change_over_median"], 1.0)


def test_report_coords_adds_named_ones_once():
    w = np.array([1.0, 5.0, 0.0, 3.0, 0.5])
    assert report_coords(w, 2, []) == [1, 3]
    # a masked coordinate (weight 0) is reported only when named, and a
    # named one already among the largest is not repeated
    assert report_coords(w, 2, [2, 3, 2]) == [1, 3, 2]
    w0 = w.copy()
    w0[2] = 0
    s = coord_stats(np.ones(5), np.ones(5), w0, report_coords(w0, 2, [2]))
    assert s["coords"]["2"]["share"] == 0 and s["coords"]["2"]["fvu"] == 1


def test_kept_steps_sorts_numerically(tmp_path):
    for s in ["9500", "10000", "369500", "cossim.pdf"]:
        (tmp_path / s).mkdir()
    assert kept_steps(str(tmp_path)) == [9500, 10000, 369500]
