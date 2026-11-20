# experiments/lcm-organism: the float16 copy of a cache (f16cache.py). Every
# row is the source's rounded to float16, across chunks that do not divide
# the cache; the sidecars are reachable under the copy's stem; meta.json
# records the dtype and the conversion; values beyond float16's range and a
# copy that would fill the disk are refused.
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

LCM = Path(__file__).parents[1] / "experiments" / "lcm-organism"
sys.path.insert(0, str(LCM))
from f16cache import copy_f16, main, rounding_cost, to_f16  # noqa: E402


def cache(tmp_path, n=1000, d=8):
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal((n, d))).astype(np.float32)
    x[3, 0], x[4, 1] = 1e-6, 1e-9  # a float16 subnormal, and a value that rounds to zero
    c = tmp_path / "c.npy"
    np.save(c, x)
    np.save(tmp_path / "c.index.npy", np.arange(3 * n).reshape(n, 3))
    np.save(tmp_path / "c.mse_weights.npy", np.ones(d, np.float32))
    (tmp_path / "c.meta.json").write_text(json.dumps({"dtype": "float32", "holdout": 100}))
    return c, x


def test_the_copy_is_the_cache_rounded_with_its_sidecars(tmp_path):
    c, x = cache(tmp_path)
    main(["--cache", str(c), "--chunk", "128", "--sample", "50", "--min-free-gb", "0"])
    out = tmp_path / "c_f16.npy"
    y = np.load(out)
    assert y.dtype == np.float16 and np.array_equal(y, x.astype(np.float16))
    assert not (tmp_path / "c_f16.npy.partial").exists()
    for side in ("index.npy", "mse_weights.npy"):
        assert os.path.samefile(tmp_path / f"c.{side}", tmp_path / f"c_f16.{side}")
    meta = json.loads((tmp_path / "c_f16.meta.json").read_text())
    assert meta["dtype"] == "float16" and meta["holdout"] == 100
    conv = meta["float16"]["conversion"]
    assert conv["subnormal"] >= 1 and conv["to_zero"] >= 1 and conv["nonfinite"] == 0
    assert meta["float16"]["rounding"]["tail"]["rows"] == 100
    assert not (tmp_path / "c_f16.c_f16.meta.json").exists()


def test_bulk_puts_the_array_elsewhere_and_keeps_the_rest_beside_out(tmp_path):
    c, x = cache(tmp_path)
    bulk = tmp_path / "other_disk"
    main(["--cache", str(c), "--bulk", str(bulk), "--sample", "50", "--min-free-gb", "0"])
    out = tmp_path / "c_f16.npy"
    assert out.is_symlink() and out.resolve() == (bulk / "c_f16.npy").resolve()
    assert np.array_equal(np.load(out), x.astype(np.float16))
    assert os.path.samefile(tmp_path / "c.index.npy", tmp_path / "c_f16.index.npy")
    assert not (tmp_path / "c_f16.index.npy").is_symlink()
    meta = json.loads((tmp_path / "c_f16.meta.json").read_text())
    assert meta["float16"]["array"] == str((bulk / "c_f16.npy").resolve())
    assert sorted(p.name for p in bulk.iterdir()) == ["c_f16.npy"]
    with pytest.raises(SystemExit, match="exists"):
        main(["--cache", str(c), "--bulk", str(bulk), "--min-free-gb", "0"])


def test_syncing_mid_copy_keeps_every_row(tmp_path):
    c, x = cache(tmp_path, n=777)
    out = tmp_path / "o.npy"
    stats = copy_f16(c, out, chunk=64, sync_every=2, log=lambda *_: None)
    y = np.load(tmp_path / "o.npy.partial")
    assert np.array_equal(y, x.astype(np.float16)) and stats["n"] == 777


def test_values_beyond_float16_are_refused():
    x = np.array([[1.0, 7e4]], np.float32)
    with pytest.raises(ValueError, match="beyond"):
        to_f16(x)
    y, s = to_f16(np.array([[np.nan, 1.0]], np.float32))
    assert s["nonfinite"] == 1 and np.isnan(y[0, 0])


def test_a_copy_that_would_fill_the_disk_is_refused(tmp_path):
    c, _ = cache(tmp_path)
    with pytest.raises(SystemExit, match="free"):
        main(["--cache", str(c), "--min-free-gb", "1e9"])
    assert not list(tmp_path.glob("c_f16*"))


def test_rounding_costs_little_and_counts_each_weighting():
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal((500, 16))).astype(np.float32)
    r = rounding_cost(x, {"flat": np.ones(16), "half": np.r_[np.ones(8), np.zeros(8)]})
    assert 0 < r["rel_err_median"] < 1e-3
    assert set(r["fvu_w"]) == {"flat", "half"} and 0 < r["fvu_w"]["flat"] < 1e-6
