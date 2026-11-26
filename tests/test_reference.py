# tests/reference: fixtures for readers of a categorical dictionary. The
# forward must reproduce every stored array, selections exactly; the legacy
# fixture must load through migrate_spec's inference of const0; and the
# fixtures must hold what their metadata says (a near tie decided in
# float32, and rescaled rows a gain-shape model codes alike at layer 0).
import sys
from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / "reference"
sys.path.insert(0, str(REF))
from build import NEAR_TIE_REL, load, run  # noqa: E402

NAMES = sorted(p.name for p in REF.iterdir() if (p / "meta.json").exists())


def test_the_fixtures_are_present():
    assert {"legacy_const0", "direct_gain_ste"} <= set(NAMES)


def test_the_direct_gain_shape_model_is_positively_homogeneous():
    """With no decoder and every layer gain-shaped (resid_first), scaling a
    row scales every layer's reconstruction and leaves every code alone;
    gain_clip's cap at the row's norm scales with it."""
    model, _, X, expected, meta = load("direct_gain_ste")
    rows, base = meta["rows"], meta["rescaled_of"]
    for key in ("rescaled_0.5", "rescaled_3"):
        r, c = rows[key], float(key.split("_")[1])
        for l in range(model.l):
            top = expected[f"L{l}/top2_index"][..., 0]
            assert (top[r] == top[base]).all(), (key, l)
            np.testing.assert_allclose(expected[f"L{l}/Y"][r],
                                       c * expected[f"L{l}/Y"][base],
                                       rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_forward_reproduces_the_fixture(name):
    model, params, X, expected, meta = load(name)
    got = run(model, params, X, meta["temperature"])
    assert set(got) == set(expected)
    for key, want in expected.items():
        if key.endswith("top2_index"):
            # the winner exactly; the runner-up too, except at the near tie
            assert (got[key][..., 0] == want[..., 0]).all(), key
            continue
        scale = max(1.0, float(np.abs(want).max()))
        np.testing.assert_allclose(got[key], want, rtol=meta["rtol"],
                                   atol=meta["atol_rel"] * scale,
                                   err_msg=key)


def test_the_legacy_spec_infers_const0():
    model, params, _, _, meta = load("legacy_const0")
    assert "const0" in meta["spec_without"]
    assert model.const0 is False
    assert params["dictencs_0"]["classifier"]["weight"].shape[-1] == model.d_in
    assert params["dictencs_1"]["classifier"]["weight"].shape[-1] == model.d_in + 1


@pytest.mark.parametrize("name", NAMES)
def test_the_near_tie_is_decided_in_float32_only(name):
    _, _, _, expected, meta = load(name)
    tie = meta["near_tie"]
    K = expected[f"L{tie['layer']}/K"][meta["rows"]["near_tie"], tie["head"]]
    s = np.sort(K)
    gap = (s[-1] - s[-2]) / np.abs(K).max()
    assert np.isclose(gap, tie["gap_over_max_logit"], rtol=1e-6)
    # far above float32's resolution, below float16's
    assert 50 * np.finfo(np.float32).eps < gap < np.finfo(np.float16).eps
    assert np.isclose(gap, NEAR_TIE_REL, rtol=0.05)
    clear = expected["L0/K"][meta["rows"]["clear_winner"]]
    s = np.sort(clear, axis=-1)
    assert ((s[:, -1] - s[:, -2]) / np.abs(clear).max(-1)).min() > 0.1


def test_rescaled_rows_keep_their_layer0_codes_under_gain_shape():
    model, _, X, expected, meta = load("direct_gain_ste")
    assert model.resid_gain
    rows = meta["rows"]
    base = meta["rescaled_of"]
    for key in ("rescaled_0.5", "rescaled_3"):
        r = rows[key]
        assert np.allclose(X[r], X[base] * float(key.split("_")[1]))
        top = expected["L0/top2_index"][..., 0]
        assert (top[r] == top[base]).all()
        np.testing.assert_allclose(expected["L0/G"][r],
                                   expected["L0/G"][base]
                                   * float(key.split("_")[1]), rtol=1e-5)
