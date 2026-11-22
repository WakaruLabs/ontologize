# experiments/ste-arm/cellshape.py: whether a head's cells are convex cones,
# whether they are convex, and whether it ignores its input's sign. Heads
# with linear logits keep every mean of their own rows, rescaled or not, and
# give negated rows another entry; an affine cell that is not a cone keeps
# its raw means and loses rescaled ones; a head whose logits are squares of
# linear forms is sign-blind and splits each cell into two opposite lobes,
# whose cross-lobe means leave the cell.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from cellshape import cell_means, shape_tests  # noqa: E402


def sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    S = rng.standard_normal((n, d))
    return S / np.linalg.norm(S, axis=1, keepdims=True)


def test_linear_heads_keep_every_mean():
    rng = np.random.default_rng(0)
    S = sphere(4000, 5, rng)
    W = rng.standard_normal((3, 6, 5))                   # 3 heads, k 6
    assign = lambda P: np.argmax(np.einsum("bd,hkd->bhk", P, W), -1)
    res = shape_tests(S, assign(S), 6, assign, lambda P: -P, 20, 8, rng)
    for key in ("mean2", "mean8", "mean32", "centroid"):
        assert np.allclose(res[key], 1.0)
        assert np.allclose(res[f"{key}_raw"], 1.0)
        assert np.allclose(res[f"{key}_radial"], 0.0)
    # argmax of -s is the argmin of s, never the argmax for k > 1
    assert np.allclose(res["negation"], 0.0)


def test_an_affine_cell_keeps_raw_means_and_loses_rescaled_ones():
    # x_1 <= 3/4 on the unit circle: the cell is convex but not a cone, so
    # a midpoint of two of its points stays in it and the midpoint pushed
    # back to the circle can leave it
    rng = np.random.default_rng(3)
    S = sphere(4000, 2, rng)
    assign = lambda P: (P[:, :1] > 0.75).astype(int)        # one head, k 2
    res = shape_tests(S, assign(S), 2, assign, lambda P: -P, 20, 64, rng)
    for key in ("mean2", "mean8", "mean32", "centroid"):
        assert np.allclose(res[f"{key}_raw"], 1.0)
    assert res["mean2"][0] < 0.99 and res["mean2_radial"][0] > 0.01
    assert np.isclose(res["mean2_radial"][0], 1.0 - res["mean2"][0])


def test_squared_logits_are_sign_blind_and_split_cells():
    rng = np.random.default_rng(1)
    S = sphere(4000, 2, rng)
    W = np.eye(2)[None]                                  # one head, k 2
    assign = lambda P: np.argmax(np.einsum("bd,hkd->bhk", P, W) ** 2, -1)
    res = shape_tests(S, assign(S), 2, assign, lambda P: -P, 20, 64, rng)
    assert np.allclose(res["negation"], 1.0)
    # a pair from opposite lobes of |u1| > |u2| averages toward the u2 axis
    assert res["mean2"][0] < 0.8


def test_cell_means_rescale_to_the_rows_norm():
    rng = np.random.default_rng(2)
    S = 3.0 * sphere(500, 4, rng)
    A = (S[:, :1] > 0).astype(int)
    P, ids = cell_means(S, A, 2, 8, 5, 20, rng)
    assert P.shape == (10, 4) and ids.shape == (10, 2)
    assert np.allclose(np.linalg.norm(P, axis=1), 3.0)
    C, cid = cell_means(S, A, 2, 0, 5, 20, rng)
    assert C.shape == (2, 4)
    assert set(map(tuple, cid)) == {(0, 0), (0, 1)}
    # the raw means are the same groups' means, before rescaling
    P2, _, R2 = cell_means(S, A, 2, 8, 5, 20, np.random.default_rng(4), raw=True)
    assert np.allclose(P2, R2 * (3.0 / np.linalg.norm(R2, axis=1, keepdims=True)),
                       atol=1e-5)
    assert (np.linalg.norm(R2, axis=1) < 3.0).all()
