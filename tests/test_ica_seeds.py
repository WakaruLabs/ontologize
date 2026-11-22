# experiments/seed-stability/ica_seeds.py: whether FastICA's sources reproduce
# on a second sample. The tests pin the matching (a relabelled, sign-flipped
# copy matches exactly and its runner-up is lower) and the comparison on
# planted sources: non-Gaussian sources fitted on two disjoint samples match
# closely, a random rotation of them does not, and two Gaussian sources, which
# no sample pins, do not.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-stability"))
from ica_seeds import abs_corr, compare, match  # noqa: E402
from kmeans_seeds import ica_frame  # noqa: E402


def test_a_relabelled_sign_flipped_copy_matches_exactly():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((5000, 6))
    perm = rng.permutation(6)
    T = S[:, perm] * rng.choice([-1, 1], 6)
    partner, val, runner = match(abs_corr(S, T))
    assert np.array_equal(perm[partner], np.arange(6))
    assert np.allclose(val, 1) and (runner < 0.1).all()


def test_non_gaussian_sources_reproduce_across_samples_and_gaussian_ones_do_not():
    rng = np.random.default_rng(1)
    n, m = 100000, 20000
    S = np.concatenate([rng.laplace(size=(n, 3)), rng.uniform(-1, 1, (n, 2)),
                        rng.standard_normal((n, 2))], 1)
    S = (S - S.mean(0)) / S.std(0)
    X = (S @ rng.standard_normal((7, 7))).astype(np.float32)
    fa = ica_frame(X[:m], 0, 1000)
    gauss = []
    for j in (1, 2, 3):                          # three disjoint samples for B
        fb = ica_frame(X[j * m:(j + 1) * m], j, 1000)
        r = compare(fa, fb, X[4 * m:])
        assert (np.sort(r["corr"])[2:] > 0.98).all()   # the five non-Gaussian
        assert np.median(r["rotated"]) < 0.8
        gauss.append(np.sort(r["corr"])[:2].min())
    # the two Gaussian sources are a rotation within their plane that each
    # sample sets anew
    assert min(gauss) < 0.95
