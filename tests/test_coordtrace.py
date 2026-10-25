# experiments/ste-arm/coordtrace.py: a coordinate's error over a run's
# checkpoints and its share of each layer's residual. The tests pin that
# layer 0's residual is the input and layer i's is the input less the
# reconstruction of the layers below it, that the shares are per row before
# they are averaged, and that the error is the last prefix's.
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from coordtrace import residual_shares, trace_sums  # noqa: E402


def test_layer_i_classifies_what_the_layers_below_left():
    X = np.array([[3.0, 4.0, 0.0]])
    # prefix 0 explains coordinate 0, prefix 1 coordinates 0 and 1
    Y = np.array([[[3.0, 0.0, 0.0]], [[3.0, 4.0, 0.0]]])
    sh = residual_shares(X, Y, [0, 1])
    # layer 0 sees X (9 + 16), layer 1 sees X - Y[0] = (0, 4, 0)
    np.testing.assert_allclose(sh, [[9 / 25, 16 / 25], [0.0, 1.0]])


def test_shares_are_per_row_then_averaged():
    # one row with a large norm must not outweigh one with a small norm
    X = np.array([[100.0, 0.0], [0.0, 0.01]])
    Y = np.zeros((1, 2, 2))
    _, share = trace_sums(lambda x: (Y[:, :len(x)], None), X, 2, [0])
    np.testing.assert_allclose(share, [[0.5]])


def test_error_is_the_last_prefix_and_batches_add_up():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)).astype(np.float32)

    def recon(x):
        return np.stack([np.zeros_like(x), x + 0.5]), None

    err, share = trace_sums(recon, X, 16, [2])
    np.testing.assert_allclose(err, 0.25, rtol=1e-6)
    assert share.shape == (2, 1)
    x = X.astype(np.float64)
    np.testing.assert_allclose(share[0, 0], (x[:, 2] ** 2 / (x ** 2).sum(1)).mean())
    np.testing.assert_allclose(share[1, 0], share[0, 0])  # X - 0 at layer 1
