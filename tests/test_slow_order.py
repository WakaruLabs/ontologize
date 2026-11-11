# experiments/slow-order/orderloss.py: the dependence term between slow
# heads. The tests pin what the term measures (1 for copies, 0 for
# independent heads, with no floor at a small batch), that its gradient
# goes where the order says, and that it leaves the Fano term alone; and
# order.py's statistics of whether two seeds' blocks match index for index.
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "slow-order"))
from order import diagonal, prefix, trend  # noqa: E402
from orderloss import (OrderedPairHyperparams, head_dependence,  # noqa: E402
                       hsic_u)

from ontologize.fns.hsic import hsic_unbiased  # noqa: E402

K, H = 8, 6


def codes(A: np.ndarray) -> jnp.ndarray:
    """(n, h) entries -> (1, n, h*k) one-hot, the ghost slot's layout."""
    return jnp.asarray(np.eye(K, dtype=np.float32)[A].reshape(1, len(A), -1))


def test_hsic_u_matches_the_package_estimator():
    rng = np.random.default_rng(0)
    A = rng.integers(0, K, (40, 2))
    P = np.eye(K, dtype=np.float32)[A]
    Kx, Ky = P[:, 0] @ P[:, 0].T, P[:, 1] @ P[:, 1].T
    off = 1.0 - np.eye(40, dtype=np.float32)
    assert np.isclose(hsic_u(jnp.asarray(Kx * off), jnp.asarray(Ky * off)),
                      hsic_unbiased(jnp.asarray(Kx), jnp.asarray(Ky)),
                      rtol=1e-5)


def test_copies_read_one_and_independent_heads_zero():
    rng = np.random.default_rng(1)
    a = rng.integers(0, K, 256)
    same = np.stack([a] * H, 1)
    relabeled = np.stack([(a + j) % K for j in range(H)], 1)
    for ordered in (True, False):
        assert np.isclose(head_dependence(codes(same), H, 0, ordered), 1.0,
                          atol=1e-5)
        # a partition is the same under any renaming of its entries
        assert np.isclose(head_dependence(codes(relabeled), H, 0, ordered),
                          1.0, atol=1e-5)
    # no floor at a small batch: the mean over draws is zero, where the
    # biased estimator's would not be
    vals = [float(head_dependence(
        codes(np.random.default_rng(s).integers(0, K, (32, H))), H, 0, True))
        for s in range(60)]
    assert abs(np.mean(vals)) < 3 * np.std(vals) / np.sqrt(len(vals)) + 1e-3


def test_only_the_first_heads_are_scored():
    rng = np.random.default_rng(2)
    a = rng.integers(0, K, 128)
    A = rng.integers(0, K, (128, H))
    A[:, 4] = A[:, 5] = a                    # copies, outside the slow block
    assert abs(float(head_dependence(codes(A), H, 4, True))) < 0.05
    assert float(head_dependence(codes(A), H, 0, True)) > 0.05


def soft(seed: int, n: int = 64) -> jnp.ndarray:
    """Soft codes with two near-copies (heads 0 and 2), so the term has a
    gradient worth routing."""
    logits = jax.random.normal(jax.random.PRNGKey(seed), (1, n, H, K))
    logits = logits.at[:, :, 2].set(logits[:, :, 0] + 0.1 * logits[:, :, 2])
    return jax.nn.softmax(3.0 * logits, -1).reshape(1, n, H * K)


def test_ordered_gradient_spares_the_earlier_head():
    P = soft(0)
    g = jax.grad(lambda p: head_dependence(p, H, 0, True))(P)
    g = np.asarray(g).reshape(-1, H, K)
    per_head = np.abs(g).sum((0, 2))
    assert per_head[0] < 1e-9               # head 0 answers to no one
    assert per_head[2] > 1e-6               # its near-copy is moved off it
    gs = jax.grad(lambda p: head_dependence(p, H, 0, False))(P)
    per_sym = np.abs(np.asarray(gs).reshape(-1, H, K)).sum((0, 2))
    assert per_sym[0] > 1e-6                # symmetric: both are moved


def test_ordered_and_symmetric_agree_in_value():
    P = soft(1)
    assert np.isclose(head_dependence(P, H, 0, True),
                      head_dependence(P, H, 0, False), rtol=1e-4, atol=1e-6)


def test_pair_loss_adds_the_term_and_keeps_the_raw_value():
    P = soft(2, 128)
    base = OrderedPairHyperparams(1, 1, 1, pair_term="fano", pair_h=H)
    with_dec = OrderedPairHyperparams(1, 1, 1, pair_term="fano", pair_h=H,
                                      dec=2.0)
    t0, r0 = base.pair_loss(P)
    t1, r1 = with_dec.pair_loss(P)
    assert np.isclose(r0, r1)               # column 2 stays the Fano bound
    assert np.isclose(t1 - t0, 2.0 * head_dependence(P, H, 0, True),
                      rtol=1e-5)


# order.py: whether two seeds' blocks match index for index

def test_diagonal_tells_an_order_from_a_permuted_match():
    rng = np.random.default_rng(0)
    s = 12
    noise = 0.05 * rng.random((s, s))
    kept = diagonal(np.eye(s) + noise)
    assert kept["on_diagonal"] == 1.0 and kept["best_is_own"] == 1.0
    assert np.isclose(kept["index_rank"], 1.0)
    assert kept["same_index"] > 1.0 and np.isclose(kept["chance"], 1 / s)
    # the same heads under another index: as well matched, off the diagonal
    perm = np.roll(np.arange(s), 5)
    moved = diagonal((np.eye(s) + noise)[:, perm])
    assert moved["on_diagonal"] == 0.0 and moved["best_is_own"] == 0.0
    assert moved["same_index"] < 0.1


def test_trend_skips_undefined_heads():
    v = np.array([0.9, 0.7, np.nan, 0.4, 0.1])
    assert np.isclose(trend(v), -1.0)
    assert np.isnan(trend(np.array([0.1, np.nan, np.nan])))


def test_prefix_reads_only_earlier_heads():
    s = 9
    W = np.zeros((s, s))
    W[np.tril_indices(s, -1)] = 1.0          # below the diagonal: earlier
    W[np.triu_indices(s, 1)] = 7.0           # later heads, never read
    assert prefix(W) == {"first_half": 1.0, "second_half": 1.0}
    W[5:, :5] = 3.0                          # late heads depend on early ones
    p = prefix(W)
    assert p["first_half"] == 1.0 and p["second_half"] > 1.5
