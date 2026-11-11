# experiments/seed-probes: translate.py (predict one seed's heads from
# another code, read against a control code) and bandagree.py (cross-seed
# contribution agreement by band of the data's spectrum). The tests plant
# the structure each instrument exists to find and the structure it must
# not credit.
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments"
                       / "seed-probes"))
from bandagree import band_scores, principal_basis, rotate  # noqa: E402
from translate import (gram, head_accuracy, lsh_codes, majority,  # noqa: E402
                       mlp_probe, onehot, probe, ridge)


def test_onehot_layout():
    S = torch.tensor([[0, 2], [1, 0]])
    F = onehot(S, 3, True)
    assert F.tolist() == [[1, 0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0, 1]]
    assert onehot(S, 3, False).shape == (2, 6)


def test_gram_accumulates_over_chunks():
    rng = np.random.default_rng(0)
    S = torch.as_tensor(rng.integers(0, 4, (50, 3)))
    T = torch.as_tensor(rng.integers(0, 4, (50, 2)))
    F_of = lambda i, j: onehot(S[i:j], 4, True)
    Y_of = lambda i, j: onehot(T[i:j], 4, False)
    G, B = gram(F_of, Y_of, 50, 7)
    F, Y = F_of(0, 50).double(), Y_of(0, 50).double()
    assert torch.allclose(G, F.T @ F) and torch.allclose(B, F.T @ Y)


def planted(n=6000, seed=0):
    """A source code of four 4-way heads; target head 0 is a function of
    source heads 0 and 1 jointly (their sum mod 4, which neither
    determines alone), head 1 copies source head 2, head 2 is noise."""
    rng = np.random.default_rng(seed)
    S = rng.integers(0, 4, (n, 4))
    T = np.stack([(S[:, 0] + S[:, 1]) % 4, S[:, 2],
                  rng.integers(0, 4, n)], 1)
    return torch.as_tensor(S), torch.as_tensor(T)


def test_probe_reads_a_copied_head_and_not_noise():
    S, T = planted()
    acc, lam = probe(lambda i, j: onehot(S[i:j], 4, True), T, 4,
                     4000, 1000, 6000, [1e-3, 1e-1], 512)
    assert acc[1] > 0.99                    # a copy is linear in the one-hot
    assert abs(acc[2] - 0.25) < 0.06        # noise stays at chance
    # a head that is a joint function of two source heads is not additive
    # in their one-hot codes, so the linear probe cannot read it: the
    # instrument measures linear translation, which the notes say
    assert abs(acc[0] - 0.25) < 0.06
    assert lam in (1e-3, 1e-1)


def test_mlp_probe_reads_a_joint_function():
    """The head the ridge cannot read (a sum of two source heads mod 4)
    is within a hidden layer's reach; noise still is not."""
    S, T = planted()
    T = T.int()                     # the codes arrive as int32 from the model
    acc, epoch = mlp_probe(lambda i, j: onehot(S[i:j], 4, False), T, 4,
                           4000, 1000, 6000, 64, 30, 0, batch=256)
    assert acc[0] > 0.9 and acc[1] > 0.99
    assert abs(acc[2] - 0.25) < 0.06
    assert 1 <= epoch <= 30


def test_probe_scores_only_rows_past_fit_and_validation():
    """Targets are a copy on the fit and validation rows and noise on the
    test rows: a probe scored on what it was fitted to would read 1."""
    S, T = planted(seed=1)
    T = T.clone()
    T[5000:, 1] = torch.as_tensor(
        np.random.default_rng(2).integers(0, 4, 1000))
    acc, _ = probe(lambda i, j: onehot(S[i:j], 4, True), T, 4,
                   4000, 1000, 6000, [1e-2], 512)
    assert abs(acc[1] - 0.25) < 0.06


def test_ridge_penalty_is_relative_to_the_gram():
    rng = np.random.default_rng(3)
    F = torch.as_tensor(rng.normal(size=(200, 5)))
    Y = torch.as_tensor(rng.normal(size=(200, 2)))
    W = ridge(F.T @ F, F.T @ Y, 0.1)
    Ws = ridge((10 * F).T @ (10 * F), (10 * F).T @ Y, 0.1)
    assert torch.allclose(10 * Ws, W)       # rescaled features, same fit


def test_head_accuracy_and_majority():
    T = torch.tensor([[0, 1], [1, 1], [2, 1], [0, 0]])
    W = torch.eye(6, dtype=torch.float64)   # features are the targets
    acc = head_accuracy(lambda i, j: onehot(T[i:j], 3, False), W, T, 3, 0, 4,
                        3)
    assert acc.tolist() == [1.0, 1.0]
    assert majority(T.numpy(), 3).tolist() == [0.5, 0.75]


def test_lsh_codes_are_deterministic_partitions():
    X = np.random.default_rng(4).normal(size=(300, 16)).astype(np.float32)
    A = lsh_codes(X, 5, 4, 0, 128)
    assert A.shape == (300, 5) and A.min() >= 0 and A.max() < 4
    assert (A == lsh_codes(X, 5, 4, 0, 64)).all()
    assert (A != lsh_codes(X, 5, 4, 1, 128)).any()
    assert len(np.unique(A[:, 0])) == 4      # every bin used


def test_principal_basis_orders_by_variance():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(4000, 4)) * np.array([1.0, 5.0, 0.2, 2.0])
    lam, Q = principal_basis(X)
    assert (np.diff(lam) <= 0).all()
    assert np.allclose(np.abs(Q[:, 0]), [0, 1, 0, 0], atol=0.05)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10)


def test_band_scores_find_agreement_only_where_it_is():
    """Two models' heads share what they write on coordinates 0:4 and
    write unrelated things on 4:12; a constant offset on every coordinate
    must not read as agreement."""
    rng = np.random.default_rng(6)
    heads, n = 6, 400
    shared = rng.normal(size=(heads, n, 4))
    A = np.concatenate([shared, rng.normal(size=(heads, n, 8))], -1)
    B = np.concatenate([shared, rng.normal(size=(heads, n, 8))], -1)
    B = B[rng.permutation(heads)]            # the same heads at other indices
    A = (A + 3.0).astype(np.float32)
    B = (B + 3.0).astype(np.float32)
    la = np.repeat(np.arange(2), 3)
    I = np.eye(12)
    res = band_scores(rotate(A, I), rotate(B, I), la, [0, 4, 12],
                      np.random.default_rng(0))
    low, high = res
    assert low["matched"] > 0.99 and low["null"] < 0.3
    assert high["matched"] < 0.3
    assert abs(high["matched"] - high["null"]) < 0.1
    assert np.isclose(low["energy_share"] + high["energy_share"], 1.0)
    assert np.isclose(low["energy_share"], 1 / 3, atol=0.05)
    assert len(low["by_layer"]) == 2
