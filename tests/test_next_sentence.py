# experiments/next-sentence: adjacent sentence pairs from a document-ordered
# cache, the split that keeps every held-out document out of training, the
# pair source, and train()'s (input, target) batches.
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from ontologize.training import ontostate
from ontologize.training.config import Hyperparams, Metadata, TrainingEnv

from conftest import KW

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "next-sentence"))
from pairs import PairSource, adjacent, pair_rows  # noqa: E402


def test_adjacent_needs_the_same_document_and_the_next_segment():
    # doc 0: segments 0 1 3 (2 was unusable and skipped); doc 1: 0 1;
    # doc 2: a single sentence
    index = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 3],
                      [1, 0, 0], [1, 1, 1],
                      [2, 0, 0]], np.int32)
    assert adjacent(index, 0, len(index)).tolist() == [0, 3]
    # a window that ends inside a document keeps only pairs inside it
    assert adjacent(index, 0, 4).tolist() == [0]
    assert adjacent(index, 3, 6).tolist() == [3]


@pytest.fixture
def cache(tmp_path):
    """60 documents of 5 sentences each, row i holding the value i, with
    every document's third segment skipped."""
    n_docs, per, d = 60, 5, KW["d_in"]
    seg = np.array([0, 1, 3, 4, 5])
    rows = np.arange(n_docs * per)
    index = np.stack([rows // per, rows % per, seg[rows % per]], 1)
    np.save(tmp_path / "c.npy",
            np.repeat(rows[:, None], d, 1).astype(np.float32))
    np.save(tmp_path / "c.index.npy", index.astype(np.int32))
    np.save(tmp_path / "c.docstart.npy", np.arange(0, n_docs * per, per))
    return tmp_path / "c.npy"


def test_pair_rows_keeps_the_held_out_documents_whole(cache):
    train, held = pair_rows(cache, train_rows=200, holdout=12)
    index = np.load(cache.with_name("c.index.npy"))
    n = len(index)
    # a 12-row request snaps up to 3 whole documents (15 rows)
    assert held.min() >= n - 15 and held.max() < n - 1
    assert train.max() + 1 < 200
    assert not set(index[train, 0]) & set(index[held, 0])
    # 3 adjacent pairs per document: 0-1, 3-4, 4-5 (the skip breaks 1-3)
    assert len(held) == 3 * 3


def test_pair_source_serves_the_next_row_or_the_same_one(cache):
    rows = pair_rows(cache, 300, 12)[0]
    nxt = PairSource(cache, rows, "next")
    slf = PairSource(cache, rows, "self")
    for j in (0, 5, len(rows) - 1):
        i = rows[j]
        (x, y), (x_s, y_s) = nxt[j], slf[j]
        assert x[0] == i and y[0] == i + 1
        assert x_s[0] == i and y_s[0] == i
    # pickles without its memmap, as grain workers need
    assert pickle.loads(pickle.dumps(nxt))[3][1][0] == rows[3] + 1
    with pytest.raises(ValueError):
        PairSource(cache, rows, "previous")


@pytest.mark.parametrize("objective", ["next", "self"])
def test_train_takes_the_target_from_the_pair(cache, tmp_path, monkeypatch,
                                              objective):
    seen = []
    update = ontostate.update

    def spy(state, lossfn, rng, X, Y, *args, **kwargs):
        seen.append((np.asarray(X)[:, 0], np.asarray(Y)[:, 0]))
        return update(state, lossfn, rng, X, Y, *args, **kwargs)

    monkeypatch.setattr(ontostate, "update", spy)
    d = KW["d_in"]
    hyper = Hyperparams(d, d, 8, 1, 5e-5, 0.0, 1.0,
                        "none", "normal", "featvar", 0.0, 0.0, 0.0,
                        ghost=False)
    model = hyper.ontologizer(d, d, KW["e_dec"], KW["k"], KW["h"], 1, n=2,
                              gate="none", forward="resid",
                              dtype_str="float32", dtype_p_str="float32")
    meta = Metadata("embedding", out_path=tmp_path / "run", save_each=2,
                    checkpoint_each=2)
    src = PairSource(cache, pair_rows(cache, 80, 12)[0], objective)
    TrainingEnv(model, hyper, meta,
                kwargs_loader={"d": d, "shuffle": True}).train(src)
    assert seen
    step = 1 if objective == "next" else 0
    for x, y in seen:
        assert np.array_equal(y, x + step)


def test_same_doc_pairs_count_segments_not_rows():
    import transitions as tr
    # doc 0: segments 0 1 3 4 (2 skipped); doc 1: segments 0 2
    index = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 3], [0, 3, 4],
                      [1, 0, 0], [1, 1, 2]], np.int32)
    i, j = tr.same_doc_pairs(index, 1)
    assert list(zip(i, j)) == [(0, 1), (2, 3)]
    assert np.array_equal(i, adjacent(index, 0, len(index)))
    i, j = tr.same_doc_pairs(index, 2)
    assert list(zip(i, j)) == [(1, 2), (4, 5)]
    assert not len(tr.same_doc_pairs(index, 9)[0])


def test_other_doc_partners_share_the_group_but_not_the_document():
    import transitions as tr
    rng = np.random.default_rng(0)
    doc = np.repeat(np.arange(6), 4)
    group = np.repeat([0, 0, 1, 1, 1, 2], 4)       # group 2 has one document
    anchors = np.arange(len(doc))
    j = tr.other_doc_partners(anchors, doc, group, rng)
    single = group[anchors] == 2
    assert (j[single] == -1).all()
    assert (doc[j[~single]] != doc[anchors[~single]]).all()
    assert (group[j[~single]] == group[anchors[~single]]).all()


def test_kappa_auc_and_retrieval():
    import transitions as tr
    rng = np.random.default_rng(0)
    A = rng.integers(8, size=(3, 20000))
    A[2] = 5                                        # one entry everywhere
    same = tr.kappa(A, A, 8)
    assert np.allclose(same[:2], 1.0) and np.isnan(same[2])
    indep = tr.kappa(A, rng.integers(8, size=A.shape), 8)
    assert np.abs(indep[:2]).max() < 0.02
    assert tr.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert tr.auc(np.ones(5), np.ones(7)) == 0.5
    # a successor identical to its anchor is always the nearest
    Z = rng.normal(size=(40, 6)).astype(np.float32)
    Z[1::2] = Z[0::2]
    i, j = np.arange(0, 40, 2), np.arange(1, 40, 2)
    negs = np.stack([rng.permutation(np.arange(0, 40, 2))[:20]
                     for _ in range(5)])
    negs = np.where(negs == i, (negs + 2) % 40, negs)
    for metric in ("dist", "cos"):
        Zm = Z / np.linalg.norm(Z, axis=1, keepdims=True) \
            if metric == "cos" else Z
        top1, rank = tr.retrieval(Zm, i, j, negs, metric)
        assert top1 == 1.0 and rank == 1.0


def test_doc_partners_stay_in_the_document_at_distance():
    from pairs import doc_partners
    # doc 0: segments 0 1 3 4; doc 1: 0 1 (nothing 2 apart); doc 2: 0 5
    index = np.array([[0, 0, 0], [0, 1, 1], [0, 2, 3], [0, 3, 4],
                      [1, 0, 0], [1, 1, 1], [2, 0, 0], [2, 1, 5]], np.int32)
    valid = {0: {2, 3}, 1: {2, 3}, 2: {0, 1}, 3: {0, 1}, 6: {7}, 7: {6}}
    seen = {r: set() for r in valid}
    for s in range(50):
        # a chunk of 3 rows cuts across documents
        rows, partners = doc_partners(index, 0, len(index), 2,
                                      np.random.default_rng(s), chunk=3)
        assert set(rows.tolist()) == set(valid)
        for r, j in zip(rows.tolist(), partners.tolist()):
            assert j in valid[r]
            seen[r].add(j)
    assert seen == valid                 # every allowed partner turns up
    rows, partners = doc_partners(index, 4, 8, 2, np.random.default_rng(0))
    assert sorted(zip(rows.tolist(), partners.tolist())) == [(6, 7), (7, 6)]


def test_stacked_records_unstack_to_paired_rows(cache):
    import jax.numpy as jnp
    from pairloss import unstack
    from pairs import StackedPairSource
    d = KW["d_in"]
    src = StackedPairSource(cache, np.array([0, 5, 10]),
                            np.array([2, 8, 12]))
    assert src[1].shape == (2 * d,)
    assert (src[1][0], src[1][d]) == (5, 8)
    assert pickle.loads(pickle.dumps(src))[2][d] == 12
    B = np.stack([src[m] for m in range(3)])      # (b, 2d), as grain batches
    U = np.asarray(unstack(jnp.asarray(B)))
    assert U.shape == (6, d)
    assert U[:, 0].tolist() == [0, 5, 10, 2, 8, 12]   # row r pairs r + b


def test_switch_share_counts_the_slow_heads_that_change():
    import jax.numpy as jnp
    from pairloss import switch_share
    rng = np.random.default_rng(0)
    l, b, h, k = 2, 4000, 6, 8

    def onehot(A):
        return jnp.asarray(np.eye(k, dtype=np.float32)[A].reshape(
            l, 2 * b, h * k))

    a = rng.integers(k, size=(l, b, h))
    assert float(switch_share(onehot(np.concatenate([a, a], 1)), h, 0)) == 0
    other = np.concatenate([a, rng.integers(k, size=(l, b, h))], 1)
    assert abs(float(switch_share(onehot(other), h, 0)) - (1 - 1 / k)) < 0.01
    # only the first `heads` heads of each layer count
    other[:, b:, :2] = other[:, :b, :2]
    assert float(switch_share(onehot(other), h, 2)) == 0
    assert float(switch_share(onehot(other), h, 0)) > 0.5


def test_fano_bound_rewards_kept_entries_not_collapse():
    import jax
    import jax.numpy as jnp
    from pairloss import fano_bound
    rng = np.random.default_rng(0)
    l, b, h, k = 1, 20000, 4, 8

    def onehot(A):
        return jnp.asarray(np.eye(k, dtype=np.float32)[A].reshape(
            l, 2 * b, h * k))

    a = rng.integers(k, size=(l, b, h))
    other = rng.integers(k, size=(l, b, h))
    kept = float(fano_bound(onehot(np.concatenate([a, a], 1)), h, 0))
    indep = float(fano_bound(onehot(np.concatenate([a, other], 1)), h, 0))
    collapsed = onehot(np.zeros((l, 2 * b, h), int))
    assert abs(kept - 1) < 0.01 and abs(indep) < 0.01
    assert abs(float(fano_bound(collapsed, h, 0))) < 1e-3
    half = np.where(rng.random((l, b, h)) < 0.5, a, other)
    assert 0.05 < float(fano_bound(onehot(np.concatenate([a, half], 1)),
                                   h, 0)) < 0.95
    # finite at a collapsed head, where the plain entropy's gradient is not
    g = jax.grad(lambda P: fano_bound(P, h, 0))(collapsed)
    assert np.isfinite(np.asarray(g)).all()


@pytest.mark.parametrize("s_pair,term", [(0.0, "agree"), (1.0, "agree"),
                                         (1.0, "fano")])
def test_pair_training_logs_the_pair_term(cache, tmp_path, s_pair, term):
    from pairloss import PairHyperparams
    from pairs import StackedPairSource, doc_partners
    d = KW["d_in"]
    index = np.load(cache.with_name("c.index.npy"))
    rows, partners = doc_partners(index, 0, 200, 2, np.random.default_rng(0))
    hyper = PairHyperparams(d, d, 8, 1, 5e-5, 0.0, 1.0,
                            "none", "normal", "featvar", 0.0, 0.0, 0.0,
                            ghost=False, s_pair=s_pair, pair_term=term,
                            pair_h=KW["h"], pair_heads=2)
    model = hyper.ontologizer(d, d, KW["e_dec"], KW["k"], KW["h"], 1, n=2,
                              gate="none", forward="resid", select="ste",
                              dtype_str="float32", dtype_p_str="float32")
    meta = Metadata("embedding", out_path=tmp_path / "run", save_each=2,
                    checkpoint_each=2)
    TrainingEnv(model, hyper, meta, kwargs_loader={"d": d, "shuffle": True}
                ).train(StackedPairSource(cache, rows, partners))
    loss = np.genfromtxt(tmp_path / "run" / "loss.csv", delimiter=",")
    raw = loss[:, 2]          # switch share, or the bound as a fraction
    assert len(loss) and np.isfinite(loss[:, 0]).all()
    assert np.isfinite(raw).all() and (raw <= 1).all()
    if term == "agree":
        assert (raw >= 0).all()


def test_setup_takes_a_hyperparams_subclass_only(cache, tmp_path):
    import train_pair
    from ontologize.training.config import Hyperparams
    cfg = train_pair.parser().parse_args(
        ["--cache", str(cache), "--out", str(tmp_path / "run")])
    assert (cfg.pair_heads, cfg.min_d, cfg.s_pair) == (38, 2, 0)
    # the recipe's --dict-init-scale picks DictWDHyperparams, which plain
    # Hyperparams does not subclass
    with pytest.raises(AssertionError, match="must subclass"):
        train_pair.train_ste.setup(train_pair.train_ste.finish(cfg),
                                   src=list(range(4)), hp=Hyperparams)


def test_seedmatch_splits_the_match_by_block():
    from seedmatch import block_match, slow_mask, slow_to_slow
    # two layers of three heads, the first of each slow: heads 0 and 3
    assert slow_mask(6, 3, 1).tolist() == [True, False, False,
                                           True, False, False]
    M = np.full((4, 4), 0.01)
    M[0, 1], M[1, 0] = 0.9, 0.8           # slow heads 0 and 1 swap places
    M[2, 2], M[3, 3] = 0.5, 0.4
    slow = np.array([True, True, False, False])
    v = block_match(M, np.flatnonzero(slow), np.flatnonzero(slow))
    assert sorted(v.tolist()) == [0.8, 0.9]
    assert slow_to_slow(M, slow, slow) == 1.0
    # the whole-model assignment can pair a slow head outside the block
    M[0, 3] = 0.95
    M[3, 3] = 0.0
    M[3, 1] = 0.6
    assert slow_to_slow(M, slow, slow) == 0.5


def test_within_label_nmi_ignores_agreement_through_the_label():
    from seedmatch import nmi, shuffle_within, within_label_nmi
    rng = np.random.default_rng(0)
    n, k = 20000, 8
    labels = rng.integers(4, size=n)
    a = rng.integers(k, size=n)
    assert abs(nmi(a, a, k) - 1.0) < 1e-9
    assert nmi(a, rng.integers(k, size=n), k) < 0.01
    # two codes that both just name the label agree overall, not within it
    a_lab, b_lab = labels % k, (3 * labels) % k
    assert nmi(a_lab, b_lab, k) > 0.99
    assert within_label_nmi(a_lab, b_lab, labels, k) == 0.0
    # identical codes agree within every label too
    assert abs(within_label_nmi(a, a, labels, k) - 1.0) < 1e-9
    # labels with too few rows are left out
    few = labels.copy()
    few[:10] = 9
    assert abs(within_label_nmi(a, a, few, k, min_rows=50) - 1.0) < 1e-9
    # the within-label shuffle keeps each label's multiset of b
    s = shuffle_within(a, labels, rng)
    for lab in range(4):
        m = labels == lab
        assert sorted(s[m].tolist()) == sorted(a[m].tolist())
    assert within_label_nmi(a, s, labels, k) < 0.01


def test_next_with_residual_layers_is_refused():
    import train_next
    cfg = train_next.parser().parse_args([])
    assert all(getattr(cfg, k) == v for k, v in train_next.RECIPE.items())
    assert (cfg.l, cfg.h, cfg.temperature, cfg.sd_k) == (1, 380, 0.00015, 0.3)
    train_next.check(cfg)
    for argv in (["--l", "2"], ["--l", "2", "--forward", "resid"]):
        with pytest.raises(SystemExit):
            train_next.check(train_next.parser().parse_args(argv))
    train_next.check(train_next.parser().parse_args(
        ["--l", "2", "--forward", "labels"]))
    train_next.check(train_next.parser().parse_args(
        ["--l", "2", "--objective", "self"]))
