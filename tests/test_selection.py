# selection.py's noise2self partition score, settings and grid helpers,
# pinned on hand-checkable cases, and both figures drawn from synthetic
# rows.
import jax.numpy as jnp
import numpy as np
import pytest

import selection
from ontologize.fns.pwak import affinity, pwak_l2


def _clustered(n_per=40, c=4, d=16, noise=0.15, seed=0):
    """c well-separated unit-norm clusters; returns (E, labels)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(c, d))
    lab = np.repeat(np.arange(c), n_per)
    E = centres[lab] + noise * rng.normal(size=(len(lab), d))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return jnp.asarray(E, jnp.float32), lab


def test_unpartitioned_is_plain_kernel_smoother():
    E, _ = _clustered()
    tau = 0.2
    D = affinity(E, tau)
    got = selection.n2s_score(jnp.ones_like(D), D, E, (1, 2))
    # by hand: row-normalized heat kernel with a zero diagonal
    En = np.asarray(E, np.float64)
    K = np.exp((En @ En.T - 1) / tau)
    np.fill_diagonal(K, 0)
    G = K / K.sum(1, keepdims=True)
    spread = ((En - En.mean(0)) ** 2).mean()
    want = [((En - G @ En) ** 2).mean() / spread,
            ((En - G @ G @ En) ** 2).mean() / spread]
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-4)
    # and it is pwak_l2 with a single one-cell head
    one = jnp.ones((E.shape[0], 1, 1))
    assert float(got[0]) == pytest.approx(float(pwak_l2(one, E, 1, tau)),
                                          rel=1e-5)


def test_joint_matches_training_stat():
    E, lab = _clustered()
    rng = np.random.default_rng(1)
    P = jnp.asarray(rng.dirichlet(np.ones(3), size=(len(lab), 2)),
                    jnp.float32)
    r = selection.layer_scores(P, E, jnp.arange(2), jnp.zeros((0, len(lab)),
                               int), (1, 3), 0.2)
    for j, s in enumerate((1, 3)):
        assert float(r["joint"][j]) == pytest.approx(
            float(pwak_l2(P, E, s, 0.2)), rel=1e-4)
    # one head: its per-head score is the joint score of that head alone
    assert float(r["head"][1, 0]) == pytest.approx(
        float(pwak_l2(P[:, 1:2], E, 1, 0.2)), rel=1e-4)


def test_true_partition_beats_random_and_unpartitioned():
    E, lab = _clustered(noise=0.3)
    P = jnp.asarray(np.eye(4)[lab][:, None], jnp.float32)  # (n, 1, 4)
    rng = np.random.default_rng(0)
    perms = jnp.asarray(np.stack([rng.permutation(len(lab))
                                  for _ in range(5)]))
    r = selection.layer_scores(P, E, jnp.arange(1), perms, (1,), 0.5)
    true = float(r["head"][0, 0])
    assert (np.asarray(r["random"])[:, 0, 0] > true).all()
    assert float(r["unpartitioned"][0]) > true
    assert true < 1.0


def test_random_partition_keeps_cell_sizes():
    lab = np.repeat(np.arange(3), [5, 10, 15])
    P = np.eye(3)[lab]
    perm = np.random.default_rng(0).permutation(len(lab))
    np.testing.assert_array_equal(P[perm].sum(0), P.sum(0))


@pytest.mark.parametrize("select,k,n", [
    ("softmax", 32, 32), ("top4", 32, 4), ("top64", 32, 32), ("ste", 2, 1),
    ("argmax", 8, 1), ("top1", 32, 1)])
def test_select_entries(select, k, n):
    assert selection.select_entries(select, k) == n


def test_capacity_follows_pareto():
    assert selection.capacity(5, 76, 32, "ste") == (0, 1900)
    assert selection.capacity(5, 32, 32, "top1") == (0, 800)
    assert selection.capacity(5, 32, 32, "top4") == (640, 3200)
    assert selection.capacity(5, 32, 32, "softmax") == (5120, 0)


def test_head_subset():
    assert selection.head_subset(4, 32).tolist() == [0, 1, 2, 3]
    s = selection.head_subset(380, 32)
    assert len(s) == 32 and s[0] == 0 and s[-1] == 379


def test_sort_values():
    assert selection.sort_values(["16", "2", "softmax", "2"]) == \
        ["16", "2", "softmax"]
    assert selection.sort_values(["16", "2", "1e-6"]) == ["1e-6", "2", "16"]


def test_assemble_grid_missing_cells_nan():
    recs = [{"name": "a", "r": 1, "c": "x", "value": 0.5},
            {"name": "b", "r": 1, "c": "x", "value": 0.3},
            {"name": "c", "r": 2, "c": "y", "value": 0.1}]
    rv, cv, V, n, names = selection.assemble_grid(recs, "r", "c")
    assert rv == [1, 2] and cv == ["x", "y"]
    assert V[0, 0] == pytest.approx(0.4) and n[0, 0] == 2
    assert np.isnan(V[0, 1]) and np.isnan(V[1, 0]) and n[1, 0] == 0
    assert V[1, 1] == pytest.approx(0.1)
    assert names[0][0] == ["a", "b"]


def test_pareto_fvu(tmp_path):
    p = tmp_path / "pareto.csv"
    p.write_text("label,coeffs,index_bits,fvu_w\n"
                 "onto m=0 (meanp origin),0,0,1.0\n"
                 "onto hard (argmax),0,800,0.7\n"
                 "sae m5120_k32,32,394,0.6\n"
                 "onto m=32 (soft),5120,0,0.2\n")
    assert selection.pareto_fvu(p) == 0.2
    assert selection.pareto_fvu(p, "hard") == 0.7
    assert selection.pareto_fvu(tmp_path / "missing.csv") is None
    assert selection.sae_fvus(p) == [("sae m5120_k32", 32, 0.6)]


def test_sae_frontier():
    sae = [("sae a", 32, 0.6), ("sae b", 16, 0.5), ("sae c", 64, 0.55),
           ("sae d", 128, 0.1)]
    assert [r[0] for r in selection.sae_frontier(sae)] == ["sae b", "sae d"]


def test_spread_labels():
    y = np.array([0.0, 0.01, 0.02, 1.0])
    z = selection.spread_labels(y, 0.1)
    assert (np.diff(np.sort(z)) >= 0.1 - 1e-12).all()
    assert np.argsort(z).tolist() == np.argsort(y).tolist()
    assert z[3] == pytest.approx(1.0, abs=0.1)


def test_grid_csv_roundtrip_and_draw(tmp_path):
    recs = [{"name": "a", "s_Hm": 1e-6, "n_sel": 1, "value": 0.7,
             "coeffs": 0, "bits": 800, "source": "computed"},
            {"name": "b", "s_Hm": 3e-5, "n_sel": 4, "value": 0.07,
             "coeffs": 640, "bits": 3200, "source": "x.csv"}]
    selection.write_grid_csv(tmp_path / "grid.csv", recs, "s_Hm", "n_sel")
    back, rk, ck = selection.read_grid_csv(tmp_path / "grid.csv")
    assert (rk, ck) == ("s_Hm", "n_sel") and back[1]["value"] == 0.07
    path = tmp_path / "grid.png"
    selection.draw_grid(back, rk, ck, "t", path, [("sae m1_k2", 2, 0.5)])
    assert path.stat().st_size > 0


def test_curve_rows_and_draw(tmp_path):
    rows = []
    for name, x, hue in [("r1", 1, "a"), ("r2", 4, "a"), ("r3", 4, "b")]:
        sc = {L: {st: np.array([0.5 + 0.1 * L]) for st in selection.STATS}
              for L in range(2)}
        rows += selection.n2s_rows(name, x, hue, sc, [1])
    assert len(rows) == 3 * 2 * len(selection.STATS)
    path = tmp_path / "c.png"
    selection.draw_curve(rows, 1, "n_sel", "hue", "t", path)
    assert path.stat().st_size > 0
