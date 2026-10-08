# assignmap.py's label, sampling and ordering helpers and its SAE block,
# pinned on hand-checkable cases, and pages drawn from synthetic
# assignments and codes.
import numpy as np
import pytest

import assignmap


@pytest.mark.parametrize("tok,cls", [
    (" the", "word"), (" The", "Word"), ("ing", "cont"), (" 42", "digit"),
    ("7", "digit"), (",", "punct"), (" (", "punct"), ("\n", "newline"),
    ("\n\n", "newline"), (" ", "space"), ("", "space")])
def test_token_class(tok, cls):
    assert assignmap.token_class(tok) == cls


def test_pos_bucket():
    b = assignmap.pos_bucket(np.array([1, 2, 3, 4, 15, 16, 63, 64, 127]))
    assert b.tolist() == ["p0:1", "p1:2-3", "p1:2-3", "p2:4-15", "p2:4-15",
                          "p3:16-63", "p3:16-63", "p4:64+", "p4:64+"]


def test_pick_rows_most_frequent_and_capped():
    labels = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 3 + ["d"])
    rows, shown = assignmap.pick_rows(labels, None, 3, 4, 0)
    assert shown == ["a", "b", "c"]  # ties broken by name
    assert [len(r) for r in rows] == [4, 3, 3]
    for r, v in zip(rows, shown):
        assert (labels[r] == v).all() and (np.diff(r) > 0).all()


def test_pick_rows_explicit_order_and_missing():
    labels = np.array(["x", "y", "y"])
    _, shown = assignmap.pick_rows(labels, ["y", "x"], 9, 9, 0)
    assert shown == ["y", "x"]
    with pytest.raises(AssertionError):
        assignmap.pick_rows(labels, ["z"], 9, 9, 0)


def test_entry_and_row_order():
    # one head, k=3: entry 2 most used, then 0, entry 1 dead
    P = np.array([[[0.0, 0.0, 1.0]],
                  [[1.0, 0.0, 0.0]],
                  [[0.0, 0.0, 1.0]]])
    order = assignmap.entry_order(P)
    assert order.tolist() == [[2, 0, 1]]
    # rows on the most-used entry come first
    assert assignmap.row_order(P, order).tolist() == [0, 2, 1]


def test_row_order_secondary_head():
    # head 0 ties all rows; head 1 decides
    P = np.zeros((3, 2, 2))
    P[:, 0, 0] = 1
    P[[0, 2], 1, 1] = 1
    P[1, 1, 0] = 1
    order = assignmap.entry_order(P)  # head 1: entry 1 used more
    assert assignmap.row_order(P, order).tolist() == [0, 2, 1]


def test_eff_entries():
    k = 4
    uniform = np.full((10, 1, k), 1 / k)
    collapsed = np.zeros((10, 1, k))
    collapsed[:, 0, 2] = 1
    assert assignmap.eff_entries(uniform)[0] == pytest.approx(k)
    assert assignmap.eff_entries(collapsed)[0] == pytest.approx(1.0)


def test_row_eff_separates_undecided_from_collapsed():
    k = 4
    uniform = np.full((10, 1, k), 1 / k)  # undecided: every row uniform
    spread = np.zeros((8, 1, k))  # healthy: one-hot rows, all entries used
    spread[np.arange(8), 0, np.arange(8) % k] = 1
    assert assignmap.row_eff(uniform)[0] == pytest.approx(k)
    assert assignmap.eff_entries(uniform)[0] == pytest.approx(k)
    assert assignmap.row_eff(spread)[0] == pytest.approx(1.0)
    assert assignmap.eff_entries(spread)[0] == pytest.approx(k)


def test_temperature_at():
    hy = {"temperature": 1.0, "temperature_end": 0.01, "anneal_steps": 100}
    assert assignmap.temperature_at(hy, 0) == pytest.approx(1.0)
    assert assignmap.temperature_at(hy, 50) == pytest.approx(0.1)
    assert assignmap.temperature_at(hy, 500) == pytest.approx(0.01)
    flat = {"temperature": 0.3, "temperature_end": None, "anneal_steps": 0}
    assert assignmap.temperature_at(flat, 10) == 0.3


def test_read_hyper_takes_last(tmp_path):
    (tmp_path / "log.jsonl").write_text(
        '{"type": "env_config", "hyper": {"temperature": 1.0}}\n'
        '{"type": "step"}\n'
        '{"type": "env_config", "hyper": {"temperature": 2.0}}\n')
    assert assignmap.read_hyper(tmp_path)["temperature"] == 2.0
    assert assignmap.read_hyper(tmp_path / "missing") is None


def test_draw_page(tmp_path):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(40, 3, 5))
    P = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    path = tmp_path / "page.png"
    assignmap.draw_page(P, [0, 2], [25, 15], ["a", "b"], "t", path, 0.5)
    assert path.stat().st_size > 0


def test_draw_all_pages(tmp_path):
    from types import SimpleNamespace
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(4), size=(30, 2, 3))  # (n, l, h, k)
    cfg = SimpleNamespace(heads=None, max_heads=2, ckpt="run", by="lang",
                          vmax=1.0)
    assignmap.draw_all(cfg, P, [10, 20], ["a", "b"], 7, tmp_path)
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "assign_l0_p0.png", "assign_l0_p1.png",
        "assign_l1_p0.png", "assign_l1_p1.png"]


def test_row_eff_skips_silent_rows():
    P = np.zeros((4, 1, 3))
    P[:2, 0] = 1 / 3  # two uniform rows; the unit is silent on the others
    assert assignmap.row_eff(P)[0] == pytest.approx(3.0)
    assert assignmap.eff_entries(P)[0] == pytest.approx(3.0)
    assert assignmap.eff_entries(np.zeros((3, 1, 4)))[0] == pytest.approx(1.0)


def test_unit_shares_groups_flat_and_silent():
    z = np.array([[2.0, 0.0, 1.0, 3.0],
                  [0.0, 0.0, 0.0, 5.0]])
    S = assignmap.unit_shares(z, 2)  # two groups of two
    np.testing.assert_allclose(S[0], [[1.0, 0.0], [0.25, 0.75]])
    np.testing.assert_allclose(S[1], [[0.0, 0.0], [0.0, 1.0]])  # g0 silent
    flat = assignmap.unit_shares(z, 0)
    assert flat.shape == (2, 1, 4)
    np.testing.assert_allclose(flat[0, 0], [2 / 6, 0, 1 / 6, 3 / 6])


def test_sae_block_flat_code_cut_to_most_used():
    z = np.zeros((4, 6))
    z[:, 3] = 1.0  # every row
    z[:2, 1] = 1.0  # half the rows
    z[3, 5] = 2.0
    blk = assignmap.sae_block(z, 0, 2, "t", 0.5)
    assert blk.names == ["code"] and blk.width == 6
    assert blk.Q.shape == (4, 1, 2)
    # latent 3 (mean share 7/12) then latent 1 (1/4); latent 5 is cut
    np.testing.assert_allclose(blk.Q[:, 0, 0], [0.5, 0.5, 1.0, 1 / 3])
    np.testing.assert_allclose(blk.Q[:, 0, 1], [0.5, 0.5, 0.0, 0.0])
    assert blk.mass[0] == pytest.approx(7 / 12 + 1 / 4)
    assert blk.fire[0] == 1.0


def test_sae_block_groups_in_index_order():
    z = np.zeros((3, 12))  # four groups of three
    z[:, 2] = 1.0  # group 0: its last latent on every row
    z[0, 4] = 1.0  # group 1 fires on row 0 only
    blk = assignmap.sae_block(z, 4, 7, "t", 1.0)  # 7 columns fit 2 groups
    assert blk.names == ["g0", "g1"] and blk.Q.shape == (3, 2, 3)
    assert blk.Q[:, 0, 0].tolist() == [1.0, 1.0, 1.0]  # most used first
    np.testing.assert_allclose(blk.fire, [1.0, 1 / 3])
    np.testing.assert_allclose(blk.mass, [1.0, 1.0])
    np.testing.assert_allclose(blk.reff, [1.0, 1.0])  # one-hot where firing


def test_sae_title():
    t = assignmap.sae_title
    assert t("r", {"m": 5120, "topk": 32}) == "SAE r (m=5120, topk 32)"
    assert t("r", {"m": 5120, "topk": 0, "groups": 160, "group_fn": "top1",
                   "prefixes": 5}) == \
        "SAE r (m=5120, 160 groups of 32, top1, 5 prefixes)"
    assert t("r", {"m": 64, "topk": 0, "l1": 3e-5}) == "SAE r (m=64, L1 3e-05)"
    assert t("r", {"m": 64, "topk": 0, "enc": "gated"}) == \
        "SAE r (m=64, gated)"


def test_sae_from_default_vmax():
    from types import SimpleNamespace
    cfg = SimpleNamespace(vmax=0.8, sae_cols=8, sae_vmax=None)
    z = np.zeros((5, 16))
    z[:, :4] = 1.0  # L0 = 4
    flat = assignmap.sae_from(cfg, z, {"m": 16, "topk": 4}, "r")
    assert flat.vmax == pytest.approx(0.5)  # 2 / L0
    grouped = assignmap.sae_from(cfg, z, {"m": 16, "topk": 0, "groups": 4,
                                          "group_fn": "top1"}, "r")
    assert grouped.vmax == 0.8 and grouped.names == ["g0", "g1"]


def test_draw_all_reorders_sae_rows_with_the_page(tmp_path, monkeypatch):
    from types import SimpleNamespace
    n = 12
    P = np.zeros((n, 1, 1, 3))
    P[np.arange(n), 0, 0, np.arange(n) % 3] = 1  # the page regroups rows
    z = np.eye(n)  # latent i fires on row i alone, so it names the row
    seen = {}

    def fake_page(Pp, heads, slices, names, title, path, vmax, sae):
        seen["P"], seen["sae"] = Pp, sae
    monkeypatch.setattr(assignmap, "draw_page", fake_page)
    cfg = SimpleNamespace(heads=None, max_heads=1, ckpt="run", by="lang",
                          vmax=1.0)
    blk = assignmap.sae_block(z, 0, n, "t", 1.0)
    assignmap.draw_all(cfg, P, [n], ["a"], 0, tmp_path, blk)
    shown = seen["sae"].Q[:, 0].argmax(-1)  # the cache row of each SAE row
    assert (shown != np.arange(n)).any()  # the page did reorder
    np.testing.assert_array_equal(seen["P"], P[shown, 0])


@pytest.mark.parametrize("groups", [0, 4])
def test_draw_page_with_sae_block(tmp_path, groups):
    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(5), size=(40, 3))
    z = np.maximum(rng.normal(size=(40, 24)), 0)
    blk = assignmap.sae_block(z, groups, 12, "SAE t", 0.5)
    path = tmp_path / "page.png"
    assignmap.draw_page(P, [0, 2], [25, 15], ["a", "b"], "t", path, 0.5, blk)
    assert path.stat().st_size > 0
