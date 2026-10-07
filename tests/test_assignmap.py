# assignmap.py's label, sampling and ordering helpers, pinned on
# hand-checkable cases, and one page drawn from synthetic assignments.
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
