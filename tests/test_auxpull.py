# experiments/ste-arm/auxpull.py: which statistics pull on the model, and
# on the parameters that write one coordinate. The tests pin that a
# parameter group is a key in the tree path (a layer's classifier is not in
# its `dict`), that the configured weights are the run's last recorded
# ones, and that the per-coordinate shares are projections on the net
# weighted gradient, which sum to 1.
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "ste-arm"))
from auxpull import TERMS, coord_pull, in_group, run_weights  # noqa: E402


def test_group_is_a_key_not_a_substring():
    tree = {"params": {"dictencs_0": {"classifier": {"w": jnp.ones(2)},
                                      "dict": {"weights": jnp.ones(3)}},
                       "decoder": {"weights": jnp.ones(4)}}}
    paths = {jax.tree_util.keystr(p): p for p, _ in jax.tree_util.tree_leaves_with_path(tree)}
    dict_leaves = [s for s, p in paths.items() if in_group(p, "dict")]
    assert dict_leaves == ["['params']['dictencs_0']['dict']['weights']"]
    assert [s for s, p in paths.items() if in_group(p, "classifier")] == \
        ["['params']['dictencs_0']['classifier']['w']"]
    assert [s for s, p in paths.items() if in_group(p, "decoder")] == \
        ["['params']['decoder']['weights']"]


def test_run_weights_reads_the_last_hyper_record(tmp_path):
    assert run_weights(str(tmp_path)) is None
    lines = [{"type": "env", "hyper": {"s_Hm": 1e-4, "s_bcossim": 1e-5}},
             {"type": "metric"},
             {"type": "env", "hyper": {"s_Hm": 2e-6, "s_bcossim": 1e-5}}]
    (tmp_path / "log.jsonl").write_text("\n".join(map(json.dumps, lines)))
    w = run_weights(str(tmp_path))
    assert w["s_Hm"] == 2e-6 and w["s_bcossim"] == 1e-5
    assert set(w) == {name for _, name in TERMS} and w["s_support"] == 0.0


def test_coord_pull_shares_sum_to_one():
    rng = np.random.default_rng(0)
    r, d, c = 50, 6, 4
    g_mse = rng.normal(size=(r, d))
    g_mse[:, c] = 0  # a coordinate the MSE weights do not reach
    grads = {"a": rng.normal(size=(r, d)), "b": rng.normal(size=(r, d)),
             "z": np.zeros((r, d))}
    grads["a"][:, c] *= 10  # "a" pulls on c far harder than elsewhere
    w = {"a": 1e-6, "b": 1e-5, "z": 1.0}
    rows, net = coord_pull(grads, g_mse, w, c)
    assert np.isclose(sum(v["net_share"] for v in rows.values()), 1.0)
    assert rows["z"]["net_share"] == 0.0 and rows["z"]["gnorm"] == 0.0
    assert rows["a"]["over_median"] > 5
    ref = np.median(np.linalg.norm(g_mse, axis=0))
    assert np.isclose(rows["b"]["pull_vs_mse"], 1e-5 * np.linalg.norm(grads["b"][:, c]) / ref)
    v = 1e-6 * grads["a"][:, c] + 1e-5 * grads["b"][:, c]
    assert np.isclose(net["net_gnorm"], np.linalg.norm(v))
    assert np.isclose(net["net_rms"], np.linalg.norm(v) / np.sqrt(r))
    assert net["mse_gnorm"] == 0.0
