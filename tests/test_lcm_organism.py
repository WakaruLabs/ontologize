# experiments/lcm-organism.
# - f16cache.py: every row of the float16 copy is the source's rounded,
#   across chunks that do not divide the cache; the sidecars are reachable
#   under the copy's stem; meta.json records the dtype and the conversion;
#   values beyond float16's range and a copy that would fill the disk are
#   refused.
# - lcm.py: a state at n does not change with rows after n, with padding,
#   or with the other documents of a batch; the scaler round-trips; a
#   prediction is scored only against the next segment.
# - train_lcm.py, on a cache whose next row depends on the row before last
#   (two interleaved rotating sequences per document): the LCM beats the
#   ridge from x_t that the context makes useless; a SIGTERM saves the step
#   and a rerun resumes from it.
# - harvest_lcm.py: the prefix keeps its row numbers, the tail follows it,
#   each row is the state at that row, and the sidecars stay aligned row for
#   row with the cache.
# - stateagree.py: a rotation reads as identity under every statistic, an
#   invertible linear map only under the linear ones, independent
#   representations under none.
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

LCM = Path(__file__).parents[1] / "experiments" / "lcm-organism"
sys.path.insert(0, str(LCM))
from f16cache import copy_f16, rounding_cost, to_f16  # noqa: E402
from f16cache import main  # noqa: E402


def cache(tmp_path, n=1000, d=8):
    rng = np.random.default_rng(0)
    x = (0.1 * rng.standard_normal((n, d))).astype(np.float32)
    x[3, 0], x[4, 1] = 1e-6, 1e-9  # a float16 subnormal, and a value that rounds to zero
    c = tmp_path / "c.npy"
    np.save(c, x)
    np.save(tmp_path / "c.index.npy", np.arange(3 * n).reshape(n, 3))
    np.save(tmp_path / "c.mse_weights.npy", np.ones(d, np.float32))
    (tmp_path / "c.meta.json").write_text(json.dumps({"dtype": "float32", "holdout": 100}))
    return c, x


def test_the_copy_is_the_cache_rounded_with_its_sidecars(tmp_path):
    c, x = cache(tmp_path)
    main(["--cache", str(c), "--chunk", "128", "--sample", "50", "--min-free-gb", "0"])
    out = tmp_path / "c_f16.npy"
    y = np.load(out)
    assert y.dtype == np.float16 and np.array_equal(y, x.astype(np.float16))
    assert not (tmp_path / "c_f16.npy.partial").exists()
    for side in ("index.npy", "mse_weights.npy"):
        assert os.path.samefile(tmp_path / f"c.{side}", tmp_path / f"c_f16.{side}")
    meta = json.loads((tmp_path / "c_f16.meta.json").read_text())
    assert meta["dtype"] == "float16" and meta["holdout"] == 100
    conv = meta["float16"]["conversion"]
    assert conv["subnormal"] >= 1 and conv["to_zero"] >= 1 and conv["nonfinite"] == 0
    assert meta["float16"]["rounding"]["tail"]["rows"] == 100
    assert not (tmp_path / "c_f16.c_f16.meta.json").exists()


def test_bulk_puts_the_array_elsewhere_and_keeps_the_rest_beside_out(tmp_path):
    c, x = cache(tmp_path)
    bulk = tmp_path / "other_disk"
    main(["--cache", str(c), "--bulk", str(bulk), "--sample", "50", "--min-free-gb", "0"])
    out = tmp_path / "c_f16.npy"
    assert out.is_symlink() and out.resolve() == (bulk / "c_f16.npy").resolve()
    assert np.array_equal(np.load(out), x.astype(np.float16))
    assert os.path.samefile(tmp_path / "c.index.npy", tmp_path / "c_f16.index.npy")
    assert not (tmp_path / "c_f16.index.npy").is_symlink()
    meta = json.loads((tmp_path / "c_f16.meta.json").read_text())
    assert meta["float16"]["array"] == str((bulk / "c_f16.npy").resolve())
    assert sorted(p.name for p in bulk.iterdir()) == ["c_f16.npy"]
    with pytest.raises(SystemExit, match="exists"):
        main(["--cache", str(c), "--bulk", str(bulk), "--min-free-gb", "0"])


def test_syncing_mid_copy_keeps_every_row(tmp_path):
    c, x = cache(tmp_path, n=777)
    out = tmp_path / "o.npy"
    stats = copy_f16(c, out, chunk=64, sync_every=2, log=lambda *_: None)
    y = np.load(tmp_path / "o.npy.partial")
    assert np.array_equal(y, x.astype(np.float16)) and stats["n"] == 777


def test_values_beyond_float16_are_refused():
    x = np.array([[1.0, 7e4]], np.float32)
    with pytest.raises(ValueError, match="beyond"):
        to_f16(x)
    y, s = to_f16(np.array([[np.nan, 1.0]], np.float32))
    assert s["nonfinite"] == 1 and np.isnan(y[0, 0])


def test_a_copy_that_would_fill_the_disk_is_refused(tmp_path):
    c, _ = cache(tmp_path)
    with pytest.raises(SystemExit, match="free"):
        main(["--cache", str(c), "--min-free-gb", "1e9"])
    assert not list(tmp_path.glob("c_f16*"))


def test_rounding_costs_little_and_counts_each_weighting():
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal((500, 16))).astype(np.float32)
    r = rounding_cost(x, {"flat": np.ones(16), "half": np.r_[np.ones(8), np.zeros(8)]})
    assert 0 < r["rel_err_median"] < 1e-3
    assert set(r["fvu_w"]) == {"flat", "half"} and 0 < r["fvu_w"]["flat"] < 1e-6


# ---------------------------------------------------------------- the LCM

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import harvest_lcm  # noqa: E402
import lcm  # noqa: E402
import stateagree  # noqa: E402
import train_lcm  # noqa: E402

TINY = dict(d_sonar=8, d_model=16, layers=2, heads=2, d_ff=32, dropout=0.0)
TINY_FLAGS = ["--layers", "2", "--d-model", "32", "--heads", "4", "--d-ff", "64",
              "--dropout", "0", "--t", "16"]


def tiny(rng=0, b=3, t=6):
    m = lcm.BaseLCM(**TINY)
    r = np.random.default_rng(rng)
    x = r.standard_normal((b, t, 8)).astype(np.float32)
    pos = np.tile(np.arange(t, dtype=np.int32), (b, 1))
    v = m.init(jax.random.PRNGKey(0), x, pos, np.ones((b, t), bool))
    return m, v, x, pos


def test_a_state_never_sees_later_rows():
    m, v, x, pos = tiny()
    mask = np.ones(x.shape[:2], bool)
    y = x.copy()
    y[:, 4:] += 5.0
    for block in (1, 2):
        sx = m.apply(v, x, pos, mask, block, method=lcm.BaseLCM.states)
        sy = m.apply(v, y, pos, mask, block, method=lcm.BaseLCM.states)
        np.testing.assert_allclose(sx[:, :4], sy[:, :4], atol=1e-6)
        assert not np.allclose(sx[:, 4:], sy[:, 4:])
    px, py = m.apply(v, x, pos, mask), m.apply(v, y, pos, mask)
    np.testing.assert_allclose(px[:, :4], py[:, :4], atol=1e-6)


def test_padding_and_other_documents_do_not_leak():
    m, v, x, pos = tiny(b=2)
    mask = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], bool)
    junk = x.copy()
    junk[0, 3:] = 100.0                      # what sits in the padding
    junk[1] = -x[1]                          # and the other document
    alone = m.apply(v, x[:1], pos[:1], mask[:1], 2, method=lcm.BaseLCM.states)
    both = m.apply(v, junk, pos, mask, 2, method=lcm.BaseLCM.states)
    np.testing.assert_allclose(both[0, :3], alone[0, :3], atol=1e-6)
    empty = m.apply(v, x, pos, np.zeros_like(mask), 2, method=lcm.BaseLCM.states)
    assert np.isfinite(np.asarray(empty)).all()


def test_the_scaler_round_trips_and_floors_a_constant_dimension():
    r = np.random.default_rng(1)
    X = r.standard_normal((2000, 8)) * np.arange(1, 9)
    X[:, 3] = 0.25
    med, iqr = lcm.fit_scaler(X)
    np.testing.assert_allclose(med[:3], np.median(X[:, :3], 0), rtol=1e-5)
    assert iqr[3] > 0 and np.isfinite(iqr).all()
    m, v, x, pos = tiny()
    v = {"params": v["params"], "scaler": {"median": jnp.asarray(med), "iqr": jnp.asarray(iqr)}}
    back = m.apply(v, m.apply(v, x, method=lcm.BaseLCM.normalize), method=lcm.BaseLCM.denormalize)
    np.testing.assert_allclose(back, x, rtol=1e-5, atol=1e-5)


def test_the_mlp_is_swiglu_from_the_package_layers_in_float32():
    m, v, x, pos = tiny()
    p = v["params"]["blocks_0"]["mlp"]
    W, D = np.asarray(p["up"]["weight"]), np.asarray(p["down"]["weights"])
    h = np.random.default_rng(4).standard_normal((3, 6, 16)).astype(np.float32)
    y = np.asarray(lcm.GLU(16, 32).apply({"params": p}, h))
    a, b = h @ W[0].T, h @ W[1].T
    np.testing.assert_allclose(y, (a / (1 + np.exp(-a)) * b) @ D.T, rtol=1e-5, atol=1e-6)
    assert {leaf.dtype for leaf in jax.tree.leaves(v["params"])} == {np.dtype("float32")}
    full = jax.eval_shape(lambda: lcm.BaseLCM().init(
        jax.random.PRNGKey(0), jnp.zeros((1, 16, 1024)), jnp.zeros((1, 16), jnp.int32),
        jnp.ones((1, 16), bool)))
    assert sum(int(np.prod(a.shape)) for a in jax.tree.leaves(full["params"])) == 104_877_056


def test_a_bilinear_mlp_loads_into_bilinear_for_its_analysis():
    from ontologize.layers.nlinear import Bilinear
    m = lcm.BaseLCM(**{**TINY, "gate": "none"})
    r = np.random.default_rng(3)
    x = r.standard_normal((2, 5, 8)).astype(np.float32)
    v = m.init(jax.random.PRNGKey(1), x, np.tile(np.arange(5, dtype=np.int32), (2, 1)),
               np.ones((2, 5), bool))
    up = v["params"]["blocks_1"]["mlp"]["up"]
    h = r.standard_normal((7, 16)).astype(np.float32)
    layer = Bilinear(d_in=16, d_out=32, n=2, biased=False, **lcm.F32)
    out = np.asarray(layer.apply({"params": up}, h))
    W = np.asarray(up["weight"])
    np.testing.assert_allclose(out, (h @ W[0].T) * (h @ W[1].T), rtol=1e-5, atol=1e-6)
    B = np.asarray(layer.apply({"params": up}, method=Bilinear.bilinearTensor))
    np.testing.assert_allclose(np.einsum("nd,ked,ne->nk", h, B, h), out, rtol=1e-4, atol=1e-5)
    assert train_lcm.tag(train_lcm.parser().parse_args(["--gate", "none"])) == "lcm_l8_d1024_bilinear_s42"
    assert train_lcm.tag(train_lcm.parser().parse_args([])) == "lcm_l8_d1024_s42"


def test_only_the_next_segment_is_a_target():
    index = np.array([[0, 0, 1], [0, 1, 2], [0, 2, 4], [0, 3, 5],   # a gap after row 1
                      [1, 0, 1], [1, 1, 2]], np.int32)
    arr = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    x, pos, mask, adj = lcm.gather(arr, index, np.array([0, 4, 0]), np.array([4, 2, 0]), 5)
    assert mask.sum(1).tolist() == [4, 2, 0] and pos[0].tolist() == [0, 1, 2, 3, 0]
    assert adj.tolist() == [[True, False, True, False], [True, False, False, False],
                            [False] * 4]
    t_adj = lcm.next_targets(mask, adj, True)
    t_all = lcm.next_targets(mask, adj, False)
    assert t_adj[0].tolist() == [True, False, True, False]
    assert t_all[0].tolist() == [True, True, True, False]
    np.testing.assert_array_equal(x[1, :2], arr[4:6])


def synth(path: Path, n_docs=320, d=8, seed=0):
    """A sentence cache whose next row is a fixed rotation of the row before
    last: each document interleaves two independent rotating sequences, so
    x_t alone says nothing about x_{t+1}. Some documents are empty, and
    some neighbouring rows skip a segment."""
    r = np.random.default_rng(seed)
    R = np.linalg.qr(r.standard_normal((d, d)))[0]
    rows, index, docstart = [], [], []
    for k in range(n_docs):
        docstart.append(len(rows))
        L = 0 if k % 40 == 7 else int(r.integers(2, 17))
        u, v, seg = r.standard_normal(d), r.standard_normal(d), 0
        for p in range(L):
            if p % 2 == 0:
                rows.append(u)
                u = R @ u
            else:
                rows.append(v)
                v = R @ v
            seg += 2 if r.random() < 0.1 else 1
            index.append((k, p, seg))
    c = path / "s.npy"
    np.save(c, 0.1 * np.asarray(rows, np.float32))
    np.save(path / "s.index.npy", np.asarray(index, np.int32))
    np.save(path / "s.docstart.npy", np.asarray(docstart, np.int64))
    n = len(rows)
    np.save(path / "s.langs.npy", np.array(["en", "fr"] * (n // 2) + ["en"] * (n % 2)))
    np.save(path / "s.kind.npy", np.array(["sentence"] * n))
    np.save(path / "w.npy", np.ones(d, np.float32))
    return c


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lcm")
    c = synth(tmp)
    out = tmp / "run"
    flags = ["--cache", str(c), "--out", str(out), *TINY_FLAGS, "--b", "32",
             "--tail-rows", "400", "--holdout", "150", "--mse-weights", str(tmp / "w.npy"),
             "--fit-docs", "200", "--scaler-rows", "1000", "--lr", "3e-3", "--warmup", "20",
             "--save-every", "100", "--keep-every", "0", "--eval-every", "0",
             "--log-every", "100"]
    assert train_lcm.main(flags + ["--max-steps", "400"]) == 0
    return tmp, c, out, flags


def records(out: Path):
    return [json.loads(ln) for ln in (out / "log.jsonl").read_text().splitlines()]


def test_the_lcm_learns_what_x_t_cannot_predict(trained):
    _, _, out, _ = trained
    log = records(out)
    base = next(r["baselines"] for r in log if "baselines" in r)
    evals = [r["eval"]["fvu_next"] for r in log if "eval" in r]
    assert base["ridge x_t"] > 0.9                   # x_t says nothing of x_{t+1}
    assert evals[-1] < 0.5 * base["ridge x_t"] and evals[-1] < evals[0]
    spec = json.loads((out / "spec.json").read_text())
    assert spec["model"]["d_model"] == 32 and spec["total_steps"] == 400


def test_a_sigterm_saves_the_step_and_a_rerun_resumes(trained):
    tmp, c, out, flags = trained
    run2 = tmp / "run2"
    args = [a if a != str(out) else str(run2) for a in flags] + ["--max-steps", "1000000",
                                                                  "--no-baselines", "--log-every", "10"]
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    proc = subprocess.Popen([sys.executable, str(LCM / "train_lcm.py"), *args], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 120
    while time.time() < deadline and not (
            (run2 / "log.jsonl").exists() and '"loss"' in (run2 / "log.jsonl").read_text()):
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(60) == train_lcm.STOPPED
    stopped = [r for r in records(run2) if r.get("event") == "stopped"][-1]["step"]
    assert train_lcm.manager(run2, train_lcm.parser().parse_args(args)).latest_step() == stopped
    assert train_lcm.main(args[:-5] + ["--max-steps", str(stopped + 5), "--no-baselines"]) == 0
    starts = [r["at"] for r in records(run2) if r.get("event") == "start"]
    assert starts == [0, stopped]


def test_the_harvest_keeps_row_numbers_and_aligned_sidecars(trained):
    tmp, c, out, _ = trained
    dst = tmp / "h" / "s_b1.npy"
    harvest_lcm.main(["--run", str(out), "--block", "1", "--prefix-rows", "300",
                      "--out", str(dst), "--b", "16", "--match-cache", str(c),
                      "--match-weights", str(tmp / "w.npy")])
    meta = json.loads(dst.with_name("s_b1.meta.json").read_text())
    P, T = meta["rows"]["prefix"][1], meta["rows"]["tail"][0]
    n = len(np.load(c))
    ds = np.load(tmp / "s.docstart.npy")
    assert P >= 300 and P in ds and T == json.loads((out / "spec.json").read_text())["split"]["train_hi"]
    src = harvest_lcm.harvest_rows(P, T, n)
    h = np.load(dst)
    assert h.dtype == np.float16 and len(h) == len(src) == meta["n"]
    for suf in (".index.npy", ".langs.npy", ".kind.npy"):
        np.testing.assert_array_equal(np.load(dst.with_name("s_b1" + suf)),
                                      np.load(tmp / ("s" + suf))[src])
    hd = np.load(dst.with_name("s_b1.docstart.npy"))
    doc = np.load(dst.with_name("s_b1.index.npy"))[:, 0]
    assert (doc[hd] != np.r_[-1, doc[hd[1:] - 1]]).all() and hd[0] == 0
    assert meta["holdout"]["harvest"] == meta["holdout"]["cache"] < len(h) - P
    # each row is the model's state at that row, given its document so far
    model, v, _, _ = train_lcm.load_run(out)
    arr, index = np.load(c), np.load(tmp / "s.index.npy")
    for row in (0, 5, P - 1, T, n - 1):
        # an empty document shares its start with the next one, so the last
        # start at or before the row is the row's own document
        lo = int(ds[np.searchsorted(ds, row, side="right") - 1])
        x, pos, mask, _ = lcm.gather(arr, index, np.array([lo]), np.array([row - lo + 1]), 16)
        want = model.apply(v, x, pos, mask, 1, method=lcm.BaseLCM.states)[0, row - lo]
        got = h[row if row < P else P + row - T]
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)
    w = np.load(dst.with_name("s_b1.mse_weights.npy"))
    assert abs(w.mean() - 1) < 1e-5 and dst.with_name("s_b1.mse_weights_matched.npy").exists()
    assert not dst.with_name("s_b1.npy.partial").exists()


def test_a_tail_only_harvest_has_no_weights(trained):
    tmp, c, out, _ = trained
    dst = tmp / "h" / "tail_b2.npy"
    harvest_lcm.main(["--run", str(out), "--block", "2", "--prefix-rows", "0", "--out", str(dst)])
    meta = json.loads(dst.with_name("tail_b2.meta.json").read_text())
    assert meta["rows"]["prefix"] == [0, 0] and meta["n"] == len(np.load(c)) - meta["rows"]["tail"][0]
    assert meta["mse_weights"] is None and not dst.with_name("tail_b2.mse_weights.npy").exists()


def test_gate_b_tells_a_contextless_state_from_one_that_reads_the_context(trained):
    import stateinput
    tmp, c, out, _ = trained
    model, v, _, spec = train_lcm.load_run(out)
    arr, index, ds = np.load(c), np.load(tmp / "s.index.npy"), np.load(tmp / "s.docstart.npy")
    sp = spec["split"]
    fit = lcm.doc_spans(ds, sp["n"], 0, sp["train_hi"])
    test = lcm.doc_spans(ds, sp["n"], sp["eval_lo"], sp["n"])
    trained_r = stateinput.measure(stateinput.States(model, v, [1, 2]), arr, index, fit, test,
                                   32, 16, 1e-3)
    # with every attention output zeroed, a state is a function of its own row
    p = jax.tree.map(np.array, v["params"])
    for k in p:
        if k.startswith("blocks_"):
            p[k]["attn"]["out"]["kernel"][:] = 0
    blind = stateinput.measure(stateinput.States(model, {"params": p, "scaler": v["scaler"]}, [1, 2]),
                               arr, index, fit, test, 32, 16, 1e-3)
    for k in (1, 2):
        assert blind[k]["r2_alone"] > 0.999
        assert trained_r[k]["rows_test"] == blind[k]["rows_test"] == sp["n"] - sp["eval_lo"]
    assert trained_r[2]["r2_alone"] < 0.99       # the trained model carries x_{n-1}


def test_agreement_tells_a_rotation_from_a_linear_map_from_nothing():
    r = np.random.default_rng(2)
    a = r.standard_normal((4000, 12)) @ np.diag(np.linspace(0.5, 3, 12))
    Q = np.linalg.qr(r.standard_normal((12, 12)))[0]
    M = r.standard_normal((12, 12)) @ np.diag(np.linspace(0.2, 5, 12))
    rot = stateagree.agree(a, a @ Q)
    lin = stateagree.agree(a, a @ M)
    ind = stateagree.agree(a, r.standard_normal((4000, 12)))
    assert rot["cka"] > 0.999 and rot["procrustes"] < 1e-6 and rot["r2_a_to_b"] > 0.99
    # the ridge's penalty shrinks the map back from a@M's small directions
    assert lin["r2_a_to_b"] > 0.99 and lin["r2_b_to_a"] > 0.95
    assert lin["procrustes_whitened"] < 1e-2 and lin["procrustes"] > 0.05 and lin["cka"] < 0.99
    assert ind["cka"] < 0.05 and ind["r2_a_to_b"] < 0.05 and ind["procrustes_whitened"] > 1.5
    assert ind["cka_shuffled"] < 0.05
