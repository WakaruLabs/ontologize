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
# - readout.py: its features are Gate B's `x_context` and never reach across
#   documents; its chunks end at documents and at the cuts; on a harvest
#   whose states are planted as an affine function of those features, it
#   reproduces the states row for row, laid out as the harvest.
# - ctxmean.py: each row is the source's plus the weighted mean of its
#   document's earlier rows (its own at a document's start), and the loss
#   weights are the source's, scaled to the source's whitened base.
# - recolour.py: a source already laid out as the target harvest is mapped
#   row for row, by one linear map with no offset.
# - axes.py: a skewed coordinate is outlying on the axes and not after a
#   rotation; one random scale per row reads the same magnitude excess on the
#   axes as along random directions, and a Gaussian none; a correlated group
#   is found on the axes in the cache's numbering and not after a rotation,
#   and the draw avoids it.
# - templates.py: a sentence is a template only when another document repeats
#   it (case, spacing and digits aside), however the file is split among
#   workers; a twin is the closest row of another document; a skeleton masks
#   names, numbers and URLs and needs words left; MinHash signatures agree
#   about as often as the sets' Jaccard index, and a shingle twin must be
#   another document's; the random control drops as many rows of each
#   language.
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
import readout  # noqa: E402
import recolour  # noqa: E402
import spectrum  # noqa: E402
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


def test_the_spectrum_reads_planted_eigenvalues_in_the_weighted_frame():
    r = np.random.default_rng(3)
    n, d = 60000, 8
    lam = np.array([8.0, 4, 2, 1, 1, 1, 1, 1])
    Q = np.linalg.qr(r.standard_normal((d, d)))[0]
    w = r.uniform(0.5, 2, d)
    lab = r.integers(0, 3, n)
    means = 3 * r.standard_normal((3, d))
    shaped = r.standard_normal((n, d)) * np.sqrt(lam) @ Q.T + means[lab]
    labels = np.array(["a", "b", "c"])[lab]
    # a label too rare for the within-language covariance, with a variance
    # that would show in it
    shaped[:5] = 100 * r.standard_normal((5, d))
    labels[:5] = "z"
    x = shaped / np.sqrt(w)
    res, lam_t, lam_w = spectrum.measure(x, w, labels, np.arange(n), 1000, [4.0])
    assert res["labels"] == 3 and res["label_rows"] == n - 5
    np.testing.assert_allclose(lam_w, lam, rtol=0.05)
    pr = lam.sum() ** 2 / (lam ** 2).sum()
    assert abs(res["within"]["participation"] - pr) < 0.05 * pr
    assert res["within"]["k_share"]["0.5"] == 2 and res["within"]["k_share"]["0.99"] == 8
    keep = slice(5, None)
    between = np.cov(means[lab][keep].T, bias=True).trace()
    total = np.cov(shaped.T, bias=True).trace()
    assert abs(res["language_share"] - (1 - (np.trace(np.cov(shaped[keep].T, bias=True))
                                             - between) / total)) < 0.02
    assert np.isclose(lam_t.sum(), total)


def test_the_spectrum_reads_a_harvest_at_the_cache_rows_it_keeps():
    m = (5, 100)
    np.testing.assert_array_equal(spectrum.file_rows(np.array([0, 4, 100, 102]), m),
                                  [0, 4, 5, 7])
    np.testing.assert_array_equal(spectrum.file_rows(np.array([3, 50]), None), [3, 50])
    with pytest.raises(SystemExit):
        spectrum.file_rows(np.array([50]), m)


def test_recolouring_gives_the_target_spectrum_with_even_coordinates():
    r = np.random.default_rng(4)
    d = 64
    rot = lambda: np.linalg.qr(r.standard_normal((d, d)))[0]  # noqa: E731
    lam_s = np.r_[np.linspace(3, 1, d - 1), 0.0]      # one direction the weights zero
    lam_t = np.geomspace(50, 0.01, d)
    U, V = rot(), rot()
    C_s, C_t = (U * lam_s) @ U.T, (V * lam_t) @ V.T
    A, info = recolour.recolour_map(C_s, C_t, seed=0)
    C_o = A @ C_s @ A.T
    got = np.linalg.eigvalsh(C_o)[::-1]
    want = np.r_[lam_t[:-1], 0.0] * info["scale"] ** 2
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9 * want.max())
    assert info["zeroed"] == 1 and np.isclose(info["unmatched_target_share"],
                                              lam_t[-1] / lam_t.sum())
    assert np.isclose(np.diag(C_o).mean(), 1.0)
    # a random rotation spreads a concentrated spectrum over every coordinate
    assert np.diag(C_o).std() < 0.5 and np.diag(C_o).min() > 0.1


def test_the_readout_features_are_gate_bs_and_stay_in_their_document():
    import stateinput
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    F = readout.context_features(x, np.array([0, 2, 5]))
    np.testing.assert_array_equal(F[:, :2], x)
    np.testing.assert_allclose(F[:, 2:4], [[0, 0], [0, 1], [0, 0], [4, 5], [5, 6], [0, 0]])
    np.testing.assert_array_equal(F[:, 4], [1, 0, 1, 0, 0, 1])
    r = np.random.default_rng(5)
    lens = np.array([3, 1, 5, 2])
    x = r.standard_normal((lens.sum(), 4)).astype(np.float32)
    t = 6
    mask = np.arange(t)[None] < lens[:, None]
    pad = np.zeros((len(lens), t, 4), np.float32)
    pad[mask] = x
    _, want = stateinput.row_features(pad, mask)
    got = readout.context_features(x, np.r_[0, np.cumsum(lens)[:-1]])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_readout_chunks_end_at_documents_and_at_the_cuts():
    ds = np.array([0, 3, 4, 9, 11, 16, 20])
    assert readout.doc_chunks(ds, [11, 16], 25, 6) == [(0, 4), (4, 9), (9, 11), (11, 16),
                                                       (16, 20), (20, 25)]
    # a document longer than a chunk is one chunk
    assert readout.doc_chunks(np.array([0, 10]), [], 12, 4) == [(0, 10), (10, 12)]


def planted_harvest(tmp_path):
    """synth's cache, and a harvest of its rows below P and from T on whose
    states are an affine map of readout.context_features: (cache, its rows,
    the harvest, P, T, the cache rows the harvest keeps, the features)."""
    c = synth(tmp_path)
    x = np.load(c)
    index = np.load(tmp_path / "s.index.npy")
    n = len(x)
    cache_first = harvest_lcm.first_rows(index[:, 0])
    P = int(cache_first[np.searchsorted(cache_first, 300)])
    T = int(cache_first[np.searchsorted(cache_first, n - 400)])
    rows = harvest_lcm.harvest_rows(P, T, n)
    r = np.random.default_rng(6)
    A, b0 = r.standard_normal((2 * x.shape[1] + 1, 12)), r.standard_normal(12)
    F = readout.context_features(x, cache_first)
    tgt = tmp_path / "h.npy"
    np.save(tgt, (F[rows] @ A + b0).astype(np.float16))
    for suf in (".index.npy", ".langs.npy", ".kind.npy"):
        np.save(tmp_path / ("h" + suf), np.load(tmp_path / ("s" + suf))[rows])
    np.save(tmp_path / "h.docstart.npy", harvest_lcm.first_rows(index[rows, 0]))
    (tmp_path / "h.meta.json").write_text(json.dumps({"rows": {"prefix": [0, P], "tail": [T, n]}}))
    return c, x, tgt, P, T, rows, F


def test_the_readout_reproduces_states_that_are_its_features_mapped(tmp_path):
    c, _, tgt, P, T, _, _ = planted_harvest(tmp_path)
    n = len(np.load(c))
    dst = tmp_path / "out" / "ro.npy"
    dst.parent.mkdir()
    readout.main(["--source", str(c), "--source-weights", str(tmp_path / "w.npy"),
                  "--target", str(tgt), "--out", str(dst), "--fit-docs", "1000",
                  "--holdout", "150", "--chunk", "50", "--alpha", "1e-9"])
    meta = json.loads((tmp_path / "out" / "ro.meta.json").read_text())
    assert meta["rows"]["prefix"] == [0, P] and meta["rows"]["tail"] == [T, n]
    assert meta["r2"]["scored_tail"] > 0.9999 and meta["r2"]["tail"] > 0.9999
    y, h = np.load(dst), np.load(tgt)
    assert y.dtype == np.float16 and y.shape == h.shape
    np.testing.assert_allclose(y.astype(np.float32), h.astype(np.float32), atol=0.02)
    for suf in (".index.npy", ".langs.npy", ".kind.npy", ".docstart.npy"):
        assert os.path.samefile(tmp_path / ("h" + suf), tmp_path / "out" / ("ro" + suf))
    w = np.load(tmp_path / "out" / "ro.mse_weights.npy")
    assert abs(w.mean() - 1) < 1e-5 and (tmp_path / "out" / "ro.mse_weights_matched.npy").exists()
    assert not (tmp_path / "out" / "ro.npy.partial").exists()
    with pytest.raises(SystemExit, match="exists"):
        readout.main(["--source", str(c), "--target", str(tgt), "--out", str(dst)])


def test_the_row_readout_sees_x_n_alone(tmp_path):
    c, x, tgt, P, T, rows, F = planted_harvest(tmp_path)
    np.testing.assert_array_equal(readout.row_features(x[:5], np.array([0, 2])), x[:5])
    (tmp_path / "out").mkdir()
    r2 = {}
    for kind in ("context", "row"):
        dst = tmp_path / "out" / f"{kind}.npy"
        readout.main(["--source", str(c), "--source-weights", str(tmp_path / "w.npy"),
                      "--target", str(tgt), "--out", str(dst), "--features", kind,
                      "--fit-docs", "1000", "--holdout", "150", "--chunk", "50", "--alpha", "1e-9"])
        meta = json.loads((tmp_path / "out" / f"{kind}.meta.json").read_text())
        r2[kind] = meta["r2"]["tail"]
        assert meta["fit"]["features"] == readout.DESCRIBED[kind]
    # the planted states read the context, which x_n alone cannot supply
    assert r2["context"] > 0.9999 and r2["row"] < 0.95
    # the row readout is the best affine map of x_n on the tail, up to the fit
    y = np.load(tmp_path / "out" / "row.npy").astype(np.float64)
    h = np.load(tgt).astype(np.float64)
    tail = slice(P, None)
    X = np.c_[x[rows][tail], np.ones(len(h) - P)]
    best = h[tail] - X @ np.linalg.lstsq(X, h[tail], rcond=None)[0]
    sst = ((h[tail] - h[tail].mean(0)) ** 2).sum()
    assert r2["row"] > 1 - (best ** 2).sum() / sst - 0.05
    assert np.isclose(r2["row"], 1 - ((y[tail] - h[tail]) ** 2).sum() / sst, atol=1e-3)


def test_the_fixed_mix_adds_the_earlier_mean_and_keeps_the_sources_weights(tmp_path):
    import ctxmean
    c, x, tgt, P, T, rows, F = planted_harvest(tmp_path)
    d = x.shape[1]
    w = np.r_[np.linspace(0.5, 2, d - 1), 0.0].astype(np.float32)   # one coordinate zeroed
    np.save(tmp_path / "w0.npy", w)
    dst = tmp_path / "out" / "mix.npy"
    dst.parent.mkdir()
    ctxmean.main(["--source", str(c), "--source-weights", str(tmp_path / "w0.npy"),
                  "--target", str(tgt), "--out", str(dst), "--weight", "0.5",
                  "--holdout", "150", "--chunk", "50"])
    y = np.load(dst)
    want = F[rows, :d] + 0.5 * F[rows, d:2 * d]
    assert y.dtype == np.float16 and y.shape == (len(rows), d)
    np.testing.assert_allclose(y.astype(np.float32), want, rtol=2e-3, atol=1e-4)
    # a document's first row is its own
    first = np.load(tmp_path / "h.docstart.npy")
    np.testing.assert_allclose(y[first].astype(np.float32), x[rows[first]], rtol=2e-3, atol=1e-4)
    side = lambda s: tmp_path / "out" / ("mix" + s)  # noqa: E731
    np.testing.assert_array_equal(np.load(side(".mse_weights.npy")), w)
    matched = np.load(side(".mse_weights_matched.npy"))
    assert matched[-1] == 0
    assert np.isclose(harvest_lcm.whitened_base(y, matched), harvest_lcm.whitened_base(x, w),
                      rtol=1e-3)
    meta = json.loads(side(".meta.json").read_text())
    assert meta["rows"]["prefix"] == [0, P] and meta["rows"]["tail"] == [T, len(x)]
    for suf in (".index.npy", ".langs.npy", ".kind.npy", ".docstart.npy"):
        assert os.path.samefile(tmp_path / ("h" + suf), side(suf))


def test_the_weighted_frame_is_the_cache_recoloured_to_itself_unrotated(tmp_path):
    import weightframe
    _, _, tgt, P, _, _, _ = planted_harvest(tmp_path)
    h = np.load(tgt).astype(np.float64)
    d = h.shape[1]
    w = harvest_lcm.inverse_variance(np.load(tgt), P)
    np.save(tmp_path / "w.npy", w.astype(np.float32))
    out = tmp_path / "out"
    out.mkdir()
    wts = ["--source-weights", str(tmp_path / "w.npy")]
    recolour.main(["--source", str(tgt), *wts, "--target", str(tgt), "--target-weights",
                   str(tmp_path / "w.npy"), "--out", str(out / "rot.npy"), "--min-rows", "10"])
    weightframe.main(["--source", str(tgt), *wts, "--scale-like", str(out / "rot.npy"),
                      "--out", str(out / "wf.npy"), "--chunk", "50"])
    c = json.loads((out / "rot.meta.json").read_text())["map"]["scale"]
    wf, rot = np.load(out / "wf.npy").astype(np.float64), np.load(out / "rot.npy").astype(np.float64)
    np.testing.assert_allclose(wf, c * np.sqrt(w) * h, rtol=2e-3, atol=1e-3)
    # the recoloured rows are these rows rotated
    M = np.linalg.lstsq(wf, rot, rcond=None)[0]
    assert ((wf @ M - rot) ** 2).sum() < 1e-4 * (rot ** 2).sum()
    np.testing.assert_allclose(M @ M.T, np.eye(d), atol=1e-2)
    # their own weights are flat, and the loss scale is the source's
    np.testing.assert_allclose(np.load(out / "wf.mse_weights.npy"), 1, rtol=1e-2)
    m = np.load(out / "wf.mse_weights_matched.npy")
    assert np.isclose(harvest_lcm.whitened_base(wf, m), harvest_lcm.whitened_base(h, w), rtol=1e-3)
    meta = json.loads((out / "wf.meta.json").read_text())
    assert meta["rows"] == json.loads((tmp_path / "h.meta.json").read_text())["rows"]
    for suf in (".index.npy", ".langs.npy", ".kind.npy", ".docstart.npy"):
        assert os.path.samefile(tmp_path / ("h" + suf), out / ("wf" + suf))
    # a coordinate the weights zero stays zero and keeps no weight; the default
    # scale brings the other coordinates' variance to about 1
    np.save(tmp_path / "w0.npy", np.r_[w[:-1], 0.0].astype(np.float32))
    weightframe.main(["--source", str(tgt), "--source-weights", str(tmp_path / "w0.npy"),
                      "--out", str(out / "wf0.npy"), "--chunk", "50"])
    wf0 = np.load(out / "wf0.npy").astype(np.float64)
    assert (wf0[:, -1] == 0).all() and np.load(out / "wf0.mse_weights.npy")[-1] == 0
    assert np.isclose(wf0[:, :-1].var(0).mean(), 1, rtol=1e-2)


def test_the_weighted_frame_rotates_all_but_the_coordinates_it_keeps(tmp_path):
    import weightframe
    _, _, tgt, P, _, _, _ = planted_harvest(tmp_path)
    d = np.load(tgt).shape[1]
    w = harvest_lcm.inverse_variance(np.load(tgt), P)
    w[1] = 0.0  # a dead coordinate stays dead and out of the rotation
    np.save(tmp_path / "w.npy", w.astype(np.float32))
    out = tmp_path / "out"
    out.mkdir()
    base = ["--source", str(tgt), "--source-weights", str(tmp_path / "w.npy"), "--chunk", "50"]
    weightframe.main([*base, "--out", str(out / "wf.npy")])
    weightframe.main([*base, "--out", str(out / "part.npy"), "--rotate-except", "0,3"])
    wf, part = (np.load(out / f).astype(np.float64) for f in ("wf.npy", "part.npy"))
    keep, dead = [0, 3], [1]
    rest = [j for j in range(d) if j not in keep + dead]
    np.testing.assert_allclose(part[:, keep], wf[:, keep], rtol=2e-3, atol=1e-3)
    assert (part[:, dead] == 0).all() and np.load(out / "part.mse_weights.npy")[1] == 0
    # the rest is an orthogonal map of the same block: norms and the loss are kept
    M = np.linalg.lstsq(wf[:, rest], part[:, rest], rcond=None)[0]
    assert ((wf[:, rest] @ M - part[:, rest]) ** 2).sum() < 1e-4 * (part ** 2).sum()
    np.testing.assert_allclose(M @ M.T, np.eye(len(rest)), atol=1e-2)
    assert np.abs(M - np.eye(len(rest))).max() > 0.1
    frame = json.loads((out / "part.meta.json").read_text())["frame"]
    assert frame["rotate_except"] == keep and frame["rotated"] == len(rest)


def test_a_kept_block_rotates_among_itself_or_takes_a_gaussian_marginal(tmp_path):
    import weightframe
    _, _, tgt, P, _, _, _ = planted_harvest(tmp_path)
    d = np.load(tgt).shape[1]
    w = harvest_lcm.inverse_variance(np.load(tgt), P)
    np.save(tmp_path / "w.npy", w.astype(np.float32))
    out = tmp_path / "out"
    out.mkdir()
    base = ["--source", str(tgt), "--source-weights", str(tmp_path / "w.npy"), "--chunk", "50",
            "--rotate-except", "0,3,5"]
    weightframe.main([*base, "--out", str(out / "keep.npy")])
    weightframe.main([*base, "--rotate-block", "0,3", "--out", str(out / "blk.npy")])
    weightframe.main([*base, "--gaussianize", "0,3", "--out", str(out / "gau.npy")])
    keep, blk, gau = (np.load(out / f).astype(np.float64) for f in ("keep.npy", "blk.npy", "gau.npy"))
    rest = [j for j in range(d) if j not in (0, 3, 5)]
    # the rest is rotated as without the option, and coordinate 5 keeps its axis
    for y in (blk, gau):
        np.testing.assert_allclose(y[:, rest], keep[:, rest], rtol=2e-3, atol=1e-3)
        np.testing.assert_allclose(y[:, 5], keep[:, 5], rtol=2e-3, atol=1e-3)
    # the block is an orthogonal map of the kept columns, not the identity
    M = np.linalg.lstsq(keep[:, [0, 3]], blk[:, [0, 3]], rcond=None)[0]
    assert ((keep[:, [0, 3]] @ M - blk[:, [0, 3]]) ** 2).sum() < 1e-4 * (blk ** 2).sum()
    np.testing.assert_allclose(M @ M.T, np.eye(2), atol=1e-2)
    assert np.abs(M - np.eye(2)).max() > 0.1
    # a Gaussianized coordinate keeps the rows' order and the prefix's mean and spread, and loses
    # its marginal's shape
    src = np.load(tgt).astype(np.float64)
    for j in (0, 3):
        a, b = keep[:, j], gau[:, j]
        assert (np.diff(b[np.argsort(src[:, j], kind="stable")]) >= 0).all()   # by the source's values
        assert abs(b[:P].mean() - a[:P].mean()) < 0.05 * a[:P].std()
        assert abs(b[:P].std() / a[:P].std() - 1) < 0.05
        z = (b[:P] - b[:P].mean()) / b[:P].std()
        assert abs((z ** 3).mean()) < 0.1 and abs((z ** 4).mean() - 3) < 0.3
    frame = json.loads((out / "gau.meta.json").read_text())["frame"]
    assert frame["gaussianize"] == [0, 3] and frame["rotate_block"] is None
    assert json.loads((out / "blk.meta.json").read_text())["frame"]["rotate_block"] == [0, 3]
    # a coordinate outside the kept ones is refused, so nothing is rotated twice
    with pytest.raises(SystemExit, match="only coordinates --rotate-except keeps"):
        weightframe.main([*base, "--rotate-block", "0,1", "--out", str(out / "bad.npy")])
    # another cache's weights, when even across the block, make the loss that cache's turned
    # within the block; uneven ones are refused
    kw = np.load(out / "keep.mse_weights.npy")
    np.save(tmp_path / "even.mse_weights.npy", np.r_[kw[:1], kw[1:3], kw[:1], kw[4:]].astype(np.float32))
    np.save(tmp_path / "even.mse_weights_matched.npy", 2 * np.load(tmp_path / "even.mse_weights.npy"))
    weightframe.main([*base, "--rotate-block", "0,3", "--weights-like", str(tmp_path / "even.npy"),
                      "--out", str(out / "blkw.npy")])
    np.testing.assert_array_equal(np.load(out / "blkw.mse_weights_matched.npy"),
                                  np.load(tmp_path / "even.mse_weights_matched.npy"))
    assert json.loads((out / "blkw.meta.json").read_text())["frame"]["weights_like"] == "even.npy"
    uneven = np.load(tmp_path / "even.mse_weights.npy")
    uneven[3] *= 2
    np.save(tmp_path / "uneven.mse_weights.npy", uneven)
    np.save(tmp_path / "uneven.mse_weights_matched.npy", uneven)
    with pytest.raises(SystemExit, match="differ across the block"):
        weightframe.main([*base, "--rotate-block", "0,3", "--weights-like", str(tmp_path / "uneven.npy"),
                          "--out", str(out / "bad2.npy")])


def test_the_gaussian_map_is_fitted_on_the_prefix_and_keeps_ties():
    import weightframe
    r = np.random.default_rng(9)
    fit = weightframe.gaussian_fit(r.exponential(size=5000))
    v = np.array([0.5, 0.5, -1.0, 1e9, 2.0])
    g = weightframe.gaussianize(v, fit)
    assert g[0] == g[1]                               # equal values stay equal
    assert g[2] == g.min() and g[3] == g.max()        # beyond the range, the end quantiles
    assert g[2] < g[0] < g[4] < g[3]                  # the order is kept
    n = len(fit[0])
    assert np.isclose(g[3], fit[1] + fit[2] * weightframe.ndtri(1 - 0.5 / n), rtol=1e-5)


def test_recolouring_to_its_own_spectrum_is_a_scaled_rotation():
    r = np.random.default_rng(8)
    d = 32
    U = np.linalg.qr(r.standard_normal((d, d)))[0]
    C = (U * np.geomspace(20, 0.05, d)) @ U.T
    A, info = recolour.recolour_map(C, C, seed=3)
    c = info["scale"]
    # only the basis changes: distances in the weighted frame are kept, up to c
    np.testing.assert_allclose(A @ A.T, c ** 2 * np.eye(d), atol=1e-9 * c ** 2)
    assert info["zeroed"] == 0 and info["unmatched_target_share"] == 0


def test_a_source_laid_out_as_the_target_is_recoloured_row_for_row(tmp_path):
    import ctxmean
    c, x, tgt, P, T, rows, F = planted_harvest(tmp_path)
    d = x.shape[1]
    # a target harvest as wide as the source, on the same rows and sidecars
    r = np.random.default_rng(7)
    t8 = tmp_path / "h8.npy"
    np.save(t8, (F[rows] @ r.standard_normal((F.shape[1], d))).astype(np.float16))
    for suf in (".index.npy", ".langs.npy", ".kind.npy", ".docstart.npy", ".meta.json"):
        os.link(tmp_path / ("h" + suf), tmp_path / ("h8" + suf))
    np.save(tmp_path / "w8.npy", np.ones(d, np.float32))
    out = tmp_path / "out"
    out.mkdir()
    ctxmean.main(["--source", str(c), "--source-weights", str(tmp_path / "w.npy"),
                  "--target", str(t8), "--out", str(out / "mix.npy"), "--holdout", "150",
                  "--chunk", "50"])
    recolour.main(["--source", str(out / "mix.npy"),
                   "--source-weights", str(out / "mix.mse_weights_matched.npy"),
                   "--target", str(t8), "--target-weights", str(tmp_path / "w8.npy"),
                   "--out", str(out / "rec.npy"), "--min-rows", "10"])
    meta = json.loads((out / "rec.meta.json").read_text())
    assert meta["source_laid_out"] and meta["rows"]["prefix"] == [0, P]
    mix, rec = np.load(out / "mix.npy").astype(np.float64), np.load(out / "rec.npy").astype(np.float64)
    assert rec.shape == mix.shape
    # row i is the map of the mix's row i: one linear map, no offset, fits every row
    M = np.linalg.lstsq(mix, rec, rcond=None)[0]
    assert ((mix @ M - rec) ** 2).sum() < 1e-4 * (rec ** 2).sum()
    for suf in (".index.npy", ".langs.npy", ".docstart.npy"):
        assert os.path.samefile(tmp_path / ("h8" + suf), out / ("rec" + suf))


# ---------------------------------------------------------------- axes.py

def planted_group(n=20000, d=64, members=(3, 10, 41), seed=0):
    """Gaussian rows whose `members` share a latent, correlating at 0.7 in pairs."""
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    z = r.standard_normal(n)
    for j in members:
        x[:, j] = np.sqrt(0.7) * z + np.sqrt(0.3) * x[:, j]
    return x


def test_a_skewed_coordinate_is_outlying_on_the_axes_and_not_after_a_rotation():
    import axes
    r = np.random.default_rng(1)
    X = r.standard_normal((20000, 32))
    X[:, 5] = r.exponential(size=20000)  # skew 2, excess kurtosis 6
    X -= X.mean(0)
    D = r.standard_normal((32, 1024))
    D /= np.linalg.norm(D, axis=0)
    a = axes.shape(X)
    assert abs(a["skew"][5] - 2) < 0.3 and abs(a["kurt"][5] - 6) < 2
    assert axes.outlying(a, axes.shape(X @ D)).tolist() == [5]
    Y = X @ axes.haar(32, r)
    assert len(axes.outlying(axes.shape(Y), axes.shape(Y @ D))) == 0


def test_one_scale_per_row_reads_the_same_magnitude_excess_in_any_basis():
    import axes
    r = np.random.default_rng(2)
    n, d = 20000, 16
    Z = r.standard_normal((n, d))
    Q = axes.haar(d, r)
    for Y, scaled in ((Z * np.exp(0.5 * r.standard_normal((n, 1))), True), (Z, False)):
        Y = Y - Y.mean(0)
        G = axes.gaussian_copy(Y, r)
        on, off = axes.excess(Y, G).mean(), axes.excess(Y @ Q, G @ Q).mean()
        if scaled:
            assert on > 0.1 and abs(on - off) < 0.01
        else:
            assert abs(on) < 0.01 and abs(off) < 0.01


def test_a_correlated_group_is_found_on_the_axes_and_not_after_a_rotation():
    import axes
    x = planted_group()
    w = np.ones(x.shape[1])
    w[0] = 0  # a dead coordinate: the group keeps the cache's numbering
    X, live = axes.weighted(x, w, np.arange(len(x)))
    C = X.T @ X / len(X)
    coords = np.flatnonzero(live)
    g = axes.group(C, coords, 0.5)
    assert g["coords"] == [3, 10, 41] and len(g["pairs"]) == 3
    assert abs(abs(g["strongest"][0][2]) - 0.7) < 0.03
    s = axes.corr_stats(C, 0.5)
    assert s["coords_with_partner"] == 3 and s["pairs_over_threshold"] == 3
    Q = axes.haar(len(C), np.random.default_rng(3))
    rot = axes.corr_stats(Q.T @ C @ Q, 0.5)
    assert rot["coords_with_partner"] == 0 and rot["rho_max"] < 0.5
    np.testing.assert_allclose(np.linalg.eigvalsh(Q.T @ C @ Q), np.linalg.eigvalsh(C), atol=1e-8)
    pick, about = axes.draw(C, coords, 5, 0.45, seed=0)
    assert len(set(pick)) == 5 and not set(pick) & {0, 3, 10, 41}
    assert pick == axes.draw(C, coords, 5, 0.45, seed=0)[0] and about["excluded"] == 3


def test_the_axes_report_reads_every_check_and_draws_outside_the_group(tmp_path):
    import axes
    x = planted_group(n=4000, d=16, members=(3, 10, 12))
    c = tmp_path / "a.npy"
    np.save(c, x.astype(np.float32))
    w = np.ones(16, np.float32)
    w[0] = 0
    np.save(tmp_path / "w.npy", w)
    out = tmp_path / "out"
    rep = axes.main(["--arm", "a", str(c), str(tmp_path / "w.npy"), "--rows", "1500",
                     "--dirs", "64", "--rotations", "2", "--disjoint", "--draw", "4",
                     "--draw-arm", "a", "--reference", "a", "--out", str(out)])
    a = rep["arms"]["a"]
    assert a["live"] == 15 and a["group"]["coords"] == [3, 10, 12]
    assert a["group"]["disjoint"]["same"] and a["group"]["sizes"]["0.5"] == 3
    assert a["covariance"]["axes"]["coords_with_partner"] == 3 and len(a["covariance"]["rotations"]) == 2
    assert abs(a["energy"]["axes"]["mean"]) < 0.01 and a["marginals"]["axes"]["max_abs_skew"] < 0.3
    assert len(rep["draw"]["coords"]) == 4 and not set(rep["draw"]["coords"]) & {0, 3, 10, 12}
    assert json.loads((out / "axes.json").read_text())["draw"] == rep["draw"]
    assert np.load(out / "a.npz")["cov"].shape == (15, 15)


# ---------------------------------------------------------------- templates.py

def test_a_sentence_is_a_template_only_when_another_document_repeats_it(tmp_path):
    import templates
    assert templates.normalize("  Price:\t12,50 EUR ") == templates.normalize("price: 99,00 eur")
    docs = [{"doc": 0, "sents": [[1, 0, 5, "Retrieved from Wiki", "x", 1.0, "line", ""],
                                 [2, 0, 5, "A sentence of its own.", "x", 1.0, "sentence", ""]]},
            {"doc": 1, "sents": [[1, 0, 5, "retrieved  from wiki", "x", 1.0, "line", ""],
                                 [3, 0, 5, "Said twice here.", "x", 1.0, "sentence", ""],
                                 [4, 0, 5, "Said twice here.", "x", 1.0, "sentence", ""]]},
            {"doc": 2, "sents": [[1, 0, 5, "Version 2 of 7", "x", 1.0, "line", ""]]},
            {"doc": 3, "sents": [[2, 0, 5, "Version 5 of 9", "x", 1.0, "line", ""]]}]
    path = tmp_path / "c.sentences.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in docs))
    for workers in (1, 3):  # spans that cut lines are read once each
        h, k, ok, d, s, tail = templates.sentence_hashes(path, workers, tail_first=2)
        assert len(h) == len(k) == len(ok) == 7 and sorted(zip(d.tolist(), s.tolist())) == sorted(
            (x["doc"], t[0]) for x in docs for t in x["sents"])
        assert tail == {(2, 1): "Version 2 of 7", (3, 2): "Version 5 of 9"}
    row_doc = np.array([0, 0, 1, 1, 2, 3])
    row_seg = np.array([1, 2, 3, 4, 1, 2])
    # repeated across documents (case, spacing and digits aside) is flagged; within one is not
    assert templates.text_flags(row_doc, row_seg, h, d, s).tolist() == [
        True, False, False, False, True, True]
    with pytest.raises(SystemExit, match="not in the sentences file"):
        templates.text_flags(np.array([5]), np.array([1]), h, d, s)


def test_a_skeleton_masks_names_numbers_and_urls_and_needs_words_left():
    import templates
    t = templates.masked_tokens
    # the converter pages' sentences differ in their currency's name and date only
    a = t("An ráta malairte the WorldCoin Tugadh suas chun dáta deireanach ar 22 Deireadh Fómhair 2019")
    b = t("An ráta malairte the Ixcoin Tugadh suas chun dáta deireanach ar 3 Nollaig 2019")
    assert a == b and a[0] == "an"   # the first word keeps its capital's word
    assert t('Aisghafa ó "https://ga.wikipedia.org/w/index.php?title=X"') == ["aisghafa", "ó", "<url>"]
    assert t("Peruvian Nuevo Sol") == ["peruvian", "<name>"]   # a run of names is one name
    assert t("pay with WorldCoin or iPhone") == ["pay", "with", "<name>", "or", "<name>"]
    assert templates.skeleton_counts(t('Aisghafa ó "https://x.org"'))
    assert templates.skeleton_counts(t("Zum Warenkorb hinzufügen"))   # two words left
    assert not templates.skeleton_counts(t("Zum Warenkorb"))           # one word left
    assert not templates.skeleton_counts(t("Alpha Beta and 12 Gamma"))  # under half words


def test_a_shingle_twin_shares_most_three_grams_with_another_documents_row():
    import templates
    sents = ["the exchange rate for this coin was last updated from the market today",   # doc 0
             "the exchange rate for that coin was last updated from the market today",   # doc 1
             "the exchange rate for this coin was last updated from the market today",   # doc 0 again
             "a cat sat on the mat and looked out of the window for a while",            # doc 2
             "we ship every order within two working days of payment"]                   # doc 3
    doc = np.array([0, 1, 0, 2, 3])
    sets = [templates.shingles(templates.masked_tokens(x)) for x in sents]
    sig = templates.minhash(sets, 60, 0)
    assert sig.shape == (5, 60) and (sig[0] == sig[2]).all()
    # the signatures agree about as often as the sets' Jaccard index
    r = np.random.default_rng(5)
    a, b = r.choice(10 ** 6, 400, replace=False), r.choice(10 ** 6, 400, replace=False)
    b[:200] = a[:200]                       # 200 shared of 600: J = 1/3
    s2 = templates.minhash([np.unique(a).astype(np.uint64), np.unique(b).astype(np.uint64)], 600, 1)
    assert abs((s2[0] == s2[1]).mean() - 1 / 3) < 0.06
    flag = templates.near_twins(sets, doc, sig, 20, 0.5)
    assert templates.jaccard(sets[0], sets[1]) >= 0.5
    assert flag.tolist() == [True, True, True, False, False]
    # the same sentence twice in one document is no template
    assert not templates.near_twins(sets[:3:2], np.array([0, 0]), sig[:3:2], 20, 0.5).any()
    with pytest.raises(ValueError, match="empty"):
        templates.minhash([np.array([], np.uint64)], 4, 0)


def test_the_random_control_drops_as_many_rows_of_each_language():
    import templates
    groups = np.array(list("aaaaabbbbbbcc"))
    drop = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], bool)
    keep = templates.random_like(drop, groups, 0)
    for g, n in (("a", 2), ("b", 1), ("c", 0)):
        assert (~keep[groups == g]).sum() == n
    assert (keep == templates.random_like(drop, groups, 0)).all()


def test_a_twin_is_the_closest_row_of_another_document():
    import templates
    r = np.random.default_rng(4)
    E = r.standard_normal((40, 16))
    doc = np.repeat(np.arange(10), 4)
    E[1] = E[0] + 1e-3 * r.standard_normal(16)  # a twin in its own document does not count
    E[20] = E[5] + 1e-3 * r.standard_normal(16)  # one in another document does
    cos = templates.twin_cos(E, doc, chunk=7)  # 40 rows: the last chunk is padded
    assert cos[20] > 0.99 and cos[5] > 0.99
    assert cos[0] < 0.9 and cos[1] < 0.9
    X = E - E.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    S[doc[:, None] == doc[None, :]] = -2
    np.testing.assert_allclose(cos, S.max(1), atol=1e-5)
