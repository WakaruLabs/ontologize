# The standalone SAE baseline trainer (sae.py): the comparison numbers for
# the Ontologizer ride on this being a correct field-standard top-k SAE
# trained on the same whitened objective.
import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import sae

D, M, TOPK, B = 32, 64, 8, 256


def make_cfg(tmp_path, **overrides):
    # planted sparse ground truth: data generated from M/2 random
    # directions with TOPK-sparse non-negative coefficients, so a working
    # SAE can drive FVU far below the constant predictor
    rng = np.random.default_rng(0)
    W = rng.normal(size=(M // 2, D)).astype(np.float32)
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    C = np.abs(rng.normal(size=(4096 + 512, M // 2))).astype(np.float32)
    thr = np.sort(C, axis=1)[:, -TOPK:][:, :1]
    X = np.where(C >= thr, C, 0.0) @ W
    cache = tmp_path / "cache.npy"
    np.save(cache, X)
    weights = tmp_path / "w.npy"
    np.save(weights, np.ones(D, np.float32))

    cfg = dict(m=M, topk=TOPK, l1=0.0, groups=0, group_fn="top1", prefixes=1,
               enc="linear", block=0,
               epochs=200, max_steps=0, lr=2e-3, adam_eps=1e-8,
               b=B, cache=str(cache), mse_weights=str(weights),
               eval_rows=512, train_rows=0, block_rows=0, buffer_blocks=16,
               aux_k=8, resample_every=0, dead_steps=50,
               sd_pre=0.0, p_drop=0.0, sd_out=0.0,
               save_each=10**6,
               seed=0, out=str(tmp_path / "out"), overwrite=False)
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def test_topk_sae_learns_planted_dictionary(tmp_path):
    fvu, l0, dead = sae.train(make_cfg(tmp_path))
    assert fvu < 0.15          # far below the constant predictor
    assert l0 <= TOPK + 0.01   # top-k enforced exactly (ties can only reduce)
    assert dead < M // 4

    params = np.load(tmp_path / "out" / "params.npz")
    norms = np.linalg.norm(params["W_dec"], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_resume_from_snapshot(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=20, save_each=10)
    sae.train(cfg)
    state = np.load(tmp_path / "out" / "state.npz")
    assert int(state["step"]) == 20
    # resumed run continues the schedule instead of restarting
    cfg2 = make_cfg(tmp_path, max_steps=40, save_each=10)
    sae.train(cfg2)
    state = np.load(tmp_path / "out" / "state.npz")
    assert int(state["step"]) == 40


def test_adam_eps_is_a_floor_in_the_loss_units(tmp_path):
    # Adam steps by m_hat / (sqrt(v_hat) + eps), so scaling the loss by c
    # is the same as dividing eps by c. With the MSE weights scaled by
    # 1e-10 the stock eps all but freezes training; eps scaled with them
    # gives the unscaled run back. FVU does not see the weights' scale
    small = tmp_path / "w_small.npy"
    np.save(small, np.full(D, 1e-10, np.float32))
    run = lambda name, **kw: sae.train(make_cfg(
        tmp_path, max_steps=300, out=str(tmp_path / name), **kw))[0]
    stock = run("stock")
    frozen = run("frozen", mse_weights=str(small))
    scaled = run("scaled", mse_weights=str(small), adam_eps=1e-18)
    assert abs(scaled - stock) < 1e-3
    assert frozen > stock + 0.1
    assert json.loads((tmp_path / "scaled" / "meta.json").read_text())[
        "adam_eps"] == 1e-18
    assert "_eps1e-15" in sae.run_name(make_cfg(tmp_path, adam_eps=1e-15))
    assert "_eps" not in sae.run_name(make_cfg(tmp_path))


def noise_model(**kw):
    """A flat linear top-k SAE over the planted data's shape, with its
    parameters, a batch and its whitening."""
    from ontologize.sae import SAE
    model = SAE(d=D, m=M, topk=TOPK, aux_k=0, **kw)
    X = jax.random.normal(jax.random.PRNGKey(1), (B, D))
    params = model.init(jax.random.PRNGKey(0), X)
    return model, params, X, jnp.ones(D), jnp.zeros(M, bool)


def test_training_noise_off_is_the_stock_loss():
    model, params, X, w, dead = noise_model()
    stock = model.apply(params, X, w, dead, method="loss")[0]
    keyed = model.apply(params, X, w, dead, jax.random.PRNGKey(2), method="loss")[0]
    assert float(stock) == float(keyed)
    # set but given no rng (encoding, evaluation): stock again
    noisy, *_ = noise_model(sd_pre=0.3, p_drop=0.1, sd_out=0.1)
    assert float(noisy.apply(params, X, w, dead, method="loss")[0]) == float(stock)
    np.testing.assert_array_equal(noisy.apply(params, X, method="encode"),
                                  model.apply(params, X, method="encode"))


def test_selection_noise_moves_the_choice_not_the_values():
    model, params, X, *_ = noise_model(sd_pre=3.0)
    pre = model.apply(params, X, method="preacts")
    z = model.apply(params, pre, jax.random.PRNGKey(3), method="noisy_activate")
    clean = model.apply(params, pre, method="activate")
    on = z > 0
    np.testing.assert_allclose(z[on], jax.nn.relu(pre)[on])  # clean values
    assert int(on.sum(-1).max()) <= TOPK
    assert not np.array_equal(np.asarray(on), np.asarray(clean > 0))


def test_winner_dropout_at_one_promotes_the_next_k():
    model, params, X, *_ = noise_model(p_drop=1.0)
    pre = model.apply(params, X, method="preacts")
    z = model.apply(params, pre, jax.random.PRNGKey(4), method="noisy_activate")
    ranks = np.argsort(-np.asarray(pre), axis=-1)
    for n in range(B):
        if (np.asarray(pre[n])[ranks[n, :2 * TOPK]] > 0).all():
            np.testing.assert_array_equal(np.sort(np.nonzero(np.asarray(z[n]))[0]),
                                          np.sort(ranks[n, TOPK:2 * TOPK]))


def test_output_noise_variance_is_the_per_latent_sum():
    model, params, X, *_ = noise_model(sd_out=0.5)
    z = model.apply(params, X, method="encode")
    W = np.asarray(params["params"]["decoder"]["weights"], np.float64)  # (d, m)
    want = 0.25 * (W ** 2) @ np.asarray(z, np.float64).var(0)
    draws = np.stack([np.asarray(model.apply(params, z, jax.random.PRNGKey(i),
                                             method="out_noise"))
                      for i in range(64)])
    np.testing.assert_allclose(draws.reshape(-1, D).var(0), want, rtol=0.1)


def test_training_noise_needs_a_flat_linear_topk_sae(tmp_path):
    with pytest.raises(AssertionError):
        sae.train(make_cfg(tmp_path, topk=0, l1=1e-3, sd_pre=0.3))
    with pytest.raises(ValueError):
        model, params, X, w, dead = noise_model(sd_pre=0.3, prefixes=2)
        model.apply(params, X, w, dead, jax.random.PRNGKey(0), method="loss")


def test_noised_run_trains_and_records_its_noise(tmp_path):
    cfg = make_cfg(tmp_path, max_steps=300, sd_pre=0.3, p_drop=0.1, sd_out=0.1)
    fvu, l0, _ = sae.train(cfg)
    assert np.isfinite(fvu) and fvu < 0.5 and l0 <= TOPK + 0.01
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert (meta["sd_pre"], meta["p_drop"], meta["sd_out"]) == (0.3, 0.1, 0.1)
    assert sae.run_name(cfg).endswith("_sdpre0.3_pdrop0.1_sdout0.1_s0")


def test_train_rows_takes_the_head_and_keeps_the_tail():
    assert sae.splits(100, 10, 0) == (90, 90)
    assert sae.splits(100, 10, 30) == (30, 90)
    assert sae.splits(100, 10, 500) == (90, 90)


def test_block_schedule_covers_whole_blocks_once():
    rng = np.random.default_rng(0)
    groups = list(sae.block_schedule(rng, 10 * 8 + 5, 8, 3))
    assert [len(s) for s, _ in groups] == [3, 3, 3, 1]
    starts = np.concatenate([s for s, _ in groups])
    # every whole block exactly once; the 5 rows past the last are left out
    assert sorted(starts.tolist()) == list(range(0, 80, 8))
    for s, perm in groups:
        assert sorted(perm.tolist()) == list(range(len(s) * 8))


@pytest.mark.parametrize("block_rows", [0, 512])
def test_resume_replays_the_schedule(tmp_path, block_rows):
    # 40 steps straight and 20 + 20 with a resume between give the same
    # parameters: the resumed run skips exactly the batches already used
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    straight = make_cfg(tmp_path / "a", max_steps=40, block_rows=block_rows,
                        buffer_blocks=2, save_each=20)
    sae.train(straight)
    for steps in (20, 40):
        sae.train(make_cfg(tmp_path / "b", max_steps=steps, block_rows=block_rows,
                           buffer_blocks=2, save_each=20))
    a = np.load(tmp_path / "a" / "out" / "params.npz")
    b = np.load(tmp_path / "b" / "out" / "params.npz")
    for k in a.files:
        np.testing.assert_array_equal(a[k], b[k])


def test_block_shuffled_sae_learns_planted_dictionary(tmp_path):
    fvu, l0, dead = sae.train(make_cfg(tmp_path, block_rows=512, buffer_blocks=2))
    assert fvu < 0.15 and l0 <= TOPK + 0.01 and dead < M // 4


def test_train_rows_run_scores_the_same_tail(tmp_path):
    # training on the first 1024 rows still learns, and its eval rows are
    # the cache's last eval_rows whatever train_rows is
    fvu, l0, _ = sae.train(make_cfg(tmp_path, train_rows=1024, epochs=400))
    assert fvu < 0.3 and l0 <= TOPK + 0.01


def test_l1_mode_is_sparse(tmp_path):
    cfg = make_cfg(tmp_path, topk=0, l1=3e-2, epochs=60,
                   out=str(tmp_path / "out_l1"))
    fvu, l0, _ = sae.train(cfg)
    assert fvu < 0.8  # sparsity costs reconstruction, but beats the mean
    assert l0 < M / 3  # the penalty actually sparsifies


def test_resample_dead_directions_and_moments():
    # resampled latents point at badly reconstructed examples, tied
    # encoder columns at 0.2x the alive norm, zero bias, and zeroed Adam
    # moments -- alive latents and b_dec untouched
    m, n = 16, 64
    rng = np.random.default_rng(5)
    params = sae.init_params(jax.random.PRNGKey(0), D, m, np.zeros(D))
    opt_state = optax.adam(1e-3).init(params)
    adam = opt_state[0]._replace(
        mu=jax.tree_util.tree_map(jnp.ones_like, params),
        nu=jax.tree_util.tree_map(jnp.ones_like, params))
    opt_state = (adam,) + tuple(opt_state[1:])

    X = jnp.asarray(rng.normal(size=(n, D)), jnp.float32)
    dead = np.zeros(m, bool)
    dead[:4] = True
    p2, o2 = sae.resample_dead(params, opt_state, X, jnp.ones(D), dead,
                               np.random.default_rng(0))

    v = np.asarray(p2["W_dec"][:4])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    Xn = np.asarray(X) - np.asarray(params["b_dec"])
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    assert np.all((v @ Xn.T).max(1) > 0.999)  # rows are centered examples

    cols = np.linalg.norm(np.asarray(p2["W_enc"][:, :4]), axis=0)
    alive_norm = np.linalg.norm(
        np.asarray(params["W_enc"])[:, 4:], axis=0).mean()
    assert np.allclose(cols, 0.2 * alive_norm, rtol=1e-4)
    assert np.all(np.asarray(p2["b_enc"][:4]) == 0.0)
    assert np.array_equal(np.asarray(p2["W_dec"][4:]),
                          np.asarray(params["W_dec"][4:]))

    mu = o2[0].mu
    assert np.all(np.asarray(mu["W_dec"])[:4] == 0)
    assert np.all(np.asarray(mu["W_dec"])[4:] == 1)
    assert np.all(np.asarray(mu["W_enc"])[:, :4] == 0)
    assert np.all(np.asarray(mu["W_enc"])[:, 4:] == 1)
    assert np.all(np.asarray(mu["b_enc"])[:4] == 0)
    assert np.all(np.asarray(mu["b_dec"]) == 1)  # b_dec never resampled


def test_resample_mode_trains(tmp_path):
    # mutually exclusive with aux-k
    with pytest.raises(AssertionError):
        sae.train(make_cfg(tmp_path, resample_every=100))  # aux_k still on
    cfg = make_cfg(tmp_path, topk=0, l1=3e-2, aux_k=0, epochs=60,
                   resample_every=40, dead_steps=20,
                   out=str(tmp_path / "out_rs"))
    fvu, l0, dead = sae.train(cfg)
    assert np.isfinite(fvu) and fvu < 0.8
    assert l0 < M / 3  # still an L1-sparse run


def test_evaluate_l1_stat(tmp_path):
    # evaluate returns (fvu, l0, l1): l1 is the held-out mean per-sample
    # |z|_1, logged unpenalized in every mode so its trajectory is
    # observable without a sparsity term in the loss
    cfg = make_cfg(tmp_path, max_steps=10, save_each=5)
    sae.train(cfg)
    params = {k: jnp.asarray(v)
              for k, v in np.load(tmp_path / "out" / "params.npz").items()}
    X_eval = np.load(cfg.cache)[-cfg.eval_rows:].astype(np.float32)
    w_sqrt = jnp.ones(D, jnp.float32)
    base_w = float((X_eval.var(0)).mean())
    fvu, l0, l1, freq = sae.evaluate(sae.make_eval(TOPK), params, X_eval,
                                     w_sqrt, base_w, B)
    assert np.isfinite(l1) and l1 > 0.0
    # manual check (equal-size batches, so batch-mean average == full mean)
    z = sae.encode(params, jnp.asarray(X_eval), TOPK, 0, "top1")
    assert l1 == pytest.approx(float(jnp.abs(z).sum(-1).mean()), rel=1e-3)
    # firing frequencies: per-latent, in [0, 1], consistent with mean L0
    assert freq.shape == (M,)
    assert np.all(freq >= 0) and np.all(freq <= 1)
    assert freq.sum() == pytest.approx(l0, rel=1e-3)
    # the full loss.csv row: step, loss, mse_w, fvu, l0, l1, dead,
    # revived, new_dead, use_ent, hifreq, wdec_maxcos, wdec_meancos,
    # fvu_train
    rows = np.loadtxt(tmp_path / "out" / "loss.csv", delimiter=",")
    assert rows.reshape(-1, 14).shape[1] == 14
    assert np.all(np.isfinite(rows))


def test_usage_stats():
    # uniform usage -> effective count = m; single latent -> 1
    eff, hi = sae.usage_stats(np.full(8, 0.5))
    assert eff == pytest.approx(8.0)
    assert hi == 8
    eff, hi = sae.usage_stats(np.array([0.5, 0, 0, 0]))
    assert eff == pytest.approx(1.0)
    assert hi == 1
    eff, hi = sae.usage_stats(np.zeros(4))  # nothing fires
    assert (eff, hi) == (0.0, 0)


def test_wdec_cos():
    # orthogonal rows -> 0; a duplicated row -> max 1, and the chunked
    # path agrees with the dense computation
    params = {"W_dec": jnp.eye(6, 12)}
    mx, mean = sae.wdec_cos(params, chunk=4)
    assert mx == pytest.approx(0.0, abs=1e-6)
    assert mean == pytest.approx(0.0, abs=1e-6)

    W = np.array(jax.random.normal(jax.random.PRNGKey(0), (6, 12)))
    W[3] = 2.0 * W[1]  # duplicate feature at a different scale
    params = {"W_dec": jnp.asarray(W)}
    mx, mean = sae.wdec_cos(params, chunk=4)
    assert mx == pytest.approx(1.0, abs=1e-5)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    C = np.abs(Wn @ Wn.T)
    np.fill_diagonal(C, 0.0)
    assert mean == pytest.approx(C.sum() / (6 * 5), rel=1e-5)


# ---- structural-ablation rungs (head / layer analogues) ----

def rand_params(seed=0, m=M, d=D):
    rng = np.random.default_rng(seed)
    return {"W_enc": jnp.asarray(rng.normal(size=(d, m)), jnp.float32),
            "b_enc": jnp.asarray(rng.normal(size=m), jnp.float32),
            "W_dec": jnp.asarray(rng.normal(size=(m, d)), jnp.float32),
            "b_dec": jnp.zeros(d)}


def test_group_top1_at_most_one_winner_per_group():
    params = rand_params()
    X = jnp.asarray(np.random.default_rng(1).normal(size=(64, D)),
                    jnp.float32)
    G = 8
    z = np.asarray(sae.encode(params, X, 0, groups=G))
    per_group = (z.reshape(64, G, M // G) > 0).sum(-1)
    assert per_group.max() <= 1
    # winner must be the group argmax of the raw logits
    pre = np.asarray((X - params["b_dec"]) @ params["W_enc"]
                     + params["b_enc"]).reshape(64, G, M // G)
    fired = z.reshape(64, G, M // G).argmax(-1)
    lit = per_group == 1
    assert (fired[lit] == pre.argmax(-1)[lit]).all()


def test_group_softmax_sums_to_one_per_group():
    params = rand_params()
    X = jnp.asarray(np.random.default_rng(2).normal(size=(64, D)),
                    jnp.float32)
    G = 8
    z = np.asarray(sae.encode(params, X, 0, groups=G, group_fn="softmax"))
    sums = z.reshape(64, G, M // G).sum(-1)
    assert np.allclose(sums, 1.0, atol=1e-5)


def test_group_top1_trains(tmp_path):
    # G = planted dirs: a per-group winner code can cover the dictionary
    cfg = make_cfg(tmp_path, topk=0, groups=M // 2, aux_k=0,
                   out=str(tmp_path / "out_g"))
    fvu, l0, _ = sae.train(cfg)
    assert l0 <= M // 2 + 0.01  # at most one winner per group
    assert fvu < 0.25  # ample groups: near the unconstrained top-k regime


def test_group_softmax_trains_and_skips_renorm(tmp_path):
    cfg = make_cfg(tmp_path, topk=0, groups=8, group_fn="softmax", aux_k=0,
                   out=str(tmp_path / "out_gs"))
    fvu, l0, _ = sae.train(cfg)
    assert fvu < 0.5  # dense mixture code reconstructs decently
    assert l0 == pytest.approx(M)  # softmax fires every latent
    # decoder rows must carry magnitude in this mode (no unit renorm)
    params = np.load(tmp_path / "out_gs" / "params.npz")
    norms = np.linalg.norm(params["W_dec"], axis=-1)
    assert not np.allclose(norms, 1.0, atol=1e-3)


def test_bilinear_encode_matches_reference():
    rng = np.random.default_rng(7)
    params = {"W_enc1": jnp.asarray(rng.normal(size=(D + 1, M)), jnp.float32),
              "W_enc2": jnp.asarray(rng.normal(size=(D + 1, M)), jnp.float32),
              "b_enc": jnp.asarray(rng.normal(size=M), jnp.float32),
              "W_dec": jnp.asarray(rng.normal(size=(M, D)), jnp.float32),
              "b_dec": jnp.asarray(rng.normal(size=D), jnp.float32)}
    X = rng.normal(size=(16, D)).astype(np.float32)

    Xa = np.concatenate([X - np.asarray(params["b_dec"]),
                         np.ones((16, 1), np.float32)], -1)
    want = ((Xa @ np.asarray(params["W_enc1"]))
            * (Xa @ np.asarray(params["W_enc2"]))
            + np.asarray(params["b_enc"]))
    got = np.asarray(sae.preacts(params, jnp.asarray(X)))
    assert np.allclose(got, want, atol=1e-4)

    # top-k applies to the bilinear pre-activations like any other
    z = np.asarray(sae.encode(params, jnp.asarray(X), TOPK))
    assert ((z > 0).sum(-1) <= TOPK).all()


def test_eigenfeatures_closed_form_matches_eigh():
    rng = np.random.default_rng(8)
    params = {"W_enc1": jnp.asarray(rng.normal(size=(D + 1, M)), jnp.float32),
              "W_enc2": jnp.asarray(rng.normal(size=(D + 1, M)), jnp.float32),
              "W_dec": jnp.asarray(rng.normal(size=(M, D)), jnp.float32)}
    E = np.asarray(sae.eigenfeatures(params))
    assert E.shape == (M, D)
    assert np.allclose(np.linalg.norm(E, axis=-1), 1.0, atol=1e-5)
    # sign convention: aligned with the latent's decoder row
    assert ((E * np.asarray(params["W_dec"])).sum(-1) >= 0).all()

    for j in range(0, M, 7):
        u = np.asarray(params["W_enc1"])[:, j]
        v = np.asarray(params["W_enc2"])[:, j]
        B = (np.outer(u, v) + np.outer(v, u)) / 2
        vals, vecs = np.linalg.eigh(B)
        top = vecs[:, np.argmax(np.abs(vals))][:D]
        top /= np.linalg.norm(top)
        assert abs(abs(top @ E[j]) - 1.0) < 1e-4, j


def test_bilinear_topk_sae_trains(tmp_path):
    cfg = make_cfg(tmp_path, enc="bilinear", lr=1e-3,
                   out=str(tmp_path / "out_bl"))
    fvu, l0, _ = sae.train(cfg)
    assert fvu < 0.3           # bilinear encoder still fits the planted data
    assert l0 <= TOPK + 0.01
    params = np.load(tmp_path / "out_bl" / "params.npz")
    assert "W_enc2" in params  # form recorded in the checkpoint itself


def test_matryoshka_prefixes_are_monotone(tmp_path):
    P = 4
    cfg = make_cfg(tmp_path, prefixes=P, out=str(tmp_path / "out_p"))
    fvu, _, _ = sae.train(cfg)
    assert fvu < 0.3

    # each nested prefix must reconstruct on its own, improving with depth
    params = {k: jnp.asarray(v)
              for k, v in np.load(tmp_path / "out_p" / "params.npz").items()}
    X = jnp.asarray(np.load(tmp_path / "cache.npy")[-512:])
    z = sae.encode(params, X, TOPK)
    base = float(X.var(0).mean())
    fvus = []
    for j in range(1, P + 1):
        mask = np.zeros(M, np.float32)
        mask[:j * (M // P)] = 1.0
        recon = sae.decode(params, z * mask)
        fvus.append(float(((recon - X) ** 2).mean(0).mean()) / base)
    assert all(a >= b - 0.02 for a, b in zip(fvus, fvus[1:])), fvus
    assert fvus[0] < 1.0  # the first block alone beats the constant predictor


# ---- the block-sparse featurizer ----

def plant_blocks(path, n=4096 + 512, k=2, seed=1):
    """Block-sparse ground truth: each row sums `k` of M // 8 random 4-d
    subspaces of R^D, at Gaussian (signed) coordinates in each."""
    rng = np.random.default_rng(seed)
    F = np.linalg.qr(rng.normal(size=(M // 8, D, 4)))[0]       # (8, D, 4)
    on = np.argsort(rng.random((n, M // 8)), -1)[:, :k]
    X = np.einsum("nkj,nkdj->nd", rng.normal(size=(n, k, 4)), F[on])
    np.save(path, X.astype(np.float32))
    return str(path)


def test_block_sparse_featurizer_learns_planted_subspaces(tmp_path):
    out = tmp_path / "out_bsf"
    cfg = make_cfg(tmp_path, cache=plant_blocks(tmp_path / "blocks.npy"),
                   block=4, topk=2, aux_k=0, out=str(out))
    fvu, l0, dead = sae.train(cfg)
    assert fvu < 0.05
    assert l0 == pytest.approx(2 * 4)   # whole blocks of signed coordinates

    params = np.load(out / "params.npz")
    assert params["W_blk"].shape == (D, M // 4, 4) and "W_enc" not in params
    blocks = params["W_dec"].reshape(M // 4, 4, D)
    assert (np.sqrt((blocks ** 2).sum((1, 2))) <= 1 + 1e-5).all()
    assert json.loads((out / "meta.json").read_text())["block"] == 4


def test_block_checkpoints_need_their_block():
    """A consumer that does not pass `block` would read topk as latents, so
    encoding refuses; decoding reads every code alike."""
    params = sae.init_params(jax.random.PRNGKey(0), D, M, np.zeros(D),
                             block=4)
    X = jnp.asarray(np.random.default_rng(0).normal(size=(16, D)),
                    jnp.float32)
    plain = sae.init_params(jax.random.PRNGKey(0), D, M, np.zeros(D))
    for p, block in ((params, 0), (params, 2), (plain, 4)):
        with pytest.raises(ValueError):
            sae.encode(p, X, 2, block=block)
    z = sae.encode(params, X, 2, block=4)
    assert ((np.asarray(z).reshape(16, M // 4, 4) != 0).any(-1).sum(-1)
            == 2).all()
    assert sae.decode(params, z).shape == (16, D)
    assert np.allclose(sae.activate(sae.preacts(params, X, block=4), 2,
                                    block=4), z)


def test_block_flags(tmp_path):
    assert sae.run_name(make_cfg(tmp_path, block=4, topk=8, seed=42)) \
        == "m64_bsf4_k8"
    for bad in (dict(),                         # aux-k still on
                dict(aux_k=0, topk=9),          # no room for the runner-ups
                dict(aux_k=0, l1=1e-3),
                dict(aux_k=0, enc="bilinear"),
                dict(aux_k=0, resample_every=100)):
        with pytest.raises(AssertionError):
            sae.train(make_cfg(tmp_path, **{"block": 4, "topk": 8, **bad}))
