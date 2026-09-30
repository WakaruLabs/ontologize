# The standalone SAE baseline trainer (sae.py): the comparison numbers for
# the Ontologizer ride on this being a correct field-standard top-k SAE
# trained on the same whitened objective.
import argparse

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
               enc="linear",
               epochs=200, max_steps=0, lr=2e-3,
               b=B, cache=str(cache), mse_weights=str(weights),
               eval_rows=512, aux_k=8, resample_every=0, dead_steps=50,
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
