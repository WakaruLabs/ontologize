# The standalone SAE baseline trainer (sae.py): the comparison numbers for
# the Ontologizer ride on this being a correct field-standard top-k SAE
# trained on the same whitened objective.
import argparse

import jax.numpy as jnp
import numpy as np
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
               eval_rows=512, aux_k=8, dead_steps=50, save_each=10**6,
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
