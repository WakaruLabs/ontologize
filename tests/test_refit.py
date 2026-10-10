# refit.py: the masked weighted least-squares refit and the linearity
# assumption that lets the Ontologizer share the machinery (output =
# flattened classifications @ a fixed direction matrix).
import jax
import jax.numpy as jnp
import numpy as np

import refit

D, F, K = 16, 20, 4


def synth(seed=0, b=64):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(F, D)).astype(np.float32)
    base = rng.normal(size=D).astype(np.float32)
    idx = np.stack([rng.choice(F, K, replace=False) for _ in range(b)])
    p = np.abs(rng.normal(size=(b, K))).astype(np.float32) + 0.5
    X = base + np.einsum("bk,bkd->bd", p, dirs[idx])
    return dirs, base, idx, p, X.astype(np.float32)


def test_refit_recovers_shrunk_coefficients():
    dirs, base, idx, p, X = synth()
    w_sqrt = jnp.ones(D)
    # the "activation rule" shrinks every coefficient by half
    raw = base + np.einsum("bk,bkd->bd", 0.5 * p, dirs[idx])
    fit = np.asarray(refit.refit_recon(
        jnp.asarray(X), jnp.asarray(dirs), jnp.asarray(idx),
        jnp.ones(idx.shape, jnp.float32), jnp.asarray(base), w_sqrt))
    assert ((raw - X) ** 2).mean() > 0.1
    assert ((fit - X) ** 2).mean() < 1e-6  # exact support => exact recovery


def test_masked_slots_contribute_nothing():
    dirs, base, idx, p, X = synth(1)
    w_sqrt = jnp.ones(D)
    mask = np.ones(idx.shape, np.float32)
    mask[:, -1] = 0.0  # pretend the last slot is a silent group
    fit = np.asarray(refit.refit_recon(
        jnp.asarray(X), jnp.asarray(dirs), jnp.asarray(idx),
        jnp.asarray(mask), jnp.asarray(base), w_sqrt))
    fit3 = np.asarray(refit.refit_recon(
        jnp.asarray(X), jnp.asarray(dirs), jnp.asarray(idx[:, :-1]),
        jnp.ones((len(X), K - 1), jnp.float32), jnp.asarray(base), w_sqrt))
    assert np.allclose(fit, fit3, atol=1e-4)


def test_sae_supports_topk_and_groups():
    import sae
    rng = np.random.default_rng(2)
    m = 24
    params = {"W_enc": jnp.asarray(rng.normal(size=(D, m)), jnp.float32),
              "b_enc": jnp.asarray(rng.normal(size=m), jnp.float32),
              "W_dec": jnp.asarray(rng.normal(size=(m, D)), jnp.float32),
              "b_dec": jnp.zeros(D)}
    X = jnp.asarray(rng.normal(size=(8, D)), jnp.float32)

    z, idx, mask = refit.sae_supports(params, X, topk=5, groups=0,
                                      group_fn="top1")
    zn = np.asarray(z)
    for b in range(8):
        active = set(np.where(zn[b] > 0)[0])
        chosen = set(np.asarray(idx[b])[np.asarray(mask[b]) > 0])
        assert chosen == active

    z, idx, mask = refit.sae_supports(params, X, topk=0, groups=6,
                                      group_fn="top1")
    gi = np.asarray(idx) // (m // 6)
    assert (gi == np.arange(6)).all()  # one slot per group, in order
    zn = np.asarray(z)
    for b in range(8):
        assert set(np.asarray(idx[b])[np.asarray(mask[b]) > 0]) \
            == set(np.where(zn[b] > 0)[0])


def test_onto_output_is_linear_in_classifications(build, X):
    """The assumption behind the onto refit path: Y = P_flat @ G for a
    fixed G measured from one-hot constant codes."""
    model, params = build()
    l, h, k = model.l, model.h, model.k
    T = 0.5

    def probe_p(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = E
        Ps = []
        for i, de in enumerate(module.dictencs):
            P = de.dict.cluster(de.classifier(Ein), T)
            Ps.append(P)
            R = R + de.dict.combine(de.dict.hfwd(P))
            if i < module.l - 1:
                Ein = module.nextinput(X, R, None)
        return jnp.stack(Ps, 1), module.decode(R)

    def probe_c(module, Ps):
        R = jnp.zeros((Ps.shape[0], module.e_dec))
        for i, de in enumerate(module.dictencs):
            R = R + de.dict.combine(de.dict.hfwd(Ps[:, i]))
        return module.decode(R)

    P, Y = model.apply(params, X, method=probe_p)
    Ff = l * h * k
    codes = jnp.eye(Ff).reshape(Ff, l, h, k)
    G = model.apply(params, codes, method=probe_c)   # (F, d_out)
    Y_lin = P.reshape(len(X), Ff) @ G
    assert np.allclose(np.asarray(Y), np.asarray(Y_lin), atol=1e-4)


def test_random_support_layout():
    b, n_head, k, m = 32, 5, 8, 3
    idx = np.asarray(refit.random_support(jax.random.PRNGKey(0), b, n_head,
                                          k, m))
    assert idx.shape == (b, n_head * m)
    per = idx.reshape(b, n_head, m)
    # each head's draws stay in its own block and never repeat
    assert (per // k == np.arange(n_head)[None, :, None]).all()
    assert all(len(set(row)) == m for row in per.reshape(-1, m))
    flat = np.asarray(refit.random_support(jax.random.PRNGKey(1), b, 1, 50,
                                           20))
    assert flat.shape == (b, 20) and flat.min() >= 0 and flat.max() < 50
    assert all(len(set(row)) == 20 for row in flat)


def test_correction_on_own_support_is_the_refit():
    """raw - offset in the support's span (an SAE's decode): correcting
    raw on its own support lands on the refit from the offset."""
    dirs, base, idx, p, X = synth(4)
    w_sqrt = jnp.asarray(np.random.default_rng(5).uniform(0.5, 2.0, D),
                         jnp.float32)
    X = X + np.random.default_rng(6).normal(size=X.shape).astype(np.float32)
    raw = base + np.einsum("bk,bkd->bd", 0.7 * p, dirs[idx])
    ones = jnp.ones(idx.shape, jnp.float32)
    fit = refit.refit_recon(jnp.asarray(X), jnp.asarray(dirs),
                            jnp.asarray(idx), ones, jnp.asarray(base), w_sqrt)
    e_fit = float((((fit - X) * w_sqrt) ** 2).mean())
    e_corr = float(refit.correction_fvu(
        jnp.asarray(X), jnp.asarray(raw.astype(np.float32)),
        jnp.asarray(dirs), jnp.asarray(idx), ones, w_sqrt))
    assert np.isclose(e_corr, e_fit, rtol=1e-4)


def test_random_support_removes_its_share_of_an_isotropic_residual():
    """|S| directions chosen without the residual remove |S|/d of it in
    expectation: the iso_share reference."""
    rng = np.random.default_rng(7)
    d, F_, S, b = 64, 200, 16, 4096
    dirs = jnp.asarray(rng.normal(size=(F_, d)), jnp.float32)
    X = jnp.asarray(rng.normal(size=(b, d)), jnp.float32)
    raw = jnp.zeros((b, d))
    idx = refit.random_support(jax.random.PRNGKey(2), b, 1, F_, S)
    e = float(refit.correction_fvu(X, raw, dirs, idx,
                                   jnp.ones((b, S), jnp.float32),
                                   jnp.ones(d)))
    share = 1.0 - e / float((X ** 2).mean())
    assert abs(share - S / d) < 0.01


def test_onto_supports_shapes_and_renorm():
    rng = np.random.default_rng(3)
    l, h, k, b, m = 2, 3, 8, 16, 2
    P = rng.exponential(size=(b, l * h * k)).astype(np.float32)
    P = P.reshape(b, l * h, k)
    P /= P.sum(-1, keepdims=True)
    origin = np.full((l * h, k), 1.0 / k, np.float32)
    idx, Pt = refit.onto_supports(jnp.asarray(P.reshape(b, -1)),
                                  jnp.asarray(origin), m, l, h, k)
    assert idx.shape == (b, l * h * m)
    # ids stay within each head's block, in head order
    heads = np.asarray(idx).reshape(b, l * h, m) // k
    assert (heads == np.arange(l * h)[None, :, None]).all()
    assert np.allclose(np.asarray(Pt).reshape(b, l * h, k).sum(-1), 1.0,
                       atol=1e-5)
