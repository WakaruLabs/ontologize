"""RBF-kernel HSIC estimators in JAX: biased (Gretton et al. 2005) and
unbiased (Song et al. 2012), plus the batched per-head pairwise-CKA
penalty this experiment wires into training.

All functions are pure jnp and jit/grad-safe. Kernel bandwidths use the
median heuristic by default (sigma^2 = median of the off-diagonal
squared distances / 2), computed under stop_gradient so the bandwidth
is a constant of the batch, not a learnable escape hatch.

Conventions:
  gram      k(x, y) = exp(-||x - y||^2 / (2 sigma^2))
  biased    HSIC_b(K, L) = tr(K H L H) / (n - 1)^2, H = I - 11^T/n
  unbiased  Song et al. 2012 eq. (5) over zero-diagonal grams; unbiased
            under independence (can be negative), needs n > 3
  cka       HSIC_b(K, L) / sqrt(HSIC_b(K, K) HSIC_b(L, L)); in [0, 1]

The head penalty `pairwise_head_cka` treats each head's classification
p (b, k) as a random variable observed over the batch and penalizes the
mean pairwise CKA between heads: independent heads score ~0, redundant
heads score ~1. Computed for all h(h-1)/2 pairs at O(h b^2) via the
sum-of-grams identity
  sum_{h<h'} <G_h, G_h'> = (||sum_h G_h||_F^2 - sum_h ||G_h||_F^2) / 2
over Frobenius-normalized centered grams. NOTE an important boundary
case: a FROZEN head emits a constant code, its centered gram is ~0, and
it contributes ~0 to every pair -- this penalty rewards independence
but does NOT punish freezing (that is s_Hm's job; see the README).

Self-test (CPU, seconds):  python -m ontologize.fns.hsic
"""
import jax
import jax.numpy as jnp

EPS = 1e-12


def sqdists(X):
    """Pairwise squared euclidean distances of rows: (n, d) -> (n, n)."""
    n2 = jnp.sum(X * X, axis=-1)
    D2 = n2[..., :, None] + n2[..., None, :] - 2.0 * (X @ X.T)
    return jnp.maximum(D2, 0.0)


def median_sigma2(D2):
    """Median-heuristic bandwidth: sigma^2 = median(off-diag D2) / 2.
    stop_gradient: the bandwidth is a batch constant. jit-safe (the
    diagonal is masked to NaN, nanmedian handles it with static shapes)."""
    n = D2.shape[-1]
    eye = jnp.eye(n, dtype=bool)
    off = jnp.where(eye, jnp.nan, D2)
    med = jnp.nanmedian(off)
    return jax.lax.stop_gradient(jnp.maximum(0.5 * med, EPS))


def rbf_gram(X, sigma2=None):
    """RBF Gram matrix of the rows of X; median-heuristic bandwidth when
    `sigma2` is None."""
    D2 = sqdists(X)
    if sigma2 is None:
        sigma2 = median_sigma2(D2)
    return jnp.exp(-D2 / (2.0 * sigma2))


def center(K):
    """Double-centering H K H without materializing H."""
    rm = K.mean(axis=-1, keepdims=True)
    cm = K.mean(axis=-2, keepdims=True)
    return K - rm - cm + K.mean(axis=(-2, -1), keepdims=True)


def hsic_biased(Kx, Ky):
    """tr(Kx H Ky H) / (n-1)^2. O(n^2) via <H Kx H, Ky>_F (symmetry)."""
    n = Kx.shape[-1]
    return jnp.sum(center(Kx) * Ky) / (n - 1) ** 2


def hsic_unbiased(Kx, Ky):
    """Song et al. 2012 eq. (5). Requires n > 3; unbiased under
    independence, so small negative values are normal."""
    n = Kx.shape[-1]
    eye = jnp.eye(n, dtype=Kx.dtype)
    Kt = Kx * (1.0 - eye)
    Lt = Ky * (1.0 - eye)
    term1 = jnp.sum(Kt * Lt)                       # tr(Kt Lt), symmetric
    term2 = Kt.sum() * Lt.sum() / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * jnp.dot(Kt.sum(0), Lt.sum(0))
    return (term1 + term2 - term3) / (n * (n - 3))


def hsic(X, Y, sigma2_x=None, sigma2_y=None, estimator="biased"):
    """HSIC between the rows of X (n, dx) and Y (n, dy)."""
    Kx = rbf_gram(X, sigma2_x)
    Ky = rbf_gram(Y, sigma2_y)
    if estimator == "unbiased":
        return hsic_unbiased(Kx, Ky)
    return hsic_biased(Kx, Ky)


def cka(X, Y, sigma2_x=None, sigma2_y=None):
    """Normalized (biased) HSIC, in [0, 1]."""
    Kx = rbf_gram(X, sigma2_x)
    Ky = rbf_gram(Y, sigma2_y)
    hxy = hsic_biased(Kx, Ky)
    hxx = hsic_biased(Kx, Kx)
    hyy = hsic_biased(Ky, Ky)
    # 0 by definition when either side is constant; see pairwise_head_cka
    sq = hxx * hyy
    return hxy * jnp.where(sq > 0, jax.lax.rsqrt(jnp.where(sq > 0, sq, 1.0)),
                           0.0)


def pairwise_head_cka(P, sigma2=0.0, estimator="unbiased"):
    """Mean pairwise CKA between heads of one layer's classifications.

    P: (b, h, k) per-sample per-head classification probabilities.
    sigma2 <= 0 selects the per-head median heuristic. Returns a scalar
    in ~[0, 1]: 0 = heads pairwise independent over the batch.

    The biased estimator carries a floor that independent heads alone
    produce, because with b samples two independent categorical
    variables still co-occur by chance and it reads that as dependence.
    The floor grows as the batch shrinks, and at a training batch it can
    exceed the dependence being measured, which makes the statistic
    useless as a training signal: there is nothing left to descend.
    Song et al.'s estimator removes it and is the default; `biased`
    remains for comparison."""
    b, h, k = P.shape
    dots = jnp.einsum("ihk,jhk->hij", P, P)              # (h, b, b)
    n2 = jnp.einsum("ihk,ihk->hi", P, P)
    D2 = jnp.maximum(n2[:, :, None] + n2[:, None, :] - 2.0 * dots, 0.0)
    if sigma2 and sigma2 > 0:
        s2 = jnp.full((h,), sigma2, P.dtype)
    else:
        s2 = jax.vmap(median_sigma2)(D2)                 # (h,)
    G = jnp.exp(-D2 / (2.0 * s2[:, None, None]))
    if estimator == "biased":
        Gc = center(G)                                   # per head
        # a head constant over the batch has an all-zero centered gram,
        # so its CKA with anything is 0. Taking that by definition,
        # rather than through sqrt(0), keeps the gradient finite: sqrt's
        # derivative is infinite there and meets a zero from upstream,
        # and one NaN under global-norm clipping poisons every
        # parameter. Same construction as the package's recip_norm.
        sq = jnp.sum(Gc * Gc, axis=(-2, -1))
        r = jnp.where(sq > 0, jax.lax.rsqrt(jnp.where(sq > 0, sq, 1.0)), 0.0)
        Ghat = Gc * r[:, None, None]
        S = Ghat.sum(0)
        total = 0.5 * (jnp.sum(S * S) - jnp.sum(Ghat * Ghat))
        return total / (h * (h - 1) / 2)

    # Song et al.'s estimator is bilinear in the two kernels, so scaling
    # each head by 1/sqrt(HSIC(a,a)) makes each pair's value its CKA and
    # every term a sum of outer products -- the same O(h b^2) pairwise
    # trick the biased path uses, rather than h^2 pair evaluations.
    n = b
    Kt = G * (1.0 - jnp.eye(n, dtype=G.dtype))           # zero diagonal
    s = Kt.sum((-2, -1))                                 # (h,)   1'K1
    rs = Kt.sum(-1)                                      # (h, n) K1
    c2, c3 = 1.0 / ((n - 1) * (n - 2)), 2.0 / (n - 2)

    def combine(fro, ss, rr):
        return (fro + c2 * ss - c3 * rr) / (n * (n - 3))

    self_h = combine(jnp.sum(Kt * Kt, (-2, -1)), s * s, jnp.sum(rs * rs, -1))
    # unbiased HSIC can be <= 0 for a head carrying no information; its
    # CKA is 0 by definition there, and this keeps the gradient finite
    g = jnp.where(self_h > 0,
                  jax.lax.rsqrt(jnp.where(self_h > 0, self_h, 1.0)), 0.0)
    Kn, sn, rn = Kt * g[:, None, None], s * g, rs * g[:, None]
    off = lambda tot, own: 0.5 * (tot - own)
    total = combine(
        off(jnp.sum(Kn.sum(0) ** 2), jnp.sum(Kn * Kn)),
        off(jnp.sum(sn) ** 2, jnp.sum(sn * sn)),
        off(jnp.sum(rn.sum(0) ** 2), jnp.sum(rn * rn)))
    return total / (h * (h - 1) / 2)


def _selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 512

    x = rng.normal(size=(n, 2)).astype(np.float32)
    y_ind = rng.normal(size=(n, 2)).astype(np.float32)
    y_dep = (x @ np.array([[0.6, -0.8], [0.8, 0.6]], np.float32)
             + 0.1 * rng.normal(size=(n, 2)).astype(np.float32))

    hb_i = float(hsic(jnp.asarray(x), jnp.asarray(y_ind)))
    hb_d = float(hsic(jnp.asarray(x), jnp.asarray(y_dep)))
    hu_i = float(hsic(jnp.asarray(x), jnp.asarray(y_ind),
                      estimator="unbiased"))
    hu_d = float(hsic(jnp.asarray(x), jnp.asarray(y_dep),
                      estimator="unbiased"))
    c_i = float(cka(jnp.asarray(x), jnp.asarray(y_ind)))
    c_d = float(cka(jnp.asarray(x), jnp.asarray(y_dep)))
    print(f"biased    ind {hb_i:.5f}  dep {hb_d:.5f}  (dep >> ind expected)")
    print(f"unbiased  ind {hu_i:+.5f} dep {hu_d:.5f}  (ind ~ 0 expected)")
    print(f"cka       ind {c_i:.4f}   dep {c_d:.4f}")
    assert hb_d > 5 * hb_i and hu_d > 5 * abs(hu_i) and c_d > 5 * c_i

    # head penalty: 4 independent heads vs 4 copies of one head
    b, h, k = 256, 4, 8
    logits = rng.normal(size=(b, h, k)).astype(np.float32)
    P_ind = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    P_dup = np.repeat(P_ind[:, :1], h, axis=1)
    p_i = float(pairwise_head_cka(jnp.asarray(P_ind)))
    p_d = float(pairwise_head_cka(jnp.asarray(P_dup)))
    # frozen heads: constant code -> ~0 contribution (documented behavior)
    P_frz = P_ind.copy()
    P_frz[:, :2] = np.eye(k, dtype=np.float32)[0]
    p_f = float(pairwise_head_cka(jnp.asarray(P_frz)))
    print(f"head cka  independent {p_i:.4f}  duplicated {p_d:.4f}  "
          f"half-frozen {p_f:.4f}")
    assert p_d > 0.9 and p_i < 0.1 and p_f <= p_i + 0.05

    # gradient flows
    g = jax.grad(lambda P: pairwise_head_cka(P))(jnp.asarray(P_dup))
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g).max()) > 0
    print("gradients finite and nonzero")
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
