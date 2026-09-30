# `NLinear.withL1` / `NLinearBlock.withL1`: the classifier path that
# produces the `L1_K` column and, with a nonzero `s_L1K`, the penalty.
#
# The name is misleading in a way worth pinning. It does NOT put an L1 on
# the layer's output. It puts one on `fn_gate(Ys[0])` -- the first factor
# of the product, after the gate activation. What that means depends
# entirely on `gate`:
#
#   gate="none"   `fn_gate` is identity, so the penalty lands on a raw
#                 bilinear factor. The layer's output is the PRODUCT, so
#                 the penalty is gameable: shrink the first factor, grow
#                 the second, and the output is unchanged while the
#                 reported L1 falls. `s_L1K` is disrecommended in the
#                 config with the note that it "tends to paradoxically
#                 cause exploding L1_K", and this degenerate direction is
#                 the obvious candidate.
#   gate="relu"   the mask gains exact zeros, and a zero there kills the
#                 output regardless of the second factor -- but this does
#                 NOT make the penalty well-posed. relu is positively
#                 homogeneous, so the same rescaling works verbatim. Only
#                 a saturating gate (sigmoid) breaks it, and the real
#                 missing piece is a bias INSIDE the gate: `NLinear` adds
#                 its bias after the product, so the gate path has none
#                 and is scale-free. A Gated SAE's `relu(Xc @ W + b)` has
#                 that bias, which is what makes its L1 meaningful.
#
# `BilinearBlock` -- what `DictEnc` actually instantiates at n=2 --
# inherits `NLinearBlock.withL1` unchanged, so these cover the live path.
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ontologize.layers.nlinear import (NLinear, NLinearBlock, Bilinear,
                                       BilinearBlock)

D_IN, D_OUT, H, B = 12, 6, 4, 8
KW = dict(dtype_str="float32", dtype_p_str="float32")


def build(cls, gate="none", activation="none", biased=True):
    kw = dict(d_in=D_IN, d_out=D_OUT, n=2, biased=biased, gate=gate,
              activation=activation, **KW)
    if cls in (NLinearBlock, BilinearBlock):
        kw["h"] = H
    mod = cls(**kw)
    X = jax.random.normal(jax.random.PRNGKey(0), (B, D_IN), jnp.float32)
    params = mod.init(jax.random.PRNGKey(1), X)
    return mod, params, X


@pytest.mark.parametrize("cls", [NLinear, Bilinear, NLinearBlock, BilinearBlock])
def test_l1_is_the_gate_mask_not_the_output(cls):
    """The defining property: `L1` is `l1(fn_gate(Ys[0]))`, computed on the
    gate alone, and not on the layer's output or on the second factor."""
    mod, params, X = build(cls, gate="relu")
    Y, L1 = mod.apply(params, X, method=cls.withL1)
    Ys = np.asarray(mod.apply(params, X, method=cls.nfwd))
    mask = np.maximum(Ys[0], 0.0)
    # `Sparse.l1` reduces with `batchmean`: each sample's L1 over its
    # feature axes, meaned over the batch
    want = np.abs(mask).reshape(B, -1).sum(-1).mean()
    assert float(L1) == pytest.approx(want, rel=1e-5)
    # and not the other two candidates
    assert float(L1) != pytest.approx(np.abs(np.asarray(Y)).sum(), rel=1e-3)
    assert float(L1) != pytest.approx(np.abs(Ys[1]).sum(), rel=1e-3)


@pytest.mark.parametrize("cls", [NLinear, Bilinear, NLinearBlock, BilinearBlock])
def test_output_matches_call_when_no_activation(cls):
    """`withL1`'s docstring claims it is `__call__` plus a loss. That holds
    only while `activation` is "none", which is what the live classifier
    uses (`activation_cl` defaults to "none" and no config overrides it).
    The companion below covers the activated case."""
    mod, params, X = build(cls, gate="relu", activation="none")
    Y, _ = mod.apply(params, X, method=cls.withL1)
    assert np.allclose(np.asarray(Y), np.asarray(mod.apply(params, X)),
                       atol=1e-6)


@pytest.mark.parametrize("cls", [NLinear, Bilinear, NLinearBlock, BilinearBlock])
def test_output_matches_call_under_an_activation_too(cls):
    """`withL1` applies `self.fn`, so it is `__call__` plus a loss at any
    activation and not only at "none". It did not always: `afee20c` added
    the call, and before it a nonzero `activation_cl` would silently have
    made the penalised forward differ from the evaluated one."""
    mod, params, X = build(cls, gate="relu", activation="relu")
    Y, _ = mod.apply(params, X, method=cls.withL1)
    assert np.allclose(np.asarray(Y), np.asarray(mod.apply(params, X)),
                       atol=1e-6)


@pytest.mark.parametrize("gate", ["none", "relu"])
@pytest.mark.parametrize("cls", [NLinear, NLinearBlock])
def test_every_homogeneous_gate_leaves_the_penalty_gameable(cls, gate):
    """Scaling the two factors inversely leaves the output identical while
    the reported L1 falls without bound. A penalty with a free direction
    along which it decreases at no cost to the loss it is meant to shape
    is the shape of the `s_L1K` pathology."""
    mod, params, X = build(cls, gate=gate, biased=False)
    Y0, L0 = mod.apply(params, X, method=cls.withL1)
    w = params["params"]["weight"]
    for c in (0.5, 0.1, 0.01):
        # weight axis 0 is `n`: factor 0 down by c, factor 1 up by 1/c
        scaled = w.at[0].multiply(c).at[1].multiply(1.0 / c)
        p2 = {"params": {**params["params"], "weight": scaled}}
        Y, L = mod.apply(p2, X, method=cls.withL1)
        assert np.allclose(np.asarray(Y), np.asarray(Y0), rtol=1e-4, atol=1e-5)
        assert float(L) == pytest.approx(c * float(L0), rel=1e-4)


@pytest.mark.parametrize("cls", [NLinear, NLinearBlock])
def test_a_saturating_gate_is_what_breaks_the_rescaling(cls):
    """sigmoid is not positively homogeneous, so the same trade changes the
    output and stops being free. Recorded as the one tested gate that makes
    the penalty well-posed, short of adding a bias to the gate path."""
    mod, params, X = build(cls, gate="sigmoid", biased=False)
    Y0, L0 = mod.apply(params, X, method=cls.withL1)
    w = params["params"]["weight"]
    scaled = w.at[0].multiply(0.1).at[1].multiply(10.0)
    Y, L = mod.apply({"params": {**params["params"], "weight": scaled}}, X,
                     method=cls.withL1)
    assert not np.allclose(np.asarray(Y), np.asarray(Y0), rtol=1e-3, atol=1e-4)
    assert float(L) > 0.9 * float(L0)          # nothing like the 0.1x above


@pytest.mark.parametrize("cls", [NLinear, NLinearBlock])
def test_a_zero_gate_kills_its_output_whatever_the_second_factor_does(cls):
    """True of relu and the reason it looks like a fix. It bounds what one
    entry can do, but says nothing about the global rescaling above."""
    mod, params, X = build(cls, gate="relu", biased=False)
    Ys = np.asarray(mod.apply(params, X, method=cls.nfwd))
    shut = np.maximum(Ys[0], 0.0) == 0.0
    assert shut.any(), "fixture produced no closed gates; nothing to test"
    w = params["params"]["weight"]
    loud = w.at[1].multiply(1e6)
    Y, _ = mod.apply({"params": {**params["params"], "weight": loud}}, X,
                     method=cls.withL1)
    assert float(np.abs(np.asarray(Y)[shut]).max()) == 0.0


@pytest.mark.parametrize("cls", [NLinear, NLinearBlock])
def test_the_l1_takes_no_gradient_unless_sparse_is_set(cls):
    """`Sparse.l1` stop-gradients unless `sparse=True`. `DictEnc` wires that
    from `sparse_K`, which `Hyperparams.s_loss` sets exactly when
    `s_L1K != 0` -- so the penalty is inert by construction until it is
    given a weight, and a direct instantiation of the layer sees no
    gradient at all."""
    mod, params, X = build(cls, gate="relu", biased=False)
    g = jax.grad(lambda p: mod.apply(p, X, method=cls.withL1)[1])(params)
    assert float(jnp.abs(g["params"]["weight"]).max()) == 0.0

    kw = dict(d_in=D_IN, d_out=D_OUT, n=2, biased=False, gate="relu",
              activation="none", sparse=True, **KW)
    if cls is NLinearBlock:
        kw["h"] = H
    live = cls(**kw)
    p2 = live.init(jax.random.PRNGKey(1), X)
    g2 = jax.grad(lambda p: live.apply(p, X, method=cls.withL1)[1])(p2)
    gw = np.asarray(g2["params"]["weight"])
    assert float(np.abs(gw[0]).max()) > 0.0       # reaches the gate factor
    assert float(np.abs(gw[1]).max()) == 0.0      # and never the second
