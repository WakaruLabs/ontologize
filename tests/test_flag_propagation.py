# Loss/sparsity flags must survive the whole chain Ontologizer -> DictEnc ->
# DictBlock. `Hyperparams.s_loss` derives each flag from its `s_*` weight
# being nonzero, and the flag is what un-stop-gradients the corresponding
# stat -- so a flag dropped in `Ontologizer.dictenc()` silently demotes a
# loss term to a logged-but-inert metric, with no error and no shape change.
#
# That is exactly what happened to `sparse_F`: it was never passed to
# DictEnc, so `Sparse.l1` always took its stop_gradient branch and `s_L1F`
# never applied gradient in any run. At the live SONAR config the term was
# 93% of the logged loss VALUE and 0% of its gradient. These tests pin the
# whole family rather than that one flag, since the rest were only
# accidentally correct.
import jax
import jax.numpy as jnp
import pytest

from ontologize.ontologizer import Ontologizer
from ontologize.training.config import Hyperparams

from conftest import KW, B

# Ontologizer flag -> where it has to land once it reaches the DictEnc
ON_DICTBLOCK = {"sparse_K": "classifier.sparse", "sparse_F": "dict.sparse",
                "entropy_loss": "dict.entropy_loss",
                "cossim_loss": "dict.cossim_loss",
                "bcossim_loss": "dict.bcossim_loss",
                "hmean_loss": "dict.hmean_loss"}
# these two are consumed by the DictEnc itself (both pwak stats read the
# layer input, which is a DictEnc concept), so they stop there
ON_DICTENC_ONLY = ["pwak_loss", "l2pwak_loss"]
FLAGS = list(ON_DICTBLOCK) + ON_DICTENC_ONLY


def get(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_reaches_dictenc(build, flag):
    # set one flag at a time: an all-on model would hide a flag that is
    # being sourced from the wrong neighbour
    model, params = build(**{flag: True})
    de = model.bind(params).dictencs[0]
    assert getattr(model, flag) is True
    assert getattr(de, flag) is True, f"{flag} lost between Ontologizer and DictEnc"
    for other in FLAGS:
        if other != flag:
            assert getattr(de, other) is False, f"{flag} leaked into {other}"


@pytest.mark.parametrize("flag,path", sorted(ON_DICTBLOCK.items()))
def test_flag_reaches_dictblock(build, flag, path):
    model, params = build(**{flag: True})
    de = model.bind(params).dictencs[0]
    assert get(de, path) is True, f"{flag} lost between DictEnc and {path}"


def l1f(model):
    def f(params, X):
        _, stats, _ = model.apply(params, X, temperature=0.5,
                                  method=Ontologizer.withStats)
        return stats[:, 1].sum()   # L1_F column of the DictEnc stats row
    return f


@pytest.mark.parametrize("s_L1F,penalizes", [(0.0, False), (1e-9, True)])
def test_s_L1F_is_a_penalty_not_just_a_metric(X, s_L1F, penalizes):
    # end to end through Hyperparams: a nonzero s_L1F has to produce
    # gradient, and a zero one has to produce none. The regression made the
    # first case behave like the second while still reporting the stat.
    d = KW["d_in"]
    hyper = Hyperparams(d, d, B, s_L1F=s_L1F)
    model = hyper.ontologizer(
        d, d, KW["e_dec"], KW["k"], KW["h"], KW["l"], n=2, gate="none",
        forward="resid", deepsup=True,
        dtype_str="float32", dtype_p_str="float32")
    assert model.sparse_F is penalizes
    params = model.init(jax.random.PRNGKey(0), X)
    g = jax.grad(l1f(model))(params, X)
    total = sum(float(jnp.abs(v).sum())
                for v in jax.tree_util.tree_leaves(g))
    assert (total > 0.0) is penalizes, f"s_L1F={s_L1F} gave grad {total}"
