"""PWAK consensus KL: pull classifications toward neighbourhood consensus
instead of toward per-sample argmax.

Temperature annealing hardens by saturating the softmax, and a saturated
softmax has no gradient -- that is the mechanism behind the frozen-head
pathology this codebase keeps patching (resid_nc layer 4 froze 22/32 heads
after step ~211k; winner dropout, sd_K annealing, and the s_Hm bonus are all
splints on the same break). Temperature also amplifies whatever the model
already believes about each sample *in isolation*: there is no external
check, so early mistakes get locked in.

The construction comes from DeePWAK (Deep learning of a Partitioned
Weighted Affinity Kernel; unpublished), which builds on noise2self
(Batson & Royer 2019) and DEWAKSS (Tjarnberg et al. 2021). There a
partitioner's soft labels K define a partition mask wak(K^T K) over the
batch, the batch's embeddings are diffused through it and decoded, and the
reconstruction error trains the partition: each sample is predicted from
its co-clustered neighbours.

There, the diffused embeddings sit in the reconstruction path; the
hope in bringing it here was that increasing the diffusion depth s might
force the model to learn sharper distinctions. DeePWAK never tested this;
it chose s by grid search. That hypothesis presupposes
the diffusion is used to reconstruct something, which is what `pwak_l2`
does -- it scores the partition by how well a sample's own layer input is
predicted by its neighbours' -- while `pwak_kl` uses the same graph only
to build a KL target, where the mechanism is absent by construction (see
BEHAVIOUR). The construction itself: within a head, the batch's
classifications P are a soft partition
matrix PP^T -- exactly `partitionmat` with the indicators relaxed. Gate the
input affinity with it, zero the diagonal, row-normalize (the WAK function),
and diffuse the labels s steps:

    G = wak((D . PP^T) - diag),    P^(s) = G^s P

Each sample's classification becomes the consensus of its co-clustered
semantic neighbours. Two properties temperature lacks:

  1. The logit scale is untouched, so the softmax never saturates and the
     classifier stays in its responsive regime at full hardness.
  2. Zeroing the diagonal keeps a sample's own classification out of its
     target: it is predicted from its neighbours, not from itself. This is
     weaker than noise2self's J-invariance, because the sample's own input
     still sets its weights, through its classification (the gate) and
     through the affinity. For a hard per-head code it is no target at all:
     a sample's neighbours are its cell-mates, which carry its own one-hot,
     so the target equals P and the KL is identically 0. And G has a zero
     diagonal but G^2 does not, so even-length walks readmit the self-edge
     (damped) at the s=2-4 actually used; DEWAKSS re-zeroes the diagonal
     at every step instead.

P^(s) also stays in the convex hull of the neighbourhood's classifications,
so it cannot manufacture confidence the data does not support.

*** BEHAVIOUR: unmeasured, and probably not what the name says. ***
No controlled measurement of either term exists. One informal run
suggested `pwak_kl` softens classifications rather than hardening them, and
the KL direction below predicts exactly that. KL(T || P) is mode-COVERING:
minimizing it forces P to put mass wherever T has mass, so a disagreeing
neighbourhood (whose consensus is flatter than the sample's own belief)
pulls the sample toward softness. The mode-SEEKING KL(P || T) is the one
that concentrates mass on whichever tag the neighbourhood supports --
swapping the arguments in the return statement of `pwak_kl` gives it. But
note the deeper mismatch recorded at the top: the hardening-by-deeper-
diffusion hypothesis assumed the diffused labels feed the reconstruction
of F, as the diffused embeddings do in DeePWAK. As a KL regularizer,
neither direction implements that mechanism.

Also untested: the frozen-head claim that motivated the whole thing. A
test needs a regime where heads actually freeze (h=32, five layers, s_Hm
off), with the frozen-head count compared against a control.

Used as a stop-gradiented target with a KL pull (not in the forward path):
per-sample inference semantics stay intact for `withArgs` and the
intervention REPL, and no gradient crosses the batch. The whole
computation is gated on `DictBlock.pwak_loss` (static): the schedule
passes `pwak_s` traced, so without that gate every run -- pwak or not --
would build the (h, b, b) graph as a loop constant XLA cannot prune.

CAVEAT, recorded here because it is easy to forget: training against a
similarity operator makes "the dictionary's cells align with that
operator's slow modes" true by construction. Any run using this needs a
different null for such alignments -- score against an affinity the
training never saw.
"""
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float

from .loss import l2

def wak(G: Float[Array, "... n n"]) -> Float[Array, "... n n"]:
    """Weighted affinity kernel: row sums normalized to 1, all-zero rows
    left at zero rather than NaN. This is the kernel from DEWAKSS, which uses
    it to impute a cell's count for a gene as the weighted average of that
    gene's counts in neighbouring cells; row normalization is what makes
    `G @ E` that average. The diagonal is untouched here; `affinity`
    zeroes it."""
    W = G.sum(-1, keepdims=True)
    return jnp.where(W > 0, G / jnp.where(W > 0, W, 1.0), 0.0)

def affinity(E: Float[Array, "b d"], tau: float) -> Float[Array, "b b"]:
    """Heat-kernel affinity over unit-normalized rows, self-edge removed.
    exp((<e_i,e_j> - 1)/tau); the diagonal is dropped so no estimate uses
    the value of the point being estimated, as in noise2self. The point
    still sets its own row of weights, so this is leave-one-out, not
    J-invariant."""
    U = E / (jnp.linalg.norm(E, axis=-1, keepdims=True)
             + jnp.finfo(E.dtype).eps)
    D = jnp.exp((U @ U.T - 1.0) / tau)
    return D * (1.0 - jnp.eye(D.shape[0], dtype=D.dtype))

def diffuse(P: Float[Array, "b h k"], E: Float[Array, "b d"],
            s: int, tau: float = 0.2) -> Float[Array, "b d"]:
    """s steps of partition-gated diffusion of `E` itself: each sample's
    value becomes the graph-weighted average of its neighbours' values.

    The gate is the JOINT co-assignment `sum_h sum_c p_ihc p_jhc` -- the
    expected number of heads that agree -- so there is one (b, b)
    transition matrix rather than one per head. Per-head gating would add
    nothing here: the dictionary is linear in P and the two contractions
    commute, so diffusing P per head and running the dictionary is
    already exactly diffusing each head's features (`diffuse_kl` + `fwd`).
    The joint graph is also h times cheaper.

    No simplex renormalization -- `E` is input space, not a distribution,
    and `wak` already makes G row-stochastic, so a convex average of E
    rows is what comes out. `s` is a Python int and the loop is unrolled,
    which keeps this reverse-differentiable so it can carry a loss term
    (`jax.lax.fori_loop` with a traced bound lowers to a `while_loop`,
    which cannot be differentiated in reverse mode)."""
    G = wak(jnp.einsum("ihc,jhc->ij", P, P) * affinity(E, tau))
    for _ in range(s):
        E = G @ E
    return E

def pwak_l2(P: Float[Array, "... h k"], E: Float[Array, "... d"],
            s: int, tau: float = 0.2) -> Float[Array, ""]:
    """Leave-one-out reconstruction error of a layer's input from its
    partition-gated neighbourhood (noise2self-style; see `affinity`), as a
    fraction of the batch's own
    spread: `l2(E, G^s E) / l2(E, mean(E))`. Reads as a fraction of
    variance: 0 when neighbours predict a sample exactly, 1 when the
    graph does no better than the batch mean. Exactly 0 at s=0 (diffusion
    is the identity), so the term is inert until the schedule ramps.

    `E` is stop-gradiented, which is the whole point: it is the external
    reference the partition is scored against, so the only gradient path
    is through the gate `PP^T`. The term cannot be reduced by changing
    what is being predicted -- only by co-assigning samples that actually
    predict each other -- and that pressure grows with `s`, because a
    leaky partition's walk escapes its block while a tight one does not.
    This is the mechanism the module docstring's hardening hypothesis
    assumed and the KL form does not implement. Because the gate is
    computed from the same input, any input-dependent partition, trained
    or random, co-assigns samples that predict each other somewhat, even
    on structureless data; a null for this score has to be an
    input-dependent partition too.

    Scoring against the input rather than against the model's own output
    is what keeps it honest. `l2(F, diffuse(F))` is minimized exactly by
    a constant `F`, so its cheapest direction is collapsing the code --
    the degeneracy this codebase already fights. Here there is no such
    direction. Unlike `pwak_kl` there is also no `log`, so exact-zero
    tags cost nothing and this is compatible with `select="top<k>"`."""
    E = jax.lax.stop_gradient(E.reshape(-1, E.shape[-1]))
    P = P.reshape(-1, P.shape[-2], P.shape[-1])
    spread = l2(E, E.mean(0, keepdims=True))
    return l2(E, diffuse(P, E, s, tau)) / (spread + jnp.finfo(E.dtype).eps)

def diffuse_kl(P: Float[Array, "b h k"], E: Float[Array, "b d"],
            s: int, tau: float = 0.2) -> Float[Array, "b h k"]:
    """s steps of partition-gated diffusion of the classifications.
    `s` may be traced (the schedule ramps it); the loop is a `fori_loop`,
    which is fine because callers stop-gradient the result."""
    D = affinity(E, tau)
    # soft co-assignment per head: (PP^T)_ij = sum_c p_ic p_jc
    G = wak(jnp.einsum("ihc,jhc->hij", P, P) * D)
    # renormalize each step: G is row-stochastic so the simplex is preserved
    # in exact arithmetic, but TF32 matmuls drift ~1e-3 per step and the KL
    # target has to be an actual distribution
    def step(_, Q):
        Q = jnp.einsum("hij,jhk->ihk", G, Q)
        return Q / (Q.sum(-1, keepdims=True) + jnp.finfo(Q.dtype).eps)
    return jax.lax.fori_loop(0, s, step, P)

def pwak_kl(P_0: Float[Array, "... h k"], E: Float[Array, "... d"],
            s: int, tau: float = 0.2, isloss: bool = False
            ) -> Float[Array, ""]:
    """KL(consensus target || classification) in bits, mean over samples and
    heads. Exactly 0 at s=0 (the target is P itself), so the term is inert
    until the schedule ramps -- diffusing while partitions are still noise
    would lock in noise, the standard self-training death spiral.

    NOTE the argument order is the mode-COVERING direction, which is why this
    softens rather than hardens (see the module docstring). Swapping T and P
    in the expression below gives the mode-seeking variant."""
    P = P_0.reshape(-1, P_0.shape[-2], P_0.shape[-1])
    E = E.reshape(-1, E.shape[-1])
    if not isloss:
        P = jax.lax.stop_gradient(P)
    T = jax.lax.stop_gradient(diffuse_kl(jax.lax.stop_gradient(P), E, s, tau))
    eps = jnp.finfo(P.dtype).eps
    return (T * (jnp.log2(T + eps) - jnp.log2(P + eps))).sum(-1).mean()
