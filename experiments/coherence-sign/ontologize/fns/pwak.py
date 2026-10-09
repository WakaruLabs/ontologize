"""PWAK consensus KL: pull classifications toward neighbourhood consensus
instead of toward per-sample argmax. Designed as a label hardener; the
MEASURED BEHAVIOUR section below found that in this KL direction it
softens instead.

Temperature annealing hardens by saturating the softmax, and a saturated
softmax has no gradient -- that is the mechanism behind the frozen-head
pathology this codebase keeps patching (resid_nc layer 4 froze 22/32 heads
after step ~211k; winner dropout, sd_K annealing, and the s_Hm bonus are all
splints on the same break). Temperature also amplifies whatever the model
already believes about each sample *in isolation*: there is no external
check, so early mistakes get locked in.

The DeePWAK construction (kewiechecki/DeePWAK) was built to give a better
hardening operator. Within a head, the batch's classifications P are a soft partition
matrix PP^T -- exactly `partitionmat` with the indicators relaxed. Gate the
input affinity with it, zero the diagonal, row-normalize (the WAK function),
and diffuse the labels s steps:

    G = wak((D . PP^T) - diag),    P^(s) = G^s P

Each sample's classification becomes the consensus of its co-clustered
semantic neighbours. Two properties temperature lacks:

  1. The logit scale is untouched, so the softmax never saturates and the
     classifier stays in its responsive regime at full hardness.
  2. Zeroing the diagonal makes the target *self-supervised* rather than
     self-fulfilling: a sample is predicted from its neighbours and not from
     itself -- noise2self's J-invariance. Strictly so only at s=1: G has a
     zero diagonal but G^2 does not, so even-length walks readmit the
     self-edge (damped) at the s=2-4 actually used. The target is mostly,
     not purely, information the sample did not already have.

P^(s) also stays in the convex hull of the neighbourhood's classifications,
so it cannot manufacture confidence the data does not support.

*** MEASURED BEHAVIOUR, and it contradicts the name of this module. ***
The 2x2 factorial (run where this code was developed; its outputs are not
in this repository) says this term does NOT harden. It softens: +0.32 bits of classification entropy in both
temperature rows, dead consistent, and KL_pwak never converges -- it
climbs 0.13 -> 0.47 bits and plateaus, i.e. the model holds a permanent
disagreement with its consensus instead of reaching a fixed point.

The cause is the target, not the KL direction below. KL(T || P) and
KL(P || T) are both minimized at P = T, so swapping the arguments in
`pwak_kl` moves the same fixed point. T is a convex combination of the
neighbourhood's classifications (above), so wherever the neighbourhood
disagrees it is flatter than a confident sample's own P, and any pull
toward it softens P. Hardening needs a target sharper than the consensus,
not another KL direction.

What it DOES do, unclaimed in advance: it regularizes. Training MSE +1.4%,
held-out MSE -0.3%, both rows; held-out slow-mode alignment 0.813 vs 0.754
for the control. Label degeneracy is row-dependent: 0.155 vs 0.211 in the
hard-T row, REVERSED in the soft-T row (0.272 vs 0.232, pwak worse) -- the
one endpoint the designed config loses to its control. So it earns its
keep as a generalization term, not as a hardening term.

Untested: the frozen-head claim that motivated the whole thing. All four arms
had zero frozen heads and zero dead tags, controls included -- h=5, k=10 with
s_Hm active never gets sick. That needs h=32, five layers, s_Hm off.

Used as a stop-gradiented target with a KL pull (not in the forward path):
per-sample inference semantics stay intact for `withArgs` and the
intervention REPL, and no gradient crosses the batch. The whole
computation is gated on `DictBlock.pwak_loss` (static): the schedule
passes `pwak_s` traced, so without that gate every run -- pwak or not --
would build the (h, b, b) graph as a loop constant XLA cannot prune.

CAVEAT, recorded here because it is easy to forget: training against a
similarity operator makes "the dictionary's cells align with the slow modes
of the discourse operator" true by construction. The crossref experiments
this code was developed alongside (not in this repository, whose
`writeup/findings.tex` is a different document) are evidence only because
nothing in training ever mentioned a kernel. Any run using this needs a
different null -- score against an affinity the training never saw.
"""
import jax
import jax.numpy as jnp

from jaxtyping import Array, Float

def wak(G: Float[Array, "... n n"]) -> Float[Array, "... n n"]:
    """Weighted affinity kernel: row sums normalized to 1, NaNs zeroed.
    DeePWAK's WAK function, verbatim in behaviour."""
    W = G.sum(-1, keepdims=True)
    return jnp.where(W > 0, G / jnp.where(W > 0, W, 1.0), 0.0)

def affinity(E: Float[Array, "b d"], tau: float) -> Float[Array, "b b"]:
    """Heat-kernel affinity over unit-normalized rows, self-edge removed.
    exp((<e_i,e_j> - 1)/tau); the diagonal is dropped so every estimate comes
    from the complement of the point being estimated (noise2self)."""
    U = E / (jnp.linalg.norm(E, axis=-1, keepdims=True)
             + jnp.finfo(E.dtype).eps)
    D = jnp.exp((U @ U.T - 1.0) / tau)
    return D * (1.0 - jnp.eye(D.shape[0], dtype=D.dtype))

def diffuse(P: Float[Array, "b h k"], E: Float[Array, "b d"],
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

    It softens rather than hardens because the target is flatter than a
    confident P, not because of the argument order: swapping T and P below
    leaves the minimizer at P = T (see the module docstring)."""
    P = P_0.reshape(-1, P_0.shape[-2], P_0.shape[-1])
    E = E.reshape(-1, E.shape[-1])
    if not isloss:
        P = jax.lax.stop_gradient(P)
    T = jax.lax.stop_gradient(diffuse(jax.lax.stop_gradient(P), E, s, tau))
    eps = jnp.finfo(P.dtype).eps
    return (T * (jnp.log2(T + eps) - jnp.log2(P + eps))).sum(-1).mean()
