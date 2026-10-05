"""`Grouped`: an SAE whose latents compete within groups rather than
globally -- the structural-ablation rung that adds Ontologizer-style heads
to a standard SAE, and nothing else.

The `m` latents are partitioned into `groups` contiguous groups of
`m // groups`. Under `group_fn="top1"` each group's winning latent keeps
its ReLU magnitude and the rest are zero, so L0 is at most `groups`; a
group whose winner is negative stays silent, which an Ontologizer head
cannot. Under `"softmax"` each group emits a distribution over its members:
dense, summing to one per group, and coefficient-bounded, so the decoder
rows are not held at unit norm and must carry magnitude themselves. That
form is one `DictEnc`-like layer without the bilinear classifier, the
residual stack or the non-negative dictionary; with a hard winner it is a
one-layer hard-code Ontologizer. Neither is an SAE in any established
sense, and results report them as simplified Ontologizer variants.

`enc` is "linear" or "bilinear" (the gated encoder sets its own sparsity
and does not compose with groups); `prefixes` composes as in `SAE`.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .sae import SAE

GROUP_FNS = ("top1", "softmax")


def group_top1(pre: Float[Array, "... m"], groups: int
               ) -> Float[Array, "... m"]:
    """Per group, the winning latent keeps its ReLU magnitude; a group
    whose winner is not positive stays silent."""
    g = pre.reshape(*pre.shape[:-1], groups, -1)
    top = g.max(-1, keepdims=True)
    return jnp.where((g >= top) & (g > 0), g, 0.0).reshape(pre.shape)


def group_softmax(pre: Float[Array, "... m"], groups: int
                  ) -> Float[Array, "... m"]:
    """Per group, a softmax over its members."""
    g = pre.reshape(*pre.shape[:-1], groups, -1)
    return jax.nn.softmax(g, -1).reshape(pre.shape)


class Grouped(SAE):
    """`SAE` with per-group competition over `groups` contiguous groups of
    latents (`group_fn` "top1" or "softmax") in place of the global top-k,
    which it requires off (`topk=0`)."""
    topk: int = 0
    groups: int = 0
    group_fn: str = "top1"

    def check(self):
        super().check()
        if self.enc == "gated":
            raise ValueError("the gated encoder sets its own sparsity and "
                             "does not compose with groups")
        if self.topk:
            raise ValueError("groups replace the global top-k; needs topk=0")
        if self.groups < 1 or self.m % self.groups:
            raise ValueError(f"m={self.m} must be a positive multiple of "
                             f"groups={self.groups}")
        if self.group_fn not in GROUP_FNS:
            raise ValueError(f"group_fn must be one of {GROUP_FNS}; got "
                             f"{self.group_fn!r}")

    @property
    def unit_rows(self) -> bool:
        """Softmax coefficients are bounded, so the rows must carry
        magnitude and the L1 norm-gaming loophole does not exist."""
        return self.group_fn != "softmax"

    def activate(self, pre: Float[Array, "... m"]) -> Float[Array, "... m"]:
        if self.group_fn == "softmax":
            return group_softmax(pre, self.groups)
        return group_top1(pre, self.groups)
