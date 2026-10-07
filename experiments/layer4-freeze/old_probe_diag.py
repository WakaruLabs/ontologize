"""`freeze_diag.py` with the probe it used before it called
`autointerp.onto_probe`, for auditing results that probe produced (see
notes.md). Same arguments and outputs as `freeze_diag.py`.

That probe classified each layer's raw next-layer input and added
`combine(hfwd(P))` without the layer gain: no constant coordinate at
layer 0, no gain-shape split, no router or fibers. It is exact on models
with none of those (`resid_nc`, `resid_nc_hm`) and wrong past layer 0 on
a gain-shape model, where the classifier reads the shrinking raw
residual with the constant coordinate and assigns every sample alike.
Do not use it to diagnose a run; use `freeze_diag.py`.

  uv run python experiments/layer4-freeze/old_probe_diag.py \\
      --ckpt data/out/sonar/hsic_check/hsic \\
      --out data/out/sonar/hsic_check/diag_old
"""
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
import freeze_diag  # noqa: E402  (puts the repo root on sys.path)


def old_make_acts(model, temperature):
    """Jitted (params, X) -> per-layer classifications (b, l, h, k) from
    the former probe."""
    import jax
    import jax.numpy as jnp
    import einops

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        Ein = E
        Ps = []
        for i, de in enumerate(module.dictencs):
            P = de.dict.cluster(de.classifier(Ein), temperature)
            Ps.append(P)
            R = R + de.dict.combine(de.dict.hfwd(P))
            if i < module.l - 1:
                Ein = module.nextinput(
                    X, R, einops.rearrange(P, "... h k -> ... (h k)"))
        return jnp.stack(Ps, 1)  # (b, l, h, k)

    @jax.jit
    def acts(params, X):
        return model.apply(params, X, method=probe)

    return acts


if __name__ == "__main__":
    freeze_diag.make_acts = old_make_acts
    freeze_diag.main()
