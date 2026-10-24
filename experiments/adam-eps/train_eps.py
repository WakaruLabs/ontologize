"""Train an arm of `experiments/ste-arm` with Adam's eps set by flag.

Adam steps by m_hat / (sqrt(v_hat) + eps), so a parameter whose gradient
scale sqrt(v_hat) is well below eps takes a step of about
sqrt(v_hat) / eps of the full one, and eps rather than the learning rate
sets how fast it trains. `--adam-eps` hands eps to optax through a subclass
of the `Hyperparams` class `train_ste.py`'s flags choose (`setup`'s `hp=`);
every other flag is `train_ste.py`'s, so an arm and its control differ in
eps alone.

    uv run python experiments/adam-eps/train_eps.py --base gpt2_l8 \\
        --dict-init-scale 1.0 --epochs 2 --adam-eps 1e-15 \\
        --out data/out/gpt2_l8/adam_eps/ste_h76_eps1e-15
"""
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ste-arm"))
import train_ste  # noqa: E402
from ontologize.training.config import Hyperparams  # noqa: E402


def with_eps(base: type) -> type:
    """`base` with Adam's eps as the field `adam_eps` (optax's default,
    1e-8), which `opt` hands to optax. Every `opt` in the chain passes its
    keyword arguments on to `optax.adam` or `optax.adamw`."""
    @dataclass
    class EpsHyperparams(base):
        adam_eps: float = 1e-8

        def opt(self, *args, **kwargs):
            return super().opt(*args, eps=self.adam_eps, **kwargs)

    EpsHyperparams.__name__ = EpsHyperparams.__qualname__ = f"Eps{base.__name__}"
    return EpsHyperparams


EpsHyperparams = with_eps(Hyperparams)


def flags_class(cfg: argparse.Namespace) -> type:
    """The `Hyperparams` class `train_ste.setup` chooses for these flags,
    which `hp=` has to subclass (setup asserts it). The flags that pick one
    are mutually exclusive there."""
    if cfg.s_hsic_heads:
        from hsic_ste import HSICSteHyperparams as HP
    elif cfg.zca:
        from whiten_enc import WhitenEncHyperparams as HP
    elif cfg.train_layer >= 0 or cfg.init_from:
        from layerft import LayerFTHyperparams as HP
    elif cfg.dict_init_scale or cfg.dict_wd:
        from dictwd import DictWDHyperparams as HP
    else:
        HP = Hyperparams
    return HP


def parser() -> argparse.ArgumentParser:
    p = train_ste.parser()
    p.add_argument("--adam-eps", type=float, default=1e-8,
                   help="Adam's eps (optax's default, 1e-8, is the control)")
    return p


def main() -> None:
    cfg = train_ste.finish(parser().parse_args())
    run = train_ste.setup(cfg, hp=with_eps(flags_class(cfg)),
                          hp_extra=dict(adam_eps=cfg.adam_eps))
    train_ste.train(*run)


if __name__ == "__main__":
    main()
