"""GPT-2 small at the `gpt2_l8` dictionaries' site: the model, an editing
hook, held-out windows from the harvest, the equal-length steps `rescale.py`
compares, and next-token metrics.

The site is the output of `transformer.h[8]` (`blocks.8.resid_post`), the
rows `encode_acts.py` harvested. Position 0 is the prepended BOS, which the
harvest drops, so every edit here leaves it alone and no metric scores the
prediction it makes.
"""
import os

# jax (the dictionaries) and torch (GPT-2) share the GPU, as in encode_acts.py
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Int

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "activations" / "gpt2_l8.npy"
DOCS = CACHE.with_name("gpt2_l8.docs.jsonl")
HOLDOUT = 32768        # sonar.py's holdout; gpt2_l8.py snaps it to whole documents
LAYER = 8
CTX = 128              # tokens per harvested window, BOS included
OUT = ROOT / "data" / "out" / "gpt2_l8" / "nonlinear_steering"

Tensor = torch.Tensor


def device_for(name: str) -> torch.device:
    """`auto` picks CUDA when it is available."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_gpt2(device: torch.device):
    """GPT-2 small in float32, the dtype the harvest ran in, frozen."""
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@contextmanager
def site_edit(model, fn: Callable[[Float[Tensor, "b t d"]],
                                  Float[Tensor, "b t d"]]) -> Iterator[None]:
    """Replace the site's output at positions >= 1 by `fn` of it; position 0
    passes through. Blocks return a tuple in some transformers versions and
    a tensor in others, so both are handled."""
    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = torch.cat([h[:, :1], fn(h[:, 1:])], dim=1)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handle = model.transformer.h[LAYER].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture(model, tokens: Int[Tensor, "b t"]) -> Float[Tensor, "b t1 d"]:
    """The site's output at positions >= 1, with no edit."""
    got: List[Tensor] = []

    def keep(h: Float[Tensor, "b t1 d"]) -> Float[Tensor, "b t1 d"]:
        got.append(h.detach())
        return h

    with site_edit(model, keep), torch.no_grad():
        model(tokens, use_cache=False)
    return got[0]


def tail_windows(full_only: bool = True
                 ) -> Tuple[Int[np.ndarray, "n ctx"], Int[np.ndarray, "n"]]:
    """Token windows (BOS first) of the documents in the cache's held-out
    tail, the split every model trained on `gpt2_l8` leaves out, and each
    window's first cache row. With `full_only`, only documents that fill
    the whole context, so a batch needs no padding."""
    from ontologize.data.loaders import doc_holdout
    n = len(np.load(CACHE, mmap_mode="r"))
    first = n - doc_holdout(CACHE, HOLDOUT)
    toks, rows = [], []
    with open(DOCS) as f:
        for line in f:
            d = json.loads(line)
            if d["row0"] < first or (full_only and len(d["tokens"]) != CTX):
                continue
            toks.append(d["tokens"])
            rows.append(d["row0"])
    return np.asarray(toks, np.int64), np.asarray(rows, np.int64)


def parity(model, toks: Int[np.ndarray, "n ctx"],
           rows: Int[np.ndarray, "n"], device: torch.device,
           n: int = 4) -> float:
    """Largest relative error, over the first `n` windows, between the
    site's output here and the rows the harvest stored for them. Small
    means this hook reads the stream the dictionaries were trained on."""
    X = np.load(CACHE, mmap_mode="r")
    H = capture(model, torch.as_tensor(toks[:n], device=device))
    H = H.float().cpu().numpy()
    err = 0.0
    for j in range(min(n, len(toks))):
        ref = np.asarray(X[rows[j]:rows[j] + CTX - 1], np.float32)
        err = max(err, float(np.linalg.norm(H[j] - ref) / np.linalg.norm(ref)))
    return err


def orth_step(h: Float[Tensor, "... d"], size: float,
              g: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
    """A step of length `size * |h|` along the part of `g` orthogonal to
    `h`: the same length as rescaling `h` by `1 +- size`, with the length of
    `h` unchanged to first order and only its direction moving."""
    hn = h / h.norm(dim=-1, keepdim=True)
    gp = g - (g * hn).sum(-1, keepdim=True) * hn
    return size * h.norm(dim=-1, keepdim=True) * gp / gp.norm(dim=-1, keepdim=True)


def next_token_stats(logits_ref: Float[Tensor, "b t v"],
                     logits: Float[Tensor, "b t v"],
                     tokens: Int[Tensor, "b t"]
                     ) -> Tuple[Float[Tensor, "b t2"], Float[Tensor, "b t2"],
                                Float[Tensor, "b t2"], Bool[Tensor, "b t2"]]:
    """For the predictions positions 1..T-2 make of tokens 2..T-1:
    KL(reference || edited) over the vocabulary, the reference and edited
    cross-entropies of the actual next token, and whether the two agree on
    the top token. Position 0's prediction is excluded, as no edit at
    positions >= 1 can reach it."""
    lr = torch.log_softmax(logits_ref[:, 1:-1].float(), -1)
    le = torch.log_softmax(logits[:, 1:-1].float(), -1)
    tgt = tokens[:, 2:, None]
    kl = (lr.exp() * (lr - le)).sum(-1)
    return (kl, -lr.gather(-1, tgt)[..., 0], -le.gather(-1, tgt)[..., 0],
            lr.argmax(-1) == le.argmax(-1))


def run_temperature(run: Path) -> float:
    """The softmax temperature an Ontologizer run trained with, from the
    first `log.jsonl` record; 1.0 if it is not recorded."""
    log = Path(run) / "log.jsonl"
    if log.exists():
        with open(log) as f:
            rec = json.loads(f.readline())
        return float(rec.get("hyper", {}).get("temperature", 1.0))
    return 1.0
