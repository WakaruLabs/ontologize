"""Shared pieces for the GPT-2 experiments: cache access, FVU, and the
splice evaluator (next-token cross-entropy with the layer's residual stream
replaced by a reconstruction).

Cache layout (see make_cache.py): acts.npy rows [0, n_train_rows) are train,
the rest are eval (held out by document). Rows are normalized activations
x' = (x - mu) / s; the splice maps reconstructions back with x = x' * s + mu.
"""
import json
from pathlib import Path

import numpy as np


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.acts = np.load(self.path / "acts.npy", mmap_mode="r")
        self.n_train = self.meta["n_train_rows"]
        self.n_eval = self.meta["n_eval_rows"]
        self.d = self.meta["d"]
        self.mu = np.load(self.path / "mean.npy")
        self.scale = self.meta["scale"]

    def eval_rows(self, n=None):
        n = self.n_eval if n is None else min(n, self.n_eval)
        return np.asarray(self.acts[self.n_train:self.n_train + n], np.float32)

    def train_batches(self, b, epochs, seed=0, block=65536, blocks_per_buffer=8):
        """Yield float32 (b, d) batches of train rows, reshuffled each epoch.

        Random row reads from the memmap are IO-bound once the cache no longer
        fits in the page cache, so rows are read in contiguous blocks: each
        epoch visits the blocks in random order, `blocks_per_buffer` blocks at
        a time are loaded into a buffer, and that buffer is shuffled at the row
        level. A background thread prefetches the next buffer."""
        import queue
        import threading
        n_blocks = self.n_train // block          # the ragged tail is dropped
        q = queue.Queue(maxsize=2)

        def producer():
            rng = np.random.default_rng(seed)
            for _ in range(epochs):
                order = rng.permutation(n_blocks)
                for j in range(0, n_blocks, blocks_per_buffer):
                    ids = np.sort(order[j:j + blocks_per_buffer])
                    buf = np.concatenate([np.asarray(self.acts[i * block:(i + 1) * block])
                                          for i in ids])
                    q.put(buf[rng.permutation(len(buf))])
            q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        while (buf := q.get()) is not None:
            for i in range(0, len(buf) - b + 1, b):
                yield buf[i:i + b].astype(np.float32)

    def steps_per_epoch(self, b, block=65536, blocks_per_buffer=8):
        n_rows = (self.n_train // block) * block
        per_buf = blocks_per_buffer * block
        full, rest = divmod(n_rows, per_buf)
        return full * (per_buf // b) + rest // b


def fvu(X, Y):
    """Fraction of variance unexplained, relative to the eval set's own mean."""
    return float(((X - Y) ** 2).sum() / ((X - X.mean(0)) ** 2).sum())


class Splicer:
    """Next-token CE of GPT-2 on the cache's eval windows, with the residual
    stream entering block `layer` (positions 1..ctx-1) replaced.

    `recon_fn` maps normalized activations (N, d) float32 -> normalized
    reconstructions (N, d). Position 0 (BOS) is never replaced, and its own
    prediction (unaffected by the splice) is excluded from the loss."""

    def __init__(self, cache: Cache, n_windows=512, batch=32, device="cuda"):
        import torch
        from transformers import GPT2LMHeadModel
        self.torch = torch
        self.cache = cache
        self.dev = torch.device(device)
        self.model = GPT2LMHeadModel.from_pretrained(
            cache.meta["model"]).to(self.dev).eval()
        self.layer = cache.meta["layer"]
        toks = np.load(cache.path / "tokens_eval.npy").astype(np.int64)
        self.tokens = toks[:n_windows]
        self.batch = batch
        self.mu = torch.tensor(cache.mu, device=self.dev)
        self.s = cache.scale
        self._replace = None
        block = self.model.transformer.h[self.layer - 1]
        block.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if self._replace is None:
            return output
        h = output[0]
        new = self._replace(h[:, 1:])
        h = self.torch.cat([h[:, :1], new.to(h.dtype)], dim=1)
        return (h,) + tuple(output[1:])

    def _ce(self):
        torch = self.torch
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(self.tokens), self.batch):
                ids = torch.from_numpy(self.tokens[i:i + self.batch]).to(self.dev)
                logits = self.model(ids).logits[:, 1:-1]
                tgt = ids[:, 2:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    tgt.reshape(-1), reduction="sum")
                tot += float(loss)
                n += tgt.numel()
        return tot / n

    def evaluate(self, recon_fns: dict):
        """Returns {'clean', 'mean_ablate', <name>: ce, <name>_recovered: frac}."""
        torch = self.torch
        self._replace = None
        out = {"clean": self._ce()}
        self._replace = lambda h: self.mu.expand_as(h)
        out["mean_ablate"] = self._ce()
        for name, fn in recon_fns.items():
            def rep(h, fn=fn):
                B, P, D = h.shape
                x = ((h.float() - self.mu) / self.s).reshape(-1, D).cpu().numpy()
                y = np.asarray(fn(x), np.float32)
                return torch.from_numpy(y).to(self.dev).reshape(B, P, D) * self.s + self.mu
            self._replace = rep
            ce = self._ce()
            out[name] = ce
            out[name + "_recovered"] = (out["mean_ablate"] - ce) / (
                out["mean_ablate"] - out["clean"])
        self._replace = None
        return out
