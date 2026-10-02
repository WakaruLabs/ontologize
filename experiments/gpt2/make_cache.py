"""Build a GPT-2 residual-stream activation cache from FineWeb (Common Crawl).

Streams HuggingFaceFW/fineweb (sample-10BT), packs documents (separated by
<|endoftext|>) into fixed windows that each start with <|endoftext|> as BOS,
runs GPT-2 small, and stores the residual stream entering block `--layer`
(HF hidden_states[layer]) for positions 1..ctx-1. Position 0 is dropped: the
BOS residual has a norm far above every other position's and would dominate
the MSE.

Train windows come first. The eval windows are taken from later documents
(the token buffer is flushed at the switch), so the eval set is held out by
document, not just by row. Every model trained on this cache must train
only on the first `n_train_rows` rows.

Activations are stored normalized as float16: x' = (x - mu) / s, with mu the
per-dim mean and s = sqrt(E||x - mu||^2), so E||x'||^2 = 1 (the unit scale the
Ontologizer's SONAR configs assume). mu and s are estimated from the first
`--stat-windows` train windows and saved, so reconstructions can be mapped
back to raw activations for splicing (x = x' * s + mu).

Outputs in --out: acts.npy (rows, d) float16, tokens_train.npy and
tokens_eval.npy (windows, ctx) uint16, mean.npy, meta.json.

  uv run python experiments/gpt2/make_cache.py --out data/gpt2_l8
  uv run python experiments/gpt2/make_cache.py --train-tokens 200000 \\
      --eval-windows 64 --out data/gpt2_l8_smoke        # quick smoke test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layer", type=int, default=8,
                   help="cache the residual stream entering this block")
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--train-tokens", type=int, default=10_000_000)
    p.add_argument("--eval-windows", type=int, default=4096)
    p.add_argument("--stat-windows", type=int, default=2048,
                   help="train windows used to estimate mu and s")
    p.add_argument("--batch", type=int, default=64, help="windows per forward")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--out", default="data/gpt2_l8")
    return p.parse_args()


def windows(docs, tok, ctx, eot):
    """Yield (ctx,) token windows: BOS + ctx-1 tokens of the packed stream.
    `docs` is an iterator of texts; send() a truthy value to flush the
    buffer (used at the train/eval switch)."""
    buf = []
    for text in docs:
        buf.extend(tok(text)["input_ids"])
        buf.append(eot)
        while len(buf) >= ctx - 1:
            flush = yield np.array([eot] + buf[:ctx - 1], dtype=np.int64)
            buf = buf[ctx - 1:]
            if flush:
                buf = []
                break


@torch.no_grad()
def resid(model, ids, layer):
    out = model(ids, output_hidden_states=True)
    return out.hidden_states[layer]  # entering block `layer`


def main():
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    tok = GPT2TokenizerFast.from_pretrained(cfg.model)
    tok.model_max_length = int(1e12)  # we chunk ourselves; silence length warnings
    model = GPT2LMHeadModel.from_pretrained(cfg.model).to(dev).eval()
    eot = tok.eos_token_id
    d = model.config.n_embd
    per = cfg.ctx - 1  # rows per window (position 0 dropped)

    n_train_w = -(-cfg.train_tokens // per)
    n_eval_w = cfg.eval_windows
    n_train_rows, n_eval_rows = n_train_w * per, n_eval_w * per
    acts = np.lib.format.open_memmap(
        out / "acts.npy", mode="w+", dtype=np.float16,
        shape=(n_train_rows + n_eval_rows, d))
    tokens_train = np.zeros((n_train_w, cfg.ctx), np.uint16)
    tokens_eval = np.zeros((n_eval_w, cfg.ctx), np.uint16)

    ds = load_dataset(cfg.dataset, name=cfg.config, split="train",
                      streaming=True)
    docs = (r["text"] for r in ds)
    gen = windows(docs, tok, cfg.ctx, eot)

    mu = s = None
    stat_buf = []
    row = 0
    t0 = time.time()

    def write(A, n):
        nonlocal row
        acts[row:row + n] = ((A - mu) / s).astype(np.float16)
        row += n

    def run(n_windows, token_store, is_train):
        nonlocal mu, s
        w = 0
        flush_first = not is_train  # start eval on a fresh document
        while w < n_windows:
            batch = []
            while len(batch) < min(cfg.batch, n_windows - w):
                if flush_first:
                    gen.send(True)
                    flush_first = False
                batch.append(next(gen))
            ids = torch.from_numpy(np.stack(batch)).to(dev)
            A = resid(model, ids, cfg.layer)[:, 1:].float().cpu().numpy()
            token_store[w:w + len(batch)] = np.stack(batch)
            A = A.reshape(-1, d)
            if mu is None:
                stat_buf.append(A)
                if sum(len(a) for a in stat_buf) >= cfg.stat_windows * per \
                        or w + len(batch) >= n_windows:
                    S = np.concatenate(stat_buf)
                    mu = S.mean(0)
                    s = float(np.sqrt(((S - mu) ** 2).sum(1).mean()))
                    print(f"stats from {len(S)} rows: |mu| {np.linalg.norm(mu):.2f}"
                          f"  s {s:.3f}  max|mu| dim {int(np.abs(mu).argmax())}")
                    for a in stat_buf:
                        write(a, len(a))
                    stat_buf.clear()
            else:
                write(A, len(A))
            w += len(batch)
            if (w // cfg.batch) % 50 == 0:
                rate = row / (time.time() - t0)
                print(f"{'train' if is_train else 'eval'} {w}/{n_windows} windows"
                      f"  {row} rows  {rate:,.0f} rows/s", flush=True)

    run(n_train_w, tokens_train, True)
    run(n_eval_w, tokens_eval, False)
    assert row == n_train_rows + n_eval_rows, (row, n_train_rows, n_eval_rows)
    acts.flush()

    np.save(out / "tokens_train.npy", tokens_train)
    np.save(out / "tokens_eval.npy", tokens_eval)
    np.save(out / "mean.npy", mu.astype(np.float32))
    meta = {"model": cfg.model, "layer": cfg.layer, "ctx": cfg.ctx, "d": d,
            "hook": f"hidden_states[{cfg.layer}] (entering block {cfg.layer})",
            "positions": f"1..{cfg.ctx - 1} (position 0 = BOS dropped)",
            "dataset": f"{cfg.dataset}/{cfg.config}",
            "normalization": "x' = (x - mean.npy) / scale; E||x'||^2 = 1",
            "scale": s, "n_train_rows": n_train_rows, "n_eval_rows": n_eval_rows,
            "n_train_windows": n_train_w, "n_eval_windows": n_eval_w,
            "dtype": "float16"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
    sys.stdout.flush()
    # HF streaming leaves non-daemon threads that hang interpreter shutdown
    os._exit(0)


if __name__ == "__main__":
    main()
