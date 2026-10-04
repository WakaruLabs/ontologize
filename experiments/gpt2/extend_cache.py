"""Extend a GPT-2 activation cache with fresh training rows.

Replays the original build's tokenization of the (deterministic) FineWeb
stream, verifying it window by window against the stored tokens, then
continues the stream past the eval documents to produce new training
windows. Layout of the new cache: [old train | new train | old eval], with
the old mean/scale, so models trained on the old cache read the new one
unchanged and the eval rows are the same held-out documents.

  uv run python experiments/gpt2/extend_cache.py --src data/gpt2_l8 --out data/gpt2_l8_40m --train-tokens 40_000_000
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

sys.path.insert(0, str(Path(__file__).parent))
from make_cache import windows, resid  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="data/gpt2_l8")
    p.add_argument("--out", default="data/gpt2_l8_40m")
    p.add_argument("--train-tokens", type=int, default=40_000_000, help="total train tokens (old + new)")
    p.add_argument("--batch", type=int, default=64)
    cfg = p.parse_args()
    src, out = Path(cfg.src), Path(cfg.out)
    meta = json.loads((src / "meta.json").read_text())
    ctx, d, per = meta["ctx"], meta["d"], meta["ctx"] - 1
    mu, s = np.load(src / "mean.npy"), meta["scale"]
    tok_train_old, tok_eval_old = np.load(src / "tokens_train.npy"), np.load(src / "tokens_eval.npy")
    n_old_w, n_eval_w = meta["n_train_windows"], meta["n_eval_windows"]
    assert tok_train_old.shape == (n_old_w, ctx) and tok_eval_old.shape == (n_eval_w, ctx)
    n_total_w = -(-cfg.train_tokens // per)
    n_add_w = n_total_w - n_old_w
    assert n_add_w > 0, "nothing to add"
    n_train_rows, n_eval_rows = n_total_w * per, n_eval_w * per
    print(f"old {n_old_w} train windows, adding {n_add_w}, eval {n_eval_w}; rows {n_train_rows + n_eval_rows}")

    out.mkdir(parents=True, exist_ok=True)
    acts = np.lib.format.open_memmap(out / "acts.npy", mode="w+", dtype=np.float16,
                                     shape=(n_train_rows + n_eval_rows, d))
    old = np.load(src / "acts.npy", mmap_mode="r")
    assert old.shape == (n_old_w * per + n_eval_rows, d)
    t0 = time.time()
    chunk = 1 << 18
    n_old = n_old_w * per
    for i in range(0, n_old, chunk):                   # old train rows, same place
        j = min(i + chunk, n_old)
        acts[i:j] = old[i:j]
    e0 = n_train_rows                                  # eval rows move to the end
    for i in range(0, n_eval_rows, chunk):
        j = min(i + chunk, n_eval_rows)
        acts[e0 + i:e0 + j] = old[n_old + i:n_old + j]
    print(f"copied old rows in {time.time() - t0:.0f}s", flush=True)

    tok = GPT2TokenizerFast.from_pretrained(meta["model"])
    tok.model_max_length = int(1e12)
    eot = tok.eos_token_id
    path, name = meta["dataset"].rsplit("/", 1)     # e.g. HuggingFaceFW/fineweb / sample-10BT
    ds = load_dataset(path, name=name, split="train", streaming=True)
    gen = windows((r["text"] for r in ds), tok, ctx, eot)

    # replay the original build exactly (make_cache.run discards the window
    # returned by the flushing send()) and verify against the stored tokens
    t0 = time.time()
    for i in range(n_old_w):
        w = next(gen)
        if i % 20000 == 0 or i == n_old_w - 1:
            assert np.array_equal(w, tok_train_old[i]), f"train window {i} differs: stream changed?"
    gen.send(True)
    for i in range(n_eval_w):
        w = next(gen)
        assert np.array_equal(w, tok_eval_old[i]), f"eval window {i} differs: stream changed?"
    print(f"replayed and verified the original stream in {time.time() - t0:.0f}s", flush=True)
    gen.send(True)                                     # drop the tail of the last eval document

    dev = torch.device("cuda")
    model = GPT2LMHeadModel.from_pretrained(meta["model"]).to(dev).eval()
    tok_new = np.zeros((n_add_w, ctx), np.uint16)
    row, w, t0 = n_old_w * per, 0, time.time()
    while w < n_add_w:
        batch = [next(gen) for _ in range(min(cfg.batch, n_add_w - w))]
        ids = torch.from_numpy(np.stack(batch)).to(dev)
        A = resid(model, ids, meta["layer"])[:, 1:].float().cpu().numpy().reshape(-1, d)
        tok_new[w:w + len(batch)] = np.stack(batch)
        acts[row:row + len(A)] = ((A - mu) / s).astype(np.float16)
        row += len(A)
        w += len(batch)
        if (w // cfg.batch) % 200 == 0:
            print(f"new {w}/{n_add_w} windows  {(row - n_old_w * per) / (time.time() - t0):,.0f} rows/s", flush=True)
    assert row == n_train_rows, (row, n_train_rows)
    acts.flush()
    np.save(out / "tokens_train.npy", np.concatenate([tok_train_old, tok_new]))
    np.save(out / "tokens_eval.npy", tok_eval_old)
    np.save(out / "mean.npy", mu.astype(np.float32))
    meta.update({"n_train_rows": n_train_rows, "n_train_windows": n_total_w,
                 "extended_from": str(src), "layout": "[old train | new train | eval]"})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s")
    sys.stdout.flush()
    os._exit(0)                                        # HF streaming threads hang shutdown


if __name__ == "__main__":
    main()
