"""How much of an embedding survives an unsteered text round trip.

`steerfid.py` (and `steer_overlay.py`, and `rl_reinforce.py`'s cycle tier)
read an intervention's effect off cycle(x) = encode(decode(x)): the
embedding decoded through the SONAR decoder (`textfid.SonarDecoder`) and
the text re-encoded exactly as steerfid's `cycle_acts` does
(TokenizeTransform, mean pool, L2 norm, as the cache was built). This
measures that instrument on UNSTEERED rows: an intervention can show up
in the cycled embedding only as far as the cycle preserves anything.

For the last --rows cache rows (textfid's rows), model-free:
  cos(x, cycle(x))         against cos(x, cycle(x')) for another row x',
                           the part any cycled text shares with any row,
                           and against cos between random pairs of rows
  cos(cycle(x), mean)      against cos(x, mean), the corpus-mean
                           direction over the first 2**17 rows
and per model, the share of (row, head) whose argmax entry on cycle(x)
is the one it picks on x, per layer and overall, against chance 1/k.
Each model runs at its schedule's temperature at the restored step.

  uv run python experiments/ste-arm/roundtrip.py --device cuda \\
      --runs data/out/sonar/multilingual/ste_h76 \\
             data/out/sonar/multilingual/sweep_softmax_shm

Writes <out>/summary.json and rows.jsonl (each row's decode). NOTE:
Ontologizer checkpoints restore on GPU JAX only.
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from jaxtyping import Float

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ENCODER_ID = "cointegrated/SONAR_200_text_encoder"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Ontologizer checkpoint dirs")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--b-decode", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/out/sonar/roundtrip")
    return p.parse_args()


def unit(V: Float[np.ndarray, "n d"]) -> Float[np.ndarray, "n d"]:
    return V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-12)


def survival(Pa: Float[np.ndarray, "n l h k"], Pb: Float[np.ndarray, "n l h k"]
             ) -> Float[np.ndarray, "l"]:
    """Per layer, the share of (row, head) whose argmax entry agrees."""
    return (Pa.argmax(-1) == Pb.argmax(-1)).mean((0, 2))


def main():
    cfg = parse_args()
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    import torch as t
    from assignmap import read_hyper, temperature_at
    from autointerp import onto_acts_fn
    from textfid import SonarDecoder
    from ontologize.data.loaders import TokenizeTransform
    from ontologize.data.pretrained import encode, pretrained_transformer

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.rows:], dtype=np.float32)
    mean = unit(np.asarray(mm[:1 << 17], dtype=np.float32).mean(0)[None])[0]

    dec = SonarDecoder(cfg.device, cfg.b_decode)
    texts = dec.texts(X)
    dev = t.device(cfg.device)
    enc, tokenizer = pretrained_transformer(ENCODER_ID, "float32", dev=dev)
    tok = TokenizeTransform(tokenizer, maxlen=512)
    C = []
    for i in range(0, len(texts), cfg.b_decode):
        batch = [tok.map({"text": s, "lang": "en"})
                 for s in texts[i:i + cfg.b_decode]]
        batch = {k: np.stack([b[k] for b in batch])
                 for k in ("input_ids", "attention_mask")}
        C.append(np.asarray(encode(enc, batch, dev)))
    C = np.concatenate(C).astype(np.float32)
    del enc

    Xu, Cu = unit(X), unit(C)
    perm = np.random.default_rng(cfg.seed).permutation(len(X))
    perm = np.where(perm == np.arange(len(X)), np.roll(perm, 1), perm)
    summary = {
        "rows": len(X),
        "cos_x_cycle": float((Xu * Cu).sum(-1).mean()),
        "cos_x_cycle_other": float((Xu * Cu[perm]).sum(-1).mean()),
        "cos_random_pairs": float((Xu * Xu[perm]).sum(-1).mean()),
        "cos_cycle_mean": float((Cu @ mean).mean()),
        "cos_x_mean": float((Xu @ mean).mean()),
        "models": {}}
    print(f"cos(x, cycle(x)) {summary['cos_x_cycle']:.3f}   another row's "
          f"cycle {summary['cos_x_cycle_other']:.3f}   random pairs "
          f"{summary['cos_random_pairs']:.3f}   cos(cycle(x), mean) "
          f"{summary['cos_cycle_mean']:.3f}   cos(x, mean) "
          f"{summary['cos_x_mean']:.3f}")

    for run in cfg.runs:
        step = ocp.CheckpointManager(
            Path(run).resolve(),
            checkpointers={"state": ocp.PyTreeCheckpointer(),
                           "spec": ocp.PyTreeCheckpointer()}).latest_step()
        T = temperature_at(read_hyper(run), step)
        acts, _, _, meta = onto_acts_fn(run, step, T)
        l, h, k = meta["l"], meta["h"], meta["k"]
        Pa = np.asarray(acts(jnp.asarray(X))).reshape(-1, l, h, k)
        Pb = np.asarray(acts(jnp.asarray(C))).reshape(-1, l, h, k)
        s = survival(Pa, Pb)
        summary["models"][Path(run).name] = {
            "step": int(meta["step"]), "temperature": float(T), "k": k,
            "survival": float(s.mean()), "survival_by_layer": s.tolist(),
            "chance": 1.0 / k}
        print(f"{Path(run).name} step {meta['step']} (T={T:g}): head argmax "
              f"survives {s.mean():.1%} (chance {1 / k:.1%}); by layer "
              + " ".join(f"{v:.1%}" for v in s))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "rows.jsonl", "w") as f:
        for i, s in enumerate(texts):
            f.write(json.dumps({"row": int(len(mm) - cfg.rows + i),
                                "decode": s}, ensure_ascii=False) + "\n")
    print(f"-> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
