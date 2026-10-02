"""Decode one head's dictionary entries to text, through the SONAR decoder.

`decode_tags.py` decodes every entry in the model (l*h*k = 12160 for the
live arm). This targets one head, so the output is readable and a single
head's hypothesis can be checked.

An entry decodes as its own additive contribution with the rest of the
model absent, renormalized to a typical embedding norm, so the text is a
probe and not a reconstruction: read it for what varies ACROSS a head's
entries, against the head-mean row printed first.

`tags()` flattens `dicts()` head-major, so entry e of head j is row
`j * k + e`.
"""
import os
import sys

if "CUDNN_INJECTED" not in os.environ:
    try:
        import nvidia.cudnn
        cudnn_lib = list(nvidia.cudnn.__path__)[0] + "/lib"
        os.environ["LD_LIBRARY_PATH"] = \
            f"{cudnn_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
        os.environ["CUDNN_INJECTED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except ImportError:
        pass

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
from pathlib import Path

import jax                                                   # noqa: E402
import jax.numpy as jnp                                      # noqa: E402

jax.random.PRNGKey(0)        # force JAX to claim its CuDNN before torch

import numpy as np                                           # noqa: E402
import orbax.checkpoint as ocp                               # noqa: E402
import torch as t                                            # noqa: E402
import torch.nn.functional as F                              # noqa: E402
from transformers import M2M100ForConditionalGeneration      # noqa: E402
from transformers.modeling_outputs import BaseModelOutput     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ontologize.data.pretrained import pretrained_transformer  # noqa: E402
from ontologize.ontologizer import Ontologizer                # noqa: E402
from ontologize.training.serialize import migrate_spec        # noqa: E402

REF = ["The weather is nice today.",
       "She walked to the store to buy some bread.",
       "The government announced new economic policies yesterday.",
       "I really enjoyed the concert last night.",
       "Scientists discovered a new species in the rainforest.",
       "Please remember to submit your report by Friday."]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--lang", default="eng_Latn",
                   help="language the decoder generates in")
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--deviation", action="store_true",
                   help="decode each entry MINUS the head mean. Entries in a "
                        "layer-0 head of the live arm are 0.74-collinear, so "
                        "the raw decode is dominated by the shared direction "
                        "and every entry returns the same sentence with "
                        "details shuffled; the deviation is the part that "
                        "actually distinguishes them")
    return p.parse_args()


def main():
    cfg = parse_args()
    dev = t.device("cuda" if t.cuda.is_available() else "cpu")
    enc, tok = pretrained_transformer("cointegrated/SONAR_200_text_encoder",
                                      dtype_str="float32", dev=dev)
    tok.src_lang = "eng_Latn"
    dec = M2M100ForConditionalGeneration.from_pretrained(
        "raxtemur/SONAR_200_text_decoder").to(dev)

    man = ocp.CheckpointManager(
        Path(cfg.checkpoint).resolve(),
        checkpointers={"state": ocp.PyTreeCheckpointer(),
                       "spec": ocp.PyTreeCheckpointer()})
    step = man.latest_step()
    model = Ontologizer(**migrate_spec(man.restore(step,
                                                   items={"spec": None})["spec"]))
    state = man.restore(step, items={"state": None})["state"]
    params = state["params"] if "opt_state" in state else state
    while "params" in params:
        params = params["params"]

    # the decoder wants the raw mean-pooled scale, not unit norm
    inp = tok(REF, return_tensors="pt", padding=True)
    with t.no_grad():
        o = enc(**{k: v.to(dev) for k, v in inp.items()})
        m = inp["attention_mask"].to(dev).unsqueeze(-1).float()
        ref = t.norm((o.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9),
                     dim=-1).mean().item()

    R, _ = model.apply({"params": params}, cfg.layer,
                       method=Ontologizer.decodeLayerEntries)
    k = model.k
    rows = np.asarray(R)[cfg.head * k:(cfg.head + 1) * k]
    mu = rows.mean(0, keepdims=True)
    cos = float((rows / np.linalg.norm(rows, axis=1, keepdims=True)
                 @ (mu / np.linalg.norm(mu)).T).mean())
    print(f"mean cosine of an entry to the head mean: {cos:.3f}")
    rows = np.concatenate([mu, rows - mu if cfg.deviation else rows])

    emb = F.normalize(t.from_numpy(rows).to(dev, t.float32), dim=-1) * ref
    bos = tok.convert_tokens_to_ids(cfg.lang)
    texts = []
    with t.no_grad():
        for i in range(0, len(emb), 8):
            out = dec.generate(
                encoder_outputs=BaseModelOutput(
                    last_hidden_state=emb[i:i + 8].unsqueeze(1)),
                forced_bos_token_id=bos, max_length=48,
                num_beams=cfg.beams, repetition_penalty=1.2)
            texts += tok.batch_decode(out, skip_special_tokens=True)

    print(f"\n{Path(cfg.checkpoint).name} step {step}: layer {cfg.layer} "
          f"head {cfg.head}, k={k}, decoded as {cfg.lang}\n")
    print(f"{'mean':>5}  {texts[0]}")
    for e, s in enumerate(texts[1:]):
        print(f"{'e' + str(e):>5}  {s}")


if __name__ == "__main__":
    main()
