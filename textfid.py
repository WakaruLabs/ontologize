"""Downstream text fidelity: the SAE-benchmark "loss recovered" analogue.

Whitened FVU measures reconstruction in embedding space; what actually
matters is whether the reconstruction still decodes to the same text.
For held-out cache rows this script decodes the original embedding and
the model's reconstruction through the SONAR/M2M100 decoder (forced
eng_Latn, so comparison happens in one language) and reports:

  chrF        character n-gram F2 between the two decodes (and exact-match
              rate) -- generation-space fidelity
  NLL         the decoder's mean per-token negative log-likelihood of the
              ORIGINAL embedding's decode, scored under the original, the
              reconstruction, and the corpus-mean embedding -- a smooth
              likelihood version of the same question, with the original
              as ceiling and the corpus mean (which decodes to the generic
              text every embedding shares) as floor

Both embeddings are rescaled to SONAR_NORM before decoding (the decoder
needs the right scale), so the metric sees direction errors, not norm
errors -- norm fidelity is already covered by FVU.

Works on sae.py runs (reconstruction = decode(encode(x))) and Ontologizer
checkpoints (full soft forward at --temperature). Rows come from the
cache tail (the sae.py eval split). Per-row decodes go to <out>/rows.jsonl
for inspection. NOTE: Ontologizer checkpoints restore on GPU JAX only.

  uv run python textfid.py --model data/out/sonar/sae/m5120_k32/params.npz
  uv run python textfid.py --model data/out/sonar/multilingual/resid_nc \\
                           --rows 512 --device cuda
"""
# disable preallocation so this can share the GPU (same as sonar.py)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import functools
import json
import numpy as np
from collections import Counter
from pathlib import Path

import sae

ENCODER_ID = "cointegrated/SONAR_200_text_encoder"
DECODER_ID = "raxtemur/SONAR_200_text_decoder"

# The norm a SONAR sentence embedding has before the cache's L2
# normalization, which the decoder was trained on: the median raw
# mean-pooled norm of 2048 mC4 corpus texts under the eval-mode encoder
# (SD 0.040; per-language medians 0.30-0.34). Every script that decodes
# a unit-norm embedding rescales it to this. Short reference sentences
# give 0.20, and decode->re-encode fidelity is lower there.
SONAR_NORM = 0.307


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="sae params.npz or onto checkpoint dir")
    p.add_argument("--cache", default="data/sonar_embeddings/mc4_4M.npy")
    p.add_argument("--rows", type=int, default=512,
                   help="cache-tail rows to decode (generation-bound)")
    p.add_argument("--b", type=int, default=4096, help="recon batch (jax)")
    p.add_argument("--b-decode", type=int, default=16,
                   help="generation/scoring batch (torch)")
    p.add_argument("--device", default="cpu",
                   help="torch device for M2M100 (cpu protects training)")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--out", default=None,
                   help="default data/out/sonar/textfid/<model name>")
    return p.parse_args()


def chrf(ref, hyp, n_max=6, beta=2.0):
    """chrF (Popović 2015): character n-gram F_beta, whitespace stripped,
    n-gram precision/recall averaged over orders 1..n_max."""
    ref = "".join(ref.split())
    hyp = "".join(hyp.split())
    if not ref or not hyp:
        return float(ref == hyp)
    ps, rs = [], []
    for n in range(1, n_max + 1):
        rg = Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        hg = Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        if not rg or not hg:
            continue
        both = sum((rg & hg).values())
        ps.append(both / sum(hg.values()))
        rs.append(both / sum(rg.values()))
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def recon_fn(path, cfg):
    """-> (jitted X -> reconstruction, model name)"""
    import jax
    import jax.numpy as jnp
    p = Path(path)
    if p.suffix == ".npz":
        params = {k: jnp.asarray(v) for k, v in np.load(p).items()}
        meta_path = p.parent / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            topk, groups, gfn = meta["topk"], meta["groups"], meta["group_fn"]
        else:
            from pareto import parse_sae_name
            _, topk = parse_sae_name(p)
            groups, gfn = 0, "top1"

        @jax.jit
        def recon(X):
            return sae.decode(params, sae.encode(params, X, topk, groups, gfn))

        return recon, p.parent.name

    from pareto import load_onto
    model, params, step = load_onto(str(p), cfg.step)
    T = cfg.temperature

    def probe(module, X):
        E, _ = module.encode(X, 0.0, None)
        R = module.resid(E)
        E_in = module.constinput(E)
        # reproduce the DictEnc's gain-shape split: under `resid_gain` it
        # classifies the unit-norm SHAPE of its input and scales its
        # contribution by the measured GAIN. Identity / no-op when off.
        for i, de in enumerate(module.dictencs):
            U, G = de.gainshape_in(E_in)
            P = de.dict.cluster(de.classifier(U), T)
            # head_outputs applies the router gain and fibers as the
            # forward does; dict.hfwd(P) alone would drop them
            R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
            if i < module.l - 1:
                E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
        return module.decode(R)

    recon = jax.jit(functools.partial(model.apply, {"params": params},
                                      method=probe))
    return recon, f"{p.name}_{step}"


class SonarDecoder:
    """The SONAR/M2M100 text decoder as every text-space eval uses it:
    forced eng_Latn, greedy, and each embedding rescaled to SONAR_NORM
    first (the decoder needs the right scale), so a comparison of two
    decodes sees direction errors, not norm errors."""

    def __init__(self, device: str = "cpu", b_decode: int = 16,
                 max_length: int = 48, ref_norm: float = SONAR_NORM):
        import torch as t
        from transformers import AutoTokenizer, M2M100ForConditionalGeneration
        self.t, self.dev = t, t.device(device)
        self.b, self.max_length = b_decode, max_length
        self.ref_norm = ref_norm
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
        self.dec = M2M100ForConditionalGeneration.from_pretrained(
            DECODER_ID).to(self.dev)
        self.dec.eval()
        self.eng = self.tokenizer.convert_tokens_to_ids("eng_Latn")
        self.pad = self.tokenizer.pad_token_id

    def scale(self, Y):
        Y = self.t.from_numpy(np.ascontiguousarray(Y)).to(self.dev,
                                                          self.t.float32)
        return self.t.nn.functional.normalize(Y, dim=-1) * self.ref_norm

    def _enc(self, Y):
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=self.scale(Y).unsqueeze(1))

    def generate(self, Y) -> list:
        """Token sequences decoded from the rows of `Y`."""
        seqs = []
        with self.t.no_grad():
            for i in range(0, len(Y), self.b):
                gen = self.dec.generate(
                    encoder_outputs=self._enc(Y[i:i + self.b]),
                    forced_bos_token_id=self.eng, max_length=self.max_length,
                    num_beams=1, repetition_penalty=1.2)
                seqs += [g for g in gen.cpu()]
        return seqs

    def text(self, seq) -> str:
        return self.tokenizer.decode(seq, skip_special_tokens=True).strip()

    def texts(self, Y) -> list:
        """Decoded text for each row of `Y`."""
        return [self.text(s) for s in self.generate(Y)]

    def nll(self, Y, seqs):
        """Mean per-token NLL of token sequences under embeddings Y."""
        t = self.t
        outs = np.empty(len(seqs))
        with t.no_grad():
            for i in range(0, len(seqs), self.b):
                batch = seqs[i:i + self.b]
                width = max(len(s) for s in batch)
                lab = t.full((len(batch), width - 1), -100, dtype=t.long)
                for j, s in enumerate(batch):  # drop the decoder-start token
                    toks = s[1:][s[1:] != self.pad]
                    lab[j, :len(toks)] = toks
                labd = lab.to(self.dev)
                logits = self.dec(encoder_outputs=self._enc(Y[i:i + self.b]),
                                  labels=labd).logits
                lp = t.log_softmax(logits, -1)
                tok_lp = lp.gather(-1, labd.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                m = (labd != -100).float()
                outs[i:i + len(batch)] = \
                    (-(tok_lp * m).sum(-1) / m.sum(-1).clamp(min=1)).cpu().numpy()
        return outs


def main():
    import jax.numpy as jnp

    cfg = parse_args()
    recon, name = recon_fn(cfg.model, cfg)
    out = Path(cfg.out) if cfg.out else Path("data/out/sonar/textfid") / name
    out.mkdir(parents=True, exist_ok=True)

    mm = np.load(cfg.cache, mmap_mode="r")
    X = np.asarray(mm[-cfg.rows:], dtype=np.float32)
    R = np.concatenate([np.asarray(recon(jnp.asarray(X[i:i + cfg.b])))
                        for i in range(0, len(X), cfg.b)])
    x_mean = np.asarray(mm[:1 << 17], dtype=np.float32).mean(0)

    dec = SonarDecoder(cfg.device, cfg.b_decode)
    generate, nll, txt = dec.generate, dec.nll, dec.text

    print(f"{name}: decoding {len(X)} tail rows on {cfg.device}")
    ref_seqs = generate(X)
    rec_seqs = generate(R)
    base_seq = generate(x_mean[None])[0]
    ref_txt = [txt(s) for s in ref_seqs]
    rec_txt = [txt(s) for s in rec_seqs]
    base_txt = txt(base_seq)

    chrf_model = np.array([chrf(a, b) for a, b in zip(ref_txt, rec_txt)])
    chrf_base = np.array([chrf(a, base_txt) for a in ref_txt])
    exact = np.mean([a == b for a, b in zip(ref_txt, rec_txt)])
    nll_x = nll(X, ref_seqs)
    nll_r = nll(R, ref_seqs)
    nll_base = nll(np.broadcast_to(x_mean, X.shape).copy(), ref_seqs)

    print(f"\nchrF(ref, recon) {chrf_model.mean():.4f}   "
          f"chrF(ref, corpus-mean) {chrf_base.mean():.4f}   "
          f"exact match {exact:.3f}")
    print(f"NLL of ref decode: under original {nll_x.mean():.4f}   "
          f"under recon {nll_r.mean():.4f}   "
          f"under corpus mean {nll_base.mean():.4f}")

    with open(out / "rows.jsonl", "w") as f:
        for i in range(len(X)):
            f.write(json.dumps(
                {"row": int(len(mm) - cfg.rows + i), "ref": ref_txt[i],
                 "recon": rec_txt[i], "chrf": float(chrf_model[i]),
                 "nll_x": float(nll_x[i]), "nll_r": float(nll_r[i])},
                ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(
        {"model": str(cfg.model), "rows": len(X),
         "chrf": chrf_model.mean(), "chrf_base": chrf_base.mean(),
         "exact": float(exact), "nll_x": nll_x.mean(),
         "nll_r": nll_r.mean(), "nll_base": nll_base.mean()}, indent=2))
    print(f"-> {out / 'rows.jsonl'}")


if __name__ == "__main__":
    main()
