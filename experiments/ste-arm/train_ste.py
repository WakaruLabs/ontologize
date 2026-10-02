"""Straight-through (hard) Ontologizer arm at raised head count.

sonar.py's live config with three things changed, each for a measured
reason:

  select="ste"   The live softmax model's code is not discrete. Its
                 end-to-end argmax reconstructs at whitened FVU 8.7e6
                 (pareto_sweep_softmax_shm), so the tag assignment the
                 interventions and decode_tags.py read is not what
                 carries the embedding. `ste` forward-passes the argmax
                 and back-propagates the softmax, so the trained model's
                 forward IS the discrete code and no gap can open.

  h (raised)     A hard code transmits exactly l*h*log2(k) bits and
                 nothing else. Reverse water-filling on the whitened
                 eval tail's covariance puts the best distortion any
                 800-bit code can reach at FVU_w 0.226, and the live
                 architecture spends exactly 800 bits -- so the current
                 head count cannot support a discrete ontology however
                 it is trained. Heads are the cheap axis (bits linear in
                 h, params linear in h, and heads run in parallel where
                 layers are sequential). The default h=76 gives 380
                 heads and 1900 bits, whose floor is FVU_w ~0.05.

  temperature    Held constant (no anneal). Under `ste` the forward is
                 temperature-invariant -- argmax(K/T) = argmax(K) -- so
                 T only sharpens the backward surrogate softmax(K/T).
                 Annealing it to the live 0.03 would drive the surrogate
                 to one-hot and starve the classifier of gradient, which
                 is the top1 failure in miniature (that arm's classifier
                 gets no gradient at all and its hard FVU_w is 0.719
                 against top2's 0.471). anneal_steps=0 also holds p_drop
                 and sd_K at their live end values, which the toy found
                 to be the safe settings under hard selection.

Everything else -- deepsup, resid conditioning, the auxiliary weights,
the whitened MSE, the cache and its 32768-row holdout -- is read from
sonar.py at import, so this arm cannot drift from the live config.

  uv run python experiments/ste-arm/train_ste.py --epochs 2     # pilot
  uv run python experiments/ste-arm/train_ste.py                # full
  uv run python experiments/ste-arm/train_ste.py --select softmax \\
      --out data/out/sonar/multilingual/softmax_h76   # capacity control

Resumable: TrainingEnv picks up the latest checkpoint in the out dir.
Score it against the SAEs and the live run with pareto.py, whose "onto
hard (argmax)" row is the number this arm exists to move.
"""
import os
# setdefault, not assignment: assigning here overrides whatever the
# caller exported, and the allocator choice is a caller's decision.
# `platform` hands every allocation straight to cudaMalloc with no
# pooling, which shares the card politely and fragments it steadily. On
# runs of this length that has truncated four arms at 6%, 23%, 34% and
# 96%, each dying on an allocation of tens of megabytes with the card
# otherwise idle. Export XLA_PYTHON_CLIENT_ALLOCATOR=bfc (or unset it,
# which is the same) for a long run; keep `platform` only when
# something else must share the GPU.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import importlib
import math

# the live training config; every constant not overridden below is read
# from here, so the arm tracks sonar.py by construction
import sonar as base

from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
from ontologize.data.loaders import NpyDataSource
from ontologize.visualize.loss import plot_loss


def _needs_encoder(biased_enc: bool, encoded: bool) -> bool:
    """`biased_enc` only reaches anything on the Linear-encoder branch."""
    if biased_enc and not encoded:
        raise SystemExit(
            "--biased-enc needs sonar.py's `encoded`; with it False the "
            "encoder is a Sparse passthrough and the flag changes nothing")
    return biased_enc


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select", default="ste",
                   help="DictBlock.select rule (ste, softmax, top<k>)")
    p.add_argument("--forward", default=None,
                   help="layer-to-layer interface (default sonar.py's "
                        "'resid'). 'resid_labels' also concatenates the "
                        "previous layer's code, widening every upper "
                        "classifier, so it is not parameter-matched to "
                        "'resid' at equal h")
    p.add_argument("--h", type=int, default=76,
                   help="heads per layer (sonar.py ships 32; 76 puts the "
                        "hard code at 1900 bits)")
    p.add_argument("--k", type=int, default=0, help="entries per head (0 = sonar.py's)")
    p.add_argument("--l", type=int, default=0, help="layers (0 = sonar.py's)")
    p.add_argument("--b", type=int, default=0,
                   help="batch size (0 = sonar.py's; lower it if the "
                        "raised head count will not fit)")
    p.add_argument("--lr", type=float, default=0.0, help="0 = sonar.py's")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="constant backward-surrogate temperature. Under "
                        "`ste` this sets ONLY the softmax the gradient "
                        "flows through, and it has to match the scale of "
                        "the logits themselves: the argmax forward is "
                        "scale-invariant, so nothing pushes |K| up, and a "
                        "T far above the measured logit spread leaves the "
                        "surrogate uniform and the estimator maximally "
                        "biased")
    p.add_argument("--sd-k", type=float, default=None,
                   help="classifier logit noise (default sonar.py's 0.02). "
                        "This is an ABSOLUTE logit-space perturbation, so "
                        "it must be read against the logit spread; at the "
                        "0.002 spread an untrained ste classifier shows, "
                        "the live value is ten times the signal and the "
                        "argmax is chosen by noise")
    p.add_argument("--noise-k", default=None,
                   help="classifier logit noise KIND (default sonar.py's "
                        "'normal', an absolute perturbation). 'batchnorm' "
                        "scales it by the logit RMS instead, which is what "
                        "an argmax forward needs: the selection is "
                        "scale-invariant, so the exploration that keeps "
                        "entries alive has to track the logit scale rather "
                        "than a fixed number of logit units")
    p.add_argument("--p-drop", type=float, default=None,
                   help="winner dropout (default sonar.py's 0.1)")
    p.add_argument("--s-kcossim", type=float, default=None,
                   help="within-head row-collinearity penalty (default "
                        "sonar.py's 0). Targets head death by a different "
                        "route than --s-hm: that one scores usage balance, "
                        "this one whether a head's entries decode to "
                        "distinct directions at all")
    p.add_argument("--kcos-target", type=float, default=0.0,
                   help="hold the per-layer MAX row cosine at a setpoint "
                        "instead of fixing --s-kcossim, which then becomes "
                        "the starting value. sonar.py's 0.2 is calibrated "
                        "against softmax runs; a hard code sits far above "
                        "it, so pick the target against the arm's own "
                        "measured value")
    p.add_argument("--kcos-eta", type=float, default=1e-3,
                   help="controller gain")
    p.add_argument("--kcos-ramp", type=int, default=50000,
                   help="steps over which the setpoint ramps from the "
                        "measured start. Required here: the loop falls "
                        "back to anneal_steps, which is 0 under a constant "
                        "temperature, and a setpoint demanded immediately "
                        "winds the multiplier to its bound")
    p.add_argument("--s-hsic-heads", type=float, default=0.0,
                   help="mean pairwise CKA between heads' classifications, "
                        "summed over layers (hsic_ste.py). Makes sharing a "
                        "variable across heads cost something; 0 = off. "
                        "Needs ghost=False, since the per-layer codes ride "
                        "the ghost slot")
    p.add_argument("--hsic-sigma2", type=float, default=1.0,
                   help="RBF bandwidth^2 for the head grams; 0 = median "
                        "heuristic. One-hot codes make 1 the natural value")
    p.add_argument("--hsic-estimator", default="unbiased",
                   choices=["biased", "unbiased"],
                   help="the biased estimator's value is dominated by a "
                        "floor independent heads alone produce at a "
                        "training batch, and it carries that floor's "
                        "gradient; it is here to reproduce the arms that "
                        "predate the switch")
    p.add_argument("--s-bcossim", type=float, default=None,
                   help="sample-similarity penalty (default sonar.py's "
                        "1e-5). Read it against the gradient, not the loss "
                        "share: equal pull on the classifier is near 2e-4, "
                        "so the shipped value applies about a twentieth of "
                        "the objective's own force")
    p.add_argument("--s-hcossim", type=float, default=None,
                   help="head-similarity penalty (default sonar.py's "
                        "1e-6, which is some 240x below equal pull)")
    p.add_argument("--s-flatcos", type=float, default=None,
                   help="flattened row-collinearity penalty "
                        "(`DictBlock.flatcos`), the between-head "
                        "complement of --s-kcossim; 0 = off")
    p.add_argument("--s-l1k", type=float, default=None,
                   help="classifier L1 penalty (`NLinearBlock.withL1`), 0 = "
                        "off, which is sonar.py's setting and is marked "
                        "disrecommended there for 'paradoxically causing "
                        "exploding L1_K'. Note what it acts on: the GATE "
                        "factor `fn_gate(Ys[0])`, not the logits. With any "
                        "positively-homogeneous gate -- identity and relu "
                        "both -- scaling that factor down and the second up "
                        "leaves the logits bit-identical while the penalty "
                        "falls, so it has a free direction to descend. Only "
                        "a saturating gate, or a bias inside the gate that "
                        "`NLinear` does not have, makes it well-posed")
    p.add_argument("--scaled", action="store_true",
                   help="build the per-head router (sonar.py ships it off). "
                        "One scalar gain per head, so the router decides how "
                        "much a head speaks and the dictionary what it says. "
                        "Needed by --s-l1s, which is otherwise inert")
    p.add_argument("--gate-router", default=None, metavar="FN",
                   help="the router's gate activation. `L1_S` is computed on "
                        "the GATE factor, not on S, so this is what decides "
                        "whether that penalty is well-posed: 'none' leaves it "
                        "on a raw bilinear factor, which a rescaling pays "
                        "down for free, while 'sigmoid' bounds it to (0,1) "
                        "and breaks the rescaling. sigmoid also never reaches "
                        "0, so no head is ever permanently shut")
    p.add_argument("--router-signed", action="store_true",
                   help="let a head's gain go negative: drop the abs() in "
                        "`DictEnc.scale`. Needs --scaled; no --gate-router "
                        "makes it inert, since the gate bounds only the gate "
                        "factor and the magnitude factor stays signed. The "
                        "abs is a fold with a kink inside the data, which "
                        "--s-l1s drives S onto; this trades that for a head "
                        "whose polarity the input decides")
    p.add_argument("--s-l1s", type=float, default=None,
                   help="router L1 penalty (`L1_S`), which needs sonar.py's "
                        "`scaled` -- off there, so this is inert without it. "
                        "Distinct from --s-l1f: that prices `hfwd(P, S)`, the "
                        "product S*|W|*c, which either factor can pay down. "
                        "This prices the router alone and leaves the "
                        "dictionary's magnitude to reconstruction, which is "
                        "the separation a gated architecture rests on")
    p.add_argument("--s-hm", type=float, default=None,
                   help="batch mean-entropy bonus (default sonar.py's "
                        "1e-6). That value is calibrated against a whitened "
                        "MSE of ~3e-6; a hard code trains at a far larger "
                        "MSE, where it is inert, so it has to be rescaled "
                        "or heads collapse onto a few entries")
    p.add_argument("--dict-init-scale", type=float, default=0.0,
                   help="rescale the dictionary at init so one layer "
                        "leaves a residual of this norm (0 = off). The "
                        "dictionary is abs()'d, so h rows sum coherently "
                        "and `gained` compounds that across layers: the "
                        "initial residual grows as ~(c*h)^l, reaching "
                        "5.25e11 at 5x380 where the live 5x76 sits at "
                        "1.84e8. This divides out h entirely -- both land "
                        "on 20.1 at a target of 1")
    p.add_argument("--dict-wd", type=float, default=0.0,
                   help="decoupled weight decay on the dictionary alone "
                        "(0 = off). The decoder is wider than its output, so "
                        "a dictionary component in its null space changes "
                        "nothing and takes no gradient, and Adam turns the "
                        "noise there into full-size steps. On ste_h76 that "
                        "left 97.7% of the dictionary's movement, and all "
                        "but 2.4% of its final energy, invisible "
                        "downstream. L1_F is aimed at the right quantity "
                        "but lands 92.5% of its pressure on the atoms doing "
                        "the work; see dictwd.py")
    p.add_argument("--init-from", default=None, metavar="CKPT",
                   help="take starting parameters from this checkpoint "
                        "rather than resuming --out. Needed with "
                        "--train-layer, whose optimizer has a different "
                        "opt_state structure than the saved one")
    p.add_argument("--train-layer", type=int, default=-1,
                   help="train only this DictEnc and freeze everything "
                        "else, for asking whether a layer has anything "
                        "left to give once its input stops moving")
    p.add_argument("--base", default="sonar", metavar="MODULE",
                   help="config module every unset value falls back to. "
                        "`sonar` is the live SONAR run; `gpt2_l8` swaps the "
                        "widths, cache and MSE weighting for the GPT-2 "
                        "activation cache encode_acts.py writes, and changes "
                        "nothing else, so the arms differ in their data and "
                        "not in what is asked of the model")
    p.add_argument("--e-dec", type=int, default=0,
                   help="dictionary/decoder-input width (0 = the base "
                        "config's)")
    p.add_argument("--d-head", type=int, default=0,
                   help="with --concat, output dimensions per head; e_dec "
                        "becomes h * d_head. Set directly rather than "
                        "derived as e_dec // h because this is the knob that "
                        "carries the construction's content: the decoder's "
                        "per-head column block is (d_out, d_head), which has "
                        "no null space at all once d_head <= d_out, so the "
                        "summing form's gauge freedom disappears rather than "
                        "merely shrinking. Needs h * d_head >= d_out for the "
                        "heads to span the output between them. 0 matches the "
                        "summing arm's total width, the one case that needs "
                        "e_dec divisible by h")
    p.add_argument("--signed", action="store_true",
                   help="let atoms subtract: drop the abs() in "
                        "`DictBlock.dicts`. Non-negative rows have "
                        "non-negative pairwise cosine exactly, so a head's "
                        "k entries share a direction that says nothing about "
                        "which of them was selected; signing them frees that "
                        "capacity, and narrowing e_dec stops costing much")
    p.add_argument("--concat", action="store_true",
                   help="heads write disjoint slices of e_dec instead of "
                        "summing into all of it (ConcatDictBlock), so "
                        "between-head orthogonality holds by construction "
                        "rather than by the s_hcossim penalty, and the "
                        "dictionary costs h times fewer parameters at matched "
                        "e_dec. See --d-head. Orthogonality is in e_dec, not "
                        "after the decoder")
    p.add_argument("--biased-dec", action="store_true",
                   help="give the decoder a bias. The dictionary is "
                        "abs()'d, so the accumulated R is a sum of "
                        "non-negative vectors whose constant part is 8 to "
                        "12x its sample-varying part. Without a bias one "
                        "matrix must both decode the variation and "
                        "attenuate that constant ~270x to land on the "
                        "embedding mean, which is why it is dense and "
                        "magnitude pruning explodes")
    p.add_argument("--encoded", action="store_true",
                   help="build the Linear encoder sonar.py switches off. "
                        "With it off, E is X exactly, so the layer-0 "
                        "classifier sees the raw embedding cloud: mean "
                        "direction 0.56, pairwise cosine 0.31, effective "
                        "dimension 107 of 1024, against 0.09/0.009/611 "
                        "for the residual layer 1 classifies. The encoder "
                        "is the only place a one-layer model can change "
                        "that, since the reconstruction target stays X")
    p.add_argument("--e-enc", type=int, default=0,
                   help="encoder width with --encoded (0 = sonar.py's)")
    p.add_argument("--zca", default=None, metavar="NPZ",
                   help="install make_zca.py's whitener as the encoder "
                        "(implies --encoded --biased-enc --e-enc d). Held "
                        "fixed unless --train-enc, so any change in code "
                        "utilization is attributable to the input geometry")
    p.add_argument("--train-enc", action="store_true",
                   help="with --zca, let the whitener train from there "
                        "instead of holding it fixed")
    p.add_argument("--biased-enc", action="store_true",
                   help="give the encoder a bias, so it can subtract a "
                        "mean and `gainshape_in` no longer normalizes an "
                        "uncentered vector. Requires sonar.py's `encoded`: "
                        "with it False the encoder is a Sparse passthrough, "
                        "`biased_enc` is read only on the Linear branch, "
                        "and this flag is silently inert")
    p.add_argument("--steps", type=int, default=0,
                   help="stop after this many steps (0 = run the epochs "
                        "out); for short configuration pilots")
    p.add_argument("--anneal", action="store_true",
                   help="use sonar.py's temperature anneal instead of a "
                        "constant T (starves the ste surrogate; for "
                        "ablation only)")
    p.add_argument("--epochs", type=int, default=0,
                   help="0 = sonar.py's (24); a small value pilots")
    p.add_argument("--seed", type=int, default=0, help="0 = sonar.py's")
    p.add_argument("--cache", default=None)
    p.add_argument("--mse-weights", default=None, metavar="NPY",
                   help="per-dim MSE weights (default: the base config's). "
                        "'none' disables target whitening")
    p.add_argument("--out", default=None,
                   help="checkpoint dir (default "
                        "<sonar out dir>/../<select>_h<h>)")
    cfg = p.parse_args()
    if cfg.kcos_target > 0 and not cfg.s_kcossim:
        # two reasons this cannot start at zero: `s_loss` gates the
        # penalty's gradient on a nonzero weight, and the controller's
        # update is multiplicative, so zero is an absorbing state
        cfg.s_kcossim = 1e-6
    return cfg


def main() -> None:
    global base
    cfg = parse_args()
    if cfg.base != base.__name__:
        base = importlib.import_module(cfg.base)
    h = cfg.h
    k = cfg.k or base.k
    l = cfg.l or base.l
    b = cfg.b or base.b
    cache = Path(cfg.cache) if cfg.cache else base.cache
    assert cache and Path(cache).exists(), \
        f"embedding cache {cache} missing (encode_corpus.py builds it)"
    assert not (cfg.d_head and cfg.e_dec), \
        "--d-head fixes e_dec at h * d_head; pass one or the other"
    if cfg.concat and cfg.d_head:
        e_dec = h * cfg.d_head
    else:
        e_dec = cfg.e_dec or base.e_dec
    d_head = e_dec // h if cfg.concat else e_dec
    # `ConcatDictBlock` raises this too, but only from inside `init`, after
    # the cache is opened and the loader built
    assert not (cfg.concat and e_dec % h), \
        f"--concat needs e_dec divisible by h; got e_dec={e_dec}, h={h}. " \
        f"Pass --d-head instead of matching the summing arm's width"
    assert not (cfg.concat and h * d_head < base.d), \
        f"--concat with d_head={d_head} spans {h * d_head} of the " \
        f"{base.d}-dimensional output; the heads cannot reach the rest"
    tag = f"_cat{d_head}" if cfg.concat else ""
    out = Path(cfg.out) if cfg.out else \
        Path(base.out).with_name(f"{cfg.select}_h{h}{tag}")
    assert out.resolve() != Path(base.out).resolve(), \
        "refusing to write into sonar.py's own output directory"

    bits = l * h * math.log2(k)
    print(f"{out.name}: select={cfg.select} h={h} k={k} l={l} b={b} "
          f"T={'anneal' if cfg.anneal else cfg.temperature} "
          f"sd_K={base.sd_K if cfg.sd_k is None else cfg.sd_k} "
          f"p_drop={base.p_drop if cfg.p_drop is None else cfg.p_drop}")
    print(f"  hard code = {l * h} heads x log2({k}) = {bits:.0f} bits/sample "
          f"({bits / (base.l * base.h * math.log2(base.k)):.2f}x the live 800)")
    if cfg.concat:
        null = max(d_head - base.d, 0)
        print(f"  concat: {d_head} dims per head, e_dec={e_dec}; dictionary "
              f"{l * h * k * d_head / 1e6:.2f}M against "
              f"{l * h * k * base.e_dec / 1e6:.2f}M summing at e_dec="
              f"{base.e_dec}; decoder {base.d * e_dec / 1e6:.2f}M")
        print(f"  decoder column blocks ({base.d}, {d_head}): per-head null "
              f"space {null} dims; h x d_head = {h * d_head}, "
              f"{h * d_head / base.d:.2f}x the {base.d}-dim output")

    # sonar.py holds the cache tail out so the eval scripts score
    # out-of-sample; keep that here or the pareto comparison is not honest.
    # `--steps` shrinks the exposed head of the cache instead of the epoch
    # count, which the loader can only take as an integer, and holds out
    # everything past it -- the eval tail included.
    holdout = base.holdout
    epochs = cfg.epochs or base.epochs
    if cfg.steps:
        rows = cfg.steps * b
        total = NpyDataSource(cache).__len__()
        assert rows < total - base.holdout, \
            f"--steps {cfg.steps} needs {rows} rows, more than the cache has"
        holdout, epochs = total - rows, 1
    src = NpyDataSource(cache, holdout=holdout)

    HP, extra = Hyperparams, {}
    # --init-from is what loads the starting parameters, so it must take
    # effect whether or not a layer is frozen. Gating both on
    # --train-layer >= 0 made `--train-layer -1 --init-from X` silently
    # ignore X and train from scratch, which is not the control it looks
    # like: it read as the model collapsing under finetuning.
    if cfg.dict_init_scale or cfg.dict_wd:
        assert not (cfg.zca or cfg.s_hsic_heads or cfg.train_layer >= 0
                    or cfg.init_from), \
            "--dict-init-scale/--dict-wd replace Hyperparams, as do --zca, " \
            "--s-hsic-heads and --train-layer/--init-from"
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        # DictWDHyperparams subclasses InitScaleHyperparams, so the two
        # compose and a decay arm keeps its control's initialization
        from dictwd import DictWDHyperparams as HP
        extra = dict(dict_init_scale=cfg.dict_init_scale,
                     dict_wd=cfg.dict_wd,
                     init_probe_cache=str(cache))
    if cfg.train_layer >= 0 or cfg.init_from:
        assert not cfg.zca and not cfg.s_hsic_heads, \
            "--train-layer, --zca and --s-hsic-heads each replace Hyperparams"
        assert cfg.train_layer < l, f"--train-layer {cfg.train_layer} >= l={l}"
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from layerft import LayerFTHyperparams as HP
        extra = dict(train_layer=cfg.train_layer,
                     init_from=cfg.init_from or "")
    if cfg.zca:
        assert not cfg.s_hsic_heads, \
            "--zca and --s-hsic-heads both replace Hyperparams"
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from whiten_enc import WhitenEncHyperparams as HP
        extra = dict(zca_path=cfg.zca, freeze_enc=not cfg.train_enc)
    if cfg.s_hsic_heads:
        assert not base.ghost, "the HSIC penalty rides the ghost slot"
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from hsic_ste import HSICSteHyperparams as HP
        extra = dict(s_hsic_heads=cfg.s_hsic_heads, hsic_h=h,
                     hsic_sigma2=cfg.hsic_sigma2,
                     hsic_estimator=cfg.hsic_estimator)
    hyper = HP(
        base.d, base.d, b, epochs, cfg.lr or base.lr,
        base.wd, cfg.temperature if not cfg.anneal else base.temperature,
        base.noise_in, cfg.noise_k or base.noise_K, base.noise_F,
        base.sd_in,
        base.sd_K if cfg.sd_k is None else cfg.sd_k, base.sd_F,
        s_g=base.s_g, s_L1F=base.s_L1F, s_H=base.s_H,
        s_L1K=base.s_L1K if cfg.s_l1k is None else cfg.s_l1k,
        s_L1S=getattr(base, 's_L1S', 0.0) if cfg.s_l1s is None
        else cfg.s_l1s,
        s_bcossim=(base.s_bcossim if cfg.s_bcossim is None
                   else cfg.s_bcossim),
        s_hcossim=(base.s_hcossim if cfg.s_hcossim is None
                   else cfg.s_hcossim),
        s_flatcos=cfg.s_flatcos or 0.0,
        s_kcossim=(base.s_kcossim if cfg.s_kcossim is None
                   else cfg.s_kcossim),
        KCOS_target=cfg.kcos_target, KCOS_eta=cfg.kcos_eta,
        KCOS_ramp=cfg.kcos_ramp,
        s_Hm=base.s_Hm if cfg.s_hm is None else cfg.s_hm,
        p_drop=base.p_drop if cfg.p_drop is None else cfg.p_drop,
        p_revive=base.p_revive,
        revive_frac=base.revive_frac,
        mse_weights=(None if cfg.mse_weights == "none"
                     else cfg.mse_weights or base.mse_weights),
        # constant T: see the module docstring. anneal_steps=0 also holds
        # p_drop and sd_K at their live end values (`schedules` returns
        # its inputs unchanged), so no companion schedule ramps either.
        temperature_end=base.temperature_end if cfg.anneal else None,
        anneal_steps=base.anneal_steps if cfg.anneal else 0,
        p_drop_start=base.p_drop_start if cfg.anneal else None,
        sd_K_end=base.sd_K_end if cfg.anneal else None,
        # sonar.py never names a seed; it takes Hyperparams' own default
        ghost=base.ghost, seed=cfg.seed or getattr(base, "seed", 42),
        **extra)

    model = hyper.ontologizer(
        base.d, base.d, e_dec, k, h, l,
        n=base.n, gate=base.gate, select=cfg.select,
        norm_rows=base.norm_rows, concat=cfg.concat, signed=cfg.signed,
        scaled=cfg.scaled or base.scaled,
        gate_router=cfg.gate_router or 'none',
        router_signed=cfg.router_signed,
        encoded=bool(cfg.zca) or cfg.encoded or base.encoded,
        # the whitener is square on the embedding, so it fixes the width
        e_enc=(base.d if cfg.zca else (cfg.e_enc or base.e_enc)),
        # Ontologizer reads biased_enc only where it builds the Linear
        # encoder, so without `encoded` this flag trains a bit-identical
        # model and costs a full run to discover that
        biased_enc=bool(cfg.zca) or _needs_encoder(
            cfg.biased_enc, cfg.encoded or base.encoded),
        biased_dec=cfg.biased_dec,
        forward=cfg.forward or base.fwd_mode, deepsup=base.deepsup,
        deepsup_sg=base.deepsup_sg,
        resid_norm=base.resid_norm, resid_const=base.resid_const,
        dtype_str=base.dtype_str, dtype_p_str=base.dtype_p_str)

    meta = Metadata(
        "embedding", base.encoder, out_path=out, threads=base.threads,
        save_each=base.save_each, checkpoint_each=base.checkpoint_each)

    env = TrainingEnv(model, hyper, meta,
                      kwargs_loader={"d": base.d, "shuffle": base.shuffle})
    env.train(src, encoder=None)
    print("Training Complete!")
    try:
        plot_loss(out, base=2)
    except Exception as e:   # the plot is a convenience, not the deliverable
        print(f"warning: plot_loss failed ({e!r}); loss.csv is intact")


if __name__ == "__main__":
    main()
