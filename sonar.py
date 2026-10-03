# Reserve a fixed share of VRAM up front rather than allocating on demand.
# On-demand allocation returns memory to the driver between steps and
# re-requests it, so a training run is only as safe as whatever else
# happens to hold the device at that instant -- one unlucky moment kills it
# mid-run. Reserving leaves the run immune to that, and half the device is
# far more than the model needs while still leaving room for the frozen
# torch encoder (loaded only when `cache` is unset) and anything else
# sharing the GPU. ONTO_MEM_FRACTION overrides the share.
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = os.environ.get(
        "ONTO_MEM_FRACTION", "0.5")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import ctypes
from pathlib import Path

# Force load the pip-installed CuDNN library before JAX initializes
try:
   import nvidia.cudnn
   # Namespace packages use __path__ which is an iterable of directory locations
   cudnn_dir = Path(list(nvidia.cudnn.__path__)[0])
   cudnn_path = cudnn_dir / "lib" / "libcudnn.so"
   
   if cudnn_path.exists():
       ctypes.CDLL(str(cudnn_path), mode=os.RTLD_GLOBAL)
except (ImportError, IndexError):
   pass

import torch as t
import jax.numpy as jnp
import einops

from datasets import load_dataset
from pathlib import Path

from ontologize.ontologizer import Ontologizer
from ontologize.training.ontostate import OntoState
from ontologize.training.config import Hyperparams, Metadata, TrainingEnv
from ontologize.data.loaders import JSONLDataSource, HFDataSource, NpyDataSource
from ontologize.data.multilingual import mc4_data

from ontologize.visualize.loss import plot_loss

# workaround to avoid requiring torch_xla if not running on TPU
if os.environ.get("USE_TPU"):
    os.environ["PJRT_DEVICE"] = "TPU"
    import torch_xla.core.xla_model as xm
    dev = xm.xla_device()
elif t.cuda.is_available():
    dev = t.device("cuda")
else:
    dev = t.device("cpu")

path = Path("data").resolve()

# Per-run overrides, so one config file can drive a sweep of architecture
# arms without forking copies of it that then drift. Every knob defined
# with `_env` below reads `ONTO_<NAME>` from the environment and otherwise
# keeps the literal default, e.g.
#   ONTO_RUN=topk4 ONTO_SELECT=top4 uv run python sonar.py
# A run stays self-describing either way: the architecture actually used is
# serialized into the checkpoint spec (`dataclasses.asdict(model)`), so the
# override does not have to be reconstructed from shell history.
def _env(name, default, cast=str):
    v = os.environ.get("ONTO_" + name)
    return default if v is None else cast(v)

dtype_str = "float32" # reducing precision breaks the embeddings
dtype_p_str = "float32" # high precision to reduce exploding gradient

# NLLB tokenizer and model
encoder = "cointegrated/SONAR_200_text_encoder"

# local training data
data = path / "text/tinystories.jsonl"
#src = JSONLDataSource(data, "text")

maxlen = 512 # sequence length
threads = 0 # not compatible with pretrained model integration

# precomputed-embedding cache built by encode_corpus.py. When set, the
# frozen SONAR encoder and tokenizer stay out of the training loop entirely
# (the ~7s/step encode+tokenize cost drops to a memmap read) and `epochs`
# becomes a true number of passes over the cached corpus (the streaming
# path never repeats data). encode_corpus.py only renames the file to this
# final name once the cache is complete, so a missing file may mean the
# encode pass is still running. None restores on-the-fly encoding.
cache = path / "sonar_embeddings/mc4_4M.npy"
shuffle = True # reshuffle the cached corpus each epoch (cache only)
# how the per-language interleave ends when `cache` is unset (streaming
# only; the cached path reads a .npy written earlier). "first_exhausted"
# caps every language at the smallest split; "all_exhausted" repeats the
# small ones rather than discarding the rest of the large ones.
stopping_strategy = _env("STOPPING_STRATEGY", "first_exhausted")
# rows excluded from the cache TAIL, matching sae.py's --eval-rows default:
# the SAE-comparison suite (sae.py, pareto.py, autointerp.py, ...) scores on
# that tail, so holding it out here makes those scores out-of-sample for the
# Ontologizer too. NOTE: checkpoints trained before this holdout landed saw
# the full cache -- their eval-tail scores are in-sample (see the sae.py
# header caveat). 0 restores full-cache training.
holdout = 32768

save_each = 250 # how often to save model & write loss
checkpoint_each = 25000 # how often to keep checkpoints indefinitely
out = path / "out/sonar/multilingual" / _env("RUN", "resid_nc_hm_c0")
# output directory (fresh dir: resid_const widens every classifier input,
# layer 0 included -- not checkpoint-compatible with the resid_nc_hm run,
# which is preserved at multilingual/resid_nc_hm). Any arm that changes the
# architecture needs its own ONTO_RUN, since a changed classifier shape or
# selection rule is not checkpoint-compatible with the default either.

# Ontologizer spec
d = 1024 # input & output dimension
e_enc = 2048 # encoder width
e_dec = 2048 # decoder height
k = 32 # number of tags per head
h = 32 # number of heads
l = 5 # number of DictEnc layers; must match the checkpoint when resuming

# classifier form. n=2/gate="none" is the pure bilinear quadratic form the
# eigendecomposition analysis (Bilinear.decompose, autointerp's "eig" mode,
# steerfid's eigenfeature steering) requires. ONTO_N=1 ONTO_GATE=relu is the
# linear relu-gated alternative: it gives up that analysis, but relu can
# produce the exact batch-wide zeros ghostgrad fires on, so dead classifier
# units become reachable -- and observable -- where bilinear + softmax makes
# them structurally impossible.
n = _env("N", 2, int)
gate = _env("GATE", "none")

# logits -> probabilities rule (DictBlock.select). "softmax" gives every tag
# in a head some mass; "top<k>" (e.g. ONTO_SELECT=top4) keeps the k largest
# logits per head and masks the rest to -inf, making the code exactly sparse
# rather than merely peaked. The two sparsify different objects: top-k acts
# on the CODE, a relu gate acts on the classifier PRE-ACTIVATIONS and leaves
# the code dense (softmax of a relu'd zero is still nonzero, just small).
select = _env("SELECT", "softmax")

# unit-L2-normalize each dictionary row, so a tag is a direction and its
# magnitude comes from `cluster` alone. ONTO_NORM_ROWS=1 turns it on. This
# bounds what s_L1F can do: the rows are non-negative and each head's
# classification sums to 1, so nothing cancels and the layer's L1 cannot
# fall below h (h*l summed over layers) however hard the penalty pushes.
norm_rows = _env("NORM_ROWS", False,
                 lambda v: v.lower() in ("1", "true", "yes"))

# layer-to-layer interface: "labels" (each layer reads the previous layer's
# classification; collapse-prone under hard temperatures -- the starvation
# probe showed layers 2-4 receive constant input) or "resid" (RVQ-style:
# each layer reads the stop-gradiented reconstruction residual; in the
# GPT-2 prototype this kept >99% of upper-layer dict entries live and gave
# monotone per-layer refinement). Changing this changes upper-layer input
# dims: not checkpoint-compatible with the other mode.
fwd_mode = "resid"
# per-prefix losses (every prefix of layers must reconstruct); enforces
# monotone per-layer refinement with fwd_mode="resid". NOTE: the loss.csv
# MSE column becomes the mean over the l prefix reconstructions.
deepsup = True
# stagewise deep supervision: stop-gradient the accumulated residual in each
# prefix decode, so layer i trains only against what layers <i left
# unexplained (pure RVQ/boosting credit assignment -- cleaner per-layer
# dictionary semantics, greedier optimum). False = joint: earlier layers
# also receive gradient from later prefix losses (the prototype-validated
# configuration). Gradient-only change; not a checkpoint compatibility issue.
deepsup_sg = False
# residual input conditioning. The first resid run collapsed to uniform
# upper-layer classifications: bilinear logits scale with ||input||^2, and
# raw residuals (norm ~0.25) left logit spreads at ~0.002*T -- flat
# softmax, vanishing gradients, classifier weights never left init.
# resid_norm classifies the unit-normalized residual direction instead.
resid_norm = True
# unbiased bilinear logits are also even functions -- blind to residual
# sign (+d and -d classify identically; NLinear's bias is added after the
# product, so biased_cl can't fix it). A constant 1 coordinate on every
# classifier input (layer 0's encoder output included) gives the quadratic
# form linear terms while keeping the pure-bilinear eigendecomposition
# analysis (over d+1 dims).
resid_const = True
scaled = False # skip scaling layer
encoded = False # skip encoder layer

b = 256 # batch size
# passes over the cached corpus, counted over the whole run: a resumed
# run skips the batches its checkpoint already trained on and stops at the
# same final step an uninterrupted one would. Raise it to extend a
# finished run.
epochs = _env("EPOCHS", 24, int)

lr = 5e-5 # bilinear MLPs require low learning rate for sability
wd = 0.0 # no weight decay

# temperature annealing: soft early (usage diversity, avoids head collapse),
# hard late (forces content into tag identity rather than soft coefficients)
temperature = 1.0
temperature_end = 0.03
anneal_steps = 50000

# input noise should scale with batch norm to account for nature of dataset
noise_in = "batchnorm"
sd_in = 0.0

# internal noise shouldn't to enforce consistent regularization
noise_K = "normal"
noise_F = "featvar"
sd_K = 0.02
sd_F = 0.1

# winner dropout: probability per head per sample that the argmax dict entry
# is masked, forcing the runner-up to win and receive gradient. Unlike logit
# noise, this reaches entries behind arbitrarily large margins (anti-collapse).
# Ramps linearly p_drop_start -> p_drop over anneal_steps: redundant while
# labels are soft, valuable in the hardening window. None disables the ramp.
p_drop = 0.1
p_drop_start = 0.0

# dead-entry revival: probability per head per sample that one dict entry
# nothing selected anywhere in the batch is promoted into the support so
# it receives gradient (DictBlock.revive_dead). Only meaningful under a
# hard select rule -- with ONTO_SELECT=top<k> an entry that leaves every
# sample's support gets exactly no gradient and can never return, which
# winner dropout cannot fix since it only ever promotes the runner-up.
p_revive = _env("P_REVIVE", 0.0, float)
# selection rate below which an entry counts as starved, as a fraction of
# the rate uniform usage would give it. Absence from the batch is far too
# strict a test at b=256: a head makes b*k_sel selections over k entries,
# so every entry is still picked occasionally long after usage has skewed.
revive_frac = _env("REVIVE_FRAC", 0.5, float)

# classifier noise schedule: sd_K enters the softmax as sd_K / T, so constant
# sd_K under the falling temperature would mean effective exploration growing
# to 0.02/0.03 = 0.67 logit units at the anneal floor -- 3x the historically
# calibrated hard-phase regime (sd_K=0.02 at T=0.1, effective 0.2). Annealing
# geometrically to sd_K_end = 0.2 * temperature_end lands exactly at that
# validated level. None disables (constant sd_K).
sd_K_end = 0.2 * temperature_end

# ghost-gradient dead-feature resurrection: only meaningful for relu-gated
# classifiers (ghostgrad fires on exact batch-wide zeros, which bilinear
# classifiers + softmax + abs()'d dicts never produce -- loss.csv showed
# MSE_ghost == MSE, i.e. the path was inert). Off by default, which saves
# ~a full extra model pass per step and reports MSE_ghost as 0; winner
# dropout is then the live anti-collapse mechanism. ONTO_GHOST=1 turns it
# on, which is only worth paying for alongside ONTO_GATE=relu -- there the
# classifier really can zero a unit across the whole batch, so dead units
# are both reachable and resurrectable rather than structurally absent.
ghost = _env("GHOST", False, lambda v: v.lower() in ("1", "true", "yes"))

# regularization terms
s_g = 1e-4 # ghost grad loss scale (no effect when ghost=False)
s_L1K = 0.0 # classifier output sparsity penalty (disrecommended)
# embedding sparsity penalty. NEEDS RECALIBRATION before it is trusted:
# until the sparse_F propagation fix, this flag never reached the DictEnc,
# so `Sparse.l1` always took its stop_gradient branch and this term applied
# NO gradient in any run -- at the live config it was 93% of the logged
# loss value and 0% of its gradient. 1e-9 was therefore tuned against a
# term that did nothing. Note also that `Sparse.l1` is an unreduced sum
# over the per-head features (b * h * e_dec = 16.8M elements here) while
# MSE is a mean, so the effective strength scales with batch size, head
# count and e_dec. ONTO_S_L1F=0 reproduces the old (inert) behaviour
# exactly, which is what the relu/top-k arms are compared under.
s_L1F = _env("S_L1F", 1e-9, float)

# hold `L1_F` at a setpoint instead of fixing `s_L1F`. A fixed coefficient
# has no equilibrium -- L1's gradient does not shrink with `F` while MSE's
# does, so it walks the code down to whatever bound exists and stalls there,
# and the best-looking coefficient is an artifact of the step budget. With
# ONTO_L1F_TARGET set, `s_L1F` becomes the multiplier the controller moves
# to sustain the target and the value above is only its starting point.
# Under ONTO_NORM_ROWS the reachable minimum is h*l (160 here), so a target
# is naturally expressed as a multiple of that: 480 is 3x the floor, where
# the pilot arms still reconstructed better than the unpenalized control.
# The setpoint is approached on a ramp rather than demanded immediately:
# the loop integrates, `L1_F` responds over thousands of steps, and asking
# for a target an order of magnitude away just winds the multiplier up to
# its bound. ONTO_L1F_RAMP is the ramp length in steps (anneal_steps when
# 0); it wants to be short enough to leave the run settled at the setpoint
# for most of training.
L1F_target = _env("L1F_TARGET", 0.0, float)
L1F_eta = _env("L1F_ETA", 1e-3, float)
L1F_ramp = _env("L1F_RAMP", 0, int)

s_H = 0.0 # classification entropy penalty
s_bcossim = 1e-5 # sample similarity penalty
s_hcossim = 1e-6 # head similarity penalty
# within-head dictionary row collinearity (DictBlock.rowcos), summed over
# layers. Measures head liveness rather than usage balance: a head whose
# entries all decode to one direction has an output independent of which
# entry wins, so s_Hm scores it perfectly while it carries nothing.
s_kcossim = _env("S_KCOSSIM", 0.0, float)
# hold the per-layer MAX of cossim_k at a setpoint instead of fixing the
# weight above. ONTO_KCOS_TARGET picks the ceiling: 0.2 is comfortably
# under the ~0.48 a collapsing layer passes through on its way up, so the
# penalty stays off while every layer is healthy and engages only on drift.
# The max rather than the sum, which falls monotonically through a collapse.
KCOS_target = _env("KCOS_TARGET", 0.0, float)
KCOS_eta = _env("KCOS_ETA", 1e-3, float)
KCOS_ramp = _env("KCOS_RAMP", 0, int)
# batch mean-entropy bonus: KL(E_batch[p] ‖ uniform) in bits, mean over
# heads, summed over layers. Entropy of the mean, not mean of the entropies:
# taxes usage imbalance without softening per-sample classifications.
# Anchors the generic code point at the uniform mixture (the resid_nc probe
# showed E[p] decoding to the corpus mean while the uniform code lands
# off-manifold, cos 0.067 to E[X]) and opposes hard-phase head freezing
# (resid_nc layer 4 froze 22/32 heads after step ~211k). Calibration: a
# fully frozen layer pays ~5 bits -> 5e-6 loss at this scale, ~28x the
# measured MSE harm of the frozen layer (+1.8e-7 whitened); healthy layers
# (~0.1 bits) pay ~1e-7, well under the final MSE ~3e-6. NOTE: adds a 9th
# loss.csv column (KL_m) from the step this lands.
# ONTO_S_HM overrides. The calibration above was derived for softmax
# selection; under top-k this term sat at ~0.7% of the loss while the
# dictionary still collapsed to ~9 of 32 effective entries per head, so
# 1e-6 is a starting point there rather than a validated level.
s_Hm = _env("S_HM", 1e-6, float)

# target whitening: per-dim inverse-variance MSE weights (mean-1 normalized,
# computed from the step-97600 census corpus embeddings). Equal-weights dims
# so dictionary capacity shifts toward low-variance directions; in the GPT-2
# prototype this took the dictionary from 401 to 920 live entries at no cost
# to raw-space reconstruction. Set to None to disable. NOTE: the loss.csv
# MSE column becomes weighted MSE from the step this is enabled.
mse_weights = str(path / "out/sonar/mse_weights.npy")

def main():
    # serialization metadata & preprocessing hyperparameters. Built first
    # because the streaming corpus below is one of its products.
    meta = Metadata(
            "embedding" if cache else "text", encoder, out_path=out,
            threads=threads,
            save_each=save_each, checkpoint_each=checkpoint_each,
            stopping_strategy=stopping_strategy)

    # HuggingFace training data
    #ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    if cache:
        src = NpyDataSource(cache, holdout=holdout)
    else:
        src = meta.mc4("allenai/c4", split="train", streaming=True)

    # training hyperparameters
    hyper = Hyperparams(
            d, d, b, epochs, lr, wd, temperature,
            noise_in, noise_K, noise_F, sd_in, sd_K, sd_F,
            s_g=s_g, s_L1K=s_L1K, s_L1F=s_L1F, s_H=s_H,
            L1F_target=L1F_target, L1F_eta=L1F_eta, L1F_ramp=L1F_ramp,
            s_bcossim=s_bcossim, s_hcossim=s_hcossim,
            s_kcossim=s_kcossim, KCOS_target=KCOS_target,
            KCOS_eta=KCOS_eta, KCOS_ramp=KCOS_ramp, s_Hm=s_Hm,
            p_drop=p_drop, p_revive=p_revive, revive_frac=revive_frac,
            mse_weights=mse_weights,
            temperature_end=temperature_end, anneal_steps=anneal_steps,
            p_drop_start=p_drop_start, sd_K_end=sd_K_end,
            ghost=ghost,
            )

    # model configuration
    model = hyper.ontologizer(
            d, d, e_dec, k, h, l, n=n, gate=gate, select=select,
            norm_rows=norm_rows,
            scaled=scaled, encoded=encoded, e_enc=e_enc,
            forward=fwd_mode, deepsup=deepsup, deepsup_sg=deepsup_sg,
            resid_norm=resid_norm, resid_const=resid_const,
            dtype_str=dtype_str, dtype_p_str=dtype_p_str)

    # load pretrained model (bypassed when training from cached embeddings)
    if cache:
        pretrained = None
        kwargs_loader = {"d": d, "shuffle": shuffle}
    else:
        pretrained, tokenizer = meta.model(dtype_str)
        kwargs_loader = {"tokenizer": tokenizer, "maxlen": maxlen}

    # initialize training environment
    env = TrainingEnv(model, hyper, meta, kwargs_loader=kwargs_loader)

    # training loop
    state = env.train(src, encoder=pretrained)
    print("Training Complete!")

    plot_loss(out, base=2)

if __name__ == "__main__":
    main()
