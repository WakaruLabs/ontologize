# disable preallocation so jax and torch can share VRAM
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"

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

save_each = 100 # how often to save model & write loss
checkpoint_each = 10000 # how often to keep checkpoints indefinitely
out = path / "out/sonar/multilingual/resid_nc_hm" # output directory (fresh
# dir: resid_const changes upper-layer classifier input dims -- not
# checkpoint-compatible with the first resid run, which is preserved at
# multilingual/resid)

# Ontologizer spec
d = 1024 # input & output dimension
e_enc = 2048 # encoder width
e_dec = 2048 # decoder height
k = 32 # number of tags per head
h = 32 # number of heads
l = 5 # number of DictEnc layers; must match the checkpoint when resuming

n = 2 # bilinear classifier
gate = "none" # pure bilinear classifier
#n = 1 # linear classifier
#gate = "relu"

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
# product, so biased_cl can't fix it). A constant 1 coordinate gives the
# quadratic form linear terms in the residual while keeping the
# pure-bilinear eigendecomposition analysis (over d+1 dims).
resid_const = True
scaled = False # skip scaling layer
encoded = False # skip encoder layer

b = 256 # batch size
epochs = 24 # overtraining seems to exacerbate mode collapse

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
# MSE_ghost == MSE, i.e. the path was inert). Disabled: saves ~a full extra
# model pass per step; the MSE_ghost column reports 0 from the step this
# lands. Winner dropout is the live anti-collapse mechanism.
ghost = False

# regularization terms
s_g = 1e-4 # ghost grad loss scale (no effect when ghost=False)
s_L1K = 0.0 # classifier output sparsity penalty (disrecommended)
s_L1F = 1e-9 # embedding sparsity penalty
s_H = 0.0 # classification entropy penalty
s_bcossim = 1e-5 # sample similarity penalty
s_hcossim = 1e-6 # head similarity penalty
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
s_Hm = 1e-6

# target whitening: per-dim inverse-variance MSE weights (mean-1 normalized,
# computed from the step-97600 census corpus embeddings). Equal-weights dims
# so dictionary capacity shifts toward low-variance directions; in the GPT-2
# prototype this took the dictionary from 401 to 920 live entries at no cost
# to raw-space reconstruction. Set to None to disable. NOTE: the loss.csv
# MSE column becomes weighted MSE from the step this is enabled.
mse_weights = str(path / "out/sonar/mse_weights.npy")

def main():
    # HuggingFace training data
    #ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    if cache:
        src = NpyDataSource(cache)
    else:
        ds = mc4_data("allenai/c4", split="train", streaming=True)
        src = HFDataSource(ds, text_key="text")

    # training hyperparameters
    hyper = Hyperparams(
            d, d, b, epochs, lr, wd, temperature,
            noise_in, noise_K, noise_F, sd_in, sd_K, sd_F,
            s_g=s_g, s_L1K=s_L1K, s_L1F=s_L1F, s_H=s_H,
            s_bcossim=s_bcossim, s_hcossim=s_hcossim, s_Hm=s_Hm,
            p_drop=p_drop, mse_weights=mse_weights,
            temperature_end=temperature_end, anneal_steps=anneal_steps,
            p_drop_start=p_drop_start, sd_K_end=sd_K_end,
            ghost=ghost,
            )

    # model configuration
    model = hyper.ontologizer(
            d, d, e_dec, k, h, l, n=n, gate=gate, scaled=scaled,
            encoded=encoded, e_enc=e_enc,
            forward=fwd_mode, deepsup=deepsup, deepsup_sg=deepsup_sg,
            resid_norm=resid_norm, resid_const=resid_const,
            dtype_str=dtype_str, dtype_p_str=dtype_p_str)

    # serialization metadata & preprocessing hyperparameters
    meta = Metadata(
            "embedding" if cache else "text", encoder, out_path=out,
            threads=threads,
            save_each=save_each, checkpoint_each=checkpoint_each)

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
