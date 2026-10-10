"""Auto-interp description adequacy (Bills et al. style detection scoring),
run identically over Ontologizer tags and SAE latents so the two methods'
interpretability is directly comparable.

Pipeline (stages write artifacts to --out and are resumable/re-runnable):

  1. harvest   activations over a cache slice -> per-feature top-activating
               rows, activation quantiles, firing rate; stratified feature
               selection; low-activation negative rows; and the feature's
               parameter-based decode EMBEDDING (Ontologizer: generic-origin
               code with the tag forced one-hot, through the model's own
               decoder; SAE: b_dec + alpha * W_dec row, alpha = mean top
               activation). Needs the checkpoint; JAX.
  2. texts     recover corpus TEXT for every row any features.npz needs, by
               re-streaming mC4 in the exact encode_corpus.py order (the
               cache stores no text; the stream order is deterministic --
               encode_corpus's own resume depends on that). Run once with
               every features.npz listed; network-bound.
  3. describe  two descriptions per feature: mode "params" decodes the
               stage-1 embedding through the SONAR/M2M100 decoder (forced
               eng_Latn; torch, no LLM); mode "acts" has an LLM judge
               summarize the top-activating texts (ranks 1..n_desc); mode
               "cacts" is the contrastive variant -- the judge sees the top
               texts NEXT TO random corpus draws and must state what
               distinguishes them, which blocks the degenerate corpus-level
               answer ("noisy non-English web scrape") the plain prompt
               invites on mC4.
  4. score     detection: for each (feature, description), the blinded
               judge sees shuffled held-out positives (ranks n_desc+1..)
               and low-activation negatives and marks which match the
               description; per-feature precision/recall/F1 to scores.csv.
               --null adds the chance floor: the same items re-graded
               against a DIFFERENT feature's same-mode description (kind
               column), so F1 - null F1 isolates what the description's
               content earns.
  5. language  no model, no judge: detection by the language spread of
               each feature's top rows, over every campaign under --out.
               Per campaign and mode, F1 and accuracy by how many distinct
               languages the rows a description was written from and
               graded on span; language purity (the description rows'
               modal-language share) against F1 and firing rate; and the
               firing-rate trend of mean F1 refitted on the multilingual
               features alone. Writes <out>/language/.

  match        optional, after harvest: language-matched rows for
               `describe --mode cacts_lang` and `score --negatives lang`.
               Per feature, negatives in the languages of its held-out
               positives and a contrastive background in the languages of
               its description rows, both drawn from the rows below the
               feature's median activation, the pool harvest draws its
               negatives from. Needs the checkpoint; writes
               <dir>/lang_rows.npz, whose rows the texts stage recovers
               like any other.

Fairness notes: prompts never mention the method; positives for scoring
are disjoint from the texts used to write the description; features are
sampled stratified by firing-rate decile; "activation" is each method's
natural quantity (tag probability / latent coefficient), whose per-feature
ranking is unaffected by the deviation-from-origin reading. Negatives are
random low-activation rows, so a description can be penalized for a true
but overly broad property -- that noise is shared by both methods. They
are not matched for language: mC4 cycles through 86 languages, so naming
the language a feature's top rows share separates them from random rows
whatever else the feature responds to. `match` with `--mode cacts_lang`
and `--negatives lang` holds language fixed between the two sides. The
judge cannot cheat by inspection: it is a bare Messages API call -- one
user turn, no tools, no system prompt, no filesystem or repo context -- so
the description string and the snippets are the only bytes it ever sees,
never the checkpoint or which method is under test. It CAN cheat by prior
(the item set is a fixed 50/50 split, and top-activating rows differ from
non-firing ones in surface register), which is what --null measures.

--dry-run writes judge prompts to disk instead of calling. Example:

  uv run python autointerp.py harvest --model onto \\
      --ckpt data/out/sonar/multilingual/resid_nc --n-features 256
  uv run python autointerp.py harvest --model sae \\
      --ckpt data/out/sonar/sae/m5120_k32/params.npz --topk 32 --n-features 256
  uv run python autointerp.py texts \\
      --features data/out/sonar/autointerp/onto/features.npz \\
                 data/out/sonar/autointerp/sae/features.npz
  uv run python autointerp.py describe --dir data/out/sonar/autointerp/onto
  uv run python autointerp.py score --dir data/out/sonar/autointerp/onto
  uv run python autointerp.py language          # every campaign under --out

  # language-matched rescore of an existing campaign
  uv run python autointerp.py match --dir data/out/sonar/autointerp/onto/onto
  uv run python autointerp.py texts \\
      --features data/out/sonar/autointerp/onto/onto/lang_rows.npz
  uv run python autointerp.py describe --mode cacts_lang \\
      --dir data/out/sonar/autointerp/onto/onto
  uv run python autointerp.py score --negatives lang \\
      --only-modes acts cacts cacts_lang \\
      --dir data/out/sonar/autointerp/onto/onto
"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import csv
import json
import re
import threading
import time
import urllib.error
import urllib.request
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from jaxtyping import Bool, Float, Int, Shaped
from pathlib import Path

ROOT = Path(__file__).parent
CACHE = ROOT / "data/sonar_embeddings/mc4_4M.npy"
OUT = ROOT / "data/out/sonar/autointerp"
ONTO_CKPT = "data/out/sonar/multilingual/resid_nc"
DECODER_ID = "raxtemur/SONAR_200_text_decoder"
ENCODER_ID = "cointegrated/SONAR_200_text_encoder"

SUMMARIZE_PROMPT = """\
The following text snippets (possibly in different languages) all strongly \
activate the same feature of a text encoder. Identify the most specific \
property they share.

{samples}

Reply with ONE English sentence describing the shared property. No preamble."""

CONTRAST_PROMPT = """\
The Group A text snippets (possibly in different languages) all strongly \
activate the same feature of a text encoder. The Group B snippets are drawn \
at random from the same corpus and do not activate it.

Group A:
{pos}

Group B:
{neg}

Identify the most specific property that distinguishes the Group A snippets \
from the Group B snippets. Do not describe properties shared by both groups \
(such as web-scraping noise, broken formatting, or simply not being English, \
if Group B shows them too).

Reply with ONE English sentence describing the distinguishing property of \
the Group A snippets. No preamble."""

DETECT_PROMPT = """\
A feature of a text encoder is described as:
"{desc}"

Which of the following numbered text snippets (possibly in different \
languages) match that description?

{samples}

Reply with ONLY a JSON list of the matching snippet numbers, e.g. [1,4]. \
Reply [] if none match."""


# ---------- pure helpers (unit-tested) ----------

def merge_topk(vals, idxs, new_vals, new_idxs, n):
    """Running per-feature top-n merge. vals/new_vals: (F, *), descending
    order not required on input; returns (F, n) sorted descending."""
    v = np.concatenate([vals, new_vals], axis=1)
    i = np.concatenate([idxs, new_idxs], axis=1)
    order = np.argsort(-v, axis=1)[:, :n]
    return (np.take_along_axis(v, order, axis=1),
            np.take_along_axis(i, order, axis=1))


def stratified_features(freq, n, seed):
    """Sample n features stratified over firing-rate deciles (all features
    when n == 0 or n >= len(freq)). Deterministic in seed."""
    if not n or n >= len(freq):
        return np.arange(len(freq))
    rng = np.random.default_rng(seed)
    deciles = np.quantile(freq, np.linspace(0, 1, 11)[1:-1])
    bins = np.digitize(freq, deciles)
    picks = []
    per = max(1, n // 10)
    for b in range(10):
        pool = np.where(bins == b)[0]
        if len(pool):
            picks.append(rng.choice(pool, min(per, len(pool)), replace=False))
    picks = np.concatenate(picks) if picks else np.zeros(0, np.int64)
    if len(picks) > n:
        picks = rng.choice(picks, n, replace=False)
    elif len(picks) < n:
        # degenerate freq (e.g. a dense softmax code where every latent
        # "fires" identically) collapses the deciles into one bin; top up
        # uniformly instead of silently returning a tiny sample
        rest = np.setdiff1d(np.arange(len(freq)), picks)
        picks = np.concatenate([picks, rng.choice(
            rest, min(n - len(picks), len(rest)), replace=False)])
    return np.sort(picks)


def parse_judge(ans, n_items):
    """Extract the judged item numbers from a (possibly chatty) reply."""
    m = re.search(r"\[[\d,\s]*\]", ans)
    if m:
        nums = re.findall(r"\d+", m.group(0))
    else:
        nums = re.findall(r"\d+", ans)
    return sorted({int(x) for x in nums if 1 <= int(x) <= n_items})


def prf(pred, truth):
    pred, truth = set(pred), set(truth)
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def contrast_rows(f, top_idx, neg_idx, pool, n_desc, n_test, seed):
    """Background rows for the contrastive description prompt: random
    low-activation corpus draws (the union of all features' negative pools),
    excluding this feature's top rows and its held-out scoring negatives so
    the description never sees what detection will grade it on."""
    excl = np.concatenate([top_idx, neg_idx[:n_test]])
    bg = np.setdiff1d(pool, excl)
    rng = np.random.default_rng(seed + f)
    return rng.choice(bg, min(n_desc, len(bg)), replace=False)


def detection_items(f, top_idx, neg_idx, n_desc, n_test, seed):
    """Held-out positives (ranks n_desc..n_desc+n_test) + negatives,
    shuffled; returns (rows, truth 1-based positions of positives)."""
    pos = top_idx[n_desc:n_desc + n_test]
    rows = np.concatenate([pos, neg_idx[:n_test]])
    order = np.random.default_rng(seed + f).permutation(len(rows))
    rows = rows[order]
    truth = [int(i) + 1 for i, o in enumerate(order) if o < len(pos)]
    return rows, truth


def null_pairing(feats, seed, texts=None):
    """Derangement over feature ids: {feature -> the OTHER feature whose
    description is used as its null control}. Empty when < 2 features.

    With `texts` ({feature -> description}) the donor is additionally
    required to differ in TEXT, not just in id: descriptions repeat verbatim
    across features (degenerate params decodes especially), and a duplicate
    donor is a self-grade wearing a different feature id. Features with no
    valid donor are dropped -- no null exists for them.

    The null re-grades a feature's own detection items against a mismatched
    description of the same mode, so item set, base rate, judge and prompt
    are all held fixed and only the description's content changes. F1 minus
    null F1 is the part of the score the description actually earns: a judge
    can score well above zero with no information by exploiting the fixed
    50/50 positive rate, or by keying on surface differences between
    top-activating and non-firing rows (length, script, boilerplate) that
    correlate with the feature without being what the description says."""
    feats = list(feats)
    if len(feats) < 2:
        return {}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(feats))
    for i in range(len(perm)):
        if perm[i] == i:  # a permutation has one preimage per index, so the
            j = (i + 1) % len(perm)  # swap can't create a new fixed point
            perm[i], perm[j] = perm[j], perm[i]
    pair = {feats[i]: feats[int(perm[i])] for i in range(len(feats))}
    if texts is None:
        return pair
    # repair duplicate-text donors by swapping two features' donors, which
    # preserves the bijection; deterministic candidate order
    order = [feats[int(k)] for k in rng.permutation(len(feats))]
    for f in [f for f in feats if texts[pair[f]] == texts[f]]:
        for g in order:
            if g != f and pair[g] != f and pair[f] != g \
                    and texts[pair[g]] != texts[f] \
                    and texts[pair[f]] != texts[g]:
                pair[f], pair[g] = pair[g], pair[f]
                break
    return {f: d for f, d in pair.items() if texts[d] != texts[f]}


def cap_pairing(pair, n, seed):
    """Keep at most n of the null pairs. The null is only an estimate of a
    floor, so its cost can be cut well below the self scores' -- the
    standard error at n=100 is already far under the effect sizes here.
    Deterministic in seed: a resume must re-pick the SAME subset or it pays
    for a second, disjoint sample instead of finishing the first."""
    if not n or n >= len(pair):
        return pair
    feats = sorted(pair)
    keep = np.random.default_rng(seed).choice(len(feats), n, replace=False)
    return {feats[int(i)]: pair[feats[int(i)]] for i in sorted(keep)}


SCORE_COLS = ["feature", "mode", "kind", "precision", "recall", "f1"]


def read_scores(path):
    """Existing rows, upgraded to the 6-column schema. Rows written before
    the null control landed have no `kind` and are all self-scored."""
    if not path.exists():
        return [], False
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    legacy = any("kind" not in r or r["kind"] is None for r in rows)
    for r in rows:
        r["kind"] = r.get("kind") or "self"
    return rows, legacy


# description modes a judge writes, the ones the language stage reports
JUDGE_MODES = ("acts", "cacts", "cacts_lang")
# row arrays of features.npz / lang_rows.npz whose texts the texts stage
# recovers
ROW_KEYS = ("top_i", "neg_i", "bg_i")


def langs_path(cache):
    """The cache's per-row language sidecar, written by encode_corpus.py."""
    cache = Path(cache)
    return cache.with_name(cache.name.replace(".npy", ".langs.npy"))


def needed_rows(paths):
    """Every cache row the texts stage must recover for these files."""
    need = set()
    for path in paths:
        dat = np.load(path)
        for key in ROW_KEYS:
            if key in dat:
                need |= set(dat[key].ravel().tolist())
    return need


def modal_language(labels: Shaped[np.ndarray, "n"]) -> tuple[str, float]:
    """The most common label and its share; a tie goes to the label that
    sorts first."""
    u, c = np.unique(labels, return_counts=True)
    i = int(c.argmax())
    return str(u[i]), float(c[i]) / len(labels)


def detection_accuracy(precision: float, recall: float,
                       n_test: int) -> tuple[float, float]:
    """Accuracy of one detection item set of n_test positives and n_test
    negatives, recovered from its precision and recall as a (low, high)
    bracket. TP = n_test * recall and FP = TP / precision - TP give the
    accuracy (TP + n_test - FP) / (2 n_test) exactly when TP > 0; with no
    true positive the false positives are unknown and it lies in [0, 1/2]."""
    tp = round(n_test * recall)
    if tp == 0:
        return 0.0, 0.5
    acc = (tp + n_test - round(tp / precision - tp)) / (2 * n_test)
    return acc, acc


def matched_rows(target: Shaped[np.ndarray, "n"], cand: Int[np.ndarray, "c"],
                 cand_langs: Shaped[np.ndarray, "c"], rng: np.random.Generator
                 ) -> tuple[Int[np.ndarray, "n"], int]:
    """len(target) distinct rows of `cand`, the i-th in language target[i]
    wherever enough candidates are in that language, drawn without
    replacement; a position whose language has run out takes a remaining
    candidate at random. Returns the rows and how many are matched."""
    if len(cand) < len(target):
        raise ValueError(f"{len(cand)} candidates for {len(target)} rows")
    out = np.full(len(target), -1, np.int64)
    free = np.ones(len(cand), bool)
    for lang in np.unique(target):
        want = np.flatnonzero(target == lang)
        have = np.flatnonzero(free & (cand_langs == lang))
        pick = rng.choice(have, min(len(want), len(have)), replace=False)
        out[want[:len(pick)]] = cand[pick]
        free[pick] = False
    short = np.flatnonzero(out < 0)
    out[short] = cand[rng.choice(np.flatnonzero(free), len(short),
                                 replace=False)]
    return out, len(target) - len(short)


def span_bins(edges, n_rows):
    """Labels of the language-count bins with these sorted lower edges, the
    last closed at n_rows, the most languages n_rows rows can span."""
    his = [e - 1 for e in edges[1:]] + [n_rows]
    return [f"{lo}-{hi}" for lo, hi in zip(edges, his)]


def spearman(a, b):
    """Rank correlation; NaN when either side is constant."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def ols_r2(y: Float[np.ndarray, "n"], *cols: Float[np.ndarray, "n"]) -> float:
    """Share of y's variance explained by least squares on `cols` and an
    intercept."""
    X = np.column_stack([np.ones(len(y)), *cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - (y - X @ beta).var() / y.var())


def trend_residuals(freq: Float[np.ndarray, "r"], f1: Float[np.ndarray, "r"],
                    fit: Bool[np.ndarray, "r"]) -> Float[np.ndarray, "r"]:
    """Each campaign's mean F1 less the least-squares line in log firing
    rate through the campaigns in `fit`; NaN for a code that never fires,
    which has no log rate."""
    fires = freq > 0
    slope, icept = np.polyfit(np.log(freq[fit & fires]), f1[fit & fires], 1)
    with np.errstate(divide="ignore"):
        line = icept + slope * np.log(freq)
    return np.where(fires, f1 - line, np.nan)


def random_span(counts: Float[np.ndarray, "c"], n: int) -> float:
    """Expected number of distinct labels among n rows drawn at random, with
    replacement, from a corpus with these label counts."""
    p = np.asarray(counts, float) / np.sum(counts)
    return float(np.sum(1 - (1 - p) ** n))


# ---------- judge plumbing ----------

def next_slot(prev, now, gap):
    """Token-bucket slot assignment: earliest permitted call time given the
    previously scheduled slot. Idle periods don't accumulate a burst."""
    t = max(prev, now)
    return t, t + gap


# the unit `rate` was calibrated in: one plain 10-snippet describe prompt.
# API spend is token-metered, not call-metered, so a call's bucket charge
# scales with its prompt size (cacts prompts carry 2x the snippets and
# cost ~2 units)
CAL_PROMPT_CHARS = 2500


def call_weight(prompt):
    return max(0.25, len(prompt) / CAL_PROMPT_CHARS)


API_URL = "https://api.anthropic.com/v1/messages"
JUDGE_MODELS = {"haiku": "claude-haiku-4-5", "sonnet": "claude-sonnet-5"}


def make_llm(judge_model, dry_dir, jobs, rate=0):
    calls = []
    model = JUDGE_MODELS.get(judge_model, judge_model)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key.startswith("/") and Path(key).is_file():
        # the var may hold a path to a token file rather than the key itself,
        # the convention on this machine; sending the path gets a 401 that
        # reads as a revoked key
        key = Path(key).read_text().strip()
    if not dry_dir and not key:
        # never fall back to anything implicit: the campaign bills this key
        raise RuntimeError("ANTHROPIC_API_KEY unset; refusing to run judge")

    # rate meters ALL judge calls (retries included) to `rate` plain-prompt-
    # equivalent cost units per hour, evenly spaced across workers -- on API
    # billing this just smooths the request rate under the org's rate limits
    gap = 3600.0 / rate if rate else 0.0
    bucket = {"t": 0.0}
    lock = threading.Lock()

    def pace(weight):
        if not gap:
            return
        now = time.monotonic()
        with lock:
            t, bucket["t"] = next_slot(bucket["t"], now, gap * weight)
        time.sleep(max(0.0, t - now))

    fails = []  # consecutive-failure circuit breaker (shared across workers)

    def llm(prompt, tag):
        if dry_dir:
            (dry_dir / f"{tag}.txt").write_text(prompt)
            return ""
        err = "timeout"
        for i in range(3):
            if len(fails) >= 3:
                # a quota/rate-limit outage fails every call: stop burning
                # quota on retries and abort the stage (resume redoes the rest)
                raise RuntimeError(
                    f"judge circuit breaker tripped; last error: {fails[-1]}")
            time.sleep(0 if i == 0 else 15 * 4 ** i)  # 60s, 240s backoff
            pace(call_weight(prompt))
            try:
                # direct Messages API call: a call pays only the prompt
                # (~1k tokens) + reply, not the ~10.6k tokens of claude -p
                # CLI context that dominated per-call cost. max_tokens
                # bounds the runaway replies degenerate descriptions
                # provoke (8.5k-token rambles measured on detect calls);
                # worst case is now ~$0.02/call, so the old --max-budget-usd
                # cap has no job left to do
                req = urllib.request.Request(
                    API_URL,
                    data=json.dumps({
                        "model": model, "max_tokens": 4000,
                        "messages": [{"role": "user", "content": prompt}],
                    }).encode(),
                    headers={"x-api-key": key,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.load(r)
                text = "".join(
                    b.get("text", "") for b in resp["content"]).strip()
                if text:
                    fails.clear()
                    return text
                err = f"empty reply (stop_reason={resp.get('stop_reason')})"
            except urllib.error.HTTPError as e:
                # 429/529 land here and ride the existing backoff/breaker;
                # the body carries the API error type and message
                err = f"HTTP {e.code}: {e.read().decode(errors='replace')[-300:]}"
            except (urllib.error.URLError, OSError) as e:
                err = str(e)[:300]
            except (ValueError, KeyError) as e:
                err = f"bad response: {e!r}"[:300]
        fails.append(err)
        # abort loudly: silently returning "" poisons scores with F1=0
        raise RuntimeError(f"judge call {tag} failed 3x; last error: {err}")

    def llm_map(items):  # [(prompt, tag)] -> lazy replies, in order
        ex = ThreadPoolExecutor(max_workers=jobs)
        try:
            # lazy so callers can persist each reply as it arrives; a
            # mid-stage abort then only costs the in-flight items
            yield from ex.map(lambda a: llm(*a), items)
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return llm_map


def snippet(text, maxlen=240):
    text = " ".join(text.split())
    return text[:maxlen]


# ---------- stage: harvest ----------

def onto_probe(module, X, temperature, inputs=False):
    """Every layer's assignments (b, l, h, k) on the plain forward pass at
    `temperature`; with inputs=True also each layer >= 1's input as its
    classifier sees it (see `onto_acts_fn`). Run under `model.apply`. Each
    layer writes `DictEnc.head_outputs`, which applies the router gain and
    fibers as the forward does; without them every layer past the first
    would classify another model's residual."""
    import jax.numpy as jnp
    E, _ = module.encode(X, 0.0, None)
    R = module.resid(E)
    E_in = module.constinput(E)
    Ps, Us = [], []
    resid_fwd = module.forward in ("resid", "resid_labels")
    # reproduce the DictEnc's gain-shape split: under `resid_gain` it
    # classifies the unit-norm SHAPE of its input and scales its
    # contribution by the measured GAIN. Identity / no-op when off.
    for i, de in enumerate(module.dictencs):
        U, G = de.gainshape_in(E_in)
        if inputs and i > 0:
            # the residual leads the input, X's width wide
            Us.append(U[..., :X.shape[-1]] if resid_fwd else U)
        P = de.dict.cluster(de.classifier(U), temperature)
        Ps.append(P)
        R = R + de.gained(de.dict.combine(de.head_outputs(U, P)), G)
        if i < module.l - 1:
            E_in = module.nextinput(X, R, P.reshape(P.shape[0], -1))
    P = jnp.stack(Ps, 1)  # (b, l, h, k)
    return (P, tuple(Us)) if inputs else P


def onto_acts_fn(ckpt, step, temperature, inputs=False):
    """Load an Ontologizer checkpoint and return (acts, embed, F, meta):
    acts(X) is the flattened (b, l*h*k) assignment code, embed decodes
    constant codes. With inputs=True a fifth element, layer_inputs(X),
    returns each layer >= 1's own input as the classifier sees it: the
    output-space residual (under a residual forward mode; the constant and
    previous-code coordinates dropped) or the forwarded code (under
    forward="labels"). Layer 0's input is X itself, so it is not repeated."""
    import jax
    import jax.numpy as jnp
    from pareto import load_onto

    model, params, step = load_onto(ckpt, step)
    params = {'params': params}
    l, h, k = model.l, model.h, model.k

    @jax.jit
    def acts(X):
        P = model.apply(params, X, temperature, method=onto_probe)
        return P.reshape(X.shape[0], -1)  # feature f = (l*h + h_i)*k + k_i

    def embed(codes):
        """Decode constant codes (F, l, h, k) through the dict + decoder.
        There is no input here, so the router gain and fibers, which read
        it, cannot apply: this is the base dictionary's decode."""
        def probe_c(module, Ps):
            R = jnp.zeros((Ps.shape[0], module.e_lat))
            for i, de in enumerate(module.dictencs):
                R = R + de.place(de.dict.combine(de.dict.hfwd(Ps[:, i])))
            return module.decode(R)
        return np.asarray(model.apply(params, jnp.asarray(codes),
                                      method=probe_c))

    meta = {"model": "onto", "step": int(step), "l": l, "h": h, "k": k,
            "temperature": temperature}
    if not inputs:
        return acts, embed, l * h * k, meta

    @jax.jit
    def layer_inputs(X):
        return model.apply(params, X, temperature, True, method=onto_probe)[1]

    return acts, embed, l * h * k, meta, layer_inputs


def entry_directions(embed, l, h, k, chunk=512):
    """Every entry's decoded direction from `onto_acts_fn`'s `embed`:
    (dirs, offset) with dirs[f] = embed(one-hot f) - embed(0) for
    f = (layer*h + head)*k + entry, and offset = embed(0), the decoder
    bias (0 without `biased_dec`). embed is affine in the code, so
    embed(P) = P_flat @ dirs + offset exactly; subtracting the offset
    keeps it out of every direction, where it would otherwise be counted
    once per head. Decoded `chunk` one-hot codes at a time, so the
    (F, F) identity is never built."""
    F = l * h * k
    offset = np.asarray(embed(np.zeros((1, l, h, k), np.float32)))[0]
    dirs = np.empty((F, len(offset)), np.float32)
    for i in range(0, F, chunk):
        f = np.arange(i, min(i + chunk, F))
        codes = np.zeros((len(f), F), np.float32)
        codes[np.arange(len(f)), f] = 1.0
        dirs[f] = np.asarray(embed(codes.reshape(len(f), l, h, k))) - offset
    return dirs, offset


def sae_acts_fn(ckpt, topk):
    import jax
    import jax.numpy as jnp
    import sae as sae_mod

    raw = np.load(ckpt)
    params = {name: jnp.asarray(raw[name]) for name in raw.files}
    m = params["W_dec"].shape[0]  # W_enc is absent on bilinear runs

    # sae.py runs record their encode rule in meta.json; --topk is only
    # the fallback for pre-meta runs
    meta_path = Path(ckpt).parent / "meta.json"
    groups, group_fn = 0, "top1"
    if meta_path.exists():
        run_meta = json.loads(meta_path.read_text())
        topk, groups, group_fn = (run_meta["topk"], run_meta["groups"],
                                  run_meta["group_fn"])

    @jax.jit
    def acts(X):
        return sae_mod.encode(params, X, topk, groups, group_fn)

    meta = {"model": "sae", "ckpt": str(ckpt), "m": m, "topk": topk,
            "groups": groups, "group_fn": group_fn,
            "enc": "bilinear" if "W_enc2" in params else "linear"}
    return acts, params, m, meta


def fire_threshold(meta):
    """The activation above which a feature fires. An Ontologizer entry's
    assignment and a grouped-softmax latent are strictly positive simplex
    weights, so firing means meaningfully above uniform (2/k); any other
    latent fires when nonzero."""
    if meta["model"] == "onto":
        return 2.0 / meta["k"]
    if meta.get("group_fn") == "softmax" and meta.get("groups"):
        return 2.0 / (meta["m"] // meta["groups"])
    return 0.0


def harvest(cfg):
    import jax.numpy as jnp

    out = Path(cfg.out) / cfg.model
    out.mkdir(parents=True, exist_ok=True)
    if cfg.model == "onto":
        acts_fn, embed_fn, F, meta = onto_acts_fn(
            cfg.ckpt or ONTO_CKPT, cfg.step, cfg.temperature)
    else:
        acts_fn, sae_params, F, meta = sae_acts_fn(cfg.ckpt, cfg.topk)

    mm = np.load(cfg.cache, mmap_mode="r")
    n = min(cfg.rows, mm.shape[0])
    b = cfg.b
    top_v = np.full((F, cfg.n_top), -np.inf, np.float32)
    top_i = np.zeros((F, cfg.n_top), np.int64)
    sub_rows, sub_acts = [], []  # activation subsample for quantiles/negatives
    stride = max(1, (n // b) // cfg.sub_batches)

    from tqdm import tqdm
    for bi, i in enumerate(tqdm(range(0, n - b + 1, b), desc="harvest")):
        A = np.asarray(acts_fn(jnp.asarray(
            np.asarray(mm[i:i + b], dtype=np.float32))))
        top_v, top_i = merge_topk(
            top_v, top_i, A.T,
            np.broadcast_to(np.arange(i, i + b), (F, b)), cfg.n_top)
        if bi % stride == 0:
            sub_rows.append(np.arange(i, i + b))
            sub_acts.append(A.astype(np.float16))
    sub_rows = np.concatenate(sub_rows)
    sub_acts = np.concatenate(sub_acts)  # (S, F)

    freq = (sub_acts > fire_threshold(meta)).mean(0)
    if getattr(cfg, "feature_ids", None):
        sel = np.asarray(sorted(set(cfg.feature_ids)), np.int64)
        assert sel.max() < sub_acts.shape[1] and sel.min() >= 0, \
            f"feature ids must be in [0, {sub_acts.shape[1]})"
    else:
        sel = stratified_features(freq, cfg.n_features, cfg.seed)

    q50 = np.quantile(sub_acts[:, sel].astype(np.float32), 0.5, axis=0)
    rng = np.random.default_rng(cfg.seed)
    neg_i = np.zeros((len(sel), cfg.n_neg), np.int64)
    for j, f in enumerate(sel):
        # <=: a sparse latent's median activation is exactly 0, and the
        # non-firing rows are precisely the negatives pool
        pool = sub_rows[sub_acts[:, f] <= q50[j]]
        pool = np.setdiff1d(pool, top_i[f])
        neg_i[j] = rng.choice(pool, cfg.n_neg, replace=len(pool) < cfg.n_neg)

    # parameter-based decode embeddings for the selected features
    if cfg.model == "onto":
        l, h, k = meta["l"], meta["h"], meta["k"]
        A_top = sub_acts.astype(np.float32)
        origin = A_top.mean(0).reshape(l, h, k)   # measured E[p] on the slice
        origin /= origin.sum(-1, keepdims=True)
        codes = np.broadcast_to(origin, (len(sel), l, h, k)).copy()
        for j, f in enumerate(sel):
            li, hi, ki = f // (h * k), (f // k) % h, f % k
            codes[j, li, hi] = np.eye(k, dtype=np.float32)[ki]
        emb = np.asarray(embed_fn(codes))
    extra = {}
    if cfg.model == "onto":
        # `emb` forces ONE head of l*h and leaves the rest at the origin, so
        # it is overwhelmingly the origin's own decode. The difference
        # against the pure origin is the tag's own contribution ("pdev").
        # That buys legibility, not detection score: a decoded sentence is
        # an instance of what the tag writes, and detection needs a
        # criterion to sort snippets by.
        emb_origin = np.asarray(embed_fn(origin[None]))
        extra = {"emb_dev": emb - emb_origin, "emb_origin": emb_origin}
    if cfg.model != "onto":
        alpha = top_v[sel, :cfg.n_desc].mean(1, keepdims=True)
        emb = (np.asarray(sae_params["b_dec"])
               + alpha * np.asarray(sae_params["W_dec"])[sel])
        if "W_enc2" in sae_params:
            # bilinear encoder: each latent's closed-form top eigenvector
            # is an input-space direction, decodable exactly like a W_dec
            # row -- the "eig" description mode
            import sae as sae_mod
            E = np.asarray(sae_mod.eigenfeatures(sae_params))[sel]
            extra["emb_eig"] = np.asarray(sae_params["b_dec"]) + alpha * E

    np.savez(out / "features.npz", sel=sel, top_i=top_i[sel], top_v=top_v[sel],
             neg_i=neg_i, freq=freq[sel], q50=q50, emb=emb, **extra)
    (out / "meta.json").write_text(json.dumps(
        {**meta, "rows": n, "n_top": cfg.n_top, "n_desc": cfg.n_desc,
         "n_neg": cfg.n_neg, "seed": cfg.seed}, indent=2))
    print(f"harvest: {len(sel)}/{F} features -> {out/'features.npz'}")


# ---------- stage: match ----------

def match(cfg):
    """Language-matched rows for an existing campaign, in lang_rows.npz next
    to its features.npz: per feature, `neg_i`, n_test negatives in the
    languages of its held-out positives, and `bg_i`, n_desc contrastive
    background rows in the languages of its description rows, with the
    matched count of each. Candidates are the harvest subsample's rows below
    the feature's median activation (q50) outside its top rows, harvest's
    own negative pool; the background also excludes both sets of scoring
    negatives, so a description never sees a row it is graded on. The
    checkpoint is --ckpt, else the SAE path harvest recorded (an
    Ontologizer's meta.json records none); its firing rates on the
    subsample must reproduce the harvest's, so a moved or retrained
    checkpoint is refused rather than matched against."""
    import jax.numpy as jnp
    from tqdm import tqdm

    d = Path(cfg.dir)
    dat = np.load(d / "features.npz")
    meta = json.loads((d / "meta.json").read_text())
    if meta["model"] == "onto":
        acts_fn, _, _, _ = onto_acts_fn(cfg.ckpt or ONTO_CKPT, meta["step"],
                                        meta["temperature"])
    else:
        acts_fn, _, _, _ = sae_acts_fn(cfg.ckpt or meta["ckpt"],
                                       meta["topk"])
    sel, top_i, n_desc = dat["sel"], dat["top_i"], meta["n_desc"]

    # the subsample harvest measured q50 on: every stride-th batch
    mm = np.load(cfg.cache, mmap_mode="r")
    n, b = meta["rows"], cfg.b
    stride = max(1, (n // b) // cfg.sub_batches)
    rows, acts = [], []
    for i in tqdm(range(0, n - b + 1, b * stride), desc="match"):
        A = np.asarray(acts_fn(jnp.asarray(
            np.asarray(mm[i:i + b], dtype=np.float32))))
        rows.append(np.arange(i, i + b))
        acts.append(A[:, sel].astype(np.float16))  # as harvest stores them
    rows, acts = np.concatenate(rows), np.concatenate(acts)  # (S,), (S, F)
    drift = np.abs((acts > fire_threshold(meta)).mean(0) - dat["freq"]).max()
    if drift > 0.01:
        raise ValueError(
            f"match: firing rates differ from the harvest's by up to "
            f"{drift:.3f}; this is not the checkpoint {d} was harvested "
            f"from (pass --ckpt)")
    langs = np.load(langs_path(cfg.cache), mmap_mode="r")
    row_langs = np.asarray(langs[rows])

    neg_i = np.zeros((len(sel), cfg.n_test), np.int64)
    bg_i = np.zeros((len(sel), n_desc), np.int64)
    neg_matched = np.zeros(len(sel), np.int64)
    bg_matched = np.zeros(len(sel), np.int64)
    for j, f in enumerate(sel):
        rng = np.random.default_rng([meta["seed"], int(f)])
        ok = (acts[:, j] <= dat["q50"][j]) & ~np.isin(rows, top_i[j])
        cand, cand_langs = rows[ok], row_langs[ok]
        pos = top_i[j, n_desc:n_desc + cfg.n_test]
        neg_i[j], neg_matched[j] = matched_rows(
            np.asarray(langs[pos]), cand, cand_langs, rng)
        keep = ~np.isin(cand, np.concatenate(
            [neg_i[j], dat["neg_i"][j, :cfg.n_test]]))
        bg_i[j], bg_matched[j] = matched_rows(
            np.asarray(langs[top_i[j, :n_desc]]), cand[keep],
            cand_langs[keep], rng)
    np.savez(d / "lang_rows.npz", sel=sel, neg_i=neg_i, bg_i=bg_i,
             neg_matched=neg_matched, bg_matched=bg_matched)
    print(f"match: {len(sel)} features, negatives "
          f"{neg_matched.sum() / neg_i.size:.1%} and background "
          f"{bg_matched.sum() / bg_i.size:.1%} language-matched -> "
          f"{d / 'lang_rows.npz'}")


def load_lang_rows(d, sel, n_test):
    """`match`'s rows for campaign d, checked against its feature sample."""
    path = Path(d) / "lang_rows.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run the match stage first")
    lr = np.load(path)
    if not np.array_equal(lr["sel"], sel):
        raise ValueError(f"{path} was drawn for another feature sample")
    if lr["neg_i"].shape[1] < n_test:
        raise ValueError(f"{path} holds {lr['neg_i'].shape[1]} negatives per "
                         f"feature, fewer than --n-test {n_test}")
    return lr


# ---------- stage: texts ----------

def texts(cfg):
    from tqdm import tqdm
    from ontologize.data.loaders import HFDataSource
    from ontologize.data.multilingual import mc4_data

    need = needed_rows(cfg.features)
    path = Path(cfg.out) / "texts.jsonl"
    have = set()
    if path.exists():
        with open(path) as f:
            have = {json.loads(line)["row"] for line in f}
    need -= have
    if not need:
        print(f"texts: all rows already in {path}")
        return
    stop = max(need)

    # texts are recovered by replaying the (deterministic) mC4 stream to the
    # cache row index; the .langs.npy sidecar written by encode_corpus.py
    # gives a per-row alignment check that catches any stream drift
    sidecar = langs_path(cfg.cache)
    L = np.load(sidecar, mmap_mode="r") if sidecar.exists() else None
    if L is None:
        print(f"WARNING: {sidecar} missing; stream alignment unverified")

    ds = mc4_data("allenai/c4", split="train", streaming=True)
    it = iter(HFDataSource(ds, text_key="text"))
    with open(path, "a") as f:
        for row in tqdm(range(stop + 1), desc="streaming mC4"):
            item = next(it)
            if L is not None and item["lang"] != L[row]:
                raise RuntimeError(
                    f"stream drift at row {row}: stream lang {item['lang']!r}"
                    f" != cached {L[row]!r}; texts cannot be recovered by"
                    " index -- re-encode or match dataset revision")
            if row in need:
                f.write(json.dumps({"row": row, "lang": item["lang"],
                                    "text": item["text"][:2000]},
                                   ensure_ascii=False) + "\n")
    print(f"texts: +{len(need)} rows -> {path}")


def load_texts(out):
    path = Path(out) / "texts.jsonl"
    with open(path) as f:
        return {json.loads(l)["row"]: json.loads(l) for l in f}


# ---------- stage: describe ----------

def describe(cfg):
    d = Path(cfg.dir)
    dat = np.load(d / "features.npz")
    meta = json.loads((d / "meta.json").read_text())
    out = d / "descriptions.jsonl"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(j["feature"], j["mode"]) for j in map(json.loads, f)}

    # write each mode's descriptions as soon as they exist: a crash in a
    # later mode (e.g. the LLM judge) must not discard earlier GPU work
    def flush(recs):
        n = 0
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        return n

    n = 0
    if cfg.mode in ("params", "both", "all"):
        n += flush(describe_params(cfg, dat, done))
    if cfg.mode in ("pdev", "all") and "emb_dev" in dat:
        n += flush(describe_params(cfg, dat, done, key="emb_dev",
                                   mode="pdev"))
    if cfg.mode in ("eig", "all"):
        if "emb_eig" in dat:
            n += flush(describe_params(cfg, dat, done, key="emb_eig",
                                       mode="eig"))
        else:
            print("describe: no emb_eig in features.npz (not a bilinear "
                  "run); skipping eig mode")
    if cfg.mode in ("acts", "both", "all"):
        n += flush(describe_acts(cfg, dat, meta, done))
    if cfg.mode == "cacts":
        n += flush(describe_acts(cfg, dat, meta, done, contrast=True))
    if cfg.mode == "cacts_lang":
        bg = load_lang_rows(d, dat["sel"], cfg.n_test)["bg_i"]
        n += flush(describe_acts(cfg, dat, meta, done, contrast=True,
                                 bg_rows=bg))
    print(f"describe: +{n} -> {out}")


def describe_params(cfg, dat, done, key="emb", mode="params"):
    import torch as t
    from transformers import AutoTokenizer, M2M100ForConditionalGeneration
    from transformers.modeling_outputs import BaseModelOutput
    from textfid import SONAR_NORM

    todo = [j for j, f in enumerate(dat["sel"]) if (int(f), mode) not in done]
    if not todo:
        return []
    dev = t.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
    ref_norm = SONAR_NORM
    dec = M2M100ForConditionalGeneration.from_pretrained(DECODER_ID).to(dev)
    dec.eval()
    ENG = tokenizer.convert_tokens_to_ids("eng_Latn")

    recs = []
    with t.no_grad():
        for i in range(0, len(todo), cfg.b_decode):
            js = todo[i:i + cfg.b_decode]
            Y = t.from_numpy(dat[key][js]).to(dev, t.float32)
            Y = t.nn.functional.normalize(Y, dim=-1) * ref_norm
            gen = dec.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=Y.unsqueeze(1)),
                forced_bos_token_id=ENG, max_length=48, num_beams=1,
                repetition_penalty=1.2)
            for j, g in zip(js, gen):
                text = tokenizer.decode(g, skip_special_tokens=True).strip()
                recs.append({"feature": int(dat["sel"][j]), "mode": mode,
                             "description": text})
    return recs


def describe_acts(cfg, dat, meta, done, contrast=False, bg_rows=None):
    """Judge-written descriptions: "acts" from the top rows alone, "cacts"
    beside a contrastive background of random rows, and "cacts_lang" beside
    `bg_rows` (feature x row), the language-matched background `match`
    draws."""
    txt = load_texts(cfg.out)
    llm_map = make_llm(cfg.judge, cfg.dry_dir(), cfg.jobs, cfg.rate)
    mode = ("cacts_lang" if bg_rows is not None else
            "cacts" if contrast else "acts")
    # contrast background: any feature's negatives are random low-activation
    # corpus draws, and they are already in texts.jsonl
    pool = np.unique(dat["neg_i"]) if contrast else None
    items, feats = [], []
    for j, f in enumerate(dat["sel"]):
        if (int(f), mode) in done:
            continue
        rows = dat["top_i"][j, :meta["n_desc"]]
        pos = "\n".join(f"- {snippet(txt[int(r)]['text'])}" for r in rows)
        if contrast:
            bg = (bg_rows[j] if bg_rows is not None else
                  contrast_rows(int(f), dat["top_i"][j], dat["neg_i"][j],
                                pool, meta["n_desc"], cfg.n_test,
                                meta["seed"]))
            neg = "\n".join(f"- {snippet(txt[int(r)]['text'])}" for r in bg)
            prompt = CONTRAST_PROMPT.format(pos=pos, neg=neg)
        else:
            prompt = SUMMARIZE_PROMPT.format(samples=pos)
        items.append((prompt, f"{mode}_{f}"))
        feats.append(int(f))
    # generator: each summary is flushed by the caller as it arrives, so an
    # aborted judge run keeps (and never re-buys) the completed calls
    for f, r in zip(feats, llm_map(items)):
        if r:
            yield {"feature": f, "mode": mode, "description": r}


# ---------- stage: score ----------

def score(cfg):
    d = Path(cfg.dir)
    dat = np.load(d / "features.npz")
    meta = json.loads((d / "meta.json").read_text())
    txt = load_texts(cfg.out)
    with open(d / "descriptions.jsonl") as f:
        descs = [json.loads(l) for l in f]
    if cfg.only_modes:
        descs = [r for r in descs if r["mode"] in cfg.only_modes]
    llm_map = make_llm(cfg.judge, cfg.dry_dir(), cfg.jobs, cfg.rate)
    sel_pos = {int(f): j for j, f in enumerate(dat["sel"])}
    # language-matched negatives grade the same positives; their rows are
    # recorded under their own kinds so both item sets live in one file
    if cfg.negatives == "lang":
        neg_i = load_lang_rows(d, dat["sel"], cfg.n_test)["neg_i"]
        kinds = {"self": "self_lang", "null": "null_lang"}
    else:
        neg_i = dat["neg_i"]
        kinds = {"self": "self", "null": "null"}

    # resume: a crashed run has already paid for some judgements
    path = d / "scores.csv"
    old, legacy = read_scores(path)
    done = {(int(r["feature"]), r["mode"], r["kind"]) for r in old}
    if legacy:  # rewrite in place so the append below stays rectangular
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(SCORE_COLS)
            w.writerows([[r[c] for c in SCORE_COLS] for r in old])

    # null control: each feature graded against another feature's
    # description of the SAME mode (matched register and verbosity)
    nulls = {}
    if cfg.null:
        by_mode = {}
        for rec in descs:
            by_mode.setdefault(rec["mode"], {})[int(rec["feature"])] = \
                rec["description"]
        for mode, m in by_mode.items():
            pair = null_pairing(sorted(m), meta["seed"], texts=m)
            if len(pair) < len(m):
                print(f"score[{mode}]: {len(m) - len(pair)}/{len(m)} features "
                      "have no distinct-text donor (duplicate descriptions); "
                      "no null for those")
            pair = cap_pairing(pair, cfg.null_n, meta["seed"])
            nulls[mode] = {f: m[donor] for f, donor in pair.items()}

    items, keys, truths = [], [], []
    for rec in descs:
        f = int(rec["feature"])
        j = sel_pos[f]
        rows, truth = detection_items(
            f, dat["top_i"][j], neg_i[j],
            meta["n_desc"], cfg.n_test, meta["seed"])
        samples = "\n".join(f"{i+1}. {snippet(txt[int(r)]['text'])}"
                            for i, r in enumerate(rows))
        for base in ("self", "null"):
            kind = kinds[base]
            desc = (rec["description"] if base == "self"
                    else nulls.get(rec["mode"], {}).get(f))
            if desc is None or (f, rec["mode"], kind) in done:
                continue
            items.append((DETECT_PROMPT.format(desc=desc, samples=samples,
                                               n=len(rows)),
                          f"detect_{rec['mode']}_{kind}_{f}"))
            keys.append((f, rec["mode"], kind))
            truths.append((truth, len(rows)))
    replies = llm_map(items)
    if cfg.dry_dir():
        for _ in replies:  # llm_map is lazy; consuming writes the prompts
            pass
        print(f"score: dry run, {len(items)} prompts written")
        return

    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(SCORE_COLS)
        for (f, mode, kind), (truth, n_items), ans in zip(keys, truths,
                                                          replies):
            if not ans:  # never record a failed judgement as F1=0
                continue
            pred = parse_judge(ans, n_items)
            p, r, f1 = prf(pred, truth)
            w.writerow([f, mode, kind, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])
    summarize_scores(path)


def summarize_scores(path):
    """Per-mode means over the WHOLE file (not just this run's additions,
    which are partial on a resume). Prints the null delta when both kinds
    are present."""
    rows, _ = read_scores(path)
    by = {}
    for r in rows:
        by.setdefault(r["mode"], {}).setdefault(r["kind"], []).append(
            float(r["f1"]))
    for mode, kinds in sorted(by.items()):
        for suffix, tag in (("", mode), ("_lang", f"{mode}, lang negatives")):
            s = kinds.get("self" + suffix, [])
            nul = kinds.get("null" + suffix, [])
            if not (s or nul):
                continue
            line = f"score[{tag}]: mean F1 {np.mean(s):.3f} (n={len(s)})" \
                if s else f"score[{tag}]: no self scores"
            if s and nul:
                line += (f" | null {np.mean(nul):.3f} (n={len(nul)})"
                         f" | delta {np.mean(s) - np.mean(nul):+.3f}")
            print(f"{line} -> {path}")


# ---------- stage: language ----------

def campaign_label(d, out, model):
    """A campaign's name: its directory under --out, less the model
    directory harvest writes into (m5120_k32/sae -> m5120_k32)."""
    try:
        parts = Path(d).resolve().relative_to(Path(out).resolve()).parts
    except ValueError:
        parts = Path(d).parts[-2:]
    if len(parts) > 1 and parts[-1] == model:
        parts = parts[:-1]
    return "/".join(parts)


def campaign_languages(d, langs, n_test, modes):
    """One campaign's per-feature language statistics, joined with its
    scores. A feature's judged rows are its n_desc description rows and
    n_test held-out positives; `lang` is the description rows' modal
    language, `purity` its share of them, and the *_share columns its share
    of what the judge sees beside them (the cacts background, the held-out
    positives and the negatives, and with lang_rows.npz the matched
    background and negatives). Returns (meta, {feature: stats}, one record
    per score row in `modes`)."""
    d = Path(d)
    dat = np.load(d / "features.npz")
    meta = json.loads((d / "meta.json").read_text())
    n_desc, seed = meta["n_desc"], meta["seed"]
    top_i, neg_i = dat["top_i"], dat["neg_i"]
    matched = (np.load(d / "lang_rows.npz")
               if (d / "lang_rows.npz").exists() else None)
    pool = np.unique(neg_i)

    def share(rows, lang):
        return float(np.mean(np.asarray(langs[np.asarray(rows)]) == lang))

    feats = {}
    for j, f in enumerate(dat["sel"]):
        f = int(f)
        seen = np.asarray(langs[top_i[j, :n_desc + n_test]])
        lang, purity = modal_language(seen[:n_desc])
        bg = contrast_rows(f, top_i[j], neg_i[j], pool, n_desc, n_test, seed)
        feats[f] = {"feature": f, "freq": float(dat["freq"][j]),
                    "n_lang": len(np.unique(seen)), "lang": lang,
                    "purity": purity, "bg_share": share(bg, lang),
                    "pos_share": float(np.mean(seen[n_desc:] == lang)),
                    "neg_share": share(neg_i[j, :n_test], lang)}
        if matched is not None:
            feats[f]["bg_share_lang"] = share(matched["bg_i"][j], lang)
            feats[f]["neg_share_lang"] = share(
                matched["neg_i"][j, :n_test], lang)
    scored = []
    for r in read_scores(d / "scores.csv")[0]:
        f = int(r["feature"])
        if r["mode"] not in modes or f not in feats:
            continue
        p, rc = float(r["precision"]), float(r["recall"])
        lo, hi = detection_accuracy(p, rc, n_test)
        scored.append({**feats[f], "mode": r["mode"], "kind": r["kind"],
                       "precision": p, "recall": rc, "f1": float(r["f1"]),
                       "acc_lo": lo, "acc_hi": hi})
    return meta, feats, scored


def language(cfg):
    """Detection against the language spread of each feature's top rows,
    per campaign, and purity against F1 across campaigns and features. A
    feature is multilingual when its judged rows span the last --lang-bins
    bin; `random_span` is what that many random rows span. Writes
    features.csv (one row per score), bins.csv and summary.json to
    <out>/language/."""
    out = Path(cfg.out)
    dirs = ([Path(p) for p in cfg.dirs] if cfg.dirs else
            sorted(p.parent for p in out.glob("**/features.npz")
                   if (p.parent / "scores.csv").exists()))
    if not dirs:
        raise SystemExit(f"language: no scored campaign under {out}")
    langs = np.load(langs_path(cfg.cache), mmap_mode="r")
    modes = cfg.only_modes or JUDGE_MODES
    edges = sorted(cfg.lang_bins)
    camps = []
    for d in dirs:
        meta, feats, scored = campaign_languages(d, langs, cfg.n_test, modes)
        camps.append((campaign_label(d, out, meta["model"]), meta, feats,
                      scored))
    n_rows = camps[0][1]["n_desc"] + cfg.n_test
    labels = span_bins(edges, n_rows)
    # what random rows span, for reading the multilingual bin
    _, counts = np.unique(np.asarray(langs[:camps[0][1]["rows"]]),
                          return_counts=True)
    span = random_span(counts, n_rows)

    def binned(recs):
        return np.searchsorted(edges, [r["n_lang"] for r in recs],
                               side="right") - 1

    def stats(recs):
        if not recs:
            return {"n": 0}
        f1 = np.array([r["f1"] for r in recs])
        return {"n": len(recs), "f1": float(f1.mean()),
                "acc": [float(np.mean([r["acc_lo"] for r in recs])),
                        float(np.mean([r["acc_hi"] for r in recs]))]}

    runs, bins_rows = {}, []
    for name, meta, feats, scored in camps:
        F = list(feats.values())
        fb = binned(F)
        run = {"model": meta["model"], "features": len(F),
               "freq": float(np.mean([r["freq"] for r in F])),
               "purity": float(np.mean([r["purity"] for r in F])),
               "pure_share": float(np.mean([r["purity"] >= cfg.pure
                                            for r in F])),
               "span": {lab: int((fb == b).sum())
                        for b, lab in enumerate(labels)},
               **{k: float(np.mean([r[k] for r in F]))
                  for k in ("bg_share", "pos_share", "neg_share",
                            "bg_share_lang", "neg_share_lang") if k in F[0]},
               "modes": {}}
        for mode, kind in sorted({(s["mode"], s["kind"]) for s in scored}):
            S = [s for s in scored if s["mode"] == mode and s["kind"] == kind]
            sb = binned(S)
            for b, lab in [(b, lab) for b, lab in enumerate(labels)] + [
                    (None, "all")]:
                part = [s for s, x in zip(S, sb) if b is None or x == b]
                st = stats(part)
                bins_rows.append([name, mode, kind, lab, st["n"],
                                  st.get("f1", ""), *st.get("acc", ["", ""])])
            multi = [s for s, x in zip(S, sb) if x == len(labels) - 1]
            pure = [s for s in S if s["purity"] >= cfg.pure]
            mixed = [s for s in S if s["purity"] < cfg.pure]
            run["modes"][mode if kind == "self" else f"{mode}/{kind}"] = {
                **stats(S), "multilingual": stats(multi),
                "pure": stats(pure), "mixed": stats(mixed),
                "rho_span": spearman([s["f1"] for s in S],
                                     [s["n_lang"] for s in S])}
        runs[name] = run

    # across campaigns and features, self scores only
    names = list(runs)
    model_level, feature_level, trend, gap = {}, {}, {}, {}
    onto = [n for n in names if runs[n]["model"] == "onto"]
    for mode in modes:
        have = [n for n in names if mode in runs[n]["modes"]]
        if len(have) < 3:
            continue
        mean_f1 = np.array([runs[n]["modes"][mode]["f1"] for n in have])
        multi_f1 = np.array([runs[n]["modes"][mode]["multilingual"]
                             .get("f1", np.nan) for n in have])
        model_level[mode] = {
            "campaigns": len(have),
            "r2_purity": ols_r2(mean_f1,
                                np.array([runs[n]["purity"] for n in have]))}
        S = [s for _, _, _, sc in camps for s in sc
             if s["mode"] == mode and s["kind"] == "self"]
        y = np.array([s["f1"] for s in S])
        pur = np.array([s["purity"] for s in S])
        # a feature that never fires is floored at a rate of 1e-6
        lf = np.log10(np.maximum([s["freq"] for s in S], 1e-6))
        feature_level[mode] = {"features": len(S),
                               "r2_purity": ols_r2(y, pur),
                               "r2_log_freq": ols_r2(y, lf),
                               "r2_both": ols_r2(y, pur, lf)}
        # the trend is fitted to the SAE campaigns and read off for all
        freq = np.array([runs[n]["freq"] for n in have])
        fit = np.array([runs[n]["model"] == "sae" for n in have])
        if (fit & (freq > 0)).sum() >= 2:
            res_all = trend_residuals(freq, mean_f1, fit)
            res_multi = trend_residuals(freq, multi_f1, fit)
            trend[mode] = {
                "fit": [n for n, x, q in zip(have, fit, freq) if x and q > 0],
                "residual": {n: [float(a), float(m)] for n, a, m
                             in zip(have, res_all, res_multi)
                             if not np.isnan(a)}}
        if len(onto) == 1 and onto[0] in have:
            ref = runs[onto[0]]["modes"][mode]

            def gaps(n):
                m = runs[n]["modes"][mode]
                return [m["f1"] - ref["f1"],
                        m["multilingual"].get("f1", np.nan)
                        - ref["multilingual"].get("f1", np.nan)]
            gap[mode] = {n: gaps(n) for n in have if n != onto[0]}

    dest = out / "language"
    dest.mkdir(parents=True, exist_ok=True)
    cols = ["campaign", "model", "feature", "mode", "kind", "f1",
            "precision", "recall", "acc_lo", "acc_hi", "n_lang", "lang",
            "purity", "freq", "bg_share", "pos_share", "neg_share",
            "bg_share_lang", "neg_share_lang"]
    with open(dest / "features.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, meta, _, scored in camps:
            for s in scored:
                w.writerow([name, meta["model"]] + [s.get(c, "")
                                                    for c in cols[2:]])
    with open(dest / "bins.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["campaign", "mode", "kind", "span", "n", "f1", "acc_lo",
                    "acc_hi"])
        w.writerows(bins_rows)
    summary = finite({
        "rows_judged": n_rows, "span_bins": labels,
        "multilingual": labels[-1], "random_span": span, "pure": cfg.pure,
        "campaigns": runs, "model_level": model_level,
        "feature_level": feature_level, "trend": trend, "gap_to_onto": gap})
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    print_language(summary)
    print(f"language: {len(camps)} campaigns -> {dest}")


def finite(obj):
    """obj with every NaN float replaced by None, which JSON can hold."""
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    return obj


def print_language(s):
    """The language stage's summary, as text."""
    fmt = lambda x, p=3: "  -  " if x is None else f"{x:.{p}f}"
    sgn = lambda x: "   -  " if x is None else f"{x:+.3f}"
    labels, runs = s["span_bins"], s["campaigns"]
    print(f"{s['rows_judged']} judged rows per feature; random rows span "
          f"{s['random_span']:.1f} languages; multilingual = "
          f"{s['multilingual']} languages; pure = purity >= {s['pure']}")
    print(f"{'campaign':18s} {'feats':>5s} {'purity':>6s} {'pure':>5s} "
          + " ".join(f"{lab:>5s}" for lab in labels)
          + "  bg/pos/neg modal-language share")
    for name, r in runs.items():
        print(f"{name:18s} {r['features']:5d} {r['purity']:6.3f} "
              f"{r['pure_share']:5.2f} "
              + " ".join(f"{r['span'][lab]:5d}" for lab in labels)
              + f"  {r['bg_share']:.3f}/{r['pos_share']:.2f}/"
              f"{r['neg_share']:.2f}")
    keys = sorted({m for r in runs.values() for m in r["modes"]})
    for mode in keys:
        print(f"\n{mode}: mean F1 (all, {labels[-1]}, pure, mixed), "
              "accuracy (all, multilingual), Spearman(F1, languages)")
        for name, r in runs.items():
            m = r["modes"].get(mode)
            if m is None:
                continue
            ml = m["multilingual"]
            print(f"  {name:18s} {fmt(m['f1'])} (n={m['n']:3d})  "
                  f"{fmt(ml.get('f1'))} (n={ml['n']:3d})  "
                  f"{fmt(m['pure'].get('f1'))}  {fmt(m['mixed'].get('f1'))}"
                  f"  [{fmt(m['acc'][0])}, {fmt(m['acc'][1])}]  "
                  + (f"[{fmt(ml['acc'][0])}, {fmt(ml['acc'][1])}]"
                     if ml["n"] else "      -       ")
                  + f"  {sgn(m['rho_span'])}")
    for mode, g in s["gap_to_onto"].items():
        print(f"\n{mode}: gap to the Ontologizer, all -> multilingual: "
              + ", ".join(f"{n} {sgn(a)} -> {sgn(b)}"
                          for n, (a, b) in g.items()))
    for mode, m in s["model_level"].items():
        f = s["feature_level"][mode]
        print(f"\n{mode}: R^2 of mean F1 on mean purity over "
              f"{m['campaigns']} campaigns {fmt(m['r2_purity'])}; per feature "
              f"(n={f['features']}) purity {fmt(f['r2_purity'])}, log firing "
              f"rate {fmt(f['r2_log_freq'])}, both {fmt(f['r2_both'])}")
    for mode, t in s["trend"].items():
        print(f"\n{mode}: distance above the firing-rate trend fitted to "
              f"{', '.join(t['fit'])}, all -> multilingual features:")
        for n, (a, b) in t["residual"].items():
            print(f"  {n:18s} {sgn(a)} -> {sgn(b)}")

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["harvest", "texts", "describe", "score",
                                     "language", "match"])
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--model", choices=["onto", "sae"], default="onto")
    p.add_argument("--ckpt", default=None,
                   help="an Ontologizer run dir (default resid_nc) or an SAE "
                        "params.npz; match defaults to the SAE path the "
                        "harvest recorded")
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--topk", type=int, default=32, help="sae harvest only")
    p.add_argument("--temperature", type=float, default=0.03)
    p.add_argument("--cache", default=str(CACHE))
    p.add_argument("--rows", type=int, default=524288,
                   help="cache rows to harvest activations over")
    p.add_argument("--b", type=int, default=4096)
    p.add_argument("--sub-batches", type=int, default=16,
                   help="batches kept for the quantile/negative subsample")
    p.add_argument("--n-features", type=int, default=256,
                   help="stratified feature sample (0 = all)")
    p.add_argument("--n-top", type=int, default=20)
    p.add_argument("--n-desc", type=int, default=10,
                   help="top texts used to WRITE the description")
    p.add_argument("--n-test", type=int, default=5,
                   help="held-out positives (and negatives) used to SCORE it")
    p.add_argument("--n-neg", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", nargs="+", help="features.npz paths (texts)")
    p.add_argument("--feature-ids", type=int, nargs="+", default=None,
                   help="harvest exactly these feature indices instead of a "
                        "stratified sample, for interrogating one head. An "
                        "Ontologizer tag is layer*h*k + head*k + entry")
    p.add_argument("--dir", help="harvest output dir (describe/score/match)")
    p.add_argument("--dirs", nargs="+", default=None,
                   help="campaign dirs for the language stage (default: "
                        "every scored campaign under --out)")
    p.add_argument("--mode",
                   choices=["params", "pdev", "acts", "cacts", "cacts_lang",
                            "eig", "both", "all"],
                   default="both",
                   help="description modes: both = params+acts; pdev = the "
                        "params decode less the pure-origin decode (onto); "
                        "eig = bilinear-SAE eigenfeature decodes; all = "
                        "params+pdev+eig+acts; cacts = contrastive acts (top "
                        "texts vs random corpus draws), run explicitly, never "
                        "part of both/all; cacts_lang = cacts against the "
                        "language-matched background of the match stage")
    p.add_argument("--only-modes", nargs="+", default=None,
                   help="score only descriptions of these modes; the language "
                        "stage reports these (default: the judge-written "
                        f"modes, {', '.join(JUDGE_MODES)})")
    p.add_argument("--negatives", choices=["random", "lang"],
                   default="random",
                   help="score stage: random = the harvest's low-activation "
                        "negatives; lang = the match stage's negatives in the "
                        "held-out positives' languages, recorded as kinds "
                        "self_lang/null_lang")
    p.add_argument("--lang-bins", type=int, nargs="+", default=[1, 4, 10, 13],
                   help="language stage: lower edges of the bins of distinct "
                        "languages among a feature's judged rows; the last "
                        "bin is the multilingual one")
    p.add_argument("--pure", type=float, default=0.8,
                   help="language stage: purity at or above which a feature "
                        "counts as language-pure")
    p.add_argument("--device", default="cpu",
                   help="torch device for the M2M100 decode")
    p.add_argument("--b-decode", type=int, default=16)
    p.add_argument("--judge", default="haiku",
                   help="judge model: short name (haiku, sonnet) or a full "
                        "Anthropic API model id; calls the Messages API "
                        "directly, billing ANTHROPIC_API_KEY")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rate", type=int, default=0,
                   help="max judge calls per hour, evenly spaced across "
                        "workers (0 = unmetered); smooths request rate "
                        "under API rate limits")
    p.add_argument("--null", action="store_true",
                   help="also grade every feature against ANOTHER feature's "
                        "same-mode description (score stage). Doubles judge "
                        "calls and yields the chance floor the real F1 has "
                        "to clear; resumable and additive to existing scores")
    p.add_argument("--null-n", type=int, default=0,
                   help="cap null controls at N features per mode (0 = every "
                        "feature). The floor only needs estimating, not "
                        "matching per-feature; the subset is seed-determined "
                        "so a resume finishes it rather than sampling anew")
    p.add_argument("--dry-run", action="store_true",
                   help="write judge prompts to <dir>/prompts/ instead of calling")
    cfg = p.parse_args()

    def dry_dir():
        if not cfg.dry_run:
            return None
        d = Path(cfg.dir or cfg.out) / "prompts"
        d.mkdir(parents=True, exist_ok=True)
        return d
    cfg.dry_dir = dry_dir

    {"harvest": harvest, "texts": texts, "describe": describe,
     "score": score, "language": language, "match": match}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
