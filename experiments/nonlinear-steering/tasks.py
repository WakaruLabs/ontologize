"""Text for the behaviours the steering vectors are trained toward, as
GPT-2 token windows: BOS, then `T` tokens of a task's text.

Each behaviour is a text source a 124M base model can model:
- `french`: French mC4 documents, from `mc4_sent16.sentences.jsonl`;
- `code`: Python source from the installed packages;
- `recipes` and `sports`: C4 documents from the `gpt2_l8` harvest, chosen by
  keyword;
- `neutral`: harvest documents matching neither keyword set, used only to
  measure crosstalk.

Train and test windows come from disjoint documents (or files).
"""
import json
import sysconfig
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from jaxtyping import Int

from common import DOCS, ROOT

SENTENCES = ROOT / "data" / "sonar_embeddings" / "mc4_sent16.sentences.jsonl"
BOS = 50256
TASKS = ("french", "code", "recipes", "sports")

RECIPES = frozenset({"recipe", "tablespoon", "teaspoon", "ingredients", "preheat",
                     "oven", "flour", "simmer", "bake", "baking", "whisk",
                     "saucepan", "cup of"})
SPORTS = frozenset({"season", "coach", "league", "tournament", "championship",
                    "playoff", "quarterback", "touchdown", "innings", "halftime",
                    "referee", "goalkeeper", "striker", "pitcher", "nba", "nfl",
                    "premier league", "scored"})
PACKAGES = ("numpy", "scipy", "jax", "flax", "transformers", "torch")


def hits(text: str, words: Set[str]) -> int:
    """How many of `words` occur in `text`, case-folded."""
    t = text.lower()
    return sum(w in t for w in words)


def topic(text: str, want: Set[str], avoid: Set[str], at_least: int = 3) -> bool:
    """`text` names at least `at_least` of `want` and at most one of `avoid`."""
    return hits(text, want) >= at_least and hits(text, avoid) <= 1


def french_texts(n: int) -> List[str]:
    """The first `n` French mC4 documents of the sentence file, each its
    usable sentences joined by spaces. The file interleaves languages, so a
    prefix holds them in proportion."""
    out: List[str] = []
    with open(SENTENCES) as f:
        for line in f:
            if '"lang": "fr"' not in line[:80]:
                continue
            d = json.loads(line)
            out.append(" ".join(s[3] for s in d["sents"]))
            if len(out) >= n:
                break
    return out


def c4_texts() -> Dict[str, List[str]]:
    """Harvest documents for `recipes`, `sports` and `neutral`."""
    out: Dict[str, List[str]] = {"recipes": [], "sports": [], "neutral": []}
    with open(DOCS) as f:
        for line in f:
            text = json.loads(line)["text"]
            if topic(text, RECIPES, SPORTS):
                out["recipes"].append(text)
            elif topic(text, SPORTS, RECIPES):
                out["sports"].append(text)
            elif hits(text, RECIPES) == 0 and hits(text, SPORTS) == 0:
                out["neutral"].append(text)
    return out


def code_texts(n: int, seed: int, min_chars: int = 2000) -> List[str]:
    """Up to `n` Python files of at least `min_chars` characters from the
    installed packages, in a seeded order."""
    site = Path(sysconfig.get_paths()["purelib"])
    files = sorted(p for pkg in PACKAGES for p in (site / pkg).rglob("*.py")
                   if p.stat().st_size >= min_chars)
    order = np.random.default_rng(seed).permutation(len(files))
    out = []
    for i in order:
        try:
            out.append(files[i].read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
        if len(out) >= n:
            break
    return out


def windows(ids: Sequence[Sequence[int]], T: int, per_doc: int,
            rng: np.random.Generator) -> Int[np.ndarray, "n t1"]:
    """Up to `per_doc` windows of `T` tokens from each token list long
    enough to hold one, at random offsets, each with BOS prepended."""
    out = []
    for seq in ids:
        if len(seq) < T:
            continue
        for start in rng.integers(0, len(seq) - T + 1, size=per_doc):
            out.append([BOS] + list(seq[start:start + T]))
    return np.asarray(out, np.int64).reshape(-1, T + 1)


def split_windows(texts: List[str], tok, T: int, rng: np.random.Generator,
                  test_frac: float = 0.2, train_per_doc: int = 4,
                  test_per_doc: int = 2, max_test: int = 128,
                  max_tokens: int = 4096
                  ) -> Tuple[Int[np.ndarray, "a t1"], Int[np.ndarray, "b t1"]]:
    """Train and test windows from disjoint documents: every
    `round(1 / test_frac)`-th document is a test document."""
    every = max(2, int(round(1 / test_frac)))
    ids = [tok(t[:20 * max_tokens])["input_ids"][:max_tokens] for t in texts]
    test = [s for i, s in enumerate(ids) if i % every == 0]
    train = [s for i, s in enumerate(ids) if i % every != 0]
    return (windows(train, T, train_per_doc, rng),
            windows(test, T, test_per_doc, rng)[:max_test])


def build(T: int, seed: int, n_french: int = 1500, n_code: int = 1500
          ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """`{task: (train windows, test windows)}` for the four tasks and
    `neutral`."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10 ** 9
    rng = np.random.default_rng(seed)
    texts = {"french": french_texts(n_french), "code": code_texts(n_code, seed),
             **c4_texts()}
    texts["neutral"] = texts["neutral"][:2000]
    return {k: split_windows(v, tok, T, rng) for k, v in texts.items()}
