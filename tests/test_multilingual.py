# load_langs interleaves one dataset per language in strict rotation
# (no `probabilities`), so the corpus is a uniform mixture. What happens
# when the smallest language runs out is `stopping_strategy`'s: the
# inherited default truncates, "all_exhausted" repeats instead.
import datasets
import pytest

from ontologize.data import multilingual

SIZES = {"aa": 6, "bb": 3, "cc": 4}      # bb is the shortest


@pytest.fixture
def fake_hub(monkeypatch):
    """Stand in for `load_dataset` so the test needs no network."""
    def fake(src, lang, *args, **kwargs):
        return datasets.Dataset.from_dict(
            {"text": [f"{lang}{i}" for i in range(SIZES[lang])],
             "url": [""] * SIZES[lang]})       # a column load_lang drops
    monkeypatch.setattr(multilingual, "load_dataset", fake)
    return fake


def texts(ds):
    return [r["text"] for r in ds]


def test_load_lang_keeps_text_and_tags_language(fake_hub):
    ds = multilingual.load_lang("src", "bb")
    assert sorted(ds.column_names) == ["lang", "text"]   # url dropped
    assert {r["lang"] for r in ds} == {"bb"}


def test_default_truncates_to_the_shortest(fake_hub):
    ds = multilingual.load_langs("src", SIZES)
    t = texts(ds)
    # rotation stops when the shortest is spent: every language appears
    # exactly as often as the shortest has rows, and no row repeats
    assert len(t) == len(SIZES) * SIZES["bb"]
    assert len(set(t)) == len(t)
    for lang in SIZES:
        assert sum(x.startswith(lang) for x in t) == SIZES["bb"]


def test_all_exhausted_keeps_every_row(fake_hub):
    ds = multilingual.load_langs("src", SIZES,
                                 stopping_strategy="all_exhausted")
    t = texts(ds)
    # nothing is discarded: every row of the LONGEST language is present
    assert len(t) >= len(SIZES) * max(SIZES.values())
    for lang, n in SIZES.items():
        assert {f"{lang}{i}" for i in range(n)} <= set(t)
    # the price is repetition of the short ones
    assert sum(x.startswith("bb") for x in t) > SIZES["bb"]


@pytest.fixture
def fake_stream(monkeypatch):
    """Streaming stand-in: what `load_dataset(..., streaming=True)` gives,
    which interleaves through a different code path than in-memory sets."""
    def fake(src, lang, *args, **kwargs):
        return datasets.Dataset.from_dict(
            {"text": [f"{lang}{i}" for i in range(SIZES[lang])],
             "url": [""] * SIZES[lang]}).to_iterable_dataset()
    monkeypatch.setattr(multilingual, "load_dataset", fake)
    return fake


def test_without_replacement_extends_the_stream(fake_stream):
    """experiments/fresh-eval relies on this: past the point where the
    default stream stops, rotation continues over the languages with rows
    left, never restarting a spent one, and everything before that point
    is the default stream unchanged."""
    default = texts(multilingual.load_langs("src", SIZES))
    full = texts(multilingual.load_langs(
        "src", SIZES, stopping_strategy="all_exhausted_without_replacement"))
    assert full[:len(default)] == default
    assert len(full) == len(set(full)) == sum(SIZES.values())
    # plain all_exhausted restarts the spent language: rows already in
    # the default stream come round again
    again = texts(multilingual.load_langs(
        "src", SIZES, stopping_strategy="all_exhausted"))
    assert set(again[len(default):]) & set(default)


def test_metadata_applies_its_strategy(monkeypatch):
    """Metadata.mc4 is the field's consumer; it must reach the
    interleaver over the whole mC4 language set."""
    from ontologize.data.langs import MC4_TO_SONAR
    from ontologize.training.config import Metadata

    short = next(iter(MC4_TO_SONAR))          # make exactly one tiny

    def fake(src, lang, *args, **kwargs):
        n = 2 if lang == short else 5
        return datasets.Dataset.from_dict(
            {"text": [f"{lang}{i}" for i in range(n)]})

    monkeypatch.setattr(multilingual, "load_dataset", fake)
    n_lang = len(MC4_TO_SONAR)
    first = Metadata(stopping_strategy="first_exhausted").mc4()
    every = Metadata(stopping_strategy="all_exhausted").mc4()
    assert len(first) == n_lang * 2            # capped by the tiny one
    assert len(every) >= n_lang * 5            # nothing discarded
    assert Metadata().stopping_strategy == "first_exhausted"


def test_strategy_is_not_forwarded_to_load_dataset(monkeypatch):
    """It is `interleave_datasets`' argument, not the hub's."""
    seen = []

    def fake(src, lang, *args, **kwargs):
        seen.append(kwargs)
        return datasets.Dataset.from_dict({"text": ["x"]})

    monkeypatch.setattr(multilingual, "load_dataset", fake)
    multilingual.load_langs("src", {"aa": 1}, split="train",
                            stopping_strategy="all_exhausted")
    assert seen and all("stopping_strategy" not in k for k in seen)
    assert all(k.get("split") == "train" for k in seen)
