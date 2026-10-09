from datasets import load_dataset, interleave_datasets

from ontologize.data.langs import MC4_TO_SONAR

def load_lang(src, lang, *args, **kwargs):
    ds = load_dataset(src, lang, *args, **kwargs)
    # Strip metadata like 'timestamp' and 'url' which have conflicting 
    # types across different languages in C4 (e.g. string vs timestamp[us])
    ds = ds.select_columns(["text"])
    ds = ds.map(lambda x: {"text": x["text"], "lang": lang})
    return ds

def load_langs(src, langs, *args, stopping_strategy="first_exhausted",
               **kwargs):
    """Round-robin interleave of one dataset per language.

    No `probabilities` are passed, so the languages are drawn in strict
    rotation and the result is a uniform mixture rather than a
    natural-frequency sample: every language contributes equally for as
    long as the stream runs.

    `stopping_strategy` is `interleave_datasets`'s, and decides what
    happens when the smallest language runs out:

      first_exhausted   (the default, and `interleave_datasets`' own)
                        undersamples: the stream ends there, so every
                        language is capped at the size of the smallest
                        and the rest of the corpus is discarded.
      all_exhausted     oversamples: exhausted languages restart and
                        repeat until every language has been seen
                        through once, so nothing is discarded but the
                        small languages appear many times over.
      all_exhausted_without_replacement
                        keeps rotating over the languages that have
                        documents left, skipping each one that runs out:
                        nothing is discarded or repeated, and the mixture
                        stops being uniform once the smallest language
                        is spent. Up to that point every strategy yields
                        the same stream, at least when streaming, so a
                        cache built under the default is an exact prefix
                        of this one (experiments/fresh-eval relies on it).

    None is free. Under the default a request larger than n_langs times
    the smallest split silently yields a shorter stream than asked for;
    under `all_exhausted` it yields duplicates instead, and without
    replacement the late rows lose the small languages. Which one is
    wanted depends on whether repeated text, a truncated corpus or a
    skewed mixture is the worse failure for the run at hand."""
    dss = [load_lang(src, x, *args, **kwargs) for x in langs.keys()]
    return interleave_datasets(dss, stopping_strategy=stopping_strategy)

def mc4_data(src, *args, **kwargs):
    """`load_langs` over the mC4 language set; `stopping_strategy`
    passes through."""
    return load_langs(src, MC4_TO_SONAR, *args, **kwargs)


