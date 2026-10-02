"""Multilingual dataset loading and interleaving utilities for HuggingFace datasets.

This module provides helper functions to load individual language splits from
HuggingFace (e.g., mC4 or C4), sanitize schema variations across languages, and
interleave multiple language streams into a unified training dataset.
"""

from datasets import load_dataset, interleave_datasets

from ontologize.data.langs import MC4_TO_SONAR

def load_lang(src, lang, *args, **kwargs):
    """Loads a single language split from a HuggingFace dataset and standardizes fields.

    Selects only the `"text"` column (stripping conflicting metadata fields such as
    `timestamp` and `url`) and attaches a `"lang"` tag to every example.

    Args:
        src: Dataset repository path or identifier (e.g. `"allenai/c4"`).
        lang: Language split identifier (e.g. `"en"`).
        *args: Additional positional arguments forwarded to `datasets.load_dataset`.
        **kwargs: Additional keyword arguments forwarded to `datasets.load_dataset`.

    Returns:
        Dataset: HuggingFace `Dataset` or `IterableDataset` with standardized `"text"` and `"lang"` fields.
    """
    ds = load_dataset(src, lang, *args, **kwargs)
    # Strip metadata like 'timestamp' and 'url' which have conflicting 
    # types across different languages in C4 (e.g. string vs timestamp[us])
    ds = ds.select_columns(["text"])
    ds = ds.map(lambda x: {"text": x["text"], "lang": lang})
    return ds

def load_langs(src, langs, *args, **kwargs):
    """Loads and interleaves multiple language splits from a HuggingFace dataset.

    Applies `load_lang` to each key in `langs` and combines them using
    `datasets.interleave_datasets`.

    Args:
        src: Dataset repository path or identifier.
        langs: Dictionary whose keys are language identifiers (e.g. `MC4_TO_SONAR`).
        *args: Additional positional arguments forwarded to `load_lang`.
        **kwargs: Additional keyword arguments forwarded to `load_lang`.

    Returns:
        Dataset: Interleaved HuggingFace dataset combining all specified language splits.
    """
    dss = [load_lang(src, x, *args, **kwargs) for x in langs.keys()]
    return interleave_datasets(dss)

def mc4_data(src, *args, **kwargs):
    """Loads and interleaves all multilingual splits defined in `MC4_TO_SONAR`.

    Convenience wrapper invoking `load_langs` with the standard `MC4_TO_SONAR`
    mapping table.

    Args:
        src: Dataset repository path or identifier (e.g. `"allenai/c4"`).
        *args: Additional positional arguments forwarded to `load_langs`.
        **kwargs: Additional keyword arguments forwarded to `load_langs`.

    Returns:
        Dataset: Interleaved HuggingFace dataset spanning all supported mC4 languages.
    """
    return load_langs(src, MC4_TO_SONAR, *args, **kwargs)



