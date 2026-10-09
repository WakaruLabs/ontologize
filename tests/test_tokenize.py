# TokenizeTransform's source-language tags. SONAR's tokenizer prefixes each
# sequence with the tag in `src_lang`, and the tag moves the embedding, so
# every mC4 config must get its own (langs.MC4_TO_SONAR) and an unknown one
# must fail rather than fall back to a default. langs.MC4_4M_TAGS keeps the
# tags the mc4_4M cache was built with.
import numpy as np
import pytest

from ontologize.data.langs import MC4_4M_TAGS, MC4_TO_SONAR
from ontologize.data.loaders import TokenizeTransform


class StubTokenizer:
    """Records the `src_lang` in force at each call, which is the tag
    SONAR's tokenizer would prefix."""
    def __init__(self):
        self.src_lang = None
        self.tags = []

    def __call__(self, text, max_length, padding, truncation):
        self.tags.append(self.src_lang)
        return {"input_ids": [7] * max_length,
                "attention_mask": [1] * max_length}


def test_every_config_gets_its_own_tag():
    tok = StubTokenizer()
    tf = TokenizeTransform(tok, maxlen=4)
    for lang in MC4_TO_SONAR:
        out = tf.map({"text": "x", "lang": lang})
        assert out["input_ids"].shape == (4,)
    assert tok.tags == list(MC4_TO_SONAR.values())


def test_unknown_language_raises():
    tf = TokenizeTransform(StubTokenizer(), maxlen=4)
    with pytest.raises(KeyError, match="'xx'"):
        tf.map({"text": "x", "lang": "xx"})


def test_mc4_4m_tags_are_the_cache_s():
    # en, fr, es, de and zh kept their tags; en-multi's fallback happened to
    # be its own; the other 80 configs were encoded under eng_Latn
    assert set(MC4_4M_TAGS) == set(MC4_TO_SONAR)
    own = {lang for lang in MC4_TO_SONAR
           if MC4_4M_TAGS[lang] == MC4_TO_SONAR[lang]}
    assert own == {"en", "en-multi", "fr", "es", "de", "zh"}
    assert all(MC4_4M_TAGS[lang] == "eng_Latn"
               for lang in MC4_TO_SONAR if lang not in own)
    tok = StubTokenizer()
    TokenizeTransform(tok, maxlen=4, tags=MC4_4M_TAGS).map(
        {"text": "x", "lang": "fi"})
    assert tok.tags == ["eng_Latn"]
