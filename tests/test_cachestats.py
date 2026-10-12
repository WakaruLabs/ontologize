# experiments/sentence-split/cachestats.py's helpers: byte ranges start at
# lines and cover the file, the scan of a sentences file and of a .npy
# match direct computation (non-finite rows flagged), repeats count copies
# per group, the geometry statistics hit their known values on constructed
# data, the self-rank finds planted matches, and the pairs respect
# documents and groups.
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def load_cachestats():
    path = Path(__file__).parents[1] / "experiments/sentence-split/cachestats.py"
    spec = importlib.util.spec_from_file_location("cachestats", path)
    mod = importlib.util.module_from_spec(spec)
    # registered so the scan's worker pool can pickle its functions by name
    sys.modules["cachestats"] = mod
    spec.loader.exec_module(mod)
    return mod


cs = load_cachestats()


def test_chunks_start_at_lines_and_cover_the_file(tmp_path):
    p = tmp_path / "x.jsonl"
    lines = [("a" * (i % 7 + 1)) + "\n" for i in range(50)]
    p.write_text("".join(lines))
    for parts in [1, 3, 8, 200]:
        r = cs.chunks(p, parts)
        assert r[0][0] == 0 and r[-1][1] == p.stat().st_size
        assert all(b == c for (_, b), (c, _) in zip(r[:-1], r[1:]))
        data = p.read_bytes()
        assert all(a == 0 or data[a - 1:a] == b"\n" for a, _ in r)


def test_scan_part_reads_rows_in_order(tmp_path):
    p = tmp_path / "s.jsonl"
    docs = [{"doc": 0, "lang": "en", "sents": [[1, 0, 3, "abc"], [2, 4, 6, "de"]]},
            {"doc": 1, "lang": "fr", "sents": []},
            {"doc": 2, "lang": "fr", "sents": [[0, 0, 3, "abc"]]}]
    p.write_text("".join(json.dumps(r) + "\n" for r in docs))
    want = frozenset([cs.text_hash("fr", "abc")])
    D, chars, h, found = cs.scan_texts(p, 2, want)
    assert D.tolist() == [[0, 2, cs.CONFIG["en"]], [1, 0, cs.CONFIG["fr"]], [2, 1, cs.CONFIG["fr"]]]
    assert chars.tolist() == [3, 2, 3]
    assert h[0] != h[2]  # the config is part of the hash
    assert found == {cs.text_hash("fr", "abc"): ("fr", "abc")}


def test_scan_rows_matches_direct_computation(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(37, 5)).astype(np.float32)
    X[11, 2] = np.nan
    p = tmp_path / "x.npy"
    np.save(p, X)
    r = cs.scan_rows(p, 8)
    good = np.isfinite(X).all(1)
    assert r["nonfinite"].tolist() == [11] and int(r["n"]) == 37
    np.testing.assert_allclose(r["norms"][good], np.linalg.norm(X[good], axis=1), rtol=1e-6)
    assert r["norms"][11] == 0
    np.testing.assert_allclose(r["s1"], X[good].sum(0), rtol=1e-5)
    np.testing.assert_allclose(r["s2"], (X[good].astype(np.float64) ** 2).sum(0), rtol=1e-5)


def test_repeats_counts_copies_per_group():
    h = np.array([5, 5, 5, 7, 9, 9, 11], np.uint64)
    g = np.array([0, 0, 0, 0, 1, 1, 1])
    stats, copies, u, cnt = cs.repeats(h, g, 3, at=(2, 3))
    assert stats["distinct"] == 4 and np.isclose(stats["copies"], 3 / 7)
    assert np.isclose(stats["rows_in_texts_seen_2x"], 5 / 7)
    assert np.isclose(stats["rows_in_texts_seen_3x"], 3 / 7)
    np.testing.assert_allclose(copies, [2 / 4, 1 / 3, 0])
    assert dict(zip(u.tolist(), cnt.tolist())) == {5: 3, 7: 1, 9: 2, 11: 1}


def test_anisotropy_and_spectrum_on_known_cases():
    rng = np.random.default_rng(1)
    assert np.isclose(cs.anisotropy(np.tile([[1.0, 0, 0]], (10, 1))), 1.0)
    assert abs(cs.anisotropy(cs.unit(rng.normal(size=(20000, 8))))) < 0.01
    iso = cs.spectrum(rng.normal(size=(20000, 16)))
    assert 15 < iso["participation"] <= 16 and iso["k90"] >= 14
    line = rng.normal(size=(5000, 1)) * np.ones((1, 16)) + 1e-6 * rng.normal(size=(5000, 16))
    one = cs.spectrum(line)
    assert one["participation"] < 1.01 and one["k50"] == 1 and one["top1"] > 0.999
    # scaling coordinates by sqrt(w) equalizes an anisotropic Gaussian
    w = np.arange(1, 17, dtype=float)
    assert cs.spectrum(rng.normal(size=(20000, 16)) / np.sqrt(w), w)["participation"] > 15


def test_eta2_and_lda_find_planted_groups():
    rng = np.random.default_rng(2)
    g = rng.integers(0, 4, 8000)
    M = 3 * rng.normal(size=(4, 6))
    X = M[g] + rng.normal(size=(8000, 6))
    one = np.ones(6)
    p = np.bincount(g) / len(g)
    between = (p[:, None] * (M - p @ M) ** 2).sum()
    assert abs(cs.eta2(X, g, one) - between / (between + 6)) < 0.02
    assert cs.eta2(rng.normal(size=(8000, 6)), g, one) < 0.01
    assert cs.acc_lda(X, g, 1e-3, rng) > 0.95
    assert abs(cs.acc_lda(rng.normal(size=(8000, 6)), g, 1e-3, rng) - 0.25) < 0.05


def test_rank_self_finds_matches_and_restricts_to_groups():
    rng = np.random.default_rng(3)
    B = cs.unit(rng.normal(size=(300, 32)))
    A = cs.unit(B + 0.05 * rng.normal(size=B.shape))
    assert (cs.rank_self(A, B, block=64) == 0).all()
    # a shuffled B: ranks are those of a random target; restricting to a
    # group can only lower them
    Bs = B[rng.permutation(300)]
    g = rng.integers(0, 5, 300)
    ra, rg = cs.rank_self(A, Bs), cs.rank_self(A, Bs, g)
    assert (rg <= ra).all() and ra.mean() > 100


def test_pairs_respect_documents_and_groups():
    rng = np.random.default_rng(4)
    counts = rng.integers(1, 8, 200)
    doc = np.repeat(np.arange(200), counts)
    g = np.repeat(rng.integers(0, 6, 200), counts)
    P = cs.pairs(doc, g, 5000, rng)
    i, j = P["adjacent"]
    assert (doc[i] == doc[j]).all() and (j == i + 1).all() and len(i) == (counts - 1).sum()
    i, j = P["one_doc_apart"]
    assert (doc[i] == doc[j]).all() and (np.abs(i - j) >= 2).all() and len(i) > 0
    i, j = P["one_lang_two_docs"]
    assert (g[i] == g[j]).all() and (doc[i] != doc[j]).all() and len(i) > 0
    i, j = P["two_langs"]
    assert (g[i] != g[j]).all() and len(i) > 0


def test_group_means_and_weight_spread():
    X = np.array([[1.0, 0], [3, 0], [0, 4]])
    np.testing.assert_allclose(cs.group_means(X, np.array([0, 0, 1])), [[2, 0], [2, 0], [0, 4]])
    var = np.array([1.0, 1.0, 1e-4])
    w = (1 / var) / (1 / var).mean()
    r = cs.weight_spread(w, var)
    assert r["argmax"] == 2 and np.isclose(r["max_over_min"], 1e4)
    assert np.isclose(r["share_of_sum_on_max"], 1e4 / (2 + 1e4))
    # inverse-variance weights give every coordinate the same whitened variance
    assert np.isclose(r["whitened_var_per_row"], 3 * w[0] * var[0])
