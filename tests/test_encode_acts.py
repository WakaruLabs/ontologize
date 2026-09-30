# `encode_acts.py`'s pure helpers. `boundaries` is the one that matters:
# both trainers hold out a contiguous TAIL of cache rows, so a holdout that
# is not a document boundary puts other positions from trained-on documents
# into the eval split, and every reconstruction number comes out optimistic
# with nothing to show it. Its job is to report the boundaries to snap to.
import numpy as np
import pytest
import torch as t

from encode_acts import blocks
from ontologize.data.loaders import doc_boundaries as boundaries


def starts(*counts):
    """Document start rows for documents of the given row counts."""
    return np.cumsum([0] + list(counts[:-1])), int(sum(counts))


def test_both_returns_land_on_document_boundaries():
    D, n = starts(10, 10, 10, 10, 10)          # starts 0 10 20 30 40, n=50
    lo, hi = boundaries(D, n, 17)
    assert (n - lo) in set(D.tolist()) | {n}
    assert (n - hi) in set(D.tolist()) | {n}


def test_brackets_the_requested_holdout():
    """The smaller tail is at most the request and the larger at least it,
    so a caller wanting to hold out no less than it asked for takes the
    larger -- which is what the mse_weights cutoff does."""
    D, n = starts(10, 10, 10, 10, 10)
    lo, hi = boundaries(D, n, 17)
    assert lo <= 17 <= hi
    assert (lo, hi) == (10, 20)


def test_exact_boundary_is_reported_as_itself():
    """A holdout already on a boundary must come back unchanged, or the
    script would tell the user to move a correct value."""
    D, n = starts(10, 10, 10, 10, 10)
    lo, hi = boundaries(D, n, 20)
    assert lo == 20 and 20 in (lo, hi)


def test_uneven_documents():
    D, n = starts(7, 3, 21, 5, 14)             # starts 0 7 10 31 36, n=50
    for holdout in range(1, n):
        lo, hi = boundaries(D, n, holdout)
        assert lo <= holdout <= hi
        assert (n - lo) in set(D.tolist()) | {n}
        assert (n - hi) in set(D.tolist()) | {n}


def test_degenerate_holdouts_do_not_raise():
    D, n = starts(10, 10, 10)
    assert boundaries(D, n, 0)[0] == 0          # nothing held out
    assert boundaries(D, n, n) == (n, n)        # everything held out


class Fake(t.nn.Module):
    """Stands in for an HF decoder without downloading one."""
    def __init__(self, attr, depth=3, nest=None):
        super().__init__()
        mods = t.nn.ModuleList([t.nn.Linear(2, 2) for _ in range(depth)])
        if nest:
            inner = t.nn.Module()
            setattr(inner, attr, mods)
            setattr(self, nest, inner)
        else:
            setattr(self, attr, mods)


@pytest.mark.parametrize("attr,nest", [("h", None),            # GPT2Model
                                       ("layers", None),       # most others
                                       ("h", "transformer"),   # GPT2LMHeadModel
                                       ("layers", "gpt_neox"), # GPTNeoX w/ head
                                       ("layers", "model")])   # Llama w/ head
def test_blocks_finds_the_list_across_naming_conventions(attr, nest):
    found = blocks(Fake(attr, nest=nest))
    assert isinstance(found, t.nn.ModuleList) and len(found) == 3


def test_blocks_raises_rather_than_guessing():
    """Silently hooking the wrong module would harvest a site that is not
    the one the meta.json claims."""
    with pytest.raises(ValueError, match="block list"):
        blocks(t.nn.Linear(2, 2))


def test_doc_holdout_passes_through_without_a_sidecar(tmp_path):
    """A sentence-embedding cache is one document per row, so every tail is
    already a clean split and the request must come back untouched -- this
    is what keeps SONAR's `holdout` unchanged now that it goes through here."""
    from ontologize.data.loaders import doc_holdout
    cache = tmp_path / "sentences.npy"
    np.save(cache, np.zeros((100, 4), np.float32))
    assert doc_holdout(cache, 17) == 17


def test_doc_holdout_snaps_up_with_a_sidecar(tmp_path):
    """Snapping UP, never down: the eval split may end up larger than asked
    for but never smaller, so the choice cannot quietly shrink it."""
    from ontologize.data.loaders import doc_holdout
    cache = tmp_path / "acts.npy"
    np.save(cache, np.zeros((50, 4), np.float32))
    np.save(tmp_path / "acts.docstart.npy", np.arange(0, 50, 10))
    assert doc_holdout(cache, 17) == 20
    assert doc_holdout(cache, 20) == 20
