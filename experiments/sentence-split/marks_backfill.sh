#!/usr/bin/env bash
# Re-split mc4_sent16 under the name mc4_sent16_marks with the builder that
# marks lines, sentences and pieces, then backfill the marks and the version
# record onto mc4_sent16 (backfill_marks.py checks the re-split chose the
# same sentences first). Run as a systemd unit with Restart=on-failure: a
# rerun resumes the split, or finds it complete and reruns the backfill,
# which finishes or reports that it already has.
set -eu
cd "$(dirname "$0")/../.."
/usr/bin/uv run --with wtpsplit --with fasttext==0.9.3 python encode_sentences.py split --name mc4_sent16_marks
exec /usr/bin/uv run python experiments/sentence-split/backfill_marks.py
