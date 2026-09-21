"""Validates the MTDVocaLiST estimator against real content with a known,
injected ground-truth offset, using the same methodology as
`test_syncnet_offset_real.py`. This estimator is built specifically to be
robust on short clips, where SyncNet's own windowed corroboration needs
more independent seconds of footage than a short clip has. On these same
two cases it recovers the injected offset exactly (0ms and 200ms), with a
much larger corroborating cluster (70+ windows vs SyncNet's 3-window
minimum). Correctness here depends on reusing SyncNet's padded face crops
while correcting for the padding (see `_TIGHT_ROW_FRAC`/`_TIGHT_COL_FRAC`
in `mtdvocalist_offset.py`); without that correction, the model's
per-window confidence is essentially uncorrelated with truth.

Skipped if either the real-content fixture or the vendored MTDVocaLiST
weights aren't present (`analyzer/scripts/fetch_real_content.sh` and
`analyzer/scripts/fetch_mtdvocalist.sh`), and always skipped in CI. Each
case takes several minutes (real face detection/tracking + many CPU
transformer forward passes), so this file deliberately covers only the
same 2 ground-truth cases as the SyncNet test, not a full sweep.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.lipsync.mtdvocalist_offset import estimate_mtdvocalist_offset, is_available

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"

pytestmark = [
    pytest.mark.skipif(not FIXTURE.exists(),
                        reason=f"real-content fixture not fetched, run analyzer/scripts/fetch_real_content.sh ({FIXTURE})"),
    pytest.mark.skipif(not is_available(),
                        reason="MTDVocaLiST not fetched, run analyzer/scripts/fetch_mtdvocalist.sh"),
]


@pytest.mark.parametrize("injected_ms", [0.0, 200.0])
def test_mtdvocalist_recovers_known_real_offset(tmp_path, injected_ms):
    shifted = tmp_path / f"shifted_{int(injected_ms)}.mkv"
    if injected_ms == 0.0:
        shifted.write_bytes(FIXTURE.read_bytes())
    else:
        fix_av_offset(FIXTURE, shifted, offset_ms=-injected_ms)  # injects true offset = injected_ms

    estimate = estimate_mtdvocalist_offset(str(shifted))

    assert estimate.n_tracks >= 1
    assert abs(estimate.offset_ms - injected_ms) <= 40.0  # exact-frame tolerance, see module docstring
    assert estimate.n_confident_windows >= 3
    assert estimate.confidence >= 5.0
