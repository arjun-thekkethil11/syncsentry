"""Validates the pretrained SyncNet estimator (M3c) against real content
with a KNOWN, injected ground-truth offset -- same methodology as
`test_mouth_offset_real.py`, which used this to *falsify* the classical
mouth-motion heuristic. This one is the positive result (see
docs/RESEARCH.md section 1e): unlike that heuristic and the Tier-1 coarse
detector, SyncNet actually recovers the injected offset.

Skipped if either the real-content fixture or the vendored syncnet_python
+ weights aren't present (`analyzer/scripts/fetch_real_content.sh` and
`analyzer/scripts/fetch_syncnet.sh`) -- and always skipped in CI, since both
are real, heavy, non-committed dependencies. Each case takes a few minutes
(real face detection/tracking + a CNN, CPU-bound), so this file
deliberately covers only 2 cases, not the full 4-point sweep done manually
during development (see docs/RESEARCH.md section 1e for those numbers).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.lipsync.syncnet_offset import estimate_syncnet_offset, is_available

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"

pytestmark = [
    pytest.mark.skipif(not FIXTURE.exists(),
                        reason=f"real-content fixture not fetched -- run analyzer/scripts/fetch_real_content.sh ({FIXTURE})"),
    pytest.mark.skipif(not is_available(),
                        reason="SyncNet not fetched -- run analyzer/scripts/fetch_syncnet.sh"),
]


@pytest.mark.parametrize("injected_ms", [0.0, 200.0])
def test_syncnet_recovers_known_real_offset(tmp_path, injected_ms):
    """Unlike the falsified mouth-motion heuristic (test_mouth_offset_real.py),
    this must actually track ground truth -- that's the whole point of using
    a pretrained model instead of a hand-crafted motion signal. Tolerance is
    1.5 frames (60ms at 25fps) to allow for the face-tracker resampling the
    clip to exactly 25fps.
    """
    shifted = tmp_path / f"shifted_{int(injected_ms)}.mkv"
    if injected_ms == 0.0:
        shifted.write_bytes(FIXTURE.read_bytes())
    else:
        fix_av_offset(FIXTURE, shifted, offset_ms=-injected_ms)  # injects true offset = injected_ms

    estimate = estimate_syncnet_offset(str(shifted))

    assert estimate.n_tracks >= 1
    assert abs(estimate.offset_ms - injected_ms) <= 60.0
    # Confidence should stay in the "trackable face" range regardless of
    # offset magnitude -- see docs/RESEARCH.md section 1e for why this is a
    # different scale than the coarse detector's [0, 1]-ish correlation.
    assert estimate.confidence >= 3.0
