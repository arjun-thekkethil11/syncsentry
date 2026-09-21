"""Validates (and falsifies) the mouth-ROI motion estimator against real
content with a known, injected ground-truth offset.

`dialogue_clip.mkv` is genuinely in sync (never processed by the fixer).
`fix_av_offset(..., offset_ms=-X)` is repurposed here as an injector rather
than a corrector: calling it with `-X` on an in-sync clip produces a file
whose true offset is exactly `+X`ms (same operation, opposite intent:
"correct a -X offset" on a 0-offset input yields a +X output). That gives
real face, real speech, and a known ground truth.

Skipped in CI, same as the rest of the real-content suite. See
`analyzer/fixtures/real_content/SOURCES.md`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.lipsync.dialogue_scenes import detect_dialogue_scenes
from syncsentry.lipsync.mouth_offset import estimate_mouth_sync_offset

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"real-content fixture not fetched, run analyzer/scripts/fetch_real_content.sh ({FIXTURE})",
)


@pytest.mark.parametrize("injected_ms", [0.0, 150.0, -200.0, 300.0])
def test_mouth_motion_estimator_does_not_reliably_track_a_known_real_offset(tmp_path, injected_ms):
    """Documents an empirically-falsified hypothesis: restricting the
    whole-frame brightness detector to a mouth-region motion signal
    (frame-to-frame pixel diff in a YuNet-landmark-derived mouth crop), even
    scoped to real dialogue scenes, was expected to beat the Tier-1
    detector on real talking-head content. It doesn't: across four known
    injected offsets on the same real clip, the estimate does not track
    ground truth and confidence stays low throughout, no better than the
    Tier-1 detector's own honest "I can't tell" on this content. Asserts
    the absence of a working correlation so a future change to this
    heuristic that starts silently reporting high confidence, without
    actually being verified to track ground truth, gets caught here rather
    than by a user.
    """
    shifted = tmp_path / f"shifted_{int(injected_ms)}.mkv"
    if injected_ms == 0.0:
        shifted.write_bytes(FIXTURE.read_bytes())
    else:
        fix_av_offset(FIXTURE, shifted, offset_ms=-injected_ms)  # injects true offset = injected_ms

    scenes = detect_dialogue_scenes(str(shifted))
    scene_intervals = [(s.start_s, s.end_s) for s in scenes]
    estimate = estimate_mouth_sync_offset(str(shifted), scenes=scene_intervals)

    # The key claim: this heuristic must not be trusted at face value. Its
    # confidence should stay below the pipeline's fix-application gate
    # regardless of the true offset, exactly like the Tier-1 detector. If
    # this ever starts failing because confidence crept up, that's worth
    # investigating (maybe the heuristic started working, or maybe it's
    # newly overconfident), but either way it should be a deliberate
    # change, not a silent one.
    assert estimate.confidence < 0.3
