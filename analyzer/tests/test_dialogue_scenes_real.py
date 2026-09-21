"""Validation against real (non-synthetic) content: skipped if not present.

These tests need a real video with real faces and real speech, which is
never committed to the repo (see `analyzer/fixtures/real_content/SOURCES.md`
for the exact source, license, and a fetch script to regenerate it locally).
CI will always skip this file; that's intentional, not a gap. Third-party
media is never committed, regardless of license.

Run locally after fetching the fixture:

    bash analyzer/scripts/fetch_real_content.sh
    pytest analyzer/tests/test_dialogue_scenes_real.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.lipsync.dialogue_scenes import detect_dialogue_scenes
from syncsentry.lipsync.face_detect import detect_face_presence
from syncsentry.lipsync.vad import detect_speech_segments

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason=f"real-content fixture not fetched, run analyzer/scripts/fetch_real_content.sh ({FIXTURE})",
)


def test_face_detector_finds_a_real_face_most_of_the_time():
    frames = detect_face_presence(str(FIXTURE), sample_fps=1.0)
    assert len(frames) >= 40
    hit_rate = sum(f.face_present for f in frames) / len(frames)
    # The clip cuts away to props/demos for part of its runtime, so a real
    # face isn't visible 100% of the time, but should be visible often
    # (empirically ~48%).
    assert hit_rate > 0.25


def test_vad_finds_speech_across_most_of_the_clip():
    segments = detect_speech_segments(str(FIXTURE))
    assert len(segments) > 0
    total_speech_s = sum(s.end_s - s.start_s for s in segments)
    # This is a continuous ~50s interview: someone is talking most of the
    # time (empirically ~85%+).
    assert total_speech_s > 25.0


def test_dialogue_scenes_are_a_real_subset_of_the_clip():
    scenes = detect_dialogue_scenes(str(FIXTURE))
    assert len(scenes) >= 2
    total = sum(s.end_s - s.start_s for s in scenes)
    # Face-AND-speech overlap must be <= speech-only coverage, and > 0.
    assert 0 < total < 50.0


def test_tier1_coarse_detector_is_unreliable_on_real_dialogue_content():
    """Documents the motivation for a learned estimator (DiVAS/SyncNet
    -style), rather than asserting a specific number that could bit-rot.

    Ground truth: this clip's audio and video are in sync (offset ~0ms), and
    it was never processed by the A/V fixer. The Tier-1 detector (global
    brightness/RMS cross-correlation) was built and validated against
    synthetic flash+beep content where brightness is the signal. On real
    talking-head footage, global frame brightness barely responds to
    speech, so the correlation peak is dominated by noise. Asserts the
    detector's own confidence score reflects that (it should not report
    high confidence in a spurious result) rather than asserting it gets
    the (meaningless, for this input) offset number right.
    """
    estimate = estimate_av_offset(str(FIXTURE), search_window_ms=900.0)
    assert estimate.confidence < 0.3
