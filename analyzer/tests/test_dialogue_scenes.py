"""Unit tests for the M3b dialogue-scene modules.

Split in two:

  * Synthetic tests (this file, minus the `real_content` ones) run in CI --
    they exercise the interval-algebra and "no face / no speech" code paths
    using our existing synthetic fixture generator, with no dependency on
    real video.
  * Real-content tests (`test_dialogue_scenes_real.py`) validate the actual
    face detector and VAD against genuine human faces/speech, which the
    synthetic tone-and-flash fixture cannot provide. They're skipped
    automatically if the real-content fixture hasn't been fetched locally
    (see `analyzer/fixtures/real_content/README.md`) -- CI never has it,
    by design (never commit third-party video, even permissively licensed).
"""
from __future__ import annotations

from syncsentry.lipsync.dialogue_scenes import _intersect, _merge_close
from syncsentry.lipsync.face_detect import detect_face_presence
from syncsentry.lipsync.vad import detect_speech_segments
from syncsentry.synth.fixture_gen import FixtureSpec, generate_fixture


def test_intersect_basic_overlap():
    a = [(0.0, 5.0), (10.0, 15.0)]
    b = [(3.0, 12.0)]
    assert _intersect(a, b) == [(3.0, 5.0), (10.0, 12.0)]


def test_intersect_no_overlap():
    a = [(0.0, 1.0)]
    b = [(2.0, 3.0)]
    assert _intersect(a, b) == []


def test_merge_close_bridges_small_gap():
    intervals = [(0.0, 1.0), (1.3, 2.0), (5.0, 6.0)]
    merged = _merge_close(intervals, max_gap_s=0.5)
    assert merged == [(0.0, 2.0), (5.0, 6.0)]


def test_merge_close_keeps_far_apart_intervals_separate():
    intervals = [(0.0, 1.0), (5.0, 6.0)]
    merged = _merge_close(intervals, max_gap_s=0.5)
    assert merged == intervals


def test_face_detect_finds_no_face_in_synthetic_fixture(tmp_path_factory):
    # The flash+beep fixture has no faces at all -- a real face detector
    # (not a stub) should correctly report zero detections throughout,
    # which is a meaningful assertion precisely because it's a *negative*
    # result from a real model, not a mocked one.
    out = tmp_path_factory.mktemp("fixt") / "no_face.mkv"
    video_path = generate_fixture(out, FixtureSpec(duration_s=4.0))
    frames = detect_face_presence(str(video_path), sample_fps=2.0)
    assert len(frames) > 0
    assert all(not f.face_present for f in frames)


def test_vad_finds_no_speech_in_synthetic_tone_fixture(tmp_path_factory):
    # A gated sine tone is exactly the kind of periodic non-speech signal a
    # real speech VAD should reject -- unlike the caption checker's energy
    # threshold onset picker (syncsentry.captions.drift_check), which is
    # tuned to fire on any loud pulse, tone or speech alike. This asserts
    # Silero VAD is doing something meaningfully different from that.
    out = tmp_path_factory.mktemp("fixt") / "tone_only.mkv"
    video_path = generate_fixture(out, FixtureSpec(duration_s=6.0, period_s=1.0, pulse_ms=200.0))
    segments = detect_speech_segments(str(video_path))
    assert segments == []
