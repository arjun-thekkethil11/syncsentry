"""Correctness tests for `fixer/piecewise_fix.py`'s ffmpeg filter-graph
splicing, using the existing synthetic flash+beep fixture generator
(`synth.fixture_gen`) with a piecewise-constant offset schedule. No real
face/video content is needed to validate that the audio-splicing itself is
sample-accurate; the already-validated coarse cross-correlation detector
(which needs no face at all) independently re-measures each region
afterward.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from syncsentry.detectors.coarse_xcorr import (
    COMMON_RATE_HZ,
    _normalize,
    _resample_uniform,
    estimate_offset_from_signals,
)
from syncsentry.fixer.piecewise_fix import fix_piecewise_offsets
from syncsentry.lipsync.piecewise_offset import OffsetSegment
from syncsentry.synth.fixture_gen import generate_piecewise_offset_fixture
from syncsentry.util.ffmpeg_io import extract_audio_envelope, extract_video_brightness, probe_duration_s


def _measure_offset_in_range(video_path: str, t0: float, t1: float) -> tuple[float, float]:
    """Re-measures the A/V offset within [t0, t1) of `video_path` using the
    coarse (no-face-needed) cross-correlation detector, independent of
    anything `piecewise_offset.py`'s face-based detection does. This is a
    separate check on the fix, not a round-trip through the same code that
    would be used to find the offset in the first place."""
    audio = extract_audio_envelope(video_path)
    video = extract_video_brightness(video_path)
    a = _normalize(_resample_uniform(audio, t0, t1, COMMON_RATE_HZ))
    v = _normalize(_resample_uniform(video, t0, t1, COMMON_RATE_HZ))
    est = estimate_offset_from_signals(a, v, COMMON_RATE_HZ, search_window_ms=800.0)
    return est.offset_ms, est.confidence


def test_fix_piecewise_offsets_corrects_each_region_independently(tmp_path):
    """Two 15s regions, injected with different, opposite-sign offsets
    (+300ms then -500ms) via a near-step control-point schedule. After
    `fix_piecewise_offsets` with the correct per-region correction, each
    region's independently re-measured residual offset must be near zero,
    proving the filter graph applies the right correction to the right
    time range, not just "some" correction somewhere in the file.
    """
    fixture = tmp_path / "piecewise_src.mkv"
    generate_piecewise_offset_fixture(
        fixture, duration_s=30.0,
        control_points=[(0.0, 300.0), (14.9, 300.0), (15.1, -500.0), (29.9, -500.0)],
        period_s=2.0, pulse_ms=80.0,
    )

    # Sanity check: the raw fixture is offset as intended in each region
    # before any fix is applied.
    raw0_ms, raw0_conf = _measure_offset_in_range(str(fixture), 1.0, 14.0)
    raw1_ms, raw1_conf = _measure_offset_in_range(str(fixture), 16.0, 29.0)
    assert raw0_conf > 0.3 and abs(raw0_ms - 300.0) < 60.0
    assert raw1_conf > 0.3 and abs(raw1_ms - (-500.0)) < 60.0

    segments = [
        OffsetSegment(start_s=0.0, end_s=15.0, offset_ms=300.0, confidence=0.9, status="trusted", n_chunks=3),
        OffsetSegment(start_s=15.0, end_s=30.0, offset_ms=-500.0, confidence=0.9, status="trusted", n_chunks=3),
    ]
    out_path = tmp_path / "piecewise_fixed.mkv"
    result = fix_piecewise_offsets(fixture, out_path, segments)

    assert result.n_shifted == 2
    assert out_path.exists()

    # Output must still span (approximately) the original duration, since
    # the whole point is to do this in one filter graph with the video
    # stream copied, not re-cut.
    assert abs(probe_duration_s(str(out_path)) - 30.0) < 0.2

    fixed0_ms, fixed0_conf = _measure_offset_in_range(str(out_path), 1.0, 14.0)
    fixed1_ms, fixed1_conf = _measure_offset_in_range(str(out_path), 16.0, 29.0)
    assert fixed0_conf > 0.3 and abs(fixed0_ms) < 60.0, f"region 1 residual {fixed0_ms}ms not corrected"
    assert fixed1_conf > 0.3 and abs(fixed1_ms) < 60.0, f"region 2 residual {fixed1_ms}ms not corrected"


def test_fix_piecewise_offsets_leaves_undetermined_segment_untouched(tmp_path):
    """A segment with `offset_ms=None` ("undetermined") must pass through
    with its audio unmodified: verified by checking its own region still
    shows the original injected offset after the "fix", not zero."""
    fixture = tmp_path / "piecewise_src2.mkv"
    generate_piecewise_offset_fixture(
        fixture, duration_s=20.0,
        control_points=[(0.0, 400.0), (19.9, 400.0)],  # one constant offset throughout
        period_s=2.0, pulse_ms=80.0,
    )

    segments = [
        OffsetSegment(start_s=0.0, end_s=20.0, offset_ms=None, confidence=None, status="undetermined"),
    ]
    out_path = tmp_path / "piecewise_untouched.mkv"
    result = fix_piecewise_offsets(fixture, out_path, segments)
    assert result.n_shifted == 0

    still_ms, still_conf = _measure_offset_in_range(str(out_path), 1.0, 19.0)
    assert still_conf > 0.3 and abs(still_ms - 400.0) < 60.0  # unchanged, not corrected to 0


def test_fix_piecewise_offsets_segment_too_short_for_its_own_offset_is_not_shifted(tmp_path):
    """A segment whose magnitude exceeds its own duration must fail open
    (left unshifted) rather than risk an invalid/degenerate atrim range."""
    fixture = tmp_path / "piecewise_src3.mkv"
    generate_piecewise_offset_fixture(
        fixture, duration_s=10.0,
        control_points=[(0.0, 100.0), (9.9, 100.0)],
        period_s=2.0, pulse_ms=80.0,
    )
    segments = [
        OffsetSegment(start_s=0.0, end_s=10.0, offset_ms=50_000.0, confidence=0.9,
                      status="trusted", n_chunks=1),  # absurd, far exceeds the 10s segment itself
    ]
    out_path = tmp_path / "piecewise_too_short.mkv"
    result = fix_piecewise_offsets(fixture, out_path, segments)
    assert result.n_shifted == 0  # fails open: did not attempt an impossible shift
    assert out_path.exists()
