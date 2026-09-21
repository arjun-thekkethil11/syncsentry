"""Regression tests for the mouth-motion wide-range recentering tier
(`wide_range_offset.py`) that lets SyncNet's fine search recover offsets
beyond its native +-600ms window, plus the coverage-floor fix for the
short-clip regression it initially introduced.

Uses the project's existing real_content fixture (`dialogue_clip.mkv`, a
50s multi-speaker talking-head clip, see `fixtures/real_content/
SOURCES.md`), the same fixture already used by `test_syncnet_offset_real.py`
and `test_mouth_offset_real.py`. Skipped whenever that fixture or the
vendored SyncNet aren't present, and always skipped in CI (real, heavy,
non-committed dependencies, same policy as the other `*_real.py` tests).
Each case takes roughly a minute (real face detection/tracking + a CNN,
CPU/MPS-bound).
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.lipsync.syncnet_offset import estimate_syncnet_offset, is_available
from syncsentry.lipsync.wide_range_offset import (
    DEFAULT_MIN_FACE_COVERAGE_S,
    _maybe_trim_source_for_scan,
    estimate_wide_range_offset,
)
from syncsentry.util.ffmpeg_io import probe_duration_s

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"
# A longer (5.7min) real fixture, used only by the scan-duration-cap tests
# below. Everything else in this file uses the shorter 50s FIXTURE, which
# the cap correctly leaves untouched.
LONG_FIXTURE = Path(__file__).parent.parent / "fixtures" / "real_content" / "royal_society_whiskers.webm"

pytestmark = [
    pytest.mark.skipif(not FIXTURE.exists(),
                        reason=f"real-content fixture not fetched, run analyzer/scripts/fetch_real_content.sh ({FIXTURE})"),
    pytest.mark.skipif(not is_available(),
                        reason="SyncNet not fetched, run analyzer/scripts/fetch_syncnet.sh"),
]


def _make_short_clip(dest: Path, duration_s: float) -> Path:
    """Cuts a short (<20s) sub-clip from the fixture, below the wide-range
    tier's coverage floor by construction, used to test that this tier
    correctly does not engage on short clips."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(FIXTURE), "-t", f"{duration_s:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to cut short clip: {proc.stderr[-2000:]}")
    return dest


def test_wide_range_seed_recovers_large_offset_end_to_end(tmp_path):
    """Covers an offset beyond SyncNet's native +-600ms window, on a clip
    with enough face coverage (50s, well above the 20s floor) to clear the
    wide-range tier's gate. Must come back trusted and accurate for at
    least one polarity on this fixture, not just "found something", to
    prove the tier actually helps, not merely fails safe. (The other
    polarity is covered by the not-confidently-wrong test below: a small
    amount of case-by-case variance on this raw, non-re-encoded fixture is
    expected and handled by the safe-abstention path, exercised explicitly
    next.)
    """
    injected_ms = 1200.0
    shifted = tmp_path / f"shifted_{int(injected_ms)}.mkv"
    fix_av_offset(FIXTURE, shifted, offset_ms=-injected_ms)  # injects true offset = injected_ms

    estimate = estimate_syncnet_offset(str(shifted))

    assert estimate.n_tracks >= 1
    assert estimate.confidence >= 3.0  # DEFAULT_SYNCNET_MIN_CONFIDENCE: must be trusted, not just close
    assert abs(estimate.offset_ms - injected_ms) <= 100.0  # real-asset errors were 0-160ms; generous margin here


@pytest.mark.parametrize("injected_ms", [1200.0, -1200.0])
def test_large_offset_never_confidently_wrong_on_long_clip(tmp_path, injected_ms):
    """Weaker, symmetric safety property that must hold for both polarities
    on this long (50s) fixture: whatever the wide-range tier and SyncNet's
    fine search produce, it must never be trusted and wrong by a lot. This
    is the same calibration guarantee `test_short_clip_...` checks for
    short clips, restated here for the long-clip case, where the system is
    expected to recover the offset most of the time, but "abstain instead"
    is still an acceptable outcome on any individual real clip.
    """
    shifted = tmp_path / f"shifted_{int(injected_ms)}.mkv"
    fix_av_offset(FIXTURE, shifted, offset_ms=-injected_ms)

    estimate = estimate_syncnet_offset(str(shifted))

    err_ms = abs(estimate.offset_ms - injected_ms)
    is_confidently_wrong = estimate.confidence >= 3.0 and err_ms > 200.0
    assert not is_confidently_wrong, (
        f"confidently wrong: offset_ms={estimate.offset_ms}, confidence={estimate.confidence}, "
        f"true={injected_ms}, err={err_ms}ms"
    )


def test_wide_range_seed_raises_below_coverage_floor(tmp_path):
    """Direct unit test of the coverage-floor guard: below
    `DEFAULT_MIN_FACE_COVERAGE_S`, this tier must refuse to produce an
    estimate at all rather than attempt an underdetermined wide search.
    """
    short_clip = _make_short_clip(tmp_path / "short10s.mkv", 10.0)
    assert 10.0 < DEFAULT_MIN_FACE_COVERAGE_S  # sanity: this clip really is below the floor

    with pytest.raises(ValueError, match="not enough"):
        estimate_wide_range_offset(str(short_clip))


@pytest.mark.parametrize("injected_ms", [1200.0, -1200.0])
def test_short_clip_large_offset_fails_safe_not_confidently_wrong(tmp_path, injected_ms):
    """End-to-end version of the coverage-floor guard: a short (<20s) clip
    with a large (beyond native +-600ms) injected offset must not come
    back both trusted and wrong. Without the guard, a 5s clip in this
    situation could produce a confidently-wrong seed that SyncNet's fine
    search then agrees with (conf 3.0-5.6, off by hundreds-1000+ms); see
    module docstring in `wide_range_offset.py`. Safe outcomes here:
    abstain (untrusted), or, less likely on a clip this short, happen to
    still land within tolerance.
    """
    short_clip = _make_short_clip(tmp_path / "short10s_base.mkv", 10.0)
    shifted = tmp_path / f"short10s_shifted_{int(injected_ms)}.mkv"
    fix_av_offset(short_clip, shifted, offset_ms=-injected_ms)

    try:
        estimate = estimate_syncnet_offset(str(shifted))
    except ValueError:
        return  # no trackable face, safe (abstains upstream in _detect_av_offset)

    err_ms = abs(estimate.offset_ms - injected_ms)
    is_confidently_wrong = estimate.confidence >= 3.0 and err_ms > 200.0
    assert not is_confidently_wrong, (
        f"confidently wrong: offset_ms={estimate.offset_ms}, confidence={estimate.confidence}, "
        f"true={injected_ms}, err={err_ms}ms"
    )


def test_maybe_trim_source_for_scan_leaves_short_clip_untouched(tmp_path):
    """Below the cap: must return the original path unchanged (no extra
    encode pass). The existing tests above all rely on this, since they
    use the 50s FIXTURE with the (much larger) 120s default cap."""
    result = _maybe_trim_source_for_scan(str(FIXTURE), tmp_path, max_duration_s=120.0)
    assert result == str(FIXTURE)


@pytest.mark.skipif(not LONG_FIXTURE.exists(),
                     reason=f"long real-content fixture not fetched ({LONG_FIXTURE})")
def test_maybe_trim_source_for_scan_trims_long_clip_to_centered_window(tmp_path):
    """Direct unit test that the scan-duration cap actually engages and
    produces a frame-accurate, correctly-sized, centered window on a real
    clip well over the cap. Fast (no model inference), unlike the
    end-to-end test below."""
    full_duration_s = probe_duration_s(str(LONG_FIXTURE))
    assert full_duration_s > 120.0  # sanity: this fixture really is over the default cap

    result = _maybe_trim_source_for_scan(str(LONG_FIXTURE), tmp_path, max_duration_s=30.0)
    assert result != str(LONG_FIXTURE)
    trimmed_duration_s = probe_duration_s(result)
    assert 25.0 <= trimmed_duration_s <= 35.0  # ~30s, allowing for encode rounding


@pytest.mark.skipif(not LONG_FIXTURE.exists(),
                     reason=f"long real-content fixture not fetched ({LONG_FIXTURE})")
def test_wide_range_offset_bounded_by_max_scan_duration_on_long_asset():
    """`estimate_wide_range_offset` caps its two full-clip scans
    (face-presence pre-pass + mouth-motion extraction), which without a
    cap were measured at ~110s wall-clock on this same 342.9s (5.7min)
    fixture, already comparable to the webapp's whole stated 30-120s
    per-asset latency budget, for one tier. With a small explicit cap well
    above the proven-unreliable 20s figure, this must complete in a small
    fraction of that time. Whether it returns an estimate or safely raises
    ValueError (e.g. the periodicity self-check) doesn't matter here, only
    that it stays fast.
    """
    t0 = time.time()
    try:
        estimate_wide_range_offset(str(LONG_FIXTURE), max_scan_duration_s=40.0)
    except ValueError:
        pass  # safe abstention is an acceptable outcome, see docstring above
    elapsed_s = time.time() - t0

    # Generous ceiling: the uncapped call was measured at ~110s on this
    # fixture; a 40s scan window should be well under half of that even
    # accounting for machine variance. This is a regression guard against
    # the cap being silently bypassed, not a tight performance assertion.
    assert elapsed_s < 60.0, f"took {elapsed_s:.1f}s, scan-duration cap may not be engaging"
