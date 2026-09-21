"""Recovery-accuracy tests for the coarse A/V offset detector.

Inject a known offset into otherwise-clean synchronized material, run the
detector, and assert the recovered offset is within tolerance of ground
truth.
"""
import pytest

from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.synth.fixture_gen import FixtureSpec, generate_fixture

TOLERANCE_MS = 15.0  # sub video-frame at 25fps (40ms); envelope hop is 5ms


@pytest.mark.parametrize("injected_offset_ms", [-400, -150, -30, -5, 0, 5, 30, 150, 400])
def test_recovers_injected_offset(tmp_path, injected_offset_ms):
    spec = FixtureSpec(duration_s=6.0, period_s=1.0, pulse_ms=80, offset_ms=injected_offset_ms)
    video_path = generate_fixture(tmp_path / "fixture.mkv", spec)

    estimate = estimate_av_offset(str(video_path))

    error = abs(estimate.offset_ms - injected_offset_ms)
    assert error <= TOLERANCE_MS, (
        f"injected={injected_offset_ms}ms estimated={estimate.offset_ms}ms error={error}ms"
    )
    assert estimate.confidence > 0.5


def test_periodic_ambiguity_boundary_is_documented(tmp_path):
    """Cross-correlating a periodic pulse train is only unambiguous up to
    +/- period/2: a lag of +period/2 and -period/2 look identical. Pins
    down that known limitation so it can't silently regress, and so the
    coarse detector's default parameters aren't changed without accounting
    for why -500ms aliases on a 1s-period fixture.
    """
    period_s = 1.0
    spec = FixtureSpec(duration_s=8.0, period_s=period_s, pulse_ms=80, offset_ms=-500)
    video_path = generate_fixture(tmp_path / "ambiguous.mkv", spec)

    estimate = estimate_av_offset(str(video_path), search_window_ms=500.0)

    # At exactly the ambiguity boundary, the detector may lock onto either
    # +500ms or -500ms: both are "correct" under periodic aliasing.
    assert abs(estimate.offset_ms) == pytest.approx(500.0, abs=TOLERANCE_MS)


def test_direction_classification(tmp_path):
    cases = {-300: "audio_leads", 0: "in_sync", 300: "audio_lags"}
    for offset_ms, expected_dir in cases.items():
        spec = FixtureSpec(duration_s=6.0, period_s=1.0, pulse_ms=80, offset_ms=offset_ms)
        video_path = generate_fixture(tmp_path / f"fixture_{offset_ms}.mkv", spec)
        estimate = estimate_av_offset(str(video_path))
        assert estimate.direction == expected_dir
