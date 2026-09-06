"""Round-trip tests for the fix modules: inject a known offset, apply the
fix, and verify the *actual* residual measured by our own detectors is near
zero -- not just that the fixer "ran without error."
"""
import pytest

from syncsentry.captions.drift_check import check_caption_drift
from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.fixer.caption_fix import fix_caption_offset
from syncsentry.synth.fixture_gen import FixtureSpec, generate_captions, generate_fixture

RESIDUAL_TOLERANCE_MS = 15.0


@pytest.mark.parametrize("injected_offset_ms", [-300, -150, -50, 50, 150, 300])
def test_av_fix_round_trip(tmp_path, injected_offset_ms):
    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=injected_offset_ms)
    broken = generate_fixture(tmp_path / "broken.mkv", spec)

    before = estimate_av_offset(str(broken))
    assert abs(before.offset_ms - injected_offset_ms) <= RESIDUAL_TOLERANCE_MS

    fixed_path = tmp_path / "fixed.mkv"
    fix_av_offset(broken, fixed_path, before.offset_ms)

    after = estimate_av_offset(str(fixed_path))
    assert abs(after.offset_ms) <= RESIDUAL_TOLERANCE_MS
    assert after.direction == "in_sync"


@pytest.mark.parametrize("injected_caption_offset_ms", [-200, -90, 90, 200])
def test_caption_fix_round_trip(tmp_path, injected_caption_offset_ms):
    spec = FixtureSpec(duration_s=8.0, period_s=1.0, pulse_ms=80, offset_ms=0.0)
    video = generate_fixture(tmp_path / "video.mkv", spec)
    broken_vtt = generate_captions(tmp_path / "broken.vtt", spec, caption_offset_ms=injected_caption_offset_ms)

    before = check_caption_drift(str(video), str(broken_vtt))
    assert before.median_offset_ms is not None

    fixed_vtt = tmp_path / "fixed.vtt"
    fix_caption_offset(broken_vtt, fixed_vtt, before.median_offset_ms)

    after = check_caption_drift(str(video), str(fixed_vtt))
    assert abs(after.median_offset_ms) <= RESIDUAL_TOLERANCE_MS
    assert len(after.flagged_cue_indices) == 0


def test_no_change_when_already_in_sync(tmp_path):
    spec = FixtureSpec(duration_s=6.0, period_s=2.0, pulse_ms=80, offset_ms=2.0)
    video = generate_fixture(tmp_path / "clean.mkv", spec)

    result = fix_av_offset(video, tmp_path / "unchanged.mkv", offset_ms=2.0)
    assert result.method == "no_change"
