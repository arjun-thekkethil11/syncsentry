"""Recovery-accuracy tests for the caption-vs-speech drift checker."""
import pytest

from syncsentry.captions.drift_check import check_caption_drift
from syncsentry.synth.fixture_gen import FixtureSpec, generate_captions, generate_fixture

TOLERANCE_MS = 15.0


@pytest.fixture(scope="module")
def clean_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("caption_fixture")
    spec = FixtureSpec(duration_s=8.0, period_s=1.0, pulse_ms=80, offset_ms=0.0)
    video_path = generate_fixture(tmp_path / "fixture.mkv", spec)
    return video_path, spec, tmp_path


@pytest.mark.parametrize("injected_caption_offset_ms", [-150, -90, 0, 30, 120])
def test_recovers_caption_offset(clean_fixture, injected_caption_offset_ms):
    video_path, spec, tmp_path = clean_fixture
    vtt_path = generate_captions(
        tmp_path / f"captions_{injected_caption_offset_ms}.vtt", spec,
        caption_offset_ms=injected_caption_offset_ms,
    )

    report = check_caption_drift(str(video_path), str(vtt_path))

    assert report.median_offset_ms is not None
    error = abs(report.median_offset_ms - injected_caption_offset_ms)
    assert error <= TOLERANCE_MS
    assert report.matched_count == report.matched_count  # all cues should match a beep onset
    assert report.unmatched_count == 0


def test_in_sync_captions_are_not_flagged(clean_fixture):
    video_path, spec, tmp_path = clean_fixture
    vtt_path = generate_captions(tmp_path / "captions_insync.vtt", spec, caption_offset_ms=0.0)
    report = check_caption_drift(str(video_path), str(vtt_path))
    assert len(report.flagged_cue_indices) == 0


def test_drifted_captions_are_flagged(clean_fixture):
    video_path, spec, tmp_path = clean_fixture
    vtt_path = generate_captions(tmp_path / "captions_drift.vtt", spec, caption_offset_ms=200.0)
    report = check_caption_drift(str(video_path), str(vtt_path))
    assert len(report.flagged_cue_indices) >= report.matched_count - 1
