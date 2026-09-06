"""End-to-end tests for `syncsentry.pipeline.run_fix_pipeline` -- the thing
an actual user runs: give it a broken asset, get back corrected files and a
short report confirming both issues were resolved.
"""
from pathlib import Path

import pytest

from syncsentry.pipeline import run_fix_pipeline
from syncsentry.synth.fixture_gen import FixtureSpec, generate_captions, generate_fixture

REAL_CLIP = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"


def test_full_pipeline_fixes_both_av_and_caption_drift(tmp_path):
    spec = FixtureSpec(duration_s=10.0, period_s=2.0, pulse_ms=80, offset_ms=180.0)
    video = generate_fixture(tmp_path / "broken.mkv", spec)
    captions = generate_captions(tmp_path / "broken.vtt", spec, caption_offset_ms=-110.0)

    summary = run_fix_pipeline(video, tmp_path / "out", captions_path=captions)

    assert summary.all_resolved
    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    cap_issue = next(i for i in summary.issues if i.name == "Captions")

    assert av_issue.had_issue and av_issue.fixed
    assert abs(av_issue.residual_offset_ms) < 15.0

    assert cap_issue.had_issue and cap_issue.fixed
    assert abs(cap_issue.residual_offset_ms) < 15.0

    assert (tmp_path / "out" / "report.txt").exists()
    assert (tmp_path / "out" / "report.json").exists()


def test_pipeline_no_issues_when_asset_is_clean(tmp_path):
    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=0.0)
    video = generate_fixture(tmp_path / "clean.mkv", spec)
    captions = generate_captions(tmp_path / "clean.vtt", spec, caption_offset_ms=0.0)

    summary = run_fix_pipeline(video, tmp_path / "out", captions_path=captions)

    assert summary.all_resolved
    assert all(not i.had_issue for i in summary.issues)


def test_pipeline_video_only_skips_caption_step(tmp_path):
    spec = FixtureSpec(duration_s=6.0, period_s=2.0, pulse_ms=80, offset_ms=200.0)
    video = generate_fixture(tmp_path / "broken.mkv", spec)

    summary = run_fix_pipeline(video, tmp_path / "out", captions_path=None)

    assert len(summary.issues) == 1
    assert summary.issues[0].name == "A/V sync"
    assert "corrected_captions" not in summary.output_files


def test_pipeline_respects_custom_av_threshold(tmp_path):
    # Regression test: run_fix_pipeline accepted an `av_threshold_ms` param
    # but never actually passed it to the detector, so a custom threshold
    # was silently ignored (the CLI's --av-threshold-ms flag was a no-op).
    # A 60ms offset is above the *default* 40ms threshold but below this
    # custom 100ms one, so it must be reported as "no issue" here.
    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=60.0)
    video = generate_fixture(tmp_path / "borderline.mkv", spec)

    summary = run_fix_pipeline(video, tmp_path / "out", av_threshold_ms=100.0)

    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.had_issue is False


@pytest.mark.skipif(not REAL_CLIP.exists(),
                     reason="real-content fixture not fetched -- run analyzer/scripts/fetch_real_content.sh")
def test_pipeline_does_not_confidently_fix_low_confidence_real_content(tmp_path):
    # Regression test for a real bug report: on real talking-head content
    # (this clip's audio/video are genuinely in sync), the Tier-1 detector
    # produces a low/negative-confidence, spuriously "detected" offset (see
    # docs/RESEARCH.md section 1b). Before the confidence gate, the pipeline
    # would apply a real audio shift based on that noise and could report a
    # residual identical to (or as spurious as) the original "detected"
    # value while still labeling it FIXED -- misleading and, worse, capable
    # of quietly degrading an asset that was never actually broken.
    summary = run_fix_pipeline(REAL_CLIP, tmp_path / "out")

    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.had_issue is False
    assert av_issue.fixed is False
    assert "low-confidence" in av_issue.note

    # The "corrected" video must be a byte-for-byte passthrough of the
    # original -- we must not silently apply an audio shift we can't verify.
    corrected = Path(summary.output_files["corrected_video"])
    assert corrected.read_bytes() == REAL_CLIP.read_bytes()


def test_report_text_handles_no_issue_with_note_without_crashing():
    # Regression test: the had_issue=False branch of SyncFixSummary.to_text
    # unconditionally formatted detected_offset_ms as a float, which crashed
    # with a TypeError whenever it was None (the "no matching speech onsets
    # found" caption case), and silently dropped `note` in every
    # had_issue=False case, including the new low-confidence A/V case above.
    from syncsentry.report.summary import IssueSummary, SyncFixSummary

    summary = SyncFixSummary(asset_name="test.mkv", issues=[
        IssueSummary(name="A/V sync", had_issue=False, detected_offset_ms=45.0,
                     fixed=False, residual_offset_ms=None, note="low-confidence detection (0.12)"),
        IssueSummary(name="Captions", had_issue=False, detected_offset_ms=None,
                     fixed=False, residual_offset_ms=None, note="no matching speech onsets found"),
    ])

    text = summary.to_text()  # must not raise
    assert "low-confidence detection (0.12)" in text
    assert "no matching speech onsets found" in text
