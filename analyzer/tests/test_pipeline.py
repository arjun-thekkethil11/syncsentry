"""End-to-end tests for `syncsentry.pipeline.run_fix_pipeline` -- the thing
an actual user runs: give it a broken asset, get back corrected files and a
short report confirming both issues were resolved.
"""
from syncsentry.pipeline import run_fix_pipeline
from syncsentry.synth.fixture_gen import FixtureSpec, generate_captions, generate_fixture


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
