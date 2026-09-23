"""End-to-end tests for `syncsentry.pipeline.run_fix_pipeline`: give it a
broken asset, get back corrected files and a short report confirming both
issues were resolved.
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
    # A 60ms offset is above the default 40ms threshold but below this
    # custom 100ms one, so it must be reported as "no issue" here.
    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=60.0)
    video = generate_fixture(tmp_path / "borderline.mkv", spec)

    summary = run_fix_pipeline(video, tmp_path / "out", av_threshold_ms=100.0)

    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.had_issue is False


@pytest.mark.skipif(not REAL_CLIP.exists(),
                     reason="real-content fixture not fetched, run analyzer/scripts/fetch_real_content.sh")
def test_pipeline_does_not_confidently_fix_low_confidence_real_content(tmp_path):
    # Regression test: on real talking-head content (this clip's
    # audio/video are genuinely in sync), the Tier-1 detector produces a
    # low/negative-confidence, spuriously "detected" offset. Without a
    # confidence gate, the pipeline would apply a real audio shift based on
    # that noise and could report a residual identical to (or as spurious
    # as) the original "detected" value while still labeling it FIXED,
    # which is misleading and, worse, capable of quietly degrading an
    # asset that was never actually broken.
    summary = run_fix_pipeline(REAL_CLIP, tmp_path / "out")

    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.had_issue is False
    assert av_issue.fixed is False
    assert "low-confidence" in av_issue.note
    assert av_issue.status == "undetermined"
    assert not summary.all_resolved  # undetermined must never report as resolved

    # The "corrected" video must be a byte-for-byte passthrough of the
    # original: an audio shift must never be applied silently if it can't
    # be verified.
    corrected = Path(summary.output_files["corrected_video"])
    assert corrected.read_bytes() == REAL_CLIP.read_bytes()


@pytest.mark.skipif(not REAL_CLIP.exists(),
                     reason="real-content fixture not fetched, run analyzer/scripts/fetch_real_content.sh")
def test_pipeline_large_offset_fix_verifies_correctly_not_spuriously_rejected(tmp_path):
    # Regression test: `run_fix_pipeline`'s post-correction residual
    # re-check used to call `_detect_av_offset` with its default
    # `recenter_large_offsets=True`, i.e. the same wide-range-seed-capable
    # detector used for the original detection. `fix_av_offset`'s
    # trim-based correction (see `fixer/av_fix.py`) leaves the corrected
    # file's audio track shorter than its video track by exactly the
    # corrected amount (there's no more real source audio to fill that
    # gap), a shape that never occurs on an original, uncorrected asset.
    # The wide-range mouth-motion detector was only ever validated against
    # original assets and produces a spurious large seed on this novel
    # shape, which fed back into SyncNet and produced a confusing,
    # unrelated residual (e.g. -2240ms after a correctly-applied +1640ms
    # correction). `fixed` was correctly left `False` (that residual's own
    # confidence was also low), but the reported number looked like the
    # fix made things worse when it hadn't.
    #
    # Fix: the residual check now explicitly passes
    # `recenter_large_offsets=False`, since a genuinely successful
    # correction's residual should be small, well inside SyncNet's native
    # +-600ms range, so recentering shouldn't be needed for a real
    # verification pass at all. This test injects a real large (1200ms)
    # offset on real content and asserts the pipeline both detects and
    # verifies it correctly end to end, not just that detection alone
    # works (that's already covered by test_syncnet_offset_real.py and
    # test_wide_range_offset_real.py).
    from syncsentry.fixer.av_fix import fix_av_offset
    broken = tmp_path / "broken_large_offset.mkv"
    fix_av_offset(REAL_CLIP, broken, offset_ms=-1200.0)  # injects true offset = +1200ms

    summary = run_fix_pipeline(broken, tmp_path / "out", av_min_confidence=0.3,
                                use_syncnet=True, syncnet_min_confidence=3.0)

    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.had_issue is True
    assert av_issue.fixed is True, (
        f"expected the correction to verify successfully, got residual="
        f"{av_issue.residual_offset_ms} confidence={av_issue.residual_confidence} note={av_issue.note!r}"
    )
    assert abs(av_issue.residual_offset_ms) <= 40.0
    # The residual's own confidence must now always be reported alongside
    # the number (previously always None/absent for this field).
    assert av_issue.residual_confidence is not None
    assert av_issue.residual_confidence >= 3.0


def test_issue_summary_status_has_no_default_and_must_be_set_explicitly():
    # Regression test: `IssueSummary.status` exists specifically to stop a
    # false "in sync" claim from silently passing as resolved. It used to
    # default to "in_sync", the single most dangerous value, so any future
    # call site that forgot to pass `status=` would silently report the
    # best possible outcome instead of failing loudly. Proves the fail-safe
    # is now enforced at construction time: omitting `status` must raise
    # TypeError, not silently default to "in_sync".
    from syncsentry.report.summary import IssueSummary

    with pytest.raises(TypeError):
        IssueSummary(name="A/V sync", had_issue=False, detected_offset_ms=None,
                     fixed=False, residual_offset_ms=None)


def test_report_text_handles_no_issue_with_note_without_crashing():
    # Regression test: the had_issue=False branch of SyncFixSummary.to_text
    # unconditionally formatted detected_offset_ms as a float, which crashed
    # with a TypeError whenever it was None (the "no matching speech onsets
    # found" caption case), and silently dropped `note` in every
    # had_issue=False case, including the new low-confidence A/V case above.
    from syncsentry.report.summary import IssueSummary, SyncFixSummary

    summary = SyncFixSummary(asset_name="test.mkv", issues=[
        IssueSummary(name="A/V sync", had_issue=False, detected_offset_ms=45.0,
                     fixed=False, residual_offset_ms=None, note="low-confidence detection (0.12)",
                     status="undetermined"),
        IssueSummary(name="Captions", had_issue=False, detected_offset_ms=None,
                     fixed=False, residual_offset_ms=None, note="no matching speech onsets found",
                     status="undetermined"),
    ])

    text = summary.to_text()  # must not raise
    assert "low-confidence detection (0.12)" in text
    assert "no matching speech onsets found" in text
    assert "UNDETERMINED" in text
    # An undetermined result must never be silently counted as resolved.
    assert not summary.all_resolved


def test_pipeline_uses_piecewise_result_when_available(tmp_path, monkeypatch):
    """Wiring test: when `_maybe_fix_piecewise` finds genuine piecewise
    structure, its result must be used instead of the normal
    single-global-offset path, and the normal path's own (expensive)
    detector must not even run. Uses a synthetic clean fixture plus a
    monkeypatched `_maybe_fix_piecewise` so this stays fast and
    deterministic rather than depending on real face content actually
    triggering the piecewise path (see `test_piecewise_offset_real.py` for
    that, slower, real-content coverage). This test is purely about the
    pipeline wiring being correct, not about the detector's own accuracy.
    """
    import syncsentry.pipeline as pipeline_mod

    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=0.0)
    video = generate_fixture(tmp_path / "clean.mkv", spec)

    canned = pipeline_mod.IssueSummary(
        name="A/V sync", had_issue=True, detected_offset_ms=None, fixed=True,
        residual_offset_ms=None, note="canned piecewise result", status="fixed",
        method="piecewise", segments=[{"start_s": 0.0, "end_s": 4.0, "offset_ms": 300.0, "status": "trusted"},
                                        {"start_s": 4.0, "end_s": 8.0, "offset_ms": -500.0, "status": "trusted"}],
    )

    def fake_maybe_fix_piecewise(video_path, corrected_video_path, av_threshold_ms,
                                  av_min_confidence, syncnet_min_confidence):
        Path(corrected_video_path).write_bytes(Path(video_path).read_bytes())
        return canned

    single_global_called = []

    def fake_single_global(*args, **kwargs):
        single_global_called.append(True)
        raise AssertionError("single-global path must not run when the piecewise path already produced a result")

    monkeypatch.setattr(pipeline_mod, "_maybe_fix_piecewise", fake_maybe_fix_piecewise)
    monkeypatch.setattr(pipeline_mod, "_fix_single_global_offset", fake_single_global)

    summary = run_fix_pipeline(video, tmp_path / "out", use_syncnet=True)

    assert not single_global_called
    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.method == "piecewise"
    assert av_issue.status == "fixed"
    assert av_issue.segments == canned.segments
    assert (tmp_path / "out" / "report.json").exists()
    assert "piecewise" in (tmp_path / "out" / "report.txt").read_text()


def test_maybe_fix_piecewise_backs_out_when_confirmed_segments_agree(tmp_path, monkeypatch):
    """Regression test for a real failure mode: the classical pre-check
    can flag multiple candidate regions as "disagreeing" purely from its
    own noise, even on a clip with one real, constant offset for its
    whole duration (measured directly on `dialogue_full50s__n500ms` in
    the blind benchmark). If SyncNet refinement then confirms several of
    those candidates and they all actually agree with each other, that
    agreement is the real signal, not the classical pre-check's
    "multiple regions" claim. The piecewise path must apply the
    agreed-on value as one global correction (and report `method`
    accordingly) instead of a partial, lower-quality per-region
    "piecewise" result.
    """
    import syncsentry.pipeline as pipeline_mod
    from syncsentry.lipsync.piecewise_offset import OffsetSegment, PiecewiseResult

    noisy_pre_check_segments = [
        OffsetSegment(0.0, 10.0, 40.0, 3.0, "trusted"),
        OffsetSegment(10.0, 20.0, 1880.0, 3.0, "trusted"),
        OffsetSegment(20.0, 30.0, 600.0, 3.0, "trusted"),
    ]

    def fake_detect_piecewise_offsets(video_path):
        return PiecewiseResult(is_piecewise=True, segments=noisy_pre_check_segments,
                                video_duration_s=30.0, n_chunks_evaluated=3, n_chunks_confident=3)

    def fake_refine(video_path, segments, min_confidence=None, abort_event=None):
        # SyncNet refinement overrides the noisy classical numbers with
        # its own; all three happen to agree on the same real offset.
        return [
            OffsetSegment(0.0, 10.0, -510.0, 8.0, "trusted"),
            OffsetSegment(10.0, 20.0, -495.0, 8.0, "trusted"),
            OffsetSegment(20.0, 30.0, -505.0, 8.0, "trusted"),
        ]

    fix_calls = []

    def fake_fix_av_offset(video_path, out_path, offset_ms, **kwargs):
        fix_calls.append(offset_ms)
        Path(out_path).write_bytes(b"corrected")

    def fake_detect_av_offset(video_path, av_threshold_ms, use_syncnet, syncnet_min_confidence,
                               av_min_confidence, use_mtdvocalist=False, recenter_large_offsets=True,
                               max_analyze_duration_s=None):
        return pipeline_mod._AVDetection(0.0, 8.0, "in_sync", 0.3, "syncnet", None)

    monkeypatch.setattr("syncsentry.lipsync.piecewise_offset.detect_piecewise_offsets",
                         fake_detect_piecewise_offsets)
    monkeypatch.setattr("syncsentry.lipsync.piecewise_offset.refine_piecewise_segments_with_syncnet",
                         fake_refine)
    monkeypatch.setattr(pipeline_mod, "fix_av_offset", fake_fix_av_offset)
    monkeypatch.setattr(pipeline_mod, "_detect_av_offset", fake_detect_av_offset)

    result = pipeline_mod._maybe_fix_piecewise(
        "input.mp4", tmp_path / "out.mp4", av_threshold_ms=50.0,
        av_min_confidence=0.3, syncnet_min_confidence=6.0,
    )

    assert result is not None
    assert result.method == "syncnet"  # not "piecewise": treated as one global offset
    assert result.status == "fixed"
    assert fix_calls == [pytest.approx(-505.0)]  # median of -510, -495, -505


def test_maybe_fix_piecewise_keeps_piecewise_path_when_segments_genuinely_disagree(tmp_path, monkeypatch):
    """Counterpart to the test above: when confirmed segments' offsets
    genuinely differ (not just classical pre-check noise), the piecewise
    path must still be taken, not backed out to a single global fix."""
    import syncsentry.pipeline as pipeline_mod
    from syncsentry.lipsync.piecewise_offset import OffsetSegment, PiecewiseResult

    pre_check_segments = [
        OffsetSegment(0.0, 10.0, 300.0, 3.0, "trusted"),
        OffsetSegment(10.0, 20.0, -500.0, 3.0, "trusted"),
    ]

    def fake_detect_piecewise_offsets(video_path):
        return PiecewiseResult(is_piecewise=True, segments=pre_check_segments,
                                video_duration_s=20.0, n_chunks_evaluated=2, n_chunks_confident=2)

    def fake_refine(video_path, segments, min_confidence=None, abort_event=None):
        return [
            OffsetSegment(0.0, 10.0, 310.0, 8.0, "trusted"),
            OffsetSegment(10.0, 20.0, -505.0, 8.0, "trusted"),
        ]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("single-global fix_av_offset must not run on genuinely piecewise content")

    monkeypatch.setattr("syncsentry.lipsync.piecewise_offset.detect_piecewise_offsets",
                         fake_detect_piecewise_offsets)
    monkeypatch.setattr("syncsentry.lipsync.piecewise_offset.refine_piecewise_segments_with_syncnet",
                         fake_refine)
    monkeypatch.setattr(pipeline_mod, "fix_av_offset", fail_if_called)
    monkeypatch.setattr("syncsentry.fixer.piecewise_fix.fix_piecewise_offsets", lambda *a, **k: None)

    result = pipeline_mod._maybe_fix_piecewise(
        "input.mp4", tmp_path / "out.mp4", av_threshold_ms=50.0,
        av_min_confidence=0.3, syncnet_min_confidence=6.0,
    )

    assert result is not None
    assert result.method == "piecewise"


def test_undetermined_duration_fraction_computes_correctly():
    from syncsentry.lipsync.piecewise_offset import OffsetSegment
    from syncsentry.pipeline import _undetermined_duration_fraction

    segs = [
        OffsetSegment(0.0, 6.0, -0.0, 5.0, "trusted"),      # 6s trusted
        OffsetSegment(6.0, 12.0, None, None, "undetermined"),  # 6s undetermined
        OffsetSegment(12.0, 18.0, 320.0, 6.0, "trusted"),   # 6s trusted
        OffsetSegment(18.0, 72.0, None, None, "undetermined"),  # 54s undetermined
    ]
    # 60/72 undetermined
    assert _undetermined_duration_fraction(segs) == pytest.approx(60.0 / 72.0)


def test_undetermined_duration_fraction_empty_is_zero():
    from syncsentry.pipeline import _undetermined_duration_fraction
    assert _undetermined_duration_fraction([]) == 0.0


def test_pipeline_falls_through_to_single_global_when_not_piecewise(tmp_path, monkeypatch):
    """The complementary case: when `_maybe_fix_piecewise` returns `None`
    (the common case for genuinely single-source content), the normal
    single-global-offset path must still run exactly as before, proving
    this pre-check doesn't silently swallow the normal case."""
    import syncsentry.pipeline as pipeline_mod

    spec = FixtureSpec(duration_s=8.0, period_s=2.0, pulse_ms=80, offset_ms=0.0)
    video = generate_fixture(tmp_path / "clean.mkv", spec)

    monkeypatch.setattr(pipeline_mod, "_maybe_fix_piecewise", lambda *a, **k: None)

    summary = run_fix_pipeline(video, tmp_path / "out", use_syncnet=False)

    av_issue = next(i for i in summary.issues if i.name == "A/V sync")
    assert av_issue.method != "piecewise"
    assert av_issue.segments is None
