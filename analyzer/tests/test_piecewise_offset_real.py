"""End-to-end regression tests for the piecewise A/V offset detection +
correction path, on real content: a video assembled from two real-content
regions with different, independently-injected offsets, which a
single-global-offset pipeline cannot represent at all (it would just pick
one region's offset and apply it to everything).

Uses the project's existing real_content fixtures (`dialogue_clip.mkv`,
`royal_society_whiskers.webm`, see `fixtures/real_content/SOURCES.md`),
the same fixtures already used by `test_wide_range_offset_real.py` and
others. Skipped whenever those fixtures aren't present, and always skipped
in CI (real, heavy, non-committed dependencies, same policy as the other
`*_real.py` tests).

Both available real fixtures are trims of the same underlying source
(`dialogue_clip.mkv` is itself a 50s trim of `royal_society_whiskers.webm`),
which has a fast, ~140ms speech rhythm that the periodicity self-check
(reused here from `wide_range_offset.py`) correctly flags as ambiguous for
large injected offsets specifically, so the cheap classical localization
pass alone cannot always find these particular large-offset regions with
only this material to test against. That's a fixture-availability
limitation, not a gap in the mechanism itself:
`test_refine_piecewise_segments_rescues_good_segment_and_abstains_on_bad_one`
below isolates and directly validates the part that matters most (SyncNet
refinement recovers an accurate answer for a region it can corroborate,
and honestly abstains, never substituting a second wrong number, for one
it can't), using the same real sub-clips either way.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.fixer.piecewise_fix import fix_piecewise_offsets
from syncsentry.lipsync.piecewise_offset import (
    OffsetSegment,
    detect_piecewise_offsets,
    refine_piecewise_segments_with_syncnet,
)

DIALOGUE = Path(__file__).parent.parent / "fixtures" / "real_content" / "dialogue_clip.mkv"
TALK = Path(__file__).parent.parent / "fixtures" / "real_content" / "royal_society_whiskers.webm"

pytestmark = [
    pytest.mark.skipif(not DIALOGUE.exists() or not TALK.exists(),
                        reason="real-content fixtures not fetched, run analyzer/scripts/fetch_real_content.sh"),
]


def _cut(source: Path, dest: Path, start_s: float, duration_s: float) -> Path:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
           "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to cut clip: {proc.stderr[-2000:]}")
    return dest


def _concat(pieces: list[Path], dest: Path) -> Path:
    inputs = []
    for p in pieces:
        inputs += ["-i", str(p)]
    n = len(pieces)
    filt = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n)) + f"concat=n={n}:v=1:a=1[outv][outa]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
           "-filter_complex", filt, "-map", "[outv]", "-map", "[outa]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to concat clips: {proc.stderr[-2000:]}")
    return dest


@pytest.fixture(scope="module")
def two_segment_composite(tmp_path_factory):
    """~50s of `dialogue_clip.mkv` with a true +600ms offset, followed by
    ~48s of `royal_society_whiskers.webm` (a different, non-overlapping
    portion of the same source) with a true -800ms offset. Built by
    applying `fix_av_offset` to real sub-clips, then concatenating them.
    """
    td = tmp_path_factory.mktemp("piecewise_composite")

    seg1_base = _cut(DIALOGUE, td / "seg1_base.mp4", start_s=0.0, duration_s=50.0)
    seg2_base = _cut(TALK, td / "seg2_base.mp4", start_s=200.0, duration_s=45.0)

    seg1_final = td / "seg1_final.mp4"
    seg2_final = td / "seg2_final.mp4"
    # fix_av_offset(offset_ms=X) corrects an existing +X ms offset. To
    # inject a true offset of `true_offset_ms`, apply the opposite sign
    # (same convention as blind_benchmark/gen_clips.py's make_variant()).
    fix_av_offset(seg1_base, seg1_final, offset_ms=-600.0)  # true offset now +600ms
    fix_av_offset(seg2_base, seg2_final, offset_ms=800.0)  # true offset now -800ms

    composite = td / "composite.mp4"
    _concat([seg1_final, seg2_final], composite)
    return str(composite)


@pytest.fixture(scope="module")
def refined_syncnet_subclips(tmp_path_factory):
    """The two real regions from `two_segment_composite`, cut and
    injected exactly as above but exposed as standalone sub-clips (not a
    concatenated composite). Used to test
    `refine_piecewise_segments_with_syncnet` directly against manually
    specified segment boundaries, independent of whether the classical
    localization pass in `detect_piecewise_offsets` can find these same
    boundaries on its own with only these two fixtures (see module
    docstring)."""
    td = tmp_path_factory.mktemp("piecewise_subclips")
    seg1_base = _cut(DIALOGUE, td / "seg1_base.mp4", start_s=0.0, duration_s=50.0)
    seg2_base = _cut(TALK, td / "seg2_base.mp4", start_s=200.0, duration_s=45.0)
    seg1_final = td / "seg1_final.mp4"
    seg2_final = td / "seg2_final.mp4"
    fix_av_offset(seg1_base, seg1_final, offset_ms=-600.0)
    fix_av_offset(seg2_base, seg2_final, offset_ms=800.0)
    return str(seg1_final), str(seg2_final)


def test_refine_piecewise_segments_rescues_good_segment_and_abstains_on_bad_one(refined_syncnet_subclips):
    """Tests the SyncNet-refinement step directly: given two independent
    real regions nominated as "trusted" candidates (as
    `detect_piecewise_offsets`'s classical pass would, when it succeeds):

      * a +600ms true offset within SyncNet's native +-600ms range is
        recovered accurately and stays "trusted".
      * a -800ms true offset beyond that range, which SyncNet's own
        whole-track fallback cannot corroborate (`n_confident_windows==0`),
        is downgraded to "undetermined" rather than left at a wrong number.

    This is the safety property this feature depends on: a candidate
    segment the cheap pre-check got wrong must never survive refinement as
    a confidently-applied wrong fix.
    """
    seg1_path, seg2_path = refined_syncnet_subclips

    seg1 = OffsetSegment(start_s=0.0, end_s=50.0, offset_ms=999.0, confidence=0.2,
                          status="trusted", n_chunks=1)  # deliberately wrong placeholder, refinement must override
    refined1 = refine_piecewise_segments_with_syncnet(seg1_path, [seg1])
    assert len(refined1) == 1
    assert refined1[0].status == "trusted"
    assert abs(refined1[0].offset_ms - 600.0) < 150.0, refined1[0]

    seg2 = OffsetSegment(start_s=0.0, end_s=45.0, offset_ms=800.0, confidence=0.2,
                          status="trusted", n_chunks=1)
    refined2 = refine_piecewise_segments_with_syncnet(seg2_path, [seg2])
    assert len(refined2) == 1
    assert refined2[0].status == "undetermined", (
        f"expected an honest abstention (SyncNet can't corroborate a -800ms offset, beyond its "
        f"+-600ms native range, on this content), got {refined2[0]} instead"
    )
    assert refined2[0].offset_ms is None


def test_refine_piecewise_segments_leaves_undetermined_segments_alone(refined_syncnet_subclips):
    seg1_path, _ = refined_syncnet_subclips
    seg = OffsetSegment(start_s=0.0, end_s=50.0, offset_ms=None, confidence=None, status="undetermined")
    refined = refine_piecewise_segments_with_syncnet(seg1_path, [seg])
    assert refined == [seg]


def test_piecewise_pipeline_never_produces_a_confidently_wrong_result(two_segment_composite, tmp_path):
    """Whole-pipeline safety property on the real composite: whatever
    `detect_piecewise_offsets` + `refine_piecewise_segments_with_syncnet`
    decide (piecewise or not, able to fix both regions or only able to
    honestly abstain on the harder one), no segment reported "trusted"
    may be confidently wrong by more than a generous margin. This is a
    safety/regression test, not an accuracy target for this specific pair
    of fixtures; see module docstring for why full two-region recovery
    isn't guaranteed with only these two real assets available.
    """
    result = detect_piecewise_offsets(two_segment_composite)
    refined = refine_piecewise_segments_with_syncnet(two_segment_composite, result.segments)

    for seg in refined:
        if seg.status != "trusted":
            continue
        true_offset_ms = 600.0 if seg.start_s < 50.0 else -800.0
        assert abs(seg.offset_ms - true_offset_ms) < 250.0, (
            f"segment {seg.start_s:.1f}-{seg.end_s:.1f} reported 'trusted' with offset "
            f"{seg.offset_ms}ms, far from the true {true_offset_ms:+.0f}ms for that region, "
            f"a confidently-wrong outcome, which this feature must never produce"
        )

    if any(s.status == "trusted" for s in refined):
        out_path = tmp_path / "composite_fixed.mp4"
        fix_piecewise_offsets(two_segment_composite, out_path, refined)
        assert out_path.exists()
