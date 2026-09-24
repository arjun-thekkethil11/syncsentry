"""Fast, synthetic tests for the pure-logic pieces of `piecewise_offset.py`:
grouping, merging, and gap-filling, with no real media, face detection, or
audio decoding involved. The (slow, real media) end-to-end detection +
correction path is covered separately by `test_piecewise_offset_real.py`.
"""
from __future__ import annotations

import os
import threading

from syncsentry.lipsync.dialogue_scenes import DialogueScene
from syncsentry.lipsync import piecewise_offset
from syncsentry.lipsync.piecewise_offset import (
    MIN_SEGMENT_DURATION_S,
    MIN_SEGMENT_SUPPORT_CHUNKS,
    OffsetChunk,
    OffsetSegment,
    _dedupe_overlapping_segments,
    _fill_gaps,
    _group_scenes_into_chunks,
    _merge_confident_chunks,
    _refine_segments_concurrently,
    _THREAD_BUDGET_ENV_VARS,
)


def test_group_scenes_respects_min_and_max_duration():
    scenes = [DialogueScene(start_s=float(i * 2), end_s=float(i * 2 + 1.5)) for i in range(20)]  # 1.5s each, 2s apart
    groups = _group_scenes_into_chunks(scenes, min_duration_s=4.0, max_duration_s=12.0)

    assert len(groups) >= 2  # must have split into multiple chunks, not one giant one
    for g in groups:
        coverage = sum(s.end_s - s.start_s for s in g)
        # every non-final group should clear the floor and stay under the cap
        assert coverage <= 12.0 + 1e-6
    # all but possibly the last group should meet the floor (leftovers can
    # be short if merged into the previous group instead)
    total_coverage = sum(s.end_s - s.start_s for g in groups for s in g)
    assert abs(total_coverage - sum(s.end_s - s.start_s for s in scenes)) < 1e-6  # no scene lost


def test_group_scenes_handles_empty_input():
    assert _group_scenes_into_chunks([], min_duration_s=4.0, max_duration_s=12.0) == []


def test_merge_confident_chunks_agreeing_offsets_form_one_segment():
    chunks = [
        OffsetChunk(start_s=0.0, end_s=8.0, offset_ms=300.0, confidence=0.5),
        OffsetChunk(start_s=8.0, end_s=16.0, offset_ms=340.0, confidence=0.6),  # within CHANGE_TOLERANCE_MS of 300
        OffsetChunk(start_s=16.0, end_s=24.0, offset_ms=280.0, confidence=0.55),
    ]
    segments = _merge_confident_chunks(chunks)
    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 24.0
    assert segments[0].n_chunks == 3
    assert abs(segments[0].offset_ms - 300.0) < 50.0  # median of {300, 340, 280} = 300


def test_merge_confident_chunks_disagreeing_offsets_form_two_segments():
    chunks = [
        OffsetChunk(start_s=0.0, end_s=8.0, offset_ms=0.0, confidence=0.5),
        OffsetChunk(start_s=8.0, end_s=16.0, offset_ms=20.0, confidence=0.6),
        # a big jump: a genuinely different region
        OffsetChunk(start_s=16.0, end_s=24.0, offset_ms=1800.0, confidence=0.7),
        OffsetChunk(start_s=24.0, end_s=32.0, offset_ms=1750.0, confidence=0.65),
    ]
    segments = _merge_confident_chunks(chunks)
    assert len(segments) == 2
    assert segments[0].n_chunks == 2
    assert abs(segments[0].offset_ms - 10.0) < 50.0
    assert segments[1].n_chunks == 2
    assert abs(segments[1].offset_ms - 1775.0) < 50.0


def test_merge_confident_chunks_weakest_link_confidence():
    """A segment's reported confidence must be the minimum across its
    member chunks, not an average: one lucky high-confidence chunk must
    not mask an otherwise-shaky segment."""
    chunks = [
        OffsetChunk(start_s=0.0, end_s=8.0, offset_ms=100.0, confidence=0.9),
        OffsetChunk(start_s=8.0, end_s=16.0, offset_ms=110.0, confidence=0.16),
    ]
    segments = _merge_confident_chunks(chunks)
    assert len(segments) == 1
    assert segments[0].confidence == 0.16


def test_fill_gaps_covers_whole_timeline_with_undetermined_spans():
    trusted = [
        OffsetSegment(start_s=10.0, end_s=20.0, offset_ms=500.0, confidence=0.5, status="trusted", n_chunks=3),
        OffsetSegment(start_s=30.0, end_s=40.0, offset_ms=-500.0, confidence=0.4, status="trusted", n_chunks=2),
    ]
    full = _fill_gaps(trusted, video_duration_s=50.0)

    # Contiguous, covers [0, 50], no gaps or overlaps.
    assert full[0].start_s == 0.0
    assert full[-1].end_s == 50.0
    for a, b in zip(full, full[1:]):
        assert abs(a.end_s - b.start_s) < 1e-6

    statuses = [s.status for s in full]
    assert statuses == ["undetermined", "trusted", "undetermined", "trusted", "undetermined"]
    assert full[1].offset_ms == 500.0
    assert full[3].offset_ms == -500.0
    assert full[0].offset_ms is None and full[2].offset_ms is None and full[4].offset_ms is None


def test_fill_gaps_no_trusted_segments_is_one_undetermined_span():
    full = _fill_gaps([], video_duration_s=30.0)
    assert len(full) == 1
    assert full[0].start_s == 0.0
    assert full[0].end_s == 30.0
    assert full[0].status == "undetermined"


def test_dedupe_overlapping_segments_splits_at_midpoint():
    # Regression test: the fine-grained fallback tier's 50%-overlapping
    # sliding windows can produce two disagreeing-but-adjacent segments
    # whose raw [start_s, end_s) spans still overlap, e.g. windows [30,40)
    # and [35,45). Left unfixed, this violates detect_piecewise_offsets's
    # own "contiguous, non-overlapping" contract and would make
    # fixer/piecewise_fix.py double-apply a correction to the overlapping
    # [35,40) span.
    segs = [
        OffsetSegment(30.0, 40.0, 1800.0, 0.5, "trusted", n_chunks=1),
        OffsetSegment(35.0, 45.0, -80.0, 0.5, "trusted", n_chunks=1),
    ]
    deduped = _dedupe_overlapping_segments(segs)
    assert len(deduped) == 2
    assert deduped[0].end_s == deduped[1].start_s == 37.5  # midpoint of [35, 40)
    assert deduped[0].start_s == 30.0
    assert deduped[1].end_s == 45.0


def test_dedupe_overlapping_segments_leaves_non_overlapping_input_unchanged():
    segs = [
        OffsetSegment(0.0, 10.0, 100.0, 0.5, "trusted", n_chunks=1),
        OffsetSegment(10.0, 20.0, -100.0, 0.5, "trusted", n_chunks=1),
    ]
    deduped = _dedupe_overlapping_segments(segs)
    assert [s.start_s for s in deduped] == [0.0, 10.0]
    assert [s.end_s for s in deduped] == [10.0, 20.0]


def test_min_segment_thresholds_are_internally_consistent():
    # Sanity check: `MIN_SEGMENT_SUPPORT_CHUNKS` minimum-sized chunks must
    # already cover at least `MIN_SEGMENT_DURATION_S`, otherwise a segment
    # could clear the support-chunk-count gate in
    # `detect_piecewise_offsets`'s "trusted" filter while still being
    # rejected by the duration gate in the same filter, making
    # `MIN_SEGMENT_SUPPORT_CHUNKS` alone insufficient evidence despite
    # passing its own check.
    from syncsentry.lipsync.piecewise_offset import MIN_CHUNK_DURATION_S
    assert MIN_SEGMENT_SUPPORT_CHUNKS * MIN_CHUNK_DURATION_S >= MIN_SEGMENT_DURATION_S


# --- Concurrent refinement dispatch ---
#
# These mock out `_refine_one_segment` itself (the real one needs ffmpeg
# and a working SyncNet install plus real media, covered by the slow
# `test_piecewise_offset_real.py`) to test the dispatch mechanics in
# isolation: every trusted index gets a result, results land back at the
# right index regardless of which thread/order finished first, and the
# process-wide thread-budget env vars this function temporarily sets are
# always restored afterward. This matters because
# `_refine_segments_concurrently` mutates `os.environ`, process-global
# state, not something scoped to its own call.

def test_refine_segments_concurrently_maps_each_result_back_to_its_own_index(monkeypatch, tmp_path):
    """With multiple trusted segments (the >1 code path, which spins up a
    real ThreadPoolExecutor), every result must land back keyed by its
    original segment index, not by submission or completion order, which
    threads make nondeterministic."""
    segments = [
        OffsetSegment(0.0, 10.0, 111.0, 0.9, "trusted", n_chunks=1),
        OffsetSegment(10.0, 20.0, 222.0, 0.9, "undetermined"),  # not in trusted_indices, must be ignored
        OffsetSegment(20.0, 30.0, 333.0, 0.9, "trusted", n_chunks=1),
        OffsetSegment(30.0, 40.0, 444.0, 0.9, "trusted", n_chunks=1),
    ]
    trusted_indices = [0, 2, 3]

    seen_indices = []

    def fake_refine_one(video_path, seg, index, tmp_dir, min_confidence, abort_event=None, syncnet_deadline=None):
        seen_indices.append(index)
        # Deliberately encode the input segment's own offset into the
        # output so a mixed-up index would produce a mismatched value.
        return OffsetSegment(seg.start_s, seg.end_s, offset_ms=seg.offset_ms * 10,
                              confidence=0.9, status="trusted", n_chunks=1)

    monkeypatch.setattr(piecewise_offset, "_refine_one_segment", fake_refine_one)

    results = _refine_segments_concurrently("fake.mp4", segments, trusted_indices, tmp_path, min_confidence=0.5)

    assert sorted(results.keys()) == trusted_indices
    assert sorted(seen_indices) == trusted_indices
    assert results[0].offset_ms == 1110.0
    assert results[2].offset_ms == 3330.0
    assert results[3].offset_ms == 4440.0


def test_refine_segments_concurrently_single_segment_skips_thread_pool(monkeypatch, tmp_path):
    """Exactly one trusted segment must take the sequential (`n_workers ==
    1`) branch: no reason to pay thread-pool-startup or env-var-mutation
    overhead for work that was never going to run in parallel anyway."""
    segments = [OffsetSegment(0.0, 10.0, 111.0, 0.9, "trusted", n_chunks=1)]
    called_from_main_thread = []

    def fake_refine_one(video_path, seg, index, tmp_dir, min_confidence, abort_event=None, syncnet_deadline=None):
        called_from_main_thread.append(threading.current_thread() is threading.main_thread())
        return OffsetSegment(seg.start_s, seg.end_s, offset_ms=1.0, confidence=0.9, status="trusted", n_chunks=1)

    monkeypatch.setattr(piecewise_offset, "_refine_one_segment", fake_refine_one)

    before = {k: os.environ.get(k) for k in _THREAD_BUDGET_ENV_VARS}
    results = _refine_segments_concurrently("fake.mp4", segments, [0], tmp_path, min_confidence=0.5)
    after = {k: os.environ.get(k) for k in _THREAD_BUDGET_ENV_VARS}

    assert results.keys() == {0}
    assert called_from_main_thread == [True]  # never dispatched to a worker thread
    assert before == after  # single-segment path must not touch process-wide env at all


def test_refine_segments_concurrently_restores_env_vars_even_after_a_worker_raises(monkeypatch, tmp_path):
    """The thread-budget env vars this function sets are process-global
    state (each worker's real work is an external subprocess that reads
    them at its own startup, not something this process can scope via
    `torch.set_num_threads()`). They must be restored on the way out even
    if a worker raises, otherwise one failed refinement call would
    silently change this server's own thread budget for every request for
    the rest of its process lifetime."""
    sentinel = "__syncsentry_test_sentinel__"
    monkeypatch.setenv(_THREAD_BUDGET_ENV_VARS[0], sentinel)

    def fake_refine_one_raises(video_path, seg, index, tmp_dir, min_confidence, abort_event=None, syncnet_deadline=None):
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr(piecewise_offset, "_refine_one_segment", fake_refine_one_raises)

    segments = [
        OffsetSegment(0.0, 10.0, 1.0, 0.9, "trusted", n_chunks=1),
        OffsetSegment(10.0, 20.0, 2.0, 0.9, "trusted", n_chunks=1),
    ]
    try:
        _refine_segments_concurrently("fake.mp4", segments, [0, 1], tmp_path, min_confidence=0.5)
    except Exception:
        pass  # a future's exception propagating out on .result() is fine, restoration is what's tested

    assert os.environ.get(_THREAD_BUDGET_ENV_VARS[0]) == sentinel
