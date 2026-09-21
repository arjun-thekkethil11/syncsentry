"""Piecewise (per-time-segment) A/V offset detection.

## Why this exists

Every other detector in this package (`syncnet_offset.py`,
`wide_range_offset.py`, `mtdvocalist_offset.py`, the coarse
`detectors/coarse_xcorr.py`) assumes an asset has one true A/V offset,
constant for its whole duration. That covers the common case: one
recording, one mux/encode pass, one sync error if any.

It breaks, by construction, for content assembled from multiple sources
with independent sync errors: a video edited together from several
differently-recorded segments (multi-camera, stitched interview clips,
inserted B-roll with its own audio mux), each with its own offset. Applied
to such content, the single-offset pipeline reports whatever offset its
one analyzed window happens to contain, then "fixes" only the region that
was sampled while making every other region's error worse. That is a
confidently wrong outcome, not just an imprecise one.

## What this module does instead

Rather than making the single-offset detectors themselves aware of this
(they would need a different aggregation strategy, and changing their
default behavior risks regressing the validated single-offset case), this
module is a separate, cheap, whole-clip pre-check:

  1. Find real dialogue scenes across the entire clip (face on screen AND
     speech active, `dialogue_scenes.detect_dialogue_scenes`): already
     cheap (2fps YuNet + Silero VAD, no CNN embedding model).
  2. Group those scenes into duration-bounded evaluation chunks: enough
     signal per chunk for one locally reliable offset reading, capped so a
     long single-source clip still yields several independently
     corroborating chunks instead of collapsing into one big window.
  3. Estimate each chunk's own local offset via mouth-motion-vs-audio
     cross-correlation (`mouth_offset`'s existing estimator, restricted to
     that chunk's own scenes): classical signal processing, not a second
     CNN pass. Both the face/mouth-crop extraction and the audio envelope
     are computed once for the whole clip, then sliced and re-correlated
     per chunk.
  4. Merge time-adjacent, confident chunks whose offsets agree into
     segments. A clip whose chunks all agree collapses back into exactly
     one segment, matching (and deliberately deferring to) the existing
     single-offset pipeline: this module only recommends switching to
     piecewise handling when the evidence doesn't fit one number, never
     forces it.

## Why this module's own classical estimate is not the final answer

Classical envelope cross-correlation is less reliable on real speech than
SyncNet's learned embeddings, which is why SyncNet exists at all. On a
two-region composite with true offsets +600ms/-800ms, even chunks that
cleared every gate above (>=20s coverage, passed the periodicity check)
landed at +880ms and +280ms, one of them wrong in direction relative to a
naive reading. Adding a periodicity check and a duration floor narrows
that gap but does not close it.

So this module's classical pass is used for exactly one thing: localizing
candidate regions that disagree (roughly where and how many, not exactly
what by). `refine_piecewise_segments_with_syncnet` below takes those
candidate regions and re-answers each one with SyncNet's own
corroboration-gated estimator (the same `n_confident_windows >= 3` bar
`pipeline.py` already requires for the single-global-offset case), scoped
to just that region, so the existing single-region SyncNet path's own
accuracy is what actually gets applied, not a new, less-trustworthy
number. A region SyncNet can't corroborate is reported "undetermined"
(left unmodified) rather than falling back to this module's own weaker
reading: an honest abstention, not a second wrong number.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from syncsentry.detectors.coarse_xcorr import _correlation_curve, _normalize, estimate_offset_from_signals
from syncsentry.lipsync.dialogue_scenes import DialogueScene, detect_dialogue_scenes
from syncsentry.lipsync.mouth_offset import _resample_scenes, extract_mouth_motion_signal
from syncsentry.lipsync.wide_range_offset import (
    DEFAULT_MIN_FACE_COVERAGE_S,
    _dominant_rhythm_period_ms,
    _rhythm_aliased_ambiguity,
)
from syncsentry.util.ffmpeg_io import _require_binary, extract_audio_envelope, probe_duration_s

Interval = tuple[float, float]

# Sample rate for per-chunk resampling and correlation. Deliberately lower
# than `coarse_xcorr.COMMON_RATE_HZ` (200Hz): the mouth-motion signal is
# itself only sampled at 15fps (`mouth_offset.extract_mouth_motion_signal`'s
# default), so a higher correlation rate would just interpolate the same
# information more finely without adding real resolution, while costing
# more per chunk (this runs once per chunk, not once per clip).
RATE_HZ = 50.0

# Accumulate consecutive dialogue scenes into an evaluation chunk until it
# has at least this much combined face+speech coverage. Reuses
# `wide_range_offset.DEFAULT_MIN_FACE_COVERAGE_S` rather than a smaller
# value of its own: a shorter floor produces confidently wrong per-chunk
# readings, since this is the same underdetermined-search failure mode
# that floor already exists to prevent for the whole-clip wide-range seed
# (see that module's docstring). This module's per-chunk correlation uses
# the same estimator (`mouth_offset`'s cross-correlation), just scoped to
# a shorter span, and the same reliability-scales-with-signal-duration
# argument applies unchanged.
MIN_CHUNK_DURATION_S = DEFAULT_MIN_FACE_COVERAGE_S

# Cap accumulation too, for the opposite reason
# `wide_range_offset.MAX_SEARCH_WINDOW_FRACTION` exists: without a cap, a
# single long, continuously-talking, single-source clip would accumulate
# into one giant chunk spanning the whole clip. That would not be wrong,
# exactly, but it would defeat the point of this module (several
# independent, time-localized readings to detect disagreement between
# regions). Kept a bit above 2x the floor so a long single-source region
# still yields at least 2 independently corroborating chunks (see
# `MIN_SEGMENT_SUPPORT_CHUNKS`) without fragmenting a real ~40-60s segment
# into oddly sized pieces.
MAX_CHUNK_DURATION_S = DEFAULT_MIN_FACE_COVERAGE_S * 2.0

# Minimum normalized cross-correlation confidence (same [0,1]-ish scale as
# `coarse_xcorr.OffsetEstimate.confidence`) for a chunk's reading to be
# used at all. Chunks below this are dropped rather than forced into a
# segment, consistent with this project's abstain-rather-than-guess
# default elsewhere (e.g. `pipeline.DEFAULT_AV_MIN_CONFIDENCE`), and
# deliberately more permissive than that constant's 0.3: this signal
# (mouth motion, face+speech-restricted) is already more targeted than
# the whole-frame-brightness signal that constant gates.
MIN_CHUNK_CONFIDENCE = 0.15

# Two chunks are "the same offset" for merging purposes if they're within
# this many ms of each other. Generous relative to this estimator's own
# noise: tight enough to tell genuinely different segments (hundreds of
# ms to seconds apart, the common case for independently muxed sources)
# apart, loose enough not to fragment one true segment into several over
# this estimator's own noise.
CHANGE_TOLERANCE_MS = 200.0

# A merged segment shorter than this isn't enough independent evidence to
# act on as its own region, matching this module's general requirement of
# real corroboration (see `MIN_SEGMENT_SUPPORT_CHUNKS`) before splitting
# the timeline at all.
MIN_SEGMENT_DURATION_S = 6.0

# Even above that duration floor, require at least this many chunks per
# segment. 1, not >= 2: each chunk here is held to the same coverage floor
# and periodicity self-check as
# `wide_range_offset.estimate_wide_range_offset`'s own single reading (see
# `MIN_CHUNK_DURATION_S` above and `_chunk_offset` below), and that module
# already trusts one reading at that bar, with corroboration happening
# downstream (SyncNet's fine search) rather than requiring a second
# independent wide-range reading first. Requiring 2 here would mean no
# segment shorter than 2x the floor could ever be trusted, which is
# stricter than the bar already used for the same estimator elsewhere.
MIN_SEGMENT_SUPPORT_CHUNKS = 1

# At least this many trusted, mutually-disagreeing segments are required
# before concluding a clip is genuinely piecewise rather than well
# described by one global offset: one segment, however confident, is just
# "the global offset", not evidence of a piecewise structure.
MIN_SEGMENTS_FOR_PIECEWISE = 2

# --- Fine-grained fallback tier ---
#
# Everything above localizes candidate regions via `detect_dialogue_scenes`
# (face on screen AND speech active, merged across gaps <= 0.5s, see
# `dialogue_scenes.py`). That works well when true splice points between
# independently offset sources land in an actual break in face visibility
# or speech (a cut to B-roll, a pause, a different camera angle). It does
# not work when clips are spliced back-to-back with no such break:
# `detect_dialogue_scenes`'s own gap-tolerant merging then reports one
# continuous "scene" that silently spans the true splice point, and any
# classical estimate computed on that scene is an uninterpretable blend of
# both true offsets on either side of it. Retuning the duration floor
# above does not fix this, since the contamination happens one layer
# downstream, in scene detection itself, before this module's own
# grouping ever runs.
#
# This tier sidesteps that by not using dialogue-scene boundaries for
# localization at all: a fixed-width, 50%-overlapping window slid across
# the entire clip's continuous mouth-motion/audio signal (independent of
# any face/speech-gap merging) will, for any true splice point, still have
# at least one window landing entirely on one side of it or the other
# (window width < half the shortest real segment this tier is meant to
# resolve). Those clean windows carry a normal, trustworthy reading; only
# the windows straddling the splice itself produce a blended, lower-
# confidence one, which the same confidence/periodicity gates used above
# filter out on their own.
#
# Only tried as a fallback, after the scene-gated tier above finds no
# piecewise evidence: the scene-gated tier's localization is more precise
# (real scene boundaries, not a fixed grid) whenever it applies, and this
# tier's fixed grid means genuinely single-source content with a brief
# real disagreement (e.g. one loud non-speech sound event) has a
# structurally higher chance of tripping the confidence gates than the
# scene-gated tier's coarser windows. That is not a reason to distrust
# this tier when it does report piecewise structure, but a reason to
# prefer the other tier's answer when both would otherwise apply.
FINE_WINDOW_S = 6.0
FINE_HOP_S = 3.0

# Skip this tier entirely beyond this duration: it scans the whole clip's
# mouth-motion signal unconditionally (no face-presence pre-filter like
# `wide_range_offset.py`'s tier, since dialogue-scene detection already
# failed to localize anything useful for this clip), at a fixed cost
# regardless of outcome. Bounding it here keeps a long asset that reaches
# this fallback from paying that cost with no localization tier having
# found anything to show for it.
FINE_MAX_VIDEO_DURATION_S = 300.0


@dataclass
class OffsetChunk:
    start_s: float
    end_s: float
    offset_ms: float
    confidence: float


@dataclass
class OffsetSegment:
    start_s: float
    end_s: float
    offset_ms: float | None  # None for "undetermined" spans
    confidence: float | None
    status: str  # "trusted" | "undetermined"
    n_chunks: int = 0
    member_offsets_ms: list[float] = field(default_factory=list)


@dataclass
class PiecewiseResult:
    is_piecewise: bool
    segments: list[OffsetSegment]  # time-ordered, contiguous, covers [0, video_duration_s]
    video_duration_s: float
    n_chunks_evaluated: int
    n_chunks_confident: int


def _group_scenes_into_chunks(scenes: list[DialogueScene], min_duration_s: float,
                               max_duration_s: float) -> list[list[DialogueScene]]:
    """Greedily accumulates consecutive (by start time) dialogue scenes
    into chunks with >= `min_duration_s` of combined face+speech coverage
    each, never letting one chunk's coverage exceed `max_duration_s`. See
    module-level comments above `MIN_CHUNK_DURATION_S`/
    `MAX_CHUNK_DURATION_S` for why both bounds matter.
    """
    chunks: list[list[DialogueScene]] = []
    current: list[DialogueScene] = []
    current_coverage = 0.0
    for scene in sorted(scenes, key=lambda s: s.start_s):
        dur = scene.end_s - scene.start_s
        if current and current_coverage + dur > max_duration_s:
            chunks.append(current)
            current, current_coverage = [], 0.0
        current.append(scene)
        current_coverage += dur
        if current_coverage >= min_duration_s:
            chunks.append(current)
            current, current_coverage = [], 0.0
    if current:
        # Leftover shorter than the floor: merge into the previous chunk
        # (if any) rather than standing alone as under-supported evidence.
        if chunks and current_coverage < min_duration_s / 2.0:
            chunks[-1] = chunks[-1] + current
        else:
            chunks.append(current)
    return chunks


def _chunk_offset(mouth_sig, audio_sig, scenes: list[DialogueScene]) -> OffsetChunk | None:
    """One chunk's local offset reading, from signals already extracted
    once for the whole clip (see `detect_piecewise_offsets`) and merely
    sliced/resampled here: the expensive part (decoding, face/mouth
    detection) is not repeated per chunk.

    Applies the same periodicity self-check as
    `wide_range_offset.estimate_wide_range_offset`: real speech's
    quasi-periodic word/pause rhythm can alias into a comparably tall
    secondary correlation peak a rhythm-period away from the true lag (see
    that module's docstring). Without this check, a chunk that otherwise
    clears every other bar (`MIN_CHUNK_DURATION_S`, `MIN_CHUNK_CONFIDENCE`)
    can still return a confidently wrong reading.
    """
    intervals: list[Interval] = [(s.start_s, s.end_s) for s in scenes]
    start_s = min(s.start_s for s in scenes)
    end_s = max(s.end_s for s in scenes)
    coverage_s = sum(e - s for s, e in intervals)

    v = _resample_scenes(mouth_sig, intervals, RATE_HZ)
    a = _resample_scenes(audio_sig, intervals, RATE_HZ)
    n = min(len(a), len(v))
    if n < 8:
        return None
    a, v = _normalize(a[:n]), _normalize(v[:n])

    # Wide enough to catch a genuinely different segment's offset (real
    # multi-source content routinely differs by seconds, not just a few
    # hundred ms), bounded by how much signal this one chunk actually has
    # (same underdetermined-search argument as
    # `wide_range_offset.MAX_SEARCH_WINDOW_FRACTION`).
    search_window_ms = min(4000.0, coverage_s * 1000.0 * 0.4)
    if search_window_ms < 50.0:
        return None
    try:
        est = estimate_offset_from_signals(a, v, RATE_HZ, search_window_ms=search_window_ms)
        lags_ms, normalized_corr, _denom = _correlation_curve(a, v, RATE_HZ, search_window_ms)
    except ValueError:
        return None

    rhythm_period_ms = _dominant_rhythm_period_ms(a, RATE_HZ)
    if rhythm_period_ms is not None:
        secondary_ratio = _rhythm_aliased_ambiguity(lags_ms, normalized_corr, est.offset_ms, rhythm_period_ms)
        if secondary_ratio is not None:
            return None  # ambiguous: same treatment as "not enough signal" above

    return OffsetChunk(start_s=start_s, end_s=end_s, offset_ms=est.offset_ms, confidence=est.confidence)


def _merge_confident_chunks(chunks: list[OffsetChunk]) -> list[OffsetSegment]:
    """Merges time-ordered confident chunks into segments: extends the
    current segment while the next chunk's offset agrees (within
    `CHANGE_TOLERANCE_MS`), starts a new one when it doesn't."""
    segments: list[OffsetSegment] = []
    for c in sorted(chunks, key=lambda c: c.start_s):
        if segments and abs(c.offset_ms - segments[-1].offset_ms) <= CHANGE_TOLERANCE_MS:
            seg = segments[-1]
            seg.end_s = c.end_s
            seg.n_chunks += 1
            seg.member_offsets_ms.append(c.offset_ms)
            seg.offset_ms = float(sorted(seg.member_offsets_ms)[len(seg.member_offsets_ms) // 2])  # running median
            seg.confidence = min(seg.confidence, c.confidence)  # weakest link, not average: one lucky chunk shouldn't mask a shaky segment
        else:
            segments.append(OffsetSegment(
                start_s=c.start_s, end_s=c.end_s, offset_ms=c.offset_ms, confidence=c.confidence,
                status="trusted", n_chunks=1, member_offsets_ms=[c.offset_ms],
            ))
    return segments


def _dedupe_overlapping_segments(segments: list[OffsetSegment]) -> list[OffsetSegment]:
    """Clips away time overlap between consecutive (by start time) trusted
    segments, splitting any overlap at its midpoint.

    Only needed for the fine-grained tier: its 50%-overlapping sliding
    windows (`FINE_HOP_S` < `FINE_WINDOW_S`) mean two adjacent windows can
    disagree enough to become two different `_merge_confident_chunks`
    segments whose raw `[start_s, end_s)` spans still overlap by up to
    `FINE_WINDOW_S - FINE_HOP_S`. The other caller of this module
    (scene-gated chunks, whose source scenes are already disjoint) never
    produces overlapping segments, so this is a no-op for it.
    `detect_piecewise_offsets`'s contract (segments are "time-ordered,
    contiguous", see `PiecewiseResult` docstring) and
    `fixer/piecewise_fix.py`'s filter-graph splicing both assume no
    overlap; leaving one in would double-apply a correction to the
    overlapping span. Only needs to consider immediate neighbors: with
    `FINE_HOP_S == FINE_WINDOW_S / 2`, window `i` can only overlap windows
    `i-1` and `i+1`, never `i-2`/`i+2`.
    """
    ordered = sorted(segments, key=lambda s: s.start_s)
    for i in range(len(ordered) - 1):
        cur, nxt = ordered[i], ordered[i + 1]
        if cur.end_s > nxt.start_s:
            midpoint = (cur.end_s + nxt.start_s) / 2.0
            cur.end_s = midpoint
            nxt.start_s = midpoint
    return [s for s in ordered if s.end_s > s.start_s]


def _fill_gaps(trusted: list[OffsetSegment], video_duration_s: float) -> list[OffsetSegment]:
    """Inserts "undetermined" segments to cover every span of
    `[0, video_duration_s]` not already covered by a trusted segment: the
    result always covers the whole timeline contiguously, so a caller
    applying corrections never has to reason about gaps separately."""
    result: list[OffsetSegment] = []
    cursor = 0.0
    eps = 0.05
    for seg in sorted(trusted, key=lambda s: s.start_s):
        if seg.start_s > cursor + eps:
            result.append(OffsetSegment(cursor, seg.start_s, None, None, "undetermined"))
        result.append(seg)
        cursor = max(cursor, seg.end_s)
    if cursor < video_duration_s - eps:
        result.append(OffsetSegment(cursor, video_duration_s, None, None, "undetermined"))
    return result


def _detect_piecewise_offsets_scene_based(video_path: str, video_duration_s: float) -> PiecewiseResult:
    """Dialogue-scene-gated localization tier. See module docstring.

    Raises `ValueError` if there isn't enough dialogue-scene signal across
    the clip to attempt this at all (e.g. no face ever on screen); callers
    should fall through to the fine-grained tier (or ultimately the
    single-offset pipeline), exactly as they already do for the other
    detectors' own `ValueError`s.
    """
    scenes = detect_dialogue_scenes(video_path)
    if len(scenes) < 2:
        raise ValueError(
            "Not enough dialogue-scene (face+speech) signal across the clip to attempt "
            "piecewise segmentation."
        )

    # Each of these decodes/scans the *whole* clip exactly once; every
    # chunk below only slices and re-correlates already-extracted arrays.
    all_intervals: list[Interval] = [(s.start_s, s.end_s) for s in scenes]
    mouth_sig = extract_mouth_motion_signal(video_path, scenes=all_intervals)
    audio_sig = extract_audio_envelope(video_path)

    chunk_groups = _group_scenes_into_chunks(scenes, MIN_CHUNK_DURATION_S, MAX_CHUNK_DURATION_S)
    chunks = [c for g in chunk_groups if (c := _chunk_offset(mouth_sig, audio_sig, g)) is not None]
    confident_chunks = [c for c in chunks if c.confidence >= MIN_CHUNK_CONFIDENCE]

    merged = _merge_confident_chunks(confident_chunks)
    trusted = [s for s in merged
               if (s.end_s - s.start_s) >= MIN_SEGMENT_DURATION_S
               and s.n_chunks >= MIN_SEGMENT_SUPPORT_CHUNKS]

    is_piecewise = len(trusted) >= MIN_SEGMENTS_FOR_PIECEWISE

    full_segments = _fill_gaps(trusted, video_duration_s)

    return PiecewiseResult(
        is_piecewise=is_piecewise, segments=full_segments, video_duration_s=video_duration_s,
        n_chunks_evaluated=len(chunks), n_chunks_confident=len(confident_chunks),
    )


def _fine_grained_chunk(mouth_sig, audio_sig, start_s: float, end_s: float) -> OffsetChunk | None:
    """One fixed-grid window's local offset reading. Structurally identical
    to `_chunk_offset` above (same resample -> normalize -> correlate ->
    periodicity-self-check pipeline); the only difference is that the
    caller slices by a fixed `[start_s, end_s)` time range instead of a set
    of dialogue-scene intervals, since this tier exists specifically for
    clips where dialogue-scene intervals can't be trusted as boundaries
    (see `FINE_WINDOW_S` module comment)."""
    v = _resample_scenes(mouth_sig, [(start_s, end_s)], RATE_HZ)
    a = _resample_scenes(audio_sig, [(start_s, end_s)], RATE_HZ)
    n = min(len(a), len(v))
    if n < 8:
        return None
    a, v = _normalize(a[:n]), _normalize(v[:n])

    search_window_ms = min(2000.0, (end_s - start_s) * 1000.0 * 0.4)
    if search_window_ms < 50.0:
        return None
    try:
        est = estimate_offset_from_signals(a, v, RATE_HZ, search_window_ms=search_window_ms)
        lags_ms, normalized_corr, _denom = _correlation_curve(a, v, RATE_HZ, search_window_ms)
    except ValueError:
        return None

    rhythm_period_ms = _dominant_rhythm_period_ms(a, RATE_HZ)
    if rhythm_period_ms is not None:
        secondary_ratio = _rhythm_aliased_ambiguity(lags_ms, normalized_corr, est.offset_ms, rhythm_period_ms)
        if secondary_ratio is not None:
            return None

    return OffsetChunk(start_s=start_s, end_s=end_s, offset_ms=est.offset_ms, confidence=est.confidence)


def _detect_piecewise_offsets_fine_grained(video_path: str, video_duration_s: float) -> PiecewiseResult:
    """Fixed-grid fallback localization tier. See `FINE_WINDOW_S` module
    comment for why this exists and when it's tried (only after the
    dialogue-scene-gated tier finds no piecewise evidence)."""
    if video_duration_s > FINE_MAX_VIDEO_DURATION_S:
        raise ValueError(
            f"Clip too long ({video_duration_s:.0f}s) for the fine-grained fallback tier "
            f"(cap {FINE_MAX_VIDEO_DURATION_S:.0f}s)."
        )

    # Whole-clip signals, extracted once regardless of how many windows
    # slice them below: no dialogue-scene restriction this time (that
    # restriction is exactly what this tier exists to route around).
    mouth_sig = extract_mouth_motion_signal(video_path)
    audio_sig = extract_audio_envelope(video_path)

    windows: list[tuple[float, float]] = []
    t = 0.0
    while t + FINE_WINDOW_S <= video_duration_s:
        windows.append((t, t + FINE_WINDOW_S))
        t += FINE_HOP_S
    if not windows and video_duration_s > 0:
        windows.append((0.0, video_duration_s))

    chunks = [c for w in windows if (c := _fine_grained_chunk(mouth_sig, audio_sig, *w)) is not None]
    confident_chunks = [c for c in chunks if c.confidence >= MIN_CHUNK_CONFIDENCE]

    merged = _merge_confident_chunks(confident_chunks)
    merged = _dedupe_overlapping_segments(merged)
    trusted = [s for s in merged
               if (s.end_s - s.start_s) >= MIN_SEGMENT_DURATION_S
               and s.n_chunks >= MIN_SEGMENT_SUPPORT_CHUNKS]

    is_piecewise = len(trusted) >= MIN_SEGMENTS_FOR_PIECEWISE

    full_segments = _fill_gaps(trusted, video_duration_s)

    return PiecewiseResult(
        is_piecewise=is_piecewise, segments=full_segments, video_duration_s=video_duration_s,
        n_chunks_evaluated=len(chunks), n_chunks_confident=len(confident_chunks),
    )


def detect_piecewise_offsets(video_path: str) -> PiecewiseResult:
    """Whole-clip piecewise offset pre-check. See module docstring.

    Two localization tiers, tried in order (see `FINE_WINDOW_S` module
    comment for why the second one exists and why it's only a fallback):

    1. Dialogue-scene-gated (`_detect_piecewise_offsets_scene_based`):
       preferred whenever it finds *any* dialogue-scene signal at all,
       since real scene boundaries are more precise than a fixed grid.
    2. Fixed-grid fallback (`_detect_piecewise_offsets_fine_grained`):
       tried only when tier 1 ran but didn't conclude the clip is
       piecewise (or had too little dialogue-scene signal to try at all).

    Raises `ValueError` only if neither tier could even attempt a verdict
    (e.g. no face ever on screen, or the clip both lacks dialogue-scene
    signal and is too long for the fallback's own cap): callers should
    fall through to the existing single-offset pipeline in that case,
    exactly as they already do for the other detectors' own `ValueError`s.
    """
    video_duration_s = probe_duration_s(video_path)

    scene_result: PiecewiseResult | None = None
    scene_error: ValueError | None = None
    try:
        scene_result = _detect_piecewise_offsets_scene_based(video_path, video_duration_s)
        if scene_result.is_piecewise:
            return scene_result
    except ValueError as exc:
        scene_error = exc

    try:
        fine_result = _detect_piecewise_offsets_fine_grained(video_path, video_duration_s)
    except ValueError as exc:
        if scene_result is not None:
            return scene_result  # tier 1 ran and gave a (non-piecewise) verdict; trust it
        raise exc from scene_error

    if fine_result.is_piecewise or scene_result is None:
        return fine_result
    return scene_result


def refine_piecewise_segments_with_syncnet(video_path: str, segments: list[OffsetSegment],
                                            min_confidence: float | None = None) -> list[OffsetSegment]:
    """Re-answers each "trusted" segment from `detect_piecewise_offsets`
    with SyncNet's own estimator, scoped to just that segment (via a
    physical, frame-accurate sub-clip, since SyncNet has no native
    start/end-time parameter). See module docstring, "Why this module's
    own classical estimate is not the final answer", for why this step
    exists rather than trusting the classical reading directly.

    Trust bar: `confidence >= min_confidence`, matching
    `pipeline._detect_av_offset`'s own bar for accepting a nonzero global
    offset, not the stricter `n_confident_windows >= 3` corroboration bar
    that module reserves for confirming "in sync". Windowed corroboration
    was tried here too but rejected: re-encoding a segment into its own
    sub-clip costs windowed agreement even at high encode quality (a real
    case went from 4 confident windows pre-cut to 0 post-cut, with the
    whole-track aggregate answer barely moving), which would make this
    step strictly harder to pass than the path it's meant to replace. The
    safety net for a nonzero reading is downstream, same as elsewhere in
    this project: `pipeline.run_fix_pipeline` re-verifies every applied
    correction against the corrected output afterward.

    "Undetermined" segments pass through unchanged. A "trusted" segment
    whose SyncNet re-check doesn't clear `min_confidence` (no trackable
    face, or a genuinely low-confidence reading) is downgraded to
    "undetermined" rather than left at its original, potentially wrong,
    classical reading.

    Costs one SyncNet face-detection/tracking pass per trusted segment
    (each internally bounded to a centered <=20s window by
    `syncnet_offset.DEFAULT_MAX_ANALYZE_DURATION_S`, regardless of the
    segment's own length). These per-segment passes are independent
    (different sub-clip, different temp files, no shared mutable state)
    and run concurrently rather than sequentially (see
    `_refine_segments_concurrently` below). This adds real latency versus
    skipping the step entirely, but it is only paid when the cheap
    classical pre-check above already found evidence of a piecewise
    structure worth spending it on, not for every clip.
    """
    min_confidence_resolved = min_confidence
    trusted_indices = [i for i, seg in enumerate(segments) if seg.status == "trusted"]
    if not trusted_indices:
        return list(segments)

    with tempfile.TemporaryDirectory() as td:
        results = _refine_segments_concurrently(
            video_path, segments, trusted_indices, Path(td), min_confidence_resolved,
        )

    refined: list[OffsetSegment] = []
    for i, seg in enumerate(segments):
        refined.append(results[i] if i in results else seg)
    return refined


# Bounds how many per-segment SyncNet refinement calls run at once (see
# `_refine_segments_concurrently`). Not simply `os.cpu_count()`: each
# individual call's dominant cost (S3FD face detection, an external
# `run_pipeline.py` subprocess) benefits from a few threads of its own, so
# uncapped parallelism would divide this machine's cores so thin per call
# that per-call latency balloons even as throughput improves. 4 is a
# modest starting point; most clips have only a handful of trusted
# candidate segments regardless, so this rarely becomes the binding
# constraint in practice.
#
# Overridable via SYNCSENTRY_MAX_REFINE_WORKERS: each concurrent worker
# eventually caches its own SyncNet model in memory (see
# `syncnet_offset.estimate_syncnet_tracks`'s thread-local cache), so on a
# host with little RAM to spare, 4 workers can mean 4x the model memory
# resident at once. Set to 1 there to keep peak memory to a single
# cached model, at the cost of refining segments one at a time.
MAX_REFINE_WORKERS = int(os.environ.get("SYNCSENTRY_MAX_REFINE_WORKERS", "4"))

# Environment variables read by the BLAS/OpenMP libraries underneath
# torch's CPU backend (checked at each subprocess's own startup, not
# settable via `torch.set_num_threads()` from this parent process, since
# each refinement call's face-detection stage is its own separate OS
# process, not a thread in this one; see
# `syncnet_offset._run_face_track_crop`). Divided across concurrent
# workers below so N simultaneous `run_pipeline.py` subprocesses share
# this machine's cores instead of each independently trying to claim all
# of them: uncapped, concurrent S3FD subprocesses thrash each other for
# cache/cores badly enough to erase most of the concurrency win.
_THREAD_BUDGET_ENV_VARS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def _refine_one_segment(video_path: str, seg: OffsetSegment, index: int, tmp_dir: Path,
                         min_confidence: float) -> OffsetSegment:
    """The per-segment work `refine_piecewise_segments_with_syncnet` farms
    out to a worker: cut this segment's own sub-clip, run SyncNet on it,
    and translate the result into an `OffsetSegment`. Factored out as its
    own function so `_refine_segments_concurrently` can dispatch it.

    Never raises: every failure mode (ffmpeg cut failed, SyncNet
    unavailable, no trackable face, low confidence) becomes an
    "undetermined" segment.
    """
    from syncsentry.lipsync.syncnet_offset import SyncNetUnavailable, estimate_syncnet_offset

    ffmpeg = _require_binary("ffmpeg")
    sub_clip = str(tmp_dir / f"segment_{index}.mp4")
    # `-preset veryfast -crf 18` (not `ultrafast`/`crf 20`, unlike
    # `syncnet_offset._maybe_trim_to_window`'s detection-only window): the
    # more aggressive preset degrades quality enough to cost SyncNet's
    # windowed corroboration specifically, even when the whole-track
    # aggregate answer barely changes. Since this step exists specifically
    # to demand that corroboration (see module docstring), the extra
    # encode cost here is worth paying for.
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
           "-ss", f"{seg.start_s:.3f}", "-t", f"{seg.end_s - seg.start_s:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", sub_clip]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return OffsetSegment(seg.start_s, seg.end_s, None, None, "undetermined")

    try:
        est = estimate_syncnet_offset(sub_clip)
    except (SyncNetUnavailable, ValueError):
        return OffsetSegment(seg.start_s, seg.end_s, None, None, "undetermined")

    if est.confidence >= min_confidence:
        return OffsetSegment(
            start_s=seg.start_s, end_s=seg.end_s, offset_ms=est.offset_ms,
            confidence=est.confidence, status="trusted", n_chunks=seg.n_chunks,
        )
    return OffsetSegment(seg.start_s, seg.end_s, None, None, "undetermined")


def _refine_segments_concurrently(video_path: str, segments: list[OffsetSegment],
                                   trusted_indices: list[int], tmp_dir: Path,
                                   min_confidence: float | None) -> dict[int, OffsetSegment]:
    """Runs `_refine_one_segment` for every trusted-segment index, in
    parallel rather than one at a time.

    Safe: each call operates on its own sub-clip file (`segment_{i}.mp4`,
    distinct per index) and its own `SyncNetInstance`/temp working
    directory inside SyncNet's own pipeline
    (`syncnet_offset._run_face_track_crop` gives each call a fresh
    `tempfile.TemporaryDirectory`), so there is no shared mutable state
    between calls for threads to race on. Completion order doesn't matter
    either, since results are assembled into a dict keyed by original
    segment index.

    Effective: most of one refinement call's wall time is spent in
    `run_pipeline.py`, an external subprocess for face detection/tracking.
    `subprocess.run()` blocks the calling Python thread on I/O and
    releases the GIL for the whole duration, so a `ThreadPoolExecutor`
    (rather than a process pool, since the real parallel work already
    happens in separate OS processes either way) genuinely runs those
    child processes concurrently on this machine's other cores.
    """
    from syncsentry.lipsync.syncnet_offset import DEFAULT_SYNCNET_MIN_CONFIDENCE

    min_confidence = min_confidence if min_confidence is not None else DEFAULT_SYNCNET_MIN_CONFIDENCE
    n_workers = max(1, min(len(trusted_indices), MAX_REFINE_WORKERS))

    if n_workers == 1:
        return {i: _refine_one_segment(video_path, segments[i], i, tmp_dir, min_confidence)
                for i in trusted_indices}

    threads_per_worker = max(1, (os.cpu_count() or n_workers) // n_workers)
    saved_env = {k: os.environ.get(k) for k in _THREAD_BUDGET_ENV_VARS}
    for k in _THREAD_BUDGET_ENV_VARS:
        os.environ[k] = str(threads_per_worker)
    try:
        results: dict[int, OffsetSegment] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_refine_one_segment, video_path, segments[i], i, tmp_dir, min_confidence): i
                for i in trusted_indices
            }
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
        return results
    finally:
        # Restore rather than delete-if-was-unset-vs-restore-if-was-set:
        # this function's caller may itself be running inside a larger
        # process (the API server) that shouldn't have its own thread
        # budget silently changed for the rest of its life.
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
