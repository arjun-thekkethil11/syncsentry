"""General wide-range A/V offset seed for `_maybe_recenter_for_large_offset`.

## Why this exists

SyncNet's native search window is only +-600ms (see
`syncnet_offset.DEFAULT_RECENTER_THRESHOLD_MS`), and the existing coarse
cross-correlation detector (whole-frame brightness vs. audio) is
unreliable at recovering large offsets on real talking-head content: on
real interview/podcast clips with true offsets in the 600-1500ms range,
its own audio-vs-video confidence stays mostly below its own confidence
gate, and where it does clear the gate it can be confidently wrong. This
module provides a better wide-range seed for those cases.

## Why mouth-motion, not whole-frame brightness, for the wide-range seed

Whole-frame brightness carries almost no speech-related signal for a
talking-head shot: the frame is dominated by a mostly-static face and
background, and the only part that actually moves in sync with speech is
the mouth, a small fraction of total pixels (see `mouth_offset.py`'s own
docstring). That module already fixes this for the narrow-range case;
this function reuses its estimator (`estimate_mouth_sync_offset`) but
widens the search window from its 500ms default to multiple seconds. A
whole-clip, unfiltered multi-second-window mouth-motion estimate is a
large, real improvement over whole-frame brightness on real talking-head
content, with no model training and no asset-specific tuning.

## Rejected alternative: splitting into several short chunks for corroboration

The existing brightness-based pre-pass requires two independent halves of
the clip to agree before trusting a wide-range estimate, and that same
principle was tried here too: both many small chunks (~8s each, clustering
agreeing chunks) and a simple 2-way half-split. Both made results worse,
not better. Real speech has quasi-periodic structure (word/pause rhythm)
that aliases inside a multi-second search window when there isn't much
signal duration to disambiguate the true lag from a spurious one a few
hundred ms to a couple of seconds away; cutting the already-modest signal
into smaller pieces makes this worse, because cross-correlation
reliability scales with available signal duration. Splitting for
corroboration is the right principle for SyncNet's own learned per-window
embeddings (`syncnet_offset._largest_agreeing_cluster`) and for the
brightness detector's coarse, low-information signal, but is
counterproductive for this already fairly informative envelope
correlation.

## A second rejected version: no minimum-duration gate at all

An earlier version of this function had no lower bound on clip/
face-coverage duration beyond a token 2-second sanity check, relying
entirely on the downstream SyncNet confidence gate to catch bad seeds.
That produced a real, dangerous regression on short single-speaker clips
(a few seconds to ~10s, with true offsets natively within SyncNet's own
+-600ms range and already working correctly): this function produced a
consistently, confidently wrong seed, and unlike the B-roll failure mode
described below, SyncNet's fine search agreed with the wrong seed, since
shifting a short clip by an arbitrary amount can still land on some
coincidentally plausible local alignment when there's little content to
disambiguate against.

Root cause: `estimate_mouth_sync_offset`'s search window was passed
through unconditionally at up to 5000ms regardless of how much signal was
actually available. A 5-second clip searched across a +-5s window is
underdetermined: there's essentially no clip left outside the search
range to rule candidates out. This is the same underlying mechanism as
the chunking failure above (cross-correlation reliability scales with
available-signal-duration relative to search-window-width), just at the
whole-clip level instead of the sub-chunk level.

Fix: `DEFAULT_MIN_FACE_COVERAGE_S` below is a duration floor, independent
of the confidence-gate question, requiring enough available signal that a
genuinely wide (multi-second) search is well-determined at all. Below
this floor, this tier is skipped entirely (falls through to the
brightness-based tier 2, or to no recentering at all, matching this
project's short-clip behavior elsewhere), rather than attempted with a
narrower window: a clip too short to support a wide search is, by
definition, also short enough that SyncNet's own native +-600ms window
already covers most of what such a short clip could plausibly need (a
multi-second true offset on a 5-8s clip would leave almost no overlapping
content to fix in the first place).

## How correctness is actually enforced instead

Given splitting doesn't work as a safety net here, this estimate is used
only as a search-center seed for SyncNet's fine, learned-embedding
search, never as a final answer on its own (see
`_maybe_recenter_for_large_offset` callers). The real safety net is
downstream: SyncNet's own windowed corroboration
(`_largest_agreeing_cluster`) and confidence gate, which fail safely
rather than confidently wrong on bad input.

Applying this function's raw seed ahead of the full fine-search pipeline
on real assets shows the expected outcomes on both sides: a correct seed
converges to a highly accurate, highly confident final answer,
independently trusted by SyncNet's own gate; a badly wrong seed (e.g. when
the on-screen face for much of the clip is unrelated narrated B-roll, not
the actual speaker) causes the fine stage to find no trackable face track
at all inside its analysis window and fail loudly with a `ValueError`,
which the existing top-level `_detect_av_offset` already catches and
falls back to the honest low-confidence coarse detector for. No
confidently-wrong result is produced by either outcome. This is why this
function deliberately carries no extra confidence gate itself beyond a
basic "was there enough signal to compute anything at all" check: adding
one trades away real accuracy for a safety property SyncNet's own
downstream gate already provides more effectively.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from syncsentry.lipsync.face_detect import detect_face_presence
from syncsentry.lipsync.mouth_offset import estimate_mouth_sync_offset_with_diagnostics
from syncsentry.util.ffmpeg_io import probe_duration_s

Interval = tuple[float, float]

DEFAULT_SEARCH_WINDOW_MS = 5000.0

# Duration floor for attempting a *wide* (multi-second) search at all. See
# module docstring, "A second rejected version", for the regression this
# fixes: without this floor, short clips can produce a consistently,
# confidently wrong seed that SyncNet's fine search then agrees with.
# Needs enough duration that the search window used below stays well
# clear of the underdetermined zone; short clips (a few seconds up to
# ~10s) fall below this and correctly skip this tier, falling back to
# already-validated short-clip behavior, while longer real-world clips
# clear it comfortably.
DEFAULT_MIN_FACE_COVERAGE_S = 20.0

# Even above the floor, the search window itself is capped to a fraction
# of available signal rather than the fixed 5000ms default
# unconditionally: belt-and-suspenders with the floor above, since the
# floor is about face coverage but a clip can have gaps.
MAX_SEARCH_WINDOW_FRACTION = 0.3

# Upper bound on how much of the *source* clip this tier's two full-clip
# scans (`_face_intervals`'s 2fps face-presence pass, then
# `extract_mouth_motion_signal`'s 15fps mouth-crop pass, both unconditional
# whenever `use_syncnet=True`) are allowed to run over. Without a cap,
# this tier's cost scales directly, and unboundedly, with total clip
# length regardless of whether a large offset even exists; the mouth-
# motion pass over face-covered scenes is the majority of that cost, not
# the face-presence pass.
#
# 120s, not the fine-search path's 20s cap
# (`syncnet_offset._maybe_trim_to_window`): this tier needs more signal
# duration than that to seed reliably (see `crop_face_tracks`'s docstring,
# "the mouth-motion wide-range tier needs as much signal duration as it
# can get"), since computing this same seed on a short centered crop
# instead of the full clip risks locking onto a materially different,
# wrong local optimum. 120s keeps a wide margin above that shorter figure
# while still bounding worst-case cost on long uploads.
DEFAULT_MAX_SCAN_DURATION_S = 120.0

# --- Periodicity self-check ---
#
# Real speech has quasi-periodic word/pause rhythm (roughly 2-5 Hz). On
# content whose audio envelope has an unusually strong periodic
# component, a metronomic speaking cadence (a trained public speaker's
# rhythm can be more regular than conversational speech) or background
# music bleeding into the envelope, the cross-correlation curve this
# module's seed is picked from can have a second, comparably tall peak
# exactly one (or a few) rhythm-periods away from the true one. Nothing in
# the single best-lag summary distinguishes a uniquely tall peak from one
# tied with a rhythm-spaced neighbor; both look like a normal confident
# result until you look at the curve's actual shape.
#
# This check does exactly that: autocorrelate the clip's own audio
# envelope to find its dominant rhythm period, if it has one strong
# enough (most natural speech doesn't), then check whether the
# cross-correlation curve has a secondary local-maximum peak near a
# multiple of that period, comparably tall to the chosen top peak. If so,
# this seed is treated as unreliable (raises, same as "not enough
# signal") rather than silently returned; the caller falls through to
# tier 2 in that case (see `_maybe_recenter_for_large_offset`).
MIN_RHYTHM_PERIOD_MS = 150.0
MAX_RHYTHM_PERIOD_MS = 2000.0
# Autocorrelation height (relative to zero-lag) needed to call a clip
# "rhythmic" at all. Most natural, non-metronomic speech stays well below
# this, so the check is a no-op for the common case.
RHYTHM_PEAK_MIN_HEIGHT = 0.15
# How tall (relative to the top peak) a rhythm-spaced secondary peak must be
# to count as "comparably tall" (i.e. genuinely ambiguous, not just a minor
# side-lobe).
AMBIGUITY_HEIGHT_RATIO = 0.85
# How close a secondary peak's lag-distance from the top peak must be to an
# exact multiple of the rhythm period to count as "rhythm-spaced" rather
# than coincidental, expressed as a fraction of the period itself.
AMBIGUITY_LAG_TOLERANCE_FRACTION = 0.2
# Peaks closer than this to the top one are treated as its own shoulder, not
# a distinct secondary peak.
MIN_PEAK_SEPARATION_MS = 80.0


@dataclass
class WideRangeEstimate:
    offset_ms: float
    confidence: float
    face_coverage_s: float
    rhythm_period_ms: float | None = None


def _face_intervals(video_path: str, sample_fps: float = 2.0, min_gap_s: float = 1.0) -> list[Interval]:
    """Merged intervals where *a* face (any face, no speaker judgment) is on
    screen. Face-presence-only: deliberately not gated on speech-VAD
    timing, which lives on the audio timeline and would be circular for a
    detector whose job is finding a potentially large audio/video
    misalignment in the first place."""
    frames = detect_face_presence(video_path, sample_fps=sample_fps)
    dt = 1.0 / sample_fps
    intervals: list[Interval] = []
    cur_start = None
    prev_t = None
    for f in frames:
        if f.face_present:
            if cur_start is None:
                cur_start = f.t_s
            prev_t = f.t_s
        else:
            if cur_start is not None:
                intervals.append((cur_start, prev_t + dt))
                cur_start = None
    if cur_start is not None:
        intervals.append((cur_start, prev_t + dt))

    merged: list[Interval] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= min_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def _dominant_rhythm_period_ms(audio_normalized: np.ndarray, rate_hz: float) -> float | None:
    """Autocorrelate the clip's own (already face/scene-restricted) audio
    envelope against itself to find its dominant rhythm period, if it has
    one strong enough to matter. Cheap: reuses a signal already extracted
    for the main estimate, no new dependency or extra ffmpeg call.

    Returns `None` when there's no clear periodic structure (peak height
    below `RHYTHM_PEAK_MIN_HEIGHT`), which is the common case for natural,
    non-metronomic speech: the periodicity check is a no-op in that case.
    """
    n = len(audio_normalized)
    if n < 8:
        return None
    ac = np.correlate(audio_normalized, audio_normalized, mode="full")
    zero_lag_idx = n - 1
    zero_lag_val = ac[zero_lag_idx]
    if zero_lag_val <= 1e-9:
        return None

    lo_samples = max(1, int((MIN_RHYTHM_PERIOD_MS / 1000.0) * rate_hz))
    hi_samples = min(n - 1, int((MAX_RHYTHM_PERIOD_MS / 1000.0) * rate_hz))
    if hi_samples <= lo_samples:
        return None

    window = ac[zero_lag_idx + lo_samples: zero_lag_idx + hi_samples + 1]
    if len(window) == 0:
        return None
    peak_offset = int(np.argmax(window))
    peak_val = window[peak_offset]
    if (peak_val / zero_lag_val) < RHYTHM_PEAK_MIN_HEIGHT:
        return None
    period_samples = lo_samples + peak_offset
    return period_samples * (1000.0 / rate_hz)


def _rhythm_aliased_ambiguity(lags_ms: np.ndarray, normalized_corr: np.ndarray,
                               top_lag_ms: float, rhythm_period_ms: float) -> float | None:
    """Returns the height ratio of the tallest rhythm-spaced secondary peak
    (relative to the top peak), or `None` if no such peak exists. See
    module-level comment above `MIN_RHYTHM_PERIOD_MS` for the full
    reasoning: a secondary peak only counts if it's both comparably tall
    and spaced from the top peak by close to a whole multiple of the
    clip's own rhythm period; generic cross-correlation side-lobes that
    don't line up with the period are not flagged.
    """
    top_val = float(np.interp(top_lag_ms, lags_ms, normalized_corr))
    if top_val <= 1e-9:
        return None

    is_peak = np.zeros(len(normalized_corr), dtype=bool)
    if len(normalized_corr) >= 3:
        is_peak[1:-1] = ((normalized_corr[1:-1] > normalized_corr[:-2])
                         & (normalized_corr[1:-1] > normalized_corr[2:]))
    peak_indices = np.nonzero(is_peak)[0]

    best_ratio: float | None = None
    for idx in peak_indices:
        lag = float(lags_ms[idx])
        dist = abs(lag - top_lag_ms)
        if dist < MIN_PEAK_SEPARATION_MS:
            continue
        n_periods = round(dist / rhythm_period_ms)
        if n_periods < 1:
            continue
        expected = n_periods * rhythm_period_ms
        tol = max(MIN_PEAK_SEPARATION_MS, rhythm_period_ms * AMBIGUITY_LAG_TOLERANCE_FRACTION)
        if abs(dist - expected) > tol:
            continue
        height_ratio = float(normalized_corr[idx]) / top_val
        if height_ratio >= AMBIGUITY_HEIGHT_RATIO and (best_ratio is None or height_ratio > best_ratio):
            best_ratio = height_ratio
    return best_ratio


def _maybe_trim_source_for_scan(video_path: str, data_dir: Path, max_duration_s: float | None) -> str:
    """Returns a path to run this tier's two full-clip scans on: `video_path`
    unchanged if it's already <= `max_duration_s` (or capping is disabled),
    otherwise a frame-accurate temp file holding just a centered
    `max_duration_s`-second window. See `DEFAULT_MAX_SCAN_DURATION_S` above
    for why this exists and why 120s specifically.

    Mirrors `syncnet_offset._maybe_trim_to_window` exactly (same `-ss`
    after `-i` frame-accurate-decode-and-re-encode approach, for the same
    reason: stream-copy's keyframe-snapping can barely trim some files at
    all). Safe to do here for the same reason it's safe there: what this
    tier ultimately returns is a relative lag between the audio and video
    streams, not an absolute timestamp, and trimming both streams together
    from the same cut points preserves that lag exactly. It does not need
    to see the parts of the clip outside the window to measure the offset
    within it.
    """
    if max_duration_s is None:
        return video_path
    try:
        duration_s = probe_duration_s(video_path)
    except (ValueError, RuntimeError):
        return video_path  # couldn't probe: fail open, scan the whole clip
    if duration_s <= max_duration_s:
        return video_path

    start_s = max((duration_s - max_duration_s) / 2.0, 0.0)
    trimmed_path = data_dir / "wide_range_scan_window.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(Path(video_path).resolve()),
           "-ss", f"{start_s:.3f}", "-t", f"{max_duration_s:.3f}",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
           "-c:a", "aac", str(trimmed_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not trimmed_path.exists():
        return video_path  # fail open rather than crash the whole detection pipeline over this
    actual_duration_s = probe_duration_s(str(trimmed_path))
    if actual_duration_s > max_duration_s * 1.5:
        return video_path  # sanity check: didn't actually trim much, not worth the extra encode pass
    return str(trimmed_path)


def estimate_wide_range_offset(
        video_path: str,
        search_window_ms: float = DEFAULT_SEARCH_WINDOW_MS,
        min_face_coverage_s: float = DEFAULT_MIN_FACE_COVERAGE_S,
        max_scan_duration_s: float | None = DEFAULT_MAX_SCAN_DURATION_S,
) -> WideRangeEstimate:
    """Whole-clip, face-presence-restricted mouth-motion-vs-audio wide-range
    seed. See module docstring for why this is deliberately not split into
    smaller pieces for corroboration, and why that's safe (the safety net
    is downstream, in SyncNet's own fine-search confidence gate).

    `max_scan_duration_s` (see `DEFAULT_MAX_SCAN_DURATION_S`): if the source
    is longer than this, both of this tier's full-clip scans run on a single
    centered window of this length instead of the whole clip. Bounds this
    tier's cost on long uploads without materially changing its accuracy
    (see `_maybe_trim_source_for_scan`). `None` disables capping (always
    scans the full clip).

    Raises `ValueError` if there's not enough face-presence signal to
    attempt this at all, or if the periodicity self-check (module-level
    comment above `MIN_RHYTHM_PERIOD_MS`) finds the chosen peak is
    ambiguous against a rhythm-spaced neighbor. Callers should fall back
    to the whole-frame brightness coarse detector in either case.
    """
    with tempfile.TemporaryDirectory() as td:
        scan_path = _maybe_trim_source_for_scan(video_path, Path(td), max_scan_duration_s)

        intervals = _face_intervals(scan_path)
        coverage_s = sum(e - s for s, e in intervals)
        if coverage_s < min_face_coverage_s:
            raise ValueError(
                f"Only {coverage_s:.1f}s of face-presence signal found "
                f"(need >= {min_face_coverage_s:.1f}s), not enough to attempt a mouth-motion "
                f"wide-range seed (see module docstring, 'A second rejected version', for why "
                f"this floor exists)."
            )

        # Belt-and-suspenders with the floor above (see MAX_SEARCH_WINDOW_FRACTION):
        # even above the floor, never search wider than a fraction of actual
        # available signal: a clip can clear the coverage floor while still
        # having gaps that make the effective usable signal shorter.
        effective_window_ms = min(search_window_ms, coverage_s * 1000.0 * MAX_SEARCH_WINDOW_FRACTION)

        diag = estimate_mouth_sync_offset_with_diagnostics(
            scan_path, scenes=intervals, search_window_ms=effective_window_ms)
    r = diag.estimate

    rhythm_period_ms = _dominant_rhythm_period_ms(diag.audio_normalized, diag.rate_hz)
    if rhythm_period_ms is not None:
        secondary_ratio = _rhythm_aliased_ambiguity(
            diag.lags_ms, diag.normalized_corr, r.offset_ms, rhythm_period_ms)
        if secondary_ratio is not None:
            raise ValueError(
                f"Wide-range seed at {r.offset_ms:.0f}ms is ambiguous: this clip has a "
                f"~{rhythm_period_ms:.0f}ms speech-rhythm period, and the correlation curve has a "
                f"secondary peak {secondary_ratio:.2f}x as tall near a multiple of that period, "
                f"likely rhythm-driven aliasing rather than a genuine offset (periodicity "
                f"self-check)."
            )

    return WideRangeEstimate(offset_ms=r.offset_ms, confidence=r.confidence,
                              face_coverage_s=coverage_s, rhythm_period_ms=rhythm_period_ms)
