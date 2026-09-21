"""Caption/subtitle timing drift checker.

Production caption pipelines typically validate captions against picture
cut points or frame PTS. That is necessary but not sufficient: what
actually matters perceptually is whether caption cues line up with
speech, independent of any separate picture/audio sync issue. A caption
track can be perfectly aligned to frame PTS and still be wrong if it was
authored against a different audio mix, a different frame rate, or a
stale EDL.

This module detects speech-like onsets directly from the audio track
(via the same RMS-envelope machinery as the coarse A/V detector,
generalized to peak-picking rather than whole-signal cross-correlation)
and compares them against WebVTT cue start times, independent of the
video track entirely. This isolates "caption vs. speech" drift from
"picture vs. audio" drift, the two failure modes production QC needs to
tell apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import webvtt

from syncsentry.util.ffmpeg_io import extract_audio_envelope


@dataclass
class CueDrift:
    index: int
    text: str
    cue_start_s: float
    nearest_onset_s: float | None
    offset_ms: float | None  # cue_start - nearest_onset; None if no onset matched


@dataclass
class CaptionDriftReport:
    cues: list[CueDrift] = field(default_factory=list)
    matched_count: int = 0
    unmatched_count: int = 0
    mean_offset_ms: float | None = None
    median_offset_ms: float | None = None
    p95_abs_offset_ms: float | None = None
    flagged_cue_indices: list[int] = field(default_factory=list)


def _detect_onsets(video_path: str, onset_threshold_std: float = 1.5,
                    min_gap_s: float = 0.15) -> np.ndarray:
    """Simple energy-based onset picker: local maxima where envelope crosses
    `onset_threshold_std` standard deviations above its mean, at least
    `min_gap_s` apart. Adequate for the synthetic beep benchmark and as a
    drop-in slot for a proper VAD (e.g. Silero-VAD) once real speech
    content is wired in.
    """
    sig = extract_audio_envelope(video_path)
    if len(sig.values) == 0:
        return np.array([])

    thresh = np.mean(sig.values) + onset_threshold_std * np.std(sig.values)
    above = sig.values > thresh

    onsets = []
    last_onset_t = -np.inf
    # Edge case: if the signal is already "above" threshold at the very
    # first sample, the pulse's rising edge happened before the observed
    # window started (e.g. a caption/pulse that begins at t=0). Count it.
    if len(above) > 0 and above[0]:
        onsets.append(sig.times_s[0])
        last_onset_t = sig.times_s[0]

    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            t = sig.times_s[i]
            if t - last_onset_t >= min_gap_s:
                onsets.append(t)
                last_onset_t = t
    return np.asarray(onsets)


def check_caption_drift(video_path: str, vtt_path: str, flag_threshold_ms: float = 80.0,
                         max_match_window_s: float = 2.0) -> CaptionDriftReport:
    """Compare WebVTT cue start times against detected audio onsets."""
    onsets = _detect_onsets(video_path)
    captions = list(webvtt.read(vtt_path))

    report = CaptionDriftReport()
    offsets_ms: list[float] = []

    for idx, cue in enumerate(captions):
        start_s = _vtt_ts_to_seconds(cue.start)
        nearest = None
        if len(onsets) > 0:
            diffs = np.abs(onsets - start_s)
            j = int(np.argmin(diffs))
            if diffs[j] <= max_match_window_s:
                nearest = float(onsets[j])

        if nearest is None:
            report.cues.append(CueDrift(idx, cue.text, start_s, None, None))
            report.unmatched_count += 1
            continue

        offset_ms = (start_s - nearest) * 1000.0
        offsets_ms.append(offset_ms)
        report.cues.append(CueDrift(idx, cue.text, start_s, nearest, offset_ms))
        report.matched_count += 1
        if abs(offset_ms) > flag_threshold_ms:
            report.flagged_cue_indices.append(idx)

    if offsets_ms:
        arr = np.asarray(offsets_ms)
        report.mean_offset_ms = float(np.mean(arr))
        report.median_offset_ms = float(np.median(arr))
        report.p95_abs_offset_ms = float(np.percentile(np.abs(arr), 95))

    return report


def _vtt_ts_to_seconds(ts: str) -> float:
    # webvtt-py's Caption.start is already "HH:MM:SS.mmm"; parse manually to
    # avoid an extra dependency.
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)
