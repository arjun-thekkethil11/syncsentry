"""Coarse, model-free A/V sync offset detector.

This is the "baseline" tier of SyncSentry's detection stack: a signal-
processing approach analogous to the waveform-peak / frame-PTS heuristics
used in production caption/VOD tooling today. It works on *any* content
(no face required) but only detects a single global offset per clip/segment
-- it cannot, by itself, distinguish drift-early / drift-late / intermittent
patterns the way the learned per-scene approach in DiVAS (CVPR 2024) does.
That per-scene classification is implemented on top of this in
`syncsentry.detectors.title_drift` (see docs/RESEARCH.md, Milestone 3).

Method: extract an audio RMS envelope and a video brightness envelope,
resample both to a common rate, normalize, and cross-correlate. The lag at
peak correlation is the estimated offset. Sign convention matches
`syncsentry.synth.fixture_gen`: positive offset_ms means audio lags video.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syncsentry.util.ffmpeg_io import Signal, extract_audio_envelope, extract_video_brightness

COMMON_RATE_HZ = 200.0  # 5ms resolution


@dataclass
class OffsetEstimate:
    offset_ms: float
    confidence: float  # normalized peak correlation, roughly in [0, 1]
    direction: str  # "in_sync" | "audio_lags" | "audio_leads"


def _resample_uniform(sig: Signal, t_start: float, t_end: float, rate_hz: float) -> np.ndarray:
    n = max(2, int((t_end - t_start) * rate_hz))
    grid = np.linspace(t_start, t_end, n)
    return np.interp(grid, sig.times_s, sig.values)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    std = np.std(x)
    if std < 1e-9:
        return x
    return x / std


def estimate_offset_from_signals(a: np.ndarray, v: np.ndarray, rate_hz: float,
                                  search_window_ms: float = 500.0,
                                  in_sync_threshold_ms: float = 40.0) -> OffsetEstimate:
    """Core cross-correlation estimator, operating on two already-uniform,
    already-normalized 1-D arrays sampled at `rate_hz`.

    Factored out of `estimate_av_offset` so callers that need many estimates
    over one file (e.g. `syncsentry.lipsync.scene_offsets`, which slides a
    window across a title) can extract the audio/video envelopes *once* and
    reuse this function per-window, instead of re-running ffmpeg extraction
    for every window.
    """
    if len(a) < 2 or len(v) < 2:
        raise ValueError("Not enough signal to estimate offset.")

    corr = np.correlate(a, v, mode="full")
    n = len(v)
    lags = np.arange(-n + 1, n)  # lag applied to `a` relative to `v`

    max_lag_samples = int((search_window_ms / 1000.0) * rate_hz)
    center = len(corr) // 2
    lo = max(0, center - max_lag_samples)
    hi = min(len(corr), center + max_lag_samples + 1)

    window_corr = corr[lo:hi]
    window_lags = lags[lo:hi]

    peak_idx = int(np.argmax(window_corr))
    peak_lag_samples = window_lags[peak_idx]
    peak_val = window_corr[peak_idx]

    # Normalize confidence by the theoretical max (||a|| * ||v||) at zero lag
    # equivalent energy, giving a rough [0, 1]-ish correlation coefficient.
    denom = np.sqrt(np.sum(a ** 2) * np.sum(v ** 2))
    confidence = float(peak_val / denom) if denom > 1e-9 else 0.0

    # lag here is samples such that a[i] aligns with v[i - lag]; converting
    # to "audio relative to video" offset: positive lag means audio's
    # pattern appears *later* in index space than video's => audio lags.
    offset_ms = float(peak_lag_samples) * (1000.0 / rate_hz)

    if abs(offset_ms) <= in_sync_threshold_ms:
        direction = "in_sync"
    elif offset_ms > 0:
        direction = "audio_lags"
    else:
        direction = "audio_leads"

    return OffsetEstimate(offset_ms=offset_ms, confidence=confidence, direction=direction)


def extract_normalized_envelopes(video_path: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Extract audio + video envelopes for `video_path`, resampled to a
    common uniform rate and normalized. Returns (audio_array, video_array,
    rate_hz, t_start_s) so callers can map array indices back to wall-clock
    time (needed by the windowed scene-offset estimator).
    """
    audio = extract_audio_envelope(video_path)
    video = extract_video_brightness(video_path)

    if len(audio.times_s) < 2 or len(video.times_s) < 2:
        raise ValueError("Not enough audio/video signal extracted to estimate offset.")

    t_start = max(audio.times_s[0], video.times_s[0])
    t_end = min(audio.times_s[-1], video.times_s[-1])
    if t_end <= t_start:
        raise ValueError("Audio and video signals do not overlap in time.")

    a = _normalize(_resample_uniform(audio, t_start, t_end, COMMON_RATE_HZ))
    v = _normalize(_resample_uniform(video, t_start, t_end, COMMON_RATE_HZ))
    return a, v, COMMON_RATE_HZ, t_start


def estimate_av_offset(video_path: str, search_window_ms: float = 500.0,
                        in_sync_threshold_ms: float = 40.0) -> OffsetEstimate:
    """Estimate the global A/V offset of a media file via envelope cross-correlation.

    `in_sync_threshold_ms` is a configurable tolerance window; real broadcast
    QC pipelines typically flag anything beyond a few tens of milliseconds
    (exact thresholds vary by org/spec), so this defaults to 40ms but should
    be tuned per-deployment rather than treated as a universal constant.
    """
    a, v, rate_hz, _ = extract_normalized_envelopes(video_path)
    return estimate_offset_from_signals(a, v, rate_hz, search_window_ms, in_sync_threshold_ms)
