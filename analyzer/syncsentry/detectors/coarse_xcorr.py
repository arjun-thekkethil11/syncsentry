"""Coarse, model-free A/V sync offset detector.

This is the "baseline" tier of SyncSentry's detection stack: a signal-
processing approach analogous to the waveform-peak/frame-PTS heuristics
used in production caption/VOD tooling today. It works on any content (no
face required) but only detects a single global offset per clip/segment;
it cannot, by itself, distinguish drift-early/drift-late/intermittent
patterns the way a learned per-scene approach can. That per-scene
classification is implemented on top of this in
`syncsentry.detectors.title_drift`.

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


def _correlation_curve(a: np.ndarray, v: np.ndarray, rate_hz: float,
                        search_window_ms: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Shared core of `estimate_offset_from_signals`: the full normalized
    cross-correlation curve within the search window, as `(lags_ms,
    normalized_corr, denom)`. Factored out so a caller that needs to
    inspect the shape of the curve, not just its single tallest peak, can
    do so without duplicating this arithmetic. Concretely,
    `wide_range_offset.py`'s periodicity self-check needs to know whether
    the top peak is uniquely tall or whether comparably-tall secondary
    peaks exist at other lags (a signature of rhythm-driven aliasing),
    information `estimate_offset_from_signals` below discards once it
    picks the single best lag.
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

    # Normalize by the theoretical max (||a|| * ||v||) at zero lag equivalent
    # energy, giving a rough [0, 1]-ish correlation coefficient at every lag,
    # not just the peak.
    denom = float(np.sqrt(np.sum(a ** 2) * np.sum(v ** 2)))
    normalized_corr = window_corr / denom if denom > 1e-9 else np.zeros_like(window_corr, dtype=float)
    lags_ms = window_lags.astype(float) * (1000.0 / rate_hz)
    return lags_ms, normalized_corr, denom


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
    lags_ms, normalized_corr, _denom = _correlation_curve(a, v, rate_hz, search_window_ms)

    peak_idx = int(np.argmax(normalized_corr))
    confidence = float(normalized_corr[peak_idx])

    # lag here is samples such that a[i] aligns with v[i - lag]; converting
    # to "audio relative to video" offset: positive lag means audio's
    # pattern appears *later* in index space than video's => audio lags.
    offset_ms = float(lags_ms[peak_idx])

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
