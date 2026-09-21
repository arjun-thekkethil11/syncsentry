"""Per-window A/V offset estimation across a whole title.

DiVAS (CVPR 2024) gets its title-level drift classification by making an
offset prediction per dialogue scene, then regressing across scenes. The
real version of "per scene" needs face detection plus speech-activity
overlap (see `syncsentry.lipsync.dialogue_scenes`); a pretrained lip-sync
embedding model as the per-scene estimator itself is deferred until real
content is available to validate it against.

This module provides the statistical half of that pipeline: fixed-length
sliding windows across the full audio/video envelope, with the existing
cross-correlation estimator (`coarse_xcorr`) computing an offset for each
window. It's a legitimate per-scene offset estimator on its own for
content where energy-based correlation works (loud, punctuated audio
events), and it's the exact interface (`list[SceneOffset]` ->
`syncsentry.lipsync.title_drift`) that a future learned per-scene
estimator slots into without changing the classification code at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syncsentry.detectors.coarse_xcorr import estimate_offset_from_signals, extract_normalized_envelopes


@dataclass
class SceneOffset:
    t_center_s: float
    t_start_s: float
    t_end_s: float
    offset_ms: float
    confidence: float


def estimate_windowed_offsets(video_path: str, window_s: float = 3.0, hop_s: float | None = None,
                               search_window_ms: float = 500.0,
                               min_confidence: float = 0.3) -> list[SceneOffset]:
    """Slide a `window_s`-wide window across the whole title (hop `hop_s`,
    default = window_s/2 for 50% overlap) and estimate an A/V offset for
    each window independently. Windows whose peak correlation confidence
    falls below `min_confidence` are dropped: a window with no strong
    audio/video events to correlate against produces a meaningless estimate,
    and DiVAS handles this the same way (dropping low-confidence per-scene
    predictions before the title-level regression).
    """
    hop_s = hop_s if hop_s is not None else window_s / 2.0
    a, v, rate_hz, t_start = extract_normalized_envelopes(video_path)

    window_samples = int(window_s * rate_hz)
    hop_samples = max(1, int(hop_s * rate_hz))

    results: list[SceneOffset] = []
    n = min(len(a), len(v))
    start_idx = 0
    while start_idx + window_samples <= n:
        end_idx = start_idx + window_samples
        a_win = a[start_idx:end_idx]
        v_win = v[start_idx:end_idx]

        try:
            estimate = estimate_offset_from_signals(a_win, v_win, rate_hz, search_window_ms)
        except ValueError:
            start_idx += hop_samples
            continue

        if estimate.confidence >= min_confidence:
            t0 = t_start + start_idx / rate_hz
            t1 = t_start + end_idx / rate_hz
            results.append(SceneOffset(
                t_center_s=(t0 + t1) / 2.0, t_start_s=t0, t_end_s=t1,
                offset_ms=estimate.offset_ms, confidence=estimate.confidence,
            ))

        start_idx += hop_samples

    return results
