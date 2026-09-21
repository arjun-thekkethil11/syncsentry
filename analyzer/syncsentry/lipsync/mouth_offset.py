"""Mouth-region-motion A/V offset estimator: a targeted alternative to the
whole-frame brightness detector (`syncsentry.detectors.coarse_xcorr`) for
real talking-head content.

On real dialogue footage, correlating whole-frame brightness against audio
energy is dominated by noise, because the only part of the frame that
actually moves in sync with speech is the mouth, a small fraction of total
pixels. Restricting the visual signal to the mouth region itself (located
via YuNet's mouth-corner landmarks, already available from the face
detector) and restricting the time range to real dialogue scenes (face and
speech overlap, `lipsync.dialogue_scenes`) fixes both problems at once,
without needing a pretrained lip-sync embedding model. This is still
classical signal processing, using the same cross-correlation core as
`coarse_xcorr.py`, just fed a much more targeted visual signal. A learned
SyncNet-style embedding of this same mouth crop (`syncnet_offset.py`)
generally performs better; this module remains useful as a lighter-weight,
model-free estimator and as the wide-range seed in `wide_range_offset.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from syncsentry.detectors.coarse_xcorr import (
    COMMON_RATE_HZ,
    OffsetEstimate,
    _correlation_curve,
    _normalize,
    _resample_uniform,
    estimate_offset_from_signals,
)
from syncsentry.lipsync.face_detect import _load_detector
from syncsentry.util.ffmpeg_io import Signal, extract_audio_envelope

Interval = tuple[float, float]


def _mouth_roi(face: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    """Bounding box around the mouth from one YuNet detection row.

    YuNet's output row is [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt,
    x_rcm, y_rcm, x_lcm, y_lcm, score]; indices 10:14 are the right/left
    mouth-corner landmarks. Those two points only give horizontal extent, so
    vertical extent is derived from face-box height (empirically, mouth
    height is roughly 0.16x face height).
    """
    x, y, w, h = face[0], face[1], face[2], face[3]
    x_rcm, y_rcm, x_lcm, y_lcm = face[10], face[11], face[12], face[13]
    cx = (x_rcm + x_lcm) / 2.0
    cy = (y_rcm + y_lcm) / 2.0
    half_w = max(abs(x_lcm - x_rcm), 1.0) * 0.9 + w * 0.05
    half_h = h * 0.16
    fh, fw = frame_shape[0], frame_shape[1]
    x0, x1 = int(max(0, cx - half_w)), int(min(fw, cx + half_w))
    y0, y1 = int(max(0, cy - half_h)), int(min(fh, cy + half_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def extract_mouth_motion_signal(video_path: str, sample_fps: float = 15.0,
                                 score_threshold: float = 0.6,
                                 scenes: list[Interval] | None = None) -> Signal:
    """Sample `video_path` at `sample_fps`, run YuNet per sampled frame, and
    return a signal of mouth-region motion magnitude (mean abs frame diff
    within the mouth ROI, between consecutive *evaluated* samples).

    If `scenes` is given, only samples inside one of those (start_s, end_s)
    intervals are evaluated: this removes non-speech/no-face dead time that
    would otherwise dilute the cross-correlation. The motion diff is reset
    (not carried over) across any gap between evaluated samples, so a scene
    boundary never gets diffed against a stale frame from a previous,
    possibly-distant scene.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, int(round(fps / sample_fps)))

    def in_scenes(t: float) -> bool:
        if not scenes:
            return True
        return any(s <= t <= e for s, e in scenes)

    detector = None
    prev_roi_gray: np.ndarray | None = None
    last_eval_idx: int | None = None
    times: list[float] = []
    values: list[float] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            t = idx / fps
            if in_scenes(t):
                if last_eval_idx is None or idx - last_eval_idx != sample_every:
                    prev_roi_gray = None  # gap since last evaluated sample: don't diff across it
                last_eval_idx = idx

                h, w = frame.shape[:2]
                if detector is None:
                    detector = _load_detector(score_threshold, (w, h))
                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)

                motion = 0.0
                if faces is not None and len(faces) > 0:
                    box = _mouth_roi(faces[0], frame.shape)
                    if box is not None:
                        x0, y0, x1, y1 = box
                        roi_gray = cv2.resize(
                            cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (32, 24),
                        )
                        if prev_roi_gray is not None:
                            motion = float(np.mean(np.abs(
                                roi_gray.astype(np.float32) - prev_roi_gray.astype(np.float32)
                            )))
                        prev_roi_gray = roi_gray
                    else:
                        prev_roi_gray = None
                else:
                    prev_roi_gray = None

                times.append(t)
                values.append(motion)
        idx += 1

    cap.release()
    return Signal(np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64))


def _resample_scenes(sig: Signal, scenes: list[Interval], rate_hz: float) -> np.ndarray:
    """Resample `sig` to `rate_hz` independently within each scene and
    concatenate. Deliberately does not interpolate across the gaps
    *between* scenes, which would reintroduce the dead-time dilution
    problem this module exists to avoid."""
    chunks = []
    for s, e in scenes:
        if e - s < 2.0 / rate_hz:
            continue
        chunks.append(_resample_uniform(sig, s, e, rate_hz))
    return np.concatenate(chunks) if chunks else np.array([])


@dataclass
class MouthSyncDiagnostics:
    """Everything `estimate_mouth_sync_offset` computes internally but
    discards after picking the single best lag: the full normalized
    correlation curve and the normalized audio signal it was computed from.

    Used by the periodicity self-check in `wide_range_offset.py`: telling
    a uniquely tall correlation peak apart from one with comparably tall
    neighbors spaced at the clip's own speech-rhythm period requires the
    shape of the curve, not just its argmax.
    """
    estimate: OffsetEstimate
    lags_ms: np.ndarray
    normalized_corr: np.ndarray
    audio_normalized: np.ndarray
    rate_hz: float


def estimate_mouth_sync_offset_with_diagnostics(
        video_path: str, scenes: list[Interval] | None = None,
        sample_fps: float = 15.0, search_window_ms: float = 500.0,
        in_sync_threshold_ms: float = 40.0) -> MouthSyncDiagnostics:
    """As `estimate_mouth_sync_offset`, but returns the underlying
    correlation curve and normalized audio signal alongside the final
    estimate. See `MouthSyncDiagnostics`. `estimate_mouth_sync_offset` below
    is now a thin wrapper around this: kept separate so existing callers
    that only want the final estimate aren't forced to handle the extra
    diagnostics.
    """
    mouth = extract_mouth_motion_signal(video_path, sample_fps=sample_fps, scenes=scenes)
    if len(mouth.times_s) < 4:
        raise ValueError("Not enough mouth-motion signal extracted (no face detected in scenes?).")

    audio = extract_audio_envelope(video_path)
    if len(audio.times_s) < 4:
        raise ValueError("Not enough audio signal extracted to estimate offset.")

    rate_hz = min(COMMON_RATE_HZ, sample_fps * 4)  # no point resampling well above the mouth signal's own rate

    if scenes:
        a = _resample_scenes(audio, scenes, rate_hz)
        v = _resample_scenes(mouth, scenes, rate_hz)
    else:
        t_start = max(mouth.times_s[0], audio.times_s[0])
        t_end = min(mouth.times_s[-1], audio.times_s[-1])
        if t_end <= t_start:
            raise ValueError("Mouth-motion and audio signals do not overlap in time.")
        a = _resample_uniform(audio, t_start, t_end, rate_hz)
        v = _resample_uniform(mouth, t_start, t_end, rate_hz)

    if len(a) < 4 or len(v) < 4:
        raise ValueError("Not enough overlapping signal within the given scenes to estimate offset.")

    a_norm = _normalize(a)
    v_norm = _normalize(v)
    estimate = estimate_offset_from_signals(a_norm, v_norm, rate_hz, search_window_ms, in_sync_threshold_ms)
    lags_ms, normalized_corr, _denom = _correlation_curve(a_norm, v_norm, rate_hz, search_window_ms)
    return MouthSyncDiagnostics(estimate, lags_ms, normalized_corr, a_norm, rate_hz)


def estimate_mouth_sync_offset(video_path: str, scenes: list[Interval] | None = None,
                                sample_fps: float = 15.0, search_window_ms: float = 500.0,
                                in_sync_threshold_ms: float = 40.0) -> OffsetEstimate:
    """Cross-correlate mouth-region motion (restricted to `scenes` if given)
    against the audio envelope. Same core estimator as
    `coarse_xcorr.estimate_offset_from_signals`; the difference is entirely
    in which visual signal it's fed and where it's allowed to look.
    """
    return estimate_mouth_sync_offset_with_diagnostics(
        video_path, scenes=scenes, sample_fps=sample_fps,
        search_window_ms=search_window_ms, in_sync_threshold_ms=in_sync_threshold_ms,
    ).estimate
