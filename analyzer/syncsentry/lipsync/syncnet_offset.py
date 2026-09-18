"""Learned lip-sync offset estimation via the pretrained SyncNet model
(Chung & Zisserman, "Out of time: automated lip sync in the wild", ACCV
Workshop 2016) -- the actual M3c estimator, after the classical mouth-motion
heuristic in `mouth_offset.py` was tried and empirically falsified (see
`docs/RESEARCH.md` sections 1d/1e).

This wraps `joonson/syncnet_python` (MIT-licensed), vendored via
`analyzer/scripts/fetch_syncnet.sh` into `analyzer/third_party/syncnet_python/`
-- never committed, same "fetch, don't commit" policy already used for the
real-content clip and its pretrained weights (`analyzer/third_party/SOURCES.md`).

Two real stages, no shortcuts:
  1. Face detection + tracking + 224x224 mouth-centered crop (S3FD,
     `run_pipeline.py`, run as a subprocess -- it's a standalone script with
     module-level side effects, not an importable library).
  2. SyncNet embedding + sliding-window L2-distance offset search on each
     cropped face track (`SyncNetInstance`, imported directly so we get real
     (offset, confidence) values, not log text to parse).

A clip can produce 0+ face tracks (e.g. a cutaway with no face produces
none); results are returned per-track and aggregated by the caller.

Sign convention: SyncNet's own `offset` is frames, positive = audio LEADS
video. This module converts to this project's convention (positive ms =
audio LAGS video, matching `syncsentry.detectors.coarse_xcorr`) via
`offset_ms = -offset_frames * (1000 / frame_rate)`.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_SYNCNET_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "syncnet_python"
_MODEL_PATH = _SYNCNET_DIR / "data" / "syncnet_v2.model"

# SyncNet's raw confidence (median_dist - min_dist across the search window)
# is NOT on the [0, 1] scale used by the coarse detector's normalized
# correlation. Empirically (docs/RESEARCH.md section 1e), genuinely
# lip-sync-trackable real content scores ~3.5-8.5 here; well below that
# suggests a track without a clear, trackable talking face.
DEFAULT_SYNCNET_MIN_CONFIDENCE = 3.0


class SyncNetUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    return _SYNCNET_DIR.exists() and _MODEL_PATH.exists()


def _check_available() -> None:
    if not is_available():
        raise SyncNetUnavailable(
            f"SyncNet not found at {_SYNCNET_DIR}. Run "
            f"`bash analyzer/scripts/fetch_syncnet.sh` once to fetch the "
            f"(MIT-licensed) reference implementation and pretrained weights -- "
            f"never committed to this repo, see analyzer/third_party/SOURCES.md."
        )


@dataclass
class SyncNetWindowResult:
    """One short (~1s) sub-segment of a face track, scored independently.

    Why this exists (see docs/RESEARCH.md section 1f, then 1g): a whole
    face-track's confidence is a single median over every frame, including
    frames where the tracked person is listening rather than talking (real
    in multi-speaker dialogue). Those frames dilute the aggregate --
    per-window scoring lets the pipeline find and use only the sub-segments
    where *this specific* face's lip motion is actually confidently
    correlated with the audio, i.e. where they're the one speaking.
    """
    track_index: int
    frame_start: int
    offset_frames: int
    confidence: float


@dataclass
class SyncNetTrackResult:
    offset_frames: int  # SyncNet's own convention: positive = audio leads video
    confidence: float  # median_dist - min_dist; NOT the [0,1] scale used elsewhere
    n_frames: int
    windows: list[SyncNetWindowResult] = field(default_factory=list)


@dataclass
class SyncNetEstimate:
    offset_ms: float
    confidence: float  # aggregated across tracks; same non-[0,1] scale as above
    direction: str  # "in_sync" | "audio_lags" | "audio_leads"
    n_tracks: int
    n_confident_windows: int = 0  # 0 means: no window cleared the bar, fell back to whole-track aggregation


def _run_face_track_crop(video_path: str, data_dir: Path, reference: str) -> None:
    """Stage 1: face detection + tracking + crop, via subprocess -- run_pipeline.py
    calls `parser.parse_args()` at module scope, so it can't be imported as a
    library; the video path is resolved to absolute since we invoke it with a
    different cwd."""
    cmd = [
        sys.executable, "run_pipeline.py",
        "--videofile", str(Path(video_path).resolve()),
        "--reference", reference,
        "--data_dir", str(data_dir.resolve()),
        "--overwrite",
    ]
    proc = subprocess.run(cmd, cwd=str(_SYNCNET_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SyncNet face-tracking stage failed:\n{proc.stderr[-4000:]}")


def _mouth_motion_per_frame(crop_file: Path, n_frames: int) -> np.ndarray:
    """Per-frame mouth-motion magnitude for a face-track crop, used as a
    cheap, fully-automated active-speaker gate (docs/RESEARCH.md section
    1h): a frame where this specific tracked face's mouth is barely moving
    is almost certainly a frame where they're listening, not talking --
    exactly the frames that dilute SyncNet's per-track/whole-track
    aggregate on multi-speaker dialogue (sections 1f/1g). No new face/
    landmark detection needed: SyncNet's own crop stage already produces a
    tight, face-centered 224x224 track (module docstring), so the mouth is
    reliably in the lower portion of every frame without re-detecting it.

    Deliberately NOT reused as an offset estimator itself -- that's
    `lipsync/mouth_offset.py`, already tried and falsified (section 1d).
    This only asks "is the mouth moving at all right now", a much easier
    and more robust question than "by how many ms is it offset from the
    audio", which is exactly why it can do a job the falsified detector
    couldn't: gate frames for SyncNet's own (independently validated,
    section 1e) offset signal, not replace it.
    """
    cap = cv2.VideoCapture(str(crop_file))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = gray.shape[0]
        frames.append(gray[int(h * 0.55):, :])  # lower ~45%: mouth/chin region of a face-centered crop
    cap.release()

    if len(frames) < 2:
        return np.zeros(n_frames)

    motion = [0.0] + [
        float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        for i in range(1, len(frames))
    ]
    motion = np.array(motion)
    if len(motion) >= n_frames:
        return motion[:n_frames]
    return np.pad(motion, (0, n_frames - len(motion)), mode="edge")


def _windowed_offsets(dist: np.ndarray, vshift: int, track_index: int,
                       window_frames: int, step_frames: int,
                       motion: np.ndarray | None = None,
                       motion_threshold: float | None = None) -> list[SyncNetWindowResult]:
    """Slide a window over a track's per-frame x per-shift distance matrix
    (`dist`, shape [n_frames, 2*vshift+1] -- SyncNet's own raw output,
    already computed by `SyncNetInstance.evaluate()` but discarded by the
    whole-track-only aggregation) and score each window exactly the way
    SyncNet scores a whole track (mean distance per shift, then
    median-minus-min for confidence), just over a short sub-segment
    instead of the whole track. A window where this face's lips are
    actually moving in sync with the audio scores high; a window where
    they're just listening scores near the track median, i.e. low.
    """
    n_frames = dist.shape[0]
    windows = []
    for start in range(0, max(n_frames - window_frames, 0) + 1, step_frames):
        chunk = dist[start:start + window_frames]
        if len(chunk) < window_frames:
            continue
        # Active-speaker gate (section 1h): skip windows where this track's
        # mouth is relatively still -- almost certainly listening, not
        # talking, in multi-speaker content. `motion_threshold` is
        # track-relative (its own median), not a fixed pixel-difference
        # value, so it self-calibrates to each track's resolution/lighting.
        if motion is not None and motion_threshold is not None:
            window_motion = float(np.mean(motion[start:start + window_frames]))
            if window_motion < motion_threshold:
                continue
        mean_dist_per_shift = chunk.mean(axis=0)
        minidx = int(np.argmin(mean_dist_per_shift))
        offset_frames = vshift - minidx
        # A minimum at the very edge of the searched shift range is the
        # signature of a spurious/noise-dominated match (the "best" shift
        # is an artifact of the search boundary, not a real local minimum),
        # not a real one -- same class of false-confidence failure already
        # documented for the coarse detector (docs/RESEARCH.md section 1b).
        # Exclude these outright rather than let a high `conf` value (which
        # a short, noisy window can produce just as easily at the boundary
        # as anywhere else) make it look trustworthy.
        if abs(offset_frames) >= vshift:
            continue
        minval = float(mean_dist_per_shift[minidx])
        conf = float(np.median(mean_dist_per_shift) - minval)
        windows.append(SyncNetWindowResult(
            track_index=track_index, frame_start=start,
            offset_frames=offset_frames, confidence=conf,
        ))
    return windows


def estimate_syncnet_tracks(video_path: str, vshift: int = 15, window_s: float = 1.0,
                             frame_rate: float = 25.0, gate_on_mouth_motion: bool = True) -> list[SyncNetTrackResult]:
    """Run the full 2-stage SyncNet pipeline and return one result per
    detected face track (see module docstring), each carrying both a
    whole-track aggregate and a set of short-window sub-scores (see
    `_windowed_offsets`) for callers that want to discount
    listening-not-speaking frames rather than average over them.

    `gate_on_mouth_motion` (default True): also require above-median mouth
    motion for a window to be scored at all (section 1h's active-speaker
    gate) -- set False to get the pre-1h windowing behavior (section 1g)
    for comparison/debugging.
    """
    _check_available()

    sys.path.insert(0, str(_SYNCNET_DIR))
    try:
        from SyncNetInstance import SyncNetInstance
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        reference = "clip"
        _run_face_track_crop(video_path, data_dir, reference)

        crop_files = sorted((data_dir / "pycrop" / reference).glob("0*.avi"))
        if not crop_files:
            return []

        opt = argparse.Namespace(batch_size=20, vshift=vshift,
                                  tmp_dir=str(data_dir / "pytmp"), reference=reference)
        s = SyncNetInstance()
        s.loadParameters(str(_MODEL_PATH))

        window_frames = max(int(round(window_s * frame_rate)), 1)
        results = []
        for track_index, crop_file in enumerate(crop_files):
            offset, conf, dist = s.evaluate(opt, videofile=str(crop_file))

            motion, motion_threshold = None, None
            if gate_on_mouth_motion:
                motion = _mouth_motion_per_frame(crop_file, n_frames=dist.shape[0])
                motion_threshold = float(np.median(motion))

            windows = _windowed_offsets(dist, vshift, track_index, window_frames=window_frames,
                                         step_frames=window_frames, motion=motion,
                                         motion_threshold=motion_threshold)
            results.append(SyncNetTrackResult(
                offset_frames=int(offset), confidence=float(conf), n_frames=len(dist),
                windows=windows,
            ))
        return results


def _largest_agreeing_cluster(windows: list[SyncNetWindowResult],
                               tolerance_frames: int = 2) -> list[SyncNetWindowResult]:
    """Group windows whose offsets agree within `tolerance_frames` and
    return the largest such group.

    Why this exists: individually-confident windows can still just be
    wrong (a short ~1s window is small enough that a spurious match can
    score just as high as a real one -- e.g. the real dialogue clip in
    docs/RESEARCH.md section 1g produced windows individually scoring
    3.5-5.5 confidence whose offsets *disagreed with each other* by up to
    560ms). Requiring several *independent* windows -- different tracks,
    different points in time -- to land on the same offset before trusting
    it is a much stronger check than any single window's confidence
    number, and is what actually distinguishes "one narrator, briefly
    silent in some windows" (should cluster tightly) from "multiple
    speakers, no single global offset any window can consistently see"
    (won't cluster at all -- see the real dialogue-clip case).
    """
    best: list[SyncNetWindowResult] = []
    for w in windows:
        cluster = [x for x in windows if abs(x.offset_frames - w.offset_frames) <= tolerance_frames]
        if len(cluster) > len(best):
            best = cluster
    return best


def estimate_syncnet_offset(video_path: str, vshift: int = 15, frame_rate: float = 25.0,
                             in_sync_threshold_ms: float = 40.0, window_s: float = 1.0,
                             min_window_confidence: float = DEFAULT_SYNCNET_MIN_CONFIDENCE,
                             cluster_tolerance_frames: int = 2,
                             min_cluster_size: int = 3,
                             gate_on_mouth_motion: bool = True) -> SyncNetEstimate:
    """Run SyncNet and aggregate into a single estimate.

    Two aggregation paths, tried in order:

    1. Windowed evidence, gated on cross-window agreement (see
       `_windowed_offsets` + `_largest_agreeing_cluster`): pool every ~1s
       window from every face track, keep the ones that individually
       clear `min_window_confidence`, then require at least
       `min_cluster_size` of *those* to independently agree with each
       other (within `cluster_tolerance_frames`) before trusting them.
       This is what makes multi-speaker dialogue tractable
       (docs/RESEARCH.md sections 1f -> 1g): a track's non-speaking frames
       no longer drag down a whole-track average, AND a single lucky/
       spurious confident window can no longer look like a real answer on
       its own -- it has to be corroborated.
    2. Whole-track fallback (the original M3c behavior): used whenever no
       cluster of agreeing confident windows is found -- either because no
       window was individually confident, or (the new, real failure mode
       this catches) confident windows exist but contradict each other.
       Correctly conservative: on a single continuous narrator this
       fallback is close to whichever windowed answer would've been found
       anyway; on real multi-speaker dialogue it correctly stays at the
       same low whole-track confidence M3c already reported before this
       change, rather than being fooled into false confidence by windowing.
    """
    tracks = estimate_syncnet_tracks(video_path, vshift=vshift, window_s=window_s, frame_rate=frame_rate,
                                      gate_on_mouth_motion=gate_on_mouth_motion)
    if not tracks:
        raise ValueError("No face tracks found (no trackable face for >= min_track frames).")

    all_windows = [w for t in tracks for w in t.windows]
    confident_windows = [w for w in all_windows if w.confidence >= min_window_confidence]
    cluster = _largest_agreeing_cluster(confident_windows, tolerance_frames=cluster_tolerance_frames)

    if len(cluster) >= min_cluster_size:
        offset_frames = statistics.median(w.offset_frames for w in cluster)
        confidence = statistics.median(w.confidence for w in cluster)
        n_confident_windows = len(cluster)
    else:
        offset_frames = statistics.median(t.offset_frames for t in tracks)
        confidence = statistics.median(t.confidence for t in tracks)
        n_confident_windows = 0

    offset_ms = -offset_frames * (1000.0 / frame_rate)  # sign flip -- see module docstring

    if abs(offset_ms) <= in_sync_threshold_ms:
        direction = "in_sync"
    elif offset_ms > 0:
        direction = "audio_lags"
    else:
        direction = "audio_leads"

    return SyncNetEstimate(offset_ms=offset_ms, confidence=confidence, direction=direction,
                            n_tracks=len(tracks), n_confident_windows=n_confident_windows)
