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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
class SyncNetTrackResult:
    offset_frames: int  # SyncNet's own convention: positive = audio leads video
    confidence: float  # median_dist - min_dist; NOT the [0,1] scale used elsewhere
    n_frames: int


@dataclass
class SyncNetEstimate:
    offset_ms: float
    confidence: float  # aggregated across tracks; same non-[0,1] scale as above
    direction: str  # "in_sync" | "audio_lags" | "audio_leads"
    n_tracks: int


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


def estimate_syncnet_tracks(video_path: str, vshift: int = 15) -> list[SyncNetTrackResult]:
    """Run the full 2-stage SyncNet pipeline and return one result per
    detected face track (see module docstring)."""
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

        results = []
        for crop_file in crop_files:
            offset, conf, dist = s.evaluate(opt, videofile=str(crop_file))
            results.append(SyncNetTrackResult(
                offset_frames=int(offset), confidence=float(conf), n_frames=len(dist),
            ))
        return results


def estimate_syncnet_offset(video_path: str, vshift: int = 15, frame_rate: float = 25.0,
                             in_sync_threshold_ms: float = 40.0) -> SyncNetEstimate:
    """Run SyncNet and aggregate across all detected face tracks into a
    single estimate: median offset (robust to any one track disagreeing),
    median confidence (conservative summary, not the best-case track)."""
    tracks = estimate_syncnet_tracks(video_path, vshift=vshift)
    if not tracks:
        raise ValueError("No face tracks found (no trackable face for >= min_track frames).")

    import statistics
    offset_frames = statistics.median(t.offset_frames for t in tracks)
    confidence = statistics.median(t.confidence for t in tracks)
    offset_ms = -offset_frames * (1000.0 / frame_rate)  # sign flip -- see module docstring

    if abs(offset_ms) <= in_sync_threshold_ms:
        direction = "in_sync"
    elif offset_ms > 0:
        direction = "audio_lags"
    else:
        direction = "audio_leads"

    return SyncNetEstimate(offset_ms=offset_ms, confidence=confidence,
                            direction=direction, n_tracks=len(tracks))
