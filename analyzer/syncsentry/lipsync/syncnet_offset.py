"""Learned lip-sync offset estimation via the pretrained SyncNet model
(Chung & Zisserman, "Out of time: automated lip sync in the wild", ACCV
Workshop 2016), used in place of the classical mouth-motion heuristic in
`mouth_offset.py` for general talking-head content.

Wraps `joonson/syncnet_python` (MIT-licensed), vendored via
`analyzer/scripts/fetch_syncnet.sh` into
`analyzer/third_party/syncnet_python/`. Never committed, same
fetch-don't-commit policy used for the real-content clips and pretrained
weights (`analyzer/third_party/SOURCES.md`).

Two real stages, no shortcuts:
  1. Face detection + tracking + 224x224 mouth-centered crop (S3FD,
     `run_pipeline.py`, run as a subprocess since it's a standalone script
     with module-level side effects, not an importable library).
  2. SyncNet embedding + sliding-window L2-distance offset search on each
     cropped face track (`SyncNetInstance`, imported directly so we get
     real (offset, confidence) values, not log text to parse).

A clip can produce 0+ face tracks (e.g. a cutaway with no face produces
none); results are returned per-track and aggregated by the caller.

Sign convention: SyncNet's own `offset` is frames, positive = audio leads
video. This module converts to this project's convention (positive ms =
audio lags video, matching `syncsentry.detectors.coarse_xcorr`) via
`offset_ms = -offset_frames * (1000 / frame_rate)`.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from syncsentry.detectors import coarse_xcorr
from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.lipsync import active_speaker
from syncsentry.util.ffmpeg_io import probe_duration_s

_SYNCNET_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "syncnet_python"
_MODEL_PATH = _SYNCNET_DIR / "data" / "syncnet_v2.model"

# SyncNet's raw confidence (median_dist - min_dist across the search window)
# is not on the [0, 1] scale used by the coarse detector's normalized
# correlation. Genuinely lip-sync-trackable real content scores roughly
# 3.5-8.5 here; well below that suggests a track without a clear,
# trackable talking face.
DEFAULT_SYNCNET_MIN_CONFIDENCE = 3.0

# Speed/quality trade-off, applied deliberately: S3FD face detection runs
# on every frame and is the dominant cost of this whole module for
# anything longer than a short clip. Detection is capped to a single
# representative window of this many seconds rather than scanning the
# whole asset. A global A/V offset (the only thing this detector looks
# for) is visible from any window that contains real, confidently
# trackable talking-head footage, so a representative sample is enough for
# the common case of one constant offset for the whole asset. This
# intentionally does not change behavior for clips already at or under
# this length, since detection already needs to work well on windows this
# size for short-clip robustness in the first place. Set to `None` (or a
# value >= the clip's real duration) to disable and scan the full clip.
#
# Overridable via SYNCSENTRY_MAX_ANALYZE_DURATION_S: S3FD's per-frame cost
# is roughly fixed on a given host, so wall-clock time for this stage
# scales close to linearly with how many seconds get analyzed. On a
# CPU-starved host (e.g. a 0.1 vCPU free-tier instance), lowering this is
# the most direct way to keep a single request's worst case bounded and
# predictable, at the cost of a smaller representative sample.
DEFAULT_MAX_ANALYZE_DURATION_S: float | None = float(
    os.environ.get("SYNCSENTRY_MAX_ANALYZE_DURATION_S", "20.0")
)

# Overridable via SYNCSENTRY_SYNCNET_TIMEOUT_S: a hard ceiling on the S3FD
# face-tracking subprocess (`_run_face_track_crop` below), the dominant
# cost of this whole module. `None` (default) means no limit, matching
# behavior before this existed. Sized generously on a constrained host
# rather than left unbounded, so a single slow request can't run for
# many minutes; on timeout this raises the same way S3FD failing outright
# does, so callers already handling `RuntimeError` here fall back to the
# coarse detector instead of the caller (or the end user) waiting
# indefinitely.
_SYNCNET_TIMEOUT_S_RAW = os.environ.get("SYNCSENTRY_SYNCNET_TIMEOUT_S")
SYNCNET_TIMEOUT_S: float | None = float(_SYNCNET_TIMEOUT_S_RAW) if _SYNCNET_TIMEOUT_S_RAW else None

# SyncNet's own sliding-window search only covers +-vshift frames (+-600ms
# at the vshift=15 default): any true offset larger than that is
# structurally unfindable no matter how good the model is, since the
# correct shift is never even in the candidate set it scores. Rather than
# raising vshift (which scales the CNN distance-matrix cost roughly
# linearly, for the entire clip, to cover a case that's rare), this uses
# the coarse cross-correlation detector, already in this codebase and
# orders of magnitude cheaper since it's plain signal cross-correlation
# rather than a CNN, as a cheap wide-range first pass. If it finds a
# confident offset bigger than this threshold, the audio is pre-shifted by
# that amount before SyncNet's normal small-vshift search runs, so the
# fine search only has to find a small residual near zero. Conditional,
# not unconditional: for the common in-range case this changes nothing
# beyond one cheap coarse-detector call, preserving existing behavior for
# small offsets exactly.
DEFAULT_RECENTER_THRESHOLD_MS = 400.0
DEFAULT_RECENTER_MIN_COARSE_CONFIDENCE = 0.15


class SyncNetUnavailable(RuntimeError):
    pass


# One cached, loaded SyncNetInstance per thread, not a single shared
# instance: `estimate_syncnet_tracks` can run concurrently across threads
# (piecewise refinement's `ThreadPoolExecutor`, see `piecewise_offset.py`),
# and nothing here has verified that `SyncNetInstance.evaluate()` is safe
# to call on the *same* instance from multiple threads at once. A
# thread-local avoids that question entirely: each thread gets its own
# instance, loaded once and reused for every later call on that thread,
# so a request that calls this twice (initial detection, then
# post-correction verification; see `pipeline.py::_detect_av_offset`) or
# a worker thread that gets reused across segments both skip the ~140MB
# weight-file reload and model re-init on every call after the first.
#
# This does mean peak memory can grow to (weights-per-instance x number
# of distinct threads that have ever called this), which matters on a
# low-RAM host: see `piecewise_offset.MAX_REFINE_WORKERS` (overridable
# via SYNCSENTRY_MAX_REFINE_WORKERS) for the knob that bounds it.
_thread_local = threading.local()


def _get_syncnet_instance() -> "SyncNetInstance":  # noqa: F821 - imported dynamically below
    cached = getattr(_thread_local, "instance", None)
    if cached is not None:
        return cached

    sys.path.insert(0, str(_SYNCNET_DIR))
    try:
        from SyncNetInstance import SyncNetInstance
    finally:
        sys.path.pop(0)

    instance = SyncNetInstance()
    instance.loadParameters(str(_MODEL_PATH))
    _thread_local.instance = instance
    return instance


def is_available() -> bool:
    return _SYNCNET_DIR.exists() and _MODEL_PATH.exists()


def _check_available() -> None:
    if not is_available():
        raise SyncNetUnavailable(
            f"SyncNet not found at {_SYNCNET_DIR}. Run "
            f"`bash analyzer/scripts/fetch_syncnet.sh` once to fetch the "
            f"(MIT-licensed) reference implementation and pretrained weights. "
            f"Never committed to this repo, see analyzer/third_party/SOURCES.md."
        )


@dataclass
class SyncNetWindowResult:
    """One short (~1s) sub-segment of a face track, scored independently.

    A whole face-track's confidence is a single median over every frame,
    including frames where the tracked person is listening rather than
    talking, which happens routinely in multi-speaker dialogue. Those
    frames dilute the aggregate. Per-window scoring lets the pipeline find
    and use only the sub-segments where this specific face's lip motion is
    actually correlated with the audio, i.e. where they're the one
    speaking.
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


@contextlib.contextmanager
def crop_face_tracks(video_path: str, reference: str = "clip",
                      max_analyze_duration_s: float | None = DEFAULT_MAX_ANALYZE_DURATION_S,
                      recenter_large_offsets: bool = True):
    """Run stage-1 face detection/tracking/crop once and yield
    `(data_dir, crop_files, pre_shift_ms)` for the duration of a `with`
    block. Factored out so a second detector
    (`lipsync/mtdvocalist_offset.py`) can reuse the exact same crop files
    as SyncNet instead of re-running face detection, the expensive part of
    this pipeline, a second time.

    `max_analyze_duration_s` (see `DEFAULT_MAX_ANALYZE_DURATION_S` above):
    if the source is longer than this, only a single centered window of
    this length is actually run through face detection. Centered rather
    than from the start, to avoid a disproportionate chance of landing on
    cold-open titles/logos/silence at the very start or credits at the
    very end. `None` disables capping (always scans the full clip).

    `recenter_large_offsets` (see `DEFAULT_RECENTER_THRESHOLD_MS` above):
    if the coarse detector finds a confident offset beyond that threshold,
    the audio actually analyzed is pre-shifted by that amount; callers
    must add the returned `pre_shift_ms` back on top of whatever the fine
    search finds on the (pre-shifted) crops to get the true total offset.
    `estimate_syncnet_tracks` below does this by folding it directly into
    each track/window's `offset_frames` right after `evaluate()`, so
    everything downstream of that point needs no further changes.
    """
    _check_available()
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        analyze_path = _maybe_trim_to_window(video_path, data_dir, max_analyze_duration_s)
        pre_shift_ms = 0.0
        if recenter_large_offsets:
            # `wide_seed_source_path=video_path` (the original, untrimmed
            # source, not `analyze_path`): the mouth-motion wide-range tier
            # needs as much signal duration as it can get, since
            # cross-correlation reliability scales with available signal
            # (see `wide_range_offset.py`); computing the same seed on a
            # short centered crop instead of the full clip risks locking
            # onto a materially different, wrong local optimum. The offset
            # itself is a global clip property, so measuring it on the full
            # source and applying the resulting shift to the (possibly
            # shorter) `analyze_path` is mathematically consistent and does
            # not reintroduce the cost `max_analyze_duration_s` exists to
            # bound, since only the cheap envelope/mouth-motion signals run
            # on the full clip, not face detection/tracking or the SyncNet
            # CNN.
            analyze_path, pre_shift_ms = _maybe_recenter_for_large_offset(
                analyze_path, data_dir, wide_seed_source_path=video_path,
            )
        _run_face_track_crop(analyze_path, data_dir, reference)
        crop_files = sorted((data_dir / "pycrop" / reference).glob("0*.avi"))
        yield data_dir, crop_files, pre_shift_ms


def _maybe_trim_to_window(video_path: str, data_dir: Path, max_duration_s: float | None) -> str:
    """Returns a path to face-detect on: `video_path` unchanged if it's
    already <= `max_duration_s` (or capping is disabled), otherwise a
    frame-accurate temp file holding just a centered `max_duration_s`-second
    window. See `crop_face_tracks` above for why.

    Deliberately does not use `-ss` (before `-i`) + `-c copy`: on files
    with sparse keyframes, stream-copy's keyframe-snapping can barely trim
    anything, defeating most of the intended savings for exactly the files
    this is supposed to help most. `-ss` placed after `-i` forces
    frame-accurate decoding instead of snapping to the nearest keyframe,
    and the output is re-encoded with a fast preset: this window is
    detection-only, never delivered to the user, so encode quality doesn't
    matter here, only that the audio/video timeline relationship it's
    re-encoded from is preserved exactly, which re-encoding does not
    disturb."""
    if max_duration_s is None:
        return video_path
    try:
        duration_s = probe_duration_s(video_path)
    except (ValueError, RuntimeError):
        return video_path  # couldn't probe: fail open, scan the whole clip
    if duration_s <= max_duration_s:
        return video_path

    start_s = max((duration_s - max_duration_s) / 2.0, 0.0)
    trimmed_path = data_dir / "analyze_window.mp4"
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


DEFAULT_RECENTER_MAX_HALF_DISAGREEMENT_MS = 250.0


def _maybe_recenter_for_large_offset(
        video_path: str, data_dir: Path,
        threshold_ms: float = DEFAULT_RECENTER_THRESHOLD_MS,
        min_coarse_confidence: float = DEFAULT_RECENTER_MIN_COARSE_CONFIDENCE,
        max_half_disagreement_ms: float = DEFAULT_RECENTER_MAX_HALF_DISAGREEMENT_MS,
        wide_seed_source_path: str | None = None,
) -> tuple[str, float]:
    """Wide-range pre-pass: finds a search center for SyncNet's fine search
    when the true offset is larger than its native +-600ms window (see
    `DEFAULT_RECENTER_THRESHOLD_MS` above for why that window exists at
    all). Two tiers, tried in order; both fail open (return the input
    unchanged) on any error, since the whole point is only to extend what
    SyncNet can find, never to interfere with the already-validated
    in-range case.

    Tier 1: mouth-motion wide-range seed (`wide_range_offset.py`),
    preferred whenever there's enough face-presence signal to attempt it.
    Carries no confidence gate of its own beyond "was there enough signal
    to compute anything", since a bad seed here fails safely downstream in
    SyncNet's own confidence gate rather than producing a confidently
    wrong result (see that module's docstring). This is the tier that
    recovers large real-world offsets that a whole-frame brightness
    detector cannot reliably seed.

    Tier 2: whole-frame brightness coarse detector plus a half-split
    self-consistency check, used as a fallback for content with no usable
    face-presence signal at all (pure screen-share, animation, off-screen
    narration), where mouth motion structurally cannot apply. The
    self-consistency check matters here specifically because whole-frame
    brightness is a much weaker, noisier signal than mouth motion (see
    `wide_range_offset.py`): without independent corroboration this tier
    can produce a confidently wrong recenter.
    """
    try:
        from syncsentry.lipsync.wide_range_offset import estimate_wide_range_offset
        wide = estimate_wide_range_offset(wide_seed_source_path or video_path)
        if abs(wide.offset_ms) >= threshold_ms:
            recentered_path = data_dir / "recentered.mp4"
            fix_av_offset(video_path, recentered_path, offset_ms=wide.offset_ms)
            return str(recentered_path), wide.offset_ms
        return video_path, 0.0  # enough face signal to judge, and it's genuinely in-range
    except (ValueError, RuntimeError):
        pass  # not enough face/mouth signal available: fall through to tier 2

    try:
        a, v, rate_hz, _ = coarse_xcorr.extract_normalized_envelopes(video_path)
        full = coarse_xcorr.estimate_offset_from_signals(a, v, rate_hz, search_window_ms=2500.0)
    except (ValueError, RuntimeError):
        return video_path, 0.0
    if abs(full.offset_ms) < threshold_ms or full.confidence < min_coarse_confidence:
        return video_path, 0.0

    mid = len(a) // 2
    try:
        half1 = coarse_xcorr.estimate_offset_from_signals(a[:mid], v[:mid], rate_hz, search_window_ms=2500.0)
        half2 = coarse_xcorr.estimate_offset_from_signals(a[mid:], v[mid:], rate_hz, search_window_ms=2500.0)
    except ValueError:
        return video_path, 0.0
    if abs(half1.offset_ms - half2.offset_ms) > max_half_disagreement_ms:
        return video_path, 0.0  # halves disagree: not a real consistent global offset

    recentered_path = data_dir / "recentered.mp4"
    try:
        fix_av_offset(video_path, recentered_path, offset_ms=full.offset_ms)
    except Exception:
        return video_path, 0.0
    return str(recentered_path), full.offset_ms


def _adaptive_min_track(video_path: str, frame_rate: float = 25.0) -> int:
    """Upstream `run_pipeline.py` hardcodes `--min_track 100` (frames, 4.0s
    @ 25fps): any face track shorter than that, or any scene shorter than
    that (also used to gate scene detection: a scene must be >= min_track
    frames before tracking is even attempted on it), is silently discarded
    before SyncNet ever sees it. That is a real problem for short clips: a
    3s clip (75 frames) can never clear a 100-frame minimum, and a 10s
    clip with a scene cut partway through can lose entirely if either half
    falls under 100 frames.

    Fix: scale `min_track` down for short input, floored at 20 frames
    (0.8s; below this there typically isn't enough signal left for
    SyncNet's own vshift=15 sliding window to produce a meaningful
    comparison) and capped at upstream's proven 100-frame default for
    anything already long enough to clear it. This is strictly a widening
    of what's accepted on short clips, never a change for already-working
    longer ones. The multiplier (`n_frames * 0.25`) is tuned to still widen
    tracking for clips with frequent scene cuts (e.g. a 10s clip cutting
    scenes every 2-3s) rather than clamping straight back to the unchanged
    100-frame default.
    """
    try:
        duration_s = probe_duration_s(video_path)
    except (ValueError, RuntimeError):
        return 100  # couldn't probe: fall back to upstream's own default, unchanged behavior
    n_frames = duration_s * frame_rate
    return max(20, min(100, int(n_frames * 0.25)))


def _run_face_track_crop(video_path: str, data_dir: Path, reference: str) -> None:
    """Stage 1: face detection + tracking + crop, via subprocess: run_pipeline.py
    calls `parser.parse_args()` at module scope, so it can't be imported as a
    library. The video path is resolved to absolute since we invoke it with a
    different cwd."""
    min_track = _adaptive_min_track(video_path)
    cmd = [
        sys.executable, "run_pipeline.py",
        "--videofile", str(Path(video_path).resolve()),
        "--reference", reference,
        "--min_track", str(min_track),
        "--data_dir", str(data_dir.resolve()),
        "--overwrite",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(_SYNCNET_DIR), capture_output=True, text=True,
                               timeout=SYNCNET_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"SyncNet face-tracking stage exceeded the {SYNCNET_TIMEOUT_S:.0f}s budget "
            f"(SYNCSENTRY_SYNCNET_TIMEOUT_S) and was killed; this host is likely too "
            f"CPU-constrained to run it on this clip in a reasonable time"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"SyncNet face-tracking stage failed:\n{proc.stderr[-4000:]}")


def _mouth_motion_speaking_mask(crop_file: Path, n_frames: int) -> np.ndarray:
    """Per-frame active-speaker mask for a face-track crop, from mouth
    motion alone: a cheap, fully automated but unimodal gate. A frame
    where this specific tracked face's mouth is barely moving relative to
    its own track is almost certainly a frame where they're listening, not
    talking. Kept as the fallback gate (`_active_speaker_mask` prefers real
    audio-visual ASD when available) since it needs nothing beyond
    SyncNet's own crop stage: no new face/landmark detection, no extra
    model.

    Not reused as an offset estimator itself; that role belongs to
    `lipsync/mouth_offset.py`. This only asks "is the mouth moving at all
    right now", a much easier and more robust question than "by how many
    ms is it offset from the audio".
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
        return np.ones(n_frames, dtype=bool)  # no signal either way: don't gate anything out

    motion = [0.0] + [
        float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
        for i in range(1, len(frames))
    ]
    motion = np.array(motion)
    if len(motion) < n_frames:
        motion = np.pad(motion, (0, n_frames - len(motion)), mode="edge")
    return motion[:n_frames] >= np.median(motion)


def _active_speaker_mask(crop_file: Path, n_frames: int, gate_mode: str) -> np.ndarray | None:
    """Per-frame active-speaker mask for one face-track crop, using
    whichever gate is actually available:

    - `"active_speaker"`: real audio-visual active-speaker detection
      (Light-ASD, `lipsync/active_speaker.py`). Jointly considers this
      face's motion and the audio, i.e. can tell "moving mouth but not
      talking" (laughing, reacting) apart from "actually the audio
      source", which mouth motion alone structurally cannot.
    - `"motion"`: the mouth-motion-only fallback.
    - `"none"`: no gating.
    - `"auto"` (default): active-speaker if `fetch_light_asd.sh` has been
      run, else motion, else none. Mirrors how `use_syncnet` itself falls
      back to the coarse detector rather than hard-failing.

    Returns `None` for `"none"` (or an unavailable explicit request), which
    callers treat as "don't gate this track at all".
    """
    if gate_mode == "none":
        return None
    if gate_mode == "auto":
        gate_mode = "active_speaker" if active_speaker.is_available() else "motion"

    if gate_mode == "active_speaker":
        if not active_speaker.is_available():
            raise active_speaker.ActiveSpeakerUnavailable(
                "gate_mode='active_speaker' requested but Light-ASD isn't fetched; "
                "run `bash analyzer/scripts/fetch_light_asd.sh`, or use gate_mode='motion'/'auto'."
            )
        return active_speaker.score_track(crop_file, n_frames=n_frames) >= 0.0
    if gate_mode == "motion":
        return _mouth_motion_speaking_mask(crop_file, n_frames=n_frames)
    raise ValueError(f"Unknown gate_mode: {gate_mode!r}")


def _windowed_offsets(dist: np.ndarray, vshift: int, track_index: int,
                       window_frames: int, step_frames: int,
                       speaking_mask: np.ndarray | None = None) -> list[SyncNetWindowResult]:
    """Slide a window over a track's per-frame x per-shift distance matrix
    (`dist`, shape [n_frames, 2*vshift+1], SyncNet's own raw output,
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
        # Active-speaker gate: skip windows where this track probably
        # isn't the audio's source right now. Majority vote over the
        # window rather than requiring every single frame, since even
        # genuine speech has brief pauses.
        if speaking_mask is not None:
            if float(np.mean(speaking_mask[start:start + window_frames])) < 0.5:
                continue
        mean_dist_per_shift = chunk.mean(axis=0)
        minidx = int(np.argmin(mean_dist_per_shift))
        offset_frames = vshift - minidx
        # A minimum at the very edge of the searched shift range is the
        # signature of a spurious, noise-dominated match (the "best" shift
        # is an artifact of the search boundary, not a real local minimum),
        # the same class of false-confidence failure seen in the coarse
        # detector. Exclude these outright rather than let a high `conf`
        # value (which a short, noisy window can produce just as easily at
        # the boundary as anywhere else) make it look trustworthy.
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
                             frame_rate: float = 25.0, gate_mode: str = "auto",
                             recenter_large_offsets: bool = True,
                             max_analyze_duration_s: float | None = DEFAULT_MAX_ANALYZE_DURATION_S,
                             ) -> list[SyncNetTrackResult]:
    """Run the full 2-stage SyncNet pipeline and return one result per
    detected face track (see module docstring), each carrying both a
    whole-track aggregate and a set of short-window sub-scores (see
    `_windowed_offsets`) for callers that want to discount
    listening-not-speaking frames rather than average over them.

    `gate_mode` (default `"auto"`): which active-speaker gate decides
    whether a window is even scored: `"active_speaker"` (real
    audio-visual ASD), `"motion"` (mouth-motion-only), `"none"` (no
    gating), or `"auto"` (prefers `"active_speaker"` if
    `fetch_light_asd.sh` has been run, else falls back to `"motion"`). See
    `_active_speaker_mask`.

    `recenter_large_offsets` (default `True`): forwarded to
    `crop_face_tracks`. Post-correction verification calls this with
    `False`: after a real correction has been applied, a large residual
    should not exist, so a verification pass should not need the
    wide-range seed running again. Concretely, `fix_av_offset`'s
    trim-based correction (see `fixer/av_fix.py`) leaves the corrected
    file's audio track shorter than its video track by exactly the
    corrected amount, since there is no more real source audio to fill
    that gap. That audio-shorter-than-video shape never occurs on an
    original, uncorrected asset, and can make the mouth-motion wide-range
    detector (tuned only on original assets) produce a spurious large seed
    on the corrected file. Disabling recentering for verification keeps
    the check to SyncNet's native +-600ms range, which a genuinely
    successful correction's residual should fall well inside of.

    `max_analyze_duration_s` (default `DEFAULT_MAX_ANALYZE_DURATION_S`):
    forwarded to `crop_face_tracks`. Callers that only need to confirm a
    small residual, rather than search for a possibly large offset (see
    `pipeline.py`'s post-correction verification call), can pass a smaller
    value: S3FD face detection (run once per analyzed frame) is this whole
    pipeline's dominant cost and scales close to linearly with frame
    count, so a smaller window meaningfully cuts verification's own cost
    without touching detection accuracy (still floored well above
    `_adaptive_min_track`'s own minimum for a trustworthy track).
    """
    _check_available()

    with crop_face_tracks(video_path, max_analyze_duration_s=max_analyze_duration_s,
                           recenter_large_offsets=recenter_large_offsets) as (
            data_dir, crop_files, pre_shift_ms):
        reference = "clip"
        if not crop_files:
            return []

        # Convert the coarse pre-shift (project ms convention, see
        # `crop_face_tracks`/`_maybe_recenter_for_large_offset` above) into
        # SyncNet's own raw frame convention so it can be added onto every
        # track/window's `offset_frames` below: everything downstream of
        # that point (aggregation, ms conversion, direction) then needs no
        # further changes. 0.0 in the untriggered, normal case, so this is
        # a no-op addition then.
        pre_shift_frames = -pre_shift_ms / (1000.0 / frame_rate)

        opt = argparse.Namespace(batch_size=20, vshift=vshift,
                                  tmp_dir=str(data_dir / "pytmp"), reference=reference)
        s = _get_syncnet_instance()

        window_frames = max(int(round(window_s * frame_rate)), 1)
        results = []
        for track_index, crop_file in enumerate(crop_files):
            offset, conf, dist = s.evaluate(opt, videofile=str(crop_file))

            speaking_mask = _active_speaker_mask(crop_file, n_frames=dist.shape[0], gate_mode=gate_mode)

            windows = _windowed_offsets(dist, vshift, track_index, window_frames=window_frames,
                                         step_frames=window_frames, speaking_mask=speaking_mask)
            if pre_shift_frames:
                windows = [
                    SyncNetWindowResult(track_index=w.track_index, frame_start=w.frame_start,
                                         offset_frames=w.offset_frames + round(pre_shift_frames),
                                         confidence=w.confidence)
                    for w in windows
                ]
            results.append(SyncNetTrackResult(
                offset_frames=int(offset) + round(pre_shift_frames), confidence=float(conf), n_frames=len(dist),
                windows=windows,
            ))
        return results


def _largest_agreeing_cluster(windows: list[SyncNetWindowResult],
                               tolerance_frames: int = 2) -> list[SyncNetWindowResult]:
    """Group windows whose offsets agree within `tolerance_frames` and
    return the largest such group.

    Individually confident windows can still be wrong: a short ~1s window
    is small enough that a spurious match can score just as high as a real
    one, and on real multi-speaker dialogue, individually confident
    windows can disagree with each other by hundreds of ms. Requiring
    several independent windows (different tracks, different points in
    time) to land on the same offset before trusting it is a much stronger
    check than any single window's confidence number, and is what
    distinguishes "one narrator, briefly silent in some windows" (should
    cluster tightly) from "multiple speakers, no single global offset any
    window can consistently see" (won't cluster at all).
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
                             gate_mode: str = "auto",
                             recenter_large_offsets: bool = True,
                             max_analyze_duration_s: float | None = DEFAULT_MAX_ANALYZE_DURATION_S,
                             ) -> SyncNetEstimate:
    """Run SyncNet and aggregate into a single estimate.

    Two aggregation paths, tried in order:

    1. Windowed evidence, gated on cross-window agreement (see
       `_windowed_offsets` + `_largest_agreeing_cluster`): pool every ~1s
       window from every face track, keep the ones that individually
       clear `min_window_confidence`, then require at least
       `min_cluster_size` of those to independently agree with each other
       (within `cluster_tolerance_frames`) before trusting them. This is
       what makes multi-speaker dialogue tractable: a track's
       non-speaking frames no longer drag down a whole-track average, and
       a single lucky or spurious confident window can no longer look like
       a real answer on its own; it has to be corroborated.
    2. Whole-track fallback: used whenever no cluster of agreeing
       confident windows is found, either because no window was
       individually confident, or because confident windows exist but
       contradict each other. Conservative by construction: on a single
       continuous narrator this fallback is close to whichever windowed
       answer would have been found anyway; on real multi-speaker dialogue
       it stays at the same low whole-track confidence rather than being
       fooled into false confidence by windowing.
    """
    tracks = estimate_syncnet_tracks(video_path, vshift=vshift, window_s=window_s, frame_rate=frame_rate,
                                      gate_mode=gate_mode, recenter_large_offsets=recenter_large_offsets,
                                      max_analyze_duration_s=max_analyze_duration_s)
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

    offset_ms = -offset_frames * (1000.0 / frame_rate)  # sign flip: see module docstring

    if abs(offset_ms) <= in_sync_threshold_ms:
        direction = "in_sync"
    elif offset_ms > 0:
        direction = "audio_lags"
    else:
        direction = "audio_leads"

    return SyncNetEstimate(offset_ms=offset_ms, confidence=confidence, direction=direction,
                            n_tracks=len(tracks), n_confident_windows=n_confident_windows)
