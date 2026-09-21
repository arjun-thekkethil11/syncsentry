"""Real dialogue-scene detection: face on screen AND speech active, overlapping in time.

`scene_offsets.py` windows a title into fixed-length chunks regardless of
content. A real "dialogue scene" is a chunk of time where there's an
actual face on screen and someone is actually talking, the two conditions
DiVAS (CVPR 2024) requires before trusting a per-scene lip-sync estimate
at all.

Pipeline: `face_detect.detect_face_presence` (YuNet, sampled) gives a
face-presence timeline; `vad.detect_speech_segments` (Silero VAD) gives
speech intervals; this module intersects the two, then merges small gaps
and drops scenes shorter than `min_duration_s` so a single dropped frame
or a half-second pause in speech doesn't fragment one real scene into many
tiny ones.

This module restricts *where* a per-scene estimator runs to scenes with an
actual talking face; swapping `scene_offsets.py`'s estimator itself for a
learned lip-sync embedding model is separate, further work.
"""
from __future__ import annotations

from dataclasses import dataclass

from syncsentry.lipsync.face_detect import detect_face_presence
from syncsentry.lipsync.vad import detect_speech_segments

Interval = tuple[float, float]


@dataclass
class DialogueScene:
    start_s: float
    end_s: float


def _face_frames_to_intervals(frames, sample_dt: float) -> list[Interval]:
    """Turn a list of per-sample FaceFrame into merged (start, end) intervals
    covering every sample where a face was present, treating each sample as
    representative of [t, t + sample_dt)."""
    intervals: list[Interval] = []
    cur_start: float | None = None
    prev_t: float | None = None
    for f in frames:
        if f.face_present:
            if cur_start is None:
                cur_start = f.t_s
            prev_t = f.t_s
        else:
            if cur_start is not None:
                intervals.append((cur_start, prev_t + sample_dt))
                cur_start = None
    if cur_start is not None:
        intervals.append((cur_start, prev_t + sample_dt))
    return intervals


def _intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Intersection of two lists of disjoint, sorted (start, end) intervals."""
    result: list[Interval] = []
    i, j = 0, 0
    a = sorted(a)
    b = sorted(b)
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            result.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _merge_close(intervals: list[Interval], max_gap_s: float) -> list[Interval]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= max_gap_s:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_dialogue_scenes(video_path: str, sample_fps: float = 2.0,
                            face_score_threshold: float = 0.6,
                            merge_gap_s: float = 0.5,
                            min_duration_s: float = 1.0) -> list[DialogueScene]:
    """Detect real dialogue scenes: time ranges where a face is on screen AND
    speech is active, merged across small gaps and filtered to a minimum
    duration.
    """
    face_frames = detect_face_presence(video_path, sample_fps=sample_fps,
                                        score_threshold=face_score_threshold)
    face_intervals = _face_frames_to_intervals(face_frames, sample_dt=1.0 / sample_fps)

    speech_segments = detect_speech_segments(video_path)
    speech_intervals = [(s.start_s, s.end_s) for s in speech_segments]

    overlap = _intersect(face_intervals, speech_intervals)
    merged = _merge_close(overlap, merge_gap_s)
    filtered = [(s, e) for s, e in merged if (e - s) >= min_duration_s]

    return [DialogueScene(start_s=s, end_s=e) for s, e in filtered]
