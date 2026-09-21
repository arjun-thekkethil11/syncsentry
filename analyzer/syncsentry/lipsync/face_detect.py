"""Real face detection via OpenCV's YuNet (opencv_zoo), sampled over time.

Why YuNet and not mediapipe's Tasks-API face detector, which is the more
"expected" choice for this kind of work: mediapipe's modern Tasks API
initializes a Metal-backed image pipeline on macOS even when a CPU
delegate is explicitly requested, and that crashes with `Check failed:
service_ Service is unavailable` in this project's execution environment.
Rather than fight a GPU-service dependency for what is fundamentally a
CPU-sized model, this module uses `cv2.FaceDetectorYN`, a small (~230KB)
ONNX model that runs through OpenCV's own DNN backend with no GPU
dependency at all. Same job (real face detection on real video), no
platform-specific crash.

Model provenance: `face_detection_yunet_2023mar.onnx` from
https://github.com/opencv/opencv_zoo (Apache-2.0), committed directly into
this repo since it's a small, redistributable model artifact (unlike the
third-party video clips under `analyzer/fixtures/real_content/`, which are
never committed; see that directory's `SOURCES.md`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

_MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceFrame:
    t_s: float
    face_present: bool
    confidence: float | None  # None when face_present is False


def _load_detector(score_threshold: float, input_size: tuple[int, int]) -> cv2.FaceDetectorYN:
    if not _MODEL_PATH.exists():
        raise RuntimeError(
            f"Face detector model not found at {_MODEL_PATH}. It should be committed to the "
            f"repo; if missing, re-fetch it from https://github.com/opencv/opencv_zoo "
            f"(models/face_detection_yunet/face_detection_yunet_2023mar.onnx)."
        )
    return cv2.FaceDetectorYN.create(
        str(_MODEL_PATH), "", input_size, score_threshold=score_threshold,
    )


def detect_face_presence(video_path: str, sample_fps: float = 2.0,
                          score_threshold: float = 0.6) -> list[FaceFrame]:
    """Sample `video_path` at `sample_fps` and run YuNet face detection on each
    sampled frame. Sampling (rather than every frame) keeps this cheap enough
    to run over a whole title: face presence changes on the timescale of
    shots (seconds), not individual frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, int(round(fps / sample_fps)))

    detector = None
    frames: list[FaceFrame] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every == 0:
            h, w = frame.shape[:2]
            if detector is None:
                detector = _load_detector(score_threshold, (w, h))
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            found = faces is not None and len(faces) > 0
            confidence = float(faces[0][-1]) if found else None
            frames.append(FaceFrame(t_s=idx / fps, face_present=found, confidence=confidence))
        idx += 1

    cap.release()
    return frames
