"""Thin wrappers around ffmpeg/ffprobe + signal extraction helpers.

These functions turn a media file into two time-aligned 1-D signals:

  * an *audio envelope* (short-time RMS energy), and
  * a *video brightness envelope* (mean luma per frame).

Both are the raw material the detectors in ``syncsentry.detectors`` correlate
against each other (or against caption timing) to estimate drift.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
import soundfile as sf


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found on PATH. Install ffmpeg (e.g. `brew install ffmpeg`)."
        )
    return path


def run(args: list[str]) -> None:
    """Run a subprocess command, raising with captured stderr on failure."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({args[0]}): {' '.join(args)}\n--- stderr ---\n{proc.stderr}"
        )


@dataclass
class Signal:
    times_s: np.ndarray  # seconds, monotonically increasing
    values: np.ndarray  # same length as times_s


def extract_audio_envelope(video_path: str, sr: int = 48000, frame_ms: float = 10.0,
                            hop_ms: float = 5.0) -> Signal:
    """Extract mono audio and compute a short-time RMS envelope.

    frame_ms/hop_ms control the temporal resolution of the envelope; 5ms hop
    gives 200 Hz effective sample rate, plenty for detecting sync drift down
    to a few milliseconds once cross-correlated.
    """
    ffmpeg = _require_binary("ffmpeg")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio.wav")
        run([
            ffmpeg, "-y", "-loglevel", "error",
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", str(sr),
            wav_path,
        ])
        audio, file_sr = sf.read(wav_path, dtype="float32", always_2d=False)

    assert file_sr == sr
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))

    n_frames = max(0, (len(audio) - frame_len) // hop_len + 1)
    if n_frames <= 0:
        return Signal(np.array([]), np.array([]))

    env = np.empty(n_frames, dtype=np.float64)
    times = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        chunk = audio[start:start + frame_len]
        env[i] = np.sqrt(np.mean(np.square(chunk)) + 1e-12)
        times[i] = (start + frame_len / 2.0) / sr

    return Signal(times, env)


def extract_video_brightness(video_path: str) -> Signal:
    """Read every frame and compute mean grayscale intensity, timestamped by fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    values = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values.append(float(np.mean(gray)))
        idx += 1
    cap.release()

    values = np.asarray(values, dtype=np.float64)
    times = (np.arange(len(values), dtype=np.float64) + 0.5) / fps
    return Signal(times, values)


def probe_duration_s(video_path: str) -> float:
    ffprobe = _require_binary("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(proc.stdout.strip())
