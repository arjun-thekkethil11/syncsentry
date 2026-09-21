"""Real speech-activity detection via Silero VAD.

The caption-drift checker (`syncsentry.captions.drift_check`) uses a simple
energy-threshold onset picker, which is deliberately not a speech
detector: it's tuned to find the rising edge of a synthetic tone pulse in
generated fixtures. Real captions are timed against real speech, which
doesn't look like a clean tone pulse, so this module uses an actual
voice-activity detector. Silero VAD is a small (~2MB) pretrained model
shipped as a pip package (`silero-vad`) with the model weights bundled: no
separate download step, unlike the face detector's ONNX file.

Implementation note: `silero_vad.read_audio` calls `torchaudio.load`, which
in torchaudio>=2.9 requires the separate `torchcodec` package for file I/O.
Rather than add another heavy dependency, this module decodes audio via
ffmpeg directly (reusing the same subprocess pattern as
`syncsentry.util.ffmpeg_io`) and hands Silero VAD a raw waveform tensor,
sidestepping torchaudio's I/O path entirely.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from syncsentry.util.ffmpeg_io import _require_binary

SAMPLE_RATE = 16000  # Silero VAD's expected input rate


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float


@lru_cache(maxsize=1)
def _get_model():
    from silero_vad import load_silero_vad
    return load_silero_vad()


def _load_wav_16k(path: str) -> torch.Tensor:
    ffmpeg = _require_binary("ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio from {path}: {proc.stderr.decode()}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(pcm)


def detect_speech_segments(video_or_audio_path: str, **vad_kwargs) -> list[SpeechSegment]:
    """Run Silero VAD over `video_or_audio_path` and return speech segments.

    `vad_kwargs` are forwarded to `silero_vad.get_speech_timestamps` (e.g.
    `min_speech_duration_ms`, `min_silence_duration_ms`) for callers that
    need to tune sensitivity for a particular corpus.
    """
    from silero_vad import get_speech_timestamps

    wav = _load_wav_16k(video_or_audio_path)
    model = _get_model()
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=SAMPLE_RATE, return_seconds=True, **vad_kwargs,
    )
    return [SpeechSegment(start_s=float(seg["start"]), end_s=float(seg["end"])) for seg in timestamps]
