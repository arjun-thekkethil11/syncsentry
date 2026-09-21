"""Synthetic ground-truth fixture generator.

Published A/V-sync papers (DiVAS, SyncNet, ModEFormer, UniSync) validate
offset-recovery accuracy by injecting a known offset into clean, in-sync
material and checking whether the detector recovers it. This module does
the same with a fully synthetic, license-free "flash + beep"
clapperboard-style pattern instead of real footage, so the benchmark
harness has zero dependency on copyrighted or restricted-access datasets
(LRS2/LRS3 etc. require request-based access) and is 100% reproducible in
CI.

Video: a white full-frame flash for `pulse_ms` every `period_s`, on a black
background, rendered at `fps`.

Audio: a `tone_hz` sine burst for `pulse_ms` every `period_s`, gated on top
of silence at sample rate `sr`.

`offset_ms` shifts the *audio* pulses relative to the video pulses:
  offset_ms > 0  ->  audio lags video   (audio event happens later)
  offset_ms < 0  ->  audio leads video  (audio event happens earlier)

This mirrors the sign convention used by `syncsentry.detectors.coarse_xcorr`.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from syncsentry.util.ffmpeg_io import run, _require_binary


@dataclass
class FixtureSpec:
    duration_s: float = 12.0
    period_s: float = 1.0
    pulse_ms: float = 80.0
    offset_ms: float = 0.0
    fps: float = 25.0
    sr: int = 48000
    tone_hz: float = 1000.0
    width: int = 320
    height: int = 240


def _synthesize_audio(spec: FixtureSpec) -> np.ndarray:
    """Sample-accurate gated sine burst (numpy), avoiding ffmpeg's frame-block
    quantized filter evaluation and any lossy-codec encoder delay, both of
    which introduce several-millisecond, slightly non-constant timing error.
    """
    n = int(round(spec.duration_s * spec.sr))
    t = np.arange(n, dtype=np.float64) / spec.sr
    period, pulse_s, offset_s = spec.period_s, spec.pulse_ms / 1000.0, spec.offset_ms / 1000.0
    phase = np.mod(t - offset_s, period)
    gate = (phase < pulse_s).astype(np.float64)
    tone = np.sin(2.0 * np.pi * spec.tone_hz * t)
    return (gate * tone).astype(np.float32)


def generate_fixture(out_path: str | Path, spec: FixtureSpec) -> Path:
    """Render a synthetic AV clip with a known, injected sync offset.

    Video pulses are rendered with ffmpeg's `drawbox` (frame-exact for
    constant frame rate output). Audio pulses are synthesized sample-exactly
    in numpy and muxed as lossless PCM to avoid quantization/encoder-delay
    error on the audio side.
    """
    ffmpeg = _require_binary("ffmpeg")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".mkv":
        raise ValueError("generate_fixture expects an .mkv output path (PCM + H.264 container).")

    video_gate = f"lt(mod(t,{spec.period_s}),{spec.pulse_ms / 1000.0})"

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        sf.write(str(wav_path), _synthesize_audio(spec), spec.sr, subtype="PCM_16")

        args = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"color=c=black:s={spec.width}x{spec.height}:r={spec.fps}:d={spec.duration_s}",
            "-i", str(wav_path),
            "-filter_complex",
            f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{video_gate}'[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le",
            "-shortest",
            str(out_path),
        ]
        run(args)
    return out_path


def _synthesize_audio_piecewise(duration_s: float, sr: int, period_s: float, pulse_ms: float,
                                 tone_hz: float, control_points: list[tuple[float, float]]) -> np.ndarray:
    """Same gated-sine-burst synthesis as `_synthesize_audio`, but the A/V
    offset at time t is linearly interpolated between `control_points`
    (t_seconds, offset_ms) instead of held constant. This is what lets us
    build synthetic "drift-early", "drift-late", and "intermittent" titles
    for testing `syncsentry.lipsync.title_drift`, using the same
    ground-truth-injection methodology as the rest of this project, just
    with a time-varying offset instead of a single number.
    """
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr
    cp_t = np.array([c[0] for c in control_points])
    cp_off_s = np.array([c[1] / 1000.0 for c in control_points])
    offset_at_t = np.interp(t, cp_t, cp_off_s)

    pulse_s = pulse_ms / 1000.0
    phase = np.mod(t - offset_at_t, period_s)
    gate = (phase < pulse_s).astype(np.float64)
    tone = np.sin(2.0 * np.pi * tone_hz * t)
    return (gate * tone).astype(np.float32)


def generate_piecewise_offset_fixture(out_path: str | Path, duration_s: float,
                                       control_points: list[tuple[float, float]],
                                       period_s: float = 2.0, pulse_ms: float = 80.0,
                                       fps: float = 25.0, sr: int = 48000,
                                       tone_hz: float = 1000.0,
                                       width: int = 320, height: int = 240) -> Path:
    """Like `generate_fixture`, but the injected A/V offset varies over time
    per `control_points` (a list of (t_seconds, offset_ms) waypoints,
    linearly interpolated). Video pulses stay at the fixed reference
    (mod(t, period) < pulse_s); only the audio pulses' timing follows the
    offset schedule.
    """
    ffmpeg = _require_binary("ffmpeg")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".mkv":
        raise ValueError("generate_piecewise_offset_fixture expects an .mkv output path.")

    video_gate = f"lt(mod(t,{period_s}),{pulse_ms / 1000.0})"

    with tempfile.TemporaryDirectory() as td:
        wav_path = Path(td) / "audio.wav"
        audio = _synthesize_audio_piecewise(duration_s, sr, period_s, pulse_ms, tone_hz, control_points)
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")

        args = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}:d={duration_s}",
            "-i", str(wav_path),
            "-filter_complex",
            f"[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='{video_gate}'[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le",
            "-shortest",
            str(out_path),
        ]
        run(args)
    return out_path


def generate_captions(out_vtt: str | Path, spec: FixtureSpec, caption_offset_ms: float = 0.0,
                       cue_len_ms: float = 700.0) -> Path:
    """Write a WebVTT file with one cue per audio pulse, optionally drifted.

    caption_offset_ms shifts caption cue *start* times relative to the true
    audio pulse times (the ground truth "speech" proxy), independent of any
    video offset. This models the real-world failure mode: captions are
    authored/retimed against the wrong reference track.

    Cue generation starts at k=1 (skipping the pulse at t=0), not k=0. WebVTT
    timestamps can't be negative, so a large negative caption_offset_ms would
    otherwise silently clamp the very first cue's ground-truth start time to
    0.000, corrupting that one cue's ground truth without erroring. Starting
    at k=1 gives every cue a full `period_s` of head-room, comfortably more
    than the +/-300ms offsets exercised in this project's tests/benchmarks.
    """
    out_vtt = Path(out_vtt)
    out_vtt.parent.mkdir(parents=True, exist_ok=True)

    def fmt(t: float) -> str:
        if t < 0.0:
            raise ValueError(
                f"caption cue start {t:.3f}s is negative, increase period_s or reduce "
                f"caption_offset_ms so every cue keeps positive head-room."
            )
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    n_pulses = int(spec.duration_s // spec.period_s)
    lines = ["WEBVTT", ""]
    for k in range(1, n_pulses):
        true_start = k * spec.period_s
        start = true_start + caption_offset_ms / 1000.0
        end = start + cue_len_ms / 1000.0
        lines.append(str(k + 1))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(f"BEEP {k}")
        lines.append("")

    out_vtt.write_text("\n".join(lines), encoding="utf-8")
    return out_vtt
