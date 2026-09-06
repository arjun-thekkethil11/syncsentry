"""Apply a correction for a detected global A/V offset.

Why this isn't just `-itsoffset` on a stream-copy remux: `-itsoffset`
combined with `-c copy` only rewrites *container-level timestamp metadata*.
It does not insert or remove any actual samples/frames. Plenty of real
consumers -- including our own frame-index/sample-index-based signal
extraction in `syncsentry.util.ffmpeg_io`, and quite a few real players --
read the raw decoded sample/frame sequence and never look at that metadata,
so a metadata-only "fix" silently does nothing for them. This was verified
empirically while building this module: an `-itsoffset -0.15` remux left the
detected offset completely unchanged.

The correction here instead re-encodes the audio track with a filter that
actually adds or removes samples, leaving video untouched (stream-copied):

  * offset_ms > 0 (audio lags video -> audio needs to happen earlier):
    trim `offset_ms` of real audio off the *start* of the track. This loses
    a sliver of lead-in audio, which is the standard, accepted trade-off
    production sync-correction tools make for small (sub-second) offsets.
  * offset_ms < 0 (audio leads video -> audio needs to happen later):
    pad the start of the audio track with `|offset_ms|` of real silence.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from syncsentry.util.ffmpeg_io import _require_binary, run


@dataclass
class AVFixResult:
    input_path: str
    output_path: str
    applied_offset_ms: float
    method: str  # "trim_audio_start" | "pad_audio_start" | "no_change"


def fix_av_offset(video_path: str | Path, out_path: str | Path, offset_ms: float,
                   min_fixable_ms: float = 5.0) -> AVFixResult:
    """Correct a detected global A/V offset by trimming or padding real audio samples."""
    ffmpeg = _require_binary("ffmpeg")
    video_path, out_path = str(video_path), str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if abs(offset_ms) < min_fixable_ms:
        run([ffmpeg, "-y", "-loglevel", "error", "-i", video_path, "-c", "copy", out_path])
        return AVFixResult(video_path, out_path, 0.0, "no_change")

    offset_s = abs(offset_ms) / 1000.0
    if offset_ms > 0:
        # Audio lags -> needs to happen earlier -> trim real samples off the start.
        audio_filter = f"atrim=start={offset_s},asetpts=PTS-STARTPTS"
        method = "trim_audio_start"
    else:
        # Audio leads -> needs to happen later -> pad with real silence.
        ms = int(round(offset_s * 1000))
        audio_filter = f"adelay={ms}|{ms}"
        method = "pad_audio_start"

    run([
        ffmpeg, "-y", "-loglevel", "error",
        "-i", video_path,
        "-c:v", "copy",
        "-af", audio_filter, "-c:a", "pcm_s16le",
        out_path,
    ])
    return AVFixResult(video_path, out_path, offset_ms, method)
