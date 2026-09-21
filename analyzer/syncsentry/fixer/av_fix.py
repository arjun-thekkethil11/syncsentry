"""Apply a correction for a detected global A/V offset.

Why this isn't just `-itsoffset` on a stream-copy remux: `-itsoffset`
combined with `-c copy` only rewrites container-level timestamp metadata.
It does not insert or remove any actual samples/frames. Plenty of real
consumers, including this project's own frame-index/sample-index-based
signal extraction in `syncsentry.util.ffmpeg_io` and quite a few real
players, read the raw decoded sample/frame sequence and never look at
that metadata, so a metadata-only "fix" does nothing for them.

The correction here instead re-encodes the audio track with a filter that
actually adds or removes samples, leaving video untouched (stream-copied):

  * offset_ms > 0 (audio lags video, so audio needs to happen earlier):
    trim `offset_ms` of real audio off the start of the track. This loses
    a sliver of lead-in audio, which is the standard, accepted trade-off
    production sync-correction tools make for small (sub-second) offsets.
  * offset_ms < 0 (audio leads video, so audio needs to happen later):
    pad the start of the audio track with `|offset_ms|` of real silence.

Tail padding: trimming samples off the start of the audio necessarily
makes the audio track shorter than the (untouched) video track by
exactly `offset_ms`, since there is no more real source audio to fill
that gap. For small (sub-second) offsets this is a barely perceptible
sliver, but for larger corrections a multi-second silent tail at the end
of a corrected video is a noticeable defect. Explicitly padding the tail
with silence (via `apad` plus forcing output duration to match the video
with `-t`) doesn't recover any real audio, since none exists, but makes
the corrected file's audio track cover its full nominal duration
explicitly and deterministically rather than leaving an implicit,
player-dependent gap. This also matters for anything downstream that
assumes audio duration is approximately video duration (e.g. this
project's own post-correction verification re-measurement; see
`pipeline.py` / `syncnet_offset.py`'s `recenter_large_offsets` parameter).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from syncsentry.util.ffmpeg_io import _require_binary, probe_duration_s, run


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
    try:
        video_duration_s = probe_duration_s(video_path)
    except (ValueError, RuntimeError):
        video_duration_s = None  # fail open: skip explicit tail padding below

    if offset_ms > 0:
        # Audio lags, so it needs to happen earlier: trim real samples off
        # the start. This leaves a real gap of exactly `offset_s` at the
        # end (no more source audio exists), so explicitly pad it with
        # silence up to the original video duration instead of leaving it
        # implicit (see module docstring). `apad` alone pads indefinitely,
        # so the `-t` below is what actually bounds it back to the real
        # video length.
        audio_filter = f"atrim=start={offset_s},asetpts=PTS-STARTPTS,apad"
        method = "trim_audio_start"
    else:
        # Audio leads, so it needs to happen later: pad the start with
        # real silence. This makes the audio track longer than before by
        # `offset_s` (harmless, since most players simply stop at video
        # end), but cap it back to the video's duration too, for a clean,
        # predictable output rather than a dangling audio tail past the
        # video's own end.
        ms = int(round(offset_s * 1000))
        audio_filter = f"adelay={ms}|{ms}"
        method = "pad_audio_start"

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", video_path,
        "-c:v", "copy",
        # AAC, not pcm_s16le: the audio filter forces a full decode and
        # re-encode regardless of codec, so there is no bit-exact PCM to
        # preserve here anyway, and AAC is the one audio codec that muxes
        # cleanly into every container this pipeline actually receives.
        # Raw PCM works fine in .mkv (why this never surfaced against the
        # synthetic .mkv test fixtures) but ffmpeg's mp4 muxer has no tag
        # for it at all and fails outright on real .mp4 uploads, the
        # overwhelmingly common real-world case.
        "-af", audio_filter, "-c:a", "aac", "-b:a", "192k",
    ]
    if video_duration_s is not None:
        # +0.05s safety margin: `-t` bounds the *whole* output (video
        # included, even though it's stream-copied), so setting it to
        # exactly the probed duration risks clipping the last video frame
        # on any float-precision rounding between what ffprobe reported and
        # what ffmpeg's own internal timestamp arithmetic computes. A few
        # extra milliseconds of trailing silence is harmless; a dropped
        # final video frame is not.
        cmd += ["-t", f"{video_duration_s + 0.05:.3f}"]
    cmd.append(out_path)
    run(cmd)
    return AVFixResult(video_path, out_path, offset_ms, method)
