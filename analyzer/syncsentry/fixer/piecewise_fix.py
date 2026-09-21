"""Applies a piecewise (per-time-segment) A/V offset correction in a
single ffmpeg pass, without re-encoding the video track at all.

Generalizes `av_fix.fix_av_offset`'s "trim-or-pad this track, then force
the result back to a fixed duration" trick (see that module's docstring
for why a metadata-only `-itsoffset` remux doesn't work at all) to a
sequence of independently-shifted audio segments, stitched back together
with ffmpeg's own `concat` audio filter inside one `-filter_complex`
graph. This avoids physically cutting the source into N pieces,
re-encoding video N times, and re-concatenating, which would also work
but wastes the one thing `fix_av_offset` already established doesn't
need to change for a correction like this: the video stream. `-c:v copy`
is used here too, for the entire output, exactly as in the
single-global-offset case: this is a strictly more general version of
the same fix, not a different, lossier one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from syncsentry.util.ffmpeg_io import _require_binary, probe_duration_s, run

# Matches `av_fix.fix_av_offset`'s own `min_fixable_ms` default. Below
# this, trimming/padding real audio samples isn't worth the risk of an
# off-by-one-sample edge case for a correction nobody would perceive
# anyway.
MIN_FIXABLE_MS = 5.0


@dataclass
class PiecewiseFixResult:
    input_path: str
    output_path: str
    n_segments: int
    n_shifted: int


def _segment_filter(idx: int, start_s: float, end_s: float, offset_ms: float | None) -> tuple[str, str, bool]:
    """One segment's ffmpeg audio-filter expression, reading directly from
    `[0:a]` (the original, undecoded-by-us audio stream; ffmpeg's filter
    graph lets many filters fan out from the same input label, one
    `atrim` per segment, each independently slicing the one decoded
    stream) and writing to its own labelled output `[a{idx}]`.

    Returns `(filter_expr, label, shifted)`. `shifted=False` means this
    segment's audio passes through unmodified: either it was never meant
    to be corrected (`offset_ms` is `None` or below `MIN_FIXABLE_MS`), or
    it's too short to safely absorb its own offset (see the `>=
    duration_s` guard below, which fails open rather than risking
    trimming/padding past this segment's own bounds).
    """
    label = f"a{idx}"
    duration_s = max(0.0, end_s - start_s)
    shifted = False

    if offset_ms is not None and duration_s > 0.0 and abs(offset_ms) >= MIN_FIXABLE_MS:
        offset_s = abs(offset_ms) / 1000.0
        if offset_s < duration_s:
            shifted = True
            if offset_ms > 0:
                # Audio lags: trim real samples off this segment's start,
                # then pad the resulting gap at this segment's end with
                # silence so the piece's own duration still comes out to
                # exactly `duration_s` (scoped-down version of
                # `fix_av_offset`'s trim-start plus apad-to-duration trick).
                f = (f"[0:a]atrim=start={start_s + offset_s:.3f}:end={end_s:.3f},"
                     f"asetpts=PTS-STARTPTS,apad,atrim=end={duration_s:.3f}[{label}]")
            else:
                # Audio leads: trim this segment's own tail short by
                # `offset_s` first, then pad that same amount of real
                # silence onto its start. Net effect: `duration_s` total,
                # content delayed by `offset_s` within this segment only.
                ms = int(round(offset_s * 1000))
                f = (f"[0:a]atrim=start={start_s:.3f}:end={end_s - offset_s:.3f},"
                     f"asetpts=PTS-STARTPTS,adelay={ms}|{ms}[{label}]")

    if not shifted:
        f = f"[0:a]atrim=start={start_s:.3f}:end={end_s:.3f},asetpts=PTS-STARTPTS[{label}]"

    return f, label, shifted


def fix_piecewise_offsets(video_path: str | Path, out_path: str | Path, segments) -> PiecewiseFixResult:
    """`segments`: time-ordered objects covering `[0, video_duration]`
    contiguously with no gaps or overlaps (as produced by
    `piecewise_offset.detect_piecewise_offsets`), each with `.start_s`,
    `.end_s`, and `.offset_ms` (`None` means "leave this span's audio
    untouched", e.g. an "undetermined" segment).

    Produces one output file spanning the same timeline: video stream
    byte-identical (`-c:v copy`, never re-encoded, never re-cut), audio
    independently corrected per segment and stitched back together in one
    ffmpeg filter graph, with no intermediate per-segment files and no
    re-encoding the video N times.
    """
    ffmpeg = _require_binary("ffmpeg")
    video_path, out_path = str(video_path), str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    filters: list[str] = []
    labels: list[str] = []
    n_shifted = 0
    for i, seg in enumerate(segments):
        f, label, shifted = _segment_filter(i, seg.start_s, seg.end_s, seg.offset_ms)
        filters.append(f)
        labels.append(f"[{label}]")
        if shifted:
            n_shifted += 1

    concat_filter = f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[aout]"
    filter_complex = ";".join(filters) + ";" + concat_filter

    try:
        video_duration_s = probe_duration_s(video_path)
    except (ValueError, RuntimeError):
        video_duration_s = None  # fail open: skip explicit duration bounding below

    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[aout]",
        # AAC, not pcm_s16le: the concat filter graph forces a full audio
        # decode and re-encode regardless of codec, so there's no
        # bit-exact PCM to preserve, and ffmpeg's mp4 muxer has no tag for
        # raw PCM at all (works in .mkv, which is why this never surfaced
        # against the synthetic test fixtures, but fails outright on a
        # real .mp4 upload). AAC muxes cleanly into every container this
        # pipeline actually receives.
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    ]
    if video_duration_s is not None:
        # Same +0.05s safety margin as `fix_av_offset` (see that module's
        # docstring for why): bounds the whole output including the
        # stream-copied video. A few extra ms of trailing silence is
        # harmless; a dropped final video frame is not.
        cmd += ["-t", f"{video_duration_s + 0.05:.3f}"]
    cmd.append(out_path)
    run(cmd)

    return PiecewiseFixResult(video_path, out_path, len(segments), n_shifted)
