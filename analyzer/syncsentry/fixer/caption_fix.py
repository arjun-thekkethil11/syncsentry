"""Apply a correction for detected caption-vs-speech drift.

Unlike the A/V fix, this doesn't touch any media samples: captions are
just timestamped text, so "fixing" drift means rewriting every cue's start
and end time by the detected offset. If captions start `offset_ms` *after*
the speech they belong to (a positive offset per
`syncsentry.captions.drift_check`), we shift every cue `offset_ms` earlier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import webvtt


@dataclass
class CaptionFixResult:
    input_path: str
    output_path: str
    applied_offset_ms: float
    cues_shifted: int


def _fmt(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _parse(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def fix_caption_offset(vtt_path: str | Path, out_path: str | Path, offset_ms: float,
                        min_fixable_ms: float = 5.0) -> CaptionFixResult:
    vtt_path, out_path = str(vtt_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    captions = list(webvtt.read(vtt_path))

    if abs(offset_ms) < min_fixable_ms:
        out_path.write_text(Path(vtt_path).read_text(encoding="utf-8"), encoding="utf-8")
        return CaptionFixResult(vtt_path, str(out_path), 0.0, 0)

    shift_s = -offset_ms / 1000.0  # cues are `offset_ms` late -> shift earlier
    lines = ["WEBVTT", ""]
    for i, cue in enumerate(captions):
        new_start = _parse(cue.start) + shift_s
        new_end = _parse(cue.end) + shift_s
        lines.append(str(i + 1))
        lines.append(f"{_fmt(new_start)} --> {_fmt(new_end)}")
        lines.append(cue.text)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return CaptionFixResult(vtt_path, str(out_path), offset_ms, len(captions))
