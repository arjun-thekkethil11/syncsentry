#!/usr/bin/env python3
"""Generates a diverse suite of ground-truth-labeled A/V sync test clips
for blind benchmarking of syncsentry's detectors. The injected offset is
known here but never passed to the detector: the detector must find it
exactly like a real upload would require.

Source content is the project's existing real_content fixtures
(`analyzer/fixtures/real_content/`, fetched via
`analyzer/scripts/fetch_real_content.sh`), not the four ad-hoc diagnostic
assets used for root-cause analysis, so these results are a genuine
held-out generalization check (different speakers/recordings/domains).

Usage:
    python3 analyzer/scripts/blind_benchmark/gen_clips.py

Output (gitignored, regenerate, don't commit):
    analyzer/scripts/blind_benchmark/_out/clips/*.mp4
    analyzer/scripts/blind_benchmark/_out/clips/manifest.json
"""
import subprocess
import sys
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYZER))
from syncsentry.fixer.av_fix import fix_av_offset  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out" / "clips"
OUT.mkdir(parents=True, exist_ok=True)

DIALOGUE = ANALYZER / "fixtures/real_content/dialogue_clip.mkv"
TALK = ANALYZER / "fixtures/real_content/royal_society_whiskers.webm"

# (name, source, start_s, duration_s); start_s/duration_s = None means "whole file".
#
# NOTE on the TALK (royal_society_whiskers.webm) timestamps below: this
# source cuts between an on-camera scientist talking and cat/mouse-whisker
# B-roll with no human face at all. Confirmed visually that 30s/120s/240s
# are B-roll (an earlier version of this suite picked those blind, and
# every one correctly fell back to the coarse detector since there's no
# face there, but that's not the short-face-clip test this suite needs).
# 60-85s is a verified continuous on-camera talking segment; all TALK
# sub-clips below are carved from inside it.
BASE_CLIPS = [
    ("dialogue_full50s", DIALOGUE, None, None),      # multi-speaker, 50s
    ("dialogue_short10s", DIALOGUE, 0.0, 10.0),       # multi-speaker, short, cuts scenes every ~2-3s
    ("talk_3s", TALK, 65.0, 3.0),                     # single speaker, very short edge case
    ("talk_5s", TALK, 68.0, 5.0),                     # single speaker, short
    ("talk_8s", TALK, 70.0, 8.0),                     # single speaker, short-ish
    ("talk_25s", TALK, 60.0, 25.0),                   # single speaker, just above the 20s coverage floor
]

# ms, positive = audio_lags (matches this project's convention). Includes
# values beyond the current SyncNet vshift=15 search range (+-600ms) on
# purpose: a deliberate edge case, not an oversight.
OFFSETS_MS = [0.0, 80.0, -80.0, 200.0, -200.0, 500.0, -500.0, 900.0, -900.0]

# Large-offset held-out set: matches the scale of offsets actually found
# in the 4 real diagnostic assets (600-1500ms), plus values beyond that
# range, on content not used to design the mouth-motion wide-range detector
# (`wide_range_offset.py`); these fixture clips (Royal Society talk +
# dialogue clip) are unrelated speakers/recordings/domains from the 4 real
# podcast/interview diagnostic assets.
LARGE_OFFSETS_MS = [1200.0, -1200.0, 1500.0, -1500.0, 2200.0, -2200.0, 3000.0, -3000.0]

# Which base clips get the large-offset variants: skip clips too short for
# the shift to leave meaningful overlap (a 3s/5s clip shifted by 2s+ has
# almost no real content left to detect anything from, so it's not a
# meaningful test of the detector, just a test of "not enough data").
LARGE_OFFSET_CLIPS = {"dialogue_full50s", "talk_25s", "talk_8s", "dialogue_short10s"}


def make_base_clip(name: str, source: Path, start_s, duration_s) -> Path:
    dest = OUT / f"{name}__base.mp4"
    if dest.exists():
        return dest
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start_s is not None:
        cmd += ["-i", str(source), "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}"]
    else:
        cmd += ["-i", str(source)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"failed to cut {name}: {proc.stderr[-2000:]}")
    return dest


def make_variant(base_name: str, base_path: Path, offset_ms: float) -> Path:
    tag = f"{offset_ms:+.0f}ms".replace("+", "p").replace("-", "n")
    dest = OUT / f"{base_name}__{tag}.mp4"
    if dest.exists():
        return dest
    fix_av_offset(base_path, dest, offset_ms=-offset_ms)  # injects true offset = offset_ms
    return dest


def main():
    import json
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    seen = {(m["base"], m["offset_ms_true"]) for m in manifest}
    for name, source, start_s, duration_s in BASE_CLIPS:
        base_path = make_base_clip(name, source, start_s, duration_s)
        offsets = list(OFFSETS_MS)
        if name in LARGE_OFFSET_CLIPS:
            offsets += LARGE_OFFSETS_MS
        for offset_ms in offsets:
            if (name, offset_ms) in seen:
                continue
            variant_path = make_variant(name, base_path, offset_ms)
            manifest.append({
                "base": name,
                "offset_ms_true": offset_ms,
                "path": str(variant_path),
            })
            print(f"  {name:20s} offset={offset_ms:+7.0f}ms -> {variant_path.name}")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} clips total -> {OUT}")


if __name__ == "__main__":
    main()
