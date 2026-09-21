#!/usr/bin/env python3
"""End-to-end verification for the mouth-motion wide-range recentering
tier and the short-clip coverage-floor fix. Two parts:

  A. Long clips (>= 20s face coverage) across the full offset sweep
     (in-range and large): confirms the mouth-motion wide-range tier both
     recovers large offsets accurately and does not regress small ones.
  B. Short clips (< 20s face coverage) at large offsets: confirms the
     coverage-floor fix makes these safely abstain (untrusted) rather than
     confidently wrong, since the wide-range tier must not engage here.

Usage:
    python3 analyzer/scripts/blind_benchmark/gen_clips.py   # if not already run
    python3 analyzer/scripts/blind_benchmark/final_verify.py
"""
import json
import sys
import time
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYZER))
from syncsentry.pipeline import _detect_av_offset  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "_out"
LOG_PATH = OUT_DIR / "final_verify.out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(line):
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def detect(path, **kw):
    kw.setdefault("av_threshold_ms", 40.0)
    kw.setdefault("use_syncnet", True)
    kw.setdefault("syncnet_min_confidence", 3.0)
    kw.setdefault("av_min_confidence", 0.3)
    t0 = time.time()
    det = _detect_av_offset(str(path), **kw)
    return det, time.time() - t0


def part_a():
    manifest = json.loads((OUT_DIR / "clips" / "manifest.json").read_text())
    log("\n=== PART A: long clips (dialogue_full50s, talk_25s), all offsets ===")
    for base in ["dialogue_full50s", "talk_25s"]:
        for m in sorted((x for x in manifest if x["base"] == base), key=lambda x: x["offset_ms_true"]):
            det, dt = detect(m["path"])
            err = abs(det.offset_ms - m["offset_ms_true"])
            trusted = det.confidence >= det.min_confidence
            flag = "BAD!" if (trusted and err > 40) else ("ok" if trusted else "untrusted")
            log(f"{base:20s} true={m['offset_ms_true']:+6.0f} pred={det.offset_ms:+8.1f} conf={det.confidence:.2f} trusted={trusted} err={err:.0f}ms t={dt:.1f}s {flag}")


def part_b():
    manifest = json.loads((OUT_DIR / "clips" / "manifest.json").read_text())
    log("\n=== PART B: short clips, large offsets (should be safe/untrusted, not confidently wrong) ===")
    for base in ["talk_8s", "dialogue_short10s", "talk_3s", "talk_5s"]:
        cases = [x for x in manifest if x["base"] == base and abs(x["offset_ms_true"]) >= 900]
        for m in cases:
            det, dt = detect(m["path"])
            err = abs(det.offset_ms - m["offset_ms_true"])
            trusted = det.confidence >= det.min_confidence
            flag = "DANGEROUS!" if (trusted and err > 200) else "safe"
            log(f"{base:20s} true={m['offset_ms_true']:+6.0f} pred={det.offset_ms:+8.1f} conf={det.confidence:.2f} trusted={trusted} err={err:.0f}ms t={dt:.1f}s {flag}")


if __name__ == "__main__":
    part_a()
    part_b()
    log("\nDONE")
