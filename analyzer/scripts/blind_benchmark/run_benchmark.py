#!/usr/bin/env python3
"""Runs syncsentry's real detection pipeline blind on every clip in the
manifest produced by `gen_clips.py` (the true offset is never passed in,
only known here for scoring after the fact) and records predicted vs
ground truth.

Usage:
    python3 analyzer/scripts/blind_benchmark/run_benchmark.py <tag> [--limit N]
    <tag> names the results file (e.g. "baseline", "wide_range_v2").

Requires the real_content fixtures + vendored SyncNet to be fetched first:
    bash analyzer/scripts/fetch_real_content.sh
    bash analyzer/scripts/fetch_syncnet.sh
    python3 analyzer/scripts/blind_benchmark/gen_clips.py
"""
import json
import sys
import time
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYZER))
from syncsentry.pipeline import _detect_av_offset  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "_out"
MANIFEST = OUT_DIR / "clips" / "manifest.json"
RESULTS_DIR = OUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one(path: str, use_syncnet=True, syncnet_min_confidence=3.0,
            av_min_confidence=0.3, av_threshold_ms=40.0, use_mtdvocalist=False):
    t0 = time.time()
    try:
        det = _detect_av_offset(path, av_threshold_ms=av_threshold_ms, use_syncnet=use_syncnet,
                                 syncnet_min_confidence=syncnet_min_confidence,
                                 av_min_confidence=av_min_confidence,
                                 use_mtdvocalist=use_mtdvocalist)
        elapsed = time.time() - t0
        return {
            "offset_ms_pred": det.offset_ms,
            "confidence": det.confidence,
            "min_confidence": det.min_confidence,
            "method": det.method,
            "direction": det.direction,
            "trusted": det.confidence >= det.min_confidence,
            "elapsed_s": round(elapsed, 1),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 (benchmark must not die on one bad clip)
        return {
            "offset_ms_pred": None, "confidence": None, "min_confidence": None,
            "method": None, "direction": None, "trusted": False,
            "elapsed_s": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    limit = None
    for a in sys.argv[2:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])

    manifest = json.loads(MANIFEST.read_text())
    if limit:
        manifest = manifest[:limit]

    out_path = RESULTS_DIR / f"{tag}.jsonl"
    results = []
    t_start = time.time()
    with out_path.open("w") as f:
        for i, case in enumerate(manifest):
            r = run_one(case["path"])
            row = {**case, **r}
            results.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            true_ms = case["offset_ms_true"]
            pred_ms = r["offset_ms_pred"]
            err = "?" if pred_ms is None else f"{abs(pred_ms - true_ms):.0f}ms"
            print(f"[{i+1}/{len(manifest)}] {case['base']:18s} true={true_ms:+6.0f}ms "
                  f"pred={pred_ms if pred_ms is not None else 'ERR':>8} conf={r['confidence']} "
                  f"method={r['method']} err={err} t={r['elapsed_s']}s "
                  f"total_elapsed={time.time()-t_start:.0f}s", flush=True)

    print(f"\nDone. {len(results)} cases -> {out_path}. Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
