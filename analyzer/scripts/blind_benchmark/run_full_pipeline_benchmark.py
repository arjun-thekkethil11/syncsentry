#!/usr/bin/env python3
"""Runs the actual production fix pipeline (`syncsentry.pipeline.run_fix_pipeline`,
the same function `api.py`'s `/v1/fix` and `cli.py`'s `fix` command call)
blind on every clip in the manifest produced by `gen_clips.py`: detection,
correction (real ffmpeg mux), and post-correction verification, exactly as
a real user's request would exercise. This supersedes `run_benchmark.py`,
which only calls `_detect_av_offset` directly and never runs the fixer or
touches a muxed output file. The benchmark should invoke the same
production pipeline and output muxing used by the web application; a
script-only or model-only metric is insufficient.

The true offset is known here (for scoring after the fact) but never
passed to the pipeline.

Usage:
    python3 analyzer/scripts/blind_benchmark/run_full_pipeline_benchmark.py <tag> [--limit N]
"""
import json
import sys
import tempfile
import time
from pathlib import Path

ANALYZER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANALYZER))
from syncsentry.pipeline import run_fix_pipeline  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "_out"
MANIFEST = OUT_DIR / "clips" / "manifest.json"
RESULTS_DIR = OUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one(path: str, use_syncnet=True, syncnet_min_confidence=3.0, av_min_confidence=0.3):
    """Runs the real production pipeline end-to-end, including writing (and
    then discarding) the actual corrected/muxed output file, matching
    exactly what `/v1/fix` does for a real upload."""
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory() as out_dir:
            summary = run_fix_pipeline(
                path, out_dir, use_syncnet=use_syncnet,
                syncnet_min_confidence=syncnet_min_confidence,
                av_min_confidence=av_min_confidence,
            )
            elapsed = time.time() - t0
            issue = next((i for i in summary.issues if i.name == "A/V sync"), None)
            if issue is None:
                return {"status": "no_issue_reported", "elapsed_s": round(elapsed, 1), "error": None}
            return {
                "status": issue.status,
                "detected_offset_ms": issue.detected_offset_ms,
                "fixed": issue.fixed,
                "residual_offset_ms": issue.residual_offset_ms,
                "residual_confidence": issue.residual_confidence,
                "confidence": issue.confidence,
                "min_confidence": issue.min_confidence,
                "method": issue.method,
                "elapsed_s": round(elapsed, 1),
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001 (benchmark must not die on one bad clip)
        return {
            "status": "exception", "detected_offset_ms": None, "fixed": False,
            "residual_offset_ms": None, "residual_confidence": None, "confidence": None,
            "min_confidence": None, "method": None,
            "elapsed_s": round(time.time() - t0, 1), "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "full_pipeline_v1"
    limit = None
    for a in sys.argv[2:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])

    manifest = json.loads(MANIFEST.read_text())
    indices = None
    for a in sys.argv[2:]:
        if a.startswith("--indices"):
            spec = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            indices = [int(x) for x in spec.split(",")]
    if indices is not None:
        manifest = [manifest[i] for i in indices]
    elif limit:
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
            # "final-output alignment error": the residual real-world
            # misalignment of the actual output the user would download.
            # 0 if genuinely in_sync and true offset really is ~0; the
            # applied-correction error if fixed; the untouched true offset
            # if nothing was applied (undetermined/not_fixed/no correction).
            if r.get("status") == "in_sync":
                final_err = abs(true_ms)
            elif r.get("fixed") and r.get("detected_offset_ms") is not None:
                final_err = abs(true_ms - r["detected_offset_ms"])
            else:
                final_err = abs(true_ms)
            print(f"[{i+1}/{len(manifest)}] {case['base']:18s} true={true_ms:+6.0f}ms "
                  f"status={r.get('status'):20s} detected={r.get('detected_offset_ms')} "
                  f"fixed={r.get('fixed')} final_err={final_err:.0f}ms t={r['elapsed_s']}s "
                  f"total_elapsed={time.time()-t_start:.0f}s", flush=True)

    print(f"\nDone. {len(results)} cases -> {out_path}. Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
