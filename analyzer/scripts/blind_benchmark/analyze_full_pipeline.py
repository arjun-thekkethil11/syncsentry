#!/usr/bin/env python3
"""Summarizes a `run_full_pipeline_benchmark.py` results file: final-output
alignment error (median/p90/worst-case), confidence calibration,
abstention rate, and runtime, computed against the real production
pipeline's output (status/fixed/detected_offset_ms), not just the raw
detector.

Usage:
    python3 analyzer/scripts/blind_benchmark/analyze_full_pipeline.py <tag>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "_out" / "results"


def load(tag):
    path = RESULTS_DIR / f"{tag}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def final_output_error(r):
    """The residual real-world misalignment of the actual file the user
    would download, given the known-true injected offset: 0 if genuinely
    in_sync (true offset was ~0, nothing touched), the applied-correction
    error if fixed (correction = detected_offset_ms, applied by
    `fix_av_offset`, exact, not re-measured, since re-measuring would just
    be testing the detector on itself again), or the untouched true offset
    if nothing was applied (undetermined/not_fixed/exception)."""
    true_ms = r["offset_ms_true"]
    if r.get("status") == "in_sync":
        return abs(true_ms)
    if r.get("fixed") and r.get("detected_offset_ms") is not None:
        return abs(true_ms - r["detected_offset_ms"])
    return abs(true_ms)


def pctl(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(int(p * (len(s) - 1)), len(s) - 1)]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "held_out_v1"
    rows = load(tag)
    for r in rows:
        r["final_err_ms"] = final_output_error(r)
        r["is_confidently_wrong"] = (
            r.get("status") in ("in_sync", "fixed") and r["final_err_ms"] > 200.0
        )

    n = len(rows)
    print(f"=== {tag}: full-production-pipeline held-out benchmark ({n} cases) ===")
    print("(Every case below ran the REAL run_fix_pipeline(): detection, ffmpeg")
    print(" correction/mux, and post-correction verification, exactly as /v1/fix")
    print(" does for a real upload. True offset was injected but never passed in.)\n")

    by_status = defaultdict(list)
    for r in rows:
        by_status[r["status"]].append(r)
    print("--- Outcome breakdown ---")
    for status in ("in_sync", "fixed", "not_fixed", "undetermined", "exception"):
        rs = by_status.get(status, [])
        if rs:
            print(f"  {status:14s} {len(rs):2d}/{n} ({100*len(rs)/n:.0f}%)")

    abstain = by_status.get("undetermined", []) + by_status.get("exception", [])
    acted = by_status.get("in_sync", []) + by_status.get("fixed", []) + by_status.get("not_fixed", [])
    print(f"\n  abstention rate (undetermined/exception, no claim either way): {100*len(abstain)/n:.0f}%")

    errs = [r["final_err_ms"] for r in rows]
    print("\n--- Final-output alignment error (ms), ALL cases (abstentions count their full untouched offset) ---")
    print(f"  mean={sum(errs)/n:.1f}  median={pctl(errs, 0.5):.1f}  p90={pctl(errs, 0.9):.1f}  worst={max(errs):.1f}")

    claimed = by_status.get("in_sync", []) + by_status.get("fixed", [])
    if claimed:
        claimed_errs = [r["final_err_ms"] for r in claimed]
        print("\n--- Final-output alignment error (ms), cases where the system made a positive claim "
              f"(in_sync or fixed, {len(claimed)}/{n}) ---")
        print(f"  mean={sum(claimed_errs)/len(claimed):.1f}  median={pctl(claimed_errs, 0.5):.1f}  "
              f"p90={pctl(claimed_errs, 0.9):.1f}  worst={max(claimed_errs):.1f}")

    print("\n--- Confidence calibration: of cases where the system claimed in_sync/fixed, "
          "how often was the final output actually within 40ms of true alignment? ---")
    if claimed:
        within_tol = sum(1 for r in claimed if r["final_err_ms"] <= 40.0)
        print(f"  {within_tol}/{len(claimed)} ({100*within_tol/len(claimed):.0f}%) within 40ms")
    dangerous = [r for r in rows if r["is_confidently_wrong"]]
    print(f"  confidently wrong (claimed in_sync/fixed but final error > 200ms): {len(dangerous)}/{n}")
    for r in dangerous:
        print(f"    {r['base']:18s} true={r['offset_ms_true']:+7.0f}ms status={r['status']:12s} "
              f"detected={r.get('detected_offset_ms')} final_err={r['final_err_ms']:.0f}ms")

    print("\n--- Abstentions: true offset the system safely declined to touch (not a failure, the safe-fallback contract) ---")
    for r in abstain:
        print(f"  {r['base']:18s} true={r['offset_ms_true']:+7.0f}ms detected={r.get('detected_offset_ms')} "
              f"method={r.get('method')} note_status={r['status']}")

    times = [r["elapsed_s"] for r in rows]
    print(f"\n--- Runtime (full pipeline incl. detect+fix+verify+mux), n={n} ---")
    print(f"  mean={sum(times)/n:.1f}s  median={pctl(times, 0.5):.1f}s  p90={pctl(times, 0.9):.1f}s  max={max(times):.1f}s")
    print("  by base clip:")
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base"]].append(r["elapsed_s"])
    for base, ts in by_base.items():
        print(f"    {base:18s} n={len(ts):2d} mean={sum(ts)/len(ts):.1f}s max={max(ts):.1f}s")

    exceptions = [r for r in rows if r.get("error")]
    if exceptions:
        print(f"\n--- Exceptions ({len(exceptions)}) ---")
        for r in exceptions:
            print(f"  {r['base']:18s} true={r['offset_ms_true']:+7.0f}ms -> {r['error']}")


if __name__ == "__main__":
    main()
