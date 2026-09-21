#!/usr/bin/env python3
"""Summarizes a benchmark results file (JSONL from run_benchmark.py) into
accuracy/timing breakdowns.

Usage:
    python3 analyzer/scripts/blind_benchmark/analyze.py <tag>
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent / "_out" / "results"

# SyncNet's default vshift=15 @ 25fps -> +-600ms is the max offset it can
# find without the wide-range recentering tier. Cases beyond this are
# expected-impossible for the baseline and analyzed separately.
IN_RANGE_MS = 600.0
EXACT_TOLERANCE_MS = 40.0


def load(tag):
    path = RESULTS_DIR / f"{tag}.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    rows = load(tag)

    print(f"=== {tag} ({len(rows)} cases) ===\n")

    for r in rows:
        r["in_range"] = abs(r["offset_ms_true"]) <= IN_RANGE_MS
        if r["offset_ms_pred"] is not None:
            r["abs_error_ms"] = abs(r["offset_ms_pred"] - r["offset_ms_true"])
            r["within_tolerance"] = r["abs_error_ms"] <= EXACT_TOLERANCE_MS
            r["direction_correct"] = (
                (r["offset_ms_true"] == 0 and abs(r["offset_ms_pred"]) <= EXACT_TOLERANCE_MS) or
                (r["offset_ms_true"] != 0 and r["offset_ms_pred"] * r["offset_ms_true"] > 0)
            )
        else:
            r["abs_error_ms"] = None
            r["within_tolerance"] = False
            r["direction_correct"] = False

    def pct(vals):
        vals = list(vals)
        return f"{100*sum(vals)/len(vals):.0f}%" if vals else "n/a"

    # Overall
    in_range = [r for r in rows if r["in_range"]]
    out_range = [r for r in rows if not r["in_range"]]
    print(f"IN-RANGE cases (|true offset| <= {IN_RANGE_MS:.0f}ms, {len(in_range)} cases):")
    print(f"  trusted (confidence cleared threshold): {pct(r['trusted'] for r in in_range)}")
    trusted_in_range = [r for r in in_range if r["trusted"]]
    print(f"  of trusted: within {EXACT_TOLERANCE_MS:.0f}ms tolerance: {pct(r['within_tolerance'] for r in trusted_in_range)}")
    print(f"  of trusted: correct direction: {pct(r['direction_correct'] for r in trusted_in_range)}")
    untrusted_in_range = [r for r in in_range if not r["trusted"]]
    if untrusted_in_range:
        would_have_been_right = [r for r in untrusted_in_range if r["within_tolerance"]]
        print(f"  NOT trusted (rejected as low-confidence): {len(untrusted_in_range)}, of those, would have been right anyway: {len(would_have_been_right)}")
    errors_ms = [r["abs_error_ms"] for r in trusted_in_range if r["abs_error_ms"] is not None]
    if errors_ms:
        sorted_errs = sorted(errors_ms)
        p90 = sorted_errs[int(0.9 * (len(sorted_errs) - 1))]
        print(f"  mean abs error (trusted only): {sum(errors_ms)/len(errors_ms):.1f}ms, "
              f"median: {sorted_errs[len(sorted_errs)//2]:.1f}ms, p90: {p90:.1f}ms, max: {max(errors_ms):.1f}ms")

    print(f"\nOUT-OF-RANGE cases (|true offset| > {IN_RANGE_MS:.0f}ms, {len(out_range)} cases), the wide-range tier's target:")
    print(f"  trusted: {pct(r['trusted'] for r in out_range)}")
    trusted_out_range = [r for r in out_range if r["trusted"]]
    print(f"  of trusted: within tolerance: {pct(r['within_tolerance'] for r in trusted_out_range)}")
    errors_ms_out = [r["abs_error_ms"] for r in trusted_out_range if r["abs_error_ms"] is not None]
    if errors_ms_out:
        sorted_errs = sorted(errors_ms_out)
        p90 = sorted_errs[int(0.9 * (len(sorted_errs) - 1))]
        print(f"  mean abs error (trusted only): {sum(errors_ms_out)/len(errors_ms_out):.1f}ms, "
              f"median: {sorted_errs[len(sorted_errs)//2]:.1f}ms, p90: {p90:.1f}ms, max: {max(errors_ms_out):.1f}ms")
    untrusted_out_range = [r for r in out_range if not r["trusted"]]
    print(f"  NOT trusted (safely abstained): {len(untrusted_out_range)}/{len(out_range)}")

    # Confidence calibration: of ALL trusted predictions (in-range + out-of-range), what fraction were correct?
    trusted_all = [r for r in rows if r["trusted"]]
    print(f"\nCONFIDENCE CALIBRATION: of {len(trusted_all)} trusted predictions, "
          f"{pct(r['within_tolerance'] for r in trusted_all)} were within {EXACT_TOLERANCE_MS:.0f}ms tolerance "
          f"(dangerous/confidently-wrong: {sum(1 for r in trusted_all if not r['within_tolerance'] and r.get('abs_error_ms', 0) and r['abs_error_ms'] > 200)})")

    # By base clip
    print("\n--- By base clip ---")
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base"]].append(r)
    for base, rs in by_base.items():
        times = [r["elapsed_s"] for r in rs]
        errs = [r["error"] for r in rs if r.get("error")]
        print(f"  {base:20s} n={len(rs):2d}  avg_time={sum(times)/len(times):.1f}s  "
              f"max_time={max(times):.1f}s  exceptions={len(errs)}")

    # Failures worth looking at (in-range but wrong or untrusted)
    print("\n--- In-range cases that were WRONG while trusted (real bugs) ---")
    bad = [r for r in in_range if r["trusted"] and not r["within_tolerance"]]
    if not bad:
        print("  (none)")
    for r in bad:
        print(f"  {r['base']:20s} true={r['offset_ms_true']:+7.0f}ms pred={r['offset_ms_pred']:+7.0f}ms "
              f"conf={r['confidence']:.2f} method={r['method']}")

    print("\n--- Out-of-range cases that were confidently WRONG (dangerous) ---")
    dangerous = [r for r in out_range if r["trusted"] and r.get("abs_error_ms") and r["abs_error_ms"] > 200]
    if not dangerous:
        print("  (none)")
    for r in dangerous:
        print(f"  {r['base']:20s} true={r['offset_ms_true']:+7.0f}ms pred={r['offset_ms_pred']:+7.0f}ms "
              f"conf={r['confidence']:.2f} method={r['method']}")

    print("\n--- Exceptions ---")
    for r in rows:
        if r.get("error"):
            print(f"  {r['base']:20s} true={r['offset_ms_true']:+7.0f}ms -> {r['error']}")


if __name__ == "__main__":
    main()
