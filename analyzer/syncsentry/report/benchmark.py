"""Benchmark harness: sweep injected offsets, measure recovery error, and
emit a markdown table and chart. Evaluates SyncSentry's own detectors
against synthetic ground truth, so results are reproducible by anyone
cloning the repo (`python -m syncsentry.cli benchmark`), with no
restricted-access datasets required.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from syncsentry.captions.drift_check import check_caption_drift
from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.synth.fixture_gen import FixtureSpec, generate_captions, generate_fixture

AV_OFFSETS_MS = [-900, -700, -500, -300, -200, -150, -100, -60, -30, -10, 0, 10, 30, 60, 100, 150, 200, 300, 500, 700, 900]
CAPTION_OFFSETS_MS = [-300, -200, -120, -80, -30, 0, 30, 80, 120, 200, 300]

# Cross-correlating a *periodic* pulse train is only unambiguous up to half
# the pulse period (a lag of +period/2 and -period/2 produce identical
# correlation peaks). We use a 2s period here so the sweep above (up to
# 900ms) stays safely inside the unambiguous +/-1000ms window. Real content
# doesn't have perfectly periodic energy, so this ambiguity is a
# synthetic-benchmark artifact rather than a production concern.
_AV_PERIOD_S = 2.0


def run_av_offset_benchmark(out_dir: Path) -> list[dict]:
    fixtures_dir = out_dir / "_fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for offset_ms in AV_OFFSETS_MS:
        spec = FixtureSpec(duration_s=12.0, period_s=_AV_PERIOD_S, pulse_ms=80, offset_ms=offset_ms)
        video_path = generate_fixture(fixtures_dir / f"av_{offset_ms}.mkv", spec)
        estimate = estimate_av_offset(str(video_path), search_window_ms=950.0)
        rows.append({
            "injected_ms": offset_ms,
            "estimated_ms": round(estimate.offset_ms, 2),
            "error_ms": round(estimate.offset_ms - offset_ms, 2),
            "confidence": round(estimate.confidence, 3),
            "direction": estimate.direction,
        })

    (out_dir / "av_offset_benchmark.json").write_text(json.dumps(rows, indent=2))
    _write_av_markdown(rows, out_dir / "av_offset_benchmark.md")
    _plot_av(rows, out_dir / "av_offset_benchmark.png")
    return rows


def run_caption_drift_benchmark(out_dir: Path) -> list[dict]:
    fixtures_dir = out_dir / "_fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    spec = FixtureSpec(duration_s=10.0, period_s=1.0, pulse_ms=80, offset_ms=0.0)
    video_path = generate_fixture(fixtures_dir / "captions_base.mkv", spec)

    rows = []
    for offset_ms in CAPTION_OFFSETS_MS:
        vtt_path = generate_captions(
            fixtures_dir / f"captions_{offset_ms}.vtt", spec, caption_offset_ms=offset_ms,
        )
        report = check_caption_drift(str(video_path), str(vtt_path))
        rows.append({
            "injected_ms": offset_ms,
            "median_estimated_ms": report.median_offset_ms,
            "error_ms": None if report.median_offset_ms is None
                        else round(report.median_offset_ms - offset_ms, 2),
            "matched_cues": report.matched_count,
            "unmatched_cues": report.unmatched_count,
            "flagged_cues": len(report.flagged_cue_indices),
        })

    (out_dir / "caption_drift_benchmark.json").write_text(json.dumps(rows, indent=2))
    _write_caption_markdown(rows, out_dir / "caption_drift_benchmark.md")
    return rows


def _write_av_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| Injected (ms) | Estimated (ms) | Error (ms) | Confidence | Direction |",
        "|---:|---:|---:|---:|:---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['injected_ms']} | {r['estimated_ms']} | {r['error_ms']} | "
            f"{r['confidence']} | {r['direction']} |"
        )
    path.write_text("\n".join(lines))


def _write_caption_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "| Injected caption offset (ms) | Recovered median (ms) | Error (ms) | Matched | Flagged |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['injected_ms']} | {r['median_estimated_ms']} | {r['error_ms']} | "
            f"{r['matched_cues']} | {r['flagged_cues']} |"
        )
    path.write_text("\n".join(lines))


def _plot_av(rows: list[dict], path: Path) -> None:
    injected = [r["injected_ms"] for r in rows]
    estimated = [r["estimated_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(injected, injected, "--", color="gray", label="Perfect recovery")
    ax.plot(injected, estimated, "o-", color="C0", label="SyncSentry coarse detector")
    ax.set_xlabel("Injected A/V offset (ms)")
    ax.set_ylabel("Estimated A/V offset (ms)")
    ax.set_title("A/V offset recovery accuracy (synthetic ground truth)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
