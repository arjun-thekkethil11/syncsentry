"""SyncSentry command-line interface.

    syncsentry gen-fixture   --out fixtures/x.mkv --offset-ms 150
    syncsentry detect-offset --video fixtures/x.mkv
    syncsentry check-captions --video fixtures/x.mkv --captions fixtures/x.vtt
    syncsentry fix           --video fixtures/x.mkv --captions fixtures/x.vtt --out-dir out/
    syncsentry classify-drift --video fixtures/x.mkv
    syncsentry dialogue-scenes --video fixtures/x.mkv
    syncsentry benchmark      --out-dir benchmark/results
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from syncsentry.captions.drift_check import check_caption_drift
from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.synth.fixture_gen import FixtureSpec, generate_captions, generate_fixture


def _print_json(obj) -> None:
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    print(json.dumps(obj, indent=2, default=str))


def cmd_gen_fixture(args: argparse.Namespace) -> None:
    spec = FixtureSpec(
        duration_s=args.duration, period_s=args.period, pulse_ms=args.pulse_ms,
        offset_ms=args.offset_ms, fps=args.fps, sr=args.sr,
    )
    out = Path(args.out)
    video_path = generate_fixture(out, spec)
    print(f"wrote {video_path}")
    if args.captions:
        vtt_path = generate_captions(
            out.with_suffix(".vtt"), spec, caption_offset_ms=args.caption_offset_ms,
        )
        print(f"wrote {vtt_path}")


def cmd_detect_offset(args: argparse.Namespace) -> None:
    estimate = estimate_av_offset(args.video, search_window_ms=args.search_window_ms)
    _print_json(estimate)


def cmd_check_captions(args: argparse.Namespace) -> None:
    report = check_caption_drift(args.video, args.captions)
    _print_json(report)


def cmd_fix(args: argparse.Namespace) -> None:
    from syncsentry.pipeline import run_fix_pipeline

    summary = run_fix_pipeline(
        video_path=args.video,
        out_dir=args.out_dir,
        captions_path=args.captions,
        av_threshold_ms=args.av_threshold_ms,
        caption_threshold_ms=args.caption_threshold_ms,
    )
    print(summary.to_text())


def cmd_classify_drift(args: argparse.Namespace) -> None:
    from syncsentry.lipsync.scene_offsets import estimate_windowed_offsets
    from syncsentry.lipsync.title_drift import classify_title_drift

    windows = estimate_windowed_offsets(args.video, window_s=args.window_s)
    if len(windows) < 3:
        print(f"Only {len(windows)} confident scene(s) found -- not enough to classify a title-level "
              f"pattern (need >= 3). Try a longer asset or a smaller --window-s.")
        return

    result = classify_title_drift(windows)
    print(f"Title drift pattern : {result.pattern}")
    print(f"Slope                : {result.slope_ms_per_s:+.2f} ms/s")
    print(f"Intercept            : {result.intercept_ms:+.2f} ms")
    print(f"Inlier ratio         : {result.inlier_ratio:.2f} ({len(windows)} scenes analyzed)")
    if result.intermittent_window_s:
        lo, hi = result.intermittent_window_s
        print(f"Intermittent window  : {lo:.1f}s - {hi:.1f}s (offset ~{result.intermittent_offset_ms:+.0f}ms)")


def cmd_dialogue_scenes(args: argparse.Namespace) -> None:
    from syncsentry.lipsync.dialogue_scenes import detect_dialogue_scenes

    scenes = detect_dialogue_scenes(
        args.video, sample_fps=args.sample_fps, min_duration_s=args.min_duration_s,
    )
    if not scenes:
        print("No dialogue scenes found (no time range with both a face on screen and active speech).")
        return

    total_s = sum(s.end_s - s.start_s for s in scenes)
    print(f"{len(scenes)} dialogue scene(s), {total_s:.1f}s total:")
    for s in scenes:
        print(f"  {s.start_s:7.2f}s - {s.end_s:7.2f}s  (dur {s.end_s - s.start_s:5.2f}s)")


def cmd_benchmark(args: argparse.Namespace) -> None:
    from syncsentry.report.benchmark import run_av_offset_benchmark, run_caption_drift_benchmark

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    av_results = run_av_offset_benchmark(out_dir)
    cap_results = run_caption_drift_benchmark(out_dir)

    print(f"A/V offset benchmark: {len(av_results)} cases -> {out_dir}")
    print(f"Caption drift benchmark: {len(cap_results)} cases -> {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syncsentry")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-fixture", help="Generate a synthetic AV fixture with a known offset")
    p.add_argument("--out", required=True, help="Output .mkv path")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--period", type=float, default=1.0)
    p.add_argument("--pulse-ms", type=float, default=80.0)
    p.add_argument("--offset-ms", type=float, default=0.0, help="Injected A/V offset (audio vs video)")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--sr", type=int, default=48000)
    p.add_argument("--captions", action="store_true", help="Also generate a matching .vtt file")
    p.add_argument("--caption-offset-ms", type=float, default=0.0)
    p.set_defaults(func=cmd_gen_fixture)

    p = sub.add_parser("detect-offset", help="Estimate global A/V offset via envelope cross-correlation")
    p.add_argument("--video", required=True)
    p.add_argument("--search-window-ms", type=float, default=500.0)
    p.set_defaults(func=cmd_detect_offset)

    p = sub.add_parser("check-captions", help="Compare WebVTT cue timing against detected audio onsets")
    p.add_argument("--video", required=True)
    p.add_argument("--captions", required=True)
    p.set_defaults(func=cmd_check_captions)

    p = sub.add_parser("fix", help="Detect and correct A/V offset and/or caption drift, with a short report")
    p.add_argument("--video", required=True)
    p.add_argument("--captions", default=None, help="Optional WebVTT file to also check/fix")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--av-threshold-ms", type=float, default=40.0)
    p.add_argument("--caption-threshold-ms", type=float, default=80.0)
    p.set_defaults(func=cmd_fix)

    p = sub.add_parser("classify-drift", help="Classify a title's sync-drift pattern (constant/drift/intermittent)")
    p.add_argument("--video", required=True)
    p.add_argument("--window-s", type=float, default=3.0)
    p.set_defaults(func=cmd_classify_drift)

    p = sub.add_parser("dialogue-scenes", help="Detect real dialogue scenes (face on screen AND speech active)")
    p.add_argument("--video", required=True)
    p.add_argument("--sample-fps", type=float, default=2.0, help="Face-detection sampling rate")
    p.add_argument("--min-duration-s", type=float, default=1.0)
    p.set_defaults(func=cmd_dialogue_scenes)

    p = sub.add_parser("benchmark", help="Run the full offset-recovery benchmark sweep and write a report")
    p.add_argument("--out-dir", default="benchmark/results")
    p.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
