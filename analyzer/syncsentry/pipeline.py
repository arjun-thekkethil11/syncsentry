"""End-to-end "detect, fix, prove it worked" pipeline for a single asset.

This is the thing an actual user runs: point it at a video (and optionally
a caption file), get back a corrected video/captions and a short report.

Order of operations matters and is deliberate:

  1. Detect the A/V offset on the *original* video.
  2. Fix it (if above threshold) -> produces a corrected video.
  3. Re-measure the A/V offset on the *corrected* video. This is real
     validation, not just trusting the fix -- and it's how the periodic
     cross-correlation ambiguity bug and the -itsoffset-does-nothing bug
     earlier in this project were actually caught.
  4. If captions were provided: detect caption drift against the
     *corrected* video's audio (not the original). This matters because
     step 2 may have physically shifted the audio track's timeline (e.g.
     trimmed samples off the front), so measuring against the original
     audio would silently give the wrong number for what to apply to the
     new captions.
  5. Fix caption drift (if above threshold) -> produces corrected captions.
  6. Re-measure caption drift on (corrected video, corrected captions) to
     confirm.

Every step re-measures against the actual output of the previous step
rather than composing corrections algebraically -- simpler to reason about
and robust to changes in how the individual fixers work.
"""
from __future__ import annotations

from pathlib import Path

from syncsentry.captions.drift_check import check_caption_drift
from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.fixer.caption_fix import fix_caption_offset
from syncsentry.report.summary import IssueSummary, SyncFixSummary

DEFAULT_AV_THRESHOLD_MS = 40.0  # ~1 frame at 25fps; tune per deployment
DEFAULT_CAPTION_THRESHOLD_MS = 80.0
# Matches the confidence gate already used by the windowed per-scene
# estimator (syncsentry/lipsync/scene_offsets.py). Found necessary here too
# via a real bug report: on real talking-head/dialogue content, global
# frame-brightness cross-correlation produces "confident-looking" but
# spurious offsets (see docs/RESEARCH.md section 1b) -- without this gate,
# the pipeline would "fix" an offset that was never really there, and the
# re-measurement afterward (also unreliable on the same content) could
# easily report the exact same bogus number as a "residual", i.e. a fix
# that visibly did nothing while still being labeled FIXED.
DEFAULT_AV_MIN_CONFIDENCE = 0.3


def run_fix_pipeline(video_path: str | Path, out_dir: str | Path,
                      captions_path: str | Path | None = None,
                      av_threshold_ms: float = DEFAULT_AV_THRESHOLD_MS,
                      caption_threshold_ms: float = DEFAULT_CAPTION_THRESHOLD_MS,
                      av_min_confidence: float = DEFAULT_AV_MIN_CONFIDENCE) -> SyncFixSummary:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = SyncFixSummary(asset_name=video_path.name)

    # --- Step 1-3: A/V sync ---
    av_estimate = estimate_av_offset(str(video_path), in_sync_threshold_ms=av_threshold_ms)
    corrected_video_path = out_dir / f"{video_path.stem}.corrected{video_path.suffix}"

    if av_estimate.confidence < av_min_confidence:
        # Not "in sync below threshold" -- we genuinely can't tell. Treating
        # this as "no issue" (rather than guessing a fix from noise) is the
        # honest answer; see docs/RESEARCH.md section 1b for why this
        # triggers on real dialogue/talking-head content specifically.
        corrected_video_path.write_bytes(video_path.read_bytes())
        summary.issues.append(IssueSummary(
            name="A/V sync", had_issue=False, detected_offset_ms=av_estimate.offset_ms,
            fixed=False, residual_offset_ms=None,
            note=(f"low-confidence detection ({av_estimate.confidence:.2f}) -- can't reliably "
                  f"tell if this asset has an A/V sync issue (common on real talking-head "
                  f"content); skipped rather than applying an unverifiable fix"),
        ))
    elif av_estimate.direction == "in_sync":
        corrected_video_path.write_bytes(video_path.read_bytes())
        summary.issues.append(IssueSummary(
            name="A/V sync", had_issue=False, detected_offset_ms=av_estimate.offset_ms,
            fixed=False, residual_offset_ms=None,
        ))
    else:
        fix_av_offset(video_path, corrected_video_path, av_estimate.offset_ms)
        residual = estimate_av_offset(str(corrected_video_path), in_sync_threshold_ms=av_threshold_ms)
        summary.issues.append(IssueSummary(
            name="A/V sync", had_issue=True, detected_offset_ms=av_estimate.offset_ms,
            fixed=(residual.confidence >= av_min_confidence and abs(residual.offset_ms) <= av_threshold_ms),
            residual_offset_ms=residual.offset_ms,
        ))

    summary.output_files["corrected_video"] = str(corrected_video_path)

    # --- Step 4-6: captions (measured against the CORRECTED video) ---
    if captions_path is not None:
        captions_path = Path(captions_path)
        cap_report = check_caption_drift(str(corrected_video_path), str(captions_path))
        corrected_captions_path = out_dir / f"{captions_path.stem}.corrected{captions_path.suffix}"

        if cap_report.median_offset_ms is None:
            note = "no matching speech onsets found"
            corrected_captions_path.write_text(captions_path.read_text(encoding="utf-8"), encoding="utf-8")
            summary.issues.append(IssueSummary(
                name="Captions", had_issue=False, detected_offset_ms=None,
                fixed=False, residual_offset_ms=None, note=note,
            ))
        elif abs(cap_report.median_offset_ms) < caption_threshold_ms:
            corrected_captions_path.write_text(captions_path.read_text(encoding="utf-8"), encoding="utf-8")
            summary.issues.append(IssueSummary(
                name="Captions", had_issue=False, detected_offset_ms=cap_report.median_offset_ms,
                fixed=False, residual_offset_ms=None,
            ))
        else:
            fix_caption_offset(captions_path, corrected_captions_path, cap_report.median_offset_ms)
            residual_report = check_caption_drift(str(corrected_video_path), str(corrected_captions_path))
            residual_ms = residual_report.median_offset_ms
            summary.issues.append(IssueSummary(
                name="Captions", had_issue=True, detected_offset_ms=cap_report.median_offset_ms,
                fixed=(residual_ms is not None and abs(residual_ms) <= caption_threshold_ms),
                residual_offset_ms=residual_ms,
            ))

        summary.output_files["corrected_captions"] = str(corrected_captions_path)

    (out_dir / "report.txt").write_text(summary.to_text(), encoding="utf-8")
    import json
    (out_dir / "report.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    return summary
