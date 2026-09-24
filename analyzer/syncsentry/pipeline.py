"""End-to-end detect/fix/verify pipeline for a single asset.

Given a video (and optionally a caption file), produces a corrected
video/captions and a short report.

Order of operations:

  1. Detect the A/V offset on the *original* video.
  2. Fix it (if above threshold), producing a corrected video.
  3. Re-measure the A/V offset on the *corrected* video to verify the fix,
     rather than trusting the detector's own estimate.
  4. If captions were provided: detect caption drift against the
     *corrected* video's audio, not the original. Step 2 may have
     physically shifted the audio track's timeline (e.g. trimmed samples
     off the front), so measuring against the original audio would give
     the wrong offset to apply to the new captions.
  5. Fix caption drift (if above threshold), producing corrected captions.
  6. Re-measure caption drift on the corrected video and captions to
     confirm.

Every step re-measures against the actual output of the previous step
rather than composing corrections algebraically. This is simpler to
reason about and stays robust to changes in how the individual fixers
work.
"""
from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from syncsentry.captions.drift_check import check_caption_drift
from syncsentry.detectors.coarse_xcorr import estimate_av_offset
from syncsentry.fixer.av_fix import fix_av_offset
from syncsentry.fixer.caption_fix import fix_caption_offset
from syncsentry.lipsync import mtdvocalist_offset
from syncsentry.lipsync import syncnet_offset
from syncsentry.lipsync.syncnet_offset import (
    DEFAULT_SYNCNET_MIN_CONFIDENCE,
    SyncNetUnavailable,
    estimate_syncnet_offset,
    is_available as syncnet_is_available,
)
from syncsentry.report.summary import IssueSummary, SyncFixSummary

# Aggregate confidence a *corroborated* (>= 3 agreeing windows) MTDVocaLiST
# estimate needs before it's trusted over SyncNet's own uncorroborated
# whole-track fallback. Set well below the ~8.0 typically seen on a good
# match: corroboration count carries the real precision here, this
# threshold just filters out clearly-nothing tracks.
DEFAULT_MTDVOCALIST_MIN_CONFIDENCE = 5.0

DEFAULT_AV_THRESHOLD_MS = 40.0  # ~1 frame at 25fps; tune per deployment
DEFAULT_CAPTION_THRESHOLD_MS = 80.0
# Matches the confidence gate used by the windowed per-scene estimator
# (syncsentry/lipsync/scene_offsets.py). On real talking-head/dialogue
# content, global frame-brightness cross-correlation can produce
# confident-looking but spurious offsets. Without this gate the pipeline
# could "fix" an offset that isn't really there, and the re-measurement
# afterward (equally unreliable on the same content) could report the
# same bogus number as a "residual", i.e. a fix that did nothing while
# still being labeled FIXED.
DEFAULT_AV_MIN_CONFIDENCE = 0.3

# The evidence bar for declaring "in sync" (a silent no-op: the corrected
# file is a byte-for-byte passthrough) must be at least as strong as the
# bar for declaring "an offset was found" (which gets its own independent
# post-fix verification pass as a safety net). A near-zero offset reported
# off SyncNet's uncorroborated whole-track fallback (`n_confident_windows
# == 0`, no independent window agreed with any other) should not clear
# the same confidence bar and get reported as confidently "in sync"
# without multiple independent windows actually agreeing on it. Matches
# `min_cluster_size` in `syncnet_offset.estimate_syncnet_offset`: the same
# corroboration bar used to accept a *nonzero* offset is required here to
# accept a *zero* one.
MIN_CONFIDENT_WINDOWS_FOR_TRUSTED_IN_SYNC = 3

# Left at the default (`None` -> `estimate_syncnet_offset`'s own 20s)
# rather than shrinking the post-correction verification window to cut
# S3FD face-detection cost (the pipeline's dominant per-call cost, scaling
# close to linearly with frame count). A shorter window can land on a
# stretch of the clip with materially weaker face/speech evidence,
# producing a low-confidence, unusable residual reading instead of a
# confident small residual. Verification only needs to confirm a small
# remaining offset, not search for a possibly-large one, but a safe
# speedup would need a face-presence pre-check to skip shrinking on
# windows that don't have enough coverage, rather than a fixed shorter
# duration. `pipeline._detect_av_offset`'s `max_analyze_duration_s`
# parameter is kept (unused by any call site below) since the plumbing
# itself is still correct.
RESIDUAL_VERIFY_MAX_ANALYZE_DURATION_S = None

# At least this many genuinely-corrected regions are required before
# taking the separate piecewise report/fix path over the normal
# single-global-offset one.
#
# The classical pre-check (`piecewise_offset.MIN_SEGMENTS_FOR_PIECEWISE`)
# requires >= 2 mutually-disagreeing candidate regions before calling a
# clip "piecewise" at all, but that pre-check runs on noisy classical
# cross-correlation: measured directly (see `dialogue_full50s__n500ms`
# in the blind benchmark, a clip with one real, constant -500ms offset
# for its whole duration), it can still report multiple "disagreeing"
# candidates purely from per-chunk noise, not genuine structure. SyncNet
# refinement confirming one of those candidates as trusted does not by
# itself rule that out: the confirmed segment's offset could just be
# this clip's one real global value, sampled in the one window that
# happened to have strong evidence. `PIECEWISE_AGREEMENT_TOLERANCE_MS`
# below is the actual check for that; this constant alone is not enough.
MIN_FIXED_SEGMENTS_FOR_PIECEWISE_PATH = 1

# If every confirmed ("trusted") segment's offset agrees with every
# other confirmed segment's within this many ms, the clip is treated as
# having one real global offset rather than genuine per-region
# structure, and `_maybe_fix_piecewise` backs out (returns `None`) to
# let the normal single-global path handle it instead, cheaper (one
# whole-clip detect/fix/verify instead of N segments' worth) and more
# complete (the single-global path either confidently fixes the whole
# duration or honestly abstains, instead of piecewise's partial,
# per-region "some regions undetermined" report). This is the actual
# gate against the noisy-classical-pre-check failure mode described
# above; requiring >= 2 confirmed segments alone would not catch it,
# since two independently-sampled windows of the same real global
# offset routinely both confirm correctly and still "agree" with each
# other. 150ms, not tighter: SyncNet's own frame quantization (40ms at
# 25fps) plus real per-window noise means even two readings of a truly
# identical offset rarely land bit-for-bit on the same ms value.
PIECEWISE_AGREEMENT_TOLERANCE_MS = 150.0

# If more than this fraction of the video's own duration ends up
# "undetermined" after refinement, the overall issue is reported
# "not_fixed" (attention needed), even when zero regions were confidently
# found bad; see `_maybe_fix_piecewise` for the full reasoning. 20%, not
# 0%: a small undetermined sliver (e.g. a few seconds of no-face
# lead-in/credits) is normal and expected even on content this path
# genuinely resolves. This only trips when most of the timeline is
# unverified rather than resolved.
MAX_UNDETERMINED_FRACTION_FOR_PIECEWISE_FIXED = 0.20

# `_maybe_fix_piecewise`'s own classical pre-check (`detect_piecewise_
# offsets`) decodes and face-scans the *whole* clip before any SyncNet
# work even starts, and before `syncnet_deadline` has any effect on it
# (that deadline only bounds SyncNet calls, not this classical stage).
# Measured directly on a real ~60s clip: 24.2s on its own, on a normal
# dev machine with a full CPU core - on Render's 0.1-CPU free tier this
# would be proportionally far worse, and would blow the whole request's
# 30s budget by itself before a single SyncNet call ever ran, regardless
# of how tight `SYNCSENTRY_SYNCNET_TIMEOUT_S` is set. Piecewise
# (multi-source) content is the rare case (see `_maybe_fix_piecewise`'s
# own docstring: single-source uploads are "the overwhelming common
# case"), so it isn't worth guaranteeing this cost on every request just
# to catch it. Skip attempting the piecewise path entirely whenever the
# configured per-call SyncNet budget is too small to plausibly afford
# both this pre-check and at least one real refinement attempt
# afterward, and fall through to the single-global path instead (one
# whole-clip detect/fix/verify, no whole-video multi-region scan).
# `None` (no per-call cap configured) always allows it, matching that
# setup's existing unbounded behavior.
PIECEWISE_MIN_SYNCNET_BUDGET_S = float(
    os.environ.get("SYNCSENTRY_PIECEWISE_MIN_SYNCNET_BUDGET_S", "45")
)


@dataclass
class _AVDetection:
    offset_ms: float
    confidence: float
    direction: str
    min_confidence: float  # threshold to compare `confidence` against: differs by method
    method: str  # "coarse" | "syncnet" | "mtdvocalist"
    fallback_note: str | None  # set when --use-syncnet was requested but couldn't run
    # Independent corroboration count (SyncNet/MTDVocaLiST only; `None` for
    # "coarse", which has no windowing concept). 0 means the whole-track
    # fallback was used: a single, uncorroborated reading. Declaring
    # "in_sync" (a silent no-op) requires stronger evidence than declaring
    # "offset detected" (which gets an independent post-fix verification
    # pass as its own safety net); holding both to the same bar would let
    # a low-evidence in-sync claim pass silently.
    n_confident_windows: int | None = None
    # True only when `use_syncnet` was requested and it failed with a
    # host-level `RuntimeError` (exceeded `SYNCSENTRY_SYNCNET_TIMEOUT_S`,
    # or the face-tracking subprocess crashed outright), as opposed to a
    # per-video reason ("no trackable face", "not installed"). Callers
    # that make a second SyncNet attempt later in the same request (the
    # post-fix residual verification below) check this to skip that
    # second attempt instead of paying the same timeout twice: a host too
    # CPU-constrained to finish once is essentially certain to be too
    # constrained to finish again moments later on the same file.
    syncnet_timed_out: bool = False


def _remaining_syncnet_budget_s(deadline: float | None) -> float | None:
    """Converts a request-wide SyncNet deadline (a `time.monotonic()`
    timestamp, set once per `run_fix_pipeline` call, see that function)
    into "how many seconds does the *next* SyncNet call get", so several
    SyncNet attempts in one request (piecewise's per-segment refinement,
    its own verification pass, and/or the single-global path's detect +
    verify) share one real ceiling on their combined wall time instead of
    each independently getting the full per-call budget.

    `None` deadline means no request-wide budget is configured
    (`SYNCSENTRY_SYNCNET_TIMEOUT_S` unset): returns `None`, so callers
    fall back to `estimate_syncnet_offset`'s own per-call default
    (unlimited), identical to behavior before this budget existed.
    """
    if deadline is None:
        return None
    return deadline - time.monotonic()


def _detect_av_offset(video_path: str, av_threshold_ms: float, use_syncnet: bool,
                       syncnet_min_confidence: float, av_min_confidence: float,
                       use_mtdvocalist: bool = False,
                       recenter_large_offsets: bool = True,
                       max_analyze_duration_s: float | None = None,
                       syncnet_deadline: float | None = None) -> _AVDetection:
    """Picks which detector actually runs.

    SyncNet is far more accurate on real talking-head content than the
    coarse detector, but takes minutes per video (face detection/tracking
    plus a CNN) versus seconds, so it's opt-in (`use_syncnet=True`) rather
    than the default. Falls back to the coarse detector (with a note
    explaining why) if SyncNet isn't fetched or finds no trackable face.

    `use_mtdvocalist` (default `False`): if SyncNet's own windowed
    corroboration doesn't clear the bar (`n_confident_windows == 0`), and
    this is explicitly requested, also tries MTDVocaLiST, a second,
    independent scorer over the same face-track evidence that works with
    far fewer independent samples than SyncNet needs. Off by default: it
    adds real latency (its own face-detection pass plus dozens of CPU
    transformer calls) for a case that's already ambiguous by definition,
    and its answer isn't automatically more trustworthy than SyncNet's.
    Still worth trying deliberately when SyncNet alone can't decide.

    `recenter_large_offsets` (default `True`): forwarded to
    `estimate_syncnet_offset`. Set `False` for the post-correction
    verification call in `run_fix_pipeline`: a genuine post-fix residual
    should be small (well inside SyncNet's native +-600ms range), and
    re-running the wide-range seed on an already-corrected file is
    unreliable because the trim-based fix leaves the corrected file's
    audio shorter than its video (see `fixer/av_fix.py`), a shape the
    wide-range mouth-motion detector isn't validated against.

    `max_analyze_duration_s` (default `None`, meaning "use
    `estimate_syncnet_offset`'s own default"): forwarded as-is. See
    `RESIDUAL_VERIFY_MAX_ANALYZE_DURATION_S` above for the one caller that
    passes something other than the default.

    `syncnet_deadline` (default `None`, meaning "no request-wide budget,
    each call gets the full per-call default"): a `time.monotonic()`
    deadline shared across every SyncNet call this *request* makes (see
    `run_fix_pipeline`). If the budget is already exhausted by the time
    this call would start, it's treated the same as a timeout (falls back
    to the coarse detector, `syncnet_timed_out=True`) without even
    attempting SyncNet; otherwise the *remaining* time, not the full
    per-call default, is passed as this call's own ceiling.
    """
    fallback_note = None
    timed_out = False
    if use_syncnet:
        remaining_budget_s = _remaining_syncnet_budget_s(syncnet_deadline)
        if remaining_budget_s is not None and remaining_budget_s <= 0:
            fallback_note = ("--use-syncnet requested but this request's shared SyncNet time budget "
                              "was already used up by an earlier attempt; used the coarse detector instead")
            timed_out = True
            coarse = estimate_av_offset(video_path, in_sync_threshold_ms=av_threshold_ms)
            return _AVDetection(coarse.offset_ms, coarse.confidence, coarse.direction,
                                 av_min_confidence, "coarse", fallback_note, syncnet_timed_out=timed_out)

        syncnet_kwargs = {}
        if max_analyze_duration_s is not None:
            syncnet_kwargs["max_analyze_duration_s"] = max_analyze_duration_s
        if remaining_budget_s is not None:
            syncnet_kwargs["timeout_s"] = remaining_budget_s
        try:
            est = estimate_syncnet_offset(video_path, in_sync_threshold_ms=av_threshold_ms,
                                           recenter_large_offsets=recenter_large_offsets, **syncnet_kwargs)
            if use_mtdvocalist and est.n_confident_windows == 0 and mtdvocalist_offset.is_available():
                try:
                    mtdv_est = mtdvocalist_offset.estimate_mtdvocalist_offset(
                        video_path, in_sync_threshold_ms=av_threshold_ms)
                    if mtdv_est.n_confident_windows >= 3:
                        return _AVDetection(mtdv_est.offset_ms, mtdv_est.confidence, mtdv_est.direction,
                                             DEFAULT_MTDVOCALIST_MIN_CONFIDENCE, "mtdvocalist", None,
                                             n_confident_windows=mtdv_est.n_confident_windows)
                except (ValueError, mtdvocalist_offset.MTDVocaLiSTUnavailable):
                    pass  # SyncNet's own (uncorroborated) answer below is still the honest fallback
            return _AVDetection(est.offset_ms, est.confidence, est.direction,
                                 syncnet_min_confidence, "syncnet", None,
                                 n_confident_windows=est.n_confident_windows)
        except SyncNetUnavailable as exc:
            fallback_note = f"--use-syncnet requested but unavailable ({exc}); used the coarse detector instead"
        except ValueError as exc:
            fallback_note = f"--use-syncnet found no trackable face ({exc}); used the coarse detector instead"
        except RuntimeError as exc:
            # Broader net than the two cases above: covers the S3FD
            # subprocess failing outright or exceeding
            # SYNCSENTRY_SYNCNET_TIMEOUT_S on a slow/constrained host.
            # Degrading to the coarse detector here (rather than letting
            # this propagate into a 422 for the whole request, as it did
            # before this existed) keeps the safe-fallback contract:
            # SyncNet being unusable right now on this host is not the
            # same as the request itself failing.
            fallback_note = f"--use-syncnet failed ({exc}); used the coarse detector instead"
            timed_out = True

    coarse = estimate_av_offset(video_path, in_sync_threshold_ms=av_threshold_ms)
    return _AVDetection(coarse.offset_ms, coarse.confidence, coarse.direction,
                         av_min_confidence, "coarse", fallback_note, syncnet_timed_out=timed_out)


def _maybe_fix_piecewise(video_path: str, corrected_video_path: Path, av_threshold_ms: float,
                          av_min_confidence: float, syncnet_min_confidence: float,
                          syncnet_deadline: float | None = None) -> IssueSummary | None:
    """Attempts the piecewise (independent per-region) A/V sync path.

    See `syncsentry.lipsync.piecewise_offset`'s module docstring for why
    this exists: content assembled from multiple independently-offset
    sources (e.g. a multi-camera edit, or clips stitched together from
    different recordings) violates the single-global-offset assumption
    every other branch of this pipeline makes. Confidently applying one
    detected offset to the whole file in that case only fixes the region
    the offset came from and makes every other region's error worse.

    Returns `None` (falls through to the normal single-global-offset path,
    unchanged) whenever there isn't clear, multi-region evidence that a
    single offset genuinely doesn't describe this file. Deliberately
    conservative: a single-source upload, the overwhelming common case,
    must never take this branch by mistake, and every gate here (the
    classical pre-check's own `piecewise_offset.
    MIN_SEGMENTS_FOR_PIECEWISE`, plus `MIN_FIXED_SEGMENTS_FOR_PIECEWISE_PATH`
    below) exists specifically to enforce that.
    """
    from syncsentry.fixer.piecewise_fix import fix_piecewise_offsets
    from syncsentry.lipsync.piecewise_offset import (
        OffsetSegment,
        detect_piecewise_offsets,
        refine_piecewise_segments_with_syncnet,
    )

    try:
        pre = detect_piecewise_offsets(video_path)
    except ValueError:
        return None  # not enough dialogue-scene signal to attempt this; normal path handles it
    if not pre.is_piecewise:
        return None

    # Shared across both refinement passes below (this one, and the
    # post-fix verification pass further down): the first segment whose
    # SyncNet call times out on this host trips it, and every later
    # segment in either pass then skips its own doomed attempt instead of
    # re-paying the same timeout. See `refine_piecewise_segments_with_
    # syncnet`'s docstring.
    syncnet_abort = threading.Event()

    refined = refine_piecewise_segments_with_syncnet(video_path, pre.segments,
                                                       min_confidence=syncnet_min_confidence,
                                                       abort_event=syncnet_abort,
                                                       syncnet_deadline=syncnet_deadline)
    fixable = [s for s in refined if s.status == "trusted" and s.offset_ms is not None
               and abs(s.offset_ms) > av_threshold_ms]
    if len(fixable) < MIN_FIXED_SEGMENTS_FOR_PIECEWISE_PATH:
        # SyncNet refinement didn't corroborate enough independently
        # different regions to justify the piecewise path (it may have
        # downgraded some or all of the classical pre-check's candidates
        # to "undetermined"; see that function's docstring). Let the
        # normal single-global path make its own attempt instead of
        # reporting a piecewise result with little or nothing to show
        # for it.
        return None

    # Every confirmed segment agreeing with every other one (see
    # `PIECEWISE_AGREEMENT_TOLERANCE_MS` above) means the classical
    # pre-check's "these regions disagree" signal was noise, not real
    # structure: this clip has one real global offset, just sampled
    # correctly in more than one window.
    #
    # Apply that already-confirmed value directly rather than discarding
    # it and telling the caller to fall back to
    # `_fix_single_global_offset`, which would re-run a full, fresh
    # whole-clip SyncNet detection pass to very likely reconfirm the same
    # number this refinement pass just spent real time establishing.
    # Still pays for one honest post-fix verification pass (never skip
    # that, see module docstring), just not a second detection pass on
    # top of it.
    fixable_offsets = [s.offset_ms for s in fixable]
    if max(fixable_offsets) - min(fixable_offsets) <= PIECEWISE_AGREEMENT_TOLERANCE_MS:
        confirmed_offset_ms = statistics.median(fixable_offsets)
        fix_av_offset(video_path, corrected_video_path, confirmed_offset_ms)
        # `use_syncnet=not syncnet_abort.is_set()`: if refinement above
        # already hit this host's SyncNet timeout on some other segment,
        # skip straight to the coarse detector here too rather than
        # paying that same timeout again for a verification pass.
        residual_det = _detect_av_offset(str(corrected_video_path), av_threshold_ms,
                                          use_syncnet=not syncnet_abort.is_set(),
                                          syncnet_min_confidence=syncnet_min_confidence,
                                          av_min_confidence=av_min_confidence,
                                          recenter_large_offsets=False,
                                          max_analyze_duration_s=RESIDUAL_VERIFY_MAX_ANALYZE_DURATION_S,
                                          syncnet_deadline=syncnet_deadline)
        residual_confident = residual_det.confidence >= residual_det.min_confidence
        fixed = residual_confident and abs(residual_det.offset_ms) <= av_threshold_ms
        note = (f"initially looked like it might have multiple independently-offset regions, but the "
                f"{len(fixable)} confirmed region(s) agreed on one offset ({confirmed_offset_ms:+.0f}ms), "
                f"so this was treated as one global correction instead of a partial per-region fix")
        if not residual_confident:
            note += (f"; residual re-check was itself low-confidence ({residual_det.confidence:.2f} < "
                     f"{residual_det.min_confidence:.2f}), so the fix could not be independently confirmed")
        elif not fixed:
            note += (f"; residual re-check confidently found a remaining {residual_det.offset_ms:+.0f}ms "
                     f"offset after the applied correction")
        return IssueSummary(
            name="A/V sync", had_issue=True, detected_offset_ms=confirmed_offset_ms, fixed=fixed,
            residual_offset_ms=(residual_det.offset_ms if residual_confident else None), note=note,
            status=("fixed" if fixed else "not_fixed"), confidence=residual_det.confidence,
            min_confidence=residual_det.min_confidence, method="syncnet",
        )

    fix_piecewise_offsets(video_path, corrected_video_path, refined)

    # Verification: re-check the corrected output, but only at the exact
    # time ranges that were actually shifted (`fixable`), not a second
    # full-clip re-scan across every region (including ones already left
    # "undetermined"/untouched, or already-in-sync regions the fixer
    # never touched). A full re-scan would re-discover the same candidate
    # regions (the corrected file's content is unchanged aside from a few
    # regions' shifted audio) and re-pay a full SyncNet call for regions
    # that were never fixed and have nothing new to verify; cost would
    # scale with how many candidate regions exist rather than how many
    # were actually changed.
    #
    # Building the verification segment list directly from `fixable`
    # (same start_s/end_s the fixer itself just used) is exact:
    # `fix_piecewise_offsets` only shifts audio within each segment's
    # existing time range, so re-checking that same range in the
    # corrected file checks precisely the span whose offset should now
    # read near zero if the fix worked.
    #
    # A genuinely successful per-region fix should leave no such range
    # SyncNet can confidently re-corroborate as still offset by more than
    # `av_threshold_ms`, mirroring the single-global path's own post-fix
    # re-measurement (see `run_fix_pipeline` below). An inconclusive
    # verification (no confident re-reading either way) is treated
    # leniently, failing open toward "fixed", rather than paying a third
    # full SyncNet pass per segment to disambiguate. This is a
    # deliberately simpler bar than the single-global path's, given the
    # added cost of getting a segment-level answer at all.
    verify_segments = [
        OffsetSegment(start_s=s.start_s, end_s=s.end_s, offset_ms=s.offset_ms,
                       confidence=s.confidence, status="trusted", n_chunks=s.n_chunks)
        for s in fixable
    ]
    verify_refined = refine_piecewise_segments_with_syncnet(
        str(corrected_video_path), verify_segments, min_confidence=syncnet_min_confidence,
        abort_event=syncnet_abort, syncnet_deadline=syncnet_deadline)
    residual_confident_and_bad = any(
        s.status == "trusted" and s.offset_ms is not None and abs(s.offset_ms) > av_threshold_ms
        for s in verify_refined
    )

    seg_dicts = [
        {"start_s": round(s.start_s, 2), "end_s": round(s.end_s, 2), "offset_ms": s.offset_ms, "status": s.status}
        for s in refined
    ]

    undetermined_fraction = _undetermined_duration_fraction(refined)
    substantial_undetermined = undetermined_fraction > MAX_UNDETERMINED_FRACTION_FOR_PIECEWISE_FIXED

    note = (f"content doesn't fit a single global A/V offset: found {len(refined)} region(s) with "
            f"independently different sync, corrected {len(fixable)} of them independently rather than "
            f"applying one offset to the whole file (see 'segments' for the per-region breakdown)")
    if substantial_undetermined:
        note += (f"; {undetermined_fraction:.0%} of the video's duration could not be independently "
                 f"confirmed either way and is left unmodified. Treat this as a partial, not complete, fix")

    fixed = not residual_confident_and_bad and not substantial_undetermined
    status = "not_fixed" if (residual_confident_and_bad or substantial_undetermined) else "fixed"

    return IssueSummary(
        name="A/V sync", had_issue=True, detected_offset_ms=None, fixed=fixed,
        residual_offset_ms=None, note=note, status=status,
        method="piecewise", confidence=None, min_confidence=None, segments=seg_dicts,
    )


def _undetermined_duration_fraction(segments) -> float:
    """Fraction of `segments`' total duration whose `status ==
    "undetermined"`.

    Used by `_maybe_fix_piecewise` (see
    `MAX_UNDETERMINED_FRACTION_FOR_PIECEWISE_FIXED`) to stop a piecewise
    result from claiming `status="fixed"` (and therefore
    `SyncFixSummary.all_resolved`) when a large fraction of the video's
    own duration is still genuinely unverified after refinement, e.g. a
    candidate the classical pre-check flagged but SyncNet couldn't
    corroborate (a deliberate abstention per
    `refine_piecewise_segments_with_syncnet`'s docstring, not a bug).
    Without this check, correcting only a couple of several flagged
    regions could still produce an "all issues resolved" claim while most
    of the timeline is left honestly "undetermined" in the per-segment
    breakdown. That's the same false-confidence failure mode
    `IssueSummary.status` and `MIN_CONFIDENT_WINDOWS_FOR_TRUSTED_IN_SYNC`
    above exist to close for the single-global path, reached via a
    different route: aggregating several honest per-segment abstentions
    into one dishonest overall claim.

    Returns 0.0 for an empty list (nothing to be undetermined about).
    """
    total_s = sum(s.end_s - s.start_s for s in segments)
    if total_s <= 0:
        return 0.0
    undetermined_s = sum(s.end_s - s.start_s for s in segments if s.status == "undetermined")
    return undetermined_s / total_s


def run_fix_pipeline(video_path: str | Path, out_dir: str | Path,
                      captions_path: str | Path | None = None,
                      av_threshold_ms: float = DEFAULT_AV_THRESHOLD_MS,
                      caption_threshold_ms: float = DEFAULT_CAPTION_THRESHOLD_MS,
                      av_min_confidence: float = DEFAULT_AV_MIN_CONFIDENCE,
                      use_syncnet: bool = False,
                      syncnet_min_confidence: float = DEFAULT_SYNCNET_MIN_CONFIDENCE,
                      use_mtdvocalist: bool = False) -> SyncFixSummary:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = SyncFixSummary(asset_name=video_path.name)
    corrected_video_path = out_dir / f"{video_path.stem}.corrected{video_path.suffix}"

    # --- Step 0: piecewise pre-check ---
    # Only attempted with SyncNet both requested and actually installed:
    # the classical pre-check alone is not reliable enough to apply a
    # correction from directly (see `piecewise_offset.py`'s own
    # docstring), and refinement needs SyncNet itself. Checking
    # `syncnet_is_available()` here, not just `use_syncnet`, matters on
    # deployments where SyncNet's weights were never fetched: without it,
    # this branch would still pay for a real face-detection and
    # speech-detection scan of the whole video (dialogue-scene
    # localization) only to discover at the refinement step that SyncNet
    # isn't there, wasting real time on CPU-constrained hosts for no
    # possible benefit. Deliberately conservative otherwise (see
    # `_maybe_fix_piecewise`'s docstring for every other gate involved):
    # returns `None`, falling through to the unchanged single-global-offset
    # path below, for the overwhelming common case of single-source
    # content.
    # Shared, request-wide ceiling on *all* SyncNet-related work this call
    # does, however many separate attempts that ends up being (piecewise
    # per-segment refinement, its own verification pass, and/or the
    # single-global path's detect + verify). `SYNCSENTRY_SYNCNET_TIMEOUT_S`
    # bounds one SyncNet call on its own (see `syncnet_offset.
    # SYNCNET_TIMEOUT_S`), but on real dialogue content the piecewise path
    # alone can make several such calls before concluding there isn't
    # enough evidence to take that path, then still falls through to the
    # single-global path's own attempt, each individually finishing well
    # inside the per-call budget yet adding up to several times it.
    # Measured directly: one real ~60s clip that never hit the per-call
    # timeout at all still took 68s wall time this way. Computed once here
    # (`None` when no per-call budget is configured, matching that
    # setup's existing unbounded behavior) and threaded through every
    # attempt below via `syncnet_deadline`/`_remaining_syncnet_budget_s`,
    # so the *last* attempt any given request makes gets whatever's
    # actually left, not a fresh full budget it has no real claim to.
    syncnet_deadline = (
        time.monotonic() + syncnet_offset.SYNCNET_TIMEOUT_S
        if use_syncnet and syncnet_offset.SYNCNET_TIMEOUT_S is not None
        else None
    )

    piecewise_worth_attempting = (
        syncnet_offset.SYNCNET_TIMEOUT_S is None
        or syncnet_offset.SYNCNET_TIMEOUT_S >= PIECEWISE_MIN_SYNCNET_BUDGET_S
    )
    piecewise_issue = None
    if use_syncnet and syncnet_is_available() and piecewise_worth_attempting:
        piecewise_issue = _maybe_fix_piecewise(str(video_path), corrected_video_path,
                                                av_threshold_ms, av_min_confidence, syncnet_min_confidence,
                                                syncnet_deadline=syncnet_deadline)

    if piecewise_issue is not None:
        summary.issues.append(piecewise_issue)
    else:
        # --- Step 1-3: A/V sync (single global offset) ---
        summary.issues.append(_fix_single_global_offset(
            video_path, corrected_video_path, av_threshold_ms, av_min_confidence,
            use_syncnet, syncnet_min_confidence, use_mtdvocalist,
            syncnet_deadline=syncnet_deadline,
        ))

    summary.output_files["corrected_video"] = str(corrected_video_path)

    # --- Step 4-6: captions (measured against the CORRECTED video) ---
    if captions_path is not None:
        captions_path = Path(captions_path)
        cap_report = check_caption_drift(str(corrected_video_path), str(captions_path))
        corrected_captions_path = out_dir / f"{captions_path.stem}.corrected{captions_path.suffix}"

        cap_matched, cap_unmatched = cap_report.matched_count, cap_report.unmatched_count
        cap_total = cap_matched + cap_unmatched
        cap_confidence = (cap_matched / cap_total) if cap_total > 0 else None

        if cap_report.median_offset_ms is None:
            # No speech onsets to compare against at all: genuinely
            # undetermined, not confirmed in sync (same principle as the
            # A/V "undetermined" states above).
            note = "no matching speech onsets found"
            corrected_captions_path.write_text(captions_path.read_text(encoding="utf-8"), encoding="utf-8")
            summary.issues.append(IssueSummary(
                name="Captions", had_issue=False, detected_offset_ms=None,
                fixed=False, residual_offset_ms=None, note=note, status="undetermined",
                confidence=cap_confidence, matched_count=cap_matched, unmatched_count=cap_unmatched,
            ))
        elif abs(cap_report.median_offset_ms) < caption_threshold_ms:
            corrected_captions_path.write_text(captions_path.read_text(encoding="utf-8"), encoding="utf-8")
            summary.issues.append(IssueSummary(
                name="Captions", had_issue=False, detected_offset_ms=cap_report.median_offset_ms,
                fixed=False, residual_offset_ms=None, status="in_sync",
                confidence=cap_confidence, matched_count=cap_matched, unmatched_count=cap_unmatched,
            ))
        else:
            fix_caption_offset(captions_path, corrected_captions_path, cap_report.median_offset_ms)
            residual_report = check_caption_drift(str(corrected_video_path), str(corrected_captions_path))
            residual_ms = residual_report.median_offset_ms
            cap_fixed = residual_ms is not None and abs(residual_ms) <= caption_threshold_ms
            summary.issues.append(IssueSummary(
                name="Captions", had_issue=True, detected_offset_ms=cap_report.median_offset_ms,
                fixed=cap_fixed, status=("fixed" if cap_fixed else "not_fixed"),
                residual_offset_ms=residual_ms,
                confidence=cap_confidence, matched_count=cap_matched, unmatched_count=cap_unmatched,
            ))

        summary.output_files["corrected_captions"] = str(corrected_captions_path)

    (out_dir / "report.txt").write_text(summary.to_text(), encoding="utf-8")
    import json
    (out_dir / "report.json").write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    return summary


def _fix_single_global_offset(video_path: Path, corrected_video_path: Path, av_threshold_ms: float,
                               av_min_confidence: float, use_syncnet: bool, syncnet_min_confidence: float,
                               use_mtdvocalist: bool, syncnet_deadline: float | None = None) -> IssueSummary:
    """The A/V sync detect/fix/verify logic for the single-global-offset
    path, extracted into its own function so `run_fix_pipeline` can
    choose between this and the piecewise path above without one giant
    nested conditional."""
    det = _detect_av_offset(str(video_path), av_threshold_ms, use_syncnet,
                             syncnet_min_confidence, av_min_confidence, use_mtdvocalist,
                             syncnet_deadline=syncnet_deadline)
    method_tag = "" if det.method == "coarse" else f" [{det.method}]"

    if det.confidence < det.min_confidence:
        # Not "in sync below threshold": we genuinely can't tell. Treating
        # this as "no issue" (rather than guessing a fix from noise) is the
        # honest answer. This case is most common on real dialogue/
        # talking-head content, where the confidence signal is weaker.
        corrected_video_path.write_bytes(video_path.read_bytes())
        note = (f"low-confidence detection{method_tag}: raw estimate {det.offset_ms:+.0f}ms "
                f"({det.direction}), confidence {det.confidence:.2f} (threshold {det.min_confidence:.2f}), "
                f"too low to trust; not applied automatically. If your own check agrees with the "
                f"direction, re-run with a lower confidence threshold to force it, but treat the "
                f"result as unverified")
        if det.fallback_note:
            note = f"{det.fallback_note}. {note}"
        return IssueSummary(
            name="A/V sync", had_issue=False, detected_offset_ms=det.offset_ms,
            fixed=False, residual_offset_ms=None, note=note, status="undetermined",
            confidence=det.confidence, min_confidence=det.min_confidence, method=det.method,
        )
    elif (det.direction == "in_sync"
          and (det.n_confident_windows is None
               or det.n_confident_windows >= MIN_CONFIDENT_WINDOWS_FOR_TRUSTED_IN_SYNC)):
        # Genuinely confirmed in sync: either the coarse detector (no
        # windowing concept, `n_confident_windows is None`) or
        # SyncNet/MTDVocaLiST with real independent-window corroboration,
        # not just one uncorroborated whole-track reading.
        corrected_video_path.write_bytes(video_path.read_bytes())
        return IssueSummary(
            name="A/V sync", had_issue=False, detected_offset_ms=det.offset_ms,
            fixed=False, residual_offset_ms=None, note=det.fallback_note, status="in_sync",
            confidence=det.confidence, min_confidence=det.min_confidence, method=det.method,
        )
    elif det.direction == "in_sync":
        # Reports a near-zero offset, but without the independent-window
        # corroboration required above: a single uncorroborated
        # whole-track reading landed near zero. Do not report this as a
        # confirmed "in sync", since that would be a false negative.
        # Treated the same as the low-confidence branch above (an honest
        # "can't confirm", not a claim either way), but with its own note
        # naming the specific reason (lack of corroboration rather than
        # raw confidence) so the two are distinguishable.
        corrected_video_path.write_bytes(video_path.read_bytes())
        note = (f"uncorroborated near-zero detection{method_tag}: raw estimate {det.offset_ms:+.0f}ms, "
                f"confidence {det.confidence:.2f} clears the trust threshold but only one whole-track "
                f"reading supports it. No independent windows agreed (need >= "
                f"{MIN_CONFIDENT_WINDOWS_FOR_TRUSTED_IN_SYNC}). Not confident enough to call this "
                f"synchronized; treat as unverified rather than assuming no correction is needed")
        if det.fallback_note:
            note = f"{det.fallback_note}. {note}"
        return IssueSummary(
            name="A/V sync", had_issue=False, detected_offset_ms=det.offset_ms,
            fixed=False, residual_offset_ms=None, note=note, status="undetermined",
            confidence=det.confidence, min_confidence=det.min_confidence, method=det.method,
        )
    else:
        fix_av_offset(video_path, corrected_video_path, det.offset_ms)
        # `recenter_large_offsets=False`: independent verification, not a
        # re-run of the same estimator that selected the correction. See
        # the docstring on `_detect_av_offset`.
        #
        # `use_syncnet and not det.syncnet_timed_out`: if the detection
        # call above already hit SyncNet's own host-level timeout, this
        # host is essentially certain to time out again on the same
        # file's corrected copy moments later. Skipping straight to the
        # coarse detector here avoids paying that same multi-second-to-
        # multi-minute timeout a second time in one request for no
        # realistic chance of a different outcome; see
        # `_AVDetection.syncnet_timed_out`.
        residual_det = _detect_av_offset(str(corrected_video_path), av_threshold_ms,
                                          use_syncnet and not det.syncnet_timed_out,
                                          syncnet_min_confidence, av_min_confidence,
                                          recenter_large_offsets=False,
                                          max_analyze_duration_s=RESIDUAL_VERIFY_MAX_ANALYZE_DURATION_S,
                                          syncnet_deadline=syncnet_deadline)
        residual_confident = residual_det.confidence >= residual_det.min_confidence
        fixed = residual_confident and abs(residual_det.offset_ms) <= av_threshold_ms
        verify_note = det.fallback_note
        if not fixed:
            # Be explicit about why verification didn't confirm the fix.
            # A bare, unexplained residual number (especially a
            # low-confidence one) reads as "the fix made it worse", which
            # is often not what happened: an un-gated residual
            # re-measurement can itself be a spurious, low-confidence
            # reading with no indication that it shouldn't be trusted.
            confidence_note = (
                f"residual re-check was itself low-confidence ({residual_det.confidence:.2f} < "
                f"{residual_det.min_confidence:.2f}). The correction may still be correct; this just "
                f"means the fix could not be independently confirmed"
                if not residual_confident else
                f"residual re-check confidently found a remaining {residual_det.offset_ms:+.0f}ms offset "
                f"after the applied correction"
            )
            verify_note = f"{verify_note}. {confidence_note}" if verify_note else confidence_note
        return IssueSummary(
            name="A/V sync", had_issue=True, detected_offset_ms=det.offset_ms,
            fixed=fixed,
            residual_offset_ms=residual_det.offset_ms, residual_confidence=residual_det.confidence,
            note=verify_note, status=("fixed" if fixed else "not_fixed"),
            confidence=det.confidence, min_confidence=det.min_confidence, method=det.method,
        )
