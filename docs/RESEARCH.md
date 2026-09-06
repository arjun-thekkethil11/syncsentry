# Research grounding

SyncSentry exists because A/V and caption synchronization QC in production
OTT pipelines is still, almost everywhere, done with heuristics: check
waveform peaks against frame PTS, check a fixed offset tolerance, flag
outliers by hand. That approach is what's typically built inside content
operations tooling at streaming companies, and it is not wrong -- it's fast,
explainable, and cheap. But it treats every sync problem as "one global
offset," when in reality there are several distinct failure modes that
need different detectors and different remediations.

This document tracks the literature that motivates SyncSentry's design and
maps each paper's finding to a concrete decision in this codebase.

## 1. Sync drift is not just "one offset" -- it has types

**[DiVAS: Video and Audio Synchronization with Dynamic Frame Rates](https://openaccess.thecvf.com/content/CVPR2024/papers/Fernandez-Labrador_DiVAS_Video_and_Audio_Synchronization_with_Dynamic_Frame_Rates_CVPR_2024_paper.pdf)** (CVPR 2024)
identifies four distinct sync failure patterns that a single "detect the
offset" model conflates: **constant offset**, **drift-early**, **drift-late**,
and **intermittent offset**. DiVAS splits a title into dialogue scenes, makes
a per-scene offset prediction, then fits a RANSAC regression line across all
scenes to classify which failure mode is present and how confident the
detection is.

- **Decision this drives:** SyncSentry's coarse detector
  (`syncsentry/detectors/coarse_xcorr.py`) estimates a single global offset
  per asset (Milestone 1/2). Milestone 3a
  (`syncsentry/lipsync/title_drift.py`) implements DiVAS's per-scene +
  RANSAC title-level classification *on top of* that detector -- windowing
  it across a title (`syncsentry/lipsync/scene_offsets.py`) instead of
  swapping in a learned model yet (that's M3b, gated on having real
  face/speech content to validate against). The classification layer itself
  is model-agnostic: it consumes a list of (scene_time, offset,
  confidence) tuples regardless of what produced them, so M3b's pretrained
  embedding model slots into the exact same `title_drift.classify_title_drift`
  without changing it. Validated on all five patterns (in_sync,
  constant_offset, drift_increasing, drift_decreasing, intermittent) against
  synthetic ground truth with a time-varying injected offset
  (`generate_piecewise_offset_fixture`) -- see `docs/ROADMAP.md`, M3a.

## 1b. Real content exposes exactly why the coarse detector isn't enough

M3a's per-window estimator (confidence-gated Tier-1 cross-correlation) was
validated only against synthetic flash+beep content, where global frame
brightness *is* the signal by construction. M3b fetched a real, permissively
licensed talking-head clip (CC BY 3.0, see
`analyzer/fixtures/real_content/SOURCES.md`) specifically to test that
assumption against genuine human speech and a genuine face, and it failed
in an informative way:

- **A single global estimate on real content produces a confident-looking
  but meaningless number.** The clip's audio and video are genuinely in
  sync (injected offset: none). `estimate_av_offset` on it reports
  `offset_ms=880.0` with `confidence=-0.039` -- a negative confidence
  correctly flags this as garbage, but only because the confidence
  calculation happens to go negative here; it isn't a principled bound.
- **The per-window confidence gate (>= 0.3, M3a's guard against exactly
  this) is not sufficient either.** Sliding a 3s window across the same
  clip, 12 of ~29 windows pass the >= 0.3 confidence threshold -- but their
  offset estimates are `{65, 185, 125, 440, 0, -85, 115, 480, 190, 185, 0,
  0}` ms against a true offset of 0ms throughout. Global frame brightness in
  a real talking-head shot (camera holds still, hands and head move,
  lighting doesn't pulse) occasionally correlates with the audio energy
  envelope by coincidence -- often enough to clear a confidence bar tuned
  against clean synthetic pulses, while still being the wrong signal
  entirely. This is a stronger and more specific finding than "the detector
  can be uncertain": it shows that a global-brightness confidence score is
  not a reliable arbiter of *its own trustworthiness* on real content.
- **This is precisely the gap DiVAS/SyncNet-family models close.** They
  don't correlate whole-frame brightness with whole-signal audio energy;
  they extract localized (mouth-region) visual embeddings and compare them
  against audio embeddings in a learned representation space, so a hand
  gesture or a lighting change isn't mistaken for a mouth movement. M3b's
  real-content test suite
  (`analyzer/tests/test_dialogue_scenes_real.py::test_tier1_coarse_detector_is_unreliable_on_real_dialogue_content`)
  pins this finding as a regression test -- asserting the *global* estimate's
  confidence stays low, which is the one part of this behavior we do want to
  rely on (never trust a single global estimate on real content without
  restricting to a scene with a real face and real speech first, which is
  exactly what `syncsentry/lipsync/dialogue_scenes.py` does before any
  learned estimator gets to run).

## 2. Lip-sync detection has moved from heuristic offset search to learned contrastive embeddings

The original **[SyncNet](http://arxiv.org/pdf/2005.08606v1)** approach and
its successors --
**[ModEFormer](https://doi.org/10.1109/icassp49357.2023.10097209)** (ICASSP
2023, modality-preserving transformer embeddings, 94.5%/90.9% on LRS2/LRS3),
**[Interpretable Convolutional SyncNet](https://arxiv.org/html/2409.00971)**
(96.5%/93.8% on LRS2/LRS3 with a balanced-BCE loss that yields an
interpretable in-sync probability), and
**[UniSync](https://arxiv.org/html/2503.16357v1)** (2025, cross-speaker
negative sampling + margin loss for broader representation compatibility) --
all frame sync detection as a *contrastive learning* problem: extract
audio and visual embeddings, and use their similarity as a function of
candidate offset to pick the true offset.

- **Decision this drives:** Milestone 3's fine-grained detector will use a
  pretrained SyncNet-family model for dialogue scenes only (where a face is
  visible and talking), as a complement to -- not a replacement for -- the
  coarse envelope-correlation detector in Milestone 1/2, which works on any
  content but only detects a single global offset and can't resolve
  sub-frame lip-sync-level drift.
- **"Dialogue scene" itself now has a real (not stubbed) definition.**
  M3b implements `syncsentry/lipsync/dialogue_scenes.py`: face-on-screen
  (OpenCV YuNet, see 2b below) intersected with speech-active (Silero VAD)
  time ranges, gap-merged and minimum-duration-filtered. Validated against
  the real clip above: 4 scenes, 25.8s of 50.0s total (52%), matching a
  manual visual check of the footage (the interview cuts to lab-bench props
  and demonstrations for the rest). This is the scene *localization* half
  of M3b; swapping in the pretrained embedding model as the per-scene
  *estimator* (replacing Tier-1 inside those scenes) is the remaining piece.

## 2b. Real face detection has its own platform gotchas -- and a smaller model was the fix

Building M3b surfaced a concrete, and non-obvious, engineering finding
worth documenting alongside the ML ones above: **mediapipe's modern Tasks
API (>=0.10, the current recommended face-detection entry point) crashes on
this project's macOS execution environment**, even when a CPU delegate is
explicitly requested:

```
F0000 ... graph_service.h:139] Check failed: service_ Service is unavailable.
    @ ... -[DrishtiMetalHelper initWithCalculatorContext:]
    @ ... mediapipe::api2::TensorsToDetectionsCalculator::Open()
```

The detection calculator unconditionally initializes a Metal-backed GPU
helper as part of its image-container conversion path, independent of the
delegate requested for the actual tensor inference -- and Metal is
unavailable in this sandboxed/headless execution context. Rather than work
around a GPU-service dependency for what's fundamentally a small CPU-sized
model, M3b switched to `cv2.FaceDetectorYN` (OpenCV's DNN-backed YuNet
face detector, from [opencv_zoo](https://github.com/opencv/opencv_zoo),
Apache-2.0): a ~230KB ONNX model, pure CPU inference through OpenCV's own
DNN backend, no GPU dependency at all. Validated against the same real
clip: face detected in 24/50 one-per-second sampled frames (48%,
confidence 0.84-0.95 when present), consistent with the interview's actual
on-camera/cutaway ratio. Same job, no platform-specific crash -- and the
project already depended on OpenCV for the video-brightness envelope
(`syncsentry/util/ffmpeg_io.py`), so this didn't add a new dependency
family, just a new model file.

## 3. Coarse, model-free detection is a legitimate first tier, with a known limitation

There isn't a single canonical paper for "cross-correlate audio energy with
a video-derived signal" -- it's the workhorse technique behind most
production broadcast QC tooling (and behind the caption-sync automation
built for Fox's OTT pipeline that this project's author shipped in
production). It's included here as Milestone 1/2 specifically *because* it
is what's actually deployed today, and because framing it rigorously (with a
synthetic ground-truth benchmark, explicit tolerance, and a documented
failure mode) is what turns a heuristic into an evaluable detector.

**Known limitation, found and documented via this project's own benchmark:**
cross-correlating a *periodic* pulse train is only unambiguous up to
`± period / 2` -- a lag of `+period/2` and `-period/2` produce identical
correlation peaks (see `tests/test_coarse_detector.py::test_periodic_ambiguity_boundary_is_documented`,
discovered when the first benchmark run aliased a -500ms injected offset on
a 1-second-period fixture into +500ms). Real dialogue audio isn't perfectly
periodic, so this exact failure mode is a synthetic-benchmark artifact more
than a production one, but the *general* lesson -- periodic or
quasi-periodic content (laugh tracks, music beds, repeated SFX) can alias a
correlation-based detector -- is real and is why Milestone 3's per-scene
model exists rather than relying on the coarse detector alone.

## 4. Caption drift is a distinct problem from picture/audio drift

Most of the caption-sync literature is folded into broadcast/timed-text
engineering practice rather than published as ML papers, but the practical
lesson (learned directly from production caption tooling) is: a caption
track can be perfectly aligned to frame PTS and still be wrong, because it
was authored/retimed against a different audio mix, frame rate, or a stale
EDL. Validating captions against *detected speech* rather than against
picture cut points is the only way to catch that class of bug.

- **Decision this drives:** `syncsentry/captions/drift_check.py` deliberately
  ignores the video track and compares WebVTT cue timing only against
  audio-derived speech/onset activity, so "caption vs. speech" drift and
  "picture vs. audio" drift are measured -- and can be root-caused --
  independently.
- **M3b adds real VAD (Silero-VAD) as a separate module
  (`syncsentry/lipsync/vad.py`)** rather than swapping it directly into
  `drift_check.py` yet. Validated against the real clip: 10 speech segments
  covering ~46s of the 50s clip (continuous talking, as expected), versus
  zero segments detected on the synthetic tone-pulse fixture -- confirming
  Silero VAD is doing something meaningfully different from (and not a
  drop-in replacement for, on synthetic content) the energy-threshold onset
  picker `drift_check.py` still uses today. Wiring real VAD into the
  caption checker itself, so it also works against real speech, is
  deferred until there's a real captioned asset to validate that swap
  against -- the same "don't build untested against content that can't
  exercise the code path" discipline as M3a.

## 4b. Fixing sync drift is not the mirror image of detecting it

Once you can detect an offset, "fixing" it looks like it should be trivial
(shift a timestamp). It isn't, for two reasons found empirically while
building `syncsentry/fixer/`:

**`-itsoffset` + stream copy is a metadata-only trick.** The obvious
first implementation -- remux with `ffmpeg -itsoffset <delta> -i audio ...
-c copy` -- changes the *container's* declared timestamp for a stream but
does not insert or remove any samples/frames. Verified with our own
detector: running it on an `-itsoffset`-remuxed file gave back the *exact
same* offset estimate as the original, unchanged. Any consumer that reads
raw decoded samples/frames by index (which is what
`syncsentry.util.ffmpeg_io` does, and what a meaningful fraction of real
players/analysis tools do) never sees a metadata-only "fix." The real fix
re-encodes the audio track with a filter that actually adds/removes samples
(`atrim` to advance, `adelay` to delay) -- see `syncsentry/fixer/av_fix.py`.

**Corrections compose, and composing them by hand is fragile.** If you fix
an A/V offset by trimming audio, you've changed that audio track's entire
timeline. Any caption-drift measurement taken against the *original* audio
is now wrong for the *corrected* file by exactly the amount you just fixed.
`syncsentry/pipeline.py` avoids algebraic composition entirely: every step
re-measures against the actual output of the previous step rather than
trusting a computed correction. This is the same "trust but verify"
discipline as the coarse-detector benchmarking in section 3 above -- and it
caught a real bug in this project's own synthetic fixture generator (the
WebVTT writer silently clamped negative cue timestamps to 0, corrupting one
cue's ground truth for any large negative caption offset -- see
`docs/ROADMAP.md`, M2.5, and the fix in `syncsentry/synth/fixture_gen.py`).

## 5. Adjacent OTT platform research (context, not yet built here)

These informed which project this repo *isn't* (see the top-level project
selection writeup in chat history / README "Why this project" section) but
are worth tracking for future milestones or a follow-up repo:

- **Multi-CDN content steering:** [CADENCE](https://doi.org/10.1145/3793853.3795756),
  [StreamWise](https://doi.org/10.1145/3712676.3714436), and
  [IMAC](https://doi.org/10.1145/3715675.3715839) (all 2025) apply
  multi-agent reinforcement learning to per-session CDN selection under the
  new ETSI TS 103 998 Content Steering standard.
- **Adaptive bitrate streaming generalization:** [SABR](https://arxiv.org/html/2509.10486),
  [SafeSABR](https://arxiv.org/abs/2605.23560v2), and
  [NSMA](https://arxiv.org/html/2607.18845v1) address out-of-distribution
  generalization in learned ABR policies (2025-2026).
- **Content-adaptive encoding:** Netflix's per-title (2015) and per-shot
  Dynamic Optimizer (2018) work, extended by 2024-2025 ML-based convex-hull
  *prediction* research that recovers 85-95% of per-shot bitrate savings at
  10-20x lower compute than brute-force trial encoding.
- **OTT content fingerprinting/anti-piracy:** multimodal redistribution
  detection and compressed-domain (HEVC syntax-element) fingerprinting,
  both active 2025 research areas.
