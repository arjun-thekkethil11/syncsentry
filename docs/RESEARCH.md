# Research grounding

Production A/V/caption sync QC is almost always heuristic: waveform peaks
vs. frame PTS, a fixed tolerance, manual QC on outliers. Fast and cheap, but
it treats every sync bug as "one global offset," when the literature below
shows drift has distinct types that need different detection and fixes.
This doc maps each paper's finding to a concrete decision in this codebase.

## 1. Sync drift has types (DiVAS, CVPR 2024)

[DiVAS](https://openaccess.thecvf.com/content/CVPR2024/papers/Fernandez-Labrador_DiVAS_Video_and_Audio_Synchronization_with_Dynamic_Frame_Rates_CVPR_2024_paper.pdf)
identifies four failure patterns a single "detect the offset" model
conflates: constant offset, drift-early, drift-late, intermittent. It
splits a title into dialogue scenes, predicts per-scene offsets, then fits
a RANSAC line across scenes to classify the pattern.

**Decision:** `syncsentry/detectors/coarse_xcorr.py` gives a single global
offset (M1/M2). `syncsentry/lipsync/title_drift.py` (M3a) implements
DiVAS's per-scene + RANSAC classification on top of it -- windowed
(`scene_offsets.py`) rather than a learned model yet, since the
classification layer is model-agnostic (consumes `(time, offset,
confidence)` tuples regardless of source, so M3c's model is a drop-in
replacement). Validated on all 5 patterns against synthetic ground truth
with time-varying injected offsets.

## 1b. Real content breaks the confidence gate M3a relies on

M3a was only validated on synthetic flash+beep content, where global frame
brightness *is* the signal. Tested against a real CC BY 3.0 talking-head
clip (see `analyzer/fixtures/real_content/SOURCES.md`) with true offset =
0ms:

- A single global estimate reports `offset_ms=880`, `confidence=-0.04` --
  negative confidence happens to flag this as garbage, but that's not a
  principled bound.
- Worse: sliding a 3s window across the same clip, 12 of ~29 windows *pass*
  the >=0.3 confidence gate -- with offsets ranging `0` to `480ms`, all
  "confident," none correct. Global brightness spuriously correlates with
  real speech energy often enough to fool a threshold tuned on clean
  pulses.

This is exactly the gap SyncNet/DiVAS-style models close: they compare
localized (mouth-region) visual embeddings against audio embeddings in a
learned space, not whole-frame brightness against whole-signal energy. Real
dialogue-scene detection (§2 below) exists specifically so a learned
estimator only ever runs where a real face + real speech are confirmed
present, rather than trusting this gate alone. Pinned as a regression test.

## 1c. That finding became a real product bug, caught by dogfooding

Running `syncsentry fix` on a real "learn English conversation" video
produced `detected -55ms -> FIXED (residual -55.0ms)`, while the overall
line correctly said "some issues remain" -- a contradiction in the same
report. Two real bugs, on top of the detector limitation above:

- `run_fix_pipeline` never checked `estimate.confidence`, only
  `direction == "in_sync"` -- so a low/negative-confidence reading (§1b's
  failure mode) was treated as real and "fixed" by trimming actual audio
  off a file that likely had no problem.
- `fixed=True` was set the moment a fix was *attempted*, not when the
  re-measured residual confirmed it worked.

**Fix:** gate the A/V step on `confidence >= 0.3` before deciding there's
an issue at all (skip + report "low-confidence, undetermined" below that);
compute `fixed` from the actual residual, not assumed. Also found: the
`--av-threshold-ms` CLI flag was accepted but never wired to the detector --
now fixed. All pinned in `analyzer/tests/test_pipeline.py`, including the
exact real-content scenario above (now asserts a byte-for-byte passthrough
instead of an unverifiable edit).

## 2. Lip-sync detection has moved to learned contrastive embeddings

[SyncNet](http://arxiv.org/pdf/2005.08606v1) and successors --
[ModEFormer](https://doi.org/10.1109/icassp49357.2023.10097209) (ICASSP
2023, 94.5%/90.9% on LRS2/LRS3), [Interpretable Conv-SyncNet](https://arxiv.org/html/2409.00971)
(96.5%/93.8%), [UniSync](https://arxiv.org/html/2503.16357v1) (2025) --
frame sync as *contrastive learning*: audio/visual embedding similarity as
a function of candidate offset picks the true offset.

**Decision:** M3's fine-grained detector will use a pretrained SyncNet-family
model, restricted to real dialogue scenes (face + speech both present),
complementing rather than replacing the coarse detector.

**Dialogue scenes are now real, not stubbed.** `syncsentry/lipsync/dialogue_scenes.py`
intersects face-presence and speech-activity intervals (gap-merged,
min-duration filtered). Validated on the real clip: 4 scenes, 25.8s of
50.0s (52%), matching a manual check. This is the scene *localization*
half of M3b; the learned *estimator* swap is M3c.

## 2b. mediapipe crashes here; YuNet doesn't

mediapipe's Tasks API (the "obvious" face-detector choice) crashes on this
project's macOS environment even with a CPU delegate requested
(`Check failed: service_ Service is unavailable`, from a Metal-backed
image-pipeline helper that initializes regardless of the requested
delegate). Switched to `cv2.FaceDetectorYN` (YuNet, [opencv_zoo](https://github.com/opencv/opencv_zoo),
Apache-2.0): ~230KB ONNX, pure CPU, no GPU dependency. Validated: face
detected in 24/50 sampled frames of the real clip (48%, confidence
0.84-0.95), matching the interview's actual on-camera ratio.

## 3. Coarse cross-correlation is a legitimate first tier, with a known limit

Not from a paper -- it's the workhorse behind production broadcast QC
(including caption-sync tooling this project's author shipped at Fox).
Included as M1/M2 because it's what's actually deployed, framed rigorously
(synthetic ground truth, explicit tolerance, documented failure mode).

**Known limitation:** cross-correlating a periodic pulse train is
unambiguous only up to `±period/2` -- found via the benchmark itself (a
-500ms injection aliased to +500ms on a 1s-period fixture), now a pinned
regression test. Real dialogue audio isn't perfectly periodic, so this
exact case is a synthetic-benchmark artifact, but the general lesson
(periodic/quasi-periodic content can alias this detector) motivated M3's
per-scene approach.

## 4. Caption drift is a distinct problem from picture/audio drift

A caption track can be perfectly aligned to frame PTS and still be wrong
(authored against a different mix/frame-rate/stale EDL). Validating against
*detected speech*, not picture cuts, is the only way to catch that.

**Decision:** `syncsentry/captions/drift_check.py` ignores video entirely,
comparing WebVTT timing only to audio-derived speech activity, so the two
drift types are measured (and root-caused) independently. Real VAD
(Silero, `syncsentry/lipsync/vad.py`) is validated separately -- 10 speech
segments over ~46s of a 50s real clip vs. zero on the synthetic tone
fixture -- but not yet wired into `drift_check.py` itself, deferred until
there's a real captioned asset to validate that swap against.

## 4b. Fixing drift isn't the mirror image of detecting it

Two things found empirically while building `syncsentry/fixer/`:

- **`-itsoffset` + stream copy is metadata-only.** It changes the
  container's declared timestamp but inserts/removes no samples -- verified
  by re-running our own detector on the remuxed file and getting back the
  *exact same* offset. The real fix re-encodes audio with `atrim`/`adelay`
  to actually shift samples.
- **Corrections compose, and composing by hand is fragile.** Fixing A/V
  offset changes the audio track's whole timeline, so a caption-drift
  measurement against the *original* audio is now wrong. `pipeline.py`
  re-measures against each step's actual output instead of composing
  corrections algebraically -- the same discipline that caught this
  project's own fixture-generator bug (WebVTT writer clamping negative cue
  timestamps to 0).

## 5. Adjacent OTT research (context, not built here)

Scoped and set aside in favor of this project's tighter fit with existing
production experience (caption sync, VOD/iVOD tooling at Fox):

- **Multi-CDN steering:** [CADENCE](https://doi.org/10.1145/3793853.3795756),
  [StreamWise](https://doi.org/10.1145/3712676.3714436), [IMAC](https://doi.org/10.1145/3715675.3715839)
  (2025) -- multi-agent RL for per-session CDN selection (ETSI TS 103 998).
- **ABR generalization:** [SABR](https://arxiv.org/html/2509.10486),
  [SafeSABR](https://arxiv.org/abs/2605.23560v2), [NSMA](https://arxiv.org/html/2607.18845v1)
  (2025-2026).
- **Content-adaptive encoding:** Netflix per-title/per-shot Dynamic
  Optimizer, extended by ML convex-hull *prediction* (85-95% of savings at
  10-20x lower compute).
- **Anti-piracy fingerprinting:** multimodal + compressed-domain (HEVC)
  detection, active 2025 research.
