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

Follow-up from a second real user report on a different real talking-head
video ("the fix is not correct"): the skip message originally hid the raw
estimate entirely, showing only a confidence score -- unhelpful if the user
has their own (visual) evidence of a real offset and wants to judge the
number themselves. Fixed: the note now always states the raw
`offset_ms`/`direction` alongside confidence, plus the exact
`--av-min-confidence` value needed to force it (explicitly labeled
unverified).

## 1d. A more targeted classical heuristic was tried -- and also falsified

§1b's fix (skip below confidence 0.3) is honest but doesn't answer "so what
*should* run on real dialogue content instead?" One hypothesis: whole-frame
brightness fails because the mouth is a tiny fraction of the frame -- so
restrict the visual signal to just the mouth (via YuNet's mouth-corner
landmarks, already available from M3b) and just real dialogue scenes
(`lipsync/mouth_offset.py`), same cross-correlation core, more targeted
input.

Tested properly, not just eyeballed: `fix_av_offset(clip, out, -X)` on the
(genuinely in-sync) real clip injects a *known* true offset of `+X`ms --
turning it into a labeled real-content dataset of one. Across four injected
offsets (0, +150, -200, +300ms), the mouth-motion estimate did not track
ground truth at all (e.g. true `+300ms` -> estimated `-33ms`), and
confidence stayed low throughout (~0.14-0.20), same as the failing
whole-frame method. Frame-to-frame pixel-difference motion in a mouth crop
just isn't a strong enough feature -- consistent with why SyncNet-family
work (§2) uses *learned* embeddings rather than hand-crafted motion energy.
Kept as a documented negative result and pinned in
`analyzer/tests/test_mouth_offset_real.py` (asserts confidence stays low,
so a future change that starts reporting high confidence without actually
being re-validated against known offsets gets caught, not shipped). This
is the concrete evidence that M3c genuinely needs a pretrained model, not
just a smaller ROI.

## 1e. The actual pretrained model (M3c) validates -- with real numbers

Rather than re-implement SyncNet's exact preprocessing (face-track cropping
at 224x224, MFCC windows, the CNN itself) and risk another subtle bug like
§1d, `syncsentry/lipsync/syncnet_offset.py` wraps the original
[joonson/syncnet_python](https://github.com/joonson/syncnet_python) (MIT)
as an external tool -- vendored via `analyzer/scripts/fetch_syncnet.sh`,
never committed (`analyzer/third_party/SOURCES.md`), same policy as the
real-content clip.

Same injected-known-offset methodology as §1d, now on the model this
project actually ships: `fix_av_offset` used as an offset *injector* on the
genuinely-in-sync real clip, 3 independently-detected face tracks per run.

| true offset | SyncNet estimate (median of 3 tracks) | confidence (median) |
|---|---|---|
| 0ms | 0ms (frame offsets: 0, +1, 0) | 6.8 |
| +150ms | +133ms (-4, -3, -3 frames) | 7.0 |
| -200ms | -213ms (+5, +6, +5 frames) | 6.7 |
| +300ms | +293ms (-8, -7, -7 frames) | 6.4 |

Every case recovers ground truth within ~1 frame (40ms at the tracker's
25fps) -- the error §1b's coarse detector and §1d's heuristic both had is
gone. Confidence stays in a consistent, meaningfully-nonzero range (~4-8.5)
across all 3 tracks and both an in-sync and 3 out-of-sync injected cases,
unlike the coarse detector's negative/near-zero confidence on this same
clip. Pinned in `analyzer/tests/test_syncnet_offset_real.py` (2 of the 4
cases above, to keep CI-skipped-but-runnable-locally test time bounded --
each case takes a few minutes: real S3FD face detection/tracking + a CNN,
CPU-only).

**Why this isn't the new default:** each run takes minutes, not
milliseconds -- fine for an opt-in `--use-syncnet` tier, not for the
default `syncsentry fix` path most users hit first. The coarse detector
(with its honest confidence gate, §1c) stays the fast default; SyncNet is
the accurate-but-slow tier for when it matters.

## 1f. Multi-speaker dialogue defeats both detectors -- for different, real reasons

A third real user report, on a different real video: a ~30s two-person
"skit" clip (two people talking to each other, cut with camera zooms, plus
a non-face outro card in the last few seconds). Both detectors agreed on
*direction* (`audio_leads`) but wildly disagreed on *magnitude* (-55ms
coarse vs -320ms SyncNet), and SyncNet itself produced 4 short face tracks
that disagreed with each other (0/7/9/10 frames) at low confidence
(0.25-0.81, well below the ~4-8.5 seen on the single-narrator validation
clip in §1e).

Root cause, confirmed by pulling actual frames: this is genuinely a harder
case than either detector was built/validated for. SyncNet's per-track
distance metric compares *one* face's lip motion against the *whole* mixed
audio track; when that face isn't the one currently speaking (the other
person is, mid-conversation), that's real noise, not model error -- this is
exactly why active-speaker detection (ASD; e.g. TalkNet) is its own
research subfield, a prerequisite this project doesn't yet implement. The
coarse detector has the analogous problem at the whole-frame level. Camera
cuts additionally fragment face tracks (explaining the 4 short,
disagreeing tracks instead of 1-3 consistent ones).

**Conclusion, not yet a fix:** both detectors' low-confidence output was
the *honest* response here, not a bug -- for a single continuous narrator
(§1e), the tools work; for genuine back-and-forth dialogue, they
legitimately can't pin a reliable global offset yet. A real next step
would be active-speaker-aware analysis (attribute each moment to whichever
person is actually talking before running SyncNet on just that speaker's
track against just that moment's audio) -- scoped as a future milestone,
not built here.

## 2. Lip-sync detection has moved to learned contrastive embeddings

[SyncNet](http://arxiv.org/pdf/2005.08606v1) and successors --
[ModEFormer](https://doi.org/10.1109/icassp49357.2023.10097209) (ICASSP
2023, 94.5%/90.9% on LRS2/LRS3), [Interpretable Conv-SyncNet](https://arxiv.org/html/2409.00971)
(96.5%/93.8%), [UniSync](https://arxiv.org/html/2503.16357v1) (2025) --
frame sync as *contrastive learning*: audio/visual embedding similarity as
a function of candidate offset picks the true offset.

**Decision:** M3's fine-grained detector uses a pretrained SyncNet model
(now implemented, §1e), restricted to real dialogue scenes (face + speech
both present), complementing rather than replacing the coarse detector.

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
