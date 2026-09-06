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
  (`syncsentry/detectors/coarse_xcorr.py`) currently estimates a single
  global offset per asset -- that's Milestone 1/2. Milestone 3
  (`syncsentry/detectors/title_drift.py`, not yet built) will add DiVAS's
  per-scene + RANSAC title-level classification on top of it, which is the
  only way to tell "this whole title is 80ms out" apart from "this title
  drifts progressively worse from minute 40 onward" -- two very different
  remediation paths in a real content-ops workflow.

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
  independently. Milestone 3 swaps the current energy-threshold onset
  picker for a proper VAD (e.g. Silero-VAD) so this works on real speech,
  not just synthetic tone bursts.

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
