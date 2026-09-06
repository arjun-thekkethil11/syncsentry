"""Tier 2: learned per-scene lip-sync detection (Milestone 3, in progress).

Planned design (see docs/RESEARCH.md sections 1-2 and docs/ROADMAP.md M3):

1. Detect dialogue scenes (shot boundaries + face + speech-active overlap).
2. Run a pretrained SyncNet-family embedding model (ModEFormer / IC-SyncNet /
   UniSync) per scene to get a per-scene offset estimate + confidence,
   accelerated via PyTorch's Apple Silicon MPS backend.
3. Fit a RANSAC regression across per-scene estimates for the whole title
   (following DiVAS, CVPR 2024) to classify the title-level drift pattern:
   constant offset, drift-early, drift-late, or intermittent offset.

This module is intentionally left as a stub until Milestone 3 -- Tier 1
(syncsentry.detectors.coarse_xcorr, syncsentry.captions.drift_check) is
built, tested, and benchmarked first.
"""
