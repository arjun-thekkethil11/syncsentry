# Roadmap

This is a living project, built incrementally. Status reflects what's
actually implemented and tested, not aspirational scope.

- [x] **M0 -- Scaffold & research.** Repo structure, literature review
      (`docs/RESEARCH.md`), toolchain (Go, ffmpeg, Python 3.11 venv).
- [x] **M1 -- Synthetic ground-truth benchmark harness.** Sample-accurate
      "flash + beep" fixture generator (`syncsentry/synth/fixture_gen.py`)
      that injects a known A/V offset and/or caption-vs-speech offset,
      with zero dependency on restricted-access datasets (LRS2/LRS3 etc.).
- [x] **M2 -- Coarse detectors (signal processing, no ML).**
      - Global A/V offset via audio-RMS / video-brightness envelope
        cross-correlation (`syncsentry/detectors/coarse_xcorr.py`).
        Benchmarked to exact (0.00ms mean error) recovery on synthetic
        fixtures across a -900ms..+900ms sweep; see
        `benchmark/results/av_offset_benchmark.md`.
      - Caption-vs-speech drift checker (`syncsentry/captions/drift_check.py`),
        benchmarked to exact median recovery across an 11-point offset sweep;
        see `benchmark/results/caption_drift_benchmark.md`.
      - 18 pytest cases encoding the "inject known offset -> assert recovery
        within tolerance" methodology, including a regression test for the
        periodic cross-correlation ambiguity boundary discovered during
        benchmarking (see `docs/RESEARCH.md` section 3).
- [x] **M2.5 -- Detect, fix, and report (the actual user-facing product).**
      - `syncsentry fix --video X --captions Y --out-dir Z`: detects A/V
        offset and caption drift, corrects both, re-measures the corrected
        output to prove the fix worked, and writes a short human-readable
        report (`report.txt` / `report.json`) instead of a dump of internal
        metrics -- see [`syncsentry/pipeline.py`](../analyzer/syncsentry/pipeline.py).
      - A/V fix (`syncsentry/fixer/av_fix.py`): trims or pads *real* audio
        samples (re-encode), not a metadata-only `-itsoffset` remux -- an
        earlier attempt at the latter was verified (via our own detector) to
        do nothing at all under stream copy.
      - Caption fix (`syncsentry/fixer/caption_fix.py`): rewrites cue
        timestamps, always re-measured against the *corrected* video's audio
        rather than composing the A/V and caption corrections algebraically.
      - Two real bugs found and fixed via round-trip testing during this
        milestone (both now pinned by regression tests):
        1. `-itsoffset` + `-c copy` silently no-ops for consumers (including
           our own frame/sample-index-based extraction) that read raw
           decoded sequences instead of container timestamp metadata.
        2. The synthetic fixture generator's WebVTT writer clamped negative
           cue timestamps to `0.000`, silently corrupting the first cue's
           ground truth whenever a large negative caption offset was
           injected -- fixed by giving every test cue a full period of
           head-room instead of masking it with a wider tolerance.
      - 32 pytest cases total (14 new: 6 A/V fix round-trips, 4 caption fix
        round-trips, 1 no-op case, 3 pipeline e2e cases).
- [ ] **M3 -- Learned per-scene / per-title drift classification.**
      - Dialogue-scene detection (face + speech-active segments).
      - SyncNet-family pretrained embedding model for per-scene offset
        (PyTorch, Apple Silicon MPS backend).
      - DiVAS-style RANSAC regression across scenes to classify
        constant-offset / drift-early / drift-late / intermittent per title.
      - Swap the synthetic-beep onset picker in the caption checker for a
        real VAD (Silero-VAD) so it works on real speech.
- [ ] **M4 -- Orchestration service (Go).**
      - Job queue + asset state (Postgres), batch runner over a catalog.
      - REST API exposing per-title sync-health reports.
      - Structured logging / metrics in the style of the observability
        work this project is modeled after.
- [ ] **M5 -- Sync-health dashboard.**
      - Per-title timeline visualization (DiVAS-style "sync movie timeline").
      - Catalog-wide summary view (what fraction of titles have drift, by
        type and severity).
- [ ] **M6 -- Real-content validation + writeup.**
      - Validate against a small set of Creative-Commons-licensed titles
        with real dialogue (kept out of git; only code + methodology is
        published).
      - Publish a benchmark comparison write-up and update the top-level
        README with final numbers.

## Why this project (vs. alternatives considered)

Before starting, four candidate OTT/platform research areas were scoped:
multi-CDN content steering, per-shot bitrate ladder optimization, ABR
reinforcement learning benchmarking, and this one (A/V + caption drift
detection). This project was chosen because it most directly extends
production experience already on the resume (caption sync, frame-accurate
VOD/iVOD tooling at Fox) into the current research frontier (DiVAS,
ModEFormer, UniSync) rather than starting a new, less-differentiated
research area from zero. See `docs/RESEARCH.md` for the literature that
grounds each milestone.
