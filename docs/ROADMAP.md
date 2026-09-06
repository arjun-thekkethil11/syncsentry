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
