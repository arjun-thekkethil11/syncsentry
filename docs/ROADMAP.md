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
- [x] **M2.6 -- HTTP API.** FastAPI service (`syncsentry/api.py`) wrapping
      the same `pipeline`/`title_drift` code the CLI and tests use --
      `POST /v1/fix`, `POST /v1/classify-drift`, `GET /healthz`, file
      download endpoints. Thin by design: no logic lives only in the API
      layer. Verified both via `TestClient` (7 tests) and a real running
      `uvicorn` server hit with `curl`. This is the Python-side worker API;
      the Go orchestrator (M4) will front it for catalog-scale batch jobs
      rather than duplicate it.
- [x] **M3a -- Statistical drift-pattern classification (DiVAS-style).**
      - `syncsentry/lipsync/scene_offsets.py`: slides a window across a
        whole title's audio/video envelopes (extracted once, not
        re-extracted per window) and estimates a per-window A/V offset
        using the already-validated Tier-1 cross-correlation detector,
        dropping low-confidence windows.
      - `syncsentry/lipsync/title_drift.py`: a from-scratch RANSAC line fit
        (numpy only) over (scene_time, scene_offset) pairs, classifying the
        title as `in_sync` / `constant_offset` / `drift_increasing` /
        `drift_decreasing` / `intermittent` / `unstable` -- the actual
        DiVAS (CVPR 2024) contribution this project is built around.
      - Extended the synthetic fixture generator
        (`generate_piecewise_offset_fixture`) to inject *time-varying*
        offset schedules (ramps, localized bumps), not just a constant
        offset, so all five drift patterns have ground-truth tests.
      - New CLI command (`syncsentry classify-drift`) and API endpoint
        (`POST /v1/classify-drift`). 6 new tests, all passing, including
        exact recovery of an injected intermittent-drift window (detected
        7.5s-10.5s vs. injected 7.3s-10.7s; detected bump 246.7ms vs.
        injected 250ms).
      - **What this is not (yet):** the per-window *estimator* is still the
        Tier-1 signal-processing detector, not a learned lip-sync embedding
        model, and scenes are fixed-length windows, not real dialogue
        scenes (face + speech-activity overlap). Both are deferred to
        **M3b** below rather than built untested against synthetic content
        that has no faces in it.
- [x] **M3b -- Real dialogue-scene detection (sourced real content to
      validate against).**
      - Sourced a small, permissively-licensed (CC BY 3.0) real talking-head
        clip from Wikimedia Commons specifically to test face detection and
        VAD against genuine faces/speech -- never committed to the repo
        (`analyzer/fixtures/real_content/SOURCES.md` documents the source +
        license; `analyzer/scripts/fetch_real_content.sh` reproducibly
        re-fetches it).
      - Real face detection: `syncsentry/lipsync/face_detect.py`, using
        OpenCV's `cv2.FaceDetectorYN` (YuNet, opencv_zoo) instead of the
        originally-planned mediapipe -- mediapipe's Tasks API crashes on
        this project's macOS execution environment (Metal GPU service
        unavailable even under a CPU delegate; see `docs/RESEARCH.md`
        section 2b). Validated: face detected in 24/50 sampled frames of the
        real clip (48%, confidence 0.84-0.95 when present), matching a
        manual check of the footage.
      - Real speech detection: `syncsentry/lipsync/vad.py`, using Silero-VAD
        (pip package, bundled weights). Validated: 10 speech segments
        covering ~46s of a 50s continuous-talking clip, vs. zero segments on
        the synthetic tone-pulse fixture (confirms it's doing something
        real, not just firing on any loud signal like the caption checker's
        onset picker does).
      - Real dialogue-scene detection: `syncsentry/lipsync/dialogue_scenes.py`
        intersects face-presence and speech-activity intervals (merging
        small gaps, dropping sub-1s scenes). Validated: 4 scenes, 25.8s of
        50.0s (52%) on the real clip, matching a manual visual check.
        New CLI command (`syncsentry dialogue-scenes`) and API endpoint
        (`POST /v1/dialogue-scenes`).
      - **A concrete finding motivating M3c below:** run against the same
        real (in-sync) clip, M3a's Tier-1 windowed estimator passes its
        confidence gate on 12 windows whose offset estimates range from 0ms
        to 480ms -- all "confident," none reliable, because global frame
        brightness only spuriously correlates with real speech energy. See
        `docs/RESEARCH.md` section 1b. 10 new tests (6 synthetic/CI-safe, 4
        real-content, auto-skipped when the fixture isn't fetched locally).
      - 53 pytest cases total (11 new: 6 dialogue-scene unit tests, 4
        real-content validation tests -- auto-skipped in CI, 1 API endpoint
        test).
- [ ] **M3c -- Learned per-scene estimator (deferred until there's a way to
      validate it against real, ideally annotated, drift).**
      - Swap the per-window estimator in `scene_offsets.py` for a
        SyncNet-family pretrained embedding model, restricted to the real
        dialogue scenes M3b now detects -- the interface (`list[SceneOffset]`
        in, same `title_drift` classification code) is already designed for
        this to be a drop-in replacement.
      - Real VAD (Silero, from M3b) wired into the caption checker itself,
        replacing the energy-threshold onset picker for real speech (kept as
        the default for synthetic-tone fixtures, where a speech-specific VAD
        wouldn't even fire).
- [ ] **M4 -- Go orchestrator: catalog batch runner.**
      - Job queue + asset state (Postgres), scheduling across a catalog of
        many assets, calling the Python API (M2.6) as a worker.
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
