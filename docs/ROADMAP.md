# Roadmap

Living project, built incrementally. Status reflects what's actually
implemented and tested, not aspirational scope.

- [x] **M0 -- Scaffold & research.** Repo structure, literature review
      (`docs/RESEARCH.md`), toolchain (Go, ffmpeg, Python 3.11).
- [x] **M1 -- Synthetic ground-truth benchmark harness.** Sample-accurate
      "flash + beep" fixture generator injecting a known A/V and/or
      caption offset -- zero dependency on restricted-access datasets.
- [x] **M2 -- Coarse detectors (signal processing, no ML).**
      Global A/V offset via audio-RMS / video-brightness cross-correlation
      (0.00ms mean error, -900..900ms sweep); caption-vs-speech drift
      checker (exact median recovery, 11-point sweep). 18 tests, including
      a regression test for the periodic cross-correlation ambiguity found
      during benchmarking (`docs/RESEARCH.md` §3).
- [x] **M2.5 -- Detect, fix, and report (the user-facing product).**
      `syncsentry fix` detects + corrects A/V offset and caption drift,
      re-measures the corrected output to prove it worked, writes a short
      report. Two real bugs found via round-trip testing (both pinned):
      `-itsoffset`+stream-copy silently no-ops for raw-sample readers; the
      fixture generator's WebVTT writer clamped negative cue timestamps to
      0. 32 tests total.
- [x] **M2.6 -- HTTP API.** FastAPI service wrapping the same
      `pipeline`/`title_drift` code the CLI and tests use --
      `POST /v1/fix`, `POST /v1/classify-drift`, `GET /healthz`.
- [x] **M3a -- Statistical drift-pattern classification (DiVAS-style).**
      Windowed per-scene A/V offset estimation + from-scratch RANSAC line
      fit classifying `in_sync` / `constant_offset` / `drift_increasing` /
      `drift_decreasing` / `intermittent` / `unstable`. Validated on all 5
      patterns against synthetic ground truth with time-varying injected
      offsets. New `classify-drift` CLI/API. *Not yet real: the per-window
      estimator is still Tier-1 signal processing, and scenes are fixed
      windows, not real dialogue scenes -- deferred to M3b.*
- [x] **M3b -- Real dialogue-scene detection.**
      Sourced a small CC BY 3.0 real talking-head clip (never committed --
      `analyzer/fixtures/real_content/SOURCES.md` + `scripts/fetch_real_content.sh`)
      to validate against genuine faces/speech.
      - Real face detection (`lipsync/face_detect.py`, OpenCV YuNet --
        mediapipe's Tasks API crashes on this project's macOS environment,
        see `docs/RESEARCH.md` §2b): 24/50 sampled frames matched (48%).
      - Real speech detection (`lipsync/vad.py`, Silero-VAD): 10 segments
        over ~46s of the 50s clip, vs. zero on synthetic tone fixtures.
      - Real dialogue scenes (`lipsync/dialogue_scenes.py`, face ∩ speech):
        4 scenes, 25.8s of 50.0s (52%), matching a manual check. New
        `dialogue-scenes` CLI/API.
      - **Key finding:** M3a's confidence gate passes 12 "confident"
        windows on this same in-sync clip with offsets ranging 0-480ms --
        motivating M3c. See `docs/RESEARCH.md` §1b.
      - **Found via dogfooding, not testing:** the finding above turned out
        to be a real product bug -- the fix pipeline had no confidence gate
        at all, and mislabeled an unverified fix as "FIXED." Now fixed and
        regression-tested. See `docs/RESEARCH.md` §1c.
      - 56 tests total (4 real-content ones auto-skip in CI).
- [x] **M3b.5 -- Tried and falsified a classical mouth-motion heuristic.**
      Hypothesis: restrict the coarse detector to mouth-ROI motion (YuNet
      landmarks) within real dialogue scenes, instead of whole-frame
      brightness -- prompted by a second real-user report ("still wrong")
      on the M3b behavior above. Validated against the real clip with 4
      *known* injected offsets (0/+150/-200/+300ms via `fix_av_offset` used
      as an injector): confidence stayed low (~0.15-0.20) and didn't track
      ground truth at all. Not wired into the pipeline -- would repeat the
      exact §1b mistake with a different detector. Kept as a pinned
      negative-result test (`test_mouth_offset_real.py`); real evidence
      that M3c needs a learned model, not a smaller ROI. 60 tests total.
      See `docs/RESEARCH.md` §1d.
- [x] **M3c -- Learned estimator: pretrained SyncNet, wired in as an opt-in tier.**
      `syncsentry fix --use-syncnet` (`lipsync/syncnet_offset.py`) wraps
      `joonson/syncnet_python` (MIT, vendored via
      `scripts/fetch_syncnet.sh`, never committed) as an external tool
      rather than reimplemented. Validated with the same known-injected-offset
      methodology that falsified M3b.5: recovers 0/+150/-200/+300ms injected
      offsets on the real clip within ~1 frame (40ms) across 3 face tracks
      each time, with consistent confidence (~4-8.5) -- unlike every prior
      attempt. See `docs/RESEARCH.md` §1e. Not the default: each run takes
      minutes (real face tracking + a CNN, CPU), so the fast coarse
      detector + confidence gate (M2.5/M3b) stays the default path; SyncNet
      is opt-in for when accuracy matters more than latency. 62 tests total
      (2 more real-content ones, auto-skip in CI). *Remaining, smaller
      piece of the original M3c scope:* wiring Silero-VAD into the caption
      checker itself for real speech (currently still a synthetic-pulse
      onset picker, see `docs/RESEARCH.md` §4) -- deferred, not blocking.
- [ ] **M4 -- Go orchestrator: catalog batch runner.** Job queue + asset
      state (Postgres), scheduling across a catalog, calling the Python
      API (M2.6) as a worker.
- [ ] **M5 -- Sync-health dashboard.** Per-title timeline visualization;
      catalog-wide drift summary.
- [ ] **M6 -- Real-content validation + writeup.** Validate against a
      small set of Creative-Commons titles (kept out of git); publish a
      benchmark comparison write-up.

## Why this project

Chosen over three other scoped candidates (multi-CDN steering, per-shot
bitrate optimization, ABR RL benchmarking) because it most directly extends
production experience already on the resume (caption sync, frame-accurate
VOD/iVOD tooling at Fox) into the current research frontier (DiVAS,
ModEFormer, UniSync). See `docs/RESEARCH.md` for the literature grounding
each milestone.
