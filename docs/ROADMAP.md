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
- [x] **M3c.5 -- Found the actual limit of both detectors: multi-speaker dialogue.**
      A third real-user report, on a third real video (a two-person
      back-and-forth clip): both the coarse detector and SyncNet agreed on
      *direction* but disagreed sharply on *magnitude*, and SyncNet's own
      face tracks disagreed with each other. Root-caused by inspecting
      actual frames: genuine back-and-forth dialogue (not a single
      continuous narrator) breaks SyncNet's per-track-vs-whole-mixed-audio
      assumption, and camera cuts fragment face tracking. Both detectors'
      low confidence here was the *honest* answer, not a bug. Documented,
      not fixed -- a real fix needs active-speaker-aware analysis (attribute
      each moment to whoever's actually talking before running SyncNet),
      scoped as future work. See `docs/RESEARCH.md` §1f.
- [x] **M3c.6 -- Windowed + cross-corroborated SyncNet (real improvement; one false positive caught and fixed).**
      Score independent ~1s sub-windows within each face track instead of
      only the whole-track median (non-speaking frames no longer dilute
      the estimate) plus a boundary-artifact filter and a
      3-independent-windows-must-agree corroboration requirement before
      trusting a windowed answer over the whole-track fallback. First cut
      (2 windows required) looked like a win on the §M3c.5 video -- until
      the pipeline's own re-measurement step (not a new check, the same
      one that caught the M2.5 bugs) showed the "fix" didn't collapse the
      residual, proving the 2-window agreement was coincidental. Raised to
      3; re-validated both the §1e ground-truth regression (still passes)
      and the hard video (now correctly falls back to the same honest
      low confidence as before, no regression). See `docs/RESEARCH.md` §1g.
- [x] **M3c.7 -- Fully-automated active-speaker gate (mouth motion), no manual step.**
      Real user ask: no manual offset entry, fully automated. Built a
      cheap active-speaker approximation reusing SyncNet's own
      face-centered crops (no new model): per-track, self-calibrating
      mouth-motion gate excludes windows where that face is likely
      listening, not talking, before SyncNet scores them at all. Passes
      the full ground-truth regression with the gate on (no regression).
      On the §M3c.5/6 video: measurably cleaned up the windows scored
      (14->5-6 per track) but the remaining confident windows still only
      formed two separate, contradicting 2-window pairs -- correctly still
      below the corroboration bar, so the honest answer stays "can't
      reliably tell" for this specific ~30s, multi-camera-cut teaser.
      Assessed as a property of this particular short/fragmented asset,
      not (only) a tooling gap -- see `docs/RESEARCH.md` §1h for why a
      longer cut of the same source would very plausibly work with zero
      further changes.
- [x] **M4 -- Go orchestrator: catalog batch runner.** Job queue + asset
      state in Postgres (`SELECT ... FOR UPDATE SKIP LOCKED` for safe
      concurrent claiming), `POST /jobs` / `GET /jobs` / `GET /jobs/{id}`,
      N worker goroutines calling the Python API (M2.6) over HTTP
      (multipart upload, same as the CLI's `curl -F`). Verified end-to-end
      against a real local Postgres + real `uvicorn` process: submitted a
      job for a fixture with a known +200ms injected offset, worker
      claimed it, called the analyzer API, and the job's stored result
      showed `detected_offset_ms: 200.0` / `residual_offset_ms: 0.0` --
      i.e. actually fixed, not just "ran without erroring." 15 Go tests
      (unit tests with an in-memory Store + `httptest` fake analyzer API;
      one real-Postgres integration test, opt-in via
      `SYNCSENTRY_TEST_DATABASE_URL`, skipped otherwise).
- [x] **M4.5 -- Webapp: browser UI for detect + fix + visualize (single asset).**
      A React/Vite/Tailwind SPA (`webapp/`) in front of the same `/v1/fix`
      the CLI and Go orchestrator already call -- no parallel pipeline.
      Upload a video (or a `.zip` bundling video + captions -- new
      `api.py` extraction path, `_extract_zip_inputs`) and optional
      captions, get back detected/residual offset bars, a confidence
      gauge (plotted against the actual threshold used, not just the raw
      number), and download links for the corrected files. Required two
      small, real backend additions, not just a frontend: (1) CORS
      middleware, since this is now a cross-origin browser client; (2)
      `IssueSummary` gained structured `confidence`/`min_confidence`/
      `method`/`matched_count`/`unmatched_count` fields -- previously that
      detail only existed embedded in a prose `note` string, fine for a
      CLI report, not for a UI gauge. Verified against the real API (not
      mocked): screenshotted the actual results page after uploading a
      real synthetic fixture with a known +175ms/-95ms injected offset and
      confirming the rendered numbers match. 62 Python tests still pass
      (+4 new: zip upload, zip-without-video rejection).
- [ ] **M5 -- Sync-health dashboard.** Per-title timeline visualization;
      catalog-wide drift summary (across jobs recorded by the M4
      orchestrator, surfaced in the M4.5 webapp).
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
