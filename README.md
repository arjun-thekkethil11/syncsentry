# SyncSentry

Detects and fixes audio/video sync drift in video files, using lip motion and speech, not captions or timestamps.

Most sync tools assume one global offset for the whole file. Real content does not always work that way: a compilation of clips can have a different offset in every clip, and some clips can drift over time instead of staying constant. SyncSentry treats this as the general case, not an edge case.

## How it works

1. **Face and speech detection.** Faces are tracked with S3FD, and speech activity with Silero VAD, to find windows where someone is actually talking on screen.
2. **Offset detection.** A pretrained lip-sync model (SyncNet) scores how well the audio and mouth movement line up at different offsets, inside each window. A lighter cross-correlation detector runs first as a coarse pass and to cover a wider offset range.
3. **Piecewise correction.** If a video does not fit a single global offset, it is split into independently-verified segments, and each one gets its own correction. Segments that cannot be confidently verified are left untouched and reported as such, instead of guessing.
4. **Verification.** After a fix is applied, the corrected file is re-checked with the same detector. A result is only reported as fixed if the residual offset actually measures near zero. If confidence is too low to trust in the first place, nothing is applied automatically either — a preview of the candidate correction is rendered instead, so it can be checked by eye/ear rather than trusted on a bare confidence number.

## Results

Evaluated on a held-out benchmark of synthetic clips with known, injected offsets (not used to tune the detector). Methodology, generation scripts, and current numbers are in [`analyzer/scripts/blind_benchmark/`](analyzer/scripts/blind_benchmark/); run `run_benchmark.py` to reproduce them on your own machine.

## Try it

```bash
uvicorn syncsentry.api:app --port 8000     # backend, from analyzer/
cd webapp && npm install && npm run dev    # frontend
```

See [`SETUP.md`](SETUP.md) for full install steps.

The webapp fills the wait (detection + fixing isn't instant) with facts about
the uploaded asset and general A/V-sync trivia instead of a bare spinner, then
shows a before/after preview of the result so nothing has to be taken on
faith:

<p>
  <img src="docs/images/webapp-processing.png" alt="Processing view with a rotating fact about the uploaded asset" width="420">
  <img src="docs/images/webapp-results.png" alt="Results view with a before/after preview of the detected fix" width="420">
</p>

## Project layout

```
analyzer/     Python package: detection, fixing, API, CLI, tests
webapp/       React frontend
```

Inside `analyzer/syncsentry/`:

```
detectors/    Cross-correlation offset detector
lipsync/      Face/speech detection, SyncNet, wide-range and piecewise offset estimation
fixer/        Applies the audio correction
pipeline.py   detect -> fix -> verify -> report
api.py        HTTP API
cli.py        Command-line interface
```

## Tech

Python, PyTorch, OpenCV, FFmpeg, FastAPI, React, Vite, TypeScript.

## Research

Built on published models rather than anything trained from scratch:

- SyncNet, lip-sync scoring: Chung & Zisserman, ["Out of Time: Automated Lip Sync in the Wild"](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/chung16a.pdf), ACCV Workshop 2016
- S3FD, face detection: Zhang et al., ["S3FD: Single Shot Scale-invariant Face Detector"](https://arxiv.org/abs/1708.05237), ICCV 2017
- Light-ASD, active speaker detection: Liao et al., ["A Light Weight Model for Active Speaker Detection"](https://arxiv.org/abs/2303.04439), CVPR 2023
- MTDVocaLiST, short-clip sync scoring: Chen et al., ["Multimodal Transformer Distillation for Audio-Visual Synchronization"](https://arxiv.org/abs/2210.15563), ICASSP 2024, distilled from Kadandale et al., ["VocaLiST"](https://arxiv.org/abs/2204.02090), Interspeech 2022
- Silero VAD, speech detection: [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad)

None of these models solve the actual problem on their own, each is built and
benchmarked for a fixed-length clip with one offset. On top of them:

- **Piecewise correction**: splits a video into independently-verified segments
  instead of assuming one global offset, for compilations or edited clips where
  each cut can carry a different offset.
- **Wide-range detection**: extends SyncNet's native +-600ms search window to
  recover offsets of several seconds, with a periodicity self-check so a
  video's natural rhythm (music, repeated motion) isn't mistaken for a real
  offset.
- **Verification pass**: every correction is re-measured with the same
  detector after the fix. A result is only reported as fixed if the residual
  offset actually lands near zero, not just because a correction was applied.

Source, license, and weight provenance for each model: [`analyzer/third_party/SOURCES.md`](analyzer/third_party/SOURCES.md).

## Background

Built by [Arjun T](mailto:arjun.thekkethil11@gmail.com).
