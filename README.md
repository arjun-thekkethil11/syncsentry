# SyncSentry

Detects and fixes audio/video sync drift in video files, using lip motion and speech, not captions or timestamps.

Most sync tools assume one global offset for the whole file. Real content does not always work that way: a compilation of clips can have a different offset in every clip, and some clips can drift over time instead of staying constant. SyncSentry treats this as the general case, not an edge case.

## How it works

1. **Face and speech detection.** Faces are tracked with S3FD, and speech activity with Silero VAD, to find windows where someone is actually talking on screen.
2. **Offset detection.** A pretrained lip-sync model (SyncNet) scores how well the audio and mouth movement line up at different offsets, inside each window. A lighter cross-correlation detector runs first as a coarse pass and to cover a wider offset range.
3. **Piecewise correction.** If a video does not fit a single global offset, it is split into independently-verified segments, and each one gets its own correction. Segments that cannot be confidently verified are left untouched and reported as such, instead of guessing.
4. **Verification.** After a fix is applied, the corrected file is re-checked with the same detector. A result is only reported as fixed if the residual offset actually measures near zero.

## Results

Evaluated on a held-out benchmark of synthetic clips with known, injected offsets (not used to tune the detector). Methodology, generation scripts, and current numbers are in [`analyzer/scripts/blind_benchmark/`](analyzer/scripts/blind_benchmark/); run `run_benchmark.py` to reproduce them on your own machine.

## Try it

```bash
uvicorn syncsentry.api:app --port 8000     # backend, from analyzer/
cd webapp && npm install && npm run dev    # frontend
```

Then open `http://localhost:5173` and upload a video.

See [`SETUP.md`](SETUP.md) for full install steps.

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

## Background

Built by [Arjun T](mailto:arjun.thekkethil11@gmail.com).
