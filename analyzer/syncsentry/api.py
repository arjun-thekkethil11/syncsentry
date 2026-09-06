"""SyncSentry HTTP API: the same detect -> fix -> report pipeline as the
CLI, exposed over HTTP so it's usable by anyone/anything without a local
Python environment (a web frontend, a CI job, a content-ops tool, or --
eventually -- the Go orchestrator's catalog batch runner, per docs/ROADMAP.md
M4).

Deliberately thin: every endpoint just validates input and calls into
`syncsentry.pipeline`, the same code path the CLI and the test suite use.
No logic lives only in this file.

Run locally:
    uvicorn syncsentry.api:app --reload --port 8000

Try it:
    curl -F "video=@asset.mkv" -F "captions=@asset.vtt" \
         http://localhost:8000/v1/fix
"""
from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from syncsentry import __version__
from syncsentry.lipsync.dialogue_scenes import detect_dialogue_scenes
from syncsentry.lipsync.scene_offsets import estimate_windowed_offsets
from syncsentry.lipsync.title_drift import classify_title_drift
from syncsentry.pipeline import run_fix_pipeline

app = FastAPI(
    title="SyncSentry",
    description="Detects and corrects A/V offset and caption drift in OTT video assets.",
    version=__version__,
)

# Jobs are written under a run directory so corrected files can be fetched
# afterward via /v1/jobs/{job_id}/files/{name}. In-memory job registry is
# fine for a single-process dev/demo deployment; M4's Go orchestrator is
# where real persistence (Postgres) and multi-worker scaling belong.
_RUNS_DIR = Path(tempfile.gettempdir()) / "syncsentry-runs"
_RUNS_DIR.mkdir(exist_ok=True)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "service": "syncsentry-api", "version": __version__}


@app.post("/v1/fix")
async def fix(video: UploadFile = File(...), captions: UploadFile | None = File(None),
               av_min_confidence: float = 0.3) -> dict:
    """Upload a video (and optionally its captions), get back a short report
    plus download links for the corrected files.

    `av_min_confidence` (default 0.3, matching the CLI): below this
    cross-correlation confidence, the A/V offset is reported as
    undetermined rather than "detected" -- real talking-head/dialogue
    content routinely produces confident-looking but spurious global
    offsets from this detector (see docs/RESEARCH.md section 1b); without
    this gate the pipeline would confidently "fix" noise.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = _RUNS_DIR / job_id
    job_dir.mkdir(parents=True)

    video_path = job_dir / video.filename
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    captions_path = None
    if captions is not None:
        captions_path = job_dir / captions.filename
        with captions_path.open("wb") as f:
            shutil.copyfileobj(captions.file, f)

    try:
        summary = run_fix_pipeline(video_path, job_dir / "out", captions_path=captions_path,
                                    av_min_confidence=av_min_confidence)
    except Exception as exc:  # noqa: BLE001 -- surface the real error to the caller
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}") from exc

    result = summary.to_dict()
    # Don't leak server-side absolute paths to API callers -- only the
    # filename (via download_urls) is a caller's business.
    result.pop("output_files", None)
    result["job_id"] = job_id
    result["download_urls"] = {
        label: f"/v1/jobs/{job_id}/files/{Path(path).name}"
        for label, path in summary.output_files.items()
    }
    return result


@app.post("/v1/classify-drift")
async def classify_drift(video: UploadFile = File(...), window_s: float = 3.0) -> dict:
    """Upload a video, get back its title-level drift pattern (in_sync /
    constant_offset / drift_increasing / drift_decreasing / intermittent /
    unstable) -- the M3a statistical layer, see docs/RESEARCH.md and
    docs/ROADMAP.md for what this is and isn't (yet) doing.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = _RUNS_DIR / job_id
    job_dir.mkdir(parents=True)
    video_path = job_dir / video.filename
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    windows = estimate_windowed_offsets(str(video_path), window_s=window_s)
    if len(windows) < 3:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(windows)} confident scene(s) found; need >= 3 to classify.",
        )

    result = classify_title_drift(windows)
    return {
        "job_id": job_id,
        "pattern": result.pattern,
        "slope_ms_per_s": result.slope_ms_per_s,
        "intercept_ms": result.intercept_ms,
        "inlier_ratio": result.inlier_ratio,
        "n_scenes": result.n_scenes,
        "intermittent_window_s": result.intermittent_window_s,
        "intermittent_offset_ms": result.intermittent_offset_ms,
    }


@app.post("/v1/dialogue-scenes")
async def dialogue_scenes(video: UploadFile = File(...), sample_fps: float = 2.0,
                           min_duration_s: float = 1.0) -> dict:
    """Upload a video, get back real dialogue scenes: time ranges where a
    face is on screen AND speech is active (YuNet face detection + Silero
    VAD, see docs/RESEARCH.md M3b). This is the scene-localization half of
    the learned per-scene estimator work; it does not itself estimate an
    offset (that's `/v1/classify-drift`, still backed by the Tier-1
    detector -- see the M3b finding in docs/RESEARCH.md on why that
    detector alone isn't trustworthy on real dialogue content).
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = _RUNS_DIR / job_id
    job_dir.mkdir(parents=True)
    video_path = job_dir / video.filename
    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    scenes = detect_dialogue_scenes(
        str(video_path), sample_fps=sample_fps, min_duration_s=min_duration_s,
    )
    return {
        "job_id": job_id,
        "n_scenes": len(scenes),
        "total_duration_s": sum(s.end_s - s.start_s for s in scenes),
        "scenes": [{"start_s": s.start_s, "end_s": s.end_s} for s in scenes],
    }


@app.get("/v1/jobs/{job_id}/files/{filename}")
def download_file(job_id: str, filename: str) -> FileResponse:
    path = _RUNS_DIR / job_id / "out" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)
