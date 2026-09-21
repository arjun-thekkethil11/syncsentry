"""SyncSentry HTTP API: the same detect, fix, report pipeline as the CLI,
exposed over HTTP for the web frontend or any other client.

Every endpoint validates input and calls into `syncsentry.pipeline`, the
same code path the CLI and the test suite use. No logic lives only in
this file.

Run locally:
    uvicorn syncsentry.api:app --reload --port 8000

Try it:
    curl -F "video=@asset.mkv" -F "captions=@asset.vtt" \
         http://localhost:8000/v1/fix
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from syncsentry import __version__
from syncsentry.lipsync.dialogue_scenes import detect_dialogue_scenes
from syncsentry.lipsync.scene_offsets import estimate_windowed_offsets
from syncsentry.lipsync.title_drift import classify_title_drift
from syncsentry.pipeline import run_fix_pipeline
from syncsentry.util.ffmpeg_io import probe_duration_s

# `uvicorn` without `--reload` never re-imports changed modules, so a
# long-running process can silently keep serving stale code after a file
# on disk changes, with no way for a caller to tell. `_CODE_FINGERPRINT`
# makes this checkable: a content hash of every detection/fix module's
# source, computed once at process import time. If a fresh hash of the
# same files on disk ever disagrees with this, the running process needs
# a restart. `_STARTUP_TIME` additionally reports uptime, so "has this
# been running suspiciously long relative to my last edit" is answerable
# without any file comparison.
_CODE_FINGERPRINT_FILES = sorted(
    (Path(__file__).parent).rglob("*.py"),
)


def _compute_code_fingerprint() -> str:
    h = hashlib.sha256()
    for f in _CODE_FINGERPRINT_FILES:
        try:
            h.update(f.read_bytes())
        except OSError:
            pass  # file removed or unreadable since the glob; don't crash healthz over it
    return h.hexdigest()[:16]


_CODE_FINGERPRINT = _compute_code_fingerprint()  # frozen at process import time
_STARTUP_TIME = time.time()


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                            capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent,
                                    capture_output=True, text=True, timeout=2)
            suffix = "-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
            return r.stdout.strip() + suffix
    except (OSError, subprocess.SubprocessError):
        pass
    return None


_GIT_COMMIT = _git_commit()

app = FastAPI(
    title="SyncSentry",
    description="Detects and corrects A/V offset and caption drift in OTT video assets.",
    version=__version__,
)

# The webapp calls this API directly from the browser. Wide open here
# because this is a local dev tool, not a multi-tenant service; tighten
# `allow_origins` before deploying anywhere real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}
_CAPTION_EXTS = {".vtt", ".srt"}


def _extract_zip_inputs(zip_path: Path, dest_dir: Path) -> tuple[Path, Path | None]:
    """Unpack an uploaded .zip and find the video (and optional captions)
    inside. Picks the first file matching each extension set, by name so
    results are deterministic, rather than assuming a fixed layout.
    """
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    video_path, captions_path = None, None
    for path in sorted(dest_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if video_path is None and suffix in _VIDEO_EXTS:
            video_path = path
        elif captions_path is None and suffix in _CAPTION_EXTS:
            captions_path = path
    if video_path is None:
        raise HTTPException(
            status_code=422,
            detail=f"No video file found in the uploaded zip (looked for {sorted(_VIDEO_EXTS)}).",
        )
    return video_path, captions_path

# Jobs are written under a run directory so corrected files can be fetched
# afterward via /v1/jobs/{job_id}/files/{name}. In-memory job registry is
# fine for a single-process deployment.
_RUNS_DIR = Path(tempfile.gettempdir()) / "syncsentry-runs"
_RUNS_DIR.mkdir(exist_ok=True)


@app.get("/healthz")
def healthz() -> dict:
    """Includes a staleness check: re-hashes the same files on disk right
    now and compares against what this process loaded at import time.
    `code_stale: true` means this process needs a restart to pick up
    on-disk changes. Deliberately cheap (source-only hashing, no
    imports), so this stays safe to poll.
    """
    current_fingerprint = _compute_code_fingerprint()
    return {
        "status": "ok",
        "service": "syncsentry-api",
        "version": __version__,
        "git_commit": _GIT_COMMIT,
        "code_fingerprint": _CODE_FINGERPRINT,
        "code_stale": current_fingerprint != _CODE_FINGERPRINT,
        "uptime_s": round(time.time() - _STARTUP_TIME, 1),
        "python_executable": sys.executable,
    }


@app.post("/v1/fix")
def fix(video: UploadFile = File(...), captions: UploadFile | None = File(None),
         av_min_confidence: float = 0.3, use_syncnet: bool = True,
         syncnet_min_confidence: float = 3.0, use_mtdvocalist: bool = False) -> dict:
    """Upload a video (and optionally its captions), get back a short report
    plus download links for the corrected files.

    `use_syncnet` defaults to `True` here to match the webapp's own
    default (`UploadForm.tsx`'s `DEFAULT_OPTIONS.useSyncnet`), so a plain
    curl/API/CLI call and a webapp upload of the same file get the same
    detector by default. The coarse detector alone is unreliable on real
    talking-head and dialogue content. `run_fix_pipeline`'s own library
    default is left at `False` for tests/scripts that don't care about
    this; every user-facing entry point (this endpoint, the CLI)
    explicitly opts into `True` instead of relying on that.

    Deliberately a plain `def`, not `async def`: `run_fix_pipeline` below
    is long-running, blocking, synchronous CPU work (real face detection,
    SyncNet/Light-ASD/MTDVocaLiST inference, minutes with
    `use_syncnet=True`) with no `await` anywhere in this body. FastAPI/
    Starlette runs plain `def` route handlers in a worker thread
    automatically; an `async def` handler runs directly on the single
    event loop, so calling blocking work from one blocks the entire
    server, including unrelated concurrent requests like `/healthz` or
    another upload, for the whole duration.

    `video` may also be a single `.zip` containing the video (and
    optionally a `.vtt`/`.srt` captions file inside it), a convenience
    for the webapp's "upload a zip" path so callers don't need to bundle
    two separate multipart fields.

    `use_syncnet`: use the pretrained SyncNet model instead of the fast
    coarse detector. Far more accurate on real talking-head content, but
    the request then takes minutes rather than seconds (real face
    detection and tracking plus a CNN, CPU-bound), so set a generous
    client timeout. Requires the model to have been fetched server-side
    (`analyzer/scripts/fetch_syncnet.sh`); silently falls back to the
    coarse detector (noted in the response) otherwise.

    `av_min_confidence` (default 0.3, matching the CLI): below this
    cross-correlation confidence, the A/V offset is reported as
    undetermined rather than detected. Real talking-head and dialogue
    content routinely produces confident-looking but spurious global
    offsets from this detector; without this gate the pipeline would
    confidently "fix" noise.

    `use_mtdvocalist` (default False, requires `use_syncnet=True`): only
    tried if SyncNet's own windowed corroboration fails outright. Off by
    default since it adds real latency (its own face-detection pass plus
    many CPU transformer calls) for a case that is already ambiguous.
    Opt in deliberately, not as a silent "try harder" default.
    """
    job_id = uuid.uuid4().hex[:12]
    job_dir = _RUNS_DIR / job_id
    job_dir.mkdir(parents=True)

    upload_path = job_dir / "upload" / video.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    # Convenience path for the webapp: a single .zip containing the video
    # (and optionally captions) instead of two separate multipart fields.
    if upload_path.suffix.lower() == ".zip":
        video_path, zip_captions_path = _extract_zip_inputs(upload_path, job_dir / "upload_extracted")
    else:
        video_path, zip_captions_path = upload_path, None

    captions_path = zip_captions_path
    if captions is not None:
        captions_path = job_dir / "upload" / captions.filename
        with captions_path.open("wb") as f:
            shutil.copyfileobj(captions.file, f)

    try:
        input_duration_s = probe_duration_s(str(video_path))
    except Exception:  # noqa: BLE001 - purely informational, never block processing on it
        input_duration_s = None

    started_at = time.monotonic()
    try:
        summary = run_fix_pipeline(video_path, job_dir / "out", captions_path=captions_path,
                                    av_min_confidence=av_min_confidence, use_syncnet=use_syncnet,
                                    syncnet_min_confidence=syncnet_min_confidence,
                                    use_mtdvocalist=use_mtdvocalist)
    except Exception as exc:  # noqa: BLE001 - surface the real error to the caller
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}") from exc
    processing_time_s = time.monotonic() - started_at

    result = summary.to_dict()
    # Don't leak server-side absolute paths to API callers; only the
    # filename (via download_urls) is a caller's business.
    result.pop("output_files", None)
    result["job_id"] = job_id
    result["input_filename"] = video_path.name
    result["input_duration_s"] = input_duration_s
    result["processing_time_s"] = round(processing_time_s, 2)
    result["used_syncnet"] = use_syncnet
    result["download_urls"] = {
        label: f"/v1/jobs/{job_id}/files/{Path(path).name}"
        for label, path in summary.output_files.items()
    }
    return result


@app.post("/v1/classify-drift")
def classify_drift(video: UploadFile = File(...), window_s: float = 3.0) -> dict:
    """Upload a video, get back its title-level drift pattern: in_sync,
    constant_offset, drift_increasing, drift_decreasing, intermittent, or
    unstable.

    Plain `def`, not `async def`; see `/v1/fix` above for why (blocking
    CPU work with no `await` in the body would otherwise stall the whole
    server, not just this request).
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
def dialogue_scenes(video: UploadFile = File(...), sample_fps: float = 2.0,
                     min_duration_s: float = 1.0) -> dict:
    """Upload a video, get back real dialogue scenes: time ranges where a
    face is on screen and speech is active (YuNet face detection plus
    Silero VAD). This is scene localization only; it does not itself
    estimate an offset (that's `/v1/classify-drift`).

    Plain `def`, not `async def`; see `/v1/fix` above for why.
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
