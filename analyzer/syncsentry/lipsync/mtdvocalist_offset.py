"""Short-clip-robust lip-sync offset estimation via MTDVocaLiST (Chen et
al., ICASSP 2024, a distilled, 83.5%-smaller version of VocaLiST,
Kadandale et al., Interspeech 2022).

SyncNet's own per-track confidence only becomes reliable once averaged
over many seconds of continuous footage, and the windowed corroboration
scheme on top of it needs at least 3 independent ~1s windows to agree
before trusting a windowed answer; both structurally need several seconds
of same-shot evidence, which genuinely short clips don't have. VocaLiST's
own published benchmark (LRS2) evaluates accuracy at context length K
frames and is dramatically better than SyncNet at the shortest one tested
(K=5 frames = 200ms @ 25fps): 92.8% vs SyncNet's 75.8%. MTDVocaLiST keeps
similar accuracy at 83.5% fewer parameters (13.3M vs 80.1M), which matters
for CPU inference here.

Practically, this changes the unit of independent evidence from "~1s
window" (SyncNet) to "5 video frames = 200ms" (this module), roughly 5x
more independent data points from the same footage. That is the actual
mechanism for being more short-clip-robust: not a better guess from the
same amount of evidence, but more independent samples to corroborate
against in the first place. Reuses the same aggregation primitives as
`syncnet_offset.py` (`SyncNetWindowResult`, `_largest_agreeing_cluster`)
and the same face-track crop files (`crop_face_tracks`) and
active-speaker gate (`_active_speaker_mask`): a second scorer over the
same evidence, not a second pipeline.

Vendored via `analyzer/scripts/fetch_mtdvocalist.sh` into
`analyzer/third_party/mtdvocalist/`, never committed (see
`analyzer/third_party/SOURCES.md`). License note: unlike SyncNet/Light-ASD
(both MIT), MTDVocaLiST ships no license of its own and is built on
VocaLiST (CC-BY-NC) and Wav2Lip (non-commercial) code; treat as
non-commercial, research-use-only.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from syncsentry.lipsync.syncnet_offset import (
    SyncNetEstimate,
    SyncNetTrackResult,
    SyncNetWindowResult,
    _active_speaker_mask,
    _largest_agreeing_cluster,
    crop_face_tracks,
)

_MTDVOCALIST_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "mtdvocalist"
_MODEL_PATH = _MTDVOCALIST_DIR / "pretrained" / "pure_MTDVocaLiST.pth"

# Exactly the constants test_stu.py (the upstream repo's own validated
# evaluation harness) uses: v_context=5 frames matches mel_step_size=16
# mel-columns at hop_size=200/sr=16000 (80 mel-columns/sec), both = 200ms.
_V_CONTEXT = 5
_MEL_STEP = 16
_IMG_SIZE = 96
_FPS = 25.0
_MEL_RATE = 80.0  # mel-columns per second, from hop_size=200 @ sr=16000

# max(scores) - median(scores) across the shift range, in this model's raw
# (unbounded) logit units: not the same scale as SyncNet's
# `DEFAULT_SYNCNET_MIN_CONFIDENCE` (an L2-distance-based median-minus-min).
# On real content, correctly in-sync windows tend to score roughly 9-14,
# so essentially every window with signal at all clears this easily.
# Deliberately conservative rather than tight, since it's the
# corroboration-cluster step (`_largest_agreeing_cluster`) doing the real
# precision work, not this threshold. Re-tune if this model is retrained
# or the crop convention changes.
DEFAULT_MTDVOCALIST_MIN_MARGIN = 5.0

_model_cache: dict[str, object] = {}


class MTDVocaLiSTUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    return _MTDVOCALIST_DIR.exists() and _MODEL_PATH.exists()


def _check_available() -> None:
    if not is_available():
        raise MTDVocaLiSTUnavailable(
            f"MTDVocaLiST not found at {_MTDVOCALIST_DIR}. Run "
            f"`bash analyzer/scripts/fetch_mtdvocalist.sh` once to fetch it. "
            f"Never committed to this repo, see analyzer/third_party/SOURCES.md."
        )


def _load_model():
    _check_available()
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["device"]

    import torch

    sys.path.insert(0, str(_MTDVOCALIST_DIR))
    try:
        from models.student_thin_200_all import SyncTransformer
    finally:
        sys.path.pop(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SyncTransformer(d_model=200).to(device)
    checkpoint = torch.load(str(_MODEL_PATH), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _model_cache["model"] = model
    _model_cache["device"] = device
    return model, device


def _extract_mel(crop_file: Path, tmp_dir: Path):
    """Mel-spectrogram for one crop's audio, normalized exactly as
    upstream's `test_stu.py` does at evaluation time: its actual runtime
    behavior, not the possibly-stale `preemphasize` doc comment in
    `hparams.py`, since the checkpoint was validated against what the
    eval script really computes."""
    import torch
    from scipy.io import wavfile
    from torchaudio.transforms import MelScale

    wav_path = tmp_dir / f"{crop_file.stem}_mtdv.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(crop_file),
        "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr[-2000:]}")

    sr, audio = wavfile.read(str(wav_path))
    wav_tensor = torch.from_numpy(audio.astype(np.float32) / 32768.0)

    n_fft, hop_size, win_size, num_mels = 800, 200, 800, 80
    min_level_db, ref_level_db, max_abs_value = -100, 20, 4.0
    top_db = -min_level_db
    min_level = np.exp(top_db / -20 * np.log(10))

    spec = torch.stft(wav_tensor, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                       window=torch.hann_window(win_size), return_complex=True)
    melscale = MelScale(n_mels=num_mels, sample_rate=sr, f_min=55, f_max=7600,
                         n_stft=n_fft // 2 + 1, norm="slaney", mel_scale="slaney")
    melspec = melscale(torch.abs(spec).float())
    melspec_db = (20 * torch.log10(torch.clamp(melspec, min=min_level))) - ref_level_db
    normalized = torch.clip((2 * max_abs_value) * ((melspec_db + top_db) / top_db) - max_abs_value,
                             -max_abs_value, max_abs_value)
    return normalized  # (num_mels, n_mel_frames)


# SyncNet's crop_video (run_pipeline.py, crop_scale=0.4 default, unchanged
# by `_run_face_track_crop`) pads the tight detected face bbox out to a
# square region of size 2.8*bs centered horizontally on the bbox but
# extended asymmetrically downward (extra 0.8*bs of chin/neck margin), by
# design, since `_mouth_motion_speaking_mask`/`active_speaker.py` rely on
# "bottom half of this crop = mouth region" and that padding is what
# guarantees it. MTDVocaLiST/VocaLiST, however, were trained on Wav2Lip's
# `preprocess.py` crop, which is the tight, unpadded face bbox itself
# (`frame[y1:y2, x1:x2]`, no margin at all) resized straight to 96x96: a
# measurably different framing/scale. Re-deriving where that tight bbox
# sits within SyncNet's already-produced crop (fixed fractions, since
# crop_scale is fixed) avoids a second face-detection pass while matching
# what this model actually expects: (row 0..0.7143, col 0.1429..0.8571) of
# the 224x224 crop.
_TIGHT_ROW_FRAC = (0.0, 0.7143)
_TIGHT_COL_FRAC = (0.1429, 0.8571)


def _extract_video_frames(crop_file: Path) -> np.ndarray:
    """Re-crop to the tight face bbox (see `_TIGHT_ROW_FRAC`/`_TIGHT_COL_FRAC`
    above), then RGB, resized to 96x96 (upstream's `img_size`), lower half
    only (rows 48:96, the mouth/chin region within the *tight* crop,
    which is what Wav2Lip/VocaLiST's own preprocessing relies on).
    Returns shape (n_frames, 48, 96, 3).
    """
    cap = cv2.VideoCapture(str(crop_file))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        r0, r1 = int(_TIGHT_ROW_FRAC[0] * h), int(_TIGHT_ROW_FRAC[1] * h)
        c0, c1 = int(_TIGHT_COL_FRAC[0] * w), int(_TIGHT_COL_FRAC[1] * w)
        tight = frame[r0:r1, c0:c1]
        rgb = cv2.cvtColor(tight, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (_IMG_SIZE, _IMG_SIZE))
        frames.append(rgb[_IMG_SIZE // 2:, :, :])
    cap.release()
    return np.array(frames)


def _score_track_positions(crop_file: Path, vshift: int, stride: int,
                            speaking_mask: np.ndarray | None) -> list[tuple[int, int, float]]:
    """Score every `stride`-th valid center position in this track across
    the full `+-vshift`-frame shift range, exactly like
    `syncnet_offset.py`'s `SyncNetInstance.evaluate()` but with
    MTDVocaLiST as the per-hypothesis scorer instead of L2 distance on raw
    SyncNet embeddings (`test_stu.py`'s `calc_pdist`, adapted to run
    per-track instead of over a whole preprocessed dataset).

    Each call is genuinely expensive on CPU (a 4-layer cross-modal
    transformer, roughly 0.3-3.6s depending on `vshift`; scaling is
    super-linear in the shift-range batch size, not linear, so a smaller
    `vshift` is a much better lever than a smaller `stride`).
    `speaking_mask` is applied here, before scoring, rather than as a
    post-hoc filter on already-computed results, since skipping
    non-speaking positions up front is this function's main defense
    against wasting the majority of its budget on frames with no useful
    signal.

    Returns a list of `(center_frame, best_shift_frames, confidence)`.
    Confidence is `max(scores) - median(scores)` across the `2*vshift+1`
    candidate shifts: how far the best shift's score stands out above the
    overall distribution, not just its immediate neighbor. Adjacent
    shifts are only 40ms apart and legitimately score similarly, so a
    best-vs-runner-up margin is not a useful confidence signal here, just
    an artifact of shift granularity. Deliberately the same shape of
    metric as SyncNet's own median-minus-min confidence in
    `syncnet_offset.py`, mirrored since higher is better here instead of
    lower.
    """
    import torch

    model, device = _load_model()

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        mel = _extract_mel(crop_file, tmp_dir)
        video = _extract_video_frames(crop_file)

    n_frames = len(video)
    if n_frames < _V_CONTEXT + 2 * vshift:
        return []  # track too short for even one full +-vshift scan

    # video[i : i+_V_CONTEXT] frames -> (V_CONTEXT, 48, 96, 3) -> channel-concat (V_CONTEXT*3, 48, 96)
    def video_window(i: int) -> np.ndarray:
        w = video[i:i + _V_CONTEXT]
        return np.concatenate([w[j] for j in range(_V_CONTEXT)], axis=-1).transpose(2, 0, 1)  # (15, 48, 96)

    def mel_window(mel_center_frame: int) -> torch.Tensor:
        mel_col = int(round(_MEL_RATE * (mel_center_frame / _FPS)))
        chunk = mel[:, mel_col:mel_col + _MEL_STEP]
        if chunk.shape[1] < _MEL_STEP:
            chunk = torch.nn.functional.pad(chunk, (0, _MEL_STEP - chunk.shape[1]))
        return chunk.unsqueeze(0)  # (1, num_mels, MEL_STEP)

    results = []
    win_size = 2 * vshift + 1
    with torch.no_grad():
        for center in range(vshift, n_frames - vshift - _V_CONTEXT, stride):
            if speaking_mask is not None and not speaking_mask[center]:
                continue
            vid_np = video_window(center)
            vid_batch = torch.from_numpy(vid_np).float().unsqueeze(0).repeat(win_size, 1, 1, 1).to(device)

            mel_batch = torch.cat([mel_window(center - vshift + s) for s in range(win_size)], dim=0).to(device)

            scores, _ = model(vid_batch, mel_batch.unsqueeze(1))
            scores = scores.cpu().numpy()

            best_idx = int(np.argmax(scores))
            confidence = float(np.max(scores) - np.median(scores))
            shift_frames = vshift - best_idx  # matches test_stu.py's `hparams.v_shift - argmax` convention
            results.append((center, shift_frames, confidence))
    return results


def estimate_mtdvocalist_tracks(video_path: str, vshift: int = 10, stride: int = 2,
                                 gate_mode: str = "auto") -> list[SyncNetTrackResult]:
    """Per-face-track offset evidence from MTDVocaLiST, in the exact same
    `SyncNetTrackResult`/`SyncNetWindowResult` shape `syncnet_offset.py`
    produces. Each "window" here is one `_V_CONTEXT`-frame (200ms)
    position instead of a ~1s SyncNet window, giving several times as many
    independent samples per second of actually-speaking footage.

    `vshift`/`stride` defaults are tuned for CPU feasibility, not
    accuracy: each scored position costs roughly 0.3-3.6s depending on
    `vshift` (see `_score_track_positions` docstring; cost is
    super-linear in `vshift`, so it's the first thing to reduce for speed,
    not `stride`). `vshift=10` (+-400ms search range) covers the
    realistic range of real A/V sync bugs, since broadcast QC pipelines
    rarely see multi-second drift from a single muxing error; if a
    coarser tier's estimate suggests a larger true offset, that's better
    evidence something other than a constant offset is going on than
    reason to widen this search.
    """
    _check_available()

    # `recenter_large_offsets=False`: keeping this detector's existing
    # stance (see docstring above: a coarser tier suggesting a large true
    # offset is treated as evidence against a single constant offset
    # existing at all here, not as something to chase) rather than
    # adopting `syncnet_offset.py`'s opposite bet that it's usually real
    # and worth recentering onto. Off by default anyway, so the two
    # philosophies don't fight each other silently.
    with crop_face_tracks(video_path, recenter_large_offsets=False) as (_data_dir, crop_files, _pre_shift_ms):
        if not crop_files:
            return []

        results = []
        for track_index, crop_file in enumerate(crop_files):
            cap = cv2.VideoCapture(str(crop_file))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            speaking_mask = _active_speaker_mask(crop_file, n_frames=n_frames, gate_mode=gate_mode)
            positions = _score_track_positions(crop_file, vshift=vshift, stride=stride, speaking_mask=speaking_mask)

            windows = []
            for center, shift_frames, margin in positions:
                # (speaking_mask already applied inside _score_track_positions,
                # before the expensive model call: nothing left to filter here.)
                windows.append(SyncNetWindowResult(
                    track_index=track_index, frame_start=center,
                    offset_frames=shift_frames, confidence=margin,
                ))

            if not positions:
                continue
            whole_track_offset = int(np.median([p[1] for p in positions]))
            whole_track_confidence = float(np.median([p[2] for p in positions]))
            results.append(SyncNetTrackResult(
                offset_frames=whole_track_offset, confidence=whole_track_confidence,
                n_frames=n_frames, windows=windows,
            ))
        return results


def estimate_mtdvocalist_offset(video_path: str, vshift: int = 10, frame_rate: float = 25.0,
                                 in_sync_threshold_ms: float = 40.0, stride: int = 2,
                                 min_window_confidence: float = DEFAULT_MTDVOCALIST_MIN_MARGIN,
                                 cluster_tolerance_frames: int = 2,
                                 min_cluster_size: int = 3,
                                 gate_mode: str = "auto") -> SyncNetEstimate:
    """Same two-path aggregation as `syncnet_offset.estimate_syncnet_offset`
    (windowed corroboration, falling back to whole-track median) applied to
    MTDVocaLiST's per-200ms-position evidence instead of SyncNet's
    per-~1s-window evidence. See module docstring for why that unit change
    is what makes this more short-clip-robust in practice, not a
    different aggregation algorithm.
    """
    tracks = estimate_mtdvocalist_tracks(video_path, vshift=vshift, stride=stride, gate_mode=gate_mode)
    if not tracks:
        raise ValueError("No face tracks found (no trackable face for >= min_track frames).")

    all_windows = [w for t in tracks for w in t.windows]
    confident_windows = [w for w in all_windows if w.confidence >= min_window_confidence]
    cluster = _largest_agreeing_cluster(confident_windows, tolerance_frames=cluster_tolerance_frames)

    if len(cluster) >= min_cluster_size:
        offset_frames = np.median([w.offset_frames for w in cluster])
        confidence = np.median([w.confidence for w in cluster])
        n_confident_windows = len(cluster)
    else:
        offset_frames = np.median([t.offset_frames for t in tracks])
        confidence = np.median([t.confidence for t in tracks])
        n_confident_windows = 0

    offset_ms = -offset_frames * (1000.0 / frame_rate)  # SyncNet sign convention: see syncnet_offset module docstring

    if abs(offset_ms) <= in_sync_threshold_ms:
        direction = "in_sync"
    elif offset_ms > 0:
        direction = "audio_lags"
    else:
        direction = "audio_leads"

    return SyncNetEstimate(offset_ms=float(offset_ms), confidence=float(confidence), direction=direction,
                            n_tracks=len(tracks), n_confident_windows=n_confident_windows)
