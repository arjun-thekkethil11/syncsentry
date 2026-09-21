"""Real audio-visual active-speaker detection (Light-ASD, Liao et al.,
CVPR 2023), used to gate SyncNet's windowed scoring (`syncnet_offset.py`)
with an actual multimodal classifier instead of a mouth-motion-only proxy.

`_mouth_motion_speaking_mask` in `syncnet_offset.py` can only ask "is this
face's mouth moving right now", a cheap, fully automated approximation of
"is this face the one talking right now", but only an approximation. On
genuine multi-speaker dialogue two things break that assumption: (a)
someone can move their mouth while not being the audio source (laughing,
reacting, mouthing along), and (b) the audio half of "is this the source
of the audio" never enters the mouth-motion signal at all, since it's
video-only. Light-ASD is trained end-to-end on real audio-visual
active-speaker data (AVA-ActiveSpeaker) to answer exactly that joint
question.

Vendored via `analyzer/scripts/fetch_light_asd.sh` into
`analyzer/third_party/light_asd/`, never committed (see
`analyzer/third_party/SOURCES.md`). Only `model.Model.ASD_Model` and
`loss.lossAV` are imported directly (plain `nn.Module`s, no hardcoded
device); `light_asd`'s own `ASD.py` wrapper class hardcodes `.cuda()` in
its constructor, so it's deliberately bypassed rather than patched.

Reuses SyncNet's own face-track crop files (same 224x224 @ 25fps +
16kHz-mono-audio .avi convention; Light-ASD's own crop stage is explicitly
"modified based on" joonson/syncnet_python, see Columbia_test.py's
`crop_video`) instead of running a second face detection/tracking pass on
the same video.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_LIGHT_ASD_DIR = Path(__file__).resolve().parent.parent.parent / "third_party" / "light_asd"
_MODEL_PATH = _LIGHT_ASD_DIR / "weight" / "pretrain_AVA_CVPR.model"

_AUDIO_FPS = 100  # python_speech_features.mfcc with winstep=0.01
_VIDEO_FPS = 25
_AUDIO_TO_VIDEO_RATIO = _AUDIO_FPS // _VIDEO_FPS  # the model's audio encoder downsamples 4x to match
_CHUNK_VIDEO_FRAMES = 150  # ~6s per forward pass: bounds memory on long tracks, no accuracy tradeoff

_model_cache: dict[str, object] = {}


class ActiveSpeakerUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    return _LIGHT_ASD_DIR.exists() and _MODEL_PATH.exists()


def _check_available() -> None:
    if not is_available():
        raise ActiveSpeakerUnavailable(
            f"Light-ASD not found at {_LIGHT_ASD_DIR}. Run "
            f"`bash analyzer/scripts/fetch_light_asd.sh` once to fetch the "
            f"(MIT-licensed) reference implementation and pretrained weights. "
            f"Never committed to this repo, see analyzer/third_party/SOURCES.md."
        )


def _load_model():
    """Loads once per process (multiple face tracks per video all reuse
    this). Bypasses `light_asd/ASD.py`'s wrapper class, which hardcodes
    `.cuda()` in its constructor, and imports the two plain `nn.Module`
    pieces it actually composes instead, so device placement stays under
    this project's control (CPU here; there's no CUDA in this deployment
    target)."""
    _check_available()

    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["loss_av"], _model_cache["device"]

    import torch

    sys.path.insert(0, str(_LIGHT_ASD_DIR))
    try:
        from model.Model import ASD_Model
        from loss import lossAV
    finally:
        sys.path.pop(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ASD_Model().to(device)
    loss_av = lossAV().to(device)

    # The checkpoint's keys were saved from a wrapper (`ASD` in ASD.py)
    # that owns this model as `self.model.*` and this loss as
    # `self.lossAV.*`. Split the combined state dict back apart by
    # prefix rather than loading it into a wrapper we're deliberately not
    # instantiating (see module docstring).
    state = torch.load(str(_MODEL_PATH), map_location=device, weights_only=True)
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    loss_av_state = {k[len("lossAV."):]: v for k, v in state.items() if k.startswith("lossAV.")}
    model.load_state_dict(model_state)
    loss_av.load_state_dict(loss_av_state)
    model.eval()
    loss_av.eval()

    _model_cache["model"] = model
    _model_cache["loss_av"] = loss_av
    _model_cache["device"] = device
    return model, loss_av, device


def _extract_audio_mfcc(crop_file: Path, tmp_dir: Path) -> np.ndarray:
    from python_speech_features import mfcc
    from scipy.io import wavfile

    wav_path = tmp_dir / f"{crop_file.stem}_asd.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(crop_file),
        "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr[-2000:]}")

    sr, audio = wavfile.read(str(wav_path))
    return mfcc(audio, sr, numcep=13, winlen=0.025, winstep=0.010)


def _extract_video_frames(crop_file: Path) -> np.ndarray:
    """Grayscale, resized to 224x224 (a no-op for SyncNet's crops, which
    are already exactly that size) then center-cropped to 112x112 --
    matches Light-ASD's own preprocessing exactly (`Columbia_test.py`
    `evaluate_network`), needed because that's what the pretrained
    weights were trained to expect."""
    cap = cv2.VideoCapture(str(crop_file))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray[56:168, 56:168]  # center 112x112 of a 224x224 frame
        frames.append(gray)
    cap.release()
    return np.array(frames)


def score_track(crop_file: Path, n_frames: int) -> np.ndarray:
    """Real per-frame active-speaker score for one SyncNet face-track crop
    file: positive means Light-ASD judges this face to be the audio's
    source at that frame, negative means not (its own decision boundary --
    same convention as the upstream repo's visualization, which colors a
    face green when `score >= 0`). Returns a `n_frames`-length array
    (padded with the last value if the track is shorter, same convention
    as `syncnet_offset._mouth_motion_speaking_mask`).
    """
    import torch

    model, loss_av, device = _load_model()

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        audio_feature = _extract_audio_mfcc(crop_file, tmp_dir)
        video_feature = _extract_video_frames(crop_file)

    if len(video_feature) == 0:
        return np.zeros(n_frames)

    # The audio encoder downsamples 4x internally to match video's frame
    # rate (MFCC is ~100fps, video is 25fps): truncate both streams to the
    # longest length where video_frames * 4 <= audio_frames still holds,
    # rather than comparing durations in seconds directly.
    usable_video_frames = min(len(video_feature), audio_feature.shape[0] // _AUDIO_TO_VIDEO_RATIO)
    if usable_video_frames < 1:
        return np.zeros(n_frames)

    scores = []
    with torch.no_grad():
        for start in range(0, usable_video_frames, _CHUNK_VIDEO_FRAMES):
            end = min(start + _CHUNK_VIDEO_FRAMES, usable_video_frames)
            v_chunk = video_feature[start:end]
            a_chunk = audio_feature[start * _AUDIO_TO_VIDEO_RATIO:end * _AUDIO_TO_VIDEO_RATIO]

            input_v = torch.FloatTensor(v_chunk).unsqueeze(0).to(device)
            input_a = torch.FloatTensor(a_chunk).unsqueeze(0).to(device)

            embed_a = model.forward_audio_frontend(input_a)
            embed_v = model.forward_visual_frontend(input_v)
            out = model.forward_audio_visual_backend(embed_a, embed_v)
            chunk_scores = loss_av.forward(out, labels=None)
            scores.extend(np.asarray(chunk_scores).reshape(-1).tolist())

    scores = np.array(scores)
    if len(scores) >= n_frames:
        return scores[:n_frames]
    return np.pad(scores, (0, n_frames - len(scores)), mode="edge")
