#!/usr/bin/env bash
# Fetches the pretrained SyncNet reference implementation + weights (M3c,
# the real learned lip-sync estimator -- see docs/RESEARCH.md sections 1d/1e
# for why the classical mouth-motion heuristic wasn't enough).
#
# Vendors joonson/syncnet_python (MIT-licensed code) into
# analyzer/third_party/syncnet_python/, and downloads its two pretrained
# model files (SyncNet itself + the S3FD face detector it depends on) from
# a Hugging Face mirror -- the original robotics.ox.ac.uk host has an
# expired/self-signed TLS cert as of this writing. See
# analyzer/third_party/SOURCES.md for full provenance + license notes.
#
# The whole third_party/ directory is gitignored: never committed, same
# "fetch, don't commit" policy already used for the real-content clip.
#
# Usage:
#   bash analyzer/scripts/fetch_syncnet.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../third_party/syncnet_python"

if [ ! -d "$DEST" ]; then
  echo "Cloning joonson/syncnet_python (MIT) ..."
  git clone --depth 1 https://github.com/joonson/syncnet_python.git "$DEST"
else
  echo "syncnet_python already cloned, skipping."
fi

mkdir -p "$DEST/data" "$DEST/detectors/s3fd/weights"

if [ ! -f "$DEST/data/syncnet_v2.model" ]; then
  echo "Downloading SyncNet weights (~52MB) from Hugging Face mirror ..."
  curl -fL -o "$DEST/data/syncnet_v2.model" \
    "https://huggingface.co/lithiumice/syncnet/resolve/main/syncnet_v2.model"
else
  echo "SyncNet weights already present, skipping."
fi

if [ ! -f "$DEST/detectors/s3fd/weights/sfd_face.pth" ]; then
  echo "Downloading S3FD face-detector weights (~86MB) from Hugging Face mirror ..."
  curl -fL -o "$DEST/detectors/s3fd/weights/sfd_face.pth" \
    "https://huggingface.co/lithiumice/syncnet/resolve/main/sfd_face.pth"
else
  echo "S3FD weights already present, skipping."
fi

echo "Installing extra Python dependencies (python_speech_features, tqdm) ..."
pip install -q python_speech_features tqdm

echo "Done. Try: syncsentry fix --video <file> --out-dir ./out --use-syncnet"
