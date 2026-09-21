#!/usr/bin/env bash
# Fetches MTDVocaLiST (Chen et al., ICASSP 2024), a distilled, much
# smaller version of VocaLiST: a cross-modal transformer trained to
# score audio/lip sync at short context lengths (as few as 5 video
# frames), unlike SyncNet which needs several continuous seconds per
# track. Used for short-clip sync detection.
#
# Weights come from a GitHub Releases asset directly.
#
# License note: unlike SyncNet/Light-ASD (both MIT), MTDVocaLiST ships no
# license file of its own and is built on VocaLiST (CC-BY-NC,
# non-commercial) and Wav2Lip (non-commercial) code. Treat this as
# non-commercial research use only. See analyzer/third_party/SOURCES.md
# for the full breakdown.
#
# third_party/ is gitignored and never committed.
#
# Usage:
#   bash analyzer/scripts/fetch_mtdvocalist.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../third_party/mtdvocalist"

if [ ! -d "$DEST" ]; then
  echo "Cloning xjchenGit/MTDVocaLiST ..."
  git clone --depth 1 https://github.com/xjchenGit/MTDVocaLiST.git "$DEST"
else
  echo "MTDVocaLiST already cloned, skipping."
fi

mkdir -p "$DEST/pretrained"
if [ ! -f "$DEST/pretrained/pure_MTDVocaLiST.pth" ]; then
  echo "Downloading pretrained weights (~51MB) from GitHub Releases ..."
  curl -fL -o "$DEST/pretrained/pure_MTDVocaLiST.pth" "https://github.com/xjchenGit/MTDVocaLiST/releases/download/v1.0/pure_MTDVocaLiST.pth"
else
  echo "MTDVocaLiST weights already present, skipping."
fi

echo "Done. Short-clip sync scoring is now available as an automatic fallback."
