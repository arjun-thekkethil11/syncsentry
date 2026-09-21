#!/usr/bin/env bash
# Fetches Light-ASD (Junhua-Liao/Light-ASD, MIT, CVPR 2023), an
# audio-visual active speaker detection model. Used as a multimodal
# (audio + video) check on top of the mouth-motion active-speaker gate:
# a face moving its mouth is not proof that face is the source of the
# audio right now, so Light-ASD is trained specifically to answer that
# question (AVA-ActiveSpeaker benchmark).
#
# Unlike the SyncNet fetch script, weights ship in the repo (a few MB
# each), so this just vendors the MIT-licensed code. See
# analyzer/third_party/SOURCES.md for provenance.
#
# Reuses SyncNet's own face-track crops (same 224x224/25fps/16kHz-mono
# convention) instead of running a second face-detection and tracking
# pass.
#
# third_party/ is gitignored and never committed.
#
# Usage:
#   bash analyzer/scripts/fetch_light_asd.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../third_party/light_asd"

if [ ! -d "$DEST" ]; then
  echo "Cloning Junhua-Liao/Light-ASD (MIT) ..."
  git clone --depth 1 https://github.com/Junhua-Liao/Light-ASD.git "$DEST"
else
  echo "Light-ASD already cloned, skipping."
fi

if [ ! -f "$DEST/weight/pretrain_AVA_CVPR.model" ]; then
  echo "ERROR: expected weight/pretrain_AVA_CVPR.model to ship in the clone but it's missing." >&2
  exit 1
fi

echo "Done. SyncNet's face-track crops will now also be scored by Light-ASD"
echo "for real active-speaker gating (automatic; no new CLI flag)."
