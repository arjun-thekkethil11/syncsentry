#!/usr/bin/env bash
# Fetches a small, permissively-licensed real-world dialogue clip for
# local validation of face detection and VAD. See
# analyzer/fixtures/real_content/SOURCES.md for the license. The output
# directory is gitignored; only this fetch script is committed.
#
# Usage:
#   bash analyzer/scripts/fetch_real_content.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../fixtures/real_content"
mkdir -p "$OUT_DIR"

SOURCE_URL="https://commons.wikimedia.org/wiki/Special:FilePath/Interview_with_biologist_Dr_Robyn_Grant_-_Five_things_you_never_knew_about_whiskers_%E2%80%93_The_Royal_Society.webm"
FULL_FILE="$OUT_DIR/royal_society_whiskers.webm"
CLIP_FILE="$OUT_DIR/dialogue_clip.mkv"

if [ ! -f "$FULL_FILE" ]; then
  echo "Downloading source clip (CC BY 3.0, The Royal Society) ..."
  curl -L --fail -o "$FULL_FILE" "$SOURCE_URL"
else
  echo "Source clip already present, skipping download."
fi

echo "Trimming to a 50s continuous talking-head segment (t=140s..190s) ..."
ffmpeg -y -loglevel error -ss 140 -i "$FULL_FILE" -t 50 \
  -c:v libx264 -pix_fmt yuv420p -c:a pcm_s16le "$CLIP_FILE"

echo "Done: $CLIP_FILE"
echo "Run 'pytest analyzer/tests/test_dialogue_scenes_real.py -v' to validate against it."
