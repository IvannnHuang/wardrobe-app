#!/usr/bin/env bash
# Downloads the on-device ML models into the app's assets and the Python prototype.
# Run once after cloning (from the repo root):  bash scripts/fetch-models.sh
# The models are bundled in the APK; the app itself never downloads anything.
set -euo pipefail
cd "$(dirname "$0")/.."

ANDROID_DIR=android/app/src/main/assets/models
PROTO_DIR=prototype/models
mkdir -p "$ANDROID_DIR" "$PROTO_DIR"

# name | url | sha256
MODELS=(
  "pose_landmarker_full.task|https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task|4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name url sha <<< "$entry"
  target="$ANDROID_DIR/$name"
  if [ ! -f "$target" ] || ! echo "$sha  $target" | sha256sum -c --status; then
    echo "Downloading $name"
    curl -sSfL -o "$target.part" "$url"
    mv "$target.part" "$target"
  fi
  if ! echo "$sha  $target" | sha256sum -c --status; then
    echo "Checksum mismatch for $name (the file at $url changed). Update the pinned sha256 on purpose." >&2
    exit 1
  fi
  cp "$target" "$PROTO_DIR/$name"
  echo "OK  $name"
done
