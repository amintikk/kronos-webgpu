#!/usr/bin/env bash
# Build and publish to the kronoswebgpu container (~/docker/kronoswebgpu), served at https://kronos.aminox.dev
# Pass a directory with exported models as first argument to publish them too (see tools/export_kronos.py).
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${KRONOS_DEPLOY_DIR:-$HOME/docker/kronoswebgpu/data}"  # override for your own server

npm run build
rsync -a --delete dist/ "$TARGET/html/"
echo "Web publicada en $TARGET/html"

if [ "${1:-}" != "" ]; then
  rsync -a --delete --include='*.onnx' --include='manifest.json' --exclude='*' "$1/" "$TARGET/models/"
  echo "Modelos publicados en $TARGET/models"
fi
