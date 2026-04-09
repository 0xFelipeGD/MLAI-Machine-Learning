#!/usr/bin/env bash
# scripts/download_models.sh — Prepare the models/ directory.
#
# MLAI does not ship pre-trained .tflite files (different users want different
# categories). Run training/ on your PC and copy the resulting files to the
# locations printed below. This script just creates the folders and prints
# instructions.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INDUST_DIR="${ROOT}/models/indust"
AGRO_DIR="${ROOT}/models/agro"

mkdir -p "${INDUST_DIR}" "${AGRO_DIR}"

cat <<EOF
================================================================
MLAI model directories ready:

  ${INDUST_DIR}
  ${AGRO_DIR}

Drop your trained models in like this:

  models/indust/padim_toothbrush.tflite     ← or padim_bottle.tflite, etc.
                                              (must match config/indust/config.yaml
                                               active_category and categories list)

  models/agro/fruit_detector.tflite         ← COCO SSD MobileNet V1 from
                                              training/README.md §3 (no training)
  models/agro/fruit_detector.labels.txt     ← bundled with the COCO SSD download
  models/agro/fruit_quality.tflite          ← from training/README.md §4
  models/agro/fruit_quality.labels.txt      ← written by train_quality.py

To train them yourself see training/README.md.

If a model file is missing the engine starts in MOCK MODE so the
rest of the system (camera, API, dashboard) still works for development.
================================================================
EOF
