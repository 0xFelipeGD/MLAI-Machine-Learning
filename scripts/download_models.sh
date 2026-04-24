#!/usr/bin/env bash
# scripts/download_models.sh — Download trained .tflite models from GitHub Releases.
#
# Why a release instead of committing / git-LFS?
#   The .tflite binaries are excluded from the repo history via .gitignore
#   (commits would bloat the repo every retrain). They live as assets on a
#   tagged GitHub Release — version-pinned, cacheable, and free.
#
# Usage:
#   ./scripts/download_models.sh                  # pulls DEFAULT_TAG
#   MLAI_MODELS_TAG=v1.2.0 ./scripts/download_models.sh
#   ./scripts/download_models.sh v1.2.0           # positional tag override
#
# Requirements: curl.
#
# The release must contain these assets (flat, not zipped):
#   fruit_detector.tflite
#   fruit_detector.labels.txt
#   fruit_quality.tflite
#   fruit_quality.labels.txt
#
# If the release or an asset is missing, this script exits non-zero so the
# error is loud — the engine would otherwise silently fall back to MOCK MODE.

set -euo pipefail

# ---- Config ------------------------------------------------------------------
REPO="0xFelipeGD/MLAI-Machine-Learning"
DEFAULT_TAG="v1.0.0"
TAG="${1:-${MLAI_MODELS_TAG:-${DEFAULT_TAG}}}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGRO_DIR="${ROOT}/models/agro"
mkdir -p "${AGRO_DIR}"

ASSETS=(
    fruit_detector.tflite
    fruit_detector.labels.txt
    fruit_quality.tflite
    fruit_quality.labels.txt
)

BASE="https://github.com/${REPO}/releases/download/${TAG}"

# ---- Download ----------------------------------------------------------------
echo "Fetching MLAI models from release ${TAG} of ${REPO}..."
for asset in "${ASSETS[@]}"; do
    dest="${AGRO_DIR}/${asset}"
    url="${BASE}/${asset}"
    echo "  -> ${asset}"
    # --fail: exit non-zero on HTTP error; -L: follow redirects; -sS: silent but show errors.
    if ! curl --fail -L -sS -o "${dest}" "${url}"; then
        echo "ERROR: failed to download ${url}" >&2
        echo "Check that release ${TAG} exists and contains ${asset} as a flat asset." >&2
        exit 1
    fi
done

# ---- Verify ------------------------------------------------------------------
echo
echo "Models installed in ${AGRO_DIR}:"
for asset in "${ASSETS[@]}"; do
    f="${AGRO_DIR}/${asset}"
    size=$(stat -c '%s' "${f}" 2>/dev/null || stat -f '%z' "${f}")
    printf "  %-32s %s bytes\n" "${asset}" "${size}"
done

echo
echo "Done. The engine will pick these up on next start."
