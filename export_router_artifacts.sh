#!/usr/bin/env bash
# Package trained router artifacts for deployment
# Usage:
#   bash export_router_artifacts.sh <MODEL_DIR> [OUTPUT_DIR]
# Example:
#   bash export_router_artifacts.sh router_model exports

set -euo pipefail

MODEL_DIR=${1:-router_model}
OUT_DIR=${2:-exports}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="router_artifacts_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"

# Ensure model dir exists
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "✗ Model directory not found: ${MODEL_DIR}" >&2
  echo "  Train first, e.g.: python train_router.py --train_data data/router_train.jsonl --output_dir router_model" >&2
  exit 1
fi

# Required files inside model dir
REQ_IN_MODEL=("router_model.pt" "router_config.json")
MISSING=0
for f in "${REQ_IN_MODEL[@]}"; do
  if [[ ! -f "${MODEL_DIR}/${f}" ]]; then
    echo "✗ Missing ${MODEL_DIR}/${f}" >&2
    MISSING=1
  fi
done
if [[ ${MISSING} -ne 0 ]]; then
  echo "Please re-train or ensure the model directory is complete." >&2
  exit 1
fi

# Optional tokenizer files (created by train_router.py)
# DistilBERT tokenizer typically includes vocab.txt, tokenizer.json, special_tokens_map.json, tokenizer_config.json

mkdir -p "${OUT_DIR}"

# Build a temp manifest to package
TMPDIR=$(mktemp -d)
MANIFEST="${TMPDIR}/MANIFEST.txt"

{
  echo "Router Deployment Artifacts"
  echo "Created: ${TIMESTAMP}"
  echo
  echo "Included:"
  echo "- ${MODEL_DIR}/ (weights + router_config.json + tokenizer files)"
  [[ -f router_config.yaml ]] && echo "- router_config.yaml"
  [[ -f requirements_router.txt ]] && echo "- requirements_router.txt"
} > "${MANIFEST}"

# Create tarball
# Include model dir + minimal config/requirements if present
ARGS=("${MODEL_DIR}")
[[ -f router_config.yaml ]] && ARGS+=("router_config.yaml")
[[ -f requirements_router.txt ]] && ARGS+=("requirements_router.txt")
ARGS+=("-C" "${TMPDIR}" "MANIFEST.txt")

tar -czf "${ARCHIVE_PATH}" "${ARGS[@]}"

# Cleanup
rm -rf "${TMPDIR}"

# Report
SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)
echo "✓ Created ${ARCHIVE_PATH} (${SIZE})"
echo "Contents:"
 tar -tzf "${ARCHIVE_PATH}" | sed 's/^/  - /'
