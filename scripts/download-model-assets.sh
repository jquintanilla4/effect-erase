#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"
DOWNLOAD_SAM31=1
DOWNLOAD_SAM3=0
DOWNLOAD_SAM21=0
DOWNLOAD_EFFECTERASE=1

usage() {
  echo "Usage: $0 [--include-sam3] [--include-sam21] [--skip-sam31] [--skip-effecterase]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-sam3)
      DOWNLOAD_SAM3=1
      shift
      ;;
    --include-sam21)
      DOWNLOAD_SAM21=1
      shift
      ;;
    --skip-sam31)
      DOWNLOAD_SAM31=0
      shift
      ;;
    --skip-effecterase)
      DOWNLOAD_EFFECTERASE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "The Hugging Face CLI ('hf') is required. Install huggingface_hub with CLI support and run 'hf auth login' first." >&2
  exit 1
fi

mkdir -p "$MODELS_DIR"

if [[ "$DOWNLOAD_SAM31" == "1" ]]; then
  mkdir -p "$MODELS_DIR/sam3.1"
  hf download facebook/sam3.1 config.json sam3.1_multiplex.pt --local-dir "$MODELS_DIR/sam3.1"
fi

if [[ "$DOWNLOAD_SAM3" == "1" ]]; then
  mkdir -p "$MODELS_DIR/sam3"
  hf download facebook/sam3 config.json sam3.pt --local-dir "$MODELS_DIR/sam3"
fi

if [[ "$DOWNLOAD_SAM21" == "1" ]]; then
  mkdir -p "$MODELS_DIR/sam2.1"
  wget -O "$MODELS_DIR/sam2.1/sam2.1_hiera_base_plus.pt" \
    "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt"
fi

if [[ "$DOWNLOAD_EFFECTERASE" == "1" ]]; then
  mkdir -p "$MODELS_DIR/EffectErase" "$MODELS_DIR/Wan-AI/Wan2.1-Fun-1.3B-InP"
  hf download FudanCVL/EffectErase EffectErase.ckpt --local-dir "$MODELS_DIR/EffectErase"
  hf download alibaba-pai/Wan2.1-Fun-1.3B-InP --local-dir "$MODELS_DIR/Wan-AI/Wan2.1-Fun-1.3B-InP"
fi

echo "Model download complete."
