#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"
DOWNLOAD_SAM31=1
DOWNLOAD_SAM3=0
DOWNLOAD_SAM21=0
DOWNLOAD_EFFECTERASE=1
HF_MAX_ATTEMPTS="${HF_MAX_ATTEMPTS:-3}"

WAN_MODEL_DIR="$MODELS_DIR/Wan-AI/Wan2.1-Fun-1.3B-InP"
WAN_REQUIRED_FILES=(
  "models_t5_umt5-xxl-enc-bf16.pth"
  "Wan2.1_VAE.pth"
  "diffusion_pytorch_model.safetensors"
  "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
)

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

ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    return
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "The Hugging Face CLI ('hf') is required and python3 is not available to install it." >&2
    exit 1
  fi

  echo "Installing Hugging Face CLI..." >&2
  python3 -m pip install --user "huggingface_hub[cli]"
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI ('hf') is still unavailable after installation." >&2
    exit 1
  fi
}

cleanup_hf_locks() {
  local local_dir="$1"
  local download_dir="$local_dir/.cache/huggingface/download"

  if [[ ! -d "$download_dir" ]]; then
    return
  fi

  find "$download_dir" -maxdepth 1 -type f \( -name "*.lock" -o -name "*.lock.stale*" \) -delete
}

hf_download_retry() {
  local local_dir="$1"
  shift

  local attempt=1
  while (( attempt <= HF_MAX_ATTEMPTS )); do
    cleanup_hf_locks "$local_dir"
    if hf download "$@" --local-dir "$local_dir"; then
      cleanup_hf_locks "$local_dir"
      return 0
    fi
    echo "hf download failed for '$*' (attempt $attempt/$HF_MAX_ATTEMPTS)." >&2
    (( attempt += 1 ))
  done

  cleanup_hf_locks "$local_dir"
  return 1
}

recover_single_hf_file() {
  local local_dir="$1"
  local target_name="$2"
  local target_path="$local_dir/$target_name"
  local download_dir="$local_dir/.cache/huggingface/download"
  local cache_file

  if [[ -f "$target_path" ]]; then
    return 0
  fi
  if [[ ! -d "$download_dir" ]]; then
    return 1
  fi

  mapfile -t incomplete_files < <(find "$download_dir" -maxdepth 1 -type f -name "*.incomplete" | sort)
  if [[ "${#incomplete_files[@]}" -ne 1 ]]; then
    return 1
  fi

  cache_file="${incomplete_files[0]}"
  if [[ ! -s "$cache_file" ]]; then
    return 1
  fi

  ln "$cache_file" "$target_path" 2>/dev/null || cp -p "$cache_file" "$target_path"
  [[ -f "$target_path" ]]
}

download_hf_single_file() {
  local repo_id="$1"
  local local_dir="$2"
  local filename="$3"

  mkdir -p "$local_dir"
  if [[ -f "$local_dir/$filename" ]]; then
    return 0
  fi

  if hf_download_retry "$local_dir" "$repo_id" "$filename"; then
    return 0
  fi

  recover_single_hf_file "$local_dir" "$filename"
}

download_sam31_assets() {
  local local_dir="$MODELS_DIR/sam3.1"

  echo "Preparing SAM 3.1 assets..." >&2
  mkdir -p "$local_dir"
  if ! download_hf_single_file "facebook/sam3.1" "$local_dir" "config.json"; then
    return 1
  fi
  if ! download_hf_single_file "facebook/sam3.1" "$local_dir" "sam3.1_multiplex.pt"; then
    return 1
  fi
}

download_sam3_assets() {
  local local_dir="$MODELS_DIR/sam3"

  echo "Preparing SAM 3 assets..." >&2
  mkdir -p "$local_dir"
  if ! download_hf_single_file "facebook/sam3" "$local_dir" "config.json"; then
    return 1
  fi
  if ! download_hf_single_file "facebook/sam3" "$local_dir" "sam3.pt"; then
    return 1
  fi
}

download_sam21_assets() {
  local target_path="$MODELS_DIR/sam2.1/sam2.1_hiera_base_plus.pt"

  if [[ -f "$target_path" ]]; then
    return 0
  fi

  echo "Preparing SAM 2.1 fallback checkpoint..." >&2
  mkdir -p "$MODELS_DIR/sam2.1"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$target_path" \
      "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt"
    return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --output "$target_path" \
      "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt"
    return 0
  fi

  echo "Either wget or curl is required to download SAM 2.1." >&2
  return 1
}

download_effecterase_assets() {
  local effecterase_dir="$MODELS_DIR/EffectErase"

  echo "Preparing EffectErase assets..." >&2
  mkdir -p "$effecterase_dir" "$WAN_MODEL_DIR"

  if ! download_hf_single_file "FudanCVL/EffectErase" "$effecterase_dir" "EffectErase.ckpt"; then
    return 1
  fi

  local required_file
  for required_file in "${WAN_REQUIRED_FILES[@]}"; do
    if [[ -f "$WAN_MODEL_DIR/$required_file" ]]; then
      continue
    fi
    if ! download_hf_single_file "alibaba-pai/Wan2.1-Fun-1.3B-InP" "$WAN_MODEL_DIR" "$required_file"; then
      return 1
    fi
  done

  for required_file in "${WAN_REQUIRED_FILES[@]}"; do
    if [[ ! -f "$WAN_MODEL_DIR/$required_file" ]]; then
      echo "Missing required Wan model asset: $WAN_MODEL_DIR/$required_file" >&2
      return 1
    fi
  done
}

ensure_hf_cli

mkdir -p "$MODELS_DIR"

sam_ready=0
if [[ "$DOWNLOAD_SAM31" == "1" ]]; then
  if download_sam31_assets; then
    sam_ready=1
  else
    echo "SAM 3.1 download failed. Falling back to SAM 2.1." >&2
    DOWNLOAD_SAM21=1
  fi
fi

if [[ "$DOWNLOAD_SAM3" == "1" ]]; then
  download_sam3_assets
fi

if [[ "$DOWNLOAD_SAM21" == "1" ]]; then
  download_sam21_assets
  sam_ready=1
fi

if [[ "$DOWNLOAD_EFFECTERASE" == "1" ]]; then
  download_effecterase_assets
fi

if [[ "$DOWNLOAD_SAM31" == "1" || "$DOWNLOAD_SAM21" == "1" ]]; then
  if [[ "$sam_ready" != "1" ]]; then
    echo "Unable to prepare any SAM checkpoint. Inference will not be available." >&2
    exit 1
  fi
fi

echo "Model download complete."
