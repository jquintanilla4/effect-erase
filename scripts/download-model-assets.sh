#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"
ASSET_MANIFEST_PATH="$MODELS_DIR/asset-manifest.tsv"
DOWNLOAD_SAM31=1
DOWNLOAD_SAM3=0
DOWNLOAD_SAM21=0
DOWNLOAD_EFFECTERASE=1
HF_MAX_ATTEMPTS="${HF_MAX_ATTEMPTS:-3}"
ASSET_PARTIALS_DIR_NAME=".asset-partials"
ASSET_WORK_PERFORMED=0
NEEDS_HF_CLI=0
REQUESTED_ASSET_WORK_NEEDED=0

WAN_MODEL_DIR="$MODELS_DIR/Wan-AI/Wan2.1-Fun-1.3B-InP"
WAN_TOKENIZER_DIR="$WAN_MODEL_DIR/google/umt5-xxl"
WAN_REQUIRED_FILES=(
  "models_t5_umt5-xxl-enc-bf16.pth"
  "Wan2.1_VAE.pth"
  "diffusion_pytorch_model.safetensors"
  "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
)
WAN_TOKENIZER_REQUIRED_FILES=(
  "tokenizer_config.json"
  "tokenizer.json"
  "spiece.model"
  "special_tokens_map.json"
)
ASSET_SPECS=(
  "DOWNLOAD_SAM31|hf|$MODELS_DIR/sam3.1|config.json"
  "DOWNLOAD_SAM31|hf|$MODELS_DIR/sam3.1|sam3.1_multiplex.pt"
  "DOWNLOAD_SAM3|hf|$MODELS_DIR/sam3|config.json"
  "DOWNLOAD_SAM3|hf|$MODELS_DIR/sam3|sam3.pt"
  "DOWNLOAD_SAM21|direct|$MODELS_DIR/sam2.1|sam2.1_hiera_base_plus.pt"
  "DOWNLOAD_EFFECTERASE|hf|$MODELS_DIR/EffectErase|EffectErase.ckpt"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_MODEL_DIR|models_t5_umt5-xxl-enc-bf16.pth"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_MODEL_DIR|Wan2.1_VAE.pth"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_MODEL_DIR|diffusion_pytorch_model.safetensors"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_MODEL_DIR|models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_TOKENIZER_DIR|tokenizer_config.json"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_TOKENIZER_DIR|tokenizer.json"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_TOKENIZER_DIR|spiece.model"
  "DOWNLOAD_EFFECTERASE|hf|$WAN_TOKENIZER_DIR|special_tokens_map.json"
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

asset_target_path() {
  local local_dir="$1"
  local filename="$2"
  echo "$local_dir/$filename"
}

asset_marker_path() {
  local local_dir="$1"
  local filename="$2"
  echo "$local_dir/$ASSET_PARTIALS_DIR_NAME/$filename.partial"
}

mark_asset_incomplete() {
  local local_dir="$1"
  local filename="$2"
  local marker_path

  marker_path="$(asset_marker_path "$local_dir" "$filename")"
  mkdir -p "$(dirname "$marker_path")"
  : > "$marker_path"
}

clear_asset_marker() {
  local local_dir="$1"
  local filename="$2"

  rm -f "$(asset_marker_path "$local_dir" "$filename")"
}

asset_size_bytes() {
  local path="$1"
  wc -c < "$path" | tr -d '[:space:]'
}

validate_pytorch_zip_checkpoint() {
  local target_path="$1"

  if ! command -v python3 >/dev/null 2>&1; then
    # Validation is a safety net, not a hard prerequisite for bootstrap.
    echo "Skipping checkpoint validation because python3 is unavailable: $target_path" >&2
    return 0
  fi

  python3 - "$target_path" <<'PY'
import sys
import zipfile

target_path = sys.argv[1]

try:
    if not zipfile.is_zipfile(target_path):
        raise RuntimeError("file is not a zip-based PyTorch checkpoint")

    with zipfile.ZipFile(target_path) as archive:
        bad_entry = archive.testzip()
        if bad_entry is not None:
            raise RuntimeError(f"archive entry failed CRC validation: {bad_entry}")
except Exception as error:
    print(error, file=sys.stderr)
    raise SystemExit(1)
PY
}

checkpoint_requires_validation() {
  local filename="$1"
  case "$filename" in
    *.pt|*.pth|*.ckpt)
      return 0
      ;;
  esac
  return 1
}

validate_checkpoint_if_needed() {
  local local_dir="$1"
  local filename="$2"
  local target_path

  target_path="$(asset_target_path "$local_dir" "$filename")"
  if ! checkpoint_requires_validation "$filename"; then
    return 0
  fi
  if [[ ! -s "$target_path" ]]; then
    return 1
  fi

  if validate_pytorch_zip_checkpoint "$target_path"; then
    clear_asset_marker "$local_dir" "$filename"
    return 0
  fi

  echo "Detected a corrupt checkpoint. Removing it so it is not reused: $target_path" >&2
  rm -f "$target_path"
  mark_asset_incomplete "$local_dir" "$filename"
  return 1
}

repair_corrupt_checkpoint_if_needed() {
  local local_dir="$1"
  local filename="$2"
  validate_checkpoint_if_needed "$local_dir" "$filename" || true
}

asset_status() {
  local kind="$1"
  local local_dir="$2"
  local filename="$3"
  local target_path

  target_path="$(asset_target_path "$local_dir" "$filename")"

  if [[ -f "$target_path" ]]; then
    if [[ -s "$target_path" ]]; then
      echo "present"
    else
      echo "incomplete"
    fi
    return
  fi

  if [[ -f "$(asset_marker_path "$local_dir" "$filename")" ]]; then
    echo "incomplete"
    return
  fi

  if [[ "$kind" == "direct" && -f "$target_path.partial" ]]; then
    echo "incomplete"
    return
  fi

  echo "missing"
}

asset_requested_label() {
  local flag_var="$1"
  if [[ "${!flag_var}" == "1" ]]; then
    echo "requested"
    return
  fi

  echo "standby"
}

asset_relative_path() {
  local local_dir="$1"
  local filename="$2"
  echo "${local_dir#$MODELS_DIR/}/$filename"
}

# The manifest is intentionally text-first and cheap to regenerate so repeat
# runs can show a complete before/after asset view without extra tooling.
render_asset_manifest() {
  local print_output="${1:-0}"
  local spec
  local flag_var
  local kind
  local local_dir
  local filename
  local status
  local scope
  local target_path
  local size_bytes
  local relative_path

  NEEDS_HF_CLI=0
  REQUESTED_ASSET_WORK_NEEDED=0

  printf "scope\tstatus\tsize_bytes\tpath\n" > "$ASSET_MANIFEST_PATH"

  if [[ "$print_output" == "1" ]]; then
    echo "Model asset manifest before download:"
  fi

  for spec in "${ASSET_SPECS[@]}"; do
    IFS="|" read -r flag_var kind local_dir filename <<< "$spec"
    status="$(asset_status "$kind" "$local_dir" "$filename")"
    scope="$(asset_requested_label "$flag_var")"
    target_path="$(asset_target_path "$local_dir" "$filename")"
    relative_path="$(asset_relative_path "$local_dir" "$filename")"
    size_bytes="0"

    if [[ -f "$target_path" ]]; then
      size_bytes="$(asset_size_bytes "$target_path")"
    fi

    printf "%s\t%s\t%s\t%s\n" "$scope" "$status" "$size_bytes" "$relative_path" >> "$ASSET_MANIFEST_PATH"

    if [[ "$print_output" == "1" ]]; then
      if [[ "$size_bytes" != "0" ]]; then
        echo "- [$scope] $status: $relative_path (${size_bytes} bytes)"
      else
        echo "- [$scope] $status: $relative_path"
      fi
    fi

    if [[ "${!flag_var}" == "1" && "$status" != "present" ]]; then
      REQUESTED_ASSET_WORK_NEEDED=1
      if [[ "$kind" == "hf" ]]; then
        NEEDS_HF_CLI=1
      fi
    fi
  done
}

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
  local target_path

  target_path="$(asset_target_path "$local_dir" "$filename")"

  mkdir -p "$local_dir"
  if [[ -f "$target_path" && ! -s "$target_path" ]]; then
    rm -f "$target_path"
  fi

  if [[ -s "$target_path" ]]; then
    if validate_checkpoint_if_needed "$local_dir" "$filename"; then
      return 0
    fi
  fi

  mark_asset_incomplete "$local_dir" "$filename"
  ASSET_WORK_PERFORMED=1

  if hf_download_retry "$local_dir" "$repo_id" "$filename"; then
    if validate_checkpoint_if_needed "$local_dir" "$filename"; then
      return 0
    fi
  fi

  if recover_single_hf_file "$local_dir" "$filename" && validate_checkpoint_if_needed "$local_dir" "$filename"; then
    return 0
  fi

  return 1
}

download_sam31_assets() {
  local local_dir="$MODELS_DIR/sam3.1"

  repair_corrupt_checkpoint_if_needed "$local_dir" "sam3.1_multiplex.pt"
  if [[ "$(asset_status "hf" "$local_dir" "config.json")" == "present" && "$(asset_status "hf" "$local_dir" "sam3.1_multiplex.pt")" == "present" ]]; then
    return 0
  fi

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

  repair_corrupt_checkpoint_if_needed "$local_dir" "sam3.pt"
  if [[ "$(asset_status "hf" "$local_dir" "config.json")" == "present" && "$(asset_status "hf" "$local_dir" "sam3.pt")" == "present" ]]; then
    return 0
  fi

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
  local local_dir="$MODELS_DIR/sam2.1"
  local filename="sam2.1_hiera_base_plus.pt"
  local target_path="$MODELS_DIR/sam2.1/sam2.1_hiera_base_plus.pt"
  local temp_path="$target_path.partial"

  repair_corrupt_checkpoint_if_needed "$local_dir" "$filename"
  if [[ "$(asset_status "direct" "$local_dir" "$filename")" == "present" ]]; then
    clear_asset_marker "$local_dir" "$filename"
    return 0
  fi

  echo "Preparing SAM 2.1 fallback checkpoint..." >&2
  mkdir -p "$local_dir"
  rm -f "$target_path"
  mark_asset_incomplete "$local_dir" "$filename"
  ASSET_WORK_PERFORMED=1

  if command -v wget >/dev/null 2>&1; then
    if wget -O "$temp_path" \
      "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt"; then
      mv "$temp_path" "$target_path"
      if validate_checkpoint_if_needed "$local_dir" "$filename"; then
        return 0
      fi
    fi
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -L --fail --output "$temp_path" \
      "https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt"; then
      mv "$temp_path" "$target_path"
      if validate_checkpoint_if_needed "$local_dir" "$filename"; then
        return 0
      fi
    fi
  fi

  echo "Either wget or curl is required to download SAM 2.1." >&2
  return 1
}

download_effecterase_assets() {
  local effecterase_dir="$MODELS_DIR/EffectErase"
  local work_needed=0
  local required_file

  repair_corrupt_checkpoint_if_needed "$effecterase_dir" "EffectErase.ckpt"
  for required_file in "${WAN_REQUIRED_FILES[@]}"; do
    repair_corrupt_checkpoint_if_needed "$WAN_MODEL_DIR" "$required_file"
  done

  if [[ "$(asset_status "hf" "$effecterase_dir" "EffectErase.ckpt")" != "present" ]]; then
    work_needed=1
  fi

  for required_file in "${WAN_REQUIRED_FILES[@]}"; do
    if [[ "$(asset_status "hf" "$WAN_MODEL_DIR" "$required_file")" != "present" ]]; then
      work_needed=1
      break
    fi
  done
  if [[ "$work_needed" != "1" ]]; then
    for required_file in "${WAN_TOKENIZER_REQUIRED_FILES[@]}"; do
      if [[ "$(asset_status "hf" "$WAN_TOKENIZER_DIR" "$required_file")" != "present" ]]; then
        work_needed=1
        break
      fi
    done
  fi

  if [[ "$work_needed" != "1" ]]; then
    return 0
  fi

  echo "Preparing EffectErase assets..." >&2
  mkdir -p "$effecterase_dir" "$WAN_MODEL_DIR" "$WAN_TOKENIZER_DIR"

  if ! download_hf_single_file "FudanCVL/EffectErase" "$effecterase_dir" "EffectErase.ckpt"; then
    return 1
  fi

  for required_file in "${WAN_REQUIRED_FILES[@]}"; do
    if [[ "$(asset_status "hf" "$WAN_MODEL_DIR" "$required_file")" == "present" ]]; then
      clear_asset_marker "$WAN_MODEL_DIR" "$required_file"
      continue
    fi
    if ! download_hf_single_file "alibaba-pai/Wan2.1-Fun-1.3B-InP" "$WAN_MODEL_DIR" "$required_file"; then
      return 1
    fi
  done

  for required_file in "${WAN_TOKENIZER_REQUIRED_FILES[@]}"; do
    if [[ "$(asset_status "hf" "$WAN_TOKENIZER_DIR" "$required_file")" == "present" ]]; then
      clear_asset_marker "$WAN_TOKENIZER_DIR" "$required_file"
      continue
    fi
    if ! download_hf_single_file "google/umt5-xxl" "$WAN_TOKENIZER_DIR" "$required_file"; then
      return 1
    fi
  done

  for required_file in "${WAN_REQUIRED_FILES[@]}"; do
    if [[ ! -f "$WAN_MODEL_DIR/$required_file" ]]; then
      echo "Missing required Wan model asset: $WAN_MODEL_DIR/$required_file" >&2
      return 1
    fi
  done
  for required_file in "${WAN_TOKENIZER_REQUIRED_FILES[@]}"; do
    if [[ ! -f "$WAN_TOKENIZER_DIR/$required_file" ]]; then
      echo "Missing required Wan tokenizer asset: $WAN_TOKENIZER_DIR/$required_file" >&2
      return 1
    fi
  done
}

mkdir -p "$MODELS_DIR"
render_asset_manifest 1

if [[ "$NEEDS_HF_CLI" == "1" ]]; then
  ensure_hf_cli
fi

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

render_asset_manifest 0

if [[ "$REQUESTED_ASSET_WORK_NEEDED" != "1" && "$ASSET_WORK_PERFORMED" != "1" ]]; then
  echo "Model asset check complete. No requested downloads needed."
else
  echo "Model download complete."
fi
echo "Asset manifest written to $ASSET_MANIFEST_PATH"
