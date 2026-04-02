#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime-config.sh"

if [[ -d "$HOME/.local/bin" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

STORAGE_ROOT_INPUT="${STORAGE_ROOT:-}"
STORAGE_ROOT="${STORAGE_ROOT_INPUT:-}"
WORKER_BOOTSTRAP_STATE_PATH="${WORKER_BOOTSTRAP_STATE_PATH:-}"
STORAGE_ROOT_EXPLICIT=0
DRY_RUN=0

usage() {
  echo "Usage: $0 [--storage-root PATH] [--dry-run]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --storage-root)
      STORAGE_ROOT="$2"
      STORAGE_ROOT_EXPLICIT=1
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
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

if [[ -n "${MODELS_DIR:-}" && -z "${WORKER_MODELS_DIR:-}" ]]; then
  WORKER_MODELS_DIR="$MODELS_DIR"
fi

if [[ -z "${WORKER_MODELS_DIR:-}" ]]; then
  STATE_PATH="$(locate_bootstrap_state)"
  if [[ -f "$STATE_PATH" ]]; then
    eval "$(state_exports "$STATE_PATH")"
  fi
fi

resolve_runtime_layout "$STORAGE_ROOT"
export_runtime_layout

MODELS_DIR="$MODELS_DIR"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Models directory not found: $MODELS_DIR"
  exit 0
fi

relative_path() {
  local path="$1"
  if [[ "$path" == "$MODELS_DIR"/* ]]; then
    echo "${path#$MODELS_DIR/}"
    return
  fi

  echo "$path"
}

mapfile -t CLEANUP_TARGETS < <(
  {
    # Zero-byte files are treated as incomplete assets by the download script.
    find "$MODELS_DIR" -type f -size 0
    # Direct-download temp files are left as *.partial until the final move.
    find "$MODELS_DIR" -type f -name "*.partial"
    # Our own marker files track interrupted Hugging Face asset downloads.
    find "$MODELS_DIR" -type f -path "*/.asset-partials/*.partial"
    # Hugging Face may leave incomplete cache blobs behind after a failed run.
    find "$MODELS_DIR" -type f -path "*/.cache/huggingface/download/*.incomplete"
    # Clearing stale lock files keeps the next download pass from tripping on them.
    find "$MODELS_DIR" -type f \( -name "*.lock" -o -name "*.lock.stale*" \) -path "*/.cache/huggingface/download/*"
  } | sort -u
)

if [[ "${#CLEANUP_TARGETS[@]}" -eq 0 ]]; then
  echo "No incomplete model asset artifacts found under $MODELS_DIR"
  exit 0
fi

echo "Incomplete model asset artifacts:"
for target in "${CLEANUP_TARGETS[@]}"; do
  echo "- $(relative_path "$target")"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only. No files were removed."
  exit 0
fi

for target in "${CLEANUP_TARGETS[@]}"; do
  rm -f "$target"
done

# Prune empty helper directories left behind by marker files or cache cleanup.
find "$MODELS_DIR" -depth -type d \( -name ".asset-partials" -o -path "*/.cache/huggingface/download" -o -path "*/.cache/huggingface" -o -name ".cache" \) -empty -delete

echo "Removed ${#CLEANUP_TARGETS[@]} incomplete model asset artifacts."
echo "You can now rerun ./scripts/setup-worker.sh or ./scripts/download-model-assets.sh"
