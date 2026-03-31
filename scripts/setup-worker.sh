#!/usr/bin/env bash
set -euo pipefail

report_error() {
  local status="$?"
  local line_no="$1"
  local command_text="$2"
  # Surface the exact failing command so wrapper targets do not collapse into
  # a generic "make: Error 1" without enough context to debug.
  echo "setup-worker.sh failed at line $line_no while running: $command_text" >&2
  exit "$status"
}

trap 'report_error $LINENO "$BASH_COMMAND"' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_PATH="$ROOT_DIR/data/bootstrap-status.json"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_BACKEND="${CUDA_BACKEND:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$CUDA_BACKEND}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
ENV_STRATEGY="${ENV_STRATEGY:-split}"
SHARED_ENV_NAME="${SHARED_ENV_NAME:-effecterase-worker}"
SAM_ENV_NAME="${SAM_ENV_NAME:-effecterase-sam}"
REMOVE_ENV_NAME="${REMOVE_ENV_NAME:-effecterase-remove}"
WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
WORKER_PORT="${WORKER_PORT:-8000}"
SAM3_PACKAGE_SPEC="${SAM3_PACKAGE_SPEC:-https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip}"
SAM2_PACKAGE_SPEC="${SAM2_PACKAGE_SPEC:-https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip}"
EFFECTERASE_PACKAGE_SPEC="${EFFECTERASE_PACKAGE_SPEC:-https://github.com/FudanCVL/EffectErase/archive/refs/heads/main.zip}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"

mkdir -p "$ROOT_DIR/data/projects"

usage() {
  echo "Usage: $0 [--env-manager conda|micromamba|auto] [--strategy split|shared-first|shared] [--skip-model-downloads]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-manager)
      ENV_MANAGER="$2"
      shift 2
      ;;
    --strategy)
      ENV_STRATEGY="$2"
      shift 2
      ;;
    --cuda-backend)
      CUDA_BACKEND="$2"
      TORCH_INDEX_URL="https://download.pytorch.org/whl/$CUDA_BACKEND"
      shift 2
      ;;
    --skip-model-downloads)
      DOWNLOAD_MODELS=0
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

ensure_micromamba() {
  if command -v micromamba >/dev/null 2>&1; then
    return
  fi

  mkdir -p "$HOME/.local/bin"
  local archive
  archive="$(mktemp)"
  curl -Ls "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "$archive"
  tar -xjf "$archive" -C "$HOME/.local/bin" --strip-components=1 "bin/micromamba"
  rm -f "$archive"
  export PATH="$HOME/.local/bin:$PATH"
}

detect_env_manager() {
  if [[ "$ENV_MANAGER" != "auto" ]]; then
    echo "$ENV_MANAGER"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return
  fi

  echo "micromamba"
}

MANAGER="$(detect_env_manager)"

if [[ "$MANAGER" == "micromamba" ]]; then
  ensure_micromamba
fi

# Track what this run actually did so repeat runs can report real status
# instead of replaying the same generic bootstrap message every time.
declare -a REUSED_ENVS=()
declare -a CREATED_ENVS=()
declare -a REPAIRED_ENVS=()
MODEL_STEP_STATE="skipped"
FALLBACK_NOTE=""
BOOTSTRAP_STATE_STATUS="ready"
BOOTSTRAP_STATE_ERROR=""

manager_run() {
  local env_name="$1"
  shift
  if [[ "$MANAGER" == "conda" ]]; then
    conda run --no-capture-output -n "$env_name" "$@"
  else
    micromamba run -n "$env_name" "$@"
  fi
}

env_exists() {
  local env_name="$1"
  if [[ "$MANAGER" == "conda" ]]; then
    conda env list | awk '{print $1}' | grep -qx "$env_name"
  else
    micromamba env list | awk '{print $1}' | grep -qx "$env_name"
  fi
}

create_env() {
  local env_name="$1"
  if env_exists "$env_name"; then
    return
  fi

  if [[ "$MANAGER" == "conda" ]]; then
    conda create -y -n "$env_name" "python=$PYTHON_VERSION" pip git ffmpeg
  else
    micromamba create -y -n "$env_name" "python=$PYTHON_VERSION" pip git ffmpeg
  fi
}

validate_env() {
  local env_name="$1"
  shift
  if ! env_exists "$env_name"; then
    return 1
  fi
  manager_run "$env_name" python -c "$*"
}

# Keep per-run action tracking in one place so the final summary can show
# which envs were reused versus the ones that needed real work.
record_env_action() {
  local action="$1"
  local env_name="$2"

  case "$action" in
    reused)
      REUSED_ENVS+=("$env_name")
      ;;
    created)
      CREATED_ENVS+=("$env_name")
      ;;
    repaired)
      REPAIRED_ENVS+=("$env_name")
      ;;
  esac
}

log_env_status() {
  local env_name="$1"
  local action="$2"

  case "$action" in
    reused)
      echo "Env ready: $env_name (probe passed, reusing existing environment)"
      ;;
    created)
      echo "Env ready: $env_name (created environment and installed dependencies)"
      ;;
    repaired)
      echo "Env ready: $env_name (probe failed, reinstalled dependencies)"
      ;;
  esac
}

join_array() {
  local delimiter="$1"
  shift
  local item
  local result=""

  for item in "$@"; do
    if [[ -n "$result" ]]; then
      result+="$delimiter"
    fi
    result+="$item"
  done

  echo "$result"
}

print_run_summary() {
  local strategy="$1"
  local worker_env="$2"
  local env_names_text="$3"

  if [[ "${#CREATED_ENVS[@]}" -eq 0 && "${#REPAIRED_ENVS[@]}" -eq 0 ]]; then
    echo "Environment summary: already ready."
  else
    echo "Environment summary: setup work applied."
  fi

  echo "Strategy: $strategy (worker env: $worker_env)"
  echo "Envs: $env_names_text"

  if [[ "${#REUSED_ENVS[@]}" -gt 0 ]]; then
    echo "Reused envs: $(join_array ", " "${REUSED_ENVS[@]}")"
  fi

  if [[ "${#CREATED_ENVS[@]}" -gt 0 ]]; then
    echo "Created envs: $(join_array ", " "${CREATED_ENVS[@]}")"
  fi

  if [[ "${#REPAIRED_ENVS[@]}" -gt 0 ]]; then
    echo "Repaired envs: $(join_array ", " "${REPAIRED_ENVS[@]}")"
  fi

  if [[ -n "$FALLBACK_NOTE" ]]; then
    echo "$FALLBACK_NOTE"
  fi

  if [[ "$MODEL_STEP_STATE" == "checked" ]]; then
    echo "Model assets: checked by download-model-assets.sh"
  else
    echo "Model assets: skipped (--skip-model-downloads)"
  fi
}

install_common_worker_deps() {
  local env_name="$1"
  manager_run "$env_name" python -m pip install --upgrade pip "setuptools<82" wheel
  manager_run "$env_name" python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
  manager_run "$env_name" python -m pip install --no-build-isolation -e "$ROOT_DIR/worker"
}

install_sam3_package() {
  local env_name="$1"
  manager_run "$env_name" python -m pip install "$SAM3_PACKAGE_SPEC"
  manager_run "$env_name" python -m pip install \
    "einops>=0.8.0" \
    "psutil>=7.0.0" \
    "pycocotools>=2.0.8"
}

install_effecterase_shared_deps() {
  local env_name="$1"
  manager_run "$env_name" python -m pip uninstall -y opencv-python
  manager_run "$env_name" python -m pip install \
    --force-reinstall \
    "modelscope>=1.28.0" \
    "numpy<2.0.0" \
    "opencv-python-headless<4.12.0.0" \
    "transformers>=4.46.2,<5"
}

install_effecterase_remove_deps() {
  local env_name="$1"
  manager_run "$env_name" python -m pip uninstall -y opencv-python
  manager_run "$env_name" python -m pip install \
    --force-reinstall \
    "modelscope>=1.28.0" \
    "numpy<2.0.0" \
    "opencv-python-headless<4.12.0.0" \
    "transformers>=4.46.2,<5"
}

install_sam2_package() {
  local env_name="$1"
  manager_run "$env_name" python -m pip install "$SAM2_PACKAGE_SPEC"
}

install_shared_env_packages() {
  install_common_worker_deps "$SHARED_ENV_NAME"
  install_sam3_package "$SHARED_ENV_NAME"
  install_sam2_package "$SHARED_ENV_NAME"
  manager_run "$SHARED_ENV_NAME" python -m pip install --no-build-isolation "$EFFECTERASE_PACKAGE_SPEC"
  install_effecterase_shared_deps "$SHARED_ENV_NAME"
}

install_split_sam_env_packages() {
  install_common_worker_deps "$SAM_ENV_NAME"
  install_sam3_package "$SAM_ENV_NAME"
  install_sam2_package "$SAM_ENV_NAME"
}

install_split_remove_env_packages() {
  install_common_worker_deps "$REMOVE_ENV_NAME"
  manager_run "$REMOVE_ENV_NAME" python -m pip install --no-build-isolation "$EFFECTERASE_PACKAGE_SPEC"
  install_effecterase_remove_deps "$REMOVE_ENV_NAME"
}

ensure_env_ready() {
  local env_name="$1"
  local probe="$2"
  local install_fn="$3"
  local env_preexisted=0
  local action=""

  if env_exists "$env_name"; then
    env_preexisted=1
  else
    create_env "$env_name"
  fi

  if validate_env "$env_name" "$probe"; then
    record_env_action "reused" "$env_name"
    log_env_status "$env_name" "reused"
    return 0
  fi

  "$install_fn"
  validate_env "$env_name" "$probe"

  if [[ "$env_preexisted" == "1" ]]; then
    action="repaired"
  else
    action="created"
  fi

  record_env_action "$action" "$env_name"
  log_env_status "$env_name" "$action"
}

ensure_shared_env() {
  ensure_env_ready "$SHARED_ENV_NAME" "$shared_probe" install_shared_env_packages
}

ensure_split_envs() {
  ensure_env_ready "$SAM_ENV_NAME" "$sam_probe" install_split_sam_env_packages
  ensure_env_ready "$REMOVE_ENV_NAME" "$remove_probe" install_split_remove_env_packages
}

download_model_assets() {
  if [[ "$DOWNLOAD_MODELS" != "1" ]]; then
    MODEL_STEP_STATE="skipped"
    return
  fi

  # The download script owns the per-file detail; this script only reports
  # whether the model asset check/download step ran during this bootstrap.
  MODEL_STEP_STATE="checked"
  "$ROOT_DIR/scripts/download-model-assets.sh"
}

verify_bootstrap() {
  local strategy="$1"
  local worker_env="$2"
  local verify_args=(--bootstrap-mode --env-manager "$MANAGER" --strategy "$strategy" --worker-env "$worker_env")

  # Pass the resolved env names directly so setup-time verification does not
  # depend on a state file that has not been written yet.
  if [[ "$strategy" == "split" ]]; then
    verify_args+=(--sam-env "$SAM_ENV_NAME" --remove-env "$REMOVE_ENV_NAME")
  fi

  # `--skip-model-downloads` is the manual/staged provisioning path. Treat
  # missing checkpoints as a warning here so env bootstrap can finish cleanly.
  if [[ "$DOWNLOAD_MODELS" != "1" ]]; then
    verify_args+=(--allow-missing-model-assets)
  fi

  "$ROOT_DIR/scripts/verify-worker.sh" "${verify_args[@]}"
}

extract_json_bool() {
  local json_input="$1"
  local key="$2"
  local value
  value="$(printf '%s\n' "$json_input" | grep -o "\"$key\":[^,}]*" | head -n1 | cut -d: -f2 | tr -d '[:space:]')"

  if [[ "$value" == "true" ]]; then
    echo "true"
    return
  fi

  echo "false"
}

refresh_bootstrap_state() {
  local strategy="$1"
  local worker_env="$2"
  local verify_args=(--json --bootstrap-mode --env-manager "$MANAGER" --strategy "$strategy" --worker-env "$worker_env")

  BOOTSTRAP_STATE_STATUS="ready"
  BOOTSTRAP_STATE_ERROR=""

  # A staged bootstrap can finish before the operator has provisioned model
  # assets, but the persisted status must stay non-ready until inference assets
  # are actually present on disk.
  if [[ "$DOWNLOAD_MODELS" != "1" ]]; then
    verify_args+=(--allow-missing-model-assets)
  else
    # An explicit success return is required here because `set -e` would treat
    # the branch condition's false status as a failure if we used bare `return`.
    return 0
  fi

  if [[ "$strategy" == "split" ]]; then
    verify_args+=(--sam-env "$SAM_ENV_NAME" --remove-env "$REMOVE_ENV_NAME")
  fi

  local verify_report
  verify_report="$("$ROOT_DIR/scripts/verify-worker.sh" "${verify_args[@]}")"

  if [[ "$(extract_json_bool "$verify_report" "modelAssetsOk")" != "true" ]]; then
    BOOTSTRAP_STATE_STATUS="incomplete"
    BOOTSTRAP_STATE_ERROR="Bootstrap completed without required model assets. Provision the model files and rerun ./scripts/verify-worker.sh."
  fi
}

write_state() {
  local strategy="$1"
  local worker_env="$2"
  local env_names_json="$3"
  local now_utc
  local error_json="null"
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  if [[ -n "$BOOTSTRAP_STATE_ERROR" ]]; then
    error_json="\"$BOOTSTRAP_STATE_ERROR\""
  fi

  cat > "$STATE_PATH" <<EOF
{"status":"$BOOTSTRAP_STATE_STATUS","envManager":"$MANAGER","envNames":$env_names_json,"activeStrategy":"$strategy","workerEnvName":"$worker_env","samEnvName":"$SAM_ENV_NAME","removeEnvName":"$REMOVE_ENV_NAME","pythonVersion":"$PYTHON_VERSION","cudaBackend":"$CUDA_BACKEND","workerHost":"$WORKER_HOST","workerPort":"$WORKER_PORT","lastValidatedAt":"$now_utc","error":$error_json}
EOF
}

shared_probe='import cv2, fastapi, torch, diffsynth, modelscope, sam2, sam3; import app.main, app.runners.effecterase_remove'
sam_probe='import fastapi, torch, sam2, sam3; import app.main'
remove_probe='import cv2, torch, diffsynth, modelscope; import app.runners.effecterase_remove'

if [[ "$ENV_STRATEGY" == "shared" || "$ENV_STRATEGY" == "shared-first" ]]; then
  if ensure_shared_env; then
    download_model_assets
    verify_bootstrap "shared" "$SHARED_ENV_NAME"
    refresh_bootstrap_state "shared" "$SHARED_ENV_NAME"
    write_state "shared" "$SHARED_ENV_NAME" "[\"$SHARED_ENV_NAME\"]"
    print_run_summary "shared" "$SHARED_ENV_NAME" "$SHARED_ENV_NAME"
    exit 0
  fi

  if [[ "$ENV_STRATEGY" == "shared" ]]; then
    echo "Shared environment setup failed." >&2
    exit 1
  fi

  FALLBACK_NOTE="Fallback: shared env setup failed, switching to split envs."
fi

ensure_split_envs
download_model_assets
verify_bootstrap "split" "$SAM_ENV_NAME"
refresh_bootstrap_state "split" "$SAM_ENV_NAME"
write_state "split" "$SAM_ENV_NAME" "[\"$SAM_ENV_NAME\",\"$REMOVE_ENV_NAME\"]"
print_run_summary "split" "$SAM_ENV_NAME" "$SAM_ENV_NAME, $REMOVE_ENV_NAME"
