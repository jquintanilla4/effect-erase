#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_PATH="$ROOT_DIR/data/bootstrap-status.json"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_BACKEND="${CUDA_BACKEND:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$CUDA_BACKEND}"
ENV_MANAGER="${ENV_MANAGER:-auto}"
ENV_STRATEGY="${ENV_STRATEGY:-shared-first}"
SHARED_ENV_NAME="${SHARED_ENV_NAME:-effecterase-worker}"
SAM_ENV_NAME="${SAM_ENV_NAME:-effecterase-sam}"
REMOVE_ENV_NAME="${REMOVE_ENV_NAME:-effecterase-remove}"
WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
WORKER_PORT="${WORKER_PORT:-8000}"
SAM3_REPO_URL="${SAM3_REPO_URL:-https://github.com/facebookresearch/sam3.git}"
EFFECTERASE_REPO_URL="${EFFECTERASE_REPO_URL:-https://github.com/FudanCVL/EffectErase.git}"

mkdir -p "$ROOT_DIR/data/projects" "$THIRD_PARTY_DIR"

usage() {
  echo "Usage: $0 [--env-manager conda|micromamba|auto] [--strategy shared-first|shared|split]"
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

ensure_repo() {
  local url="$1"
  local target="$2"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" pull --ff-only || true
    return
  fi
  git clone "$url" "$target"
}

validate_env() {
  local env_name="$1"
  shift
  if ! env_exists "$env_name"; then
    return 1
  fi
  manager_run "$env_name" python -c "$*"
}

install_common_worker_deps() {
  local env_name="$1"
  manager_run "$env_name" python -m pip install --upgrade pip setuptools wheel
  manager_run "$env_name" python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
  manager_run "$env_name" python -m pip install -e "$ROOT_DIR/worker"
}

install_shared_env() {
  create_env "$SHARED_ENV_NAME"
  install_common_worker_deps "$SHARED_ENV_NAME"
  ensure_repo "$SAM3_REPO_URL" "$THIRD_PARTY_DIR/sam3"
  ensure_repo "$EFFECTERASE_REPO_URL" "$THIRD_PARTY_DIR/EffectErase"
  manager_run "$SHARED_ENV_NAME" python -m pip install -e "$THIRD_PARTY_DIR/sam3"
  manager_run "$SHARED_ENV_NAME" python -m pip install -e "$THIRD_PARTY_DIR/EffectErase"
}

install_split_envs() {
  create_env "$SAM_ENV_NAME"
  create_env "$REMOVE_ENV_NAME"
  install_common_worker_deps "$SAM_ENV_NAME"
  install_common_worker_deps "$REMOVE_ENV_NAME"
  ensure_repo "$SAM3_REPO_URL" "$THIRD_PARTY_DIR/sam3"
  ensure_repo "$EFFECTERASE_REPO_URL" "$THIRD_PARTY_DIR/EffectErase"
  manager_run "$SAM_ENV_NAME" python -m pip install -e "$THIRD_PARTY_DIR/sam3"
  manager_run "$REMOVE_ENV_NAME" python -m pip install -e "$THIRD_PARTY_DIR/EffectErase"
}

write_state() {
  local strategy="$1"
  local worker_env="$2"
  local env_names_json="$3"
  local now_utc
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  cat > "$STATE_PATH" <<EOF
{"status":"ready","envManager":"$MANAGER","envNames":$env_names_json,"activeStrategy":"$strategy","workerEnvName":"$worker_env","samEnvName":"$SAM_ENV_NAME","removeEnvName":"$REMOVE_ENV_NAME","pythonVersion":"$PYTHON_VERSION","cudaBackend":"$CUDA_BACKEND","workerHost":"$WORKER_HOST","workerPort":"$WORKER_PORT","lastValidatedAt":"$now_utc","error":null}
EOF
}

shared_probe='import fastapi, torch, diffsynth; import app.main'
sam_probe='import fastapi, torch; import app.main'
remove_probe='import torch, diffsynth'

if [[ "$ENV_STRATEGY" == "shared" || "$ENV_STRATEGY" == "shared-first" ]]; then
  if install_shared_env && validate_env "$SHARED_ENV_NAME" "$shared_probe"; then
    write_state "shared" "$SHARED_ENV_NAME" "[\"$SHARED_ENV_NAME\"]"
    echo "Worker bootstrap complete with shared env: $SHARED_ENV_NAME"
    exit 0
  fi

  if [[ "$ENV_STRATEGY" == "shared" ]]; then
    echo "Shared environment setup failed." >&2
    exit 1
  fi
fi

install_split_envs
validate_env "$SAM_ENV_NAME" "$sam_probe"
validate_env "$REMOVE_ENV_NAME" "$remove_probe"
write_state "split" "$SAM_ENV_NAME" "[\"$SAM_ENV_NAME\",\"$REMOVE_ENV_NAME\"]"
echo "Worker bootstrap complete with split envs: $SAM_ENV_NAME, $REMOVE_ENV_NAME"
