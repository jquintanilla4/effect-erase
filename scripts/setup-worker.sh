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
source "$ROOT_DIR/scripts/lib/runtime-config.sh"

if [[ -d "$HOME/.local/bin" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

# Runpod shells commonly run as the root Unix user. Pip's "running as root"
# warning is about the account, not whether we are inside the target conda/mamba
# environment, so silence it to keep bootstrap logs focused on real issues.
if [[ "${EUID:-$(id -u)}" == "0" && -z "${PIP_ROOT_USER_ACTION:-}" ]]; then
  export PIP_ROOT_USER_ACTION=ignore
fi

ENV_MANAGER_INPUT="${ENV_MANAGER:-}"
STORAGE_ROOT_INPUT="${STORAGE_ROOT:-}"
BOOTSTRAP_STATE_INPUT="${WORKER_BOOTSTRAP_STATE_PATH:-}"
HF_AUTH_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_BACKEND="${CUDA_BACKEND:-cu128}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$CUDA_BACKEND}"
ENV_MANAGER="${ENV_MANAGER_INPUT:-auto}"
STORAGE_ROOT="${STORAGE_ROOT_INPUT:-}"
WORKER_BOOTSTRAP_STATE_PATH="${BOOTSTRAP_STATE_INPUT:-}"
ENV_STRATEGY="${ENV_STRATEGY:-split}"
SAM_ENV_NAME="${SAM_ENV_NAME:-effecterase-sam}"
REMOVE_ENV_NAME="${REMOVE_ENV_NAME:-effecterase-remove}"
VOID_ENV_NAME="${VOID_ENV_NAME:-effecterase-void}"
WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
WORKER_PORT="${WORKER_PORT:-8000}"
# Pin direct upstream source installs so cold bootstrap behavior stays stable
# across reruns until we intentionally update these refs in-repo.
SAM3_PACKAGE_REF="${SAM3_PACKAGE_REF:-bfbed072a07a6a52c8d5fdc75a7a186251a835b1}"
SAM2_PACKAGE_REF="${SAM2_PACKAGE_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"
EFFECTERASE_PACKAGE_REF="${EFFECTERASE_PACKAGE_REF:-3dd007f6b2c60d13921c12c4a31051b32a530007}"
VOID_PACKAGE_REF="${VOID_PACKAGE_REF:-41adfdd71619df0c7834173c53d7f9518db5f247}"
FLASH_ATTENTION_HOPPER_REF="${FLASH_ATTENTION_HOPPER_REF:-83f9e450cd10e20701fb109db9c7703d376f282b}"
SAM3_PACKAGE_SPEC="${SAM3_PACKAGE_SPEC:-https://github.com/facebookresearch/sam3/archive/$SAM3_PACKAGE_REF.zip}"
SAM2_PACKAGE_SPEC="${SAM2_PACKAGE_SPEC:-https://github.com/facebookresearch/sam2/archive/$SAM2_PACKAGE_REF.zip}"
EFFECTERASE_PACKAGE_SPEC="${EFFECTERASE_PACKAGE_SPEC:-https://github.com/FudanCVL/EffectErase/archive/$EFFECTERASE_PACKAGE_REF.zip}"
VOID_REPO_URL="${VOID_REPO_URL:-https://github.com/Netflix/void-model.git}"
VOID_REPO_DIR="${VOID_REPO_DIR:-$ROOT_DIR/third_party/void-model}"
FLASH_ATTENTION_HOPPER_SPEC="${FLASH_ATTENTION_HOPPER_SPEC:-git+https://github.com/Dao-AILab/flash-attention.git@$FLASH_ATTENTION_HOPPER_REF#subdirectory=hopper}"
# Keep the underscored name as the documented option, but honor the older
# shell-history variant too so existing Runpod commands still skip the FA3 build.
SKIP_SAM_FA3="${SKIP_SAM_FA3:-${SKIPSAMFA3:-0}}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
INTERACTIVE_MODE="${BOOTSTRAP_INTERACTIVE:-auto}"
CLI_ENV_MANAGER_SET=0
CLI_STORAGE_ROOT_SET=0
STATE_HINT_PATH=""
STORAGE_ROOT_EXPLICIT=0
RUNTIME_ROOT_MANAGED=""
STATE_STORAGE_ROOT=""
STATE_ENV_MANAGER=""
STATE_HF_HOME=""

usage() {
  echo "Usage: $0 [--env-manager conda|micromamba|auto] [--storage-root PATH] [--interactive|--non-interactive] [--strategy split] [--skip-model-downloads]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-manager)
      ENV_MANAGER="$2"
      CLI_ENV_MANAGER_SET=1
      shift 2
      ;;
    --storage-root)
      STORAGE_ROOT="$2"
      STORAGE_ROOT_EXPLICIT=1
      CLI_STORAGE_ROOT_SET=1
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
    --interactive)
      INTERACTIVE_MODE="always"
      shift
      ;;
    --non-interactive)
      INTERACTIVE_MODE="never"
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

if [[ "$ENV_STRATEGY" != "split" ]]; then
  echo "Unsupported strategy '$ENV_STRATEGY'. setup-worker.sh now supports split environments only." >&2
  exit 1
fi

if [[ "$CLI_STORAGE_ROOT_SET" != "1" && -n "$STORAGE_ROOT_INPUT" ]]; then
  STORAGE_ROOT_EXPLICIT=1
fi

can_prompt() {
  if [[ "$INTERACTIVE_MODE" == "always" ]]; then
    return 0
  fi
  if [[ "$INTERACTIVE_MODE" == "never" ]]; then
    return 1
  fi
  [[ -t 0 && -t 1 ]]
}

prompt_value() {
  local label="$1"
  local default_value="${2:-}"
  local secret="${3:-0}"
  local prompt="$label"

  if [[ -n "$default_value" ]]; then
    prompt+=" [$default_value]"
  fi
  prompt+=": "

  local response=""
  if [[ "$secret" == "1" ]]; then
    read -r -s -p "$prompt" response
    echo
  else
    read -r -p "$prompt" response
  fi

  if [[ -z "$response" ]]; then
    response="$default_value"
  fi

  printf '%s\n' "$response"
}

select_env_manager() {
  local conda_available=0
  local micromamba_available=0
  local default_choice="conda"
  local selected=""

  if [[ "$ENV_MANAGER" != "auto" ]]; then
    echo "$ENV_MANAGER"
    return
  fi

  if command -v conda >/dev/null 2>&1; then
    conda_available=1
  fi
  if command -v micromamba >/dev/null 2>&1; then
    micromamba_available=1
  fi

  if [[ "$conda_available" == "1" && "$micromamba_available" == "0" ]]; then
    echo "conda"
    return
  fi
  if [[ "$conda_available" == "0" && "$micromamba_available" == "1" ]]; then
    echo "micromamba"
    return
  fi
  if [[ -n "$STATE_ENV_MANAGER" && "$STATE_ENV_MANAGER" != "auto" ]]; then
    echo "$STATE_ENV_MANAGER"
    return
  fi
  if [[ "$conda_available" == "1" && "$micromamba_available" == "1" ]] && is_runpod; then
    echo "micromamba"
    return
  fi

  if is_runpod; then
    default_choice="micromamba"
  fi

  if ! can_prompt; then
    if [[ "$conda_available" == "1" ]]; then
      echo "conda"
      return
    fi
    echo "micromamba"
    return
  fi

  while true; do
    selected="$(prompt_value "Python environment manager (conda/micromamba)" "$default_choice")"
    case "$selected" in
      conda|micromamba)
        echo "$selected"
        return
        ;;
    esac
    echo "Choose either 'conda' or 'micromamba'." >&2
  done
}

resolve_hf_home() {
  if [[ -n "$HF_HOME_PATH" ]]; then
    echo "$HF_HOME_PATH"
    return
  fi
  if [[ -n "$STATE_HF_HOME" ]]; then
    echo "$STATE_HF_HOME"
    return
  fi
  if [[ -n "${HF_HOME:-}" ]]; then
    echo "$HF_HOME"
    return
  fi
  if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
    echo "$XDG_CACHE_HOME/huggingface"
    return
  fi
  echo "$HOME/.cache/huggingface"
}

hf_auth_present() {
  local hf_home
  local token_path

  if [[ -n "$HF_AUTH_TOKEN" || -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    return 0
  fi

  hf_home="$(resolve_hf_home)"
  token_path="${HF_TOKEN_PATH:-$hf_home/token}"
  [[ -s "$token_path" ]]
}

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

ensure_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    return
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "The Hugging Face CLI ('hf') is required and python3 is not available to install it." >&2
    exit 1
  fi

  echo "Installing Hugging Face CLI..." >&2
  python3 -m pip install --user "huggingface_hub"
  export PATH="$HOME/.local/bin:$PATH"

  if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI ('hf') is still unavailable after installation." >&2
    exit 1
  fi
}

prompt_storage_root_if_needed() {
  local default_root

  if [[ -n "$STORAGE_ROOT" ]]; then
    return
  fi

  if [[ -n "$STATE_STORAGE_ROOT" ]]; then
    STORAGE_ROOT="$STATE_STORAGE_ROOT"
    return
  fi

  default_root="$(default_storage_root)"
  STORAGE_ROOT="$default_root"
}

hf_auth_guidance() {
  echo "Hugging Face auth is required for default bootstrap because facebook/sam3.1 is gated." >&2
  echo "Run 'hf auth login' first or provide HF_TOKEN / HUGGING_FACE_HUB_TOKEN." >&2
}

require_hf_auth_for_bootstrap() {
  if [[ "$DOWNLOAD_MODELS" != "1" ]]; then
    return
  fi
  if hf_auth_present; then
    return
  fi

  if can_prompt; then
    HF_AUTH_TOKEN="$(prompt_value "Hugging Face token for gated SAM 3.1 downloads" "" 1)"
    if [[ -z "$HF_AUTH_TOKEN" ]]; then
      hf_auth_guidance
      exit 1
    fi

    ensure_hf_cli
    hf auth login --token "$HF_AUTH_TOKEN"
    unset HF_AUTH_TOKEN

    if hf_auth_present; then
      return
    fi

    echo "Hugging Face auth login did not produce a usable token at ${HF_TOKEN_PATH:-$(resolve_hf_home)/token}." >&2
    hf_auth_guidance
    exit 1
  fi

  hf_auth_guidance
  exit 1
}

STATE_HINT_PATH="$(locate_bootstrap_state)"
if [[ -f "$STATE_HINT_PATH" ]]; then
  STATE_STORAGE_ROOT="$(state_field "$STATE_HINT_PATH" "storageRoot" || true)"
  STATE_ENV_MANAGER="$(state_field "$STATE_HINT_PATH" "envManager" || true)"
  STATE_HF_HOME="$(state_field "$STATE_HINT_PATH" "hfHome" || true)"
fi

prompt_storage_root_if_needed
resolve_runtime_layout "$STORAGE_ROOT"
export_runtime_layout
ensure_runtime_dirs

require_hf_auth_for_bootstrap

if [[ "$ENV_MANAGER" == "auto" && -n "$STATE_ENV_MANAGER" && "$STATE_ENV_MANAGER" != "auto" && "$CLI_ENV_MANAGER_SET" != "1" && -z "$ENV_MANAGER_INPUT" ]]; then
  ENV_MANAGER="$STATE_ENV_MANAGER"
fi

MANAGER="$(select_env_manager)"

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
SAM_FA3_STATUS="unknown"
SAM_FA3_NOTE=""
SAM_FA3_ENV_NAME=""

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

remove_env() {
  local env_name="$1"
  if ! env_exists "$env_name"; then
    return
  fi

  if [[ "$MANAGER" == "conda" ]]; then
    conda remove -y -n "$env_name" --all
  else
    micromamba env remove -y -n "$env_name"
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
      echo "Env ready: $env_name (probe failed, recreated environment and reinstalled dependencies)"
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

append_fallback_note() {
  local note="$1"
  if [[ -n "$FALLBACK_NOTE" ]]; then
    FALLBACK_NOTE+=$'\n'
  fi
  FALLBACK_NOTE+="$note"
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
  echo "Runtime root: $STORAGE_ROOT"
  echo "Models dir: $MODELS_DIR"
  echo "Bootstrap state: $STATE_PATH"

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

  if [[ -n "$SAM_FA3_STATUS" && "$SAM_FA3_STATUS" != "unknown" ]]; then
    echo "SAM FA3: $SAM_FA3_STATUS"
  fi

  if [[ -n "$SAM_FA3_NOTE" ]]; then
    echo "SAM FA3 note: $SAM_FA3_NOTE"
  fi

  if [[ "$MODEL_STEP_STATE" == "checked" ]]; then
    echo "Model assets: checked by download-model-assets.sh"
  else
    echo "Model assets: skipped (--skip-model-downloads)"
  fi
}

install_common_worker_deps() {
  local env_name="$1"
  echo "[$env_name] Installing shared worker dependencies..."
  # Newer preview generation depends on ffmpeg even when the env predated this
  # requirement, so reinstall/bootstrap runs must enforce the binary explicitly
  # instead of only relying on the original conda create template.
  if [[ "$MANAGER" == "conda" ]]; then
    conda install -y -n "$env_name" ffmpeg
  else
    micromamba install -y -n "$env_name" ffmpeg
  fi
  manager_run "$env_name" python -m pip install --upgrade pip "setuptools<82" wheel
  manager_run "$env_name" python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
  manager_run "$env_name" python -m pip install --no-build-isolation -e "$ROOT_DIR/worker"
  echo "[$env_name] Normalizing NumPy + headless OpenCV..."
  # Normalize back to the headless OpenCV stack even on repaired envs so any
  # transitive GUI OpenCV install does not leak into the worker runtime.
  manager_run "$env_name" python -m pip uninstall -y opencv-python
  manager_run "$env_name" python -m pip install --force-reinstall "numpy<2.0.0" "opencv-python-headless<4.12.0.0"
}

ensure_void_repo_checkout() {
  mkdir -p "$(dirname "$VOID_REPO_DIR")"

  if [[ -d "$VOID_REPO_DIR/.git" ]]; then
    local current_ref
    current_ref="$(git -C "$VOID_REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$current_ref" == "$VOID_PACKAGE_REF" ]]; then
      return 0
    fi
    git -C "$VOID_REPO_DIR" fetch --depth 1 origin "$VOID_PACKAGE_REF"
  else
    rm -rf "$VOID_REPO_DIR"
    git clone --depth 1 "$VOID_REPO_URL" "$VOID_REPO_DIR"
  fi

  git -C "$VOID_REPO_DIR" checkout --detach "$VOID_PACKAGE_REF"
}

install_sam3_package() {
  local env_name="$1"
  echo "[$env_name] Installing SAM 3..."
  manager_run "$env_name" python -m pip install "$SAM3_PACKAGE_SPEC"
  manager_run "$env_name" python -m pip install \
    "einops>=0.8.0" \
    "psutil>=7.0.0" \
    "pycocotools>=2.0.8"
}

sam_fa3_gpu_info() {
  local env_name="$1"
  manager_run "$env_name" python - <<'PY'
import json

try:
    import torch
except Exception as error:
    print(json.dumps({
        "cuda_available": False,
        "major": None,
        "minor": None,
        "name": None,
        "error": f"{type(error).__name__}: {error}",
    }))
    raise SystemExit(0)

if not torch.cuda.is_available():
    print(json.dumps({
        "cuda_available": False,
        "major": None,
        "minor": None,
        "name": None,
        "error": None,
    }))
    raise SystemExit(0)

props = torch.cuda.get_device_properties(0)
print(json.dumps({
    "cuda_available": True,
    "major": int(props.major),
    "minor": int(props.minor),
    "name": props.name,
    "error": None,
}))
PY
}

configure_sam_fa3() {
  local env_name="$1"
  if [[ -n "$SAM_FA3_ENV_NAME" && "$SAM_FA3_ENV_NAME" == "$env_name" && "$SAM_FA3_STATUS" != "unknown" ]]; then
    return 0
  fi
  SAM_FA3_ENV_NAME="$env_name"

  if [[ "$SKIP_SAM_FA3" == "1" ]]; then
    SAM_FA3_STATUS="disabled"
    SAM_FA3_NOTE="Skipping FlashAttention 3 because SKIP_SAM_FA3=1; sam3.1 will continue with use_fa3=false."
    return 0
  fi

  local gpu_info
  gpu_info="$(sam_fa3_gpu_info "$env_name")"

  local cuda_available
  cuda_available="$(printf '%s\n' "$gpu_info" | python3 -c 'import json,sys; print("true" if json.load(sys.stdin)["cuda_available"] else "false")')"
  local gpu_major
  gpu_major="$(printf '%s\n' "$gpu_info" | python3 -c 'import json,sys; value=json.load(sys.stdin)["major"]; print("" if value is None else value)')"
  local gpu_minor
  gpu_minor="$(printf '%s\n' "$gpu_info" | python3 -c 'import json,sys; value=json.load(sys.stdin)["minor"]; print("" if value is None else value)')"
  local gpu_name
  gpu_name="$(printf '%s\n' "$gpu_info" | python3 -c 'import json,sys; value=json.load(sys.stdin)["name"]; print("" if value is None else value)')"
  local gpu_error
  gpu_error="$(printf '%s\n' "$gpu_info" | python3 -c 'import json,sys; value=json.load(sys.stdin)["error"]; print("" if value is None else value)')"

  if [[ "$cuda_available" != "true" ]]; then
    SAM_FA3_STATUS="disabled"
    if [[ -n "$gpu_error" ]]; then
      SAM_FA3_NOTE="Skipping FlashAttention 3 because CUDA detection failed: $gpu_error"
    else
      SAM_FA3_NOTE="Skipping FlashAttention 3 because CUDA is not available during bootstrap."
    fi
    return 0
  fi

  if [[ -z "$gpu_major" || "$gpu_major" -lt 9 ]]; then
    SAM_FA3_STATUS="disabled"
    SAM_FA3_NOTE="Skipping FlashAttention 3 on ${gpu_name:-unknown GPU} (compute capability ${gpu_major:-?}.${gpu_minor:-?}); sam3.1 will use use_fa3=false."
    return 0
  fi

  # Hopper-or-newer GPUs are the only ones where we try the upstream FA3 build.
  if manager_run "$env_name" python -m pip install "packaging" "ninja"; then
    if manager_run "$env_name" python -m pip install --no-build-isolation "$FLASH_ATTENTION_HOPPER_SPEC"; then
      SAM_FA3_STATUS="enabled"
      SAM_FA3_NOTE="Installed FlashAttention 3 for ${gpu_name} (compute capability ${gpu_major}.${gpu_minor})."
      return 0
    fi
  fi

  SAM_FA3_STATUS="unavailable"
  SAM_FA3_NOTE="FlashAttention 3 install failed on ${gpu_name} (compute capability ${gpu_major}.${gpu_minor}); sam3.1 will continue with use_fa3=false."
  return 0
}

install_effecterase_remove_deps() {
  local env_name="$1"
  echo "[$env_name] Installing EffectErase runtime dependencies..."
  manager_run "$env_name" python -m pip uninstall -y opencv-python
  # Keep the CUDA torch stack that install_common_worker_deps already pinned to
  # the PyTorch wheel index. A blanket force-reinstall here sends pip back
  # through dependency resolution against the default index, which redownloads
  # torch and its CUDA runtime again and dramatically increases bootstrap time
  # and disk pressure on Runpod.
  manager_run "$env_name" python -m pip install \
    "accelerate>=0.25.0" \
    "albumentations" \
    "beautifulsoup4" \
    "datasets>=4.8.4,<5" \
    "decord" \
    "diffusers>=0.30.1,<=0.31.0" \
    "einops>=0.8.0" \
    "fsspec>=2023.1.0,<=2026.2.0" \
    "ftfy" \
    "func_timeout" \
    "imageio[ffmpeg]" \
    "imageio[pyav]" \
    "modelscope>=1.28.0" \
    "numpy<2.0.0" \
    "opencv-python-headless<4.12.0.0" \
    "omegaconf" \
    "Pillow" \
    "safetensors" \
    "scikit-image" \
    "sentencepiece" \
    "setuptools<82" \
    "tensorboard" \
    "timm" \
    "tomesd" \
    "torchdiffeq" \
    "torchsde" \
    "transformers>=4.46.2,<5"
}

install_void_runtime_deps() {
  local env_name="$1"
  echo "[$env_name] Installing VOID runtime dependencies..."
  ensure_void_repo_checkout
  manager_run "$env_name" python -m pip uninstall -y opencv-python
  manager_run "$env_name" python -m pip install \
    "absl-py" \
    "accelerate>=1.12.0" \
    "diffusers==0.33.1" \
    "einops==0.8.0" \
    "func-timeout==4.3.5" \
    "huggingface_hub>=0.35.0" \
    "imageio==2.37.0" \
    "imageio-ffmpeg==0.6.0" \
    "kornia==0.8.1" \
    "loguru==0.7.3" \
    "mediapy==1.2.4" \
    "ml_collections==1.1.0" \
    "numpy==1.26.4" \
    "omegaconf==2.3.0" \
    "opencv-python-headless==4.10.0.84" \
    "peft==0.17.1" \
    "Pillow==11.3.0" \
    "safetensors==0.6.2" \
    "scikit-image==0.25.2" \
    "sentencepiece==0.2.1" \
    "timm==1.0.19" \
    "tomesd==0.1.3" \
    "torchdiffeq==0.2.5" \
    "torchsde==0.2.6" \
    "transformers==4.57.1"
}

install_sam2_package() {
  local env_name="$1"
  echo "[$env_name] Installing SAM 2..."
  # Upstream SAM 2 declares torch as a build dependency, so build isolation can
  # trigger a second heavyweight torch install in a temp env and stall bootstrap.
  # Reuse the env's existing torch install and skip the optional CUDA extension.
  manager_run "$env_name" env SAM2_BUILD_CUDA=0 python -m pip install -v --no-build-isolation "$SAM2_PACKAGE_SPEC"
}

install_split_sam_env_delta() {
  install_sam3_package "$SAM_ENV_NAME"
  configure_sam_fa3 "$SAM_ENV_NAME"
  install_sam2_package "$SAM_ENV_NAME"
}

install_split_remove_env_delta() {
  echo "[$REMOVE_ENV_NAME] Installing EffectErase package..."
  install_effecterase_remove_deps "$REMOVE_ENV_NAME"
  manager_run "$REMOVE_ENV_NAME" python -m pip install --no-build-isolation --no-deps "$EFFECTERASE_PACKAGE_SPEC"
}

install_split_void_env_packages() {
  install_common_worker_deps "$VOID_ENV_NAME"
  install_void_runtime_deps "$VOID_ENV_NAME"
}

install_split_sam_env_packages() {
  install_common_worker_deps "$SAM_ENV_NAME"
  install_split_sam_env_delta
}

install_split_remove_env_packages() {
  install_common_worker_deps "$REMOVE_ENV_NAME"
  install_split_remove_env_delta
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

  # A brand-new env only contains python/pip/git/ffmpeg, so the first probe is
  # expected to fail before dependency installation. Keep that probe quiet so
  # bootstrap output does not look like a system-Python failure on Runpod.
  if validate_env "$env_name" "$probe" >/dev/null 2>&1; then
    record_env_action "reused" "$env_name"
    log_env_status "$env_name" "reused"
    return 0
  fi

  if [[ "$env_preexisted" == "1" ]]; then
    remove_env "$env_name"
    create_env "$env_name"
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

ensure_split_envs_direct() {
  ensure_env_ready "$SAM_ENV_NAME" "$sam_probe" install_split_sam_env_packages
  configure_sam_fa3 "$SAM_ENV_NAME"
  ensure_env_ready "$REMOVE_ENV_NAME" "$remove_probe" install_split_remove_env_packages
  ensure_env_ready "$VOID_ENV_NAME" "$void_probe" install_split_void_env_packages
}

ensure_split_envs() {
  local sam_ready=0
  local remove_ready=0
  local void_ready=0

  if validate_env "$SAM_ENV_NAME" "$sam_probe" >/dev/null 2>&1; then
    sam_ready=1
  fi
  if validate_env "$REMOVE_ENV_NAME" "$remove_probe" >/dev/null 2>&1; then
    remove_ready=1
  fi
  if validate_env "$VOID_ENV_NAME" "$void_probe" >/dev/null 2>&1; then
    void_ready=1
  fi

  if [[ "$sam_ready" != "1" && "$remove_ready" != "1" && "$void_ready" != "1" ]]; then
    ensure_split_envs_direct
    return 0
  fi

  if [[ "$sam_ready" == "1" ]]; then
    record_env_action "reused" "$SAM_ENV_NAME"
    log_env_status "$SAM_ENV_NAME" "reused"
  else
    ensure_env_ready "$SAM_ENV_NAME" "$sam_probe" install_split_sam_env_packages
  fi
  configure_sam_fa3 "$SAM_ENV_NAME"

  if [[ "$remove_ready" == "1" ]]; then
    record_env_action "reused" "$REMOVE_ENV_NAME"
    log_env_status "$REMOVE_ENV_NAME" "reused"
  else
    ensure_env_ready "$REMOVE_ENV_NAME" "$remove_probe" install_split_remove_env_packages
  fi

  if [[ "$void_ready" == "1" ]]; then
    record_env_action "reused" "$VOID_ENV_NAME"
    log_env_status "$VOID_ENV_NAME" "reused"
  else
    ensure_env_ready "$VOID_ENV_NAME" "$void_probe" install_split_void_env_packages
  fi
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
    verify_args+=(--sam-env "$SAM_ENV_NAME" --remove-env "$REMOVE_ENV_NAME" --void-env "$VOID_ENV_NAME")
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
    verify_args+=(--sam-env "$SAM_ENV_NAME" --remove-env "$REMOVE_ENV_NAME" --void-env "$VOID_ENV_NAME")
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
  now_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  BOOTSTRAP_STATE_STATUS="$BOOTSTRAP_STATE_STATUS" \
  MANAGER="$MANAGER" \
  ENV_NAMES_JSON="$env_names_json" \
  BOOTSTRAP_STRATEGY="$strategy" \
  BOOTSTRAP_WORKER_ENV="$worker_env" \
  SAM_ENV_NAME="$SAM_ENV_NAME" \
  REMOVE_ENV_NAME="$REMOVE_ENV_NAME" \
  VOID_ENV_NAME="$VOID_ENV_NAME" \
  PYTHON_VERSION="$PYTHON_VERSION" \
  CUDA_BACKEND="$CUDA_BACKEND" \
  SAM_FA3_STATUS="$SAM_FA3_STATUS" \
  SAM_FA3_NOTE="$SAM_FA3_NOTE" \
  WORKER_HOST="$WORKER_HOST" \
  WORKER_PORT="$WORKER_PORT" \
  LAST_VALIDATED_AT="$now_utc" \
  BOOTSTRAP_STATE_ERROR="$BOOTSTRAP_STATE_ERROR" \
  STORAGE_ROOT="$STORAGE_ROOT" \
  DATA_DIR="$DATA_DIR" \
  PROJECTS_DIR="$PROJECTS_DIR" \
  MODELS_DIR="$MODELS_DIR" \
  STATE_PATH="$STATE_PATH" \
  HF_HOME_PATH="$HF_HOME_PATH" \
  HF_HUB_CACHE_PATH="$HF_HUB_CACHE_PATH" \
  PIP_CACHE_PATH="$PIP_CACHE_PATH" \
  MAMBA_ROOT_PREFIX_PATH="$MAMBA_ROOT_PREFIX_PATH" \
  CONDA_ENVS_PATH_VALUE="$CONDA_ENVS_PATH_VALUE" \
  CONDA_PKGS_DIRS_VALUE="$CONDA_PKGS_DIRS_VALUE" \
  TMPDIR_PATH="$TMPDIR_PATH" \
  python3 - "$STATE_PATH" <<'PY'
import json
import os
import sys

state_path = sys.argv[1]
env_names = json.loads(os.environ["ENV_NAMES_JSON"])

def value(name: str):
    raw = os.environ.get(name, "")
    return raw or None

sam_fa3_status = value("SAM_FA3_STATUS")
if sam_fa3_status == "unknown":
    sam_fa3_status = None

data = {
    "status": os.environ["BOOTSTRAP_STATE_STATUS"],
    "envManager": os.environ["MANAGER"],
    "envNames": env_names,
    "activeStrategy": os.environ["BOOTSTRAP_STRATEGY"],
    "workerEnvName": os.environ["BOOTSTRAP_WORKER_ENV"],
    "samEnvName": value("SAM_ENV_NAME"),
    "removeEnvName": value("REMOVE_ENV_NAME"),
    "voidEnvName": value("VOID_ENV_NAME"),
    "pythonVersion": value("PYTHON_VERSION"),
    "cudaBackend": value("CUDA_BACKEND"),
    "samFa3Status": sam_fa3_status,
    "samFa3Note": value("SAM_FA3_NOTE"),
    "workerHost": value("WORKER_HOST"),
    "workerPort": value("WORKER_PORT"),
    "lastValidatedAt": value("LAST_VALIDATED_AT"),
    "error": value("BOOTSTRAP_STATE_ERROR"),
    "storageRoot": value("STORAGE_ROOT"),
    "dataDir": value("DATA_DIR"),
    "projectsDir": value("PROJECTS_DIR"),
    "modelsDir": value("MODELS_DIR"),
    "bootstrapStatePath": value("STATE_PATH"),
    "hfHome": value("HF_HOME_PATH"),
    "hfHubCache": value("HF_HUB_CACHE_PATH"),
    "pipCacheDir": value("PIP_CACHE_PATH"),
    "mambaRootPrefix": value("MAMBA_ROOT_PREFIX_PATH"),
    "condaEnvsPath": value("CONDA_ENVS_PATH_VALUE"),
    "condaPkgsDirs": value("CONDA_PKGS_DIRS_VALUE"),
    "tempDir": value("TMPDIR_PATH"),
}

with open(state_path, "w", encoding="utf-8") as handle:
    json.dump(data, handle, separators=(",", ":"))
PY
}

# Keep lazy runtime imports in the bootstrap probes so previously created envs
# are repaired before requests hit code paths that need the new dependency.
sam_probe='import shutil; assert shutil.which("ffmpeg"), "ffmpeg not found in environment PATH"; import fastapi, torch, sam2, sam3, supervision, google.genai; import app.main'
remove_probe='import shutil; assert shutil.which("ffmpeg"), "ffmpeg not found in environment PATH"; import cv2, torch, diffsynth, modelscope, supervision; import app.runners.effecterase_remove'
void_probe='import shutil; assert shutil.which("ffmpeg"), "ffmpeg not found in environment PATH"; import absl, huggingface_hub, loguru, mediapy, ml_collections, peft, torch, transformers; import app.runners.void_download_assets, app.runners.void_remove'

ensure_void_repo_checkout

ensure_split_envs
download_model_assets
verify_bootstrap "split" "$SAM_ENV_NAME"
refresh_bootstrap_state "split" "$SAM_ENV_NAME"
write_state "split" "$SAM_ENV_NAME" "[\"$SAM_ENV_NAME\",\"$REMOVE_ENV_NAME\",\"$VOID_ENV_NAME\"]"
print_run_summary "split" "$SAM_ENV_NAME" "$SAM_ENV_NAME, $REMOVE_ENV_NAME, $VOID_ENV_NAME"
