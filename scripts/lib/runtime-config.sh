#!/usr/bin/env bash

is_runpod() {
  [[ -n "${RUNPOD_POD_ID:-}" || -n "${RUNPOD_VOLUME_ID:-}" ]]
}

default_storage_root() {
  if is_runpod; then
    echo "/workspace/effect-erase-runtime"
    return
  fi

  echo "$ROOT_DIR"
}

default_state_path_for_root() {
  local storage_root="$1"
  echo "$storage_root/data/bootstrap-status.json"
}

locate_bootstrap_state() {
  local -a candidates=()
  local candidate
  local fallback=""

  if [[ -n "${WORKER_BOOTSTRAP_STATE_PATH:-}" ]]; then
    candidates+=("$WORKER_BOOTSTRAP_STATE_PATH")
  fi

  if [[ -n "${STORAGE_ROOT:-}" ]]; then
    candidates+=("$(default_state_path_for_root "$STORAGE_ROOT")")
  fi

  if is_runpod; then
    candidates+=("$(default_state_path_for_root "/workspace/effect-erase-runtime")")
  fi

  candidates+=("$ROOT_DIR/data/bootstrap-status.json")

  for candidate in "${candidates[@]}"; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -z "$fallback" ]]; then
      fallback="$candidate"
    fi
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  echo "$fallback"
}

json_field() {
  local json_path="$1"
  local key="$2"

  python3 - "$json_path" "$key" <<'PY'
import json
import sys

json_path, key = sys.argv[1:3]

try:
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
except FileNotFoundError:
    raise SystemExit(1)

value = data.get(key)
if value is None:
    raise SystemExit(0)
if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (int, float)):
    print(value)
else:
    print(str(value))
PY
}

state_field() {
  local json_path="$1"
  local key="$2"

  if [[ ! -f "$json_path" ]]; then
    return 1
  fi

  json_field "$json_path" "$key"
}

state_exports() {
  local json_path="$1"

  python3 - "$json_path" <<'PY'
import json
import shlex
import sys

json_path = sys.argv[1]
with open(json_path, encoding="utf-8") as handle:
    data = json.load(handle)

field_map = {
    "storageRoot": "STORAGE_ROOT",
    "dataDir": "WORKER_DATA_DIR",
    "projectsDir": "WORKER_PROJECTS_DIR",
    "modelsDir": "WORKER_MODELS_DIR",
    "bootstrapStatePath": "WORKER_BOOTSTRAP_STATE_PATH",
    "hfHome": "HF_HOME",
    "hfHubCache": "HF_HUB_CACHE",
    "pipCacheDir": "PIP_CACHE_DIR",
    "mambaRootPrefix": "MAMBA_ROOT_PREFIX",
    "condaEnvsPath": "CONDA_ENVS_PATH",
    "condaPkgsDirs": "CONDA_PKGS_DIRS",
}

for json_key, env_key in field_map.items():
    value = data.get(json_key)
    if value in (None, ""):
        continue
    print(f"export {env_key}={shlex.quote(str(value))}")
PY
}

resolved_storage_root_for_mode() {
  local requested_root="$1"
  if [[ -n "$requested_root" ]]; then
    echo "$requested_root"
    return
  fi

  default_storage_root
}

should_manage_runtime_root() {
  local storage_root="$1"

  if [[ -n "${RUNTIME_ROOT_MANAGED:-}" ]]; then
    [[ "$RUNTIME_ROOT_MANAGED" == "1" ]]
    return
  fi

  if [[ -n "${STORAGE_ROOT_EXPLICIT:-}" && "$STORAGE_ROOT_EXPLICIT" == "1" ]]; then
    return 0
  fi

  if is_runpod; then
    return 0
  fi

  [[ "$storage_root" != "$ROOT_DIR" ]]
}

resolve_runtime_layout() {
  local requested_root="${1:-}"
  local storage_root

  storage_root="$(resolved_storage_root_for_mode "$requested_root")"

  STORAGE_ROOT="$storage_root"
  DATA_DIR="${WORKER_DATA_DIR:-$storage_root/data}"
  PROJECTS_DIR="${WORKER_PROJECTS_DIR:-$DATA_DIR/projects}"
  MODELS_DIR="${WORKER_MODELS_DIR:-$storage_root/models}"
  STATE_PATH="${WORKER_BOOTSTRAP_STATE_PATH:-$DATA_DIR/bootstrap-status.json}"

  HF_HOME_PATH="${HF_HOME:-}"
  HF_HUB_CACHE_PATH="${HF_HUB_CACHE:-}"
  PIP_CACHE_PATH="${PIP_CACHE_DIR:-}"
  MAMBA_ROOT_PREFIX_PATH="${MAMBA_ROOT_PREFIX:-}"
  CONDA_ENVS_PATH_VALUE="${CONDA_ENVS_PATH:-}"
  CONDA_PKGS_DIRS_VALUE="${CONDA_PKGS_DIRS:-}"

  if should_manage_runtime_root "$storage_root"; then
    HF_HOME_PATH="${HF_HOME_PATH:-$storage_root/cache/huggingface}"
    HF_HUB_CACHE_PATH="${HF_HUB_CACHE_PATH:-$HF_HOME_PATH/hub}"
    PIP_CACHE_PATH="${PIP_CACHE_PATH:-$storage_root/cache/pip}"
    MAMBA_ROOT_PREFIX_PATH="${MAMBA_ROOT_PREFIX_PATH:-$storage_root/micromamba}"
    CONDA_ENVS_PATH_VALUE="${CONDA_ENVS_PATH_VALUE:-$storage_root/conda/envs}"
    CONDA_PKGS_DIRS_VALUE="${CONDA_PKGS_DIRS_VALUE:-$storage_root/conda/pkgs}"
  fi
}

export_runtime_layout() {
  export STORAGE_ROOT
  export MODELS_DIR
  export WORKER_DATA_DIR="$DATA_DIR"
  export WORKER_PROJECTS_DIR="$PROJECTS_DIR"
  export WORKER_MODELS_DIR="$MODELS_DIR"
  export WORKER_BOOTSTRAP_STATE_PATH="$STATE_PATH"

  if [[ -n "$HF_HOME_PATH" ]]; then
    export HF_HOME="$HF_HOME_PATH"
  fi
  if [[ -n "$HF_HUB_CACHE_PATH" ]]; then
    export HF_HUB_CACHE="$HF_HUB_CACHE_PATH"
  fi
  if [[ -n "$PIP_CACHE_PATH" ]]; then
    export PIP_CACHE_DIR="$PIP_CACHE_PATH"
  fi
  if [[ -n "$MAMBA_ROOT_PREFIX_PATH" ]]; then
    export MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX_PATH"
  fi
  if [[ -n "$CONDA_ENVS_PATH_VALUE" ]]; then
    export CONDA_ENVS_PATH="$CONDA_ENVS_PATH_VALUE"
  fi
  if [[ -n "$CONDA_PKGS_DIRS_VALUE" ]]; then
    export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS_VALUE"
  fi
}

ensure_runtime_dirs() {
  mkdir -p "$DATA_DIR" "$PROJECTS_DIR" "$MODELS_DIR" "$(dirname "$STATE_PATH")"

  if [[ -n "$HF_HOME_PATH" ]]; then
    mkdir -p "$HF_HOME_PATH"
  fi
  if [[ -n "$HF_HUB_CACHE_PATH" ]]; then
    mkdir -p "$HF_HUB_CACHE_PATH"
  fi
  if [[ -n "$PIP_CACHE_PATH" ]]; then
    mkdir -p "$PIP_CACHE_PATH"
  fi
  if [[ -n "$MAMBA_ROOT_PREFIX_PATH" ]]; then
    mkdir -p "$MAMBA_ROOT_PREFIX_PATH"
  fi
  if [[ -n "$CONDA_ENVS_PATH_VALUE" ]]; then
    mkdir -p "$CONDA_ENVS_PATH_VALUE"
  fi
  if [[ -n "$CONDA_PKGS_DIRS_VALUE" ]]; then
    mkdir -p "$CONDA_PKGS_DIRS_VALUE"
  fi
}
