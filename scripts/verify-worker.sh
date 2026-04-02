#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime-config.sh"

ENV_MANAGER="${ENV_MANAGER:-auto}"
STORAGE_ROOT="${STORAGE_ROOT:-}"
WORKER_BOOTSTRAP_STATE_PATH="${WORKER_BOOTSTRAP_STATE_PATH:-}"
STORAGE_ROOT_EXPLICIT=0
STRATEGY=""
WORKER_ENV=""
SAM_ENV=""
REMOVE_ENV=""
JSON_OUTPUT=0
BOOTSTRAP_MODE=0
ALLOW_MISSING_MODEL_ASSETS=0

if [[ -d "$HOME/.local/bin" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

usage() {
  echo "Usage: $0 [--json] [--bootstrap-mode] [--allow-missing-model-assets] [--env-manager conda|micromamba|auto] [--storage-root PATH] [--strategy split|shared] [--worker-env NAME] [--sam-env NAME] [--remove-env NAME]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    --bootstrap-mode)
      BOOTSTRAP_MODE=1
      shift
      ;;
    --allow-missing-model-assets)
      ALLOW_MISSING_MODEL_ASSETS=1
      shift
      ;;
    --env-manager)
      ENV_MANAGER="$2"
      shift 2
      ;;
    --storage-root)
      STORAGE_ROOT="$2"
      STORAGE_ROOT_EXPLICIT=1
      shift 2
      ;;
    --strategy)
      STRATEGY="$2"
      shift 2
      ;;
    --worker-env)
      WORKER_ENV="$2"
      shift 2
      ;;
    --sam-env)
      SAM_ENV="$2"
      shift 2
      ;;
    --remove-env)
      REMOVE_ENV="$2"
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

load_state_defaults() {
  if [[ ! -f "$STATE_PATH" ]]; then
    echo "Bootstrap state file not found at $STATE_PATH. Run ./scripts/setup-worker.sh first or pass explicit env details." >&2
    exit 1
  fi

  eval "$(state_exports "$STATE_PATH")"

  if [[ -z "$STRATEGY" ]]; then
    STRATEGY="$(state_field "$STATE_PATH" "activeStrategy" || true)"
  fi
  if [[ -z "$WORKER_ENV" ]]; then
    WORKER_ENV="$(state_field "$STATE_PATH" "workerEnvName" || true)"
  fi
  if [[ -z "$SAM_ENV" ]]; then
    SAM_ENV="$(state_field "$STATE_PATH" "samEnvName" || true)"
  fi
  if [[ -z "$REMOVE_ENV" ]]; then
    REMOVE_ENV="$(state_field "$STATE_PATH" "removeEnvName" || true)"
  fi
  if [[ "$ENV_MANAGER" == "auto" ]]; then
    ENV_MANAGER="$(state_field "$STATE_PATH" "envManager" || true)"
  fi
}

STATE_PATH="$(locate_bootstrap_state)"
if [[ -f "$STATE_PATH" ]]; then
  eval "$(state_exports "$STATE_PATH")"
fi

if [[ "$ENV_MANAGER" == "auto" && -f "$STATE_PATH" ]]; then
  ENV_MANAGER="$(state_field "$STATE_PATH" "envManager" || true)"
fi

if [[ -z "$STRATEGY" || -z "$WORKER_ENV" ]]; then
  load_state_defaults
fi

MANAGER="$ENV_MANAGER"
if [[ "$MANAGER" == "auto" ]]; then
  if command -v conda >/dev/null 2>&1; then
    MANAGER="conda"
  else
    MANAGER="micromamba"
  fi
fi

if [[ "$STRATEGY" != "shared" && "$STRATEGY" != "split" ]]; then
  echo "Unsupported strategy '$STRATEGY'. Expected 'shared' or 'split'." >&2
  exit 1
fi

if [[ "$STRATEGY" == "split" ]]; then
  if [[ -z "$SAM_ENV" || -z "$REMOVE_ENV" ]]; then
    echo "Split verification requires both --sam-env and --remove-env." >&2
    exit 1
  fi
fi

VERIFY_ARGS=(aggregate --manager "$MANAGER" --strategy "$STRATEGY" --worker-env "$WORKER_ENV")
if [[ "$STRATEGY" == "split" ]]; then
  VERIFY_ARGS+=(--sam-env "$SAM_ENV" --remove-env "$REMOVE_ENV")
fi
if [[ "$BOOTSTRAP_MODE" == "1" ]]; then
  VERIFY_ARGS+=(--bootstrap-mode)
fi
if [[ "$ALLOW_MISSING_MODEL_ASSETS" == "1" ]]; then
  VERIFY_ARGS+=(--allow-missing-model-assets)
fi
if [[ "$JSON_OUTPUT" == "1" ]]; then
  VERIFY_ARGS+=(--json)
fi

cd "$ROOT_DIR"

if [[ "$MANAGER" == "conda" ]]; then
  exec conda run --no-capture-output -n "$WORKER_ENV" python -m app.verify_worker "${VERIFY_ARGS[@]}"
fi

exec micromamba run -n "$WORKER_ENV" python -m app.verify_worker "${VERIFY_ARGS[@]}"
