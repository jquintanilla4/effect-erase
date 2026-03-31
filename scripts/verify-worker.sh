#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_PATH="$ROOT_DIR/data/bootstrap-status.json"
ENV_MANAGER="${ENV_MANAGER:-auto}"
STRATEGY=""
WORKER_ENV=""
SAM_ENV=""
REMOVE_ENV=""
JSON_OUTPUT=0

if [[ -d "$HOME/.local/bin" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

usage() {
  echo "Usage: $0 [--json] [--env-manager conda|micromamba|auto] [--strategy split|shared] [--worker-env NAME] [--sam-env NAME] [--remove-env NAME]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    --env-manager)
      ENV_MANAGER="$2"
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

parse_state() {
  local key="$1"
  grep -o "\"$key\":\"[^\"]*\"" "$STATE_PATH" | head -n1 | cut -d'"' -f4
}

load_state_defaults() {
  if [[ ! -f "$STATE_PATH" ]]; then
    echo "Bootstrap state file not found at $STATE_PATH. Run ./scripts/setup-worker.sh first or pass explicit env details." >&2
    exit 1
  fi

  # The state file is flat JSON, so shell parsing keeps this helper dependency-free.
  if [[ -z "$STRATEGY" ]]; then
    STRATEGY="$(parse_state activeStrategy)"
  fi
  if [[ -z "$WORKER_ENV" ]]; then
    WORKER_ENV="$(parse_state workerEnvName)"
  fi
  if [[ -z "$SAM_ENV" ]]; then
    SAM_ENV="$(parse_state samEnvName)"
  fi
  if [[ -z "$REMOVE_ENV" ]]; then
    REMOVE_ENV="$(parse_state removeEnvName)"
  fi
  if [[ "$ENV_MANAGER" == "auto" ]]; then
    ENV_MANAGER="$(parse_state envManager)"
  fi
}

MANAGER="$(detect_env_manager)"

if [[ -z "$STRATEGY" || -z "$WORKER_ENV" ]]; then
  load_state_defaults
  MANAGER="$(detect_env_manager)"
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
if [[ "$JSON_OUTPUT" == "1" ]]; then
  VERIFY_ARGS+=(--json)
fi

cd "$ROOT_DIR"

if [[ "$MANAGER" == "conda" ]]; then
  exec conda run --no-capture-output -n "$WORKER_ENV" python -m app.verify_worker "${VERIFY_ARGS[@]}"
fi

exec micromamba run -n "$WORKER_ENV" python -m app.verify_worker "${VERIFY_ARGS[@]}"
