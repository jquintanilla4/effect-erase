#!/usr/bin/env bash
set -euo pipefail

report_error() {
  local status="$?"
  local line_no="$1"
  local command_text="$2"
  # Mirror bootstrap diagnostics here so startup failures point at the exact
  # handoff that failed instead of only returning a generic make error.
  echo "start-worker.sh failed at line $line_no while running: $command_text" >&2
  exit "$status"
}

trap 'report_error $LINENO "$BASH_COMMAND"' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime-config.sh"

if [[ -d "$HOME/.local/bin" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

ENV_MANAGER="${ENV_MANAGER:-auto}"
STORAGE_ROOT="${STORAGE_ROOT:-}"
WORKER_BOOTSTRAP_STATE_PATH="${WORKER_BOOTSTRAP_STATE_PATH:-}"
STORAGE_ROOT_EXPLICIT=0
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-manager)
      ENV_MANAGER="$2"
      shift 2
      ;;
    --storage-root)
      STORAGE_ROOT="$2"
      STORAGE_ROOT_EXPLICIT=1
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

bootstrap_recovery_command() {
  local command="make bootstrap"
  local manager="${1:-}"
  local storage_root="${2:-}"

  if [[ -n "$manager" && "$manager" != "auto" ]]; then
    command+=" ENV_MANAGER=$manager"
  fi
  if [[ -n "$storage_root" && "$storage_root" != "$ROOT_DIR" ]]; then
    command+=" STORAGE_ROOT=$storage_root"
  fi

  printf '%s\n' "$command"
}

STATE_PATH="$(locate_bootstrap_state)"
if [[ ! -f "$STATE_PATH" ]]; then
  echo "Bootstrap state file not found at $STATE_PATH." >&2
  echo "Run $(bootstrap_recovery_command "$ENV_MANAGER" "$STORAGE_ROOT") first." >&2
  exit 1
fi

eval "$(state_exports "$STATE_PATH")"

STATE_STATUS="$(state_field "$STATE_PATH" "status" || true)"
STATE_MANAGER="$(state_field "$STATE_PATH" "envManager" || true)"
WORKER_ENV="$(state_field "$STATE_PATH" "workerEnvName" || true)"
STATE_ERROR="$(state_field "$STATE_PATH" "error" || true)"
STATE_STORAGE_ROOT="$(state_field "$STATE_PATH" "storageRoot" || true)"

if [[ -z "$STATE_STORAGE_ROOT" ]]; then
  STATE_STORAGE_ROOT="$STORAGE_ROOT"
fi

if [[ "$STATE_STATUS" != "ready" ]]; then
  echo "Bootstrap is not ready (status: ${STATE_STATUS:-unknown})." >&2
  if [[ -n "$STATE_ERROR" ]]; then
    echo "$STATE_ERROR" >&2
  fi
  echo "Run $(bootstrap_recovery_command "${STATE_MANAGER:-$ENV_MANAGER}" "$STATE_STORAGE_ROOT") and retry." >&2
  exit 1
fi

if [[ "$ENV_MANAGER" != "auto" && -n "$STATE_MANAGER" && "$ENV_MANAGER" != "$STATE_MANAGER" ]]; then
  echo "Bootstrap state expects env manager '$STATE_MANAGER', but startup was asked to use '$ENV_MANAGER'." >&2
  echo "Run $(bootstrap_recovery_command "$STATE_MANAGER" "$STATE_STORAGE_ROOT") or start the worker with --env-manager $STATE_MANAGER." >&2
  exit 1
fi

MANAGER="${STATE_MANAGER:-$ENV_MANAGER}"

if [[ -z "$WORKER_ENV" ]]; then
  echo "Bootstrap state at $STATE_PATH does not define workerEnvName." >&2
  echo "Run $(bootstrap_recovery_command "$MANAGER" "$STATE_STORAGE_ROOT") and retry." >&2
  exit 1
fi

export WORKER_HOST="$HOST"
export WORKER_PORT="$PORT"
export ENV_MANAGER="$MANAGER"

cd "$ROOT_DIR"
echo "Starting worker with $MANAGER env '$WORKER_ENV' on $HOST:$PORT"

if [[ "$MANAGER" == "conda" ]]; then
  exec conda run --no-capture-output -n "$WORKER_ENV" python -m uvicorn app.main:app --app-dir worker --host "$HOST" --port "$PORT"
fi

exec micromamba run -n "$WORKER_ENV" python -m uvicorn app.main:app --app-dir worker --host "$HOST" --port "$PORT"
