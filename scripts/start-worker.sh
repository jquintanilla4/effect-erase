#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_PATH="$ROOT_DIR/data/bootstrap-status.json"
ENV_MANAGER="${ENV_MANAGER:-auto}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-manager)
      ENV_MANAGER="$2"
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

export WORKER_HOST="$HOST"
export WORKER_PORT="$PORT"
export ENV_MANAGER

"$ROOT_DIR/scripts/setup-worker.sh" --env-manager "$ENV_MANAGER" "$@"

if [[ ! -f "$STATE_PATH" ]]; then
  echo "Bootstrap state file not found at $STATE_PATH" >&2
  exit 1
fi

parse_state() {
  local key="$1"
  grep -o "\"$key\":\"[^\"]*\"" "$STATE_PATH" | head -n1 | cut -d'"' -f4
}

MANAGER="$(parse_state envManager)"
WORKER_ENV="$(parse_state workerEnvName)"

cd "$ROOT_DIR"

if [[ "$MANAGER" == "conda" ]]; then
  exec conda run --no-capture-output -n "$WORKER_ENV" python -m uvicorn app.main:app --app-dir worker --host "$HOST" --port "$PORT"
fi

exec micromamba run -n "$WORKER_ENV" python -m uvicorn app.main:app --app-dir worker --host "$HOST" --port "$PORT"

