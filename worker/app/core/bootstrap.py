import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.api import BootstrapStatus


def load_bootstrap_status(path: Path) -> BootstrapStatus:
    if not path.exists():
        return BootstrapStatus(
            status="missing",
            envManager="unknown",
            envNames=[],
            activeStrategy="unknown",
            workerEnvName=None,
            samEnvName=None,
            removeEnvName=None,
            pythonVersion=None,
            cudaBackend=None,
            lastValidatedAt=datetime.now(timezone.utc).isoformat(),
            error="Bootstrap state file not found.",
        )

    data = json.loads(path.read_text())
    return BootstrapStatus(**data)

