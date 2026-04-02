import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG_SCRIPT = REPO_ROOT / "scripts" / "lib" / "runtime-config.sh"


class RuntimeConfigScriptTests(unittest.TestCase):
    def locate_bootstrap_state(self, *, root_dir: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        script = (
            "set -euo pipefail\n"
            f"ROOT_DIR={root_dir!s}\n"
            f"source {RUNTIME_CONFIG_SCRIPT!s}\n"
            "locate_bootstrap_state\n"
        )

        return subprocess.run(
            ["bash", "-lc", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_locate_bootstrap_state_ignores_repo_state_when_runpod_defaults_apply(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_root = Path(temp_dir)
            (fake_root / "data").mkdir()
            (fake_root / "data" / "bootstrap-status.json").write_text("{}", encoding="utf-8")

            result = self.locate_bootstrap_state(
                root_dir=fake_root,
                extra_env={"RUNPOD_POD_ID": "pod-test"},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "/workspace/effect-erase-runtime/data/bootstrap-status.json")

    def test_locate_bootstrap_state_uses_explicit_storage_root_without_repo_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_root = Path(temp_dir) / "repo"
            fake_root.mkdir()
            (fake_root / "data").mkdir()
            (fake_root / "data" / "bootstrap-status.json").write_text("{}", encoding="utf-8")
            runtime_root = Path(temp_dir) / "runtime"

            result = self.locate_bootstrap_state(
                root_dir=fake_root,
                extra_env={"STORAGE_ROOT": str(runtime_root)},
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), str(runtime_root / "data" / "bootstrap-status.json"))


if __name__ == "__main__":
    unittest.main()
