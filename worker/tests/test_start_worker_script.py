import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
START_WORKER_SCRIPT = REPO_ROOT / "scripts" / "start-worker.sh"


class StartWorkerScriptTests(unittest.TestCase):
    def test_start_worker_fails_fast_when_bootstrap_state_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "missing-bootstrap.json"
            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["WORKER_BOOTSTRAP_STATE_PATH"] = str(state_path)

            result = subprocess.run(
                ["bash", str(START_WORKER_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Bootstrap state file not found", result.stderr)
        self.assertIn("make bootstrap", result.stderr)

    def test_start_worker_fails_fast_when_bootstrap_state_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state_path = temp_path / "bootstrap-status.json"
            state_path.write_text(
                json.dumps(
                    {
                        "status": "incomplete",
                        "envManager": "micromamba",
                        "envNames": ["effecterase-sam", "effecterase-remove"],
                        "activeStrategy": "split",
                        "workerEnvName": "effecterase-sam",
                        "error": "Bootstrap completed without required model assets.",
                        "storageRoot": "/workspace/effect-erase-runtime",
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["WORKER_BOOTSTRAP_STATE_PATH"] = str(state_path)

            result = subprocess.run(
                ["bash", str(START_WORKER_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Bootstrap is not ready", result.stderr)
        self.assertIn("Bootstrap completed without required model assets.", result.stderr)
        self.assertIn("make bootstrap ENV_MANAGER=micromamba STORAGE_ROOT=/workspace/effect-erase-runtime", result.stderr)

    def test_start_worker_uses_ready_state_without_rerunning_bootstrap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            state_path = temp_path / "bootstrap-status.json"
            conda_args_path = temp_path / "conda-args.txt"
            bin_dir = temp_path / "bin"
            conda_path = bin_dir / "conda"
            bin_dir.mkdir()
            conda_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s\\n' \"$@\" > \"$CONDA_ARGS_LOG\"\n",
                encoding="utf-8",
            )
            conda_path.chmod(0o755)
            state_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "envManager": "conda",
                        "envNames": ["effecterase-sam", "effecterase-remove"],
                        "activeStrategy": "split",
                        "workerEnvName": "effecterase-sam",
                        "samEnvName": "effecterase-sam",
                        "removeEnvName": "effecterase-remove",
                        "storageRoot": "/workspace/effect-erase-runtime",
                        "dataDir": "/workspace/effect-erase-runtime/data",
                        "projectsDir": "/workspace/effect-erase-runtime/data/projects",
                        "modelsDir": "/workspace/effect-erase-runtime/models",
                        "bootstrapStatePath": str(state_path),
                        "condaEnvsPath": "/workspace/effect-erase-runtime/conda/envs",
                        "condaPkgsDirs": "/workspace/effect-erase-runtime/conda/pkgs",
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["WORKER_BOOTSTRAP_STATE_PATH"] = str(state_path)
            env["CONDA_ARGS_LOG"] = str(conda_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(START_WORKER_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            conda_args = conda_args_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Starting worker with conda env 'effecterase-sam'", result.stdout)
        self.assertEqual(conda_args[:4], ["run", "--no-capture-output", "-n", "effecterase-sam"])
        self.assertIn("uvicorn", conda_args)
        self.assertNotIn("setup-worker.sh", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
