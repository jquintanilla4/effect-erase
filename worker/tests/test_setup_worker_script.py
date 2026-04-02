import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_WORKER_SCRIPT = REPO_ROOT / "scripts" / "setup-worker.sh"


class SetupWorkerScriptTests(unittest.TestCase):
    def test_setup_worker_installs_sam2_without_build_isolation(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'env SAM2_BUILD_CUDA=0 python -m pip install -v --no-build-isolation "$SAM2_PACKAGE_SPEC"',
            script_text,
        )

    def test_setup_worker_requires_hf_auth_before_env_setup_in_non_interactive_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            conda_args_path = temp_path / "conda-args.txt"
            bin_dir = temp_path / "bin"
            conda_path = bin_dir / "conda"
            runtime_root = temp_path / "runtime"
            bin_dir.mkdir()
            conda_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s ' \"$@\" >> \"$CONDA_ARGS_LOG\"\n"
                "printf '\\n' >> \"$CONDA_ARGS_LOG\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            conda_path.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["CONDA_ARGS_LOG"] = str(conda_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(SETUP_WORKER_SCRIPT), "--non-interactive", "--env-manager", "conda", "--storage-root", str(runtime_root)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Hugging Face auth is required for default bootstrap", result.stderr)
        self.assertIn("hf auth login", result.stderr)
        self.assertFalse(conda_args_path.exists(), "bootstrap should fail before touching conda when HF auth is missing")

    def test_setup_worker_reuses_saved_hf_login_without_prompting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hf_home = temp_path / "hf-home"
            token_path = hf_home / "token"
            conda_args_path = temp_path / "conda-args.txt"
            bin_dir = temp_path / "bin"
            conda_path = bin_dir / "conda"
            runtime_root = temp_path / "runtime"
            hf_home.mkdir(parents=True)
            token_path.write_text("hf_test_token\n", encoding="utf-8")
            bin_dir.mkdir()
            conda_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s ' \"$@\" >> \"$CONDA_ARGS_LOG\"\n"
                "printf '\\n' >> \"$CONDA_ARGS_LOG\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            conda_path.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["HF_HOME"] = str(hf_home)
            env["CONDA_ARGS_LOG"] = str(conda_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(SETUP_WORKER_SCRIPT), "--non-interactive", "--env-manager", "conda", "--storage-root", str(runtime_root)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            conda_args = conda_args_path.read_text(encoding="utf-8")

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Hugging Face auth is required for default bootstrap", result.stderr)
        self.assertIn("env list", conda_args)


if __name__ == "__main__":
    unittest.main()
