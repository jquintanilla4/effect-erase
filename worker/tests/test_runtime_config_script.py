import json
import os
from pathlib import Path
import shlex
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

    def resolve_runtime_layout(self, *, root_dir: Path, extra_env: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)

        script = (
            "set -euo pipefail\n"
            f"ROOT_DIR={shlex.quote(str(root_dir))}\n"
            f"source {shlex.quote(str(RUNTIME_CONFIG_SCRIPT))}\n"
            "resolve_runtime_layout \"\"\n"
            "export_runtime_layout\n"
            "python3 - <<'PY'\n"
            "import json\n"
            "import os\n"
            "print(json.dumps({\n"
            "    'storageRoot': os.environ.get('STORAGE_ROOT', ''),\n"
            "    'hfHome': os.environ.get('HF_HOME', ''),\n"
            "    'hfHubCache': os.environ.get('HF_HUB_CACHE', ''),\n"
            "    'pipCacheDir': os.environ.get('PIP_CACHE_DIR', ''),\n"
            "    'mambaRootPrefix': os.environ.get('MAMBA_ROOT_PREFIX', ''),\n"
            "    'condaEnvsPath': os.environ.get('CONDA_ENVS_PATH', ''),\n"
            "    'condaPkgsDirs': os.environ.get('CONDA_PKGS_DIRS', ''),\n"
            "    'tmpDir': os.environ.get('TMPDIR', ''),\n"
            "    'tmp': os.environ.get('TMP', ''),\n"
            "    'temp': os.environ.get('TEMP', ''),\n"
            "}))\n"
            "PY\n"
        )

        result = subprocess.run(
            ["bash", "-lc", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return json.loads(result.stdout)

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

    def test_resolve_runtime_layout_moves_runpod_temp_and_caches_under_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = self.resolve_runtime_layout(
                root_dir=Path(temp_dir),
                extra_env={"RUNPOD_POD_ID": "pod-test"},
            )

        workspace_root = "/workspace/effect-erase-runtime"
        self.assertEqual(layout["storageRoot"], workspace_root)
        self.assertEqual(layout["hfHome"], f"{workspace_root}/cache/huggingface")
        self.assertEqual(layout["hfHubCache"], f"{workspace_root}/cache/huggingface/hub")
        self.assertEqual(layout["pipCacheDir"], f"{workspace_root}/cache/pip")
        self.assertEqual(layout["mambaRootPrefix"], f"{workspace_root}/micromamba")
        self.assertEqual(layout["condaEnvsPath"], f"{workspace_root}/conda/envs")
        self.assertEqual(layout["condaPkgsDirs"], f"{workspace_root}/conda/pkgs")
        self.assertEqual(layout["tmpDir"], f"{workspace_root}/tmp")
        self.assertEqual(layout["tmp"], f"{workspace_root}/tmp")
        self.assertEqual(layout["temp"], f"{workspace_root}/tmp")


if __name__ == "__main__":
    unittest.main()
