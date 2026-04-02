import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_MODEL_ASSETS_SCRIPT = REPO_ROOT / "scripts" / "download-model-assets.sh"


class DownloadModelAssetsScriptTests(unittest.TestCase):
    def test_download_model_assets_does_not_fall_back_to_sam21_when_sam31_download_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            hf_args_path = temp_path / "hf-args.txt"
            bin_dir = temp_path / "bin"
            hf_path = bin_dir / "hf"
            runtime_root = temp_path / "runtime"
            bin_dir.mkdir()
            hf_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s ' \"$@\" >> \"$HF_ARGS_LOG\"\n"
                "printf '\\n' >> \"$HF_ARGS_LOG\"\n"
                "exit 1\n",
                encoding="utf-8",
            )
            hf_path.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["HF_TOKEN"] = "hf_test_token"
            env["HF_ARGS_LOG"] = str(hf_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                ["bash", str(DOWNLOAD_MODEL_ASSETS_SCRIPT), "--storage-root", str(runtime_root)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            hf_args = hf_args_path.read_text(encoding="utf-8")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to prepare required SAM 3.1 assets", result.stderr)
        self.assertIn("--skip-sam31 --include-sam21", result.stderr)
        self.assertNotIn("Falling back to SAM 2.1", result.stderr)
        self.assertIn("download", hf_args)


if __name__ == "__main__":
    unittest.main()
