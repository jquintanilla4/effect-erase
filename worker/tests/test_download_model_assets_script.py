import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_MODEL_ASSETS_SCRIPT = REPO_ROOT / "scripts" / "download-model-assets.sh"


class DownloadModelAssetsScriptTests(unittest.TestCase):
    def test_download_model_assets_materializes_from_cache_without_local_dir_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_root = temp_path / "runtime"
            sam3_dir = runtime_root / "models" / "sam3"
            sam3_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "import pathlib, zipfile; "
                        "path = pathlib.Path(__import__('sys').argv[1]); "
                        "archive = zipfile.ZipFile(path, 'w'); "
                        "archive.writestr('checkpoint.txt', 'ok'); "
                        "archive.close()"
                    ),
                    str(sam3_dir / "sam3.pt"),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            hf_args_path = temp_path / "hf-args.txt"
            bin_dir = temp_path / "bin"
            hf_path = bin_dir / "hf"
            bin_dir.mkdir()
            hf_path.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf '%s\\n' \"$*\" >> \"$HF_ARGS_LOG\"\n"
                "cache_dir=''\n"
                "positional=()\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  case \"$1\" in\n"
                "    --cache-dir)\n"
                "      cache_dir=\"$2\"\n"
                "      shift 2\n"
                "      ;;\n"
                "    --*)\n"
                "      shift\n"
                "      ;;\n"
                "    *)\n"
                "      positional+=(\"$1\")\n"
                "      shift\n"
                "      ;;\n"
                "  esac\n"
                "done\n"
                "repo_id=\"${positional[1]}\"\n"
                "filename=\"${positional[2]}\"\n"
                "target=\"$cache_dir/${repo_id//\\//__}__${filename//\\//__}\"\n"
                "mkdir -p \"$(dirname \"$target\")\"\n"
                "printf '{\"repo\":\"%s\",\"file\":\"%s\"}\\n' \"$repo_id\" \"$filename\" > \"$target\"\n"
                "printf '%s\\n' \"$target\"\n",
                encoding="utf-8",
            )
            hf_path.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["HF_ARGS_LOG"] = str(hf_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                [
                    "bash",
                    str(DOWNLOAD_MODEL_ASSETS_SCRIPT),
                    "--storage-root",
                    str(runtime_root),
                    "--skip-sam31",
                    "--skip-effecterase",
                    "--include-sam3",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            hf_args = hf_args_path.read_text(encoding="utf-8")
            config_present = (sam3_dir / "config.json").is_file()
            legacy_cache_present = (sam3_dir / ".cache" / "huggingface").exists()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--cache-dir", hf_args)
        self.assertNotIn("--local-dir", hf_args)
        self.assertTrue(config_present)
        self.assertFalse(legacy_cache_present)

    def test_download_model_assets_uses_central_hf_cache_instead_of_local_dir(self):
        script_text = DOWNLOAD_MODEL_ASSETS_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('hf download "$@" --cache-dir "$cache_dir"', script_text)
        self.assertNotIn('--local-dir "$local_dir"', script_text)

    def test_download_model_assets_disables_xet_and_cleans_legacy_local_cache(self):
        script_text = DOWNLOAD_MODEL_ASSETS_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"', script_text)
        self.assertIn('cleanup_legacy_local_dir_cache "$local_dir"', script_text)
        self.assertIn('rm -rf "$local_dir/.cache/huggingface"', script_text)

    def test_download_model_assets_installs_plain_huggingface_hub_cli(self):
        script_text = DOWNLOAD_MODEL_ASSETS_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('python3 -m pip install --user "huggingface_hub"', script_text)
        self.assertNotIn('huggingface_hub[cli]', script_text)

    def test_download_model_assets_treats_partial_markers_as_incomplete_until_finalize(self):
        script_text = DOWNLOAD_MODEL_ASSETS_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('marker_path="$(asset_marker_path "$local_dir" "$filename")"', script_text)
        self.assertIn('if [[ -f "$marker_path" ]]; then', script_text)
        self.assertIn('clear_asset_marker "$local_dir" "$filename"', script_text)

    def test_download_model_assets_clears_stale_sam31_partial_marker_when_assets_are_already_valid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_root = temp_path / "runtime"
            sam31_dir = runtime_root / "models" / "sam3.1"
            partial_path = sam31_dir / ".asset-partials" / "config.json.partial"
            sam31_dir.mkdir(parents=True, exist_ok=True)
            (sam31_dir / "config.json").write_text('{"model":"sam3.1"}\n', encoding="utf-8")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text("", encoding="utf-8")
            subprocess.run(
                [
                    "python3",
                    "-c",
                    (
                        "import pathlib, zipfile; "
                        "path = pathlib.Path(__import__('sys').argv[1]); "
                        "archive = zipfile.ZipFile(path, 'w'); "
                        "archive.writestr('checkpoint.txt', 'ok'); "
                        "archive.close()"
                    ),
                    str(sam31_dir / "sam3.1_multiplex.pt"),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            env = os.environ.copy()
            env["HOME"] = temp_dir

            result = subprocess.run(
                [
                    "bash",
                    str(DOWNLOAD_MODEL_ASSETS_SCRIPT),
                    "--storage-root",
                    str(runtime_root),
                    "--skip-effecterase",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertFalse(partial_path.exists(), "stale partial markers should be cleared once the target file is already valid")
        self.assertIn("Model asset check complete. No requested downloads needed.", result.stdout)

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
