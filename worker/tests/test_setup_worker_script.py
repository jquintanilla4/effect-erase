import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_WORKER_SCRIPT = REPO_ROOT / "scripts" / "setup-worker.sh"

FAKE_CONDA_SCRIPT = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -euo pipefail

    LOG_PATH="${CONDA_ARGS_LOG:?}"
    STATE_DIR="${FAKE_CONDA_STATE_DIR:?}"
    mkdir -p "$STATE_DIR"
    printf '%s\\n' "$*" >> "$LOG_PATH"

    env_path() {
      printf '%s/%s\\n' "$STATE_DIR" "$1"
    }

    marker_path() {
      printf '%s/%s\\n' "$(env_path "$1")" "$2"
    }

    has_marker() {
      [[ -f "$(marker_path "$1" "$2")" ]]
    }

    set_marker() {
      mkdir -p "$(env_path "$1")"
      : > "$(marker_path "$1" "$2")"
    }

    clear_env() {
      rm -rf "$(env_path "$1")"
    }

    print_env_list() {
      echo "# conda environments:"
      for dir in "$STATE_DIR"/*; do
        [[ -d "$dir" ]] || continue
        basename "$dir"
      done | sort
    }

    probe_passes() {
      local env_name="$1"
      local probe="$2"

      if [[ "$probe" == *"sam2, sam3"* ]]; then
        has_marker "$env_name" common_ready && has_marker "$env_name" sam_ready
        return $?
      fi
      if [[ "$probe" == *"diffsynth, modelscope"* ]]; then
        has_marker "$env_name" common_ready && has_marker "$env_name" remove_ready
        return $?
      fi
      if [[ "$probe" == *"import cv2, fastapi, torch, supervision; import app.main"* ]]; then
        has_marker "$env_name" common_ready
        return $?
      fi

      return 1
    }

    mark_from_pip() {
      local env_name="$1"
      shift
      local joined=" $* "

      if [[ "$joined" == *" opencv-python-headless<4.12.0.0 "* ]]; then
        set_marker "$env_name" common_ready
      fi
      if [[ "$joined" == *" /facebookresearch/sam3/archive/"* || "$joined" == *" einops>=0.8.0 "* || "$joined" == *" /facebookresearch/sam2/archive/"* ]]; then
        set_marker "$env_name" sam_ready
      fi
      if [[ "$joined" == *" /FudanCVL/EffectErase/archive/"* || "$joined" == *" modelscope>=1.28.0 "* ]]; then
        set_marker "$env_name" remove_ready
      fi
    }

    command="${1:-}"
    shift || true

    case "$command" in
      env)
        subcommand="${1:-}"
        shift || true
        case "$subcommand" in
          list)
            print_env_list
            exit 0
            ;;
          remove)
            env_name=""
            while [[ $# -gt 0 ]]; do
              case "$1" in
                -n)
                  env_name="$2"
                  shift 2
                  ;;
                *)
                  shift
                  ;;
              esac
            done
            clear_env "$env_name"
            exit 0
            ;;
        esac
        ;;
      create)
        env_name=""
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -n)
              env_name="$2"
              shift 2
              ;;
            *)
              shift
              ;;
          esac
        done
        mkdir -p "$(env_path "$env_name")"
        exit 0
        ;;
      remove)
        env_name=""
        while [[ $# -gt 0 ]]; do
          case "$1" in
            -n)
              env_name="$2"
              shift 2
              ;;
            *)
              shift
              ;;
          esac
        done
        clear_env "$env_name"
        exit 0
        ;;
      install)
        exit 0
        ;;
      run)
        env_name=""
        while [[ $# -gt 0 ]]; do
          case "$1" in
            --no-capture-output)
              shift
              ;;
            -n)
              env_name="$2"
              shift 2
              ;;
            *)
              break
              ;;
          esac
        done
        if [[ "${1:-}" == "env" ]]; then
          shift
          while [[ $# -gt 0 && "$1" == *=* ]]; do
            shift
          done
        fi
        if [[ "${1:-}" != "python" ]]; then
          exit 0
        fi
        shift
        case "${1:-}" in
          -c)
            probe="$2"
            if probe_passes "$env_name" "$probe"; then
              exit 0
            fi
            exit 1
            ;;
          -m)
            module_name="${2:-}"
            shift 2
            case "$module_name" in
              pip)
                action="${1:-}"
                shift || true
                case "$action" in
                  install)
                    mark_from_pip "$env_name" "$@"
                    ;;
                  uninstall)
                    ;;
                esac
                exit 0
                ;;
              app.verify_worker)
                if [[ " $* " == *" --json "* ]]; then
                  if [[ " $* " == *" --allow-missing-model-assets "* ]]; then
                    printf '{"modelAssetsOk":false}\\n'
                  else
                    printf '{"modelAssetsOk":true}\\n'
                  fi
                else
                  printf 'Bootstrap verification\\n'
                fi
                exit 0
                ;;
            esac
            ;;
        esac
        exit 0
        ;;
    esac

    exit 0
    """
)


def seed_fake_env(state_dir: Path, env_name: str, markers: list[str]) -> None:
    env_dir = state_dir / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    for marker in markers:
        (env_dir / marker).write_text("", encoding="utf-8")


class SetupWorkerScriptTests(unittest.TestCase):
    def run_setup_with_fake_conda(self, *, initial_envs: dict[str, list[str]] | None = None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            runtime_root = temp_path / "runtime"
            state_path = runtime_root / "data" / "bootstrap-status.json"
            conda_args_path = temp_path / "conda-args.txt"
            fake_conda_state_dir = temp_path / "fake-conda-state"
            bin_dir = temp_path / "bin"
            conda_path = bin_dir / "conda"
            bin_dir.mkdir()
            conda_path.write_text(FAKE_CONDA_SCRIPT, encoding="utf-8")
            conda_path.chmod(0o755)
            fake_conda_state_dir.mkdir()

            for env_name, markers in (initial_envs or {}).items():
                seed_fake_env(fake_conda_state_dir, env_name, markers)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["SKIP_SAM_FA3"] = "1"
            env["CONDA_ARGS_LOG"] = str(conda_args_path)
            env["FAKE_CONDA_STATE_DIR"] = str(fake_conda_state_dir)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                [
                    "bash",
                    str(SETUP_WORKER_SCRIPT),
                    "--non-interactive",
                    "--env-manager",
                    "conda",
                    "--storage-root",
                    str(runtime_root),
                    "--skip-model-downloads",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            conda_args = conda_args_path.read_text(encoding="utf-8").splitlines() if conda_args_path.exists() else []
            bootstrap_state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None

        return result, conda_args, bootstrap_state

    def test_setup_worker_silences_pip_root_warning_for_root_runpod_shells(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('export PIP_ROOT_USER_ACTION=ignore', script_text)

    def test_setup_worker_installs_sam2_without_build_isolation(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'env SAM2_BUILD_CUDA=0 python -m pip install -v --no-build-isolation "$SAM2_PACKAGE_SPEC"',
            script_text,
        )

    def test_setup_worker_pins_direct_upstream_package_refs(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('SAM3_PACKAGE_REF="${SAM3_PACKAGE_REF:-bfbed072a07a6a52c8d5fdc75a7a186251a835b1}"', script_text)
        self.assertIn('SAM2_PACKAGE_REF="${SAM2_PACKAGE_REF:-2b90b9f5ceec907a1c18123530e92e794ad901a4}"', script_text)
        self.assertIn('EFFECTERASE_PACKAGE_REF="${EFFECTERASE_PACKAGE_REF:-3dd007f6b2c60d13921c12c4a31051b32a530007}"', script_text)
        self.assertIn('FLASH_ATTENTION_HOPPER_REF="${FLASH_ATTENTION_HOPPER_REF:-83f9e450cd10e20701fb109db9c7703d376f282b}"', script_text)
        self.assertNotIn("archive/refs/heads/main.zip", script_text)
        self.assertIn('flash-attention.git@$FLASH_ATTENTION_HOPPER_REF#subdirectory=hopper', script_text)

    def test_setup_worker_quiets_expected_initial_probe_failures(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'if validate_env "$env_name" "$probe" >/dev/null 2>&1; then',
            script_text,
        )

    def test_setup_worker_logs_env_scoped_install_steps(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('echo "[$env_name] Installing shared worker dependencies..."', script_text)
        self.assertIn('echo "[$env_name] Installing SAM 2..."', script_text)
        self.assertIn('echo "[$env_name] Installing EffectErase runtime dependencies..."', script_text)

    def test_setup_worker_defaults_storage_root_without_prompting(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn('prompt_value "Runtime storage root"', script_text)
        self.assertIn('STORAGE_ROOT="$default_root"', script_text)

    def test_setup_worker_supports_skipping_sam_fa3_build(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('SKIP_SAM_FA3="${SKIP_SAM_FA3:-${SKIPSAMFA3:-0}}"', script_text)
        self.assertIn('if [[ "$SKIP_SAM_FA3" == "1" ]]; then', script_text)
        self.assertIn('Skipping FlashAttention 3 because SKIP_SAM_FA3=1', script_text)

    def test_setup_worker_avoids_repeating_same_run_sam_fa3_configuration(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('SAM_FA3_ENV_NAME=""', script_text)
        self.assertIn(
            'if [[ -n "$SAM_FA3_ENV_NAME" && "$SAM_FA3_ENV_NAME" == "$env_name" && "$SAM_FA3_STATUS" != "unknown" ]]; then',
            script_text,
        )

    def test_setup_worker_installs_effecterase_package_without_deps(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(
            'python -m pip install --no-build-isolation --no-deps "$EFFECTERASE_PACKAGE_SPEC"',
            script_text,
        )

    def test_setup_worker_pins_effecterase_runtime_dependency_overrides(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"numpy<2.0.0"', script_text)
        self.assertIn('"transformers>=4.46.2,<5"', script_text)
        self.assertIn('"fsspec>=2023.1.0,<=2026.2.0"', script_text)
        self.assertIn('"setuptools<82"', script_text)
        self.assertIn('"opencv-python-headless<4.12.0.0"', script_text)

    def test_setup_worker_keeps_existing_torch_stack_for_effecterase_layers(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        shared_fn = script_text.split("install_effecterase_shared_deps() {", 1)[1].split("install_effecterase_remove_deps() {", 1)[0]
        remove_fn = script_text.split("install_effecterase_remove_deps() {", 1)[1].split("install_sam2_package() {", 1)[0]

        self.assertNotIn("--force-reinstall", shared_fn)
        self.assertNotIn("--force-reinstall", remove_fn)
        self.assertIn("Keep the CUDA torch stack", shared_fn)
        self.assertIn("Keep the CUDA torch stack", remove_fn)

    def test_setup_worker_removes_split_clone_bootstrap_path(self):
        script_text = SETUP_WORKER_SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("clone_env()", script_text)
        self.assertNotIn("ensure_split_envs_via_clone()", script_text)
        self.assertNotIn("--clone", script_text)
        self.assertNotIn("effecterase-split-base", script_text)

    def test_setup_worker_prefers_micromamba_on_runpod_when_auto_detecting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            conda_args_path = temp_path / "conda-args.txt"
            micromamba_args_path = temp_path / "micromamba-args.txt"
            bin_dir = temp_path / "bin"
            conda_path = bin_dir / "conda"
            micromamba_path = bin_dir / "micromamba"
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
            micromamba_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s ' \"$@\" >> \"$MICROMAMBA_ARGS_LOG\"\n"
                "printf '\\n' >> \"$MICROMAMBA_ARGS_LOG\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            micromamba_path.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["RUNPOD_POD_ID"] = "pod-test"
            env["CONDA_ARGS_LOG"] = str(conda_args_path)
            env["MICROMAMBA_ARGS_LOG"] = str(micromamba_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                [
                    "bash",
                    str(SETUP_WORKER_SCRIPT),
                    "--non-interactive",
                    "--env-manager",
                    "auto",
                    "--storage-root",
                    str(runtime_root),
                    "--skip-model-downloads",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            micromamba_args = micromamba_args_path.read_text(encoding="utf-8")

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(conda_args_path.exists(), "Runpod auto bootstrap should not pick conda when micromamba is available")
        self.assertIn("env list", micromamba_args)

    def test_setup_worker_reuses_recorded_env_manager_on_runpod_before_preferring_micromamba(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            conda_args_path = temp_path / "conda-args.txt"
            micromamba_args_path = temp_path / "micromamba-args.txt"
            bin_dir = temp_path / "bin"
            conda_path = bin_dir / "conda"
            micromamba_path = bin_dir / "micromamba"
            runtime_root = temp_path / "runtime"
            state_path = runtime_root / "data" / "bootstrap-status.json"
            hf_home = temp_path / "hf-home"
            token_path = hf_home / "token"
            bin_dir.mkdir()
            (runtime_root / "data").mkdir(parents=True)
            hf_home.mkdir()
            token_path.write_text("hf_test_token\n", encoding="utf-8")
            state_path.write_text('{"envManager":"conda"}', encoding="utf-8")
            conda_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s ' \"$@\" >> \"$CONDA_ARGS_LOG\"\n"
                "printf '\\n' >> \"$CONDA_ARGS_LOG\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            conda_path.chmod(0o755)
            micromamba_path.write_text(
                "#!/usr/bin/env sh\n"
                "printf '%s ' \"$@\" >> \"$MICROMAMBA_ARGS_LOG\"\n"
                "printf '\\n' >> \"$MICROMAMBA_ARGS_LOG\"\n"
                "exit 99\n",
                encoding="utf-8",
            )
            micromamba_path.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = temp_dir
            env["HF_HOME"] = str(hf_home)
            env["RUNPOD_POD_ID"] = "pod-test"
            env["CONDA_ARGS_LOG"] = str(conda_args_path)
            env["MICROMAMBA_ARGS_LOG"] = str(micromamba_args_path)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"

            result = subprocess.run(
                [
                    "bash",
                    str(SETUP_WORKER_SCRIPT),
                    "--non-interactive",
                    "--env-manager",
                    "auto",
                    "--storage-root",
                    str(runtime_root),
                    "--skip-model-downloads",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            conda_args = conda_args_path.read_text(encoding="utf-8")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("env list", conda_args)
        self.assertFalse(
            micromamba_args_path.exists(),
            "Runpod auto bootstrap should reuse the recorded conda manager when bootstrap state says conda",
        )

    def test_setup_worker_uses_direct_split_bootstrap_for_fresh_envs(self):
        result, conda_args, bootstrap_state = self.run_setup_with_fake_conda()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertFalse(any("effecterase-split-base" in line for line in conda_args))
        self.assertIn("create -y -n effecterase-sam python=3.12 pip git ffmpeg", conda_args)
        self.assertIn("create -y -n effecterase-remove python=3.12 pip git ffmpeg", conda_args)
        self.assertTrue(any("effecterase-sam python -m pip install --index-url" in line for line in conda_args))
        self.assertTrue(any("effecterase-remove python -m pip install --index-url" in line for line in conda_args))
        self.assertTrue(
            any(
                "effecterase-remove python -m pip install --no-build-isolation --no-deps https://github.com/FudanCVL/EffectErase/archive/3dd007f6b2c60d13921c12c4a31051b32a530007.zip"
                in line
                for line in conda_args
            )
        )
        self.assertEqual(bootstrap_state["envNames"], ["effecterase-sam", "effecterase-remove"])
        self.assertNotIn("effecterase-split-base", json.dumps(bootstrap_state))

    def test_setup_worker_repairs_both_split_envs_directly_when_both_need_rebuild(self):
        result, conda_args, _ = self.run_setup_with_fake_conda(
            initial_envs={
                "effecterase-sam": [],
                "effecterase-remove": [],
            }
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertFalse(any("effecterase-split-base" in line for line in conda_args))
        self.assertTrue(any("effecterase-sam python -m pip install --index-url" in line for line in conda_args))
        self.assertTrue(any("effecterase-remove python -m pip install --index-url" in line for line in conda_args))
        self.assertIn("Repaired envs: effecterase-sam, effecterase-remove", result.stdout)

    def test_setup_worker_keeps_direct_single_env_repair_path(self):
        result, conda_args, bootstrap_state = self.run_setup_with_fake_conda(
            initial_envs={
                "effecterase-sam": ["common_ready", "sam_ready"],
            }
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertFalse(any("effecterase-split-base" in line for line in conda_args))
        self.assertFalse(any("--clone effecterase-split-base" in line for line in conda_args))
        self.assertIn("Reused envs: effecterase-sam", result.stdout)
        self.assertIn("Created envs: effecterase-remove", result.stdout)
        self.assertEqual(bootstrap_state["envNames"], ["effecterase-sam", "effecterase-remove"])

    def test_setup_worker_recreates_invalid_single_split_env_before_repair(self):
        result, conda_args, bootstrap_state = self.run_setup_with_fake_conda(
            initial_envs={
                "effecterase-sam": ["common_ready", "sam_ready"],
                "effecterase-remove": [],
            }
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertFalse(any("effecterase-split-base" in line for line in conda_args))
        remove_index = conda_args.index("remove -y -n effecterase-remove --all")
        create_index = conda_args.index("create -y -n effecterase-remove python=3.12 pip git ffmpeg")
        install_index = next(
            index
            for index, line in enumerate(conda_args)
            if "effecterase-remove python -m pip install --index-url" in line
        )
        self.assertLess(remove_index, create_index)
        self.assertLess(create_index, install_index)
        self.assertIn("Reused envs: effecterase-sam", result.stdout)
        self.assertIn("Repaired envs: effecterase-remove", result.stdout)
        self.assertEqual(bootstrap_state["envNames"], ["effecterase-sam", "effecterase-remove"])

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
