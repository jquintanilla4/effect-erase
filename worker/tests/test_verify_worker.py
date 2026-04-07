from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

# Repo-root discovery does not put worker/ on sys.path, so add it here before
# importing the worker package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import verify_worker


def _env_report(env_name, role, *, imports_ok=True, cuda_ok=True, ffmpeg_ok=True):
    return {
        "envName": env_name,
        "role": role,
        "label": verify_worker.PROBE_DEFINITIONS[role]["label"],
        "imports": {
            "ok": imports_ok,
            "probe": verify_worker.PROBE_DEFINITIONS[role]["code"],
            "error": None if imports_ok else "import failure",
        },
        "cuda": {
            "ok": cuda_ok,
            "torchImportOk": True,
            "torchVersion": "2.8.0",
            "cudaAvailable": cuda_ok,
            "deviceCount": 1 if cuda_ok else 0,
            "firstDeviceName": "GPU 0" if cuda_ok else None,
            "error": None,
        },
        "tools": {
            "ffmpeg": {
                "ok": ffmpeg_ok,
                "path": "/usr/bin/ffmpeg" if ffmpeg_ok else None,
                "error": None if ffmpeg_ok else "ffmpeg not found in environment PATH.",
            },
        },
    }


def _model_report(*, sam_ok=True, effecterase_ok=True, void_ok=False):
    return {
        "sam": {
            "ok": sam_ok,
            "localModels": ["sam3.1"] if sam_ok else [],
            "checks": [],
            "sam2Config": {
                "path": "/tmp/sam2.yaml",
                "exists": True,
                "source": "package",
            },
        },
        "effectErase": {
            "ok": effecterase_ok,
            "requiredPaths": [],
        },
        "void": {
            "ok": void_ok,
            "requiredPaths": [],
        },
    }


class VerifyWorkerAggregateTests(unittest.TestCase):
    def _aggregate(self, *, runtime_mode="auto", imports_ok=True, cuda_ok=True, ffmpeg_ok=True, sam_ok=True, effecterase_ok=True, bootstrap_mode=False, allow_missing_model_assets=False):
        with (
            patch.object(verify_worker, "_runtime_mode", return_value=runtime_mode),
            patch.object(verify_worker, "_run_probe", side_effect=[
                _env_report("effecterase-sam", "sam", imports_ok=imports_ok, cuda_ok=cuda_ok, ffmpeg_ok=ffmpeg_ok),
                _env_report("effecterase-remove", "remove", imports_ok=imports_ok, cuda_ok=cuda_ok, ffmpeg_ok=ffmpeg_ok),
                _env_report("effecterase-void", "void", imports_ok=imports_ok, cuda_ok=cuda_ok, ffmpeg_ok=ffmpeg_ok),
            ]),
            patch.object(
                verify_worker,
                "_model_report",
                return_value=_model_report(sam_ok=sam_ok, effecterase_ok=effecterase_ok),
            ),
        ):
            return verify_worker._aggregate(
                "conda",
                "split",
                "effecterase-sam",
                "effecterase-sam",
                "effecterase-remove",
                "effecterase-void",
                bootstrap_mode=bootstrap_mode,
                allow_missing_model_assets=allow_missing_model_assets,
            )

    def test_strict_verification_requires_full_readiness(self):
        report = self._aggregate()

        self.assertTrue(report["ok"])
        self.assertTrue(report["realInferenceReady"])
        self.assertTrue(report["policy"]["cudaRequired"])
        self.assertTrue(report["policy"]["modelAssetsRequired"])

    def test_bootstrap_can_allow_missing_model_assets_for_staged_setups(self):
        report = self._aggregate(
            bootstrap_mode=True,
            allow_missing_model_assets=True,
            sam_ok=False,
            effecterase_ok=False,
        )

        self.assertTrue(report["ok"])
        self.assertTrue(report["bootstrapCompatible"])
        self.assertFalse(report["realInferenceReady"])
        self.assertFalse(report["policy"]["modelAssetsRequired"])

    def test_bootstrap_allows_missing_cuda_when_runtime_is_mock(self):
        report = self._aggregate(
            runtime_mode="mock",
            bootstrap_mode=True,
            cuda_ok=False,
        )

        self.assertTrue(report["ok"])
        self.assertFalse(report["realInferenceReady"])
        self.assertFalse(report["policy"]["cudaRequired"])

    def test_bootstrap_still_fails_without_cuda_in_auto_mode(self):
        report = self._aggregate(
            runtime_mode="auto",
            bootstrap_mode=True,
            cuda_ok=False,
        )

        self.assertFalse(report["ok"])
        self.assertTrue(report["policy"]["bootstrapMode"])
        self.assertTrue(report["policy"]["cudaRequired"])

    def test_bootstrap_fails_when_ffmpeg_is_missing(self):
        report = self._aggregate(ffmpeg_ok=False)

        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["toolsOk"])

    def test_strict_verification_keeps_assets_required_even_in_mock_mode(self):
        report = self._aggregate(
            runtime_mode="mock",
            sam_ok=False,
            effecterase_ok=False,
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["realInferenceReady"])
        self.assertTrue(report["policy"]["modelAssetsRequired"])


class VerifyWorkerPathCheckTests(unittest.TestCase):
    def test_path_check_flags_corrupt_pytorch_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "broken.pth"
            checkpoint_path.write_bytes(b"not-a-valid-archive")

            report = verify_worker._path_check("text_encoder", checkpoint_path)

        self.assertTrue(report["exists"])
        self.assertFalse(report["ok"])
        self.assertIn("zip-based PyTorch checkpoint", report["error"])

    def test_path_check_keeps_non_checkpoint_files_on_existence_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            report = verify_worker._path_check("sam3.1 config", config_path)

        self.assertTrue(report["exists"])
        self.assertTrue(report["ok"])
        self.assertIsNone(report["error"])


if __name__ == "__main__":
    unittest.main()
