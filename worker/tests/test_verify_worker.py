import unittest
from unittest.mock import patch

from app import verify_worker


def _env_report(env_name, role, *, imports_ok=True, cuda_ok=True):
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
    }


def _model_report(*, sam_ok=True, effecterase_ok=True):
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
    }


class VerifyWorkerAggregateTests(unittest.TestCase):
    def _aggregate(self, *, runtime_mode="auto", imports_ok=True, cuda_ok=True, sam_ok=True, effecterase_ok=True, bootstrap_mode=False, allow_missing_model_assets=False):
        with (
            patch.object(verify_worker, "_runtime_mode", return_value=runtime_mode),
            patch.object(verify_worker, "_run_probe", side_effect=[
                _env_report("effecterase-sam", "sam", imports_ok=imports_ok, cuda_ok=cuda_ok),
                _env_report("effecterase-remove", "remove", imports_ok=imports_ok, cuda_ok=cuda_ok),
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

    def test_strict_verification_keeps_assets_required_even_in_mock_mode(self):
        report = self._aggregate(
            runtime_mode="mock",
            sam_ok=False,
            effecterase_ok=False,
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["realInferenceReady"])
        self.assertTrue(report["policy"]["modelAssetsRequired"])


if __name__ == "__main__":
    unittest.main()
