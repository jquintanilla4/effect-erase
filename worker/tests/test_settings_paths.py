import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings


class SettingsPathTests(unittest.TestCase):
    def test_storage_overrides_relocate_derived_worker_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "runtime-data"
            models_dir = Path(temp_dir) / "runtime-models"

            with patch.dict(
                os.environ,
                {
                    "WORKER_DATA_DIR": str(data_dir),
                    "WORKER_MODELS_DIR": str(models_dir),
                },
            ):
                settings = Settings()

        self.assertEqual(settings.projects_dir, data_dir / "projects")
        self.assertEqual(settings.bootstrap_state_path, data_dir / "bootstrap-status.json")
        self.assertEqual(settings.sam_checkpoint_path, models_dir / "sam3.1" / "sam3.1_multiplex.pt")
        self.assertEqual(settings.sam_legacy_checkpoint_path, models_dir / "sam3" / "sam3.pt")
        self.assertEqual(settings.sam2_checkpoint_path, models_dir / "sam2.1" / "sam2.1_hiera_base_plus.pt")
        self.assertEqual(settings.void_model_dir, models_dir / "VOID")
        self.assertEqual(settings.void_base_model_dir, models_dir / "VOID" / "CogVideoX-Fun-V1.5-5b-InP")
        self.assertEqual(settings.effecterase_model_dir, models_dir / "EffectErase")
        self.assertEqual(settings.effecterase_lora_path, models_dir / "EffectErase" / "EffectErase.ckpt")
        self.assertEqual(settings.effecterase_wan_model_dir, models_dir / "Wan-AI" / "Wan2.1-Fun-1.3B-InP")

    def test_gemini_api_key_supports_standard_environment_names(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "gemini-test-key",
            },
            clear=False,
        ):
            settings = Settings()

        self.assertEqual(settings.gemini_api_key, "gemini-test-key")
        self.assertTrue(settings.gemini_configured())


if __name__ == "__main__":
    unittest.main()
