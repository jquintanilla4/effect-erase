from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

# Repo-root discovery does not put worker/ on sys.path, so add it here before
# importing the worker package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.runners.effecterase_remove as remove_runner


class FakeTensor:
    def to(self, *_args, **_kwargs):
        return self


class ResolveNumFramesTests(unittest.TestCase):
    def test_resolve_num_frames_clamps_to_available_clip_length(self):
        with patch.object(remove_runner, "video_frame_count", side_effect=[56, 56]):
            resolved = remove_runner.resolve_num_frames("/tmp/source.mp4", "/tmp/mask.mp4", 81)

        self.assertEqual(resolved, 56)

    def test_resolve_num_frames_rejects_mask_length_mismatches(self):
        with patch.object(remove_runner, "video_frame_count", side_effect=[56, 48]):
            with self.assertRaises(ValueError) as context:
                remove_runner.resolve_num_frames("/tmp/source.mp4", "/tmp/mask.mp4", 81)

        self.assertIn("Source frames=56, mask frames=48", str(context.exception))


class TokenizerAssetTests(unittest.TestCase):
    def test_require_wan_tokenizer_assets_rejects_missing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_encoder_path = Path(temp_dir) / "models_t5_umt5-xxl-enc-bf16.pth"
            text_encoder_path.touch()

            with self.assertRaises(RuntimeError) as context:
                remove_runner.require_wan_tokenizer_assets(text_encoder_path.as_posix())

        self.assertIn("Missing Wan tokenizer assets", str(context.exception))


class RunTests(unittest.TestCase):
    def test_load_effecterase_models_reports_cuda_oom_clearly(self):
        fake_manager = MagicMock()
        fake_manager.load_models.side_effect = [None, remove_runner.torch.OutOfMemoryError("oom")]
        args = SimpleNamespace(
            dit_path="/tmp/dit.safetensors",
            text_encoder_path="/tmp/text_encoder.pth",
            vae_path="/tmp/vae.pth",
            image_encoder_path="/tmp/image_encoder.pth",
            pretrained_lora_path="/tmp/effecterase.ckpt",
            lora_alpha=1.0,
        )

        with self.assertRaises(RuntimeError) as context:
            remove_runner.load_effecterase_models(fake_manager, args)

        self.assertIn("ran out of memory", str(context.exception))
        self.assertIn("release the SAM worker state", str(context.exception))

    def test_load_effecterase_models_reports_exact_bad_asset_path(self):
        fake_manager = MagicMock()
        fake_manager.load_models.side_effect = [None, RuntimeError("archive corrupted")]
        args = SimpleNamespace(
            dit_path="/tmp/dit.safetensors",
            text_encoder_path="/tmp/text_encoder.pth",
            vae_path="/tmp/vae.pth",
            image_encoder_path="/tmp/image_encoder.pth",
            pretrained_lora_path="/tmp/effecterase.ckpt",
            lora_alpha=1.0,
        )

        with self.assertRaises(RuntimeError) as context:
            remove_runner.load_effecterase_models(fake_manager, args)

        self.assertIn(args.text_encoder_path, str(context.exception))
        self.assertIn("likely corrupted", str(context.exception))

    def test_run_passes_resolved_clip_length_into_pipeline(self):
        fake_pipe = MagicMock()
        fake_pipe.return_value = (["frame-1"], None)
        fake_pipe.enable_vram_management = MagicMock()
        fake_manager = MagicMock()
        args = SimpleNamespace(
            fg_bg_path="/tmp/source.mp4",
            mask_path="/tmp/mask.mp4",
            output_path="/tmp/output.mp4",
            num_frames=81,
            frame_interval=1,
            height=480,
            width=832,
            seed=2025,
            cfg=1.0,
            lora_alpha=1.0,
            num_inference_steps=50,
            tiled=False,
            use_teacache=False,
            text_encoder_path="/tmp/text_encoder.pth",
            vae_path="/tmp/vae.pth",
            dit_path="/tmp/dit.safetensors",
            image_encoder_path="/tmp/image_encoder.pth",
            pretrained_lora_path="/tmp/effecterase.ckpt",
        )

        video_tensor = FakeTensor()
        first_image_tensor = FakeTensor()

        with (
            patch.object(remove_runner, "resolve_num_frames", return_value=56),
            patch.object(remove_runner, "read_video_frames", side_effect=[(video_tensor, object()), (video_tensor, object())]) as read_video_frames_mock,
            patch.object(remove_runner, "crop_square_from_pil", return_value=first_image_tensor),
            patch.object(remove_runner, "save_video"),
            patch.object(remove_runner, "ModelManager", return_value=fake_manager),
            patch.object(remove_runner, "require_wan_tokenizer_assets"),
            patch.object(remove_runner.WanRemovePipeline, "from_model_manager", return_value=fake_pipe),
            patch.object(remove_runner.torch.cuda, "is_available", return_value=True),
            patch.object(remove_runner.torch, "inference_mode", side_effect=lambda: nullcontext()),
            patch.object(remove_runner.torch, "autocast", side_effect=lambda **_kwargs: nullcontext()),
        ):
            remove_runner.run(args)

        self.assertEqual(read_video_frames_mock.call_args_list[0].args[1], 56)
        self.assertEqual(read_video_frames_mock.call_args_list[1].args[1], 56)
        self.assertEqual(fake_pipe.call_args.kwargs["num_frames"], 56)


if __name__ == "__main__":
    unittest.main()
