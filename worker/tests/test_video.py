from pathlib import Path
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Repo-root discovery does not put worker/ on sys.path, so add it here before
# importing the worker package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.video import write_mask_overlay_video, write_mask_video


class WriteMaskVideoTests(unittest.TestCase):
    def test_write_mask_video_uses_browser_safe_h264_settings(self):
        process = MagicMock()
        process.stdin = MagicMock()
        process.stderr = io.BytesIO(b"")
        process.wait.return_value = 0
        masks = [np.zeros((4, 6), dtype=np.uint8), np.ones((4, 6), dtype=np.uint8) * 255]

        with patch("app.models.video.subprocess.Popen", return_value=process) as popen:
            metadata = write_mask_video(Path("/tmp/mask_sequence.mp4"), masks, 24.0, 6, 4)

        command = popen.call_args.args[0]
        self.assertIn("libopenh264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertEqual(process.stdin.write.call_count, 2)
        first_frame_bytes = process.stdin.write.call_args_list[0].args[0]
        self.assertEqual(len(first_frame_bytes), 4 * 6 * 3)
        process.stdin.close.assert_called_once()
        self.assertEqual(metadata.frame_count, 2)

    def test_write_mask_overlay_video_resolves_color_by_index(self):
        class FakeDetections:
            def __init__(self, xyxy, mask):
                self.xyxy = xyxy
                self.mask = mask

        class FakeMaskAnnotator:
            instances = []

            def __init__(self, *, color, color_lookup):
                self.color = color
                self.color_lookup = color_lookup
                self.annotate_calls = []
                FakeMaskAnnotator.instances.append(self)

            def annotate(self, *, scene, detections):
                self.annotate_calls.append((scene, detections))
                return scene

        class FakeSupervisionModule:
            class Color:
                def __init__(self, r, g, b):
                    self.r = r
                    self.g = g
                    self.b = b

            class ColorLookup:
                INDEX = "index"

            Detections = FakeDetections
            MaskAnnotator = FakeMaskAnnotator

        source_frames = [np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8) * 50]
        mask_frames = [np.zeros((4, 6), dtype=np.uint8), np.ones((4, 6), dtype=np.uint8) * 255]

        with patch.dict("sys.modules", {"supervision": FakeSupervisionModule}):
            with patch(
                "app.models.video.iterate_video_frames",
                side_effect=[iter(source_frames), iter(mask_frames)],
            ):
                with patch("app.models.video._write_browser_safe_mp4") as write_mp4:
                    metadata = write_mask_overlay_video(
                        Path("/tmp/mask_overlay.mp4"),
                        Path("/tmp/source.mp4"),
                        Path("/tmp/mask.mp4"),
                        24.0,
                        6,
                        4,
                    )

        annotator = FakeMaskAnnotator.instances[0]
        self.assertEqual(annotator.color_lookup, FakeSupervisionModule.ColorLookup.INDEX)
        self.assertEqual(len(annotator.annotate_calls), 2)
        self.assertEqual(metadata.frame_count, 2)
        write_mp4.assert_called_once()


if __name__ == "__main__":
    unittest.main()
