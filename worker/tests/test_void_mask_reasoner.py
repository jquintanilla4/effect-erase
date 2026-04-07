from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.void_mask_reasoner import (
    VoidReasoningResult,
    VoidQuadmaskBuilder,
    combine_black_and_grey_frames,
    gridify_masks,
    masks_from_grid_localizations,
    normalize_reasoning,
)
from app.core.config import Settings


class VoidMaskReasonerTests(unittest.TestCase):
    def test_normalize_reasoning_falls_back_to_scene_description_for_background_prompt(self):
        reasoning = normalize_reasoning(
            {
                "integral_belongings": [{"noun": "Bike", "why": "ridden by subject"}],
                "affected_objects": [{"noun": "Shadow", "category": "visual_artifact"}],
                "scene_description": "Empty street after the rider is removed.",
            }
        )

        self.assertEqual(reasoning.integral_belongings[0]["noun"], "bike")
        self.assertEqual(reasoning.affected_objects[0]["noun"], "shadow")
        self.assertEqual(reasoning.background_prompt, "Empty street after the rider is removed.")

    def test_gridify_masks_fills_the_entire_cell_when_any_pixel_is_set(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = True

        [gridified] = gridify_masks([mask], grid_rows=2, grid_cols=2)

        self.assertTrue(gridified[:4, :4].all())
        self.assertFalse(gridified[4:, 4:].any())

    def test_combine_black_and_grey_frames_uses_all_quadmask_values(self):
        black = np.array([[0, 255], [0, 255]], dtype=np.uint8)
        grey = np.array([[255, 127], [127, 255]], dtype=np.uint8)

        [combined] = combine_black_and_grey_frames([black], [grey])

        self.assertEqual(combined[0, 0], 0)
        self.assertEqual(combined[0, 1], 127)
        self.assertEqual(combined[1, 0], 63)
        self.assertEqual(combined[1, 1], 255)

    def test_masks_from_grid_localizations_hold_regions_until_next_keyframe(self):
        masks = masks_from_grid_localizations(
            [
                {"frame": 0, "grid_regions": [{"row": 0, "col": 0}]},
                {"frame": 2, "grid_regions": [{"row": 1, "col": 1}]},
            ],
            total_frames=4,
            frame_shape=(8, 8),
            grid_rows=2,
            grid_cols=2,
        )

        self.assertTrue(masks[0][:4, :4].all())
        self.assertTrue(masks[1][:4, :4].all())
        self.assertFalse(masks[2][:4, :4].any())
        self.assertTrue(masks[2][4:, 4:].all())
        self.assertTrue(masks[3][4:, 4:].all())

    def test_reference_primary_mask_uses_first_non_empty_frame(self):
        builder = VoidQuadmaskBuilder(Settings())
        black_frames = [
            np.full((4, 4), 255, dtype=np.uint8),
            np.array(
                [
                    [255, 0, 255, 255],
                    [255, 0, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                ],
                dtype=np.uint8,
            ),
        ]

        frame_index, primary_mask = builder._reference_primary_mask(black_frames)

        self.assertEqual(frame_index, 1)
        self.assertTrue(primary_mask[:, 1].any())

    def test_reference_primary_mask_rejects_all_empty_masks(self):
        builder = VoidQuadmaskBuilder(Settings())
        black_frames = [np.full((4, 4), 255, dtype=np.uint8)]

        with self.assertRaisesRegex(RuntimeError, "non-empty propagated mask"):
            builder._reference_primary_mask(black_frames)

    def test_build_passes_selected_reference_frame_to_gemini_analysis(self):
        builder = VoidQuadmaskBuilder(Settings())
        black_frames = [
            np.full((4, 4), 255, dtype=np.uint8),
            np.array(
                [
                    [255, 0, 255, 255],
                    [255, 0, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                ],
                dtype=np.uint8,
            ),
        ]

        with patch("app.services.void_mask_reasoner.load_video_metadata") as load_video_metadata, \
             patch("app.services.void_mask_reasoner.build_black_mask_frames", return_value=black_frames), \
             patch.object(builder.analyzer, "analyze") as analyze, \
             patch("app.services.void_mask_reasoner._materialize_source"), \
             patch("app.services.void_mask_reasoner.write_lossless_mask_video"), \
             patch("app.services.void_mask_reasoner.combine_black_and_grey_frames", return_value=black_frames):
            load_video_metadata.return_value = type(
                "Metadata",
                (),
                {"fps": 24.0, "width": 4, "height": 4, "frame_count": 2},
            )()
            analyze.return_value = (
                VoidReasoningResult([], [], "empty scene", "clean background", 0.9),
                2,
                2,
            )

            builder.build(
                source_video_path=Path("/tmp/source.mp4"),
                mask_video_path=Path("/tmp/mask.mp4"),
                sequence_dir=Path("/tmp/void-sequence"),
            )

        self.assertEqual(analyze.call_args.args[2], 1)


if __name__ == "__main__":
    unittest.main()
