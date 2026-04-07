from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.void_mask_reasoner import (
    combine_black_and_grey_frames,
    gridify_masks,
    masks_from_grid_localizations,
    normalize_reasoning,
)


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
        self.assertTrue(masks[2][4:, 4:].all())
        self.assertTrue(masks[3][4:, 4:].all())


if __name__ == "__main__":
    unittest.main()
