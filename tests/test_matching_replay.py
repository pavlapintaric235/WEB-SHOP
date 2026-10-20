"""Saved inventory proposals must never masquerade as fresh reference inference."""

import unittest

import numpy as np
from PIL import Image

from check_matching import replay_segments
from sam_vision import encode_mask


class MatchingReplayTests(unittest.TestCase):
    def test_saved_masks_and_qualities_are_used_only_for_their_original_photo(self):
        image = Image.new("RGB", (80, 100), "red")
        mask = np.zeros((100, 80), dtype=bool)
        mask[10:90, 20:60] = True
        snapshot = {"raw_detections": [{"mask": encode_mask(mask), "mask_quality": 0.95}]}
        segment = replay_segments(snapshot, image)
        masks, qualities = segment(image.copy())
        np.testing.assert_array_equal(masks[0], mask)
        self.assertEqual(qualities, [0.95])
        with self.assertRaisesRegex(ValueError, "original inventory image"):
            segment(Image.new("RGB", image.size, "blue"))
        with self.assertRaises(ValueError):
            segment(Image.new("RGB", (20, 30), "red"))

    def test_mismatched_saved_resolution_is_rejected(self):
        snapshot = {"raw_detections": [{"mask": encode_mask(np.ones((20, 30), dtype=bool)), "mask_quality": 1.0}]}
        with self.assertRaisesRegex(ValueError, "resolution"):
            replay_segments(snapshot, Image.new("RGB", (80, 100), "red"))

    def test_empty_saved_scene_remains_empty(self):
        image = Image.new("RGB", (80, 100), "red")
        self.assertEqual(replay_segments({"raw_detections": []}, image)(image), ([], []))


if __name__ == "__main__":
    unittest.main()
