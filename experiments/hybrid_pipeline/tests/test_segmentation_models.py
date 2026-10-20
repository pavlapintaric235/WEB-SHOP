import unittest

import numpy as np
from PIL import Image

from experiments.hybrid_pipeline.crop_ablation import crop_for_query
from experiments.hybrid_pipeline.evaluate_hybrid_masks import match_masks, score_instances
from experiments.hybrid_pipeline.mask_runtime import (
    CALIBRATED_MASK_THRESHOLDS, MaskFewShotMatcher, masked_context_crop,
)
from experiments.hybrid_pipeline.segmentation_models import (
    duplicate_proposal_pairs, mask_polygons, resize_mask,
)


class SegmentationModelTests(unittest.TestCase):
    def test_duplicate_proposals_are_flagged_without_suppression(self):
        proposals = [
            {"box": [10, 10, 30, 30]},
            {"box": [10, 10, 30, 30]},
            {"box": [25, 10, 45, 30]},
        ]
        self.assertEqual(len(proposals), 3)
        self.assertEqual(duplicate_proposal_pairs(proposals), [[1, 2]])

    def test_mask_resize_uses_nearest_neighbor_binary_values(self):
        mask = np.asarray([[0, 1], [1, 0]], dtype=bool)
        resized = resize_mask(mask, (4, 4))
        self.assertEqual(resized.shape, (4, 4))
        self.assertTrue(set(np.unique(resized)) <= {False, True})

    def test_mask_polygons_keeps_external_contour(self):
        mask = np.zeros((40, 50), dtype=bool)
        mask[5:30, 8:43] = True
        polygons = mask_polygons(mask)
        self.assertEqual(len(polygons), 1)
        self.assertGreaterEqual(len(polygons[0]), 4)

    def test_mask_context_crop_retains_real_context_ring(self):
        pixels = np.zeros((40, 40, 3), dtype=np.uint8)
        pixels[:] = (220, 30, 20)
        pixels[12:28, 12:28] = (20, 40, 220)
        mask = np.zeros((40, 40), dtype=bool)
        mask[12:28, 12:28] = True
        crop = masked_context_crop(Image.fromarray(pixels), mask, fraction=0.25)
        self.assertEqual(crop.getpixel((1, 1)), (220, 30, 20))
        self.assertEqual(crop.getpixel((crop.width // 2, crop.height // 2)), (20, 40, 220))

    def test_recognize_keeps_mask_for_review_overlay(self):
        matcher = object.__new__(MaskFewShotMatcher)
        matcher.match_crops = lambda crops: [{"sku": "SKU-1", "status": "recognized"}]
        mask = np.zeros((12, 14), dtype=bool)
        mask[2:9, 4:11] = True
        image = Image.new("RGB", (14, 12), "white")
        result = matcher.recognize(image, [{"mask": mask, "box": [4, 2, 11, 9]}])[0]
        self.assertIs(result["mask"], mask)
        self.assertEqual(result["sku"], "SKU-1")

    def test_crop_ablation_uses_working_resolution_photo(self):
        working = Image.new("RGB", (20, 12), (220, 30, 20))
        original = Image.new("RGB", (200, 120), "white")
        mask = np.zeros((12, 20), dtype=bool)
        mask[3:9, 5:15] = True
        row = {"photo": original, "photo_working": working, "mask": mask}
        crop = crop_for_query(row, "polygon_masked_context")
        self.assertLessEqual(crop.width, working.width)
        self.assertLessEqual(crop.height, working.height)

    def test_mask_matcher_cutoffs_come_from_calibration(self):
        self.assertEqual(CALIBRATED_MASK_THRESHOLDS[frozenset({"siglip2"})], (0.55, 0.05))
        self.assertEqual(CALIBRATED_MASK_THRESHOLDS[frozenset({"siglip2", "mobilenetv2"})],
                         (0.65, 0.05))

    def test_predicted_masks_score_correct_sku_separately_from_mask_match(self):
        mask = np.zeros((12, 14), dtype=bool)
        mask[2:9, 4:11] = True
        target = {"annotation_id": 7, "sku": "SKU-1", "mask": mask}
        predicted = [{"mask": mask}]
        matches = match_masks(predicted, [mask])
        row = {"sku": "SKU-1", "status": "recognized", "similarity": 0.9,
               "margin": 0.5, "ranked_scores": [{"sku": "SKU-1", "similarity": 0.9}],
               "branch_scores": {"mobilenetv2": {"SKU-1": 0.9},
                                 "siglip2": {"SKU-1": 0.9}}}
        result = score_instances([target], [row], matches)
        self.assertEqual(result["metrics"]["siglip2"]["registered"]["correct"], 1)
        self.assertEqual(result["metrics"]["siglip2+mobilenetv2"]["registered"]["correct"], 1)
        self.assertEqual(result["metrics"]["siglip2"]["registered"]["not_detected"], 0)


if __name__ == "__main__":
    unittest.main()
