"""The diagnostic evaluator keeps unregistered products in detection ground truth."""

import unittest
from collections import Counter
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from tools.evaluate_oct17_benchmark import (annotation_sku, load_coco, matcher_audit,
                                           render_markdown, render_matcher_preview, score_arm, summarize,
                                           wilson_interval)


class Oct17BenchmarkEvaluationTests(unittest.TestCase):
    def test_cvat_product_attribute_accepts_explicit_unregistered_sku(self):
        annotation = {"id": 12, "attributes": {"sku": "__undefined__"}}
        self.assertEqual(annotation_sku(annotation, "product"), "__undefined__")

    def test_missing_or_empty_sku_is_not_silently_called_unregistered(self):
        for attributes in ({}, {"sku": ""}, {"sku": "   "}):
            with self.subTest(attributes=attributes), self.assertRaisesRegex(ValueError, "missing a selected SKU"):
                annotation_sku({"id": 13, "attributes": attributes}, "product")

    def test_coco_loader_preserves_each_product_instances_sku_attribute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), "white").save(image_dir / "photo.png")
            manifest_path = root / "instances_default.json"
            manifest_path.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "photo.png", "width": 20, "height": 20}],
                "categories": [{"id": 1, "name": "product"}],
                "annotations": [
                    {"id": 4, "image_id": 1, "category_id": 1,
                     "segmentation": [[1, 1, 10, 1, 10, 10, 1, 10]],
                     "attributes": {"sku": "REGISTERED-1"}},
                    {"id": 5, "image_id": 1, "category_id": 1,
                     "segmentation": [[11, 11, 18, 11, 18, 18, 11, 18]],
                     "attributes": {"sku": "__undefined__"}},
                ],
            }), encoding="utf-8")

            loaded = load_coco(manifest_path, image_dir)["photo.png"]

        self.assertEqual([item["sku"] for item in loaded], ["REGISTERED-1", "__undefined__"])
        self.assertEqual([item["annotation_id"] for item in loaded], [4, 5])

    def test_coco_loader_accepts_image_border_and_rejects_coordinates_beyond_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (20, 20), "white").save(image_dir / "photo.png")
            manifest_path = root / "instances_default.json"
            coco = {
                "images": [{"id": 1, "file_name": "photo.png", "width": 20, "height": 20}],
                "categories": [{"id": 1, "name": "product"}],
                "annotations": [{"id": 7, "image_id": 1, "category_id": 1,
                                 "segmentation": [[1, 1, 20, 1, 20, 20, 1, 20]],
                                 "attributes": {"sku": "__undefined__"}}],
            }
            manifest_path.write_text(json.dumps(coco), encoding="utf-8")
            self.assertEqual(len(load_coco(manifest_path, image_dir)["photo.png"]), 1)

            coco["annotations"][0]["segmentation"] = [[1, 1, 20.01, 1, 20, 20, 1, 20]]
            manifest_path.write_text(json.dumps(coco), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Out-of-frame polygon"):
                load_coco(manifest_path, image_dir)

    def test_unregistered_product_is_detection_positive_and_unknown_abstention_is_correct(self):
        mask = np.ones((8, 8), dtype=bool)
        result = score_arm(
            "run", "photo.png", "mobile_sam", [{"sku": "__undefined__", "mask": mask}],
            [{"sku": None, "mask": mask, "status": "unknown"}], 0.1, {}, Counter(), 0.2, True,
        )

        self.assertEqual(result["detection_metrics"]["recall_iou50"], 1.0)
        self.assertEqual(result["sku_metrics"]["unregistered_ground_truth"], 1)
        self.assertEqual(result["sku_metrics"]["correct_unknown_abstentions"], 1)
        self.assertEqual(result["sku_metrics"]["false_sku_assignments_on_unregistered"], 0)

    def test_registered_sku_assignment_to_unregistered_product_is_identity_error(self):
        mask = np.ones((8, 8), dtype=bool)
        result = score_arm(
            "run", "photo.png", "mobile_sam", [{"sku": "__undefined__", "mask": mask}],
            [{"sku": "REGISTERED-1", "mask": mask, "status": "matched"}], 0.1, {}, Counter(), 0.2, True,
        )

        self.assertEqual(result["detection_metrics"]["recall_iou50"], 1.0)
        self.assertEqual(result["sku_metrics"]["correct_unknown_abstentions"], 0)
        self.assertEqual(result["sku_metrics"]["false_sku_assignments_on_unregistered"], 1)
        self.assertEqual(result["sku_metrics"]["wrong"], 1)

    def test_recognized_instance_removed_by_filter_is_not_reported_as_missed_detection(self):
        mask = np.ones((8, 8), dtype=bool)
        result = score_arm(
            "run", "photo.png", "mobile_sam", [{"sku": "REGISTERED-1", "mask": mask,
                                                  "annotation_id": 33}],
            [], 0.1, {}, Counter(), 0.2, True,
            raw_predictions=[{"mask_array": mask, "sku": "REGISTERED-1", "status": "recognized",
                              "reason": "reference_match", "matches": []}],
            post_filter_predictions=[{"mask_array": mask, "sku": None, "status": "rejected",
                                      "reason": "duplicate_or_nested_mask"}],
        )

        self.assertEqual(result["detection_metrics"]["recall_iou50"], 0.0)
        self.assertEqual(result["objects"][0]["outcome"], "recognized_then_removed_by_filter")
        self.assertEqual(result["objects"][0]["reason_after_filter"], "duplicate_or_nested_mask")

    def test_wilson_interval_reports_small_sample_warning(self):
        result = wilson_interval(8, 10)
        self.assertEqual((result["successes"], result["trials"]), (8, 10))
        self.assertAlmostEqual(result["estimate"], 0.8)
        self.assertLess(result["wilson_95"][0], 0.8)
        self.assertGreater(result["wilson_95"][1], 0.8)
        self.assertIn("Small sample", result["warning"])

    def test_matcher_audit_keeps_reference_margin_and_geometry_evidence(self):
        mask = np.ones((8, 5), dtype=bool)
        prediction = {"sku": None, "status": "uncertain", "reason": "ambiguous_references",
                      "method": "mobile_sam_resnet18", "similarity": 0.82, "match_margin": 0.03,
                      "matches": [
                          {"sku": "SKU-A", "similarity": 0.82, "reference_image": "a.jpg", "reference_view": 1},
                          {"sku": "SKU-B", "similarity": 0.79, "reference_image": "b.jpg", "reference_view": 2},
                      ],
                      "sift_verification_attempted": True,
                      "geometric_verification": {"status": "unresolved"}}
        audit = matcher_audit(prediction, {"mask": mask, "attributes": {"occluded": True}})
        self.assertEqual(audit["best_candidate"]["sku"], "SKU-A")
        self.assertEqual(audit["runner_up_candidate"]["reference_image"], "b.jpg")
        self.assertAlmostEqual(audit["similarity_margin"], 0.03)
        self.assertTrue(audit["local_feature_verification"]["attempted"])
        self.assertEqual(audit["local_feature_verification"]["final_status"], "unresolved")
        self.assertTrue(audit["ground_truth_occluded_attribute"])
        self.assertAlmostEqual(audit["crop_quality"]["aspect_ratio"], 5 / 8)

    def test_summary_renders_confidence_intervals_for_small_diagnostic_sample(self):
        mask = np.ones((5, 5), dtype=bool)
        row = score_arm("run", "photo.png", "B_human_masks",
                        [{"sku": "SKU-1", "mask": mask, "annotation_id": 1}],
                        [{"sku": "SKU-1", "mask": mask, "status": "recognized"}],
                        0.0, {}, Counter(), 0.1, False)
        report = summarize([row], False, {
            "detection_iou_threshold": 0.5,
            "reviewed_coco": "corrected.zip",
            "human_mask_arm": "oracle masks",
        })
        rendered = render_markdown(report)
        self.assertEqual(report["arms"]["B_human_masks"]["sku_confidence_intervals_95"]["SKU-1"]
                         ["correct_identity_per_ground_truth"]["trials"], 1)
        self.assertIn("Wilson 95% intervals", rendered)
        self.assertIn("corrected.zip", rendered)

    def test_matcher_preview_uses_polygon_crop_and_reference_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "ref.png"
            Image.new("RGB", (12, 8), "blue").save(reference)
            output = root / "preview.png"
            target = {"sku": "SKU-1", "annotation_id": 9,
                      "polygons": [np.asarray([[3, 4], [15, 4], [15, 16], [3, 16]])],
                      "width": 20, "height": 20}
            prediction = {"matches": [
                {"sku": "SKU-2", "similarity": 0.8, "reference_image": str(reference), "reference_view": 1},
                {"sku": "SKU-3", "similarity": 0.7, "reference_image": str(reference), "reference_view": 2},
            ]}
            render_matcher_preview(Image.new("RGB", (20, 20), "white"), target, prediction, output)
            with Image.open(output) as preview:
                self.assertEqual(preview.size, (1440, 675))


if __name__ == "__main__":
    unittest.main()
