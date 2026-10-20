"""Focused regression tests for the isolated new-product workflow."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate_scene_predictions as scene_eval
from experiment_io import load_manifest, load_registry


class DatasetAuditTests(unittest.TestCase):
    def test_audit_counts_and_reference_mapping(self):
        rows = load_manifest()
        self.assertEqual(len(rows), 72)
        self.assertEqual({group: sum(row["group"] == group for row in rows)
                          for group in ("owner", "employee", "customer")},
                         {"owner": 42, "employee": 25, "customer": 5})
        registry = load_registry()
        self.assertEqual(len(registry), 10)
        self.assertEqual(sum(len(item["reference_files"]) for item in registry), 42)


class SceneScoringTests(unittest.TestCase):
    def test_iou_and_split_unknown_duplicate_metrics(self):
        self.assertAlmostEqual(scene_eval.box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 1 / 7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "predictions.csv"
            fields = ["photo_filename", "detection_id", "sku_id", "confidence", "x1", "y1", "x2", "y2"]
            rows = [
                ["employee.jpeg", "e1", "SKU-A", 0.9, 0, 0, 10, 10],
                ["employee.jpeg", "e2", "SKU-A", 0.8, 0, 0, 10, 10],
                ["employee.jpeg", "e3", "SKU-WRONG", 0.7, 20, 20, 30, 30],
                ["customer.jpeg", "c1", "__unknown__", 0.6, 0, 0, 10, 10],
            ]
            with predictions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            labels = {
                "employee.jpeg": {"review_complete": True, "dataset_type": "employee", "arrangement_group": "single",
                                  "instances": [
                                      {"instance_id": "a", "sku_id": "SKU-A", "box_xyxy": [0, 0, 10, 10]},
                                      {"instance_id": "b", "sku_id": "SKU-B", "box_xyxy": [20, 20, 30, 30]},
                                  ]},
                "customer.jpeg": {"review_complete": True, "dataset_type": "customer", "arrangement_group": "single",
                                  "instances": [
                                      {"instance_id": "u", "sku_id": "__unknown__", "box_xyxy": [0, 0, 10, 10]},
                                  ]},
            }
            with patch.object(scene_eval, "load_scene_labels", return_value=labels), \
                 patch.object(scene_eval, "RESULTS", root / "results"):
                report = scene_eval.evaluate(predictions)
            self.assertEqual((report["true_positives"], report["false_positives"], report["false_negatives"]), (3, 1, 0))
            self.assertEqual(report["duplicate_detections"], 1)
            self.assertEqual((report["sku_correct"], report["sku_incorrect"], report["sku_unknown"]), (1, 1, 1))
            self.assertEqual(report["unknown_rejection_recall"], 1.0)
            self.assertEqual(set(report["metrics_by_capture_type"]), {"employee", "customer"})


if __name__ == "__main__":
    unittest.main()

