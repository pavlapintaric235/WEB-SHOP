from __future__ import annotations

import unittest

from dataset_utils import convert_segmentation_text, parse_segmentation_row, polygon_to_detection
from evaluate_detectors import box_iou, score_iou


class PolygonConversionTests(unittest.TestCase):
    def test_polygon_becomes_axis_aligned_yolo_box(self):
        points = parse_segmentation_row("0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8")
        self.assertEqual(polygon_to_detection(points), (0.3, 0.5, 0.4, 0.6000000000000001))

    def test_self_intersection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Self-intersecting"):
            parse_segmentation_row("0 0 0 1 1 0 1 0.8 0.2")

    def test_exact_duplicate_polygon_is_rejected_even_if_rotated(self):
        text = "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n0 0.8 0.8 0.8 0.1 0.1 0.1 0.1 0.8"
        with self.assertRaisesRegex(ValueError, "Exact duplicate"):
            convert_segmentation_text(text)

    def test_detection_conversion_uses_extrema_not_polygon_sequence_as_box(self):
        converted, count = convert_segmentation_text("0 0.1 0.1 0.5 0.2 0.4 0.8 0.2 0.6")
        self.assertEqual(count, 1)
        self.assertEqual(list(map(float, converted.split()[1:])), [0.3, 0.45, 0.4, 0.7])


class DetectionMetricTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0.0)

    def test_greedy_iou_metrics(self):
        predictions = [
            {"xyxy": [0, 0, 10, 10], "confidence": 0.9},
            {"xyxy": [20, 20, 30, 30], "confidence": 0.8},
        ]
        metrics = score_iou(predictions, [[0, 0, 10, 10], [40, 40, 50, 50]])
        self.assertEqual((metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]), (1, 1, 1))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
