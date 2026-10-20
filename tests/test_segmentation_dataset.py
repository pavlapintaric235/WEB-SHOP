import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from tools.segmentation_dataset import (
    export_dataset,
    import_cvat,
    parse_yolo_label,
    polygon_self_intersections,
    validate_dataset,
)


class SegmentationDatasetTests(unittest.TestCase):
    def test_simple_polygon_has_no_crossings(self):
        points = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        self.assertEqual(polygon_self_intersections(points), [])

    def test_self_crossing_polygon_is_detected(self):
        points = [(0.1, 0.1), (0.9, 0.9), (0.1, 0.8), (0.9, 0.2), (0.7, 0.95)]
        self.assertTrue(polygon_self_intersections(points))

    def test_label_parser_accepts_empty_negative_file(self):
        self.assertEqual(parse_yolo_label("", "empty.txt"), [])

    def test_label_parser_rejects_out_of_range_coordinates(self):
        with self.assertRaisesRegex(ValueError, "normalized"):
            parse_yolo_label("0 -0.1 0 0.8 0 0.2 0.8", "bad.txt")

    def test_confirmed_negative_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "approved/images").mkdir(parents=True)
            (root / "approved/labels").mkdir(parents=True)
            Image.new("RGB", (20, 20), "white").save(root / "approved/images/empty.png")
            (root / "approved/labels/empty.txt").write_text("", encoding="utf-8")
            (root / "approved/review_manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "images": {"empty.png": {
                    "reviewed": True, "all_visible_products_reviewed": True,
                    "negative": True, "instances": 0, "group_id": "negative_01",
                }},
            }), encoding="utf-8")
            report = validate_dataset(root)
            self.assertEqual(report["images"], 1)
            self.assertEqual(report["approved_instances"], 0)
            self.assertEqual(report["errors"], 0)
            self.assertEqual(report["warnings"], 0)

    def test_unconfirmed_empty_image_is_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "approved/images").mkdir(parents=True)
            (root / "approved/labels").mkdir(parents=True)
            Image.new("RGB", (20, 20), "white").save(root / "approved/images/empty.png")
            (root / "approved/labels/empty.txt").write_text("", encoding="utf-8")
            (root / "approved/review_manifest.json").write_text('{"images": {}}', encoding="utf-8")
            report = validate_dataset(root)
            self.assertIn("empty_labels_not_confirmed_negative", {issue["code"] for issue in report["issues"]})

    def test_cvat_import_requires_review_and_maps_same_pixels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "proposals" / "batch01"
            (batch / "images").mkdir(parents=True)
            (batch / "labels").mkdir()
            image_path = batch / "images" / "scene.png"
            Image.new("RGB", (20, 20), "white").save(image_path)
            polygon = "0 0.1 0.1 0.9 0.1 0.5 0.9\n"
            archive_path = root / "reviewed.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data.yaml", "names:\n  0: product\n")
                archive.writestr("images/train/scene.png", image_path.read_bytes())
                archive.writestr("labels/train/scene.txt", polygon)

            with self.assertRaisesRegex(ValueError, "--reviewed"):
                import_cvat(archive_path, "batch01", dataset=root)
            imported = import_cvat(archive_path, "batch01", reviewed=True, dataset=root)
            self.assertEqual(len(imported["scene.png"][1]), 1)
            self.assertEqual((root / "approved/labels/scene.txt").read_text(), polygon)
            report = validate_dataset(root)
            self.assertEqual(report["errors"], 0)

    def test_export_contains_only_approved_reviewed_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir, label_dir = root / "approved/images", root / "approved/labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir()
            Image.new("RGB", (20, 20), "white").save(image_dir / "scene.png")
            (label_dir / "scene.txt").write_text("0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8")
            (root / "approved/review_manifest.json").write_text(json.dumps({
                "images": {"scene.png": {
                    "reviewed": True, "all_visible_products_reviewed": True,
                    "negative": False, "instances": 1, "group_id": "layout1",
                }},
            }), encoding="utf-8")
            output = export_dataset(root, "export01")
            with zipfile.ZipFile(output / "ultralytics_yolo_segmentation.zip") as archive:
                self.assertIn("images/train/scene.png", archive.namelist())
                self.assertIn("labels/train/scene.txt", archive.namelist())
                self.assertEqual(archive.read("labels/train/scene.txt").decode().split(),
                                 "0 0.1 0.1 0.9 0.1 0.5 0.9".split())


if __name__ == "__main__":
    unittest.main()
