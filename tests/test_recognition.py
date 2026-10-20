"""Instance separation, ambiguity rejection, and real-mask rendering contracts."""

import unittest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from sam_vision import (SAMRecognizer, aggregate_reference_scores, classify_scores, decode_mask,
                        encode_mask, prune_instances, render_detections)
from images import product_references
from vision import duplicate_overlap, plausible_inventory_box


class RecognitionTests(unittest.TestCase):
    products = [{"sku": "A", "name": "Product A"}, {"sku": "B", "name": "Product B"}]

    def test_close_scores_remain_unassigned(self):
        result = classify_scores([0.91, 0.89], self.products)
        self.assertIsNone(result["sku"])
        self.assertEqual(result["reason"], "ambiguous_references")
        self.assertEqual(len(result["matches"]), 2)

    def test_weak_best_match_remains_unassigned(self):
        result = classify_scores([0.70, 0.4], self.products)
        self.assertIsNone(result["sku"])
        self.assertEqual(result["reason"], "below_similarity_threshold")

    def test_strong_separated_match_maps_to_its_sku(self):
        result = classify_scores([0.4, 0.94], self.products)
        self.assertEqual(result["sku"], "B")
        self.assertEqual(result["score_type"], "cosine_similarity")
        self.assertAlmostEqual(result["match_margin"], 0.54)

    def test_single_reference_still_requires_similarity(self):
        self.assertIsNone(classify_scores([0.4], self.products[:1])["sku"])
        self.assertEqual(classify_scores([0.95], self.products[:1])["sku"], "A")

    def test_nonfinite_match_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            classify_scores([float("nan"), 0.9], self.products)

    def test_best_view_is_used_and_margin_compares_different_skus(self):
        scores, winners = aggregate_reference_scores(np.array([[0.70, 0.94, 0.93, 0.60]]), [0, 0, 0, 1], 2)
        result = classify_scores(scores[0], self.products)
        self.assertEqual(result["sku"], "A")
        self.assertAlmostEqual(result["match_margin"], 0.34)
        self.assertEqual(winners[0, 0], 1)

    def test_extra_views_never_average_away_a_good_match_or_add_votes(self):
        scores, _ = aggregate_reference_scores(np.array([[0.91, 0.3, 0.3, 0.88]]), [0, 0, 0, 1], 2)
        np.testing.assert_allclose(scores, [[0.91, 0.88]])
        self.assertIsNone(classify_scores(scores[0], self.products)["sku"])

    def test_multiview_weak_or_ambiguous_matches_stay_unknown(self):
        scores, _ = aggregate_reference_scores(np.array([[0.6, 0.7, 0.5], [0.94, 0.95, 0.94]]), [0, 0, 1], 2)
        self.assertEqual(classify_scores(scores[0], self.products)["reason"], "below_similarity_threshold")
        self.assertEqual(classify_scores(scores[1], self.products)["reason"], "ambiguous_references")

    def test_reference_aggregation_handles_empty_scene_and_rejects_invalid_mapping(self):
        scores, winners = aggregate_reference_scores(np.empty((0, 3)), [0, 0, 1], 2)
        self.assertEqual(scores.shape, (0, 2))
        self.assertEqual(winners.shape, (0, 2))
        with self.assertRaises(ValueError):
            aggregate_reference_scores([[0.9, 0.4]], [0, 0], 2)
        with self.assertRaises(ValueError):
            aggregate_reference_scores([[float("nan")]], [0], 1)

    def test_single_reference_catalogue_is_backward_compatible(self):
        legacy = {"sku": "A", "image_path": "original.png"}
        self.assertEqual(product_references(legacy), [{"id": None, "image_path": "original.png"}])
        scores, winners = aggregate_reference_scores([[0.6, 0.94]], [0, 1], 2)
        self.assertEqual(classify_scores(scores[0], self.products)["sku"], "B")
        np.testing.assert_array_equal(winners, [[0, 1]])

    def test_recognizer_uses_all_views_but_returns_one_detection_and_provenance(self):
        from threading import RLock
        products = [
            {"sku": "A", "name": "A", "image_path": "a1", "references": [
                {"id": 1, "image_path": "a1"}, {"id": 2, "image_path": "a2"}]},
            {"sku": "B", "name": "B", "image_path": "b1"},
        ]
        instance = SAMRecognizer.__new__(SAMRecognizer)
        instance.lock = RLock()
        instance.reference = Mock(side_effect=[
            (np.array([0.7, np.sqrt(1 - 0.7**2)]), Image.new("RGB", (30, 50), "red")),
            (np.array([0.94, np.sqrt(1 - 0.94**2)]), Image.new("RGB", (30, 50), "blue")),
            (np.array([0.5, np.sqrt(1 - 0.5**2)]), Image.new("RGB", (30, 50), "green")),
        ])
        mask = np.zeros((100, 80), dtype=bool)
        mask[10:90, 10:60] = True
        instance.segment = Mock(return_value=([mask], [0.95]))
        instance.encoder = Mock()
        instance.encoder.encode.return_value = np.array([[1.0, 0.0]])
        progress = Mock()
        detections, annotated, board, raw = instance.recognize(
            Image.new("RGB", (80, 100), "white"), products, progress=progress
        )
        self.assertEqual(instance.reference.call_count, 3)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["sku"], "A")
        self.assertEqual(raw[0]["matches"][0]["reference_id"], 2)
        self.assertEqual(raw[0]["matches"][0]["reference_image"], "a2")
        self.assertEqual(board.size, (960, 320))
        self.assertEqual(annotated.size, (80, 100))
        events = [call.args[0] for call in progress.call_args_list]
        self.assertEqual([event["completed"] for event in events if event["stage"] == "references"], [0, 1, 2, 3])
        self.assertEqual([event["stage"] for event in events[-3:]], ["photo", "matching", "rendering"])
        self.assertEqual(events[1]["view"], 2)
        self.assertEqual(set(instance.last_timings), {"references_seconds", "photo_segmentation_seconds",
                                                     "matching_seconds", "rendering_seconds", "total_seconds"})
        self.assertTrue(all(seconds >= 0 for seconds in instance.last_timings.values()))

    def test_mask_encoding_roundtrip_and_original_resolution(self):
        mask = np.zeros((37, 29), dtype=bool)
        mask[8:30, 11:20] = True
        np.testing.assert_array_equal(decode_mask(encode_mask(mask)), mask)
        self.assertEqual(decode_mask(encode_mask(mask), (58, 74)).shape, (74, 58))

    def test_ambiguous_recognizer_checks_close_skus_beyond_the_displayed_top_three(self):
        from threading import RLock
        products = [{"sku": sku, "name": sku, "image_path": sku + ".png"} for sku in ("A", "B", "C", "D")]
        instance = SAMRecognizer.__new__(SAMRecognizer)
        instance.lock = RLock()
        instance.reference = Mock(side_effect=[
            (np.array([score, np.sqrt(1 - score**2)]), Image.new("RGB", (30, 50), "red"))
            for score in (0.94, 0.93, 0.92, 0.91)
        ])
        mask = np.zeros((100, 80), dtype=bool)
        mask[10:90, 10:60] = True
        instance.segment = Mock(return_value=([mask], [0.95]))
        instance.encoder = Mock()
        instance.encoder.encode.return_value = np.array([[1.0, 0.0]])
        instance.label_verifier = Mock()
        instance.label_verifier.compare.return_value = [
            {"sku": sku, "verified": sku in ("A", "D")} for sku in ("A", "B", "C", "D")
        ]
        detections, _, _, raw = instance.recognize(Image.new("RGB", (80, 100), "white"), products)
        compared = instance.label_verifier.compare.call_args.args[1]
        self.assertEqual({reference["sku"] for reference, crop in compared}, {"A", "B", "C", "D"})
        self.assertIsNone(detections[0]["sku"])
        self.assertEqual(raw[0]["geometric_verification"]["candidate_skus"], ["A", "B", "C", "D"])

    def test_spotlight_preserves_only_mask_pixels_not_entire_box(self):
        image = Image.new("RGB", (40, 40), (200, 150, 100))
        mask = np.zeros((40, 40), dtype=bool)
        mask[20:35, 20:35] = True
        mask[25:30, 25:30] = False
        detection = {"sku": "A", "name": "A", "box": [20, 20, 35, 35], "mask": encode_mask(mask)}
        spotlight, kind = render_detections(image, [detection], spotlight=True)
        self.assertEqual(kind, "instance_mask")
        self.assertEqual(spotlight.size, image.size)
        self.assertEqual(spotlight.getpixel((23, 28)), image.getpixel((23, 28)))
        self.assertEqual(spotlight.getpixel((27, 27)), (60, 45, 30))

    def test_uncertain_mask_is_not_brightened(self):
        image = Image.new("RGB", (60, 80), (200, 150, 100))
        mask = np.zeros((80, 60), dtype=bool)
        mask[30:70, 20:50] = True
        detection = {"sku": None, "name": "Uncertain product", "box": [20, 30, 50, 70], "mask": encode_mask(mask)}
        spotlight, _ = render_detections(image, [detection], spotlight=True)
        self.assertEqual(spotlight.getpixel((35, 50)), (60, 45, 30))

    def test_legacy_spotlight_is_explicitly_rectangular(self):
        image = Image.new("RGB", (40, 40), "white")
        detection = {"sku": "A", "name": "A", "box": [20, 20, 35, 35]}
        _, kind = render_detections(image, [detection], spotlight=True)
        self.assertEqual(kind, "bounding_box")

    @staticmethod
    def record(mask, sku="A"):
        return {"sku": sku, "name": sku, "status": "recognized", "mask_quality": 0.95}

    def test_tall_adjacent_products_are_not_merged(self):
        self.assertFalse(duplicate_overlap([0, 0, 20, 100], [22, 0, 42, 100]))
        masks = []
        for left in (10, 32):
            mask = np.zeros((120, 60), dtype=bool)
            mask[10:110, left:left + 20] = True
            masks.append(mask)
        kept = prune_instances([self.record(mask) for mask in masks], masks)
        self.assertEqual(len(kept), 2)

    def test_closeup_object_is_not_rejected_due_to_area(self):
        self.assertTrue(plausible_inventory_box({"box": [10, 10, 90, 90]}, (100, 100), 3))

    def test_reference_selector_rejects_larger_table_and_keeps_full_product(self):
        table = np.zeros((120, 100), dtype=bool)
        table[80:, :99] = True
        bottle = np.zeros_like(table)
        bottle[10:110, 40:60] = True
        table[bottle] = False
        cap = np.zeros_like(table)
        cap[10:25, 40:60] = True
        self.assertGreater(table.sum(), bottle.sum())
        self.assertLess(table.mean(), 0.5)
        self.assertEqual(SAMRecognizer.rejection(table), "background_or_scene_mask")
        self.assertEqual(SAMRecognizer.select_reference_mask([table, cap, bottle], [1.0, 0.99, 0.9]), 2)

    def test_partial_table_with_short_contact_on_third_edge_is_background(self):
        table = np.zeros((100, 100), dtype=bool)
        table[60:, :40] = True
        table[88:, :] = True
        self.assertLess(table.mean(), 0.5)
        self.assertEqual(SAMRecognizer.rejection(table), "background_or_scene_mask")

    def test_large_closeup_and_two_edge_cropped_product_remain_valid(self):
        bounded = np.zeros((100, 100), dtype=bool)
        bounded[5:95, 5:95] = True
        cropped = np.zeros_like(bounded)
        cropped[:, 10:90] = True
        self.assertIsNone(SAMRecognizer.rejection(bounded))
        self.assertIsNone(SAMRecognizer.rejection(cropped))

    def test_reference_policy_rejects_two_edge_walls_and_floors_but_not_scene_closeups(self):
        bottle = np.zeros((120, 100), dtype=bool)
        bottle[10:110, 40:60] = True
        cap = np.zeros_like(bottle)
        cap[10:25, 40:60] = True
        wall = np.zeros_like(bottle)
        wall[:80, 20:] = True
        wall[bottle] = False
        floor = np.zeros_like(bottle)
        floor[80:, 20:] = True
        floor[bottle] = False
        for background in (wall, floor):
            with self.subTest(background=background.sum()):
                self.assertIsNone(SAMRecognizer.rejection(background))
                self.assertEqual(SAMRecognizer.reference_rejection(background), "reference_touches_frame")
                self.assertEqual(SAMRecognizer.select_reference_mask([background, cap, bottle], [1.0, 0.99, 0.9]), 2)

    def test_reference_policy_rejects_one_edge_background_strip_and_near_frame_gaps(self):
        strip = np.zeros((120, 100), dtype=bool)
        strip[20:100, 1:20] = True
        self.assertIsNone(SAMRecognizer.rejection(strip))
        self.assertEqual(SAMRecognizer.reference_rejection(strip), "reference_touches_frame")
        with self.assertRaisesRegex(ValueError, "No isolated product"):
            SAMRecognizer.select_reference_mask([strip], [1.0])

    def test_strict_reference_policy_keeps_wide_products_and_stray_edge_pixels(self):
        package = np.zeros((100, 120), dtype=bool)
        package[35:65, 10:110] = True
        package[0, 0] = package[-1, 0] = package[-1, -1] = True
        self.assertIsNone(SAMRecognizer.reference_rejection(package))
        self.assertEqual(SAMRecognizer.select_reference_mask([package], [0.9]), 0)

    def test_few_stray_border_pixels_do_not_discard_a_product(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[15:85, 30:70] = True
        mask[0, 0] = mask[-1, 0] = mask[-1, -1] = True
        self.assertIsNone(SAMRecognizer.rejection(mask))

    def test_reference_selector_refuses_scene_only_or_misaligned_results(self):
        table = np.zeros((100, 100), dtype=bool)
        table[70:] = True
        with self.assertRaisesRegex(ValueError, "No isolated product"):
            SAMRecognizer.select_reference_mask([table, np.ones_like(table)], [0.99, 1.0])
        with self.assertRaises(ValueError):
            SAMRecognizer.select_reference_mask([table], [])

    def test_background_candidate_cannot_be_assigned_even_with_a_strong_encoder(self):
        from threading import RLock
        table = np.zeros((100, 100), dtype=bool)
        table[70:] = True
        instance = SAMRecognizer.__new__(SAMRecognizer)
        instance.lock = RLock()
        instance.reference = Mock(return_value=(np.ones(512), Image.new("RGB", (30, 50), "red")))
        instance.segment = Mock(return_value=([table], [1.0]))
        instance.encoder = Mock()
        instance.encoder.encode.return_value = np.empty((0, 512))
        products = [{"sku": "A", "name": "A", "image_path": "a.png"}]
        detections, _, _, raw = instance.recognize(Image.new("RGB", (100, 100), "white"), products)
        self.assertEqual(detections, [])
        self.assertEqual(raw[0]["reason"], "background_or_scene_mask")
        self.assertIsNone(raw[0]["sku"])
        self.assertEqual(instance.encoder.encode.call_args.args[0], [])

    def test_cap_mask_does_not_add_another_product(self):
        whole = np.zeros((100, 60), dtype=bool)
        whole[10:90, 10:50] = True
        cap = np.zeros_like(whole)
        cap[10:30, 10:50] = True
        records = [self.record(whole), self.record(cap)]
        self.assertEqual(len(prune_instances(records, [whole, cap])), 1)
        self.assertEqual(records[1]["reason"], "duplicate_or_nested_mask")

    def test_parent_spanning_two_instances_does_not_count_a_third(self):
        first, second = np.zeros((100, 100), dtype=bool), np.zeros((100, 100), dtype=bool)
        first[10:90, 10:40] = True
        second[10:90, 50:80] = True
        parent = first | second
        masks = [parent, first, second]
        records = [self.record(mask) for mask in masks]
        self.assertEqual(len(prune_instances(records, masks)), 2)
        self.assertEqual(records[0]["reason"], "mask_contains_multiple_instances")

    def test_reference_cache_survives_restart_and_detects_replaced_photo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.new("RGB", (50, 80), "red").save(path)
            product = {"sku": "A", "image_path": str(path)}
            mask = np.zeros((80, 50), dtype=bool)
            mask[10:70, 10:40] = True

            def recognizer():
                instance = SAMRecognizer.__new__(SAMRecognizer)
                instance.references = {}
                instance.reference_dir = Path(directory) / "cache"
                instance.segment = Mock(return_value=([mask], [0.95]))
                instance.encoder = Mock()
                instance.encoder.encode.return_value = np.ones((1, 512), dtype=np.float32)
                return instance

            first = recognizer()
            _, red_crop = first.reference(product)
            reloaded = recognizer()
            reloaded.reference(product)
            reloaded.segment.assert_not_called()
            reloaded.encoder.encode.assert_not_called()
            Image.new("RGB", (50, 80), "blue").save(path)
            changed = recognizer()
            _, blue_crop = changed.reference(product)
            changed.segment.assert_called_once()
            self.assertNotEqual(red_crop.getpixel((15, 20)), blue_crop.getpixel((15, 20)))

    def test_reference_selection_version_ignores_legacy_bad_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "reference.png"
            Image.new("RGB", (80, 100), "red").save(path)
            cache = root / "cache"
            cache.mkdir()
            legacy = cache / (hashlib.sha256(b"mobile-sam12-resnet18-v1" + path.read_bytes()).hexdigest() + ".npz")
            np.savez_compressed(legacy, feature=np.ones(512), crop=np.asarray(Image.new("RGB", (70, 20), "blue")))
            instance = SAMRecognizer.__new__(SAMRecognizer)
            stat = path.stat()
            instance.references = {(str(path.resolve()), stat.st_mtime_ns, stat.st_size):
                                   (np.ones(512), Image.new("RGB", (70, 20), "blue"))}
            instance.reference_dir = cache
            bottle = np.zeros((100, 80), dtype=bool)
            bottle[5:95, 30:50] = True
            table = np.zeros_like(bottle)
            table[65:] = True
            table[bottle] = False
            instance.segment = Mock(return_value=([table, bottle], [1.0, 0.95]))
            instance.encoder = Mock()
            instance.encoder.encode.return_value = np.ones((1, 512), dtype=np.float32)
            _, crop = instance.reference({"sku": "A", "image_path": str(path)})
            instance.segment.assert_called_once()
            self.assertEqual(crop.size, (20, 90))
            self.assertEqual(crop.getpixel((10, 40)), (255, 0, 0))
            self.assertEqual(len(list(cache.glob("*.npz"))), 2)
            with np.load(legacy, allow_pickle=False) as original:
                self.assertEqual(original["crop"].shape, (20, 70, 3))

    def test_v2_cardboard_cache_is_not_reused_from_disk_or_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "reference.png"
            Image.new("RGB", (80, 100), "red").save(path)
            version = "mobile-sam12-resnet18-v2-border-foreground"
            cache = root / "cache"
            cache.mkdir()
            legacy = cache / (hashlib.sha256(version.encode("ascii") + path.read_bytes()).hexdigest() + ".npz")
            bad_crop = Image.new("RGB", (70, 20), "blue")
            np.savez_compressed(legacy, feature=np.ones(512), crop=np.asarray(bad_crop))
            instance = SAMRecognizer.__new__(SAMRecognizer)
            stat = path.stat()
            instance.references = {(version, str(path.resolve()), stat.st_mtime_ns, stat.st_size):
                                   (np.ones(512), bad_crop)}
            instance.reference_dir = cache
            bottle = np.zeros((100, 80), dtype=bool)
            bottle[5:95, 30:50] = True
            wall = np.zeros_like(bottle)
            wall[:70, 20:] = True
            wall[bottle] = False
            instance.segment = Mock(return_value=([wall, bottle], [1.0, 0.95]))
            instance.encoder = Mock()
            instance.encoder.encode.return_value = np.ones((1, 512), dtype=np.float32)
            _, crop = instance.reference({"sku": "A", "image_path": str(path)})
            instance.segment.assert_called_once()
            self.assertEqual(crop.size, (20, 90))
            self.assertEqual(crop.getpixel((10, 40)), (255, 0, 0))
            self.assertTrue(legacy.is_file())

    def test_background_only_reference_errors_without_caching_or_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "reference.png"
            Image.new("RGB", (80, 100), "red").save(path)
            instance = SAMRecognizer.__new__(SAMRecognizer)
            instance.references = {}
            instance.reference_dir = root / "cache"
            wall = np.zeros((100, 80), dtype=bool)
            wall[:70, 20:] = True
            instance.segment = Mock(return_value=([wall], [1.0]))
            instance.encoder = Mock()
            with self.assertRaisesRegex(ValueError, "background visible on every side"):
                instance.reference({"sku": "A", "image_path": str(path)})
            instance.encoder.encode.assert_not_called()
            self.assertEqual(instance.references, {})
            self.assertFalse(instance.reference_dir.exists())


if __name__ == "__main__":
    unittest.main()
