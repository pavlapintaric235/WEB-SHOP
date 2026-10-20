"""Geometric evidence is independent of global embedding similarity."""

import unittest

import cv2
import numpy as np
from PIL import Image

from local_features import (LocalFeatures, LabelVerifier, extract_features, resolve_ambiguity,
                            resolve_low_score_geometry, should_verify_low_score_geometry,
                            verify_geometry)
from sam_vision import classify_scores


class LocalFeatureTests(unittest.TestCase):
    products = [{"sku": "A", "name": "A"}, {"sku": "B", "name": "B"}]

    @staticmethod
    def texture(seed):
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, (360, 320, 3), dtype=np.uint8)
        return Image.fromarray(pixels)

    @staticmethod
    def evidence(sku, verified=True, view=1):
        return {"sku": sku, "verified": verified, "inliers": 35 if verified else 0,
                "inlier_ratio": 0.9, "reference_coverage": 0.3, "query_coverage": 0.3,
                "reference_id": view, "reference_view": view, "reference_image": sku + ".png"}

    def test_real_sift_verifies_a_perspective_transformed_label(self):
        source = self.texture(7)
        corners = np.float32([[0, 0], [319, 0], [319, 359], [0, 359]])
        target = np.float32([[30, 35], [340, 20], [355, 385], [15, 365]])
        transform = cv2.getPerspectiveTransform(corners, target)
        query = Image.fromarray(cv2.warpPerspective(np.asarray(source), transform, (380, 420),
                                                    borderValue=(124, 124, 124)))
        result = verify_geometry(extract_features(source), extract_features(query))
        self.assertTrue(result["verified"], result)
        self.assertGreaterEqual(result["inliers"], 12)
        self.assertGreater(result["query_coverage"], 0.08)

    def test_unrelated_and_featureless_images_are_not_verified(self):
        reference = extract_features(self.texture(7))
        for query in (self.texture(21), Image.new("RGB", (320, 360), (124, 124, 124))):
            with self.subTest(query=query.getpixel((0, 0))):
                self.assertFalse(verify_geometry(reference, extract_features(query))["verified"])

    def test_many_matches_in_a_tiny_patch_are_not_enough(self):
        points = np.float32([(x, y) for x in range(100, 140, 10) for y in range(100, 140, 10)])
        descriptors = np.eye(len(points), 128, dtype=np.float32)
        reference = LocalFeatures(points, descriptors, (640, 640))
        query = LocalFeatures(points + 30, descriptors.copy(), (640, 640))
        result = verify_geometry(reference, query)
        self.assertEqual(result["inliers"], 16)
        self.assertFalse(result["verified"])

    def test_reflected_and_collinear_correspondences_are_not_verified(self):
        points = np.float32([(x, y) for x in (60, 230, 400, 570) for y in (60, 230, 400, 570)])
        descriptors = np.eye(len(points), 128, dtype=np.float32)
        reference = LocalFeatures(points, descriptors, (640, 640))
        mirrored = points.copy()
        mirrored[:, 0] = 640 - mirrored[:, 0]
        self.assertFalse(verify_geometry(reference, LocalFeatures(mirrored, descriptors, (640, 640)))["verified"])
        line = np.float32([(x, x) for x in range(10, 170, 10)])
        self.assertFalse(verify_geometry(LocalFeatures(line, descriptors, (640, 640)),
                                         LocalFeatures(line + 10, descriptors, (640, 640)))["verified"])

    def test_verifier_caches_references_by_crop_content_not_name(self):
        verifier = LabelVerifier()
        reference = {"sku": "A", "reference_id": 1, "view": 1, "image_path": "a.png"}
        image = self.texture(7)
        self.assertTrue(verifier.compare(image, [(reference, image)])[0]["verified"])
        self.assertEqual(len(verifier.references), 1)
        verifier.compare(image, [(reference, image.copy())])
        self.assertEqual(len(verifier.references), 1)
        verifier.compare(image, [(reference, self.texture(21))])
        self.assertEqual(len(verifier.references), 2)

    def test_only_unambiguous_geometric_agreement_resolves_a_close_embedding(self):
        record = classify_scores([0.91, 0.89], self.products)
        resolve_ambiguity(record, [self.evidence("A", view=2), self.evidence("B", False)])
        self.assertEqual(record["sku"], "A")
        self.assertEqual(record["reason"], "geometric_reference_match")
        self.assertEqual(record["embedding_reason"], "ambiguous_references")
        self.assertAlmostEqual(record["confidence"], 0.91)
        self.assertAlmostEqual(record["match_margin"], 0.02)
        self.assertEqual(record["geometric_verification"]["reference_view"], 2)

    def test_two_verified_skus_stay_unknown_even_if_one_has_more_views(self):
        record = classify_scores([0.91, 0.89], self.products)
        evidence = [self.evidence("A", view=1), self.evidence("A", view=2), self.evidence("B")]
        resolve_ambiguity(record, evidence)
        self.assertIsNone(record["sku"])
        self.assertEqual(record["geometric_verification"]["status"], "unresolved")

    def test_geometry_disagreeing_with_embedding_or_finding_no_label_abstains(self):
        for first, second in ((False, True), (False, False)):
            record = classify_scores([0.91, 0.89], self.products)
            resolve_ambiguity(record, [self.evidence("A", first), self.evidence("B", second)])
            self.assertIsNone(record["sku"])

    def test_strong_label_evidence_does_not_override_a_weak_embedding(self):
        record = classify_scores([0.79, 0.77], self.products)
        resolve_ambiguity(record, [self.evidence("A"), self.evidence("B", False)])
        self.assertIsNone(record["sku"])
        self.assertEqual(record["reason"], "below_similarity_threshold")

    def test_unique_geometry_can_recover_a_borderline_match_across_all_skus(self):
        record = classify_scores([0.73, 0.65], self.products)
        record["mask_area_ratio"] = 0.02
        evidence = [self.evidence("A", False), self.evidence("B")]
        resolve_low_score_geometry(record, evidence, ["A", "B"])
        self.assertEqual(record["sku"], "B")
        self.assertEqual(record["reason"], "geometric_reference_match")
        self.assertEqual(record["embedding_reason"], "below_similarity_threshold")
        self.assertEqual(record["geometric_verification"]["embedding_winner"], "A")

    def test_low_score_recovery_requires_one_sku_and_enough_visible_mask(self):
        for scores, area, evidence in (
            ([0.69, 0.65], 0.02, [self.evidence("A"), self.evidence("B", False)]),
            ([0.73, 0.65], 0.004, [self.evidence("A"), self.evidence("B", False)]),
            ([0.73, 0.65], 0.02, [self.evidence("A"), self.evidence("B")]),
        ):
            record = classify_scores(scores, self.products)
            record["mask_area_ratio"] = area
            resolve_low_score_geometry(record, evidence, ["A", "B"])
            self.assertIsNone(record["sku"])

    def test_low_score_recovery_requires_evidence_from_every_registered_sku(self):
        record = classify_scores([0.73, 0.65], self.products)
        record["mask_area_ratio"] = 0.02
        with self.assertRaises(ValueError):
            resolve_low_score_geometry(record, [self.evidence("A")], ["A", "B"])

    def test_low_score_geometry_search_only_runs_for_ambiguous_usable_masks(self):
        record = classify_scores([0.73, 0.72], self.products)
        record["mask_area_ratio"] = 0.02
        self.assertTrue(should_verify_low_score_geometry(record, 0.05))
        self.assertFalse(should_verify_low_score_geometry(record, 0.01))
        record["mask_area_ratio"] = 0.004
        self.assertFalse(should_verify_low_score_geometry(record, 0.05))

    def test_omitted_or_extra_competing_skus_are_rejected(self):
        for evidence in ([self.evidence("A")], [self.evidence("A"), self.evidence("B"), self.evidence("C")]):
            record = classify_scores([0.91, 0.89], self.products)
            with self.assertRaises(ValueError):
                resolve_ambiguity(record, evidence)

    def test_verified_fourth_candidate_prevents_a_false_resolution(self):
        record = classify_scores([0.94, 0.93, 0.92, 0.91],
                                 self.products + [{"sku": "C", "name": "C"}, {"sku": "D", "name": "D"}])
        evidence = [self.evidence(sku, sku in ("A", "D")) for sku in ("A", "B", "C", "D")]
        resolve_ambiguity(record, evidence, ["A", "B", "C", "D"])
        self.assertIsNone(record["sku"])


if __name__ == "__main__":
    unittest.main()
