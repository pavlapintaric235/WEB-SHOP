import unittest

import numpy as np

from experiments.hybrid_pipeline.hybrid_core import (
    abstaining_decision,
    apply_manual_corrections,
    count_by_sku,
    fuse_sku_scores,
    scores_for_features,
    scores_for_reference_views,
    unit_vector,
)


class HybridCoreTests(unittest.TestCase):
    def test_unknown_when_score_is_below_cutoff(self):
        result = abstaining_decision({"A": 0.49, "B": 0.32}, minimum=0.5, margin=0.04)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["sku"])
        self.assertEqual(result["reason"], "below_similarity_threshold")

    def test_unknown_when_top_two_margin_is_too_small(self):
        result = abstaining_decision({"A": 0.82, "B": 0.80}, minimum=0.5, margin=0.04)
        self.assertEqual(result["reason"], "ambiguous_margin")

    def test_equal_branch_fusion(self):
        scores = fuse_sku_scores({"dino": {"A": 0.9, "B": 0.2},
                                  "mobile": {"A": 0.3, "B": 0.8}})
        self.assertAlmostEqual(scores["A"], 0.6)
        self.assertAlmostEqual(scores["B"], 0.5)

    def test_few_shot_best_view_per_sku(self):
        index = {"A": {"mobile": [
                    {"feature": unit_vector([1, 0]), "image": "a1", "sha256": "x"},
                    {"feature": unit_vector([0, 1]), "image": "a2", "sha256": "y"}]},
                 "B": {"mobile": [
                    {"feature": unit_vector([0, 1]), "image": "b1", "sha256": "z"}]}}
        scores, branches, refs = scores_for_features({"mobile": np.asarray([0.9, 0.1])}, index)
        self.assertGreater(scores["A"], scores["B"])
        self.assertEqual(refs["mobile"]["A"]["path"], "a1")
        self.assertEqual(branches["mobile"]["A"], scores["A"])

    def test_reference_view_scoring_preserves_best_view_and_path(self):
        index = {"A": {"dino": [
                    {"feature": unit_vector([1, 0]), "image": "front.jpg", "sha256": "a"},
                    {"feature": unit_vector([0, 1]), "image": "side.jpg", "sha256": "b"}]},
                 "B": {"dino": [
                    {"feature": unit_vector([0, 1]), "image": "b.jpg", "sha256": "c"}]}}
        scores, _, refs = scores_for_reference_views({"dino": np.asarray([0.8, 0.6])}, index)
        self.assertGreater(scores["A"], scores["B"])
        self.assertEqual(refs["dino"]["A"]["path"], "front.jpg")
        self.assertAlmostEqual(refs["dino"]["A"]["similarity"], scores["A"])

    def test_counts_ignore_unknown_instances(self):
        self.assertEqual(count_by_sku([
            {"sku": "A", "status": "recognized"},
            {"sku": "A", "status": "recognized"},
            {"sku": None, "status": "unknown"},
        ]), {"A": 2})

    def test_manual_confirmation_changes_sku_and_quantity_not_prediction(self):
        detections = [{"sku": None, "status": "unknown", "similarity": 0.4},
                      {"sku": "A", "status": "recognized"}]
        corrected, counts = apply_manual_corrections(
            detections, [{"sku": "B", "quantity": 2}, {"sku": "__unknown__", "quantity": 1}])
        self.assertEqual(corrected[0]["predicted_sku"], None)
        self.assertEqual(corrected[0]["sku"], "B")
        self.assertEqual(corrected[0]["confirmed_quantity"], 2)
        self.assertEqual(corrected[1]["predicted_sku"], "A")
        self.assertEqual(counts, {"B": 2})


if __name__ == "__main__":
    unittest.main()
