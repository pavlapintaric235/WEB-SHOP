from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if (PACKAGE_ROOT / "project").is_dir():
    sys.path.insert(0, str(PACKAGE_ROOT))
    import run_inference as runner
else:
    from experiments.runpod_gpu_inference import run_inference as runner


class FakeEncoder:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)

    def encode(self, images, batch_size=8):
        return np.tile(self.vector, (len(images), 1))


def reference(vector, path):
    return {"feature": np.asarray(vector, dtype=np.float32), "image": path, "sha256": "test"}


class RunPodInferenceTests(unittest.TestCase):
    def test_encoder_arms_and_fusion_keep_independent_decisions(self):
        index = {
            "ILX-001": {
                "mobilenetv2": [reference([1, 0], "a")],
                "siglip2": [reference([1, 0], "a")],
            },
            "ILX-002": {
                "mobilenetv2": [reference([0, 1], "b")],
                "siglip2": [reference([0, 1], "b")],
            },
        }
        matcher = SimpleNamespace(
            encoders={"mobilenetv2": FakeEncoder([1, 0]), "siglip2": FakeEncoder([1, 0])},
            index=index,
        )
        with patch.object(runner, "sync", lambda: None):
            decisions, _ = runner.encode_and_match(matcher, [Image.new("RGB", (12, 12))])
        for mode in ("mobilenetv2", "siglip2", "siglip2+mobilenetv2"):
            self.assertEqual(decisions[0][mode]["sku"], "ILX-001")
            self.assertEqual(decisions[0][mode]["status"], "recognized")

    def test_same_scene_views_are_never_summed(self):
        rows = [
            {"backend": "baseline", "recognition_mode": "fusion", "scene_id": "B10",
             "manifest_order": 0, "image": "view-a.jpeg",
             "predictions": [{"sku": "ILX-001", "status": "recognized"} for _ in range(2)]},
            {"backend": "baseline", "recognition_mode": "fusion", "scene_id": "B10",
             "manifest_order": 1, "image": "view-b.jpeg",
             "predictions": [{"sku": "ILX-001", "status": "recognized"} for _ in range(3)]},
        ]
        summary = runner.primary_view_summaries(
            rows,
            {"B10": {"view_group_class": "same_arrangement"}},
            {"B10": {"total": 11, "status": "USER-CONFIRMED"}},
            {"B10": {"ILX-001": 2}},
        )[0]
        self.assertEqual(summary["view_count"], 2)
        self.assertEqual(summary["primary_detection_count"], 2)
        self.assertEqual(summary["non_additive_max_per_sku_across_views"], {"ILX-001": 3})
        self.assertIn("never summed", summary["multi_view_warning"])


if __name__ == "__main__":
    unittest.main()
