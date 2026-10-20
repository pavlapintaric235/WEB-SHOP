"""Offline handler checks using an in-memory S3 client; no AWS credentials."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handler
from capture_gate import capture_decision, quality


SOURCE_B01 = (
    Path(__file__).resolve().parents[3]
    / "data/recognition_checks/final_comparative_gpu_benchmark_20261018"
    / "inventorylens_final_comparative_gpu_benchmark/data/source/images/scenes/B01/IMG_4754.jpeg"
)
BUCKET = "inventorylens-local-fake"


class FakeS3:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.results: dict[str, dict] = {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        assert Bucket == BUCKET
        data = self.objects[Key]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes,
                   ContentType: str, ServerSideEncryption: str) -> None:
        assert Bucket == BUCKET
        assert ContentType == "application/json"
        assert ServerSideEncryption == "AES256"
        self.results[Key] = json.loads(Body)


def event_for(key: str, bucket: str = BUCKET) -> dict:
    return {"Records": [{
        "eventName": "ObjectCreated:Put",
        "s3": {"bucket": {"name": bucket}, "object": {"key": key}},
    }]}


def encoded(image: np.ndarray) -> bytes:
    ok, output = cv2.imencode(".jpg", image)
    assert ok
    return output.tobytes()


class HandlerOfflineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = SOURCE_B01.read_bytes()
        cls.image = cv2.imdecode(np.frombuffer(cls.original, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert cls.image is not None

    def run_image(self, name: str, data: bytes) -> dict:
        key = f"incoming/offline/{name}.jpg"
        fake = FakeS3({key: data})
        with patch.dict(os.environ, {"AWS_REGION": "eu-north-1", "PILOT_BUCKET": BUCKET}):
            with patch.object(handler, "_s3_client", fake):
                summary = handler.lambda_handler(event_for(key), None)
        self.assertEqual(summary["results"][0]["result_key"], f"results/offline/{name}.json")
        return fake.results[f"results/offline/{name}.json"]

    def test_original_has_matching_hash_metrics_and_proceed_action(self) -> None:
        result = self.run_image("b01", self.original)
        self.assertEqual(result["image_sha256"], hashlib.sha256(self.original).hexdigest())
        self.assertEqual(result["quality"], quality(self.image))
        self.assertEqual(result["capture_decision"], capture_decision(quality(self.image)))
        self.assertEqual(result["capture_decision"]["action"], "proceed")
        self.assertEqual(result["status"], "ok")

    def test_blur_and_dark_request_retake(self) -> None:
        blurred = encoded(cv2.GaussianBlur(self.image, (0, 0), sigmaX=6))
        dark = encoded(cv2.convertScaleAbs(self.image, alpha=0.07, beta=0))
        self.assertEqual(self.run_image("blur", blurred)["capture_decision"]["reason"],
                         "Hold the camera steadier")
        self.assertEqual(self.run_image("dark", dark)["capture_decision"]["reason"],
                         "Use more light")

    def test_invalid_and_oversize_do_not_proceed(self) -> None:
        self.assertEqual(self.run_image("invalid", b"not an image")["status"], "rejected")
        oversize = self.run_image("oversize", b"x" * (handler.MAX_IMAGE_BYTES + 1))
        self.assertEqual(oversize["capture_decision"]["action"], "retake")
        self.assertIsNone(oversize["image_sha256"])

    def test_wrong_bucket_and_non_input_prefix_are_safe(self) -> None:
        fake = FakeS3({})
        with self.assertRaises(ValueError):
            handler.process_record(event_for("incoming/offline/b01.jpg", "other")["Records"][0], fake, BUCKET)
        ignored = handler.process_record(event_for("results/offline/b01.json")["Records"][0], fake, BUCKET)
        self.assertEqual(ignored["status"], "ignored")
        self.assertFalse(fake.results)

    def test_non_image_filename_returns_rejection_without_reading_object(self) -> None:
        fake = FakeS3({})
        outcome = handler.process_record(
            event_for("incoming/offline/note.txt")["Records"][0], fake, BUCKET
        )
        self.assertEqual(outcome["status"], "rejected")
        self.assertEqual(fake.results[outcome["result_key"]]["capture_decision"]["action"], "retake")


if __name__ == "__main__":
    unittest.main()
