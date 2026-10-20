"""Isolated S3-to-Lambda OpenCV 5 capture pilot; no inventory writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from importlib.metadata import PackageNotFoundError, version
from urllib.parse import unquote_plus

import cv2
import numpy as np

from capture_gate import capture_decision, quality


MAX_IMAGE_BYTES = 20 * 1024 * 1024
KEY_PATTERN = re.compile(
    r"^incoming/([A-Za-z0-9_-]{1,64})/([A-Za-z0-9_-]{1,64})\.(jpe?g|png)$",
    re.IGNORECASE,
)
_s3_client = None


def opencv_package_version() -> str:
    try:
        return version("opencv-python-headless")
    except PackageNotFoundError:
        return version("opencv-python")


def _result_key(source_key: str) -> str:
    match = KEY_PATTERN.fullmatch(source_key)
    if match is None:
        raise ValueError("Image key must match incoming/<run-id>/<image-id>.jpg|jpeg|png")
    return f"results/{match.group(1)}/{match.group(2)}.json"


def _retake(reason: str) -> dict[str, str]:
    return {"action": "retake", "reason": reason}


def _write_result(s3_client, bucket: str, result_key: str, result: dict) -> None:
    s3_client.put_object(
        Bucket=bucket,
        Key=result_key,
        Body=json.dumps(result, sort_keys=True, allow_nan=False).encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


def process_record(record: dict, s3_client, expected_bucket: str) -> dict:
    """Process one S3 event with an injected client for offline verification."""
    event_name = record.get("eventName", "")
    if not event_name.startswith("ObjectCreated:"):
        return {"status": "ignored", "reason": "Not an ObjectCreated event"}

    source = record["s3"]
    bucket = source["bucket"]["name"]
    if bucket != expected_bucket:
        raise ValueError("S3 event bucket does not match the configured pilot bucket")

    source_key = unquote_plus(source["object"]["key"])
    if not source_key.startswith("incoming/"):
        return {"status": "ignored", "reason": "Outside incoming prefix"}
    try:
        result_key = _result_key(source_key)
    except ValueError:
        result_key = f"results/rejected/{hashlib.sha256(source_key.encode('utf-8')).hexdigest()}.json"
        result = {
            "schema_version": 1,
            "status": "rejected",
            "source_bucket": bucket,
            "source_key": source_key,
            "image_sha256": None,
            "quality": None,
            "capture_decision": _retake("Use a JPEG or PNG name under incoming/<run-id>/"),
        }
        _write_result(s3_client, bucket, result_key, result)
        return {"status": "rejected", "result_key": result_key}
    started = time.perf_counter()

    response = s3_client.get_object(Bucket=bucket, Key=source_key)
    body = response["Body"]
    try:
        content_length = response.get("ContentLength")
        if content_length is not None and content_length > MAX_IMAGE_BYTES:
            image_bytes = None
        else:
            image_bytes = body.read(MAX_IMAGE_BYTES + 1)
    finally:
        body.close()

    result = {
        "schema_version": 1,
        "status": "ok",
        "source_bucket": bucket,
        "source_key": source_key,
        "image_sha256": None,
        "quality": None,
        "capture_decision": None,
        "runtime": {
            "cv2_version": cv2.__version__,
            "opencv_package_version": opencv_package_version(),
            "numpy_version": np.__version__,
        },
    }
    if image_bytes is None or len(image_bytes) > MAX_IMAGE_BYTES:
        result["status"] = "rejected"
        result["capture_decision"] = _retake("Image exceeds the 20 MiB pilot limit")
    else:
        result["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        metrics = quality(image)
        result["quality"] = metrics
        result["capture_decision"] = capture_decision(metrics)
        if not metrics["decode_ok"]:
            result["status"] = "rejected"

    result["processing_ms"] = round((time.perf_counter() - started) * 1000, 2)
    _write_result(s3_client, bucket, result_key, result)
    return {"status": result["status"], "result_key": result_key}


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3")
    return _s3_client


def lambda_handler(event: dict, context: object) -> dict:
    if event.get("Event") == "s3:TestEvent":
        return {"status": "ignored-test-event"}
    if os.environ.get("AWS_REGION") != "eu-north-1":
        raise RuntimeError("Pilot Lambda must run in selected Region eu-north-1")
    bucket = os.environ["PILOT_BUCKET"]
    records = event["Records"]
    if len(records) > 10:
        raise ValueError("Pilot accepts at most 10 records per invocation")
    s3_client = _get_s3_client()
    return {"results": [process_record(record, s3_client, bucket) for record in records]}
