"""Explicit-click AWS capture check for the isolated guided-scan demo."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


RESOURCES = Path(__file__).resolve().parent / "aws_pilot/deployment_resources_20261020.json"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PILOT_INPUTS = 200
RESULT_TIMEOUT_SECONDS = 90


def image_type(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8"):
        return "jpg", "image/jpeg"
    raise ValueError("AWS capture accepts original JPEG or PNG bytes only")


def checked_result(s3, bucket: str, source_key: str, result_key: str,
                   image_sha256: str) -> dict | None:
    try:
        response = s3.get_object(Bucket=bucket, Key=result_key)
    except s3.exceptions.NoSuchKey:
        return None
    with response["Body"] as body:
        result = json.load(body)
    if result.get("source_bucket") != bucket or result.get("source_key") != source_key:
        raise ValueError("AWS capture result refers to a different source photo")
    if result.get("image_sha256") != image_sha256:
        raise ValueError("AWS capture result hash does not match this photo")
    if result.get("status") != "ok" or result.get("runtime", {}).get("cv2_version", "").split(".")[0] != "5":
        raise ValueError("AWS capture result is not a successful OpenCV 5 check")
    if result.get("capture_decision", {}).get("action") not in {"retake", "proceed"}:
        raise ValueError("AWS capture result has no usable action")
    return result


def run_capture(data: bytes, image_sha256: str) -> dict:
    """Upload exact bytes only when called, then return a matching AWS result."""
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("AWS capture requires a photo of at most 20 MiB")
    digest = hashlib.sha256(data).hexdigest()
    if digest != image_sha256:
        raise ValueError("The selected photo changed before AWS capture")
    extension, content_type = image_type(data)
    config = json.loads(RESOURCES.read_text(encoding="utf-8"))
    if config["region"] != "eu-north-1":
        raise ValueError("The pilot must use selected Region eu-north-1")
    import boto3

    session = boto3.Session(profile_name=config["profile"], region_name=config["region"])
    identity = session.client("sts").get_caller_identity()
    if identity["Account"] != config["project_account_id"]:
        raise ValueError("AWS session is not signed in to the WEB&SHOP project")
    s3 = session.client("s3")
    bucket = config["bucket"]
    if s3.get_bucket_location(Bucket=bucket).get("LocationConstraint") != config["region"]:
        raise ValueError("Pilot bucket is outside the selected Region")
    source_key = f"incoming/live_demo_20261020/{digest}.{extension}"
    result_key = f"results/live_demo_20261020/{digest}.json"
    existing_result = checked_result(s3, bucket, source_key, result_key, digest)
    if existing_result is not None:
        return existing_result
    listed = s3.list_objects_v2(Bucket=bucket, Prefix="incoming/", MaxKeys=MAX_PILOT_INPUTS)
    if listed.get("KeyCount", 0) >= MAX_PILOT_INPUTS:
        raise ValueError("The pilot has reached its 200-photo input cap")
    s3.put_object(
        Bucket=bucket,
        Key=source_key,
        Body=data,
        ContentType=content_type,
        ServerSideEncryption="AES256",
        Metadata={"sha256": digest},
    )
    deadline = time.monotonic() + RESULT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        result = checked_result(s3, bucket, source_key, result_key, digest)
        if result is not None:
            return result
        time.sleep(1)
    raise TimeoutError("No matching AWS capture result arrived within 90 seconds")
