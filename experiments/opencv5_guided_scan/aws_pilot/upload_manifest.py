"""Upload only approved, hash-verified pilot originals to the selected Region."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

from build_upload_manifest import FIELDS, OUTPUT, build


REGION = "eu-north-1"
PROFILE = "WEB&SHOP"
MAX_UNIQUE_UPLOADS = 200
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
RESULT_TIMEOUT_SECONDS = 90
EXPECTED = Path(__file__).with_name("local_expected_decisions_20261020.csv")


def verified_upload_rows() -> list[dict[str, str]]:
    with OUTPUT.open(encoding="utf-8", newline="") as source:
        frozen = list(csv.DictReader(source))
    current = [{field: str(row[field]) for field in FIELDS} for row in build()]
    if frozen != current:
        raise ValueError("The source photos or frozen upload manifest changed; review before upload")
    uploads = [row for row in current if row["upload"] == "yes"]
    if len(uploads) > MAX_UNIQUE_UPLOADS:
        raise ValueError("The manifest exceeds the 200 unique-photo pilot cap")
    if sum(int(row["bytes"]) for row in uploads) > MAX_TOTAL_BYTES:
        raise ValueError("The manifest exceeds the 1 GiB pilot upload cap")
    if len({row["s3_key"] for row in uploads}) != len(uploads):
        raise ValueError("The manifest contains duplicate upload keys")
    return uploads


def expected_by_key(uploads: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    with EXPECTED.open(encoding="utf-8", newline="") as source:
        expected = {row["s3_key"]: row for row in csv.DictReader(source)}
    if set(expected) != {row["s3_key"] for row in uploads}:
        raise ValueError("Local expected decisions do not match the approved upload keys")
    for row in uploads:
        if expected[row["s3_key"]]["sha256"] != row["sha256"]:
            raise ValueError(f"Expected-result hash mismatch: {row['s3_key']}")
    return expected


def wait_for_result(s3, bucket: str, source: dict[str, str], expected: dict[str, str]) -> None:
    key = source["s3_key"].replace("incoming/", "results/", 1)
    key = key.rsplit(".", 1)[0] + ".json"
    deadline = time.monotonic() + RESULT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
        except s3.exceptions.NoSuchKey:
            time.sleep(1)
            continue
        with response["Body"] as body:
            result = json.load(body)
        decision = result.get("capture_decision") or {}
        if result.get("source_key") != source["s3_key"]:
            raise ValueError(f"Result points to a different source: {key}")
        if result.get("image_sha256") != source["sha256"]:
            raise ValueError(f"Result image hash mismatch: {key}")
        if result.get("status") != "ok":
            raise ValueError(f"Lambda rejected an approved original: {key}")
        if (decision.get("action"), decision.get("reason")) != (
            expected["action"], expected["reason"]
        ):
            raise ValueError(f"Lambda decision differs from local OpenCV baseline: {key}")
        metrics = result.get("quality") or {}
        for field in ("width", "height"):
            if metrics.get(field) != int(expected[field]):
                raise ValueError(f"Lambda {field} differs from local baseline: {key}")
        for field in ("lap_var_1024", "dark_pct", "bright_pct"):
            actual = metrics.get(field)
            reference = float(expected[field])
            tolerance = max(1.0, abs(reference) * 0.02) if field == "lap_var_1024" else 0.2
            if actual is None or abs(actual - reference) > tolerance:
                raise ValueError(f"Lambda {field} differs from local baseline: {key}")
        return
    raise TimeoutError(f"No matching Lambda result after {RESULT_TIMEOUT_SECONDS}s: {key}")


def upload(uploads: list[dict[str, str]], bucket: str, limit: int | None = None) -> None:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config
    from botocore.exceptions import ClientError

    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    s3 = session.client(
        "s3",
        config=Config(retries={"total_max_attempts": 2, "mode": "standard"}),
    )
    location = s3.get_bucket_location(Bucket=bucket).get("LocationConstraint")
    if location != REGION:
        raise ValueError(f"Pilot bucket must be in selected Region {REGION}; got {location!r}")

    transfer = TransferConfig(multipart_threshold=20 * 1024 * 1024 + 1)
    expected = expected_by_key(uploads)
    selected = uploads[:limit] if limit is not None else uploads
    sent = skipped = 0
    for row in selected:
        key = row["s3_key"]
        try:
            existing = s3.head_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if (existing.get("ContentLength") != int(row["bytes"])
                    or existing.get("Metadata", {}).get("sha256") != row["sha256"]):
                raise ValueError(f"Existing object conflicts with approved manifest: {key}")
            wait_for_result(s3, bucket, row, expected[key])
            skipped += 1
            continue

        path = Path(row["source_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source changed immediately before upload: {path}")
        content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        s3.upload_file(
            str(path), bucket, key,
            ExtraArgs={
                "ContentType": content_type,
                "ServerSideEncryption": "AES256",
                "Metadata": {"sha256": row["sha256"]},
            },
            Config=transfer,
        )
        wait_for_result(s3, bucket, row, expected[key])
        sent += 1
        print(f"Verified {sent + skipped}/{len(selected)}: {key}", flush=True)
    print(f"Complete: {sent} uploaded, {skipped} already present")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", help="Private pilot bucket in eu-north-1")
    parser.add_argument("--execute", action="store_true", help="Actually upload approved originals")
    parser.add_argument("--limit", type=int, help="Process only the first N unique originals")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= MAX_UNIQUE_UPLOADS:
        parser.error("--limit must be between 1 and 200")
    try:
        uploads = verified_upload_rows()
        byte_count = sum(int(row["bytes"]) for row in uploads)
        print(f"Verified {len(uploads)} unique originals, {byte_count} bytes")
        if args.execute:
            if not args.bucket:
                parser.error("--bucket is required with --execute")
            upload(uploads, args.bucket, args.limit)
        else:
            print("Dry run only. Upload requires --execute and --bucket.")
        return 0
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"Pilot upload stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
