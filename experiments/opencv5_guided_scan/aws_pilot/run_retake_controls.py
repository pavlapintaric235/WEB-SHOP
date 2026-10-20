"""Upload two in-memory B01 retakes and compare Lambda with local OpenCV."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import boto3
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_gate import capture_decision, quality
from upload_manifest import PROFILE, REGION, wait_for_result


BUCKET = "inventorylens-opencv5-pilot-478400396665-20261020"
SOURCE_B01 = (
    Path(__file__).resolve().parents[3]
    / "data/recognition_checks/final_comparative_gpu_benchmark_20261018"
    / "inventorylens_final_comparative_gpu_benchmark/data/source/images/scenes/B01/IMG_4754.jpeg"
)


def encoded(image: np.ndarray) -> bytes:
    ok, output = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("OpenCV could not encode a retake control")
    return output.tobytes()


def main() -> None:
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    s3 = session.client("s3")
    location = s3.get_bucket_location(Bucket=BUCKET).get("LocationConstraint")
    if location != REGION:
        raise RuntimeError(f"Pilot bucket is not in selected Region {REGION}")
    original = cv2.imdecode(np.frombuffer(SOURCE_B01.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError("Original B01 photo could not be decoded")
    controls = {
        "blur": cv2.GaussianBlur(original, (0, 0), sigmaX=6),
        "dark": cv2.convertScaleAbs(original, alpha=0.07, beta=0),
    }
    for name, image in controls.items():
        data = encoded(image)
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        metrics = quality(decoded)
        decision = capture_decision(metrics)
        if decision["action"] != "retake":
            raise RuntimeError(f"Local {name} control did not request retake")
        digest = hashlib.sha256(data).hexdigest()
        key = f"incoming/inventorylens_20261020/quality_control_{name}.jpg"
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=data,
            ContentType="image/jpeg",
            ServerSideEncryption="AES256",
            Metadata={"sha256": digest},
        )
        source = {"s3_key": key, "sha256": digest}
        expected = {"action": decision["action"], "reason": decision["reason"],
                    **{field: str(value) for field, value in metrics.items()}}
        wait_for_result(s3, BUCKET, source, expected)
        print(f"{name}: Lambda matched local retake reason: {decision['reason']}", flush=True)


if __name__ == "__main__":
    main()
