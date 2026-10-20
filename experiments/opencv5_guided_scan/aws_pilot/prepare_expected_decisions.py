"""Record local OpenCV capture decisions for the approved unique originals."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_gate import capture_decision, quality
from upload_manifest import verified_upload_rows


OUTPUT = Path(__file__).with_name("local_expected_decisions_20261020.csv")
FIELDS = [
    "source_group", "role", "s3_key", "sha256", "action", "reason", "decode_ok",
    "width", "height", "lap_var_1024", "dark_pct", "bright_pct",
]


def main() -> None:
    rows = []
    for source in verified_upload_rows():
        image_bytes = Path(source["source_path"]).read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        metrics = quality(image)
        decision = capture_decision(metrics)
        rows.append({
            "source_group": source["source_group"],
            "role": source["role"],
            "s3_key": source["s3_key"],
            "sha256": source["sha256"],
            "action": decision["action"],
            "reason": decision["reason"],
            **metrics,
        })
    with OUTPUT.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} local results to {OUTPUT}")
    print(f"Actions: {dict(Counter(row['action'] for row in rows))}")
    print(f"Reasons: {dict(Counter(row['reason'] for row in rows))}")


if __name__ == "__main__":
    main()
