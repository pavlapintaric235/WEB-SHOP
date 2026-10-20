"""Audit downloaded pilot results against the frozen local OpenCV baseline."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "runs/aws_pilot_20261020/results"
EXPECTED = ROOT / "local_expected_decisions_20261020.csv"
REPORTS = ROOT / "lambda_reports_20261020.json"
OUTPUT = ROOT.parent / "runs/aws_pilot_20261020/audit_summary_20261020.json"
CONTROL_REASONS = {
    "quality_control_blur.json": "Hold the camera steadier",
    "quality_control_dark.json": "Use more light",
}


def compare_metrics(result: dict, expected: dict[str, str]) -> None:
    metrics = result["quality"]
    for field in ("width", "height"):
        if metrics[field] != int(expected[field]):
            raise ValueError(f"{field} mismatch: {result['source_key']}")
    for field in ("lap_var_1024", "dark_pct", "bright_pct"):
        reference = float(expected[field])
        tolerance = max(1.0, abs(reference) * 0.02) if field == "lap_var_1024" else 0.2
        if abs(metrics[field] - reference) > tolerance:
            raise ValueError(f"{field} mismatch: {result['source_key']}")


def main() -> None:
    with EXPECTED.open(encoding="utf-8", newline="") as source:
        expected_rows = list(csv.DictReader(source))
    expected_by_file = {
        row["s3_key"].rsplit("/", 1)[1].rsplit(".", 1)[0] + ".json": row
        for row in expected_rows
    }
    actual_files = {path.name: path for path in RESULTS.glob("*.json")}
    if set(actual_files) != set(expected_by_file) | set(CONTROL_REASONS):
        raise ValueError("Downloaded result files differ from 176 originals plus two controls")
    actions = Counter()
    groups = Counter()
    processing_ms = []
    for filename, row in expected_by_file.items():
        result = json.loads(actual_files[filename].read_text(encoding="utf-8"))
        if result["status"] != "ok" or result["source_key"] != row["s3_key"]:
            raise ValueError(f"Status/source mismatch: {filename}")
        if result["image_sha256"] != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {filename}")
        decision = result["capture_decision"]
        if (decision["action"], decision["reason"]) != (row["action"], row["reason"]):
            raise ValueError(f"Decision mismatch: {filename}")
        compare_metrics(result, row)
        runtime = result["runtime"]
        if runtime != {"cv2_version": "5.0.0", "numpy_version": "2.5.4",
                       "opencv_package_version": "5.0.0.93"}:
            raise ValueError(f"Runtime package mismatch: {filename}")
        actions[decision["action"]] += 1
        groups[(row["source_group"], decision["action"])] += 1
        processing_ms.append(result["processing_ms"])
    for filename, reason in CONTROL_REASONS.items():
        result = json.loads(actual_files[filename].read_text(encoding="utf-8"))
        if (result["status"], result["capture_decision"]) != (
            "ok", {"action": "retake", "reason": reason}
        ):
            raise ValueError(f"Retake control mismatch: {filename}")

    reports = json.loads(REPORTS.read_text(encoding="utf-8-sig"))
    billed_ms = [float(match.group(1)) for item in reports
                 if (match := re.search(r"Billed Duration: ([0-9.]+) ms", item))]
    memory_mb = [int(match.group(1)) for item in reports
                 if (match := re.search(r"Max Memory Used: ([0-9]+) MB", item))]
    if len(billed_ms) != 179:
        raise ValueError(f"Expected 179 Lambda reports including import test, found {len(billed_ms)}")
    summary = {
        "date": "2026-10-20",
        "region": "eu-north-1",
        "originals_verified": len(expected_rows),
        "retake_controls_verified": len(CONTROL_REASONS),
        "mismatches": 0,
        "original_actions": dict(actions),
        "by_group_and_action": {f"{group}:{action}": count for (group, action), count in sorted(groups.items())},
        "lambda_reports_including_import_test": len(billed_ms),
        "lambda_billed_ms_total": sum(billed_ms),
        "lambda_billed_ms_max": max(billed_ms),
        "lambda_peak_memory_mb": max(memory_mb),
        "handler_processing_ms_p95_originals": sorted(processing_ms)[math.ceil(0.95 * len(processing_ms)) - 1],
        "estimated_lambda_charge_usd_before_credits": round(
            sum(billed_ms) / 1000 * 0.5 * 0.0000166667 + len(billed_ms) * 0.0000002, 8
        ),
        "note": "Lambda estimate uses AWS eu-north-1 x86 tier-one rates; S3, logs, taxes, and other project usage are additional. Billing is delayed.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
