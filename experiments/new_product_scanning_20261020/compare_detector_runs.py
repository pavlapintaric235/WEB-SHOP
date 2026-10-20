#!/usr/bin/env python3
"""Compare two detector runs to each other, box by box, without any ground truth.

`evaluate_scene_predictions.py` is the only thing that can produce accuracy, and it needs
reviewed scene labels that do not exist yet. Until a person has reviewed the scenes, the
one honest quantitative question that *can* be answered is narrower:

    where two independent detectors place a box on the same photograph, do they agree?

Agreement is not correctness. Two detectors can agree and both be wrong, and a box that
only one of them found may well be the right one. What this does establish is whether a
new detector reproduces the geometry of an existing trusted run, which is the evidence
needed before anyone relies on the new one.

    python experiments/new_product_scanning_20261020/compare_detector_runs.py \\
        --a results/imported_runpod_scene_results_20261020_v5_flat/predictions.csv \\
        --b results/local_owlv2_sam21_20261020/predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parent


def box_iou(first, second) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def load(path: Path) -> dict[str, list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["photo_filename"]].append({
            "detection_id": row.get("detection_id", ""),
            "box": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
            "confidence": float(row.get("detector_confidence") or row.get("confidence") or 0.0),
        })
    return grouped


def match(first: list[dict], second: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching, strongest overlap first."""
    pairs = sorted(
        ((index_a, index_b, box_iou(a["box"], b["box"]))
         for index_a, a in enumerate(first) for index_b, b in enumerate(second)),
        key=lambda item: -item[2])
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = []
    for index_a, index_b, iou in pairs:
        if iou < threshold or index_a in used_a or index_b in used_b:
            continue
        used_a.add(index_a)
        used_b.add(index_b)
        matched.append((index_a, index_b, iou))
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", type=Path, required=True, help="Reference predictions CSV.")
    parser.add_argument("--b", type=Path, required=True, help="Predictions CSV to compare.")
    parser.add_argument("--iou", type=float, default=0.5, help="Matching IoU threshold.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the JSON.")
    args = parser.parse_args()

    first = load(args.a if args.a.is_absolute() else EXPERIMENT / args.a)
    second = load(args.b if args.b.is_absolute() else EXPERIMENT / args.b)
    shared = sorted(set(first) & set(second))

    per_photo = []
    total_a = total_b = total_matched = 0
    iou_sum = 0.0
    for name in shared:
        matched = match(first[name], second[name], args.iou)
        ious = [iou for _, _, iou in matched]
        per_photo.append({
            "photo_filename": name,
            "boxes_a": len(first[name]),
            "boxes_b": len(second[name]),
            "matched": len(matched),
            "mean_iou": round(sum(ious) / len(ious), 4) if ious else 0.0,
            "only_in_a": len(first[name]) - len(matched),
            "only_in_b": len(second[name]) - len(matched),
        })
        total_a += len(first[name])
        total_b += len(second[name])
        total_matched += len(matched)
        iou_sum += sum(ious)

    report = {
        "status": "detector_agreement_only_not_accuracy",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "iou_threshold": args.iou,
        "run_a": str(args.a),
        "run_b": str(args.b),
        "photographs_in_both": len(shared),
        "boxes_a": total_a,
        "boxes_b": total_b,
        "matched_pairs": total_matched,
        "mean_iou_of_matched": round(iou_sum / total_matched, 4) if total_matched else 0.0,
        "share_of_a_matched": round(total_matched / total_a, 4) if total_a else 0.0,
        "share_of_b_matched": round(total_matched / total_b, 4) if total_b else 0.0,
        "limitations": [
            "This is agreement between two models, not accuracy. Neither run is ground truth.",
            "An unmatched box is not necessarily wrong; it may be an item the other run missed.",
            "Only box geometry is compared. Masks and SKU identities are not scored here.",
        ],
        "per_photo": per_photo,
    }

    destination = args.output or (EXPERIMENT / "results/detector_agreement.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Photographs in both runs: {len(shared)}")
    print(f"  boxes A={total_a}  B={total_b}  matched at IoU>={args.iou}: {total_matched}")
    print(f"  mean IoU of matched pairs: {report['mean_iou_of_matched']}")
    print(f"  share of A matched: {report['share_of_a_matched']}")
    print(f"  share of B matched: {report['share_of_b_matched']}")
    print(f"\n  report: {destination}")
    print("  status: agreement between detectors - NOT accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
