"""Evaluate the DINO held-out boxes against the user's item-box review export."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / "artifacts" / "third_set_review_20261020"
ANNOTATIONS = HERE.parent / "annotations"
PREDICTIONS = ARTIFACTS / "predictions.json"
LABELS = ARTIFACTS / "dino_inference" / "results" / "labels"
REVIEW = HERE.parent / "annotations" / "dino_item_boxes_review_20261023.json"
OUT_JSON = ARTIFACTS / "dino_ground_truth_eval_20261023.json"
OUT_CSV = ARTIFACTS / "dino_ground_truth_eval_by_photo_20261023.csv"
IOU_THRESHOLD = 0.5


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def counts(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    model_data = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    reviewed = {scene["filename"]: scene for scene in review["scenes"]}
    model_scenes = {Path(scene["original_filename"]).name: scene for scene in model_data["scenes"]}
    if review.get("schema_version") != 1 or review.get("coordinate_space") != "EXIF-corrected upright image pixels; xyxy":
        raise ValueError("Unexpected review schema or coordinate space")
    if len(reviewed) != 30 or set(reviewed) != set(model_scenes):
        raise ValueError("Review and DINO prediction exports must contain the same 30 photos")

    rows = []
    group_counts = {"Employee": [0, 0, 0], "Customer": [0, 0, 0]}
    total_gt = total_det = 0
    for name, scene in model_scenes.items():
        gt_scene = reviewed[name]
        if not gt_scene.get("review_complete"):
            raise ValueError(f"Scene not marked complete: {name}")
        if gt_scene.get("source_sha256") != scene["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        image_path = ARTIFACTS / scene["local_path"]
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != scene["sha256"]:
            raise ValueError(f"Packaged source hash mismatch: {name}")
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
        if [width, height] != [gt_scene["width"], gt_scene["height"]]:
            raise ValueError(f"Source dimension mismatch: {name}")

        gt_boxes = gt_scene["boxes_xyxy"]
        for number, box in enumerate(gt_boxes, 1):
            if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Invalid item box {number} in {name}: {box}")
        detections = []
        labels_path = LABELS / f"{Path(name).stem}.txt"
        if labels_path.exists():
            for line in labels_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 16:
                    box = [float(value) for value in fields[4:8]]
                    detections.append({"box": box, "confidence": float(fields[-1])})
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        matched_gt: set[int] = set()
        tp = fp = 0
        for detection in detections:
            candidates = [(iou(detection["box"], box), idx) for idx, box in enumerate(gt_boxes) if idx not in matched_gt]
            best_iou, best_idx = max(candidates, default=(0.0, -1))
            if best_iou >= IOU_THRESHOLD:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)
        error = len(detections) - len(gt_boxes)
        row = {"photo_filename": name, "group": gt_scene["group"], "ground_truth_items": len(gt_boxes),
               "dino_detections": len(detections), "tp_iou50": tp, "fp_iou50": fp, "fn_iou50": fn,
               "count_error_detections_minus_items": error}
        rows.append(row)
        totals = group_counts[gt_scene["group"]]
        totals[0] += tp; totals[1] += fp; totals[2] += fn
        total_gt += len(gt_boxes); total_det += len(detections)

    totals = group_counts["Employee"][0] + group_counts["Customer"][0], group_counts["Employee"][1] + group_counts["Customer"][1], group_counts["Employee"][2] + group_counts["Customer"][2]
    result = {
        "status": "PROVISIONAL_HUMAN_REVIEWED_BOX_EVALUATION",
        "source_review": str(REVIEW),
        "model": {"created_utc": model_data["created_utc"], "confidence_threshold": model_data["confidence_threshold"],
                  "iou_threshold_for_detections": model_data["iou_threshold"], "scenes": model_data["held_out_images"]},
        "matching": {"metric_iou_threshold": IOU_THRESHOLD, "method": "detections sorted by confidence; one-to-one greedy match to highest-IoU unmatched ground-truth box"},
        "ground_truth": {"scenes": len(rows), "items": total_gt, "complete_scenes": sum(bool(s.get("review_complete")) for s in review["scenes"])},
        "dino": {"detections": total_det},
        "overall": counts(*totals),
        "by_group": {group: counts(*values) for group, values in group_counts.items()},
        "limitations": ["Scores depend on the user's one-box-per-visible-item annotations, including decisions in crowded and partially occluded scenes.",
                        "This evaluates generic product localization only; it does not measure SKU recognition.",
                        "The 30 held-out photos have been seen during earlier model review and are not a pristine blind test."],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"overall": result["overall"], "by_group": result["by_group"], "ground_truth": result["ground_truth"], "dino": result["dino"], "out_json": str(OUT_JSON), "out_csv": str(OUT_CSV)}, indent=2))


if __name__ == "__main__":
    main()
