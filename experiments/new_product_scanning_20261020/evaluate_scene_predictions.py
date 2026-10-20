"""Score predictions against completed human-reviewed boxes from this experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiment_io import RESULTS, load_registry, load_scene_labels, sha256_file

REQUIRED = {"photo_filename", "detection_id", "sku_id", "acceptance_status", "confidence", "x1", "y1", "x2", "y2"}


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(prediction_path: Path, threshold: float = 0.5) -> dict:
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError("Predictions CSV needs columns: " + ", ".join(sorted(REQUIRED)))
        prediction_rows = list(reader)
    labels = load_scene_labels()
    reviewed = {name: record for name, record in labels.items() if record.get("review_complete")}
    if not reviewed:
        raise RuntimeError("No scene photos are marked review_complete; refusing to compute metrics.")
    confirmed_skus = {row["sku_id"] for row in load_registry() if row.get("identity_confirmed")}
    for name, record in reviewed.items():
        boxes = [tuple(item["box_xyxy"]) for item in record.get("instances", [])]
        if len(boxes) != len(set(boxes)):
            raise ValueError(f"Duplicate physical-instance boxes in {name}; fix the review before scoring.")

    pred_by_photo: dict[str, list[dict]] = defaultdict(list)
    for item in prediction_rows:
        name = item["photo_filename"]
        if name not in reviewed:
            continue
        box = [float(item[key]) for key in ("x1", "y1", "x2", "y2")]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid predicted box in {name}: {box}")
        if item["acceptance_status"] not in {"accepted", "abstained"}:
            raise ValueError(f"Invalid acceptance_status for {name}: {item['acceptance_status']}")
        if item["acceptance_status"] == "abstained" and item["sku_id"] not in {"", "__unknown__"}:
            raise ValueError(f"Abstained prediction must not carry an accepted SKU in {name}.")
        if item["acceptance_status"] == "accepted" and item["sku_id"] in {"", "__unknown__"}:
            raise ValueError(f"Accepted prediction needs a SKU in {name}.")
        pred_by_photo[name].append({**item, "box": box})

    image_rows, matched_rows, totals_by_type = [], [], defaultdict(lambda: defaultdict(int))
    tp = fp = fn = sku_correct = sku_wrong = sku_unknown = accepted = duplicate_count = 0
    unverified_identity_matches = merged_detection_count = 0
    true_unknown = correctly_rejected_unknown = 0
    absolute_count_errors = []

    for name, record in sorted(reviewed.items()):
        truth = [item for item in record.get("instances", []) if item.get("eligible_for_bbox_scoring", True)]
        predictions = pred_by_photo.get(name, [])
        pairs = sorted(
            ((box_iou(pred["box"], gt["box_xyxy"]), p, g)
             for p, pred in enumerate(predictions) for g, gt in enumerate(truth)),
            reverse=True,
        )
        used_predictions, used_truth, matches = set(), set(), []
        for overlap, pidx, gidx in pairs:
            if overlap < threshold or pidx in used_predictions or gidx in used_truth:
                continue
            used_predictions.add(pidx)
            used_truth.add(gidx)
            matches.append((pidx, gidx, overlap))

        image_tp, image_fp, image_fn = len(matches), len(predictions) - len(matches), len(truth) - len(matches)
        tp, fp, fn = tp + image_tp, fp + image_fp, fn + image_fn
        absolute_count_errors.append(abs(len(predictions) - len(truth)))
        by_type = totals_by_type[record.get("dataset_type", "unknown")]
        by_type["images"] += 1
        by_type["tp"] += image_tp
        by_type["fp"] += image_fp
        by_type["fn"] += image_fn
        by_type["gt"] += len(truth)
        by_type["pred"] += len(predictions)
        by_type["count_abs_error"] += abs(len(predictions) - len(truth))
        linked = defaultdict(int)
        for gt in truth:
            if gt.get("source_detection_id") and gt.get("detection_relation") == "one_of_multiple_in_merged_detection":
                linked[gt["source_detection_id"]] += 1
        image_merged = sum(count > 1 for count in linked.values())
        merged_detection_count += image_merged

        for pidx, gidx, overlap in matches:
            pred, gt = predictions[pidx], truth[gidx]
            predicted_sku, true_sku = pred["sku_id"], gt["sku_id"]
            is_unknown = pred["acceptance_status"] == "abstained"
            identity_verified = true_sku == "__unknown__" or true_sku in confirmed_skus
            if not identity_verified:
                unverified_identity_matches += 1
            elif true_sku == "__unknown__":
                true_unknown += 1
                correctly_rejected_unknown += int(is_unknown)
            if not identity_verified:
                outcome = "unverified_identity_excluded"
            elif is_unknown:
                sku_unknown += 1
                outcome = "abstained"
            else:
                accepted += 1
                sku_correct += int(predicted_sku == true_sku)
                sku_wrong += int(predicted_sku != true_sku)
                outcome = "correct" if predicted_sku == true_sku else "incorrect"
            matched_rows.append({
                "photo_filename": name, "dataset_type": record.get("dataset_type", ""),
                "arrangement_group": record.get("arrangement_group", ""),
                "detection_id": pred["detection_id"], "ground_truth_instance_id": gt.get("instance_id", ""),
                "box_iou": overlap, "predicted_sku_id": predicted_sku, "ground_truth_sku_id": true_sku,
                "sku_outcome": outcome,
                "confidence": float(pred["confidence"]),
            })

        for pidx, pred in enumerate(predictions):
            if pidx not in used_predictions and any(box_iou(pred["box"], truth[gidx]["box_xyxy"]) >= threshold for gidx in used_truth):
                duplicate_count += 1
        image_rows.append({
            "photo_filename": name, "dataset_type": record.get("dataset_type", ""),
            "arrangement_group": record.get("arrangement_group", ""),
            "ground_truth_instances": len(truth), "predicted_instances": len(predictions),
            "true_positives": image_tp, "false_positives": image_fp, "false_negatives": image_fn,
            "merged_detections_marked_by_reviewer": image_merged,
            "absolute_count_error": abs(len(predictions) - len(truth)),
        })

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    split_metrics = {}
    for split, values in totals_by_type.items():
        split_precision = values["tp"] / (values["tp"] + values["fp"]) if values["tp"] + values["fp"] else 0.0
        split_recall = values["tp"] / (values["tp"] + values["fn"]) if values["tp"] + values["fn"] else 0.0
        split_metrics[split] = {
            "images": values["images"], "ground_truth_instances": values["gt"], "predicted_instances": values["pred"],
            "precision": split_precision, "recall": split_recall,
            "mean_absolute_count_error": values["count_abs_error"] / values["images"],
        }
    report = {
        "status": "scored_human_reviewed_scene_predictions", "iou_threshold": threshold,
        "prediction_csv_sha256": sha256_file(prediction_path), "review_complete_images": len(reviewed),
        "ground_truth_instances": tp + fn, "predicted_instances": tp + fp,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "detection_precision": precision, "detection_recall": recall,
        "detection_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "duplicate_detections": duplicate_count,
        "merged_detections_marked_by_reviewer": merged_detection_count,
        "mean_absolute_count_error_per_image": sum(absolute_count_errors) / len(absolute_count_errors),
        "matched_detections": len(matched_rows), "sku_correct": sku_correct,
        "sku_incorrect": sku_wrong, "sku_unknown": sku_unknown,
        "sku_accuracy_on_verified_matched": sku_correct / (sku_correct + sku_wrong + sku_unknown)
        if sku_correct + sku_wrong + sku_unknown else None,
        "sku_accuracy_accepted_only": sku_correct / accepted if accepted else None,
        "unverified_identity_matches_excluded_from_sku_metrics": unverified_identity_matches,
        "genuinely_unknown_ground_truth_instances": true_unknown,
        "correct_unknown_rejections": correctly_rejected_unknown,
        "unknown_rejection_recall": correctly_rejected_unknown / true_unknown if true_unknown else None,
        "metrics_by_capture_type": split_metrics,
        "limitations": [
            "Only review-complete images and scoreable individual boxes are included.",
            "Boxes are matched one-to-one by IoU; polygon and mask IoU are not evaluated.",
            "Repeated views are scored separately and never summed into a physical inventory count.",
            "A small scene set does not establish a general visibility or crowding limit.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / f"scene_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "per_image.csv", image_rows)
    _write_csv(output / "matched_predictions.csv", matched_rows)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["output_dir"] = str(output)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.predictions, args.iou)


if __name__ == "__main__":
    main()

