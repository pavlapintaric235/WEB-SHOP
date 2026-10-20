"""Score few-shot SKU recognition on automatic and human-provided product masks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from . import run_benchmark as fixed
from .detectors import configure_ultralytics
from .evaluate_mask_backends import MobileSAMProposals
from .hybrid_core import ROOT, abstaining_decision
from .mask_runtime import CALIBRATED_MASK_THRESHOLDS, MaskFewShotMatcher
from .segmentation_models import DEFAULT_SAM_WEIGHTS, YOLOPromptedMobileSAM, render_masks, resize_mask

os.environ.setdefault("HF_HOME", str(ROOT / ".venv" / "huggingface_cache"))

MODEL_NAMES = ("mobilenetv2", "siglip2", "siglip2+mobilenetv2")
ENCODER_NAMES = ("siglip2", "mobilenetv2")
MASK_IOU_THRESHOLD = 0.50


def match_masks(predictions: list[dict], truth_masks: list[np.ndarray]) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, target in enumerate(truth_masks):
            intersection = np.count_nonzero(prediction["mask"] & target)
            union = np.count_nonzero(prediction["mask"] | target)
            iou = intersection / union if union else 0.0
            candidates.append((float(iou), pred_index, truth_index))
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    used_predictions, used_truth = set(), set()
    matches = []
    for iou, pred_index, truth_index in candidates:
        if iou < MASK_IOU_THRESHOLD:
            break
        if pred_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(pred_index)
        used_truth.add(truth_index)
        matches.append((pred_index, truth_index, iou))
    return matches


def decision_for(row: dict, model_name: str) -> dict:
    if model_name == "siglip2+mobilenetv2":
        return {key: row.get(key) for key in
                ("sku", "status", "reason", "similarity", "runner_up_similarity", "margin", "ranked_scores")}
    scores = row.get("branch_scores", {}).get(model_name, {})
    minimum, margin = CALIBRATED_MASK_THRESHOLDS[frozenset({model_name})]
    return abstaining_decision(scores, minimum, margin)


def score_instances(truth_items: list[dict], predicted_rows: list[dict],
                    matches: list[tuple[int, int, float]]) -> dict:
    pair_by_truth = {truth_index: (pred_index, iou) for pred_index, truth_index, iou in matches}
    matched_predictions = {pred_index for pred_index, _, _ in matches}
    model_metrics = {}
    details = []
    for model_name in MODEL_NAMES:
        registered = {"correct": 0, "wrong": 0, "unknown": 0, "not_detected": 0}
        undefined = {"correctly_rejected": 0, "false_registered_assignment": 0, "not_detected": 0}
        matched = 0
        for truth_index, target in enumerate(truth_items):
            pair = pair_by_truth.get(truth_index)
            actual = target["sku"]
            if pair is None:
                category = "not_detected"
                if actual == "__undefined__":
                    undefined["not_detected"] += 1
                else:
                    registered["not_detected"] += 1
                prediction_index, iou, decision = None, None, None
            else:
                prediction_index, iou = pair
                decision = decision_for(predicted_rows[prediction_index], model_name)
                predicted = decision.get("sku")
                matched += 1
                if actual == "__undefined__":
                    category = "correctly_rejected" if predicted is None else "false_registered_assignment"
                    undefined[category] += 1
                elif predicted == actual:
                    category = "correct"
                    registered[category] += 1
                elif predicted is None:
                    category = "unknown"
                    registered[category] += 1
                else:
                    category = "wrong"
                    registered[category] += 1
            details.append({
                "model": model_name, "annotation_id": target["annotation_id"], "actual_sku": actual,
                "prediction_index": prediction_index, "mask_iou": iou,
                "predicted_sku": decision.get("sku") if decision else None,
                "decision": category,
                "top_five": decision.get("ranked_scores", []) if decision else [],
            })
        false_positive_predictions = len(predicted_rows) - len(matched_predictions)
        model_metrics[model_name] = {
            "matched_instances": matched,
            "registered_ground_truth": len(truth_items) - sum(x["sku"] == "__undefined__" for x in truth_items),
            "registered": registered,
            "undefined_ground_truth": sum(x["sku"] == "__undefined__" for x in truth_items),
            "undefined": undefined,
            "false_positive_masks": false_positive_predictions,
        }
    return {"metrics": model_metrics, "per_instance": details}


def detection_summary(truth_items: list[dict], predictions: list[dict],
                      matches: list[tuple[int, int, float]]) -> dict:
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(truth_items) - tp
    return {
        "ground_truth_instances": len(truth_items), "predicted_instances": len(predictions),
        "true_positives_iou50": tp, "false_positives": fp, "false_negatives": fn,
        "precision_iou50": tp / max(1, tp + fp), "recall_iou50": tp / max(1, tp + fn),
        "mean_matched_mask_iou": float(np.mean([iou for _, _, iou in matches])) if matches else None,
        "count_error": len(predictions) - len(truth_items),
    }


def summarize_human_rows(rows: list[dict]) -> dict:
    output = {}
    for model_name in MODEL_NAMES:
        registered = {"correct": 0, "wrong": 0, "unknown": 0}
        undefined = {"correctly_rejected": 0, "false_registered_assignment": 0}
        for target, row in rows:
            predicted = decision_for(row, model_name).get("sku")
            if target["sku"] == "__undefined__":
                key = "correctly_rejected" if predicted is None else "false_registered_assignment"
                undefined[key] += 1
            elif predicted == target["sku"]:
                registered["correct"] += 1
            elif predicted is None:
                registered["unknown"] += 1
            else:
                registered["wrong"] += 1
        output[model_name] = {
            "registered_instances": sum(x["sku"] != "__undefined__" for x, _ in rows),
            "registered": registered,
            "undefined_instances": sum(x["sku"] == "__undefined__" for x, _ in rows),
            "undefined": undefined,
        }
    return output


def serializable_row(row: dict) -> dict:
    result = {}
    for key, value in row.items():
        if key in {"mask", "branch_scores", "best_reference_by_branch"}:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path(r"C:\Users\U938\Downloads\inventory6.zip"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--sam-weights", type=Path, default=DEFAULT_SAM_WEIGHTS)
    args = parser.parse_args()
    output = args.output or ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017" / \
        f"hybrid_mask_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    history = fixed.load_history()
    coco = fixed.load_coco_zip(args.coco.expanduser().resolve(strict=True))
    products = fixed.load_catalog()
    configure_ultralytics()
    sam_only = MobileSAMProposals(args.sam_weights.expanduser().resolve(strict=True))
    yolo_sam = YOLOPromptedMobileSAM(sam_weights=args.sam_weights.expanduser().resolve(strict=True))
    matcher = MaskFewShotMatcher(products, ENCODER_NAMES, device="cpu", local_files_only=True)

    report = {
        "scope": "five October 17 development scenes; not final untouched evaluation",
        "eligible_ground_truth_policy": "Use only annotations not flagged ambiguous by the existing Inventory6 evaluator; source labels are unchanged.",
        "mask_match_rule": "One-to-one greedy mask IoU >= 0.50.",
        "recognition_thresholds": {
            model: {"minimum_similarity": CALIBRATED_MASK_THRESHOLDS[
                        frozenset(model.split("+"))][0],
                    "minimum_margin": CALIBRATED_MASK_THRESHOLDS[
                        frozenset(model.split("+"))][1],
                    "calibration_source": "fixed three-scene calibration; polygon-masked human crops"}
            for model in MODEL_NAMES
        },
        "held_out_caveat": "The two held-out photographs show the same arrangement; metrics are not independent-scene generalization.",
        "models": {"mobile_sam_automatic": {"per_image": []},
                   "sku110k_yolo_plus_mobile_sam": {"per_image": []}},
        "human_masks": {"per_image": [], "models": {}},
        "timing_note": "CPU timings are local, model-loading excluded; do not compare to GPU measurements.",
    }
    prediction_csv, human_csv = [], []
    score_by_sku = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "instances": 0, "correct": 0, "wrong": 0, "unknown": 0, "not_detected": 0
    })))

    for run_id in fixed.RUN_IDS:
        filename = Path(history[run_id]["original_image"]).name
        image_path = Path(history[run_id]["original_image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Original photograph unavailable: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")

        truth_items = [item for item in coco[filename] if not fixed.is_ambiguous(run_id, item)]
        truth_masks = [resize_mask(fixed.truth_mask(item), image.size) for item in truth_items]
        truth_instances = [
            {"annotation_id": item["annotation_id"], "sku": item["sku"],
             "mask": mask, "box": [0, 0, image.width, image.height]}
            for item, mask in zip(truth_items, truth_masks)
        ]

        human_started = perf_counter()
        human_recognized = matcher.recognize(image, truth_instances)
        human_seconds = perf_counter() - human_started
        human_pairs = list(zip(truth_items, human_recognized))
        human_metric = summarize_human_rows(human_pairs)
        report["human_masks"]["per_image"].append({
            "image": filename, "eligible_instances": len(truth_items),
            "ambiguous_instances_excluded": len(coco[filename]) - len(truth_items),
            "recognition_seconds": human_seconds, "models": human_metric,
        })
        for target, row in human_pairs:
            decisions = {model: decision_for(row, model) for model in MODEL_NAMES}
            human_csv.append({"image": filename, "annotation_id": target["annotation_id"],
                "actual_sku": target["sku"], **{
                    f"{model}_{field}": decisions[model].get(field)
                    for model in MODEL_NAMES for field in ("sku", "status", "similarity", "margin")
                }})

        truth_dir = output / "human_masks"
        truth_dir.mkdir(exist_ok=True)
        render_masks(image, truth_instances).save(truth_dir / f"{Path(filename).stem}.png")

        for name, backend, kwargs in (
            ("mobile_sam_automatic", sam_only, {}),
            ("sku110k_yolo_plus_mobile_sam", yolo_sam,
             {"conf": args.detector_conf, "max_det": args.max_det}),
        ):
            predictions, segmentation_timing = (
                backend.predict(image, **kwargs) if kwargs else backend.predict(image)
            )
            mask_matches = match_masks(predictions, truth_masks)
            recognition_started = perf_counter()
            recognized = matcher.recognize(image, predictions)
            recognition_seconds = perf_counter() - recognition_started
            recognition = score_instances(truth_items, recognized, mask_matches)
            scene = {
                "image": filename,
                "ambiguous_instances_excluded": len(coco[filename]) - len(truth_items),
                "detection": detection_summary(truth_items, predictions, mask_matches),
                "segmentation_timing": segmentation_timing,
                "recognition_seconds": recognition_seconds,
                "sku_models": recognition["metrics"],
            }
            report["models"][name]["per_image"].append(scene)
            overlay_dir = output / name
            overlay_dir.mkdir(exist_ok=True)
            render_masks(image, predictions).save(overlay_dir / f"{Path(filename).stem}.png")

            match_by_prediction = {
                pred_index: (truth_index, iou)
                for pred_index, truth_index, iou in mask_matches
            }
            for prediction_index, row in enumerate(recognized):
                truth_pair = match_by_prediction.get(prediction_index)
                target = truth_items[truth_pair[0]] if truth_pair else None
                for model in MODEL_NAMES:
                    decision = decision_for(row, model)
                    if target and target["sku"] != "__undefined__":
                        stats = score_by_sku[name][model][target["sku"]]
                        stats["instances"] += 1
                        stats[decision["sku"] == target["sku"] and "correct"
                              or ("unknown" if decision["sku"] is None else "wrong")] += 1
                    prediction_csv.append({
                        "image": filename, "backend": name, "prediction_index": prediction_index + 1,
                        "matched_annotation_id": target["annotation_id"] if target else None,
                        "actual_sku": target["sku"] if target else None,
                        "mask_iou": truth_pair[1] if truth_pair else None,
                        "model": model, "predicted_sku": decision.get("sku"),
                        "status": decision.get("status"), "similarity": decision.get("similarity"),
                        "margin": decision.get("margin"),
                        "top_five": json.dumps(decision.get("ranked_scores", []), ensure_ascii=False),
                    })
            for truth_index, target in enumerate(truth_items):
                if target["sku"] != "__undefined__" and truth_index not in {
                        matched_truth for _, matched_truth, _ in mask_matches}:
                    for model in MODEL_NAMES:
                        score_by_sku[name][model][target["sku"]]["instances"] += 1
                        score_by_sku[name][model][target["sku"]]["not_detected"] += 1

    for model in MODEL_NAMES:
        report["human_masks"]["models"][model] = {
            "registered_instances": sum(row["models"][model]["registered_instances"]
                                        for row in report["human_masks"]["per_image"]),
            "registered": {key: sum(row["models"][model]["registered"][key]
                                    for row in report["human_masks"]["per_image"])
                           for key in ("correct", "wrong", "unknown")},
            "undefined_instances": sum(row["models"][model]["undefined_instances"]
                                       for row in report["human_masks"]["per_image"]),
            "undefined": {key: sum(row["models"][model]["undefined"][key]
                                   for row in report["human_masks"]["per_image"])
                          for key in ("correctly_rejected", "false_registered_assignment")},
        }

    for backend in report["models"].values():
        per_image = backend["per_image"]
        tp = sum(row["detection"]["true_positives_iou50"] for row in per_image)
        fp = sum(row["detection"]["false_positives"] for row in per_image)
        fn = sum(row["detection"]["false_negatives"] for row in per_image)
        backend["summary"] = {
            "images": len(per_image),
            "ground_truth_instances": sum(row["detection"]["ground_truth_instances"] for row in per_image),
            "predicted_instances": sum(row["detection"]["predicted_instances"] for row in per_image),
            "true_positives_iou50": tp, "false_positives": fp, "false_negatives": fn,
            "precision_iou50": tp / max(1, tp + fp), "recall_iou50": tp / max(1, tp + fn),
            "mean_absolute_count_error": float(np.mean([
                abs(row["detection"]["count_error"]) for row in per_image])),
            "sku_models": {model: {
                "registered_gt_correct": sum(row["sku_models"][model]["registered"]["correct"]
                                             for row in per_image),
                "registered_gt_wrong": sum(row["sku_models"][model]["registered"]["wrong"]
                                           for row in per_image),
                "registered_gt_unknown": sum(row["sku_models"][model]["registered"]["unknown"]
                                             for row in per_image),
                "registered_gt_not_detected": sum(row["sku_models"][model]["registered"]["not_detected"]
                                                  for row in per_image),
                "undefined_correctly_rejected": sum(row["sku_models"][model]["undefined"]["correctly_rejected"]
                                                     for row in per_image),
                "undefined_false_registered_assignment": sum(
                    row["sku_models"][model]["undefined"]["false_registered_assignment"] for row in per_image),
                "undefined_not_detected": sum(row["sku_models"][model]["undefined"]["not_detected"]
                                              for row in per_image),
            } for model in MODEL_NAMES},
        }
    report["per_sku_registered_results"] = {
        backend: {
            model: {sku: dict(values) for sku, values in sorted(skus.items())}
            for model, skus in score_by_sku[backend].items()
        }
        for backend in report["models"]
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "predicted_mask_recognition.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_csv[0]))
        writer.writeheader()
        writer.writerows(prediction_csv)
    with (output / "human_mask_recognition.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(human_csv[0]))
        writer.writeheader()
        writer.writerows(human_csv)

    lines = ["# End-to-End Mask and SKU Benchmark", "",
        "Five development scenes only. Calibration cutoffs come from human polygon masks; 01:42/01:44 are one held-out arrangement.", "",
        "| Arm | Masks | Precision@IoU.50 | Recall@IoU.50 | Count MAE | SigLIP2 correct / wrong / unknown / missed | Fusion correct / wrong / unknown / missed |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, arm in report["models"].items():
        summary = arm["summary"]
        sig = summary["sku_models"]["siglip2"]
        fusion = summary["sku_models"]["siglip2+mobilenetv2"]
        lines.append(f"| {name} | {summary['predicted_instances']} | {summary['precision_iou50']:.3f} | "
            f"{summary['recall_iou50']:.3f} | {summary['mean_absolute_count_error']:.2f} | "
            f"{sig['registered_gt_correct']} / {sig['registered_gt_wrong']} / {sig['registered_gt_unknown']} / {sig['registered_gt_not_detected']} | "
            f"{fusion['registered_gt_correct']} / {fusion['registered_gt_wrong']} / {fusion['registered_gt_unknown']} / {fusion['registered_gt_not_detected']} |")
    lines += ["", "## Human Masks", "",
        "| Encoder | Registered correct / wrong / unknown | Undefined rejected / false assignment |",
        "|---|---:|---:|"]
    for model, metric in report["human_masks"]["models"].items():
        lines.append(f"| {model} | {metric['registered']['correct']} / {metric['registered']['wrong']} / "
                     f"{metric['registered']['unknown']} | {metric['undefined']['correctly_rejected']} / "
                     f"{metric['undefined']['false_registered_assignment']} |")
    lines += ["", "These results are diagnostics, not independent-scene accuracy estimates. See report.json and per-object CSVs for per-image outcomes."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
