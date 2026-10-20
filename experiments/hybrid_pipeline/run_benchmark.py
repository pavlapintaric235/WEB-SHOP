"""Run the isolated hybrid diagnostic against Inventory6 without writing to production."""

from __future__ import annotations

import csv
import json
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .detectors import (RetailYOLODetector, YOLOEVisualPromptFallback,
                        configure_ultralytics)
from .hybrid_core import (MobileNetV2Encoder, abstaining_decision, build_reference_index,
                          crop_masked, load_catalog, match_crop, render_boxes,
                          scores_for_features)

ROOT = Path(__file__).resolve().parents[2]
RUN_IDS = [
    "20261017-014015-b2a337ec", "20261017-014239-235d7e51",
    "20261017-014423-e7b3fb42", "20261017-014554-b9412539",
    "20261017-014722-cc4ab8d3",
]
CALIBRATION = {RUN_IDS[0], RUN_IDS[3], RUN_IDS[4]}
DISPUTED_SKUS = {
    RUN_IDS[1]: {"ENERGIA-POWER-250", "JANA-ANTIOX-500"},
    RUN_IDS[2]: {"ENERGIA-POWER-250", "FENTIMANS-GINGER-200", "JANA-ANTIOX-500"},
    RUN_IDS[3]: {"DM-EINTOPF-400", "ENERGIA-POWER-250", "FRESHONA-BEANS-800"},
    RUN_IDS[4]: {"ENERGIA-POWER-250"},
}
DISPUTED_IDS = {
    RUN_IDS[1]: {20, 25, 32},
    RUN_IDS[2]: {48, 49, 50, 51},
    RUN_IDS[3]: {65},
}


def load_history() -> dict[str, dict]:
    rows = [json.loads(line) for line in (ROOT / "data/test_runs.jsonl").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    found = {row["run_id"]: row for row in rows if row.get("run_id") in RUN_IDS}
    if set(found) != set(RUN_IDS):
        raise RuntimeError("Expected five saved October 17 recognition snapshots.")
    return found


def load_coco_zip(path: Path) -> dict[str, list[dict]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if Path(name).name == "instances_default.json"]
        if len(names) != 1:
            raise ValueError("Expected one COCO instances_default.json in the Inventory6 ZIP.")
        manifest = json.loads(archive.read(names[0]))
    image_map = {item["id"]: item for item in manifest["images"]}
    categories = {item["id"]: item["name"] for item in manifest["categories"]}
    if len(image_map) != 5 or set(categories.values()) != {"product"}:
        raise ValueError("Expected exactly five COCO image records and one 'product' category.")
    basenames = [Path(item["file_name"]).name for item in image_map.values()]
    if len(set(basenames)) != len(basenames):
        raise ValueError("COCO image basenames are ambiguous.")
    output = {Path(item["file_name"]).name: [] for item in image_map.values()}
    for annotation in manifest["annotations"]:
        image = image_map[annotation["image_id"]]
        file_name = Path(image["file_name"]).name
        polygons = []
        for flat in annotation["segmentation"]:
            points = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
            if len(points) < 3 or not np.isfinite(points).all():
                raise ValueError(f"Invalid COCO polygon {annotation['id']}.")
            width, height = int(image["width"]), int(image["height"])
            if (points[:, 0].min() < 0 or points[:, 1].min() < 0
                    or points[:, 0].max() > width or points[:, 1].max() > height):
                raise ValueError(f"Polygon {annotation['id']} is outside its COCO image.")
            polygons.append(points)
        attrs = annotation.get("attributes") or {}
        output[file_name].append({
            "annotation_id": int(annotation["id"]),
            "sku": attrs.get("sku") or categories[annotation["category_id"]],
            "polygons": polygons,
            "size": (int(image["width"]), int(image["height"])),
        })
    if sum(map(len, output.values())) != 79:
        raise ValueError("Expected 79 Inventory6 annotations; refusing a mismatched export.")
    return output


def truth_mask(item: dict) -> np.ndarray:
    width, height = item["size"]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in item["polygons"]:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def bbox(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)] if len(xs) else None


def box_iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, (a[2] - a[0]) * (a[3] - a[1])) + max(0, (b[2] - b[0]) * (b[3] - b[1])) - intersection
    return intersection / union if union else 0.0


def pair_predictions(truth: list[dict], predictions: list[dict], threshold=0.5) -> list[tuple[int, int, float]]:
    """Maximum-IoU one-to-one assignment via scipy, with a dependency-free greedy fallback."""
    if not truth or not predictions:
        return []
    matrix = np.asarray([[box_iou(row["box"], pred["box"]) for pred in predictions] for row in truth])
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-matrix)
        return [(int(i), int(j), float(matrix[i, j])) for i, j in zip(rows, cols)
                if matrix[i, j] >= threshold]
    except ImportError:
        pairs = sorted(((matrix[i, j], i, j) for i in range(len(truth))
                        for j in range(len(predictions))), reverse=True)
        used_truth, used_pred, result = set(), set(), []
        for overlap, i, j in pairs:
            if overlap < threshold:
                break
            if i not in used_truth and j not in used_pred:
                used_truth.add(i); used_pred.add(j); result.append((i, j, float(overlap)))
        return result


def is_ambiguous(run_id: str, item: dict) -> bool:
    return (item["annotation_id"] in DISPUTED_IDS.get(run_id, set())
            or item["sku"] in DISPUTED_SKUS.get(run_id, set()))


def masked_crop(photo: Image.Image, mask: np.ndarray) -> Image.Image:
    return crop_masked(photo, mask)


def calibrate_threshold(rows: list[dict]) -> dict:
    candidates = []
    for minimum in np.arange(0.25, 0.901, 0.025):
        for margin in np.arange(0.0, 0.301, 0.025):
            utility = 0.0
            for row in rows:
                decision = abstaining_decision(row["scores"], float(minimum), float(margin))
                actual = row["sku"]
                predicted = decision["sku"]
                if actual == "__undefined__":
                    utility += 1.0 if predicted is None else -3.0
                elif predicted == actual:
                    utility += 2.0
                elif predicted is not None:
                    utility -= 3.0
            candidates.append((utility, float(minimum), float(margin)))
    best_utility = max(item[0] for item in candidates)
    # Prefer the stricter abstention pair among calibration ties.
    _, minimum, margin = max((row for row in candidates if row[0] == best_utility),
                             key=lambda row: (row[1] + row[2], row[1], row[2]))
    return {"minimum_similarity": minimum, "minimum_margin": margin,
            "calibration_objects": len(rows), "calibration_utility": best_utility,
            "rule": "max calibration utility (correct +2, wrong registered -3, undefined rejection +1, false assignment -3); ties prefer stricter cutoffs"}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path(r"C:\Users\U938\Downloads\inventory6.zip"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detector", choices=("sku110k", "yoloe"), default="sku110k")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Detector proposal cutoff; fixed and reported, not tuned on held-out scenes.")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.coco.is_file():
        raise FileNotFoundError(args.coco)

    history = load_history()
    truth_files = load_coco_zip(args.coco)
    expected_names = {Path(history[run_id]["original_image"]).name for run_id in RUN_IDS}
    if set(truth_files) != expected_names:
        raise ValueError(f"COCO image mapping differs from the saved five originals: {sorted(truth_files)}")
    products = load_catalog()
    skus = {item["sku"] for item in products}
    for items in truth_files.values():
        for item in items:
            if item["sku"] != "__undefined__" and item["sku"] not in skus:
                raise ValueError(f"COCO SKU is not in the read-only registry: {item['sku']}")

    encoder = MobileNetV2Encoder()
    index = build_reference_index(products, {encoder.name: encoder})
    output_dir = ROOT / "experiments/hybrid_pipeline/runs" / datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    configure_ultralytics()
    detector = (RetailYOLODetector(device=args.device) if args.detector == "sku110k"
                else YOLOEVisualPromptFallback(device=args.device))

    examples = []
    scene_data = {}
    calibration_vectors = []
    image_paths = {}
    for run_id in RUN_IDS:
        snapshot = history[run_id]
        filename = Path(snapshot["original_image"]).name
        photo_path = Path(snapshot["original_image"])
        if not photo_path.is_file():
            photo_path = ROOT / "data/test_runs" / filename
        if not photo_path.is_file() or filename not in truth_files:
            raise FileNotFoundError(f"Original photo/COCO mapping missing for {filename}.")
        with Image.open(photo_path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        if any(item["size"] != photo.size for item in truth_files[filename]):
            raise ValueError(f"COCO dimensions do not match original photograph {filename}.")
        image_paths[run_id] = photo_path
        gt = []
        crops = []
        for item in truth_files[filename]:
            mask = truth_mask(item)
            row = {**item, "mask": mask, "box": bbox(mask), "ambiguous": is_ambiguous(run_id, item)}
            gt.append(row)
            crop = masked_crop(photo, mask)
            crops.append(crop)
            if run_id in CALIBRATION and not row["ambiguous"]:
                calibration_vectors.append((run_id, row, crop))
        start = perf_counter()
        detections = detector.predict(photo, products, conf=args.conf, image_size=args.imgsz)
        detector_seconds = perf_counter() - start
        print(f"Prepared {run_id}: {len(detections)} proposals in {detector_seconds:.2f}s", flush=True)
        for prediction in detections:
            x1, y1, x2, y2 = prediction["box"]
            prediction["crop"] = (masked_crop(photo, prediction["mask"])
                                  if prediction.get("mask") is not None and prediction["mask"].any()
                                  else photo.crop((max(0, int(x1)), max(0, int(y1)),
                                                   min(photo.width, int(x2)), min(photo.height, int(y2)))))
        scene_data[run_id] = {"photo": photo, "gt": gt, "crops": crops, "predictions": detections,
                              "detector_seconds": detector_seconds, "filename": filename,
                              "oracle_seconds": 0.0}

    # Calibration is fitted only on the predefined 01:40, 01:45, 01:47 scenes.
    calibration_crops = [row[2] for row in calibration_vectors]
    calibration_features = encoder.encode(calibration_crops)
    calibration_rows = []
    for (_, target, _), feature in zip(calibration_vectors, calibration_features):
        fused, _, _ = scores_for_features({encoder.name: feature}, index)
        calibration_rows.append({"sku": target["sku"], "scores": fused})
    settings = calibrate_threshold(calibration_rows)

    # Embed detection and human-mask crops in batches, then reuse those vectors for scoring.
    proposal_rows, oracle_rows = [], []
    for run_id, scene in scene_data.items():
        start = perf_counter()
        vectors = encoder.encode([item["crop"] for item in scene["predictions"]])
        for item, vector in zip(scene["predictions"], vectors):
            item["embedding"] = vector
            item["decision"] = abstaining_decision(
                scores_for_features({encoder.name: vector}, index)[0],
                settings["minimum_similarity"], settings["minimum_margin"])
        scene["matching_seconds"] = perf_counter() - start
        gt_crops = [masked_crop(scene["photo"], row["mask"]) for row in scene["gt"]]
        start = perf_counter()
        gt_vectors = encoder.encode(gt_crops)
        for row, vector in zip(scene["gt"], gt_vectors):
            scores, _, references = scores_for_features({encoder.name: vector}, index)
            row["decision"] = abstaining_decision(scores, settings["minimum_similarity"],
                                                   settings["minimum_margin"])
            row["best_reference"] = references[encoder.name].get(row["decision"].get("sku"))
        scene["oracle_matching_seconds"] = perf_counter() - start
        arm_name = ("C_yolo11_sku110k_mobilenetv2" if args.detector == "sku110k"
                    else "C_yoloe_visual_prompt_mobilenetv2")
        for arm, predicted, match_time in ((arm_name, scene["predictions"], scene["matching_seconds"]),):
            pairs = pair_predictions(scene["gt"], predicted)
            pair_map = {j: (i, overlap) for i, j, overlap in pairs}
            for j, item in enumerate(predicted):
                pair = pair_map.get(j)
                gt_row = scene["gt"][pair[0]] if pair else None
                item["matched_annotation_id"] = gt_row["annotation_id"] if gt_row else None
                item["matched_iou"] = pair[1] if pair else None
                item["matched_truth_sku"] = gt_row["sku"] if gt_row else None
                item["match_is_ambiguous"] = bool(gt_row and gt_row["ambiguous"])
                proposal_rows.append({"run_id": run_id, "filename": scene["filename"],
                                      "arm": arm, "annotation_id": item["matched_annotation_id"],
                                      "truth_sku": item["matched_truth_sku"],
                                      "predicted_sku": item["decision"].get("sku"),
                                      "decision": item["decision"]["status"],
                                      "similarity": item["decision"].get("similarity"),
                                      "runner_up_similarity": item["decision"].get("runner_up_similarity"),
                                      "margin": item["decision"].get("margin"),
                                      "iou": item["matched_iou"], "ambiguous_pair": item["match_is_ambiguous"]})
        for row in scene["gt"]:
            oracle_rows.append({"run_id": run_id, "filename": scene["filename"],
                                "annotation_id": row["annotation_id"], "truth_sku": row["sku"],
                                "predicted_sku": row["decision"].get("sku"),
                                "decision": row["decision"]["status"],
                                "similarity": row["decision"].get("similarity"),
                                "runner_up_similarity": row["decision"].get("runner_up_similarity"),
                                "margin": row["decision"].get("margin"),
                                "ambiguous": row["ambiguous"]})

    per_image = []
    for run_id, scene in scene_data.items():
        gt = scene["gt"]
        predictions = scene["predictions"]
        pairs = pair_predictions(gt, predictions)
        matched_gt = {i for i, _, _ in pairs}
        matched_pred = {j for _, j, _ in pairs}
        ignored_gt = {i for i, row in enumerate(gt) if row["ambiguous"]}
        ignored_pred = {j for i, j, _ in pairs if i in ignored_gt}
        valid_gt = [i for i, row in enumerate(gt) if not row["ambiguous"]]
        tp = sum(1 for i, j, _ in pairs if i not in ignored_gt)
        fp = len(predictions) - tp - len(ignored_pred)
        fn = sum(1 for i in valid_gt if i not in matched_gt)
        correct = wrong = unknown = 0
        for i, j, overlap in pairs:
            if gt[i]["ambiguous"]:
                continue
            sku = predictions[j]["decision"].get("sku")
            if gt[i]["sku"] == "__undefined__":
                if sku is None: unknown += 1
                else: wrong += 1
            elif sku == gt[i]["sku"]: correct += 1
            elif sku is None: unknown += 1
            else: wrong += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_image.append({"run_id": run_id, "filename": scene["filename"],
                          "split": "calibration" if run_id in CALIBRATION else "held_out_same_arrangement",
                          "ground_truth": len(gt), "scored_ground_truth": len(valid_gt),
                          "excluded_ambiguous_ground_truth": len(ignored_gt), "detections": len(predictions),
                          "ignored_ambiguous_predictions": len(ignored_pred),
                          "scored_predictions": len(predictions) - len(ignored_pred),
                          "matched_iou50": tp, "precision_iou50": p, "recall_iou50": r,
                          "false_positives": fp, "missed": fn,
                          "count_error": len(predictions) - len(ignored_pred) - len(valid_gt),
                          "absolute_count_error": abs(len(predictions) - len(ignored_pred) - len(valid_gt)),
                          "correct_sku": correct, "wrong_sku": wrong, "unknown": unknown,
                          "detector_seconds": scene["detector_seconds"],
                          "matching_seconds": scene["matching_seconds"],
                          "human_mask_matching_seconds": scene["oracle_matching_seconds"],
                          "end_to_end_seconds": scene["detector_seconds"] + scene["matching_seconds"]})
        overlay = render_boxes(scene["photo"], [{**item, "sku": item["decision"].get("sku"),
                                                   "similarity": item["decision"].get("similarity")}
                                                for item in predictions])
        overlay.save(output_dir / f"{run_id}_yolo11_mobilenet.png", optimize=True)

    tile_w, tile_h, header_h = 430, 470, 48
    contact = Image.new("RGB", (tile_w * 3, header_h + tile_h * len(RUN_IDS)), "white")
    contact_draw = ImageDraw.Draw(contact)
    headings = ["Original photo", "Existing MobileSAM overlay", "YOLO11 SKU-110K + MobileNetV2"]
    for column, title in enumerate(headings):
        contact_draw.text((column * tile_w + 10, 14), title, fill="#17212b")
    for row_index, run_id in enumerate(RUN_IDS):
        paths = [image_paths[run_id], ROOT / "data/test_runs" / f"{run_id}-test.png",
                 output_dir / f"{run_id}_yolo11_mobilenet.png"]
        for column, path in enumerate(paths):
            x, y = column * tile_w, header_h + row_index * tile_h
            tile = Image.new("RGB", (tile_w, tile_h), "#eef1f4")
            if path.is_file():
                with Image.open(path) as source:
                    preview = ImageOps.contain(source.convert("RGB"), (tile_w - 12, tile_h - 38))
                tile.paste(preview, ((tile_w - preview.width) // 2, 30 + (tile_h - 38 - preview.height) // 2))
            ImageDraw.Draw(tile).text((8, 6), run_id[-6:], fill="#17212b")
            contact.paste(tile, (x, y))
    contact.save(output_dir / "comparison_contact_sheet.png", optimize=True)

    # Historical baseline values are copied into this new report, never rewritten.
    historical_path = ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017/evaluation_inventory6_20261017/report.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    scored_human = [row for row in oracle_rows if not row["ambiguous"]]
    registered = [row for row in scored_human if row["truth_sku"] != "__undefined__"]
    unregistered = [row for row in scored_human if row["truth_sku"] == "__undefined__"]
    human_correct = sum(row["predicted_sku"] == row["truth_sku"] for row in registered)
    human_wrong = sum(row["predicted_sku"] not in (None, row["truth_sku"]) for row in registered)
    human_unknown = sum(row["predicted_sku"] is None for row in registered)
    per_sku_counts: dict[str, Counter] = {}
    for row in scored_human:
        stats = per_sku_counts.setdefault(row["truth_sku"], Counter())
        stats["instances"] += 1
        if row["truth_sku"] == "__undefined__":
            stats["correctly_rejected_unknown"] += row["predicted_sku"] is None
            stats["false_registered_assignment"] += row["predicted_sku"] is not None
        elif row["predicted_sku"] == row["truth_sku"]:
            stats["correct"] += 1
        elif row["predicted_sku"] is None:
            stats["unknown"] += 1
        else:
            stats["wrong"] += 1
    per_sku = [{"sku": sku, **dict(stats)} for sku, stats in sorted(per_sku_counts.items())]
    detection_summary = {}
    for split_name, subset in (("all_five", per_image),
                               ("calibration", [x for x in per_image if x["run_id"] in CALIBRATION]),
                               ("held_out", [x for x in per_image if x["run_id"] not in CALIBRATION])):
        tp = sum(row["matched_iou50"] for row in subset)
        fp = sum(row["false_positives"] for row in subset)
        fn = sum(row["missed"] for row in subset)
        detection_summary[split_name] = {
            "precision_iou50": tp / (tp + fp) if tp + fp else None,
            "recall_iou50": tp / (tp + fn) if tp + fn else None,
            "matched": tp, "false_positives": fp, "missed": fn,
            "mean_absolute_count_error": (sum(row["absolute_count_error"] for row in subset) / len(subset)
                                           if subset else None),
        }

    mobile_sam_rows = []
    for run_id, scene in scene_data.items():
        predictions = [{"box": row["box"], "sku": row.get("sku"), "status": row.get("status")}
                       for row in history[run_id].get("detections", []) if row.get("box")]
        pairs = pair_predictions(scene["gt"], predictions)
        matched_gt = {i for i, _, _ in pairs}
        ignored_gt = {i for i, row in enumerate(scene["gt"]) if row["ambiguous"]}
        ignored_predictions = {j for i, j, _ in pairs if i in ignored_gt}
        valid_gt = len(scene["gt"]) - len(ignored_gt)
        tp = sum(i not in ignored_gt for i, _, _ in pairs)
        fp = len(predictions) - tp - len(ignored_predictions)
        fn = sum(not scene["gt"][i]["ambiguous"] and i not in matched_gt
                 for i in range(len(scene["gt"])))
        correct = wrong = unknown = 0
        for i, j, _ in pairs:
            if scene["gt"][i]["ambiguous"]:
                continue
            actual = scene["gt"][i]["sku"]
            predicted = predictions[j].get("sku") if predictions[j].get("status") == "recognized" else None
            if actual == "__undefined__":
                if predicted is None: unknown += 1
                else: wrong += 1
            elif predicted == actual: correct += 1
            elif predicted is None: unknown += 1
            else: wrong += 1
        mobile_sam_rows.append({"run_id": run_id, "filename": scene["filename"],
                                "split": "calibration" if run_id in CALIBRATION else "held_out_same_arrangement",
                                "valid_ground_truth": valid_gt, "excluded_ambiguous_ground_truth": len(ignored_gt),
                                "final_detections": len(predictions),
                                "ignored_ambiguous_predictions": len(ignored_predictions),
                                "scored_predictions": len(predictions) - len(ignored_predictions),
                                "matched_box_iou50": tp, "false_positives": fp, "missed": fn,
                                "count_error": len(predictions) - len(ignored_predictions) - valid_gt,
                                "absolute_count_error": abs(len(predictions) - len(ignored_predictions) - valid_gt),
                                "precision_box_iou50": tp / (tp + fp) if tp + fp else 0.0,
                                "recall_box_iou50": tp / (tp + fn) if tp + fn else 0.0,
                                "correct_sku": correct, "wrong_sku": wrong, "unknown": unknown})

    def mobile_sam_summary(subset: list[dict]) -> dict:
        tp = sum(row["matched_box_iou50"] for row in subset)
        fp = sum(row["false_positives"] for row in subset)
        fn = sum(row["missed"] for row in subset)
        return {"precision_box_iou50": tp / (tp + fp) if tp + fp else None,
                "recall_box_iou50": tp / (tp + fn) if tp + fn else None,
                "matched": tp, "false_positives": fp, "missed": fn,
                "mean_absolute_count_error": (sum(row["absolute_count_error"] for row in subset) / len(subset)
                                               if subset else None)}

    def human_match_summary(subset: list[dict]) -> dict:
        registered_rows = [r for r in subset if r["truth_sku"] != "__undefined__"]
        undefined_rows = [r for r in subset if r["truth_sku"] == "__undefined__"]
        correct = sum(r["predicted_sku"] == r["truth_sku"] for r in registered_rows)
        wrong = sum(r["predicted_sku"] not in (None, r["truth_sku"]) for r in registered_rows)
        unknown = len(registered_rows) - correct - wrong
        rejected = sum(r["predicted_sku"] is None for r in undefined_rows)
        return {"registered_instances": len(registered_rows), "correct": correct,
                "wrong_registered_assignment": wrong, "unknown": unknown,
                "registered_accuracy": correct / len(registered_rows) if registered_rows else None,
                "undefined_instances": len(undefined_rows), "undefined_correctly_rejected": rejected,
                "undefined_false_registered_assignment": len(undefined_rows) - rejected}

    human_split_rows = {"all_five": human_match_summary(scored_human),
                        "calibration": human_match_summary([r for r in scored_human if r["run_id"] in CALIBRATION]),
                        "held_out": human_match_summary([r for r in scored_human if r["run_id"] not in CALIBRATION])}

    report = {
        "purpose": "Isolated experimental benchmark; five development photos only, not untouched evaluation.",
        "models": {"detector": ("SKU-110K YOLO11n ONNX single-class retail proposal detector (research-only)"
                                  if args.detector == "sku110k" else
                                  "Local yoloe-26s-seg.pt visual-prompt fallback; NOT SKU-110K pretrained"),
                   "recognizer": "ImageNet-pretrained MobileNetV2 V2 pooled features + per-SKU normalized centroid cosine matching",
                   "dino_v3": "not run: gated local weights/repository absent",
                   "sku110k": "YOLO11n ONNX run when --detector sku110k; detection-only generic object class"},
        "split": {"calibration": sorted(CALIBRATION), "held_out": [x for x in RUN_IDS if x not in CALIBRATION],
                  "held_out_caveat": "01:42 and 01:44 are related views of one arrangement; holdout has only one scene group."},
        "ambiguity_exclusion": {"disputed_skus_by_run": {k: sorted(v) for k, v in DISPUTED_SKUS.items()},
                                "disputed_annotation_ids_by_run": {k: sorted(v) for k, v in DISPUTED_IDS.items()},
                                "approach": "Ignore annotations in disputed scene/SKU groups and explicit invalid/tiny IDs for scored metrics; retain them in raw output."},
        "calibration": settings,
        "arms": {"A_mobile_sam_historical_inventory6": historical["arms"].get("A_mobile_sam"),
                 "A_mobile_sam_filtered_box_iou50": {
                     "all_five": mobile_sam_summary(mobile_sam_rows),
                     "calibration": mobile_sam_summary([r for r in mobile_sam_rows if r["run_id"] in CALIBRATION]),
                     "held_out": mobile_sam_summary([r for r in mobile_sam_rows if r["run_id"] not in CALIBRATION]),
                     "per_image": mobile_sam_rows,
                     "note": "Saved final MobileSAM outputs rescored against the same ambiguity-filtered GT; no new SAM inference."},
                 "E_human_masks_existing_resnet18_sift_historical": historical["arms"].get("B_human_masks"),
                 "B_sku110k_yolo_dinov3": {"status": "not run; SKU-110K detector is available, but DINOv3 gated weights are unavailable"},
                 "C_yolo11_sku110k_mobilenetv2" if args.detector == "sku110k" else "C_yoloe_visual_prompt_mobilenetv2": {"status": "run", "summary": detection_summary,
                     "per_image": per_image,
                     "warning": ("Generic retail object detector; pretrained on SKU-110K, subject to research/non-commercial source-data restrictions."
                                 if args.detector == "sku110k" else
                                 "Explicit visual-prompt fallback, not SKU-110K-pretrained." )},
                 "D_yolo11_sku110k_dinov3_mobilenetv2": {"status": "not run; DINOv3 weights unavailable"},
                 "human_masks_mobilenetv2": {"status": "run",
                     "split_summary": human_split_rows,
                     "registered": {"instances": len(registered), "correct": human_correct,
                         "wrong": human_wrong, "unknown": human_unknown,
                         "accuracy": human_correct / max(1, len(registered))},
                     "undefined": {"instances": len(unregistered),
                         "correctly_rejected": sum(row["predicted_sku"] is None for row in unregistered),
                         "false_registered_assignment": sum(row["predicted_sku"] is not None for row in unregistered)},
                     "per_sku_csv": "per_sku.csv", "per_object_csv": "human_mask_recognition.csv"}},
        "limitations": ["The human-mask arm here uses MobileNetV2 embeddings, not existing ResNet18/SIFT; historical report contains the existing matcher arm.",
                        "Per-scene timing is CPU-only and local; timings are not comparable to GPU timing.",
                        "Only the explicitly selected YOLOE fallback uses registered exemplars as visual prompts; SKU-110K detector results are generic object proposals."],
        "source_historical_report": str(historical_path.resolve()),
        "outputs": {"per_image_csv": "per_image.csv", "mobile_sam_csv": "mobile_sam_filtered.csv",
                    "per_sku_csv": "per_sku.csv", "per_proposal_csv": "proposals.csv",
                    "human_mask_csv": "human_mask_recognition.csv", "overlays": "*_yolo11_mobilenet.png",
                    "comparison_contact_sheet": "comparison_contact_sheet.png"},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, data in (("per_image.csv", per_image), ("mobile_sam_filtered.csv", mobile_sam_rows),
                       ("per_sku.csv", per_sku),
                       ("proposals.csv", proposal_rows),
                       ("human_mask_recognition.csv", oracle_rows)):
        if data:
            with (output_dir / name).open("w", newline="", encoding="utf-8") as stream:
                fieldnames = list(dict.fromkeys(key for row in data for key in row))
                writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader(); writer.writerows(data)
    lines = ["# Hybrid Prototype Benchmark", "", report["purpose"], "",
             f"MobileNetV2 calibration cutoff: similarity >= {settings['minimum_similarity']:.3f}, "
             f"margin >= {settings['minimum_margin']:.3f}.", "",
             "| Scene | Split | Valid GT | Predictions | P@IoU.50 | R@IoU.50 | Correct SKU | Wrong | Unknown | Detector s | Matcher s |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_image:
        lines.append(f"| {row['filename']} | {row['split']} | {row['scored_ground_truth']} | {row['detections']} | "
                     f"{row['precision_iou50']:.3f} | {row['recall_iou50']:.3f} | {row['correct_sku']} | "
                     f"{row['wrong_sku']} | {row['unknown']} | {row['detector_seconds']:.2f} | {row['matching_seconds']:.2f} |")
    c_all = detection_summary["all_five"]
    a_all = mobile_sam_summary(mobile_sam_rows)
    mean_detector = sum(row["detector_seconds"] for row in per_image) / len(per_image)
    mean_match = sum(row["matching_seconds"] for row in per_image) / len(per_image)
    lines += ["", "## Aggregate", "",
              f"YOLO: {c_all['matched']} matched; precision {c_all['precision_iou50']:.3f}, recall {c_all['recall_iou50']:.3f} at box IoU >= .50.",
              f"Saved MobileSAM: {a_all['matched']} matched; precision {a_all['precision_box_iou50']:.3f}, recall {a_all['recall_box_iou50']:.3f}, mean absolute count error {a_all['mean_absolute_count_error']:.2f} under the same box-IoU rule.",
              f"YOLO mean absolute count error: {c_all['mean_absolute_count_error']:.2f} after ignoring proposals paired to disputed annotations.",
              f"YOLO matched-object SKU outcomes: {sum(r['correct_sku'] for r in per_image)} correct, {sum(r['wrong_sku'] for r in per_image)} wrong registered assignment, {sum(r['unknown'] for r in per_image)} unknown.",
              f"MobileNetV2 human-mask crops: {human_correct}/{len(registered)} registered SKU correct, {human_wrong} wrong assignments, {human_unknown} unknown; undefined rejection {sum(row['predicted_sku'] is None for row in unregistered)}/{len(unregistered)}.",
              f"Mean CPU time per scene for YOLO + MobileNet: detector {mean_detector:.2f}s, matcher {mean_match:.2f}s, total {mean_detector + mean_match:.2f}s.",
              "Existing ResNet18/SIFT human-mask results are also included from the prior Inventory6 report, but that score uses all 79 unresolved annotations and is not directly comparable to this filtered sample.", "",
              "## Interpretation", "", ("The detector arm is the downloaded YOLO11n SKU-110K generic object detector. The checkpoint is research-only and inherits restrictive upstream data terms; it is not cleared for commercial deployment." if args.detector == "sku110k" else "The runnable detector column uses the installed YOLOE checkpoint with visual reference prompts, so it is a fallback demonstration rather than a generic retail-detector comparison."), "DINOv3 branches were not run because DINOv3 weights remain gated and unavailable.", "",
              "The five images are development diagnostics. The 01:42/01:44 pair is a single held-out arrangement group, and Inventory6 labels are provisional. Do not use these results as final generalization evidence."]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
