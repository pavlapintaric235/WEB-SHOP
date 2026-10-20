#!/usr/bin/env python3
"""Evaluate NVIDIA Retail DINO boxes against the isolated five-image CVAT holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np


MEAN_BGR = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
SCALE = 0.01735207357279195
EXPECTED_IMAGES = 5
EXPECTED_INSTANCES = 33


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nvidia_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "unreported (TensorRT provider/profile still required)"


def read_manifest(root: Path) -> list[dict[str, str]]:
    manifest = root / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} manifest records, found {len(rows)}")
    return rows


def read_truth(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    boxes = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if fields[0] != "0" or len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Expected class-0 polygon at {label_path}:{line_no}")
        coords = np.asarray([float(value) for value in fields[1:]], dtype=np.float32).reshape(-1, 2)
        if not np.isfinite(coords).all() or (coords < 0).any() or (coords > 1).any():
            raise ValueError(f"Invalid normalized polygon at {label_path}:{line_no}")
        xy = coords * np.asarray([width, height], dtype=np.float32)
        box = np.asarray([xy[:, 0].min(), xy[:, 1].min(),
                          xy[:, 0].max(), xy[:, 1].max()], dtype=np.float32)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Degenerate polygon box at {label_path}:{line_no}")
        boxes.append(box)
    return boxes


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(predictions: list[dict], truths: list[np.ndarray], threshold: float) -> dict:
    candidates = sorted(
        ((box_iou(pred["box"], truth), pi, ti)
         for pi, pred in enumerate(predictions) for ti, truth in enumerate(truths)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matches = []
    for overlap, pi, ti in candidates:
        if overlap < threshold:
            break
        if pi not in used_predictions and ti not in used_truths:
            used_predictions.add(pi)
            used_truths.add(ti)
            matches.append({"prediction_index": pi + 1, "ground_truth_index": ti + 1,
                            "box_iou": float(overlap)})
    tp = len(matches)
    fp, fn = len(predictions) - tp, len(truths) - tp
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / len(predictions) if predictions else 0.0,
        "recall": tp / len(truths) if truths else 0.0,
        "f1": 2 * tp / (len(predictions) + len(truths)) if predictions or truths else 0.0,
        "count_error": len(predictions) - len(truths),
        "mean_matched_box_iou": float(np.mean([row["box_iou"] for row in matches])) if matches else None,
        "matches": matches,
        "unmatched_prediction_indices": [i + 1 for i in range(len(predictions)) if i not in used_predictions],
        "unmatched_ground_truth_indices": [i + 1 for i in range(len(truths)) if i not in used_truths],
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def preprocess(image_bgr: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    import cv2

    source_height, source_width = image_bgr.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    tensor = ((canvas.astype(np.float32) - MEAN_BGR) * SCALE).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32), scale


def decode(outputs: dict[str, np.ndarray], image_shape: tuple[int, int],
           network_shape: tuple[int, int], scale: float, threshold: float) -> list[dict]:
    logits = np.asarray(outputs["pred_logits"], dtype=np.float32)
    boxes = np.asarray(outputs["pred_boxes"], dtype=np.float32)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if logits.ndim != 2 or logits.shape[1] != 2 or boxes.shape != (len(logits), 4):
        raise ValueError(f"Unexpected Retail DINO output shapes: {logits.shape}, {boxes.shape}")
    probabilities = sigmoid(logits)
    classes = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(probabilities)), classes]
    network_height, network_width = network_shape
    image_height, image_width = image_shape
    predictions = []
    for index in np.argsort(scores)[::-1]:
        score = float(scores[index])
        if classes[index] != 1 or score < threshold:
            continue
        cx, cy, bw, bh = boxes[index]
        coords = np.asarray([
            ((cx - bw / 2) * network_width) / scale,
            ((cy - bh / 2) * network_height) / scale,
            ((cx + bw / 2) * network_width) / scale,
            ((cy + bh / 2) * network_height) / scale,
        ], dtype=np.float32)
        coords[[0, 2]] = np.clip(coords[[0, 2]], 0, image_width)
        coords[[1, 3]] = np.clip(coords[[1, 3]], 0, image_height)
        if coords[2] > coords[0] and coords[3] > coords[1]:
            predictions.append({"box": coords, "confidence": score, "class_id": 1})
    return predictions


def draw_overlay(image_path: Path, truths: list[np.ndarray], predictions: list[dict],
                 metrics: dict, output_path: Path) -> None:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to reopen image for overlay: {image_path}")
    matches = {item["prediction_index"] - 1: item for item in metrics["matches"]}
    for index, box in enumerate(truths, 1):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(image, f"GT{index}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 220, 0), 2, cv2.LINE_AA)
    for index, prediction in enumerate(predictions):
        box = prediction["box"]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        matched = matches.get(index)
        color = (255, 200, 0) if matched else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"P{index + 1} {prediction['confidence']:.2f}"
        if matched:
            label += f" GT{matched['ground_truth_index']} IoU {matched['box_iou']:.2f}"
        cv2.putText(image, label, (x1, min(image.shape[0] - 8, y2 + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Unable to write overlay: {output_path}")


def main() -> int:
    import cv2
    import onnxruntime as ort

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Extracted CVAT holdout package root")
    parser.add_argument("--model", type=Path, required=True, help="Isolated NVIDIA DINO ONNX with TRT domain remapped")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.50)
    args = parser.parse_args()

    root = args.dataset.resolve(strict=True)
    model_path = args.model.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"TensorRT EP unavailable: {ort.get_available_providers()}")

    rows = read_manifest(root)
    total_truths = sum(int(row["instances"]) for row in rows)
    if total_truths != EXPECTED_INSTANCES:
        raise ValueError(f"Expected {EXPECTED_INSTANCES} manifest instances, found {total_truths}")

    images_dir = root / "dataset" / "images" / "val"
    labels_dir = root / "dataset" / "labels" / "val"
    output.mkdir(parents=True)
    (output / "runtime").mkdir()
    engine_cache = output / "engine_cache"
    engine_cache.mkdir()

    # Validate packaged source bytes before loading any image.
    image_records = []
    for row in rows:
        image_path = root / row["dataset_image"]
        label_path = root / row["dataset_label"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Manifest image/label missing: {image_path} / {label_path}")
        if sha256(image_path) != row["image_sha256"]:
            raise ValueError(f"Image SHA-256 mismatch: {image_path.name}")
        if sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"Label SHA-256 mismatch: {label_path.name}")
        image_records.append((row, image_path, label_path))
    if len(list(images_dir.glob("*"))) != EXPECTED_IMAGES or len(list(labels_dir.glob("*.txt"))) != EXPECTED_IMAGES:
        raise ValueError("Holdout directory contents differ from the five-image manifest")

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(output / "runtime" / "nvidia_retail_dino_trt")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {
        "trt_engine_cache_enable": "True",
        "trt_engine_cache_path": str(engine_cache),
        "trt_fp16_enable": "False",
    }
    session = ort.InferenceSession(
        str(model_path), sess_options=options,
        providers=[("TensorrtExecutionProvider", provider_options), "CUDAExecutionProvider"],
    )
    providers = session.get_providers()
    if not providers or providers[0] != "TensorrtExecutionProvider":
        raise RuntimeError(f"TensorRT is not the selected primary provider: {providers}")
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    batch_is_compatible = len(shape) == 4 and (not isinstance(shape[0], int) or shape[0] == 1)
    if (not batch_is_compatible or shape[1] != 3 or
            not isinstance(shape[2], int) or not isinstance(shape[3], int)):
        raise RuntimeError(f"Unexpected DINO input shape: {shape}")
    net_h, net_w = int(shape[2]), int(shape[3])
    output_names = {item.name for item in session.get_outputs()}
    if not {"pred_logits", "pred_boxes"}.issubset(output_names):
        raise RuntimeError(f"Unexpected DINO outputs: {output_names}")

    # Build/cache the TensorRT engine outside scored image timings.
    first_row, first_image_path, _ = image_records[0]
    first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Unable to load {first_image_path}")
    warm_tensor, _ = preprocess(first_image, net_w, net_h)
    warm_start = time.perf_counter()
    session.run(None, {input_meta.name: warm_tensor})
    warmup_seconds = time.perf_counter() - warm_start

    per_image = []
    prediction_rows = []
    overlay_dir = output / "overlays"
    for row, image_path, label_path in image_records:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to load {image_path}")
        image_h, image_w = image.shape[:2]
        truths = read_truth(label_path, image_w, image_h)
        if len(truths) != int(row["instances"]):
            raise ValueError(
                f"Label count mismatch for {image_path.name}: manifest={row['instances']}, labels={len(truths)}"
            )
        prep_start = time.perf_counter()
        tensor, resize_scale = preprocess(image, net_w, net_h)
        preprocess_seconds = time.perf_counter() - prep_start
        inference_start = time.perf_counter()
        raw = session.run(None, {input_meta.name: tensor})
        inference_seconds = time.perf_counter() - inference_start
        decode_start = time.perf_counter()
        named_outputs = dict(zip([item.name for item in session.get_outputs()], raw))
        predictions = decode(named_outputs, (image_h, image_w), (net_h, net_w), resize_scale, args.confidence)
        postprocess_seconds = time.perf_counter() - decode_start
        metrics = match_boxes(predictions, truths, args.match_iou)
        draw_overlay(image_path, truths, predictions, metrics,
                     overlay_dir / f"{image_path.stem}_nvidia_retail_dino.jpg")
        per_image.append({
            "image": image_path.name,
            "scene_group": row["scene_group"],
            "image_sha256": row["image_sha256"],
            "ground_truth_instances": len(truths),
            "predicted_instances": len(predictions),
            "box_metrics": metrics,
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
        })
        for index, prediction in enumerate(predictions, 1):
            x1, y1, x2, y2 = [float(value) for value in prediction["box"]]
            match = next((item for item in metrics["matches"] if item["prediction_index"] == index), None)
            prediction_rows.append({
                "image": image_path.name,
                "prediction_index": index,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": prediction["confidence"],
                "matched_ground_truth_index": match["ground_truth_index"] if match else "",
                "box_iou": match["box_iou"] if match else "",
            })

    profile_path = Path(session.end_profiling())
    profile_events = json.loads(profile_path.read_text(encoding="utf-8"))
    node_events = [event for event in profile_events if event.get("cat") == "Node" and
                   event.get("ph") == "X" and "provider" in event.get("args", {})]
    node_provider_counts: dict[str, int] = {}
    for event in node_events:
        name = str(event["args"]["provider"])
        node_provider_counts[name] = node_provider_counts.get(name, 0) + 1
    if node_provider_counts.get("TensorrtExecutionProvider", 0) == 0:
        raise RuntimeError(f"No TensorRT execution nodes found in ORT profile: {node_provider_counts}")
    if any("CPUExecutionProvider" in key for key in node_provider_counts):
        raise RuntimeError(f"CPU graph execution detected; refusing to report GPU benchmark: {node_provider_counts}")

    tp = sum(row["box_metrics"]["true_positives"] for row in per_image)
    fp = sum(row["box_metrics"]["false_positives"] for row in per_image)
    fn = sum(row["box_metrics"]["false_negatives"] for row in per_image)
    gt_count = sum(row["ground_truth_instances"] for row in per_image)
    pred_count = sum(row["predicted_instances"] for row in per_image)
    report = {
        "status": "PASS",
        "scope": "Separate five-scene CVAT holdout evaluation; no training, threshold tuning, or production data changes.",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "images": len(per_image),
        "ground_truth_instances": gt_count,
        "predicted_instances": pred_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "micro_precision": tp / pred_count if pred_count else 0.0,
        "micro_recall": tp / gt_count if gt_count else 0.0,
        "micro_f1": 2 * tp / (pred_count + gt_count) if pred_count + gt_count else 0.0,
        "mean_absolute_count_error": float(np.mean([abs(row["box_metrics"]["count_error"]) for row in per_image])),
        "total_count_error_pred_minus_gt": pred_count - gt_count,
        "timing_seconds": {
            "engine_build_warmup_excluded_from_image_totals": warmup_seconds,
            "preprocess_total": sum(row["preprocess_seconds"] for row in per_image),
            "detector_inference_total": sum(row["inference_seconds"] for row in per_image),
            "postprocess_total": sum(row["postprocess_seconds"] for row in per_image),
        },
        "thresholds": {"product_confidence": args.confidence, "one_to_one_box_iou": args.match_iou},
        "runtime": {
            "gpu": nvidia_gpu_name(),
            "onnxruntime": ort.__version__,
            "providers": providers,
            "input_name": input_meta.name,
            "input_shape": shape,
            "profile": str(profile_path),
            "profile_node_provider_counts": node_provider_counts,
        },
        "per_image": per_image,
        "limitations": [
            "Only five grouped validation scenes; treat metrics as diagnostic, not a reliable generalization estimate.",
            "NVIDIA Retail DINO is a product detector only; this evaluation does not perform SKU recognition or segmentation.",
            "Polygon ground truth is reduced to its enclosing bounding box; this is box detection evaluation, not mask IoU.",
            "A matched box means IoU >= the fixed 0.50 threshold under one-to-one greedy matching.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["image", "scene_group", "ground_truth_instances", "predicted_instances", "true_positives",
                  "false_positives", "false_negatives", "precision", "recall", "f1", "count_error",
                  "preprocess_seconds", "inference_seconds", "postprocess_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in per_image:
            writer.writerow({"image": row["image"], "scene_group": row["scene_group"],
                             "ground_truth_instances": row["ground_truth_instances"],
                             "predicted_instances": row["predicted_instances"],
                             **{key: row["box_metrics"][key] for key in ("true_positives", "false_positives",
                                  "false_negatives", "precision", "recall", "f1", "count_error")},
                             "preprocess_seconds": row["preprocess_seconds"],
                             "inference_seconds": row["inference_seconds"],
                             "postprocess_seconds": row["postprocess_seconds"]})
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]) if prediction_rows else
                                ["image", "prediction_index", "x1", "y1", "x2", "y2", "confidence",
                                 "matched_ground_truth_index", "box_iou"])
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps({key: report[key] for key in (
        "status", "images", "ground_truth_instances", "predicted_instances", "true_positives",
        "false_positives", "false_negatives", "micro_precision", "micro_recall", "micro_f1",
        "mean_absolute_count_error", "total_count_error_pred_minus_gt", "timing_seconds", "runtime",
    )}, indent=2))
    print(f"RESULTS_DIR={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
