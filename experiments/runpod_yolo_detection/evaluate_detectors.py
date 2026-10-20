"""Compare retail-pretrained and fine-tuned generic product detectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

def list_images(source: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if source.is_file():
        if source.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported image file: {source}")
        return [source]
    if source.is_dir():
        images = sorted(path for path in source.rglob("*") if path.suffix.lower() in extensions)
        if images:
            return images
    raise FileNotFoundError(f"No supported images found under {source}")


def load_expected_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "images" in data:
        return {str(name): int(item["total_products"]) for name, item in data["images"].items()}
    if "source_image" in data and "total_products" in data:
        return {str(data["source_image"]): int(data["total_products"])}
    raise ValueError("Expected-count JSON must contain images mapping or source_image/total_products")


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def load_ground_truth_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError(f"Expected detection row '0 cx cy width height' at {label_path}:{line_no}")
        cx, cy, box_w, box_h = map(float, fields[1:])
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < box_w <= 1 and 0 < box_h <= 1):
            raise ValueError(f"Invalid normalized ground-truth box at {label_path}:{line_no}")
        if cx - box_w / 2 < -1e-7 or cy - box_h / 2 < -1e-7 or cx + box_w / 2 > 1.0000001 or cy + box_h / 2 > 1.0000001:
            raise ValueError(f"Ground-truth box extends outside the image at {label_path}:{line_no}")
        boxes.append([
            (cx - box_w / 2) * width,
            (cy - box_h / 2) * height,
            (cx + box_w / 2) * width,
            (cy + box_h / 2) * height,
        ])
    return boxes


def score_iou(predictions: list[dict], ground_truth: list[list[float]], threshold: float = 0.5) -> dict:
    matched: set[int] = set()
    true_positives = 0
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        candidates = [(box_iou(prediction["xyxy"], truth), index) for index, truth in enumerate(ground_truth) if index not in matched]
        if candidates:
            best_iou, best_index = max(candidates)
            if best_iou >= threshold:
                matched.add(best_index)
                true_positives += 1
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return {
        "iou_threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0,
        "recall": true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0,
    }


def run_model(label: str, weights: Path, images: list[Path], output_dir: Path, args: argparse.Namespace) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if getattr(model, "task", None) != "detect":
        raise ValueError(f"{label} model is not a detection model")
    started = perf_counter()
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        save=False,
    )
    elapsed = perf_counter() - started
    model_dir = output_dir / label
    model_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for image_path, result in zip(images, results):
        boxes = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().tolist()
            confidence = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for coords, score, class_id in zip(xyxy, confidence, classes):
                boxes.append({"xyxy": [round(float(value), 2) for value in coords], "confidence": round(float(score), 6), "class_id": int(class_id)})
        plotted = result.plot()
        from PIL import Image
        import numpy as np

        Image.fromarray(np.asarray(plotted)[:, :, ::-1]).save(model_dir / f"{image_path.stem}_detections.jpg", quality=92)
        rows.append({
            "image": image_path.name,
            "model": label,
            "count": len(boxes),
            "boxes": boxes,
            "timing_ms": getattr(result, "speed", None),
            "model_batch_elapsed_seconds": round(elapsed, 6),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Retail-pretrained YOLO11n .pt checkpoint")
    parser.add_argument("--fine-tuned", type=Path, required=True, help="Fine-tuned best.pt or last.pt")
    parser.add_argument("--source", type=Path, required=True, help="Untouched test image or directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; must not already exist")
    parser.add_argument("--expected-counts", type=Path, help="Optional count-only truth JSON; does not provide IoU labels")
    parser.add_argument("--labels-dir", type=Path, help="Optional YOLO detection-label directory for IoU@0.5 precision/recall")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    baseline = args.baseline.expanduser().resolve(strict=True)
    fine_tuned = args.fine_tuned.expanduser().resolve(strict=True)
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    images = list_images(source)
    expected = load_expected_counts(args.expected_counts.expanduser().resolve(strict=True) if args.expected_counts else None)
    labels_dir = args.labels_dir.expanduser().resolve(strict=True) if args.labels_dir else None
    output.mkdir(parents=True)
    records = run_model("retail_pretrained", baseline, images, output, args)
    records += run_model("fine_tuned", fine_tuned, images, output, args)
    comparison = []
    for image in images:
        base_record = next(row for row in records if row["model"] == "retail_pretrained" and row["image"] == image.name)
        tuned_record = next(row for row in records if row["model"] == "fine_tuned" and row["image"] == image.name)
        gt_count = expected.get(image.name)
        item = {
            "image": image.name,
            "ground_truth_total_count": gt_count,
            "retail_pretrained_count": base_record["count"],
            "fine_tuned_count": tuned_record["count"],
            "retail_pretrained_abs_count_error": abs(base_record["count"] - gt_count) if gt_count is not None else None,
            "fine_tuned_abs_count_error": abs(tuned_record["count"] - gt_count) if gt_count is not None else None,
        }
        if labels_dir is not None:
            from PIL import Image

            label_path = labels_dir / f"{image.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing ground-truth labels for {image.name}: {label_path}")
            with Image.open(image) as opened:
                width, height = opened.size
            truth = load_ground_truth_boxes(label_path, width, height)
            item["ground_truth_box_count"] = len(truth)
            item["retail_pretrained_iou_metrics"] = score_iou(base_record["boxes"], truth)
            item["fine_tuned_iou_metrics"] = score_iou(tuned_record["boxes"], truth)
        else:
            item["iou_metrics"] = "not computed: verified box annotations were not supplied"
        comparison.append(item)
    summary = {
        "scope": "generic product detection only; no SKU classification",
        "parameters": {"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "max_det": args.max_det, "device": args.device},
        "source_images": [str(path) for path in images],
        "count_comparison": comparison,
        "predictions": records,
        "note": "Count-only ground truth supports absolute count error. Detection precision/recall and IoU require human-verified object boxes.",
    }
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "images": len(images), "count_comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
