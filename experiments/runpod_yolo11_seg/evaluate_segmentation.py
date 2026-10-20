"""Compare pretrained and fine-tuned YOLO11-seg checkpoints on a labeled split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent))
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True, help="Pretrained yolo11n-seg.pt")
    parser.add_argument("--fine-tuned", type=Path, required=True, help="Run's best.pt")
    parser.add_argument("--output", type=Path, required=True, help="New, empty output directory")
    parser.add_argument("--split", choices=("val", "test"), default="val",
                        help="Use validation now, or a separately supplied untouched test YAML after training.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def read_truth(path: Path, size: tuple[int, int]) -> list[np.ndarray]:
    width, height = size
    masks = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if fields[0] != "0" or len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError(f"Expected single-class YOLO segmentation polygon at {path}:{line_no}")
        points = np.asarray(list(map(float, fields[1:])), dtype=np.float32).reshape(-1, 2)
        polygon = np.round(points * np.asarray([width, height])).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 1)
        masks.append(mask.astype(bool))
    return masks


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def match_masks(predictions: list[dict], truth: list[np.ndarray], threshold: float = 0.5) -> dict:
    pairs = sorted(
        ((mask_iou(pred["mask"], target), pred_index, truth_index)
         for pred_index, pred in enumerate(predictions)
         for truth_index, target in enumerate(truth)), reverse=True,
    )
    used_pred, used_truth, matched_iou = set(), set(), []
    for overlap, pred_index, truth_index in pairs:
        if overlap < threshold:
            break
        if pred_index not in used_pred and truth_index not in used_truth:
            used_pred.add(pred_index)
            used_truth.add(truth_index)
            matched_iou.append(overlap)
    tp = len(used_pred)
    return {
        "iou_threshold": threshold,
        "true_positives": tp,
        "false_positives": len(predictions) - tp,
        "false_negatives": len(truth) - tp,
        "precision": tp / max(1, len(predictions)),
        "recall": tp / max(1, len(truth)),
        "matched_mask_iou_mean": float(np.mean(matched_iou)) if matched_iou else None,
        "count_error": len(predictions) - len(truth),
        "absolute_count_error": abs(len(predictions) - len(truth)),
    }


def evaluate_model(label: str, weights: Path, images: list[Path], labels_dir: Path,
                   output: Path, args) -> dict:
    model = YOLO(str(weights))
    if getattr(model, "task", None) != "segment":
        raise ValueError(f"{label} is not a segmentation checkpoint (task={model.task!r})")
    val_start = perf_counter()
    native_metrics = model.val(data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
                               device=args.device, plots=False, verbose=False)
    val_elapsed = perf_counter() - val_start
    seg_metrics = getattr(native_metrics, "seg", None)
    native_map = {
        "mask_map50": float(seg_metrics.map50) if seg_metrics is not None else None,
        "mask_map50_95": float(seg_metrics.map) if seg_metrics is not None else None,
        "box_map50": float(native_metrics.box.map50) if getattr(native_metrics, "box", None) else None,
        "box_map50_95": float(native_metrics.box.map) if getattr(native_metrics, "box", None) else None,
        "validation_seconds": val_elapsed,
    }
    per_image = []
    pred_dir = output / label
    pred_dir.mkdir(parents=True, exist_ok=False)
    for image_path in images:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        truth = read_truth(labels_dir / f"{image_path.stem}.txt", image.size)
        start = perf_counter()
        result = model.predict(source=image, imgsz=args.imgsz, conf=args.conf, device=args.device,
                               retina_masks=True, verbose=False, save=False)[0]
        inference_seconds = perf_counter() - start
        predictions = []
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(masks))
            for raw_mask, score in zip(masks, scores):
                if raw_mask.shape != (image.height, image.width):
                    raw_mask = cv2.resize(raw_mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST)
                mask = raw_mask > 0.5
                if mask.any():
                    predictions.append({"mask": mask, "confidence": float(score)})
        metrics = match_masks(predictions, truth)
        per_image.append({"image": image_path.name, "ground_truth_instances": len(truth),
                          "predicted_instances": len(predictions), "inference_seconds": inference_seconds,
                          **metrics})
        plotted = result.plot()
        Image.fromarray(np.asarray(plotted)[:, :, ::-1]).save(pred_dir / f"{image_path.stem}_overlay.jpg", quality=92)
    return {"checkpoint": str(weights.resolve()), "native_ultralytics_metrics": native_map,
            "per_image": per_image,
            "summary": {
                "images": len(per_image),
                "gt_instances": sum(row["ground_truth_instances"] for row in per_image),
                "predicted_instances": sum(row["predicted_instances"] for row in per_image),
                "micro_precision_iou50": sum(row["true_positives"] for row in per_image) /
                    max(1, sum(row["predicted_instances"] for row in per_image)),
                "micro_recall_iou50": sum(row["true_positives"] for row in per_image) /
                    max(1, sum(row["ground_truth_instances"] for row in per_image)),
                "mean_absolute_count_error": float(np.mean([row["absolute_count_error"] for row in per_image])),
                "mean_matched_mask_iou": float(np.mean([row["matched_mask_iou_mean"] for row in per_image
                                                          if row["matched_mask_iou_mean"] is not None]))
                    if any(row["matched_mask_iou_mean"] is not None for row in per_image) else None,
            }}


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output}")
    data = __import__("yaml").safe_load(args.data.resolve(strict=True).read_text(encoding="utf-8"))
    root = Path(data.get("path") or args.data.resolve().parent)
    if not root.is_absolute():
        root = (args.data.resolve().parent / root).resolve()
    if args.split not in data:
        raise ValueError(f"Dataset YAML has no {args.split!r} split")
    image_dir = root / data[args.split]
    labels_dir = root / "labels" / image_dir.name
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"No images found for split {args.split!r} under {image_dir}")
    if args.split == "val" and len(images) != 5:
        raise ValueError(f"Expected five grouped validation photographs, found {len(images)}")
    output.mkdir(parents=True)
    results = {}
    for label, checkpoint in (("pretrained", args.baseline), ("fine_tuned", args.fine_tuned)):
        results[label] = evaluate_model(label, checkpoint.expanduser().resolve(strict=True),
                                        images, labels_dir, output, args)
    report = {
        "scope": (f"{len(images)} grouped validation images only; not an untouched final test"
                  if args.split == "val" else
                  f"{len(images)} images from a separately supplied test split; no training or tuning"),
        "data_yaml": str(args.data.resolve()),
        "evaluated_split": args.split,
        "split_groups": ("5 distinct validation groups; all protected October 17 diagnostics and IMG_4699 excluded"
                         if args.split == "val" else
                         "Independent test group structure is the responsibility of the separately supplied manifest"),
        "thresholds": {"confidence": args.conf, "mask_iou": 0.5},
        "models": results,
        "caveat": "Native mask mAP is measured by Ultralytics; the supplemental one-to-one mask-IoU counts are thresholded at 0.50.",
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["summary"] | value["native_ultralytics_metrics"]
                      for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
