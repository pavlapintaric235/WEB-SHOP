"""Compare raw MobileSAM proposals with retail-YOLO boxes refined by MobileSAM."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from . import run_benchmark as fixed
from .hybrid_core import ROOT
from .segmentation_models import (DEFAULT_SAM_WEIGHTS, YOLOPromptedMobileSAM,
                                  render_masks, resize_mask)


class MobileSAMProposals:
    def __init__(self, weights: Path, device: str = "cpu"):
        from ultralytics import SAM
        from ultralytics.models.sam.predict import Predictor

        class ProposalPredictor(Predictor):
            def generate(self, im, *args, **kwargs):
                kwargs.update(points_stride=12, points_batch_size=16, crop_n_layers=0,
                              conf_thres=0.80, stability_score_thresh=0.92)
                return super().generate(im, *args, **kwargs)

        self.model = SAM(str(weights))
        self.predictor_class = ProposalPredictor
        self.device = device

    def predict(self, image: Image.Image) -> tuple[list[dict], dict]:
        started = perf_counter()
        working = image.convert("RGB").copy()
        working.thumbnail((1280, 1280))
        results = self.model.predict(source=working, predictor=self.predictor_class,
            device=self.device, imgsz=1024, conf=0.0, iou=0.7,
            verbose=False, save=False)
        elapsed = perf_counter() - started
        if not results or results[0].masks is None:
            return [], {"sam_seconds": elapsed, "proposal_count": 0}
        raw_masks = results[0].masks.data.detach().cpu().numpy()
        scores = results[0].boxes.conf.detach().cpu().numpy().tolist()
        output = []
        for index, raw in enumerate(raw_masks):
            mask = resize_mask(raw > 0.5, image.size)
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            output.append({"instance_id": index + 1,
                "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                "mask": mask, "detector_confidence": float(scores[index]) if index < len(scores) else 0.0,
                "backend": "mobile_sam_automatic"})
        return output, {"sam_seconds": elapsed, "proposal_count": len(output)}


def pair_masks(predictions: list[dict], truth: list[np.ndarray], threshold: float = 0.5) -> dict:
    pairs = []
    for pred_index, prediction in enumerate(predictions):
        for truth_index, target in enumerate(truth):
            intersection = np.count_nonzero(prediction["mask"] & target)
            union = np.count_nonzero(prediction["mask"] | target)
            pairs.append((intersection / union if union else 0.0, pred_index, truth_index))
    pairs.sort(reverse=True)
    used_predictions, used_truth, ious = set(), set(), []
    for iou, pred_index, truth_index in pairs:
        if iou < threshold:
            break
        if pred_index not in used_predictions and truth_index not in used_truth:
            used_predictions.add(pred_index)
            used_truth.add(truth_index)
            ious.append(float(iou))
    tp, fp, fn = len(used_predictions), len(predictions) - len(used_predictions), len(truth) - len(used_truth)
    return {"tp": tp, "fp": fp, "fn": fn,
        "precision_iou50": tp / max(1, tp + fp), "recall_iou50": tp / max(1, tp + fn),
        "mean_matched_mask_iou": float(np.mean(ious)) if ious else None,
        "predicted_count": len(predictions), "ground_truth_count": len(truth),
        "count_error": len(predictions) - len(truth), "absolute_count_error": abs(len(predictions) - len(truth))}


def box_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path(r"C:\Users\U938\Downloads\inventory6.zip"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=80)
    parser.add_argument("--sam-weights", type=Path, default=DEFAULT_SAM_WEIGHTS)
    args = parser.parse_args()
    output = args.output or ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017" / \
        f"mask_backend_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    history = fixed.load_history()
    coco = fixed.load_coco_zip(args.coco.expanduser().resolve(strict=True))
    sam_only = MobileSAMProposals(args.sam_weights.expanduser().resolve(strict=True))
    yolo_sam = YOLOPromptedMobileSAM(sam_weights=args.sam_weights.expanduser().resolve(strict=True))
    report = {"scope": "five existing October 17 development photographs; not final generalization evidence",
        "ground_truth_policy": "Existing CVAT annotations marked ambiguous by the fixed Inventory6 evaluator are excluded from scored masks, not modified.",
        "matching": "One-to-one maximum-IoU greedy mask match at IoU >= 0.50.",
        "models": {"mobile_sam_automatic": {"per_image": []}, "sku110k_yolo_plus_mobile_sam": {"per_image": []}},
        "limitations": ["The five diagnostic photographs have unresolved annotation flags and are development data.",
                        "The two fixed held-out images are two views of one arrangement.",
                        "No trained YOLO11-seg model is available locally; no results are imputed for it."]}

    for run_id in fixed.RUN_IDS:
        filename = Path(history[run_id]["original_image"]).name
        image_path = Path(history[run_id]["original_image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Original source photograph unavailable: {image_path}")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        truth_items = [item for item in coco[filename] if not fixed.is_ambiguous(run_id, item)]
        truth_masks = [fixed.truth_mask(item) for item in truth_items]
        truth_masks = [resize_mask(mask, image.size) for mask in truth_masks]
        truth_instances = [{"mask": mask, "box": box_from_mask(mask)} for mask in truth_masks]
        truth_dir = output / "human_masks"
        truth_dir.mkdir(exist_ok=True)
        render_masks(image, truth_instances).save(truth_dir / f"{Path(filename).stem}_ground_truth.png")
        (output / "mobile_sam_automatic").mkdir(exist_ok=True)
        (output / "sku110k_yolo_plus_mobile_sam").mkdir(exist_ok=True)

        for name, backend, kwargs in (
            ("mobile_sam_automatic", sam_only, {}),
            ("sku110k_yolo_plus_mobile_sam", yolo_sam,
             {"conf": args.detector_conf, "max_det": args.max_det}),
        ):
            predictions, timings = backend.predict(image, **kwargs) if kwargs else backend.predict(image)
            metrics = pair_masks(predictions, truth_masks)
            row = {"run_id": run_id, "image": filename, "ambiguous_ground_truth_excluded":
                   len(coco[filename]) - len(truth_items), "eligible_ground_truth": len(truth_items),
                   **metrics, "timings": timings}
            report["models"][name]["per_image"].append(row)
            render_masks(image, predictions).save(output / name / f"{Path(filename).stem}_overlay.png")

    for name, arm in report["models"].items():
        if not (output / name).exists():
            (output / name).mkdir()
        if arm["per_image"]:
            total_tp = sum(row["tp"] for row in arm["per_image"])
            total_fp = sum(row["fp"] for row in arm["per_image"])
            total_fn = sum(row["fn"] for row in arm["per_image"])
            arm["summary"] = {"images": len(arm["per_image"]),
                "eligible_ground_truth": sum(row["eligible_ground_truth"] for row in arm["per_image"]),
                "predicted_instances": sum(row["predicted_count"] for row in arm["per_image"]),
                "true_positives_iou50": total_tp, "false_positives": total_fp, "false_negatives": total_fn,
                "micro_precision_iou50": total_tp / max(1, total_tp + total_fp),
                "micro_recall_iou50": total_tp / max(1, total_tp + total_fn),
                "mean_absolute_count_error": float(np.mean([row["absolute_count_error"] for row in arm["per_image"]])),
                "mean_detector_seconds": float(np.mean([row["timings"].get("detector_seconds", 0.0)
                                                        for row in arm["per_image"]])),
                "mean_sam_seconds": float(np.mean([row["timings"].get("sam_seconds", 0.0)
                                                   for row in arm["per_image"]])),
            }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: arm.get("summary") for name, arm in report["models"].items()}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
