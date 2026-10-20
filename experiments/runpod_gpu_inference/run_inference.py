"""Run GPU-only detector and few-shot SKU inference on the packaged scenes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.chdir(ROOT)
sys.path.insert(0, str(PROJECT))

from experiments.hybrid_pipeline.mask_runtime import CALIBRATED_MASK_THRESHOLDS, MaskFewShotMatcher, masked_crop
from experiments.hybrid_pipeline.segmentation_models import YOLO11Segmentation, YOLOPromptedMobileSAM, render_masks
from experiments.hybrid_pipeline.hybrid_core import (abstaining_decision, fuse_sku_scores,
                                                      scores_for_reference_views)


def sync() -> None:
    import torch
    torch.cuda.synchronize()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def make_backend(name: str, weights: Path, device: str):
    if name == "sku110k_yolo_plus_mobile_sam":
        return YOLOPromptedMobileSAM(
            detector_weights=weights,
            sam_weights=PROJECT / "data" / "models" / "mobile_sam.pt",
            device=device,
        )
    if name == "yolo11n_seg_coco_pretrained":
        segmenter = YOLO11Segmentation(weights, device=device)
        segmenter.model.model.to(device)
        return segmenter
    raise ValueError(name)


def encode_and_match(matcher: MaskFewShotMatcher, crops: list[Image.Image]) -> tuple[dict, dict]:
    import torch

    output_features, encoder_seconds = {}, {}
    for name, encoder in matcher.encoders.items():
        sync()
        started = perf_counter()
        output_features[name] = encoder.encode(crops, batch_size=8 if name == "siglip2" else 32)
        sync()
        encoder_seconds[name] = perf_counter() - started
    rows = []
    scoring_start = perf_counter()
    for crop_index in range(len(crops)):
        branch_scores = {}
        best_references = {}
        decisions = {}
        for name in matcher.encoders:
            scores, _, references = scores_for_reference_views(
                {name: output_features[name][crop_index]}, matcher.index)
            branch_scores[name] = scores
            best_references[name] = references[name]
            minimum, margin = CALIBRATED_MASK_THRESHOLDS[frozenset({name})]
            decision = abstaining_decision(scores, minimum, margin)
            decision.update({
                "encoder": name,
                "branch_scores": scores,
                "best_reference_by_branch": references[name],
                "thresholds": {"minimum_similarity": minimum, "minimum_margin": margin,
                               "source": "existing calibration-only mask thresholds; unchanged"},
            })
            decisions[name] = decision
        combined = fuse_sku_scores(branch_scores)
        minimum, margin = CALIBRATED_MASK_THRESHOLDS[frozenset(matcher.encoders)]
        fused = abstaining_decision(combined, minimum, margin)
        fused.update({
            "encoder": "siglip2+mobilenetv2",
            "branch_scores": branch_scores,
            "best_reference_by_branch": best_references,
            "thresholds": {"minimum_similarity": minimum, "minimum_margin": margin,
                           "source": "existing calibration-only mask thresholds; unchanged"},
        })
        decisions["siglip2+mobilenetv2"] = fused
        rows.append(decisions)
    sync()
    encoder_seconds["score_and_abstention_cpu"] = perf_counter() - scoring_start
    return rows, encoder_seconds


def export_mask(path: Path, mask) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype("uint8"), mode="L").save(path)


def primary_view_summaries(raw: list[dict], scene_meta: dict[str, dict], count_truth: dict[str, dict],
                           sku_truth: dict[str, dict[str, int]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["backend"], row["recognition_mode"], row["scene_id"])].append(row)
    summaries = []
    for (backend, mode, scene_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item["manifest_order"])
        primary = rows[0]
        per_view = []
        non_additive_max: Counter = Counter()
        for row in rows:
            counts = Counter(item["sku"] for item in row["predictions"] if item["status"] == "recognized")
            non_additive_max |= counts
            per_view.append({"image": row["image"], "detection_count": len(row["predictions"]),
                             "recognized_count": sum(counts.values()), "unknown_count": sum(
                                 item["status"] != "recognized" for item in row["predictions"]),
                             "sku_counts": dict(sorted(counts.items()))})
        predicted_counts = Counter(item["sku"] for item in primary["predictions"]
                                   if item["status"] == "recognized")
        truth = count_truth.get(scene_id)
        score_total = truth is not None
        gt_sku = sku_truth.get(scene_id, {})
        scored_skus = []
        for sku, expected in sorted(gt_sku.items()):
            predicted = int(predicted_counts.get(sku, 0))
            scored_skus.append({"sku": sku, "expected": expected, "predicted": predicted,
                                "absolute_error": abs(predicted - expected)})
        meta = scene_meta.get(scene_id, {})
        summaries.append({
            "backend": backend,
            "recognition_mode": mode,
            "scene_id": scene_id,
            "view_count": len(rows),
            "view_group_class": meta.get("view_group_class"),
            "primary_view_rule": "first scene photo in the verified photos.csv manifest; no GT-based view selection",
            "primary_image": primary["image"],
            "per_view": per_view,
            "primary_detection_count": len(primary["predictions"]),
            "primary_recognized_sku_counts": dict(sorted(predicted_counts.items())),
            "primary_unknown_count": sum(item["status"] != "recognized" for item in primary["predictions"]),
            "non_additive_max_per_sku_across_views": dict(sorted(non_additive_max.items())),
            "multi_view_warning": "View counts are never summed. Max-per-SKU is evidence only, not a fused physical inventory count." if len(rows) > 1 else None,
            "user_confirmed_total": truth,
            "total_count_absolute_error": abs(len(primary["predictions"]) - truth["total"]) if score_total else None,
            "total_count_exact_match": len(primary["predictions"]) == truth["total"] if score_total else None,
            "user_confirmed_sku_counts_scored": scored_skus,
            "count_ground_truth_status": truth.get("status") if truth else "unverified_not_scored",
        })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-yolo11-seg", action="store_true",
                        help="Also run the separately downloaded COCO-pretrained YOLO11n-seg model")
    parser.add_argument("--conf", type=float, default=0.12,
                        help="Retail detector proposal floor; defaults to current experiment value")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: this package is GPU-only and will not silently fall back to CPU.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device visible")
    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    if "3090" not in gpu_name:
        print(f"WARNING: package target was RTX 3090; current GPU is {gpu_name}", flush=True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.init()

    registry_path = ROOT / "data" / "source" / "product_registry_experiment.json"
    all_products = json.loads(registry_path.read_text(encoding="utf-8"))
    products = [item for item in all_products if item.get("references")]
    if not products:
        raise RuntimeError("No reference-bearing experiment SKUs are available")

    setup_start = perf_counter()
    matcher = MaskFewShotMatcher(products, ("mobilenetv2", "siglip2"), device=device,
                                 local_files_only=True,
                                 siglip_model_path=ROOT / "models" / "siglip2")
    torch.cuda.synchronize()
    reference_setup_seconds = perf_counter() - setup_start
    for name, encoder in matcher.encoders.items():
        model = getattr(encoder, "model", None)
        parameter = next(model.parameters(), None) if model is not None else None
        if parameter is None or parameter.device.type != "cuda":
            raise RuntimeError(f"{name} encoder failed CUDA placement validation")

    retail_weights = PROJECT / "experiments" / "hybrid_pipeline" / "weights" / "sku110k-yolo11-n640.pt"
    backends = [("sku110k_yolo_plus_mobile_sam", make_backend("sku110k_yolo_plus_mobile_sam", retail_weights, device))]
    if args.include_yolo11_seg:
        seg_weights = retail_weights.parent / "yolo11n-seg.pt"
        if not seg_weights.is_file():
            raise FileNotFoundError("Run `python prepare_models.py --with-yolo11-seg` first")
        backends.append(("yolo11n_seg_coco_pretrained", make_backend("yolo11n_seg_coco_pretrained", seg_weights, device)))

    photos_path = ROOT / "data" / "source" / "photos.csv"
    photo_rows = [row for row in read_csv(photos_path) if row["role"] == "scene"]
    scene_meta = {row["scene_id"]: row for row in read_csv(ROOT / "data" / "source" / "scenes.csv")}
    draft_rows = read_csv(ROOT / "data" / "source" / "scene_counts_DRAFT.csv")
    sku_truth = defaultdict(dict)
    for row in draft_rows:
        if row["human_verified_YES_NO"].upper() == "YES":
            sku_truth[row["scene_id"]][row["experiment_sku"]] = int(row["proposed_quantity"])
    count_truth = {}
    for scene_id, row in scene_meta.items():
        if row.get("ground_truth_status", "").startswith("USER-CONFIRMED") and row.get("user_confirmed_total"):
            count_truth[scene_id] = {"total": int(row["user_confirmed_total"]),
                                     "status": row["ground_truth_status"]}

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    overlays = output / "overlays"
    masks_dir = output / "masks"
    per_image, flat_rows, manual_rows = [], [], []
    started_all = perf_counter()
    for photo_index, photo in enumerate(photo_rows):
        image_path = ROOT / "data" / "source" / "images" / photo["relative_path"]
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image_prediction_rows = {}
        for backend_name, backend in backends:
            torch.cuda.reset_peak_memory_stats()
            sync()
            photo_start = perf_counter()
            if backend_name == "sku110k_yolo_plus_mobile_sam":
                detections, segmentation_times = backend.predict(image, conf=args.conf, imgsz=1280,
                                                                   max_det=args.max_det)
                detector_seconds = segmentation_times["detector_seconds"]
                sam_seconds = segmentation_times["sam_seconds"]
                duplicate_pairs = segmentation_times.get("duplicate_proposal_pairs", [])
            else:
                detections, segmentation_times = backend.predict(image, conf=0.25, imgsz=960,
                                                                   max_det=args.max_det)
                detector_seconds = segmentation_times["model_seconds"]
                sam_seconds = 0.0
                duplicate_pairs = []
            sync()
            crop_started = perf_counter()
            crops = [masked_crop(image, item["mask"]) for item in detections]
            crop_seconds = perf_counter() - crop_started
            match_rows, encoder_times = encode_and_match(matcher, crops)
            recognition_modes = ("mobilenetv2", "siglip2", "siglip2+mobilenetv2")
            for mode in recognition_modes:
                predictions = []
                drawable = []
                for index, (detection, decision_rows) in enumerate(zip(detections, match_rows), 1):
                    decision = decision_rows[mode]
                    sku = decision.get("sku")
                    entry = next((item for item in products if item["sku"] == sku), None) if sku else None
                    row = {
                        "instance_id": index,
                        "box_xyxy_pixels": [float(value) for value in detection["box"]],
                        "detector_confidence": float(detection.get("detector_confidence", 0.0)),
                        "detector_class": int(detection.get("detector_class", 0)),
                        "sku": sku,
                        "status": decision["status"],
                        "unknown_reason": decision.get("reason") if decision["status"] != "recognized" else None,
                        "similarity": decision.get("similarity"),
                        "runner_up_similarity": decision.get("runner_up_similarity"),
                        "similarity_margin": decision.get("margin"),
                        "thresholds": decision.get("thresholds"),
                        "top5": decision.get("ranked_scores", []),
                        "best_reference_by_branch": decision.get("best_reference_by_branch", {}),
                        "branch_scores": decision.get("branch_scores", {}),
                        "catalog_sku_candidate": entry.get("catalog_sku_candidate") if entry else None,
                        "crosswalk_status": entry.get("crosswalk_status") if entry else None,
                        "mask_polygons_xy_pixels": detection.get("mask_polygons", []),
                        "mask_png": None,
                    }
                    mask_path = masks_dir / backend_name / safe_stem(photo["original_filename"]) / f"instance_{index:03d}.png"
                    export_mask(mask_path, detection["mask"])
                    row["mask_png"] = str(mask_path.relative_to(output)).replace("\\", "/")
                    predictions.append(row)
                    drawable.append({**detection, "sku": sku,
                                     "similarity": decision.get("similarity"),
                                     "status": decision["status"]})
                    if decision["status"] != "recognized":
                        manual_rows.append({"backend": backend_name, "mode": mode,
                                            "image": photo["original_filename"], "scene_id": photo["scene_id"],
                                            "instance_id": index, "issue": "unknown_or_abstained",
                                            "reason": decision.get("reason"),
                                            "top1_candidate": (decision.get("ranked_scores") or [{}])[0].get("sku"),
                                            "similarity": decision.get("similarity"),
                                            "margin": decision.get("margin")})
                    elif entry and entry.get("crosswalk_status") != "verified":
                        manual_rows.append({"backend": backend_name, "mode": mode,
                                            "image": photo["original_filename"], "scene_id": photo["scene_id"],
                                            "instance_id": index, "issue": "candidate_catalog_crosswalk",
                                            "reason": entry.get("crosswalk_status"), "top1_candidate": sku,
                                            "similarity": decision.get("similarity"),
                                            "margin": decision.get("margin")})

                overlay = render_masks(image, drawable)
                overlay_path = overlays / backend_name / mode / f"{safe_stem(photo['original_filename'])}.jpg"
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                overlay.save(overlay_path, quality=92)
                record = {
                    "backend": backend_name,
                    "recognition_mode": mode,
                    "image": photo["original_filename"],
                    "image_path": photo["relative_path"],
                    "scene_id": photo["scene_id"],
                    "view_label": photo["view_label"],
                    "manifest_order": photo_index,
                    "data_partition": photo["data_partition"],
                    "ground_truth_status": scene_meta.get(photo["scene_id"], {}).get("ground_truth_status"),
                    "overlay": str(overlay_path.relative_to(output)).replace("\\", "/"),
                    "detection_count": len(predictions),
                    "recognized_count": sum(row["status"] == "recognized" for row in predictions),
                    "unknown_count": sum(row["status"] != "recognized" for row in predictions),
                    "sku_counts": dict(sorted(Counter(row["sku"] for row in predictions
                                                         if row["status"] == "recognized").items())),
                    "duplicate_proposal_pairs": duplicate_pairs,
                    "timings_seconds": {
                        "detector_gpu": detector_seconds,
                        "mobile_sam_gpu": sam_seconds,
                        "mask_crop_cpu": crop_seconds,
                        "mobilenetv2_gpu": encoder_times.get("mobilenetv2", 0.0),
                        "siglip2_gpu": encoder_times.get("siglip2", 0.0),
                        "score_and_abstention_cpu": encoder_times.get("score_and_abstention_cpu", 0.0),
                        "reference_model_and_feature_setup_once_seconds": reference_setup_seconds,
                        "photo_total_seconds": perf_counter() - photo_start,
                    },
                    "gpu": {"device": gpu_name,
                            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
                            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1)},
                    "predictions": predictions,
                }
                image_prediction_rows[(backend_name, mode)] = record
                per_image.append(record)
                flat_rows.append(record)
                for first, second in duplicate_pairs:
                    manual_rows.append({"backend": backend_name, "mode": mode,
                                        "image": photo["original_filename"], "scene_id": photo["scene_id"],
                                        "instance_id": f"{first}|{second}",
                                        "issue": "near_identical_detector_box_proposals",
                                        "reason": "IoU >= 0.985; flagged only, not merged or removed"})
            backend_total_seconds = perf_counter() - photo_start
            for record in image_prediction_rows.values():
                if record["backend"] == backend_name and record["image"] == photo["original_filename"]:
                    record["timings_seconds"]["photo_total_seconds"] = backend_total_seconds
            print(f"[{photo_index + 1}/{len(photo_rows)}] {photo['original_filename']}: "
                  f"{sum(len(v['predictions']) for v in image_prediction_rows.values())} backend/mode rows so far",
                  flush=True)

    scene_summaries = primary_view_summaries(flat_rows, scene_meta, count_truth, sku_truth)
    scored_totals = [row for row in scene_summaries if row["user_confirmed_total"]]
    scored_skus = [sku_row for row in scene_summaries for sku_row in row["user_confirmed_sku_counts_scored"]]
    metrics = {}
    for backend_name in sorted({row["backend"] for row in scene_summaries}):
        for mode in ("mobilenetv2", "siglip2", "siglip2+mobilenetv2"):
            current = [row for row in scene_summaries if row["backend"] == backend_name and row["recognition_mode"] == mode]
            totals = [row for row in current if row["user_confirmed_total"]]
            sku_rows = [entry for row in current for entry in row["user_confirmed_sku_counts_scored"]]
            label = f"{backend_name}/{mode}"
            errors = [row["total_count_absolute_error"] for row in totals]
            metrics[label] = {
                "confirmed_scene_total_groups_scored": len(totals),
                "scene_total_exact_matches": sum(row["total_count_exact_match"] for row in totals),
                "scene_total_mae": sum(errors) / len(errors) if errors else None,
                "scene_total_results": [{"scene_id": row["scene_id"], "expected": row["user_confirmed_total"]["total"],
                                         "predicted_primary_view": row["primary_detection_count"],
                                         "absolute_error": row["total_count_absolute_error"]} for row in totals],
                "confirmed_sku_rows_scored": len(sku_rows),
                "confirmed_sku_exact_matches": sum(row["absolute_error"] == 0 for row in sku_rows),
                "confirmed_sku_mae": (sum(row["absolute_error"] for row in sku_rows) / len(sku_rows)
                                      if sku_rows else None),
                "sku_results": [{"scene_id": row["scene_id"], **sku} for row in current
                                for sku in row["user_confirmed_sku_counts_scored"]],
                "object_precision_recall_mask_iou": None,
                "object_level_metric_note": "Unavailable: no per-instance boxes/polygons are included in the reviewed input package.",
            }

    report = {
        "title": "InventoryLens RTX 3090 GPU inference experiment",
        "run_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "gpu": {"name": gpu_name, "torch": torch.__version__, "cuda_build": torch.version.cuda},
        "dataset": {"unique_original_photos": 41, "scene_photos_inferred": len(photo_rows),
                    "provided_reference_photos": sum(row["role"] == "reference" for row in read_csv(photos_path)),
                    "scene_groups": len(scene_meta), "models_not_trained": True,
                    "held_out_unmodified_photos_inferred_as_diagnostics": True},
        "runtime": {"total_wall_seconds": perf_counter() - started_all,
                    "reference_setup_seconds_once": reference_setup_seconds,
                    "stage_timings_are_device_synchronized": True},
        "metrics": metrics,
        "scene_summaries": scene_summaries,
        "limitations": [
            "No instance-level ground-truth masks or boxes are present. Detection precision, recall, duplicate rates against truth, and mask IoU cannot be computed.",
            "Only user-confirmed scene totals and explicit human-verified SKU count rows are scored; all remaining scene/SKU counts are provisional and unscored.",
            "A scene group is scored using its first manifest photo. Same-arrangement view counts are never summed.",
            "The non-additive max-per-SKU summary across views is evidence for manual review, not a count estimate.",
            "The model's returned unknowns are detector proposals rejected by existing calibration-only thresholds; truly unregistered-object rejection accuracy is unavailable without object-level labels.",
            "Unverified catalogue crosswalks and candidate mappings are flagged; their SKU accuracy should not be read as verified catalogue recognition.",
        ],
    }
    (output / "results.json").write_text(json.dumps({"report": report, "images": per_image}, indent=2,
                                                         ensure_ascii=False), encoding="utf-8")
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["backend", "recognition_mode", "image", "scene_id", "instance_id", "status", "sku",
                  "catalog_sku_candidate", "crosswalk_status", "detector_confidence", "similarity",
                  "runner_up_similarity", "similarity_margin", "unknown_reason", "mask_png",
                  "box_xyxy_pixels", "mask_polygons_xy_pixels", "top5", "best_reference_by_branch"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in per_image:
            for prediction in item["predictions"]:
                writer.writerow({"backend": item["backend"], "recognition_mode": item["recognition_mode"],
                                 "image": item["image"], "scene_id": item["scene_id"], **prediction,
                                 "box_xyxy_pixels": json.dumps(prediction["box_xyxy_pixels"]),
                                 "mask_polygons_xy_pixels": json.dumps(prediction["mask_polygons_xy_pixels"]),
                                 "top5": json.dumps(prediction["top5"]),
                                 "best_reference_by_branch": json.dumps(prediction["best_reference_by_branch"])})
    with (output / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted({key for row in manual_rows for key in row}) or ["backend", "mode", "image", "issue"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manual_rows)
    with (output / "scene_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        flat_metrics = []
        for scene in scene_summaries:
            flat_metrics.append({key: value for key, value in scene.items()
                                 if key not in {"per_view", "primary_recognized_sku_counts",
                                                "non_additive_max_per_sku_across_views", "user_confirmed_sku_counts_scored"}})
        fields = sorted({key for row in flat_metrics for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in flat_metrics:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    report_lines = [
        "# InventoryLens RunPod Inference Report", "",
        f"- GPU: {gpu_name} (PyTorch {torch.__version__}, CUDA build {torch.version.cuda})",
        f"- Scene photos inferred: {len(photo_rows)}; total original photos in source archive: 41",
        f"- Wall time: {report['runtime']['total_wall_seconds']:.1f} seconds",
        f"- Reference model/index setup: {reference_setup_seconds:.1f} seconds (one-time)", "",
        "## Interpretation", "",
        "This run reports count agreement only for user-confirmed totals and SKU rows. There are no object masks/boxes in the input package, so object detection precision/recall and segmentation IoU are not measurable. Unknown-rate is a model abstention rate, not a verified unknown-product rejection score. Same-arrangement images are not added together.", "",
        "## Confirmed-count metrics", "",
        "| Detector / matcher | Confirmed total groups | Exact totals | Total-count MAE | Confirmed SKU rows exact | SKU-count MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, values in metrics.items():
        report_lines.append(f"| {key} | {values['confirmed_scene_total_groups_scored']} | "
                            f"{values['scene_total_exact_matches']} | "
                            f"{values['scene_total_mae'] if values['scene_total_mae'] is not None else 'n/a'} | "
                            f"{values['confirmed_sku_exact_matches']}/{values['confirmed_sku_rows_scored']} | "
                            f"{values['confirmed_sku_mae'] if values['confirmed_sku_mae'] is not None else 'n/a'} |")
    report_lines.extend(["", "## Per-image overlays", "", "Open `overlays/` for masks/predictions. See `predictions.csv` for pixel-coordinate polygons and top candidates, `scene_metrics.csv` for confirmed-count scoring, and `manual_review.csv` for abstentions, candidate crosswalks and near-duplicate detector proposals.", ""])
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"output": str(output), "gpu": gpu_name, "scene_images": len(photo_rows),
                      "prediction_rows": len(per_image), "manual_review_flags": len(manual_rows),
                      "report": str(output / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
