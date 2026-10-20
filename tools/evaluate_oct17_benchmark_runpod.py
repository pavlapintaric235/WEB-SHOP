"""Evaluate detector proposals and the unchanged SKU matcher on reviewed CVAT masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sam_vision as sam_module
from images import product_references
from model_assets import SAM_FILE
from sam_vision import (MAX_IMAGE_SIDE, REFERENCE_CACHE_VERSION, SAMRecognizer,
                        decode_mask)


RUN_IDS = [
    "20261017-014015-b2a337ec",
    "20261017-014239-235d7e51",
    "20261017-014423-e7b3fb42",
    "20261017-014554-b9412539",
    "20261017-014722-cc4ab8d3",
]


def annotation_sku(annotation: dict, category: str) -> str:
    attributes = annotation.get("attributes") or {}
    sku = attributes.get("sku") if category == "product" else category
    if not isinstance(sku, str) or not sku.strip():
        raise ValueError(f"Annotation {annotation.get('id')} is missing a selected SKU. "
                         "Choose a registered SKU or __undefined__ for a known unregistered product.")
    return sku


def load_history(path: Path) -> dict[str, dict]:
    contents = path.read_text(encoding="utf-8")
    if contents.lstrip().startswith("["):
        rows = json.loads(contents)
    else:
        rows = [json.loads(line) for line in contents.splitlines() if line.strip()]
    found = {row["run_id"]: row for row in rows if row.get("run_id") in RUN_IDS}
    if set(found) != set(RUN_IDS):
        raise ValueError("The five expected October 17 snapshots are not all present.")
    return found


def load_coco(path: Path, image_dir: Path) -> dict[str, list[dict]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            manifests = [name for name in archive.namelist()
                         if Path(name).name == "instances_default.json"]
            if len(manifests) != 1:
                raise ValueError("COCO ZIP must contain exactly one instances_default.json manifest.")
            coco = json.loads(archive.read(manifests[0]))
    else:
        coco = json.loads(path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco.get("images", [])}
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    if not images or not categories:
        raise ValueError("COCO export must contain images and SKU categories.")
    filenames = {image_id: Path(item["file_name"]).name for image_id, item in images.items()}
    if len(set(filenames.values())) != len(filenames):
        raise ValueError("COCO image basenames are not unique; refusing ambiguous mapping.")
    result: dict[str, list[dict]] = {filename: [] for filename in filenames.values()}
    for image in images.values():
        image_path = image_dir / filenames[image["id"]]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing reviewed image: {image_path}")
        with Image.open(image_path) as photo:
            oriented = ImageOps.exif_transpose(photo)
            if oriented.size != (int(image["width"]), int(image["height"])):
                raise ValueError(f"COCO dimensions do not match {image['file_name']}.")
    for annotation in coco.get("annotations", []):
        image = images.get(annotation.get("image_id"))
        category = categories.get(annotation.get("category_id"))
        if image is None or category is None:
            raise ValueError("COCO annotation has an unknown image/category reference.")
        sku = annotation_sku(annotation, category)
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list):
            raise ValueError("Expected polygon COCO segmentation; RLE requires explicit conversion.")
        width, height = int(image["width"]), int(image["height"])
        polygons = []
        for flat in segmentation:
            points = np.asarray(flat, dtype=np.float64)
            if points.size < 6 or points.size % 2 or not np.isfinite(points).all():
                raise ValueError(f"Invalid polygon in annotation {annotation.get('id')}.")
            xy = points.reshape(-1, 2)
            if (xy[:, 0].min() < 0 or xy[:, 1].min() < 0
                    or xy[:, 0].max() > width or xy[:, 1].max() > height):
                raise ValueError(f"Out-of-frame polygon in annotation {annotation.get('id')}.")
            polygons.append(xy)
        if not polygons:
            raise ValueError(f"Empty polygon mask in annotation {annotation.get('id')}.")
        result[filenames[image["id"]]].append({"polygons": polygons, "width": width, "height": height,
                                               "sku": sku, "attributes": annotation.get("attributes") or {},
                                               "annotation_id": annotation.get("id")})
    if any(not items for items in result.values()):
        raise ValueError("Every benchmark image must have reviewed product annotations.")
    return result


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def match_instances(truth: list[dict], predictions: list[dict], threshold=0.5) -> list[tuple[int, int, float]]:
    overlaps = [[mask_iou(gt["mask"], pred["mask"]) for pred in predictions] for gt in truth]
    adjacency = [sorted((j for j, value in enumerate(row) if value >= threshold),
                        key=lambda j: overlaps[i][j], reverse=True)
                 for i, row in enumerate(overlaps)]
    assigned_gt: dict[int, int] = {}

    def augment(gt_index: int, visited: set[int]) -> bool:
        for pred_index in adjacency[gt_index]:
            if pred_index in visited:
                continue
            visited.add(pred_index)
            if pred_index not in assigned_gt or augment(assigned_gt[pred_index], visited):
                assigned_gt[pred_index] = gt_index
                return True
        return False

    for gt_index in sorted(range(len(truth)), key=lambda i: len(adjacency[i])):
        augment(gt_index, set())
    return [(gt_index, pred_index, overlaps[gt_index][pred_index])
            for pred_index, gt_index in assigned_gt.items()]


def working_photo(photo: Image.Image) -> Image.Image:
    image = photo.convert("RGB").copy()
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def attach_sift_evidence(trace: dict, working: Image.Image) -> None:
    """Join in-process SIFT calls back to the exact proposal crop, read-only."""
    evidence_by_hash = trace.get("sift_by_crop_hash", {})
    for record in trace.get("before_prune", []):
        mask = record.get("mask_array")
        evidence = []
        if mask is not None and np.asarray(mask).any():
            crop = sam_module.masked_crop(working, np.asarray(mask, dtype=bool))
            crop_hash = hashlib.sha256(np.asarray(crop.convert("RGB")).tobytes()).hexdigest()
            evidence = evidence_by_hash.get(crop_hash, [])
        record["sift_verification_attempted"] = bool(evidence)
        record["sift_verification_evidence"] = evidence


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> dict:
    if trials <= 0:
        return {"successes": successes, "trials": trials, "estimate": None,
                "wilson_95": None, "warning": "No observations."}
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return {"successes": successes, "trials": trials, "estimate": p,
            "wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
            "warning": "Small sample; interval is descriptive, not evidence of production performance."
            if trials < 30 else None}


def matcher_audit(prediction: dict, target: dict) -> dict:
    trace = prediction.get("matcher_trace") or prediction
    matches = trace.get("matches") or []
    best = matches[0] if matches else None
    runner_up = matches[1] if len(matches) > 1 else None
    best_score = best.get("similarity") if best else trace.get("similarity")
    runner_score = runner_up.get("similarity") if runner_up else None
    margin = (best_score - runner_score) if best_score is not None and runner_score is not None else trace.get("match_margin")
    mask = target.get("mask")
    if mask is not None and np.asarray(mask).any():
        ys, xs = np.nonzero(mask)
        box_width, box_height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        crop_quality = {"box_xyxy_working_image": [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                                                     int(ys.max()) + 1],
                        "box_width": box_width, "box_height": box_height,
                        "aspect_ratio": round(box_width / box_height, 4),
                        "mask_fill_ratio": round(float(mask.sum()) / (box_width * box_height), 4)}
    else:
        crop_quality = None
    geometry = trace.get("geometric_verification")
    sift_evidence = trace.get("sift_verification_evidence") or []
    return {
        "recognizer_method": trace.get("method"),
        "best_candidate": best,
        "runner_up_candidate": runner_up,
        "best_similarity": best_score,
        "runner_up_similarity": runner_score,
        "similarity_margin": margin,
        "local_feature_verification": {
            "attempted": bool(trace.get("sift_verification_attempted") or geometry or sift_evidence),
            "final_status": (geometry or {}).get("status") if geometry else None,
            "outcome": geometry,
            "per_reference_evidence": sift_evidence,
        },
        "reference_view_review": "Compare the best/runner-up reference images against the crop; viewpoint coverage needs visual review.",
        "crop_quality": crop_quality,
        "ground_truth_occluded_attribute": (target.get("attributes") or {}).get("occluded"),
        "orientation_and_occlusion_visual_review_required": True,
    }


def render_matcher_preview(photo: Image.Image, target: dict, prediction: dict, path: Path) -> None:
    mask = Image.new("L", photo.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    for polygon in target["polygons"]:
        mask_draw.polygon([tuple(point) for point in polygon], fill=255)
    bounds = mask.getbbox()
    if not bounds:
        return
    left, top, right, bottom = bounds
    pad = max(35, int(max(right - left, bottom - top) * 0.12))
    crop_box = (max(0, left - pad), max(0, top - pad),
                min(photo.width, right + pad), min(photo.height, bottom + pad))
    product_crop = photo.crop(crop_box).convert("RGB")
    mask_crop = mask.crop(crop_box)
    pixels = np.asarray(product_crop).copy()
    pixels[np.asarray(mask_crop) == 0] = (124, 124, 124)
    crop_tile = Image.fromarray(pixels)

    audit = matcher_audit(prediction, {**target, "mask": None})
    candidates = (prediction.get("matches") or [])[:2]
    tile_w, tile_h = 480, 560
    header_h = 115
    board = Image.new("RGB", (tile_w * 3, tile_h + header_h), "white")
    draw = ImageDraw.Draw(board)
    title = (f"{target['sku']} | annotation {target.get('annotation_id')} | matcher returned unknown | "
             f"best={audit['best_similarity']} margin={audit['similarity_margin']}")
    draw.text((12, 10), title, fill="#17212b")
    sift = audit["local_feature_verification"]
    draw.text((12, 43), f"SIFT attempted: {sift['attempted']} | final: {sift['final_status'] or 'no verified outcome'}",
              fill="#334155")
    draw.text((12, 74), "Human mask crop, best reference, runner-up reference", fill="#334155")
    tiles = [crop_tile]
    labels = ["Ground-truth product crop"]
    for rank in range(2):
        match = candidates[rank] if rank < len(candidates) else None
        reference_path = Path(match["reference_image"]) if match and match.get("reference_image") else None
        label = "No candidate"
        tile = Image.new("RGB", (tile_w, tile_h), "#e5e7eb")
        if match:
            score = match.get("similarity")
            label = f"{rank + 1}. {match.get('sku')} | score {score} | view {match.get('reference_view')}"
            if reference_path and reference_path.is_file():
                with Image.open(reference_path) as ref_image:
                    tile = ImageOps.contain(ref_image.convert("RGB"), (tile_w, tile_h))
        tiles.append(tile)
        labels.append(label)
    for index, tile in enumerate(tiles):
        contained = ImageOps.contain(tile, (tile_w, tile_h))
        board.paste(contained, (index * tile_w + (tile_w - contained.width) // 2, header_h))
        draw.text((index * tile_w + 8, header_h - 27), labels[index][:70], fill="#17212b")
    path.parent.mkdir(parents=True, exist_ok=True)
    board.save(path, format="PNG", optimize=True)


def resized_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (mask.shape[1], mask.shape[0]) == size:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def rasterize_truth(item: dict, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = np.asarray([width / item["width"], height / item["height"]])
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in item["polygons"]:
        points = np.rint(polygon * scale).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)
    if not mask.any():
        raise ValueError(f"Polygon annotation {item['annotation_id']} rasterizes to an empty mask.")
    return mask.astype(bool)


def read_reference_products(registry_json: Path, business_id: int) -> list[dict]:
    registry = json.loads(registry_json.read_text(encoding="utf-8"))
    if int(registry.get("business_id", -1)) != business_id:
        raise ValueError("Saved reference registry business ID does not match the requested test run.")
    products = registry.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("Saved reference registry must contain a nonempty products list.")
    missing = []
    cache_dir = SAM_FILE.parent / "reference_cache"
    for product in products:
        for reference in product_references(product):
            path = Path(reference["image_path"])
            if not path.is_file():
                missing.append(f"{product['sku']}:{path.name} (reference file missing)")
                continue
            fingerprint = hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii") + path.read_bytes()).hexdigest()
            if not (cache_dir / f"{fingerprint}.npz").is_file():
                missing.append(f"{product['sku']}:{path.name}")
    if missing:
        raise RuntimeError("Read-only benchmark stopped; reference cache missing for: " + ", ".join(missing[:8]))
    return products


def reference_signature(product: dict) -> tuple:
    references = []
    for reference in product_references(product):
        path = Path(reference["image_path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        references.append((reference.get("id"), digest))
    return product["sku"], product["name"], tuple(references)


@contextmanager
def timed_matcher(model: SAMRecognizer, oracle: bool):
    """Instrument this process only; keep production matcher source and thresholds unchanged."""
    times = Counter()
    trace = {"before_prune": [], "after_prune": [], "sift_by_crop_hash": {}}
    encoder_encode = model.encoder.encode
    verifier_compare = model.label_verifier.compare
    original_prune = sam_module.prune_instances
    original_rejection = model.rejection

    def encode(images):
        start = perf_counter()
        try:
            return encoder_encode(images)
        finally:
            times["resnet18_seconds"] += perf_counter() - start

    def compare(query, references):
        start = perf_counter()
        try:
            evidence = verifier_compare(query, references)
            crop_hash = hashlib.sha256(np.asarray(query.convert("RGB")).tobytes()).hexdigest()
            trace["sift_by_crop_hash"].setdefault(crop_hash, []).append(evidence)
            return evidence
        finally:
            times["sift_verification_seconds"] += perf_counter() - start

    def prune(records, masks):
        start = perf_counter()
        try:
            trace["before_prune"] = [
                {**record, "mask_array": np.asarray(mask, dtype=bool).copy()}
                for record, mask in zip(records, masks)
            ]
            if oracle:
                result = [record.copy() for record in records if record["status"] != "rejected"]
            else:
                result = original_prune(records, masks)
            trace["after_prune"] = [
                {**record, "mask_array": np.asarray(mask, dtype=bool).copy()}
                for record, mask in zip(records, masks)
            ]
            return result
        finally:
            times["geometry_prune_seconds"] += perf_counter() - start

    model.encoder.encode = encode
    model.label_verifier.compare = compare
    model.rejection = (lambda mask: None) if oracle else original_rejection
    sam_module.prune_instances = prune
    try:
        yield times, trace
    finally:
        model.encoder.encode = encoder_encode
        model.label_verifier.compare = verifier_compare
        model.rejection = original_rejection
        sam_module.prune_instances = original_prune


def coco_file_for_snapshot(snapshot: dict) -> str:
    return Path(snapshot["original_image"]).name


def run_yolo_masks(model, image: Image.Image, device: str | None,
                   confidence: float, image_size: int,
                   all_classes_as_proposals: bool) -> tuple[list[np.ndarray], float]:
    start = perf_counter()
    results = model.predict(image, conf=confidence, imgsz=image_size, device=device,
                            classes=None if all_classes_as_proposals else [0],
                            verbose=False, retina_masks=True)
    elapsed = perf_counter() - start
    masks = []
    if results and results[0].masks is not None:
        for mask in results[0].masks.data.cpu().numpy():
            masks.append(resized_mask(mask > 0.5, image.size))
    return masks, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True,
                        help="Human-reviewed COCO 1.0 JSON from the separate CVAT task.")
    parser.add_argument("--images", type=Path, required=True,
                        help="Directory containing the original benchmark photos.")
    parser.add_argument("--history", type=Path, default=ROOT / "data/test_runs.jsonl")
    parser.add_argument("--registry-json", type=Path, required=True,
                        help="Read-only product/reference snapshot; SQLite is not read.")
    parser.add_argument("--business-id", type=int, default=1)
    parser.add_argument("--yolo-weights", type=Path,
                        help="Optional generic object-segmentation weights.")
    parser.add_argument("--all-yolo-classes-as-proposals", action="store_true",
                        help="Use every model class as an untyped instance proposal (for COCO weights).")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017/evaluation")
    args = parser.parse_args()
    snapshots = load_history(args.history)
    truth_by_file = load_coco(args.coco, args.images)
    expected_files = {coco_file_for_snapshot(snapshots[run_id]) for run_id in RUN_IDS}
    if set(truth_by_file) != expected_files:
        raise ValueError(f"Reviewed image mapping mismatch. Expected {sorted(expected_files)}, "
                         f"got {sorted(truth_by_file)}")
    products = read_reference_products(args.registry_json, args.business_id)
    product_skus = {item["sku"] for item in products}
    historical_products = snapshots[RUN_IDS[0]]["products"]
    historical_skus = {item["sku"] for item in historical_products}
    if (product_skus != historical_skus or
            {item["sku"]: reference_signature(item) for item in products} !=
            {item["sku"]: reference_signature(item) for item in historical_products}):
        raise ValueError("Current SKU/reference catalogue differs from the saved test runs; "
                         "refusing to compare against a changed matcher.")
    truth_skus = {item["sku"] for objects in truth_by_file.values() for item in objects}
    unrecognized_labels = truth_skus - product_skus - {"__undefined__"}
    if unrecognized_labels:
        raise ValueError(f"Reviewed labels are neither registered SKUs nor __undefined__: "
                         f"{sorted(unrecognized_labels)}")
    if args.yolo_weights and not args.yolo_weights.is_file():
        raise FileNotFoundError(args.yolo_weights)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark output: {args.output}")
    args.output.mkdir(parents=True)

    config_dir = args.output / "_ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir.resolve())
    model = SAMRecognizer()
    real_segment = model.segment
    for product in products:
        for reference in product_references(product):
            model.reference({**product, "image_path": reference["image_path"]})

    yolo_model = None
    if args.yolo_weights:
        from ultralytics import YOLO
        yolo_model = YOLO(str(args.yolo_weights))
        names = yolo_model.names
        class_names = list(names.values()) if isinstance(names, dict) else list(names)
        if (not args.all_yolo_classes_as_proposals and
                (len(class_names) != 1 or str(class_names[0]).strip().lower() != "product")):
            raise ValueError("YOLO weights must be a single-class segmentation model named 'product'.")

    output_rows = []
    for run_id in RUN_IDS:
        snapshot = snapshots[run_id]
        filename = coco_file_for_snapshot(snapshot)
        with Image.open(args.images / filename) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        working = working_photo(photo)
        expected_size = (working.width, working.height)
        truth = [{**item, "mask": rasterize_truth(item, expected_size)}
                 for item in truth_by_file[filename]]
        minimum = float(snapshot["confidence"])
        margin = float(snapshot["minimum_match_margin"])

        detector_inputs: dict[str, tuple[list[np.ndarray], float]] = {}
        start = perf_counter()
        sam_masks, sam_qualities = real_segment(working)
        detector_inputs["A_mobile_sam"] = (sam_masks, perf_counter() - start)
        if yolo_model is not None:
            detector_inputs["B_yolo"] = run_yolo_masks(yolo_model, working, args.device,
                                                       args.yolo_conf, args.imgsz,
                                                       args.all_yolo_classes_as_proposals)

        arms = ["A_mobile_sam"] + (["B_yolo"] if yolo_model is not None else [])
        for arm in arms:
            masks, detector_seconds = detector_inputs[arm]
            model.segment = lambda image, masks=masks, qualities=(sam_qualities if arm == "A_mobile_sam"
                                                                  else [1.0] * len(masks)): (masks, qualities)
            with timed_matcher(model, oracle=False) as (stage_times, trace):
                start = perf_counter()
                detections, _, _, raw = model.recognize(photo, products, minimum, margin)
                total_recognition = perf_counter() - start
            attach_sift_evidence(trace, working)
            predicted = [{"mask": decode_mask(item["mask"], expected_size), "sku": item.get("sku"),
                          "status": item.get("status"), "reason": item.get("reason")}
                         for item in detections if item.get("mask")]
            output_rows.append(score_arm(run_id, filename, arm, truth, predicted,
                                         detector_seconds, model.last_timings, stage_times,
                                         total_recognition, detection_applicable=True,
                                         raw_predictions=trace["before_prune"],
                                         post_filter_predictions=trace["after_prune"]))

        # Oracle arm routes reviewed polygons through the same ResNet18/SIFT matching code,
        # while bypassing detector-only size rejection and geometric proposal pruning.
        oracle_masks = [item["mask"] for item in truth]
        model.segment = lambda image, masks=oracle_masks: (masks, [1.0] * len(masks))
        with timed_matcher(model, oracle=True) as (stage_times, trace):
            start = perf_counter()
            _, _, _, raw = model.recognize(photo, products, minimum, margin)
            total_recognition = perf_counter() - start
        attach_sift_evidence(trace, working)
        oracle_predictions = [{"mask": truth[index]["mask"],
                               "sku": raw[index].get("sku"),
                               "status": raw[index].get("status"),
                               "reason": raw[index].get("reason"),
                               "matcher_trace": raw[index]}
                              for index in range(min(len(truth), len(raw)))]
        human_row = score_arm(run_id, filename, "C_human_masks", truth,
                              oracle_predictions, 0.0, model.last_timings, stage_times,
                              total_recognition, detection_applicable=False,
                              raw_predictions=trace["before_prune"],
                              post_filter_predictions=trace["after_prune"])
        output_rows.append(human_row)
        for index, target in enumerate(truth):
            if target["sku"] == "__undefined__" or index >= len(raw) or raw[index].get("sku") is not None:
                continue
            preview_path = (args.output / "matcher_review_unknown" / run_id /
                            f"annotation_{target['annotation_id']}.png")
            render_matcher_preview(photo, target, raw[index], preview_path)
            for object_row in human_row["objects"]:
                if object_row.get("annotation_id") == target.get("annotation_id"):
                    object_row.setdefault("matcher_audit", {})["manual_review_preview"] = str(preview_path.resolve())
                    break

    report = summarize(output_rows, yolo_available=yolo_model is not None,
                       settings={"sku_thresholds": "per-run saved values; unchanged",
                                 "detection_iou_threshold": 0.5,
                                 "human_mask_arm": ("Ground-truth polygons passed to the unchanged SKU matcher; "
                                                    "detector-only mask rejection and cross-proposal pruning bypassed "
                                                    "to isolate SKU assignment."),
                                 "yolo_device": args.device or "ultralytics default",
                                 "yolo_confidence": args.yolo_conf if yolo_model else None,
                                 "yolo_imgsz": args.imgsz if yolo_model else None,
                                 "reviewed_coco": str(args.coco.resolve()),
                                 "reference_registry_json": str(args.registry_json.resolve()),
                                 "all_yolo_classes_are_generic_proposals": args.all_yolo_classes_as_proposals,
                                 "ground_truth_annotations": sum(len(items) for items in truth_by_file.values()),
                                 "ground_truth_undefined_annotations": sum(
                                     item["sku"] == "__undefined__"
                                     for objects in truth_by_file.values() for item in objects),
                                 "data_quality_note": (
                                     "All 79 reviewed COCO annotations are scored provisionally, including 7 explicitly "
                                     "undefined objects. Earlier review flagged some small/boundary ambiguities; no labels "
                                     "were changed or excluded automatically."
                                 ),
                                 "mobile_sam_device": "cpu (existing pipeline behavior)",
                                 "resnet18_device": "cpu (existing matcher behavior)",
                                 "sift_device": "cpu",
                                 "photos_are_development_diagnostics": True})
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Evaluation report: {args.output / 'report.json'}")


def score_arm(run_id: str, filename: str, arm: str, truth: list[dict],
              predictions: list[dict], detector_seconds: float, timings: dict,
              stage_times: Counter, total_recognition: float,
              detection_applicable: bool, raw_predictions: list[dict] | None = None,
              post_filter_predictions: list[dict] | None = None) -> dict:
    matches = match_instances(truth, predictions) if detection_applicable else [
        (index, index, 1.0) for index in range(min(len(truth), len(predictions)))
    ]
    matched_truth = {item[0] for item in matches}
    matched_predictions = {item[1] for item in matches}
    raw_predictions = raw_predictions or []
    post_filter_predictions = post_filter_predictions or []
    raw_masks = [item.get("mask_array") for item in raw_predictions]
    raw_overlaps = [[mask_iou(target["mask"], mask) if mask is not None else 0.0
                     for mask in raw_masks] for target in truth]
    raw_matches = match_instances(
        truth, [{"mask": mask} for mask in raw_masks if mask is not None]
    ) if raw_masks else []
    raw_gt_to_prediction = {gt: pred for gt, pred, _ in raw_matches}
    raw_prediction_to_gt = {pred: gt for gt, pred, _ in raw_matches}
    post_by_raw = {index: record for index, record in enumerate(post_filter_predictions)}
    per_sku = defaultdict(lambda: Counter())
    object_rows = []
    outcome_counts = Counter()
    for index, target in enumerate(truth):
        row = per_sku[target["sku"]]
        row["ground_truth"] += 1
        if target["sku"] == "__undefined__":
            row["ground_truth_unregistered"] += 1
        if index not in matched_truth:
            row["missed"] += 1
            best_raw = int(np.argmax(raw_overlaps[index])) if raw_overlaps[index] else None
            best_iou = raw_overlaps[index][best_raw] if best_raw is not None else 0.0
            raw_record = raw_predictions[best_raw] if best_raw is not None else None
            post_record = post_by_raw.get(best_raw) if best_raw is not None else None
            if best_iou == 0:
                best_raw = None
                raw_record = None
                post_record = None
            shared_match = (best_raw in raw_prediction_to_gt and
                            raw_prediction_to_gt[best_raw] != index) if best_raw is not None else False
            if best_iou >= 0.5 and raw_record and raw_record.get("status") == "rejected":
                failure = "proposal_rejected_before_matching"
            elif best_iou >= 0.5 and post_record and post_record.get("status") == "rejected":
                failure = ("recognized_then_removed_by_filter" if raw_record and
                           raw_record.get("status") == "recognized" else "removed_by_filter_after_matching")
            elif best_iou >= 0.5 and shared_match:
                failure = "overlapping_or_merged_proposal_competed_for_another_instance"
            elif best_iou > 0:
                failure = "partial_or_incorrect_segmentation_possible"
            else:
                failure = "not_detected_by_proposal_masks"
            object_rows.append({
                "annotation_id": target.get("annotation_id"), "truth_sku": target["sku"],
                "outcome": failure, "best_raw_mask_iou": round(best_iou, 5),
                "raw_proposal_index": best_raw,
                "raw_status": raw_record.get("status") if raw_record else None,
                "raw_reason": raw_record.get("reason") if raw_record else None,
                "status_after_filter": post_record.get("status") if post_record else None,
                "reason_after_filter": post_record.get("reason") if post_record else None,
                "top_sku_candidates": raw_record.get("matches", [])[:3] if raw_record else [],
                "matcher_audit": matcher_audit(raw_record, target) if raw_record else None,
                "diagnostic_note": ("IoU overlap identifies a candidate region, not proof that the mask is a fragment or merge."
                                    if failure == "partial_or_incorrect_segmentation_possible" else None),
            })
            outcome_counts[failure] += 1
    for truth_index, prediction_index, _ in matches:
        target, prediction = truth[truth_index], predictions[prediction_index]
        actual, expected = prediction.get("sku"), target["sku"]
        if expected == "__undefined__":
            outcome = "correct_unknown_abstention" if actual is None else "false_sku_assignment"
        else:
            outcome = "correct" if actual == expected else "wrong" if actual else "unknown"
        per_sku[expected][outcome] += 1
        outcome_counts[outcome] += 1
        raw_record = prediction.get("matcher_trace")
        if raw_record is None and raw_masks and prediction.get("mask") is not None:
            overlaps = [mask_iou(prediction["mask"], mask) if mask is not None else 0.0
                        for mask in raw_masks]
            raw_index = int(np.argmax(overlaps)) if overlaps else None
            if raw_index is not None and overlaps[raw_index] >= 0.99:
                raw_record = raw_predictions[raw_index]
        object_rows.append({
            "annotation_id": target.get("annotation_id"), "truth_sku": expected,
            "predicted_sku": actual, "recognition_status": prediction.get("status"),
            "recognition_reason": prediction.get("reason"), "outcome": outcome,
            "mask_iou": round(next(item[2] for item in matches
                                    if item[0] == truth_index and item[1] == prediction_index), 5),
            "matcher_audit": matcher_audit(raw_record or prediction, target),
        })
    known_sku_correct = sum(row["correct"] for sku, row in per_sku.items() if sku != "__undefined__")
    known_sku_wrong = sum(row["wrong"] for sku, row in per_sku.items() if sku != "__undefined__")
    unknown_on_registered = sum(row["unknown"] for sku, row in per_sku.items() if sku != "__undefined__")
    unregistered_counts = per_sku.get("__undefined__", Counter())
    unregistered_gt = unregistered_counts["ground_truth_unregistered"]
    correct_unknown_abstentions = unregistered_counts["correct_unknown_abstention"]
    false_sku_assignments = unregistered_counts["false_sku_assignment"]
    matched_registered = sum(1 for truth_index, _, _ in matches
                             if truth[truth_index]["sku"] != "__undefined__")
    matched_unregistered = sum(1 for truth_index, _, _ in matches
                               if truth[truth_index]["sku"] == "__undefined__")
    true_positive = len(matches)
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    mask_iou_mean = statistics.mean(item[2] for item in matches) if matches else None
    return {
        "run_id": run_id, "image": filename, "arm": arm,
        "truth_instances": len(truth), "predicted_instances": len(predictions),
        "detection_metrics": {"applicable": detection_applicable,
                              "precision_iou50": precision if detection_applicable else None,
                              "recall_iou50": recall if detection_applicable else None,
                              "precision_wilson95": wilson_interval(true_positive, len(predictions)) if detection_applicable else None,
                              "recall_wilson95": wilson_interval(true_positive, len(truth)) if detection_applicable else None,
                              "absolute_count_error": abs(len(predictions) - len(truth)) if detection_applicable else None,
                              "mean_mask_iou_matched": mask_iou_mean if detection_applicable else None},
        "sku_metrics": {"matched_objects": len(matches), "correct": known_sku_correct,
                        "wrong": known_sku_wrong + false_sku_assignments,
                        "unknown": unknown_on_registered,
                        "unregistered_ground_truth": unregistered_gt,
                        "correct_unknown_abstentions": correct_unknown_abstentions,
                        "false_sku_assignments_on_unregistered": false_sku_assignments,
                        "unknown_predictions_total": sum(prediction.get("sku") is None
                                                          for prediction in predictions),
                        "registered_sku_accuracy_on_matched": (known_sku_correct / matched_registered
                                                                if matched_registered else None),
                        "registered_sku_accuracy_wilson95": wilson_interval(known_sku_correct, matched_registered),
                        "registered_sku_correct_per_ground_truth": wilson_interval(
                            known_sku_correct,
                            sum(row["ground_truth"] for sku, row in per_sku.items() if sku != "__undefined__")),
                        "correct_abstention_rate_on_unregistered": (
                            correct_unknown_abstentions / matched_unregistered if matched_unregistered else None),
                        "unknown_rejection_wilson95": wilson_interval(
                            correct_unknown_abstentions,
                            correct_unknown_abstentions + false_sku_assignments),
                        "identity_outcome_accuracy_on_matched": (
                            (known_sku_correct + correct_unknown_abstentions) / len(matches) if matches else None),
                        "by_ground_truth_sku": {sku: dict(counts) for sku, counts in sorted(per_sku.items())}},
        "runtime_seconds": {"detector": round(detector_seconds, 4),
                            "resnet18_encoder": round(stage_times["resnet18_seconds"], 4),
                            "sift_verification": round(stage_times["sift_verification_seconds"], 4),
                            "geometry_pruning": round(stage_times["geometry_prune_seconds"], 4),
                            "matcher_filter_prune_total": round(timings.get("matching_seconds", 0), 4),
                            "recognizer_total_with_render": round(total_recognition, 4),
                            "end_to_end_detector_plus_recognizer": round(detector_seconds + total_recognition, 4)},
        "objects": sorted(object_rows, key=lambda item: item["annotation_id"] or -1),
        "outcome_counts": dict(outcome_counts),
        "unmatched_predictions": [
            {"prediction_index": index, "predicted_sku": prediction.get("sku"),
             "status": prediction.get("status"), "reason": prediction.get("reason"),
             "best_truth_iou": round(max((mask_iou(target["mask"], prediction["mask"])
                                            for target in truth), default=0.0), 5)}
            for index, prediction in enumerate(predictions) if index not in matched_predictions
        ] if detection_applicable else [],
    }


def summarize(rows: list[dict], yolo_available: bool, settings: dict) -> dict:
    by_arm = defaultdict(list)
    by_sku = defaultdict(lambda: Counter())
    for row in rows:
        by_arm[row["arm"]].append(row)
        for sku, counts in row["sku_metrics"]["by_ground_truth_sku"].items():
            by_sku[(row["arm"], sku)].update(counts)
    arms = {}
    for arm, group in by_arm.items():
        timings = [row["runtime_seconds"] for row in group]
        relevant = [row for row in group if row["detection_metrics"]["applicable"]]
        matched = sum(row["sku_metrics"]["matched_objects"] for row in relevant)
        predicted = sum(row["predicted_instances"] for row in relevant)
        truth_count = sum(row["truth_instances"] for row in relevant)
        sku_counts = {sku: dict(by_sku[(arm, sku)])
                      for selected_arm, sku in sorted(by_sku) if selected_arm == arm}
        registered_gt = sum(counts.get("ground_truth", 0) for sku, counts in sku_counts.items()
                            if sku != "__undefined__")
        registered_correct = sum(counts.get("correct", 0) for sku, counts in sku_counts.items()
                                 if sku != "__undefined__")
        registered_matched = sum(counts.get("correct", 0) + counts.get("wrong", 0) + counts.get("unknown", 0)
                                 for sku, counts in sku_counts.items() if sku != "__undefined__")
        undefined = sku_counts.get("__undefined__", {})
        unknown_detected = undefined.get("correct_unknown_abstention", 0) + undefined.get("false_sku_assignment", 0)
        sku_intervals = {}
        for sku, counts in sku_counts.items():
            gt = counts.get("ground_truth", 0)
            missed = counts.get("missed", 0)
            success = counts.get("correct", 0) if sku != "__undefined__" else counts.get("correct_unknown_abstention", 0)
            sku_intervals[sku] = {
                "correct_identity_per_ground_truth": wilson_interval(success, gt),
                "correct_identity_when_detected": wilson_interval(success, max(0, gt - missed)),
            }
        arms[arm] = {
            "images": len(group),
            "ground_truth_instances": sum(row["truth_instances"] for row in group),
            "final_predictions": sum(row["predicted_instances"] for row in group),
            "matched_instances_iou50": sum(row["sku_metrics"]["matched_objects"] for row in group),
            "micro_detection_precision_iou50": (
                sum(row["sku_metrics"]["matched_objects"] for row in relevant)
                / sum(row["predicted_instances"] for row in relevant)
                if relevant and sum(row["predicted_instances"] for row in relevant)
                else (0.0 if relevant else None)),
            "micro_detection_recall_iou50": (
                sum(row["sku_metrics"]["matched_objects"] for row in relevant)
                / sum(row["truth_instances"] for row in relevant)
                if relevant and sum(row["truth_instances"] for row in relevant)
                else (0.0 if relevant else None)),
            "mean_detection_precision_iou50": mean_key(relevant, "precision_iou50"),
            "mean_detection_recall_iou50": mean_key(relevant, "recall_iou50"),
            "mean_absolute_count_error": mean_key(relevant, "absolute_count_error"),
            "mean_mask_iou_matched": mean_key(relevant, "mean_mask_iou_matched"),
            "sku_correct": sum(row["sku_metrics"]["correct"] for row in group),
            "sku_wrong": sum(row["sku_metrics"]["wrong"] for row in group),
            "sku_unknown": sum(row["sku_metrics"]["unknown"] for row in group),
            "unregistered_ground_truth": sum(row["sku_metrics"]["unregistered_ground_truth"]
                                              for row in group),
            "correct_unknown_abstentions": sum(row["sku_metrics"]["correct_unknown_abstentions"]
                                                for row in group),
            "false_sku_assignments_on_unregistered": sum(
                row["sku_metrics"]["false_sku_assignments_on_unregistered"] for row in group),
            "sku_unknown_predictions_total": sum(row["sku_metrics"]["unknown_predictions_total"]
                                                 for row in group),
            "runtime_mean_seconds": {key: round(statistics.mean(item[key] for item in timings), 4)
                                      for key in timings[0]},
            "sku_by_ground_truth": {sku: dict(by_sku[(arm, sku)])
                                     for selected_arm, sku in sorted(by_sku) if selected_arm == arm},
            "failure_outcomes": dict(sum((Counter(row["outcome_counts"]) for row in group), Counter())),
            "confidence_intervals_95": {
                "micro_detection_precision_iou50": wilson_interval(matched, predicted) if relevant else None,
                "micro_detection_recall_iou50": wilson_interval(matched, truth_count) if relevant else None,
                "registered_sku_accuracy_when_detected": wilson_interval(registered_correct, registered_matched),
                "registered_sku_correct_per_ground_truth": wilson_interval(registered_correct, registered_gt),
                "unknown_rejection_on_detected_unregistered": wilson_interval(
                    undefined.get("correct_unknown_abstention", 0), unknown_detected),
                "note": "Object-level Wilson 95% intervals; objects are clustered within only five scenes, so these are descriptive and likely optimistic about scene-to-scene uncertainty. Treat every metric as a small development sample."
            },
            "sku_confidence_intervals_95": sku_intervals,
        }
    return {"purpose": "Development diagnostic only; these photos are not an untouched test set.",
            "status": "YOLO arm omitted: supply --yolo-weights after a future approved training run."
                      if not yolo_available else "All three arms evaluated.",
            "settings": settings, "arms": arms, "per_image": rows}


def render_markdown(report: dict) -> str:
    lines = [
        "# October 17 Recognition Benchmark",
        "",
        "Development diagnostics only; these five photographs are not an untouched test set.",
        "Historical test logs, saved SKU/reference snapshots and inventory were read-only. No training was performed.",
        "MobileSAM, ResNet18 and SIFT follow the existing CPU execution path; the YOLO proposal model uses the configured device. Stage timings are not directly comparable across different hardware.",
        "",
        "## Overall",
        "",
        "| Arm | GT objects | Outputs/masks | Detection P (micro) | Detection R (micro) | SKU correct | SKU wrong | Registered unknown | Unknown-product abstentions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in report["arms"].items():
        is_detector_arm = any(row["arm"] == arm and row["detection_metrics"]["applicable"]
                              for row in report["per_image"])
        precision = f"{values['micro_detection_precision_iou50']:.3f}" if is_detector_arm else "n/a"
        recall = f"{values['micro_detection_recall_iou50']:.3f}" if is_detector_arm else "n/a"
        lines.append(
            f"| `{arm}` | {values['ground_truth_instances']} | {values['final_predictions']} | "
            f"{precision} | {recall} | {values['sku_correct']} | "
            f"{values['sku_wrong']} | {values['sku_unknown']} | "
            f"{values['correct_unknown_abstentions']} / {values['unregistered_ground_truth']} "
            f"GT; matched {values['correct_unknown_abstentions']} / "
            f"{values['unregistered_ground_truth'] - values['sku_by_ground_truth'].get('__undefined__', {}).get('missed', 0)} |"
        )
    lines += [
        "",
        "Detection precision/recall count one-to-one mask matches at IoU >= 0.50. The human-mask arm has no detector, so its detection metrics are not applicable; its rows report only SKU matching on reviewed masks. It uses saved per-run SKU thresholds but bypasses detector-only mask rejection and cross-proposal pruning to isolate the matcher.",
        "",
        "## Confidence Intervals",
        "",
        "Wilson 95% intervals are descriptive object-level intervals. Objects are clustered within just five scenes, so the intervals understate scene-to-scene uncertainty; every estimate is a small development sample, not a production-performance claim.",
        "",
        "| Arm | Metric | Successes / trials | Estimate | Wilson 95% |",
        "|---|---|---:|---:|---:|",
    ]
    for arm, values in report["arms"].items():
        for metric, interval in values["confidence_intervals_95"].items():
            if metric == "note" or interval is None:
                continue
            ci = interval["wilson_95"]
            label = metric.replace("_", " ")
            if ci is None:
                ci_text, estimate = "n/a", "n/a"
            else:
                ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                estimate = f"{interval['estimate']:.3f}"
            lines.append(f"| `{arm}` | {label} | {interval['successes']} / {interval['trials']} | {estimate} | {ci_text} |")
    lines += [
        "",
        "## Runtime Means",
        "",
        "| Arm | Detector | ResNet18 | SIFT | Geometry prune | Matcher/filter | Recognizer incl. render | Detector + recognizer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    runtime_keys = ["detector", "resnet18_encoder", "sift_verification", "geometry_pruning",
                    "matcher_filter_prune_total", "recognizer_total_with_render",
                    "end_to_end_detector_plus_recognizer"]
    for arm, values in report["arms"].items():
        runtime = values["runtime_mean_seconds"]
        lines.append("| `" + arm + "` | " + " | ".join(f"{runtime[key]:.3f}s" for key in runtime_keys) + " |")
    lines += ["", "## By Scene", "", "| Image | Arm | GT | Output | Det P/R | Correct / wrong / unknown | Count error | Runtime (detector + recognizer) |",
              "|---|---|---:|---:|---|---|---:|---:|"]
    for row in report["per_image"]:
        detection = row["detection_metrics"]
        pr = (f"{detection['precision_iou50']:.3f} / {detection['recall_iou50']:.3f}"
              if detection["applicable"] else "n/a (human masks)")
        metrics = row["sku_metrics"]
        runtime = row["runtime_seconds"]
        error = detection["absolute_count_error"] if detection["applicable"] else "n/a"
        lines.append(
            f"| `{row['image']}` | `{row['arm']}` | {row['truth_instances']} | "
            f"{row['predicted_instances']} | {pr} | {metrics['correct']} / {metrics['wrong']} / "
            f"{metrics['unknown']} | {error} | {runtime['detector']:.3f}s + "
            f"{runtime['recognizer_total_with_render']:.3f}s |"
        )
    lines += ["", "## By SKU", "", "Counts are ground-truth object outcomes. `missed` is the number with no final IoU-0.50 detection match. The confidence interval column is correct identity per ground-truth instance (or correct abstention for `__undefined__`).",
              "", "| Arm | Ground-truth SKU | Instances | Correct | Wrong | Unknown | Missed | Correct unknown abstention | False SKU assignment on undefined | Correct/GT (95% Wilson) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, values in report["arms"].items():
        for sku, counts in values["sku_by_ground_truth"].items():
            interval = values["sku_confidence_intervals_95"][sku]["correct_identity_per_ground_truth"]
            ci = interval["wilson_95"]
            ci_text = (f"{interval['successes']}/{interval['trials']} [{ci[0]:.2f}, {ci[1]:.2f}]"
                       if ci is not None else "n/a")
            lines.append(
                f"| `{arm}` | `{sku}` | {counts.get('ground_truth', 0)} | {counts.get('correct', 0)} | "
                f"{counts.get('wrong', 0)} | {counts.get('unknown', 0)} | {counts.get('missed', 0)} | "
                f"{counts.get('correct_unknown_abstention', 0)} | {counts.get('false_sku_assignment', 0)} | {ci_text} |"
            )
    lines += ["", "## Unknown-Match Investigation", "",
              "For each registered ground-truth object returned unknown on the human-mask arm, the JSON stores the best and runner-up SKU/reference images, cosine scores and margin, SIFT attempt/evidence, crop geometry, and a side-by-side crop/reference preview. Occlusion and viewpoint suitability still require human visual judgment; the COCO occlusion attribute is shown but is not inferred from the pixels.", ""]
    for row in report["per_image"]:
        if row["arm"] != "C_human_masks":
            continue
        unknowns = [obj for obj in row["objects"]
                    if obj["outcome"] == "unknown" and obj.get("truth_sku") != "__undefined__"]
        if not unknowns:
            continue
        lines += [f"### {row['image']}", "", "| Annotation | True SKU | Best candidate (score, reference) | Runner-up (score) | Margin | SIFT | Preview |",
                  "|---:|---|---|---|---:|---|---|"]
        for obj in unknowns:
            audit = obj.get("matcher_audit") or {}
            best, runner = audit.get("best_candidate") or {}, audit.get("runner_up_candidate") or {}
            sift = audit.get("local_feature_verification") or {}
            reference = best.get("reference_image", "")
            preview = audit.get("manual_review_preview", "")
            lines.append(f"| {obj.get('annotation_id')} | `{obj.get('truth_sku')}` | `{best.get('sku', 'n/a')}` ({best.get('similarity', 'n/a')}, `{reference}`) | `{runner.get('sku', 'n/a')}` ({runner.get('similarity', 'n/a')}) | {audit.get('similarity_margin', 'n/a')} | {sift.get('attempted', False)} / `{sift.get('final_status') or 'no verified outcome'}` | `{preview}` |")
        lines.append("")
    lines += ["", "## Per-Object Outcomes", "", "The JSON report contains every CVAT annotation ID, predicted SKU/status, mask IoU, and failure stage. For MobileSAM misses it also records the best raw-proposal IoU/status/reason and top SKU candidates. Partial overlap is labeled as possible segmentation error, not asserted as a proven fragment or merge.", ""]
    for row in report["per_image"]:
        failures = [obj for obj in row["objects"] if obj["outcome"] not in (
            "correct", "correct_unknown_abstention")]
        if failures:
            lines.append(f"### {row['image']} / `{row['arm']}`")
            lines.append("")
            lines.append("| Annotation | True SKU | Prediction | Outcome | IoU / best raw IoU | Reason |")
            lines.append("|---:|---|---|---|---:|---|")
            for obj in failures:
                prediction = obj.get("predicted_sku") or "unknown/not output"
                iou = obj.get("mask_iou", obj.get("best_raw_mask_iou"))
                reason = obj.get("recognition_reason") or obj.get("raw_reason") or obj.get("reason_after_filter") or ""
                lines.append(f"| {obj.get('annotation_id')} | `{obj['truth_sku']}` | `{prediction}` | `{obj['outcome']}` | {iou if iou is not None else 'n/a'} | `{reason}` |")
            lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- MobileSAM omissions, incomplete masks, and masks rejected before matching are detector/segmentation-side opportunities for a future detector, but each proposed YOLO mask still needs an identical matching evaluation.",
        "- Wrong SKU assignments or unknown outcomes on a matched human mask are matcher/reference problems, not detector problems.",
        "- An unknown/unregistered product correctly abstained only when its human mask reached matching and the matcher returned no registered SKU.",
        "- `recognized_then_removed_by_filter` identifies an accepted pre-filter SKU subsequently suppressed by geometry pruning; this is distinct from a missed proposal.",
        "- All 79 current CVAT polygons are scored provisionally (72 registered SKU labels and 7 `__undefined__`). Earlier review flagged a few small or boundary-ambiguous regions; this evaluator does not silently alter or omit them.",
        "- No thresholds, production code, catalogue, embeddings, history, or training data were changed by running this report.",
        "",
        "## Settings",
        "",
        f"- CVAT export: `{report['settings']['reviewed_coco']}`",
        f"- Mask match threshold: IoU >= {report['settings']['detection_iou_threshold']}",
        "- Per-run similarity and margin thresholds: saved historical values, unchanged.",
        f"- Human-mask matcher arm: {report['settings']['human_mask_arm']}",
        f"- MobileSAM device: `{report['settings']['mobile_sam_device']}`",
        f"- ResNet18/SIFT devices: `{report['settings']['resnet18_device']}` / `{report['settings']['sift_device']}`",
        f"- YOLO device: `{report['settings']['yolo_device']}`; generic all-class proposal mode: `{report['settings']['all_yolo_classes_are_generic_proposals']}`",
        f"- Ground-truth note: {report['settings']['data_quality_note']}",
        "",
    ]
    return "\n".join(lines)


def mean_key(rows: list[dict], key: str):
    values = [row["detection_metrics"][key] for row in rows
              if row["detection_metrics"][key] is not None]
    return round(statistics.mean(values), 5) if values else None


if __name__ == "__main__":
    main()
