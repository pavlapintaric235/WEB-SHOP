"""Compare few-shot image encoders on the fixed, ambiguity-filtered Inventory6 crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import run_benchmark as fixed
from .hf_encoders import DinoV2SmallEncoder, SigLIP2BaseEncoder
from .hybrid_core import (MobileNetV2Encoder, abstaining_decision,
                          build_reference_index, product_reference_paths,
                          scores_for_features, scores_for_reference_views)

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".venv" / "huggingface_cache"))
INVENTORY6 = Path(r"C:\Users\U938\Downloads\inventory6.zip")
HISTORICAL_BENCHMARK = EXP / "runs" / "benchmark_20261017_165512" / "report.json"
GRID_MINIMUM = np.arange(0.25, 0.901, 0.025)
GRID_MARGIN = np.arange(0.0, 0.301, 0.025)


def pad_neutral(image: Image.Image, fraction: float = 0.10) -> Image.Image:
    image = image.convert("RGB")
    pad = max(1, round(max(image.size) * fraction))
    canvas = Image.new("RGB", (image.width + 2 * pad, image.height + 2 * pad), (124, 124, 124))
    canvas.paste(image, (pad, pad))
    return canvas


def working_photo(photo: Image.Image) -> Image.Image:
    image = photo.convert("RGB").copy()
    image.thumbnail((1280, 1280))
    return image


def calibrate(rows: list[dict]) -> dict:
    best = None
    for minimum in GRID_MINIMUM:
        for margin in GRID_MARGIN:
            utility = 0
            for row in rows:
                predicted = abstaining_decision(row["scores"], minimum, margin)["sku"]
                actual = row["sku"]
                utility += ((1 if predicted is None else -3) if actual == "__undefined__"
                            else 2 if predicted == actual else -3 if predicted else 0)
            candidate = (utility, minimum + margin, minimum, margin)
            if best is None or candidate > best:
                best = candidate
    return {"minimum_similarity": float(best[2]), "minimum_margin": float(best[3]),
            "objects": len(rows), "utility": int(best[0]),
            "selection": "calibration split only; correct +2, incorrect -3, undefined rejection +1, false assignment -3; ties prefer stricter cutoffs"}


def make_records(history: dict, coco: dict, images_dir: Path) -> list[dict]:
    records = []
    for run_id in fixed.RUN_IDS:
        filename = Path(history[run_id]["original_image"]).name
        photo_path = Path(history[run_id]["original_image"])
        if not photo_path.is_file():
            photo_path = images_dir / filename
        with Image.open(photo_path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        working = working_photo(photo)
        for item in coco[filename]:
            if fixed.is_ambiguous(run_id, item):
                continue
            original_mask = fixed.truth_mask(item)
            mask = cv2.resize(original_mask.astype(np.uint8), working.size,
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            crop = fixed.masked_crop(working, mask)
            original_crop = fixed.masked_crop(photo, original_mask)
            records.append({"run_id": run_id, "filename": filename,
                            "split": "calibration" if run_id in fixed.CALIBRATION else "held_out",
                            "annotation_id": item["annotation_id"], "sku": item["sku"],
                            "mask": mask, "crop_tight": crop, "crop_padded": pad_neutral(crop),
                            "crop_original_resolution": original_crop,
                            "crop_quality": {"width": crop.width, "height": crop.height,
                                "aspect_ratio": round(crop.width / max(1, crop.height), 4),
                                "orientation_hint": "landscape" if crop.width > crop.height else "portrait_or_square",
                                "mask_fill_ratio": round(float(mask.sum()) / max(1, crop.width * crop.height), 4)},
                            "photo": photo, "photo_working": working,
                            "confidence": float(history[run_id]["confidence"]),
                            "saved_margin": float(history[run_id]["minimum_match_margin"])})
    return records


def ensure_reference_cache(products: list[dict]) -> None:
    from model_assets import SAM_FILE
    from sam_vision import REFERENCE_CACHE_VERSION
    cache = SAM_FILE.parent / "reference_cache"
    missing = []
    for product in products:
        for path in product_reference_paths(product):
            digest = hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii") + path.read_bytes()).hexdigest()
            if not (cache / f"{digest}.npz").is_file():
                missing.append(path.name)
    if missing:
        raise RuntimeError("Read-only reference crop cache incomplete; refusing to create or update it: "
                           + ", ".join(missing[:8]))


def load_production_gallery(products: list[dict]):
    from local_features import LabelVerifier
    from model_assets import SAM_FILE
    from sam_vision import REFERENCE_CACHE_VERSION, ReferenceEncoder, SAMRecognizer
    from threading import RLock

    ensure_reference_cache(products)
    recognizer = object.__new__(SAMRecognizer)
    recognizer.encoder = ReferenceEncoder()
    recognizer.label_verifier = LabelVerifier()
    recognizer.references = {}
    recognizer.reference_dir = SAM_FILE.parent / "reference_cache"
    recognizer.lock = RLock()
    views, product_indices, raw_crops = [], [], []
    for index, product in enumerate(products):
        for view_no, reference in enumerate(product_reference_paths(product), 1):
            entry = {**product, "image_path": str(reference), "view": view_no,
                     "reference_id": None}
            feature, crop = recognizer.reference(entry)
            views.append(entry)
            raw_crops.append(crop)
            product_indices.append(index)
    return recognizer, views, product_indices, raw_crops


def build_model_index(encoder, products, reference_crops):
    vectors = encoder.encode(reference_crops)
    index = {product["sku"]: {encoder.name: []} for product in products}
    position = 0
    for product in products:
        for path, crop in zip(product_reference_paths(product), reference_crops[position:]):
            # Gallery order is grouped by SKU, as returned by load_production_gallery.
            index[product["sku"]][encoder.name].append({
                "feature": vectors[position], "image": str(path),
                "sha256": hashlib.sha256(str(path).encode()).hexdigest()})
            position += 1
    return index


def make_gallery_index(encoder, products, views, reference_crops, padded=False):
    source = [pad_neutral(image) for image in reference_crops] if padded else reference_crops
    started = perf_counter()
    vectors = encoder.encode(source)
    elapsed = perf_counter() - started
    index = {product["sku"]: {encoder.name: []} for product in products}
    for view, vector in zip(views, vectors):
        index[view["sku"]][encoder.name].append({
            "feature": vector, "image": view["image_path"],
            "sha256": hashlib.sha256(view["image_path"].encode()).hexdigest()})
    return index, elapsed


def embed_model(encoder, records, products, views, reference_crops, crop_key):
    padded = crop_key == "padded_10pct"
    index, reference_seconds = make_gallery_index(encoder, products, views, reference_crops, padded)
    crops = [row["crop_padded"] if padded else row["crop_tight"] for row in records]
    started = perf_counter()
    features = encoder.encode(crops)
    elapsed = perf_counter() - started
    output = []
    for row, feature in zip(records, features):
        scores, branches, refs = scores_for_reference_views({encoder.name: feature}, index)
        ranked = [{"sku": sku, "similarity": value,
                   "reference": refs[encoder.name].get(sku, {}).get("path")}
                  for sku, value in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
        output.append({**row, "scores": scores, "top": ranked,
                       "best_reference": ranked[0]["reference"] if ranked else None,
                       "best_reference_by_sku": refs[encoder.name],
                       "branch_scores": branches})
    return output, elapsed, reference_seconds, index


def metric_summary(rows, minimum=None, margin=None):
    registered = [row for row in rows if row["sku"] != "__undefined__"]
    undefined = [row for row in rows if row["sku"] == "__undefined__"]
    correct_top1 = sum(bool(row["top"] and row["top"][0]["sku"] == row["sku"]) for row in registered)
    top5 = sum(any(candidate["sku"] == row["sku"] for candidate in row["top"][:5]) for row in registered)
    accepted = {id(row): abstaining_decision(row["scores"], minimum, margin) for row in rows}
    right = sum(accepted[id(row)]["sku"] == row["sku"] for row in registered)
    wrong = sum(accepted[id(row)]["sku"] is not None and accepted[id(row)]["sku"] != row["sku"]
                for row in registered)
    unknown = len(registered) - right - wrong
    accepted_rows = [row for row in registered if accepted[id(row)]["sku"] is not None]
    accepted_top5 = sum(any(candidate["sku"] == row["sku"] for candidate in row["top"][:5])
                        for row in accepted_rows)
    undef_decisions = [accepted[id(row)] for row in undefined]
    rejected = sum(decision["sku"] is None for decision in undef_decisions)
    return {"registered": len(registered), "top1_correct_before_abstention": correct_top1,
            "top1_accuracy_before_abstention": correct_top1 / max(1, len(registered)),
            "top5_contains_correct_before_abstention": top5,
            "top5_recall_before_abstention": top5 / max(1, len(registered)),
            "accepted_registered": len(accepted_rows),
            "top5_contains_correct_after_abstention": accepted_top5,
            "top5_recall_after_abstention_all_registered": accepted_top5 / max(1, len(registered)),
            "top5_recall_among_accepted": accepted_top5 / max(1, len(accepted_rows)),
            "accepted_correct": right, "accepted_incorrect": wrong, "abstained_registered": unknown,
            "coverage_registered": (right + wrong) / max(1, len(registered)),
            "accuracy_among_accepted": right / max(1, right + wrong),
            "undefined": len(undefined), "undefined_rejected": rejected,
            "undefined_false_assignments": len(undefined) - rejected}


def candidate_rows(rows, settings):
    output = []
    for row in rows:
        decision = abstaining_decision(row["scores"], settings["minimum_similarity"],
                                       settings["minimum_margin"])
        top = row["top"]
        output.append({"run_id": row["run_id"], "filename": row["filename"],
                       "split": row["split"], "annotation_id": row["annotation_id"],
                       "actual_sku": row["sku"], "top1_sku": top[0]["sku"] if top else None,
                       "top1_similarity": top[0]["similarity"] if top else None,
                       "top2_sku": top[1]["sku"] if len(top) > 1 else None,
                       "top2_similarity": top[1]["similarity"] if len(top) > 1 else None,
                       "top5": json.dumps(top[:5]), "abstained_sku": decision["sku"],
                       "decision": decision["reason"],
                       "best_reference": row.get("best_reference"),
                       "best_reference_name": Path(row["best_reference"]).name if row.get("best_reference") else None,
                       "crop_quality": json.dumps(row.get("crop_quality", {}))})
    return output


def score_encoder(rows, encoder, products, views, reference_crops, crop_key):
    print(f"Embedding {encoder.name} {crop_key}: {len(reference_crops)} references and {len(rows)} queries on CPU.",
          flush=True)
    encoded, inference_seconds, reference_seconds, index = embed_model(
        encoder, rows, products, views, reference_crops, crop_key)
    calibration_rows = [row for row in encoded if row["split"] == "calibration"]
    settings = calibrate(calibration_rows)
    summaries = {}
    for split in ("calibration", "held_out", "all"):
        subset = encoded if split == "all" else [row for row in encoded if row["split"] == split]
        summaries[split] = metric_summary(subset, settings["minimum_similarity"], settings["minimum_margin"])
    return {"encoder": encoder.name, "model_id": getattr(encoder, "model_id", "TorchVision MobileNetV2 IMAGENET1K_V2"),
            "model_revision": getattr(encoder, "revision", "local TorchVision checkpoint hash-verified"),
            "crop_mode": crop_key, "thresholds": settings,
            "metrics": summaries, "query_feature_seconds": inference_seconds,
            "reference_feature_seconds": reference_seconds,
            "per_object": candidate_rows(encoded, settings), "encoded": encoded, "index": index}


def score_mobilenet_previous_setup(rows, encoder, products, previous_settings):
    """Reproduce the earlier whole-reference, centroid matcher on the same query crops."""
    index = build_reference_index(products, {encoder.name: encoder})
    crops = [row["crop_original_resolution"] for row in rows]
    started = perf_counter()
    features = encoder.encode(crops)
    query_seconds = perf_counter() - started
    encoded = []
    for row, feature in zip(rows, features):
        scores, _, refs = scores_for_features({encoder.name: feature}, index)
        top = [{"sku": sku, "similarity": value, "reference": refs[encoder.name][sku]["path"]}
               for sku, value in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
        encoded.append({**row, "scores": scores, "top": top,
                        "best_reference": top[0]["reference"] if top else None})
    settings = {"minimum_similarity": previous_settings["minimum_similarity"],
                "minimum_margin": previous_settings["minimum_margin"],
                "source": "unchanged cutoff from the previous MobileNetV2 benchmark"}
    metrics = {split: metric_summary(
        encoded if split == "all" else [row for row in encoded if row["split"] == split],
        settings["minimum_similarity"], settings["minimum_margin"])
               for split in ("calibration", "held_out", "all")}
    output = candidate_rows(encoded, settings)
    return {"encoder": encoder.name, "gallery": "raw registration photographs + per-SKU centroid (previous setup)",
            "crop_mode": "historical full-resolution query crops; included only to reproduce the prior 12/44 report, not the controlled comparison",
            "thresholds": settings,
            "metrics": metrics, "query_feature_seconds": query_seconds,
            "per_object": output, "per_sku": by_sku(output)}


def run_resnet_sift(records, history, products, recognizer, views, product_indices):
    from sam_vision import aggregate_reference_scores
    import sam_vision as sam_module

    reference_vectors = np.stack([recognizer.reference(view)[0] for view in views])
    sku_order = [product["sku"] for product in products]
    view_skus = [view["sku"] for view in views]
    rows_out, features_by_run = [], {}
    groups = defaultdict(list)
    for row in records:
        groups[row["run_id"]].append(row)
    for run_id in fixed.RUN_IDS:
        group = groups[run_id]
        if not group:
            continue
        photo = group[0]["photo"]
        masks = [row["mask"] for row in group]
        recognizer.segment = lambda image, masks=masks: (masks, [1.0] * len(masks))
        recognizer.rejection = lambda mask: None
        query_vectors = []
        crops = [row["crop_tight"] for row in group]
        crop_hashes = {hashlib.sha256(np.asarray(crop.convert("RGB")).tobytes()).hexdigest(): index
                       for index, crop in enumerate(crops)}
        sift_by_crop = defaultdict(list)
        original_encode = recognizer.encoder.encode
        original_compare = recognizer.label_verifier.compare

        def tracked(images):
            result = original_encode(images)
            if len(images) == len(group):
                query_vectors.extend(result)
            return result

        recognizer.encoder.encode = tracked
        def tracked_compare(query, references):
            evidence = original_compare(query, references)
            digest = hashlib.sha256(np.asarray(query.convert("RGB")).tobytes()).hexdigest()
            sift_by_crop[digest].append(evidence)
            return evidence
        recognizer.label_verifier.compare = tracked_compare
        original_prune = sam_module.prune_instances
        sam_module.prune_instances = lambda result, masks: [item for item in result if item["status"] != "rejected"]
        try:
            thresholds = float(history[run_id]["confidence"]), float(history[run_id]["minimum_match_margin"])
            started = perf_counter()
            _, _, _, raw = recognizer.recognize(photo, products, *thresholds)
            total_seconds = perf_counter() - started
        finally:
            recognizer.encoder.encode = original_encode
            recognizer.label_verifier.compare = original_compare
            sam_module.prune_instances = original_prune
        if len(query_vectors) != len(group) or len(raw) != len(group):
            raise RuntimeError(f"Production matcher output did not align with eligible crops for {run_id}.")
        scores, winners = aggregate_reference_scores(np.asarray(query_vectors) @ reference_vectors.T,
                                                      product_indices, len(products))
        for index, row in enumerate(group):
            ordered = sorted(((sku, float(scores[index, sku_order.index(sku)])) for sku in sku_order),
                             key=lambda pair: (-pair[1], pair[0]))
            raw_match = raw[index]
            top = [{"sku": sku, "similarity": score,
                    "reference": views[int(winners[index, sku_order.index(sku)])]["image_path"]}
                   for sku, score in ordered[:5]]
            rows_out.append({"run_id": run_id, "filename": row["filename"], "split": row["split"],
                             "annotation_id": row["annotation_id"], "actual_sku": row["sku"],
                             "top1_sku": top[0]["sku"], "top1_similarity": top[0]["similarity"],
                             "top2_sku": top[1]["sku"] if len(top) > 1 else None,
                             "top2_similarity": top[1]["similarity"] if len(top) > 1 else None,
                             "top5": json.dumps(top), "abstained_sku": raw_match.get("sku"),
                             "decision": raw_match.get("reason"), "matcher_method": raw_match.get("method"),
                             "sift_attempted": bool(sift_by_crop.get(hashlib.sha256(
                                 np.asarray(crops[index].convert("RGB")).tobytes()).hexdigest())),
                             "sift_evidence": json.dumps(sift_by_crop.get(hashlib.sha256(
                                 np.asarray(crops[index].convert("RGB")).tobytes()).hexdigest(), [])),
                             "saved_similarity_threshold": thresholds[0],
                             "saved_margin_threshold": thresholds[1],
                             "timing_seconds": round(total_seconds, 6),
                             "best_reference": top[0]["reference"],
                             "best_reference_name": Path(top[0]["reference"]).name,
                             "crop_quality": json.dumps(row["crop_quality"])})
    summary = {}
    for split in ("calibration", "held_out", "all"):
        subset = [row for row in rows_out if split == "all" or row["split"] == split]
        registered = [row for row in subset if row["actual_sku"] != "__undefined__"]
        undefined = [row for row in subset if row["actual_sku"] == "__undefined__"]
        summary[split] = {
            "registered": len(registered),
            "top1_correct_before_abstention": sum(row["top1_sku"] == row["actual_sku"] for row in registered),
            "top5_contains_correct_before_abstention": sum(any(item["sku"] == row["actual_sku"] for item in json.loads(row["top5"])) for row in registered),
            "accepted_registered": sum(row["abstained_sku"] is not None for row in registered),
            "top5_contains_correct_after_abstention": sum(
                row["abstained_sku"] is not None
                and any(item["sku"] == row["actual_sku"] for item in json.loads(row["top5"]))
                for row in registered),
            "accepted_correct": sum(row["abstained_sku"] == row["actual_sku"] for row in registered),
            "accepted_incorrect": sum(row["abstained_sku"] is not None and row["abstained_sku"] != row["actual_sku"] for row in registered),
            "abstained_registered": sum(row["abstained_sku"] is None for row in registered),
            "undefined": len(undefined),
            "undefined_rejected": sum(row["abstained_sku"] is None for row in undefined),
            "undefined_false_assignments": sum(row["abstained_sku"] is not None for row in undefined),
            "matcher_seconds_sum": round(sum({row["run_id"]: row["timing_seconds"] for row in subset}.values()), 4)}
        summary[split]["top5_recall_after_abstention_all_registered"] = (
            summary[split]["top5_contains_correct_after_abstention"] / max(1, len(registered)))
        summary[split]["top5_recall_among_accepted"] = (
            summary[split]["top5_contains_correct_after_abstention"]
            / max(1, summary[split]["accepted_registered"]))
    return {"encoder": "resnet18_sift_production", "crop_mode": "same human polygon crops",
            "thresholds": "per-run saved original confidence and margin; unchanged",
            "metrics": summary, "per_object": rows_out}


def reference_quality(products, views, crops):
    rows = []
    for product in products:
        for view, crop in zip([entry for entry in views if entry["sku"] == product["sku"]],
                              [crop for entry, crop in zip(views, crops) if entry["sku"] == product["sku"]]):
            with Image.open(view["image_path"]) as raw:
                raw_width, raw_height = raw.size
            pixels = np.asarray(crop.convert("RGB"))
            gray_background = np.all(pixels == (124, 124, 124), axis=2)
            gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
            rows.append({"sku": product["sku"], "path": view["image_path"],
                         "filename": Path(view["image_path"]).name, "width": crop.width,
                         "height": crop.height, "aspect": round(crop.width / max(1, crop.height), 3),
                         "raw_width": raw_width, "raw_height": raw_height,
                         "foreground_fill_ratio": round(1.0 - float(gray_background.mean()), 4),
                         "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
                         "masked_crop_pixels": crop.width * crop.height})
    return rows


def by_sku(per_object):
    groups = defaultdict(list)
    for row in per_object:
        groups[row["actual_sku"]].append(row)
    result = []
    for sku, rows in sorted(groups.items()):
        result.append({"sku": sku, "instances": len(rows),
                       "top1_correct": sum(row.get("top1_sku") == sku for row in rows),
                       "top5_contains": sum(any(item["sku"] == sku for item in json.loads(row["top5"])) for row in rows),
                       "accepted_correct": sum(row.get("abstained_sku") == sku for row in rows),
                       "incorrect_assignment": sum(row.get("abstained_sku") not in (None, sku) for row in rows),
                       "unknown": sum(row.get("abstained_sku") is None for row in rows)})
    return result


def by_scene(per_object):
    groups = defaultdict(list)
    for row in per_object:
        groups[row["filename"]].append(row)
    results = []
    for filename, rows in sorted(groups.items()):
        registered = [row for row in rows if row["actual_sku"] != "__undefined__"]
        undefined = [row for row in rows if row["actual_sku"] == "__undefined__"]
        results.append({"filename": filename, "split": rows[0]["split"],
                        "registered": len(registered),
                        "top1_correct": sum(row["top1_sku"] == row["actual_sku"] for row in registered),
                        "top5_contains": sum(any(item["sku"] == row["actual_sku"]
                                                  for item in json.loads(row["top5"])) for row in registered),
                        "accepted_correct": sum(row.get("abstained_sku") == row["actual_sku"] for row in registered),
                        "incorrect": sum(row.get("abstained_sku") not in (None, row["actual_sku"])
                                          for row in registered),
                        "unknown": sum(row.get("abstained_sku") is None for row in registered),
                        "undefined": len(undefined),
                        "undefined_rejected": sum(row.get("abstained_sku") is None for row in undefined)})
    return results


def top1_correct_rejections(per_object):
    rejected = [row for row in per_object if row["actual_sku"] != "__undefined__"
                and row.get("top1_sku") == row["actual_sku"]
                and row.get("abstained_sku") is None]
    return {"count": len(rejected), "by_reason": dict(Counter(row.get("decision") for row in rejected)),
            "annotation_ids": [row["annotation_id"] for row in rejected]}


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=INVENTORY6)
    parser.add_argument("--images", type=Path, default=ROOT / "data/test_runs")
    parser.add_argument("--models", nargs="+", choices=("dinov2", "mobilenetv2", "siglip2", "dinov2+mobilenetv2"),
                        default=("dinov2", "mobilenetv2", "siglip2", "dinov2+mobilenetv2"))
    parser.add_argument("--allow-download", action="store_true",
                        help="Permit Hugging Face downloads after preflight confirms sufficient free space.")
    parser.add_argument("--output", type=Path, help="Optional new output directory; never overwritten.")
    parser.add_argument("--no-padding-ablation", action="store_true")
    args = parser.parse_args()

    history = fixed.load_history()
    coco = fixed.load_coco_zip(args.coco)
    products = fixed.load_catalog()
    eligible = make_records(history, coco, args.images)
    ensure_reference_cache(products)
    recognizer, views, product_indices, reference_crops = load_production_gallery(products)
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        transformers_version = None
    result = {"purpose": "Experimental, fixed Inventory6 human-mask recognition comparison; no classifier training.",
              "runtime": {"device": "CPU", "torch": __import__("torch").__version__,
                          "transformers": transformers_version},
              "eligible_objects": len(eligible),
              "registered": sum(row["sku"] != "__undefined__" for row in eligible),
              "undefined": sum(row["sku"] == "__undefined__" for row in eligible),
              "split": {"calibration_run_ids": sorted(fixed.CALIBRATION),
                        "held_out_run_ids": [run for run in fixed.RUN_IDS if run not in fixed.CALIBRATION],
                        "caveat": "Held-out 01:42 and 01:44 are two views of one arrangement; not independent final evaluation."},
              "threshold_policy": "Original ResNet18/SIFT thresholds remain per-run historical values. Each pretrained embedding model uses only calibration objects to select abstention thresholds; held-out images are never used for threshold selection.",
              "models": {}, "reference_gallery_count": len(views),
              "reference_quality": reference_quality(products, views, reference_crops)}

    production = run_resnet_sift(eligible, history, products, recognizer, views, product_indices)
    production["per_sku"] = by_sku(production["per_object"])
    production["per_scene"] = by_scene(production["per_object"])
    production["correct_top1_rejected"] = top1_correct_rejections(production["per_object"])
    result["models"]["resnet18_sift"] = production
    encoder_instances = {}
    score_cache = {}
    if "mobilenetv2" in args.models:
        encoder_instances["mobilenetv2"] = MobileNetV2Encoder()
        mobile_previous = score_mobilenet_previous_setup(
            eligible, encoder_instances["mobilenetv2"], products,
            json.loads(HISTORICAL_BENCHMARK.read_text(encoding="utf-8"))["calibration"])
        mobile_previous["per_scene"] = by_scene(mobile_previous["per_object"])
        mobile_previous["correct_top1_rejected"] = top1_correct_rejections(mobile_previous["per_object"])
        result["models"]["mobilenetv2_previous_gallery"] = mobile_previous
    for model_name in args.models:
        needed = {"mobilenetv2": 0.0, "dinov2": 1.0, "siglip2": 2.5,
                  "dinov2+mobilenetv2": 1.0}[model_name]
        free_gb = shutil.disk_usage(EXP).free / (1024 ** 3)
        if needed and (free_gb < needed) and args.allow_download:
            result["models"][model_name] = {"status": "not_loaded",
                "reason": f"Download skipped: only {free_gb:.3f} GiB free; preflight requires at least {needed:.1f} GiB."}
            continue
        try:
            local_only = not args.allow_download
            branches = (model_name.split("+") if "+" in model_name else [model_name])
            encoders = []
            for branch in branches:
                if branch not in encoder_instances:
                    if branch == "dinov2":
                        encoder_instances[branch] = DinoV2SmallEncoder(local_files_only=local_only)
                    elif branch == "mobilenetv2":
                        encoder_instances[branch] = MobileNetV2Encoder()
                    elif branch == "siglip2":
                        encoder_instances[branch] = SigLIP2BaseEncoder(local_files_only=local_only)
                encoders.append(encoder_instances[branch])

            for crop_mode in (("tight", "padded_10pct") if not args.no_padding_ablation else ("tight",)):
                branch_data = []
                for encoder in encoders:
                    cache_key = (encoder.name, crop_mode)
                    if cache_key not in score_cache:
                        score_cache[cache_key] = score_encoder(eligible, encoder, products, views,
                                                               reference_crops, crop_mode)
                    branch_data.append(score_cache[cache_key])
                if len(encoders) == 1:
                    scored = branch_data[0]
                    key = model_name if crop_mode == "tight" else f"{model_name}_{crop_mode}"
                    result["models"][key] = {k: v for k, v in scored.items() if k not in {"encoded", "index"}}
                    result["models"][key]["per_sku"] = by_sku(scored["per_object"])
                    result["models"][key]["per_scene"] = by_scene(scored["per_object"])
                    result["models"][key]["correct_top1_rejected"] = top1_correct_rejections(scored["per_object"])
                    result["models"][key]["reference_feature_seconds"] = scored["reference_feature_seconds"]
                else:
                    # Equal weight per encoder; each branch contributes its best-view cosine.
                    fused = []
                    for row_index, source_row in enumerate(eligible):
                        scores = {sku: sum(branch["encoded"][row_index]["scores"][sku]
                                           for branch in branch_data) / len(branch_data)
                                  for sku in products_by_sku(products)}
                        top_decision = abstaining_decision(scores, 0.0, 0.0)
                        fused.append({**source_row, "scores": scores,
                                      "top": top_decision["ranked_scores"],
                                      "best_reference": branch_data[0]["encoded"][row_index]["best_reference"]})
                    calibration_rows = [row for row in fused if row["split"] == "calibration"]
                    settings = calibrate(calibration_rows)
                    metrics = {split: metric_summary(
                        fused if split == "all" else [row for row in fused if row["split"] == split],
                        settings["minimum_similarity"], settings["minimum_margin"])
                               for split in ("calibration", "held_out", "all")}
                    output = candidate_rows(fused, settings)
                    key = model_name if crop_mode == "tight" else f"{model_name}_{crop_mode}"
                    result["models"][key] = {"encoder": model_name, "crop_mode": crop_mode,
                        "branches": [{"model_id": getattr(encoder, "model_id", encoder.name),
                                      "revision": getattr(encoder, "revision", "local TorchVision weights")}
                                     for encoder in encoders],
                        "thresholds": settings, "metrics": metrics, "per_object": output,
                        "per_sku": by_sku(output),
                        "per_scene": by_scene(output),
                        "correct_top1_rejected": top1_correct_rejections(output),
                        "query_feature_seconds": sum(branch["query_feature_seconds"] for branch in branch_data),
                        "reference_feature_seconds": sum(branch["reference_feature_seconds"] for branch in branch_data)}
        except Exception as error:
            result["models"][model_name] = {"status": "unavailable", "reason": str(error)}

    mobile_json = json.loads(HISTORICAL_BENCHMARK.read_text(encoding="utf-8"))
    result["detector_context"] = {
        "source": str(HISTORICAL_BENCHMARK),
        "note": "Detector metrics are from the saved fixed-split benchmark, not a new detector run. Recognition models here use the exact eligible human masks.",
        "YOLO_Sku110k": mobile_json["arms"].get("C_yolo11_sku110k_mobilenetv2", {}).get("summary"),
        "MobileSAM": mobile_json["arms"].get("A_mobile_sam_filtered_box_iou50", {}).get("all_five")}

    if args.output:
        output_dir = args.output
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir}")
        output_dir.mkdir(parents=True)
        (output_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        for name, model in result["models"].items():
            if model.get("per_object"):
                write_rows(output_dir / f"{name}_per_object.csv", model["per_object"])
            if model.get("per_sku"):
                write_rows(output_dir / f"{name}_per_sku.csv", model["per_sku"])
        write_rows(output_dir / "reference_quality.csv", result["reference_quality"])
        print(output_dir.resolve())
    print(json.dumps({"eligible_objects": result["eligible_objects"],
                      "registered": result["registered"], "undefined": result["undefined"],
                      "models": {name: {"status": value.get("status", "run"),
                          "metrics": value.get("metrics"), "thresholds": value.get("thresholds"),
                          "per_sku": value.get("per_sku"), "per_scene": value.get("per_scene"),
                          "correct_top1_rejected": value.get("correct_top1_rejected"),
                          "reason": value.get("reason")}
                                 for name, value in result["models"].items()},
                      "detector_context": result["detector_context"]}, indent=2))


def products_by_sku(products):
    return [product["sku"] for product in products]


if __name__ == "__main__":
    main()
