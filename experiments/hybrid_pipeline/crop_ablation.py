"""Compare crop representations for few-shot SKU matching on the fixed Inventory6 split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageOps

from . import run_benchmark as fixed
from .fewshot_compare import calibrate, make_records, load_production_gallery
from .hf_encoders import SigLIP2BaseEncoder
from .hybrid_core import (MobileNetV2Encoder, abstaining_decision, fuse_sku_scores,
                          product_reference_paths, scores_for_reference_views)
from .mask_runtime import masked_context_crop

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
INVENTORY6 = Path(r"C:\Users\U938\Downloads\inventory6.zip")


def heuristic_foreground(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]], axis=0)
    background = np.median(border, axis=0)
    return np.linalg.norm(pixels - background, axis=2) > 35


def tight_box_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty foreground mask")
    return image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def crop_for_query(row: dict, mode: str) -> Image.Image:
    if mode == "polygon_masked":
        return row["crop_tight"]
    if mode == "tight_bbox":
        return tight_box_crop(row["photo_working"], row["mask"])
    if mode == "polygon_masked_context":
        return masked_context_crop(row["photo_working"], row["mask"], fraction=0.10)
    raise ValueError(mode)


def crop_for_reference(path: Path, cached_masked_crop: Image.Image, mode: str) -> Image.Image:
    if mode == "polygon_masked":
        return cached_masked_crop
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    mask = heuristic_foreground(image)
    if mode == "tight_bbox":
        return tight_box_crop(image, mask)
    if mode == "polygon_masked_context":
        return masked_context_crop(image, mask, fraction=0.10)
    raise ValueError(mode)


def make_index(encoder, products, views, reference_crops):
    started = perf_counter()
    vectors = encoder.encode(reference_crops)
    elapsed = perf_counter() - started
    index = {product["sku"]: {encoder.name: []} for product in products}
    for view, vector in zip(views, vectors):
        index[view["sku"]][encoder.name].append({
            "feature": vector, "image": view["image_path"],
            "sha256": hashlib.sha256(view["image_path"].encode("utf-8")).hexdigest(),
        })
    return index, elapsed


def score_branch(rows, encoder, index, mode):
    crops = [crop_for_query(row, mode) for row in rows]
    started = perf_counter()
    vectors = encoder.encode(crops)
    elapsed = perf_counter() - started
    output = []
    for row, crop, vector in zip(rows, crops, vectors):
        scores, _, references = scores_for_reference_views({encoder.name: vector}, index)
        ranked = [{"sku": sku, "similarity": score,
                   "reference": references[encoder.name][sku]["path"]}
                  for sku, score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
        output.append({**row, "crop_mode": mode, "scores": scores, "top": ranked,
                       "crop_size": list(crop.size)})
    return output, elapsed


def metric_summary(rows: list[dict], minimum: float, margin: float) -> dict:
    registered = [row for row in rows if row["sku"] != "__undefined__"]
    undefined = [row for row in rows if row["sku"] == "__undefined__"]
    decisions = [abstaining_decision(row["scores"], minimum, margin) for row in rows]
    top1 = sum(bool(row["top"] and row["top"][0]["sku"] == row["sku"]) for row in registered)
    top5 = sum(any(item["sku"] == row["sku"] for item in row["top"][:5]) for row in registered)
    correct = sum(decision["sku"] == row["sku"] for row, decision in zip(rows, decisions)
                  if row["sku"] != "__undefined__")
    wrong = sum(decision["sku"] not in (None, row["sku"]) for row, decision in zip(rows, decisions)
                if row["sku"] != "__undefined__")
    accepted_unknown = sum(decision["sku"] is not None for row, decision in zip(rows, decisions)
                           if row["sku"] == "__undefined__")
    return {"objects": len(rows), "registered": len(registered), "undefined": len(undefined),
            "raw_top1_correct": top1, "raw_top1_accuracy": top1 / max(1, len(registered)),
            "raw_top5_contains_correct": top5, "raw_top5_recall": top5 / max(1, len(registered)),
            "accepted_correct": correct, "accepted_wrong": wrong,
            "registered_unknown": len(registered) - correct - wrong,
            "undefined_correctly_rejected": len(undefined) - accepted_unknown,
            "undefined_false_assignments": accepted_unknown,
            "accepted_accuracy_registered": correct / max(1, correct + wrong),
            "thresholds": {"minimum_similarity": minimum, "minimum_margin": margin}}


def serialize_rows(rows, minimum, margin, model_name, mode):
    output = []
    for row in rows:
        decision = abstaining_decision(row["scores"], minimum, margin)
        output.append({"run_id": row["run_id"], "image": row["filename"], "split": row["split"],
            "annotation_id": row["annotation_id"], "actual_sku": row["sku"], "model": model_name,
            "crop_mode": mode, "predicted_sku": decision["sku"] or "__unknown__",
            "decision_reason": decision["reason"], "top1_sku": row["top"][0]["sku"],
            "top1_score": row["top"][0]["similarity"],
            "top5": json.dumps(row["top"], ensure_ascii=False),
            "crop_size": json.dumps(row["crop_size"]),
            "minimum_similarity": minimum, "minimum_margin": margin})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=INVENTORY6)
    parser.add_argument("--images", type=Path, default=ROOT / "data/test_runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--models", nargs="+", choices=("mobilenetv2", "siglip2", "siglip2+mobilenetv2"),
                        default=("mobilenetv2", "siglip2", "siglip2+mobilenetv2"))
    args = parser.parse_args()
    output = args.output or EXP / "runs" / f"crop_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    os.environ.setdefault("HF_HOME", str(ROOT / ".venv" / "huggingface_cache"))

    history = fixed.load_history()
    coco = fixed.load_coco_zip(args.coco)
    products = fixed.load_catalog()
    rows = make_records(history, coco, args.images)
    _, views, _, cached_reference_crops = load_production_gallery(products)
    modes = ("tight_bbox", "polygon_masked", "polygon_masked_context")
    encoders = {}
    if "mobilenetv2" in args.models or "siglip2+mobilenetv2" in args.models:
        encoders["mobilenetv2"] = MobileNetV2Encoder(device="cpu")
    if "siglip2" in args.models or "siglip2+mobilenetv2" in args.models:
        encoders["siglip2"] = SigLIP2BaseEncoder(device="cpu", local_files_only=True)

    report = {"scope": "fixed five-scene Inventory6 human-mask benchmark; no training",
        "objects": len(rows), "registered": sum(row["sku"] != "__undefined__" for row in rows),
        "undefined": sum(row["sku"] == "__undefined__" for row in rows),
        "split": {"calibration": sorted(fixed.CALIBRATION),
                  "held_out": [run for run in fixed.RUN_IDS if run not in fixed.CALIBRATION],
                  "warning": "Both held-out photographs show the same arrangement; results are not independent-scene generalization."},
        "reference_count": len(views), "models": {}, "crop_definitions": {
            "tight_bbox": "Unmasked tight rectangle around the human instance polygon; reference photos use border-color foreground crop.",
            "polygon_masked": "Background outside human polygon replaced by neutral gray; references use existing cached MobileSAM foreground crops.",
            "polygon_masked_context": "Polygon-masked object rectangle plus a 10% real-image context ring; reference foreground mask is a border-color heuristic.",
        }, "threshold_policy": "Each encoder/crop pair calibrates similarity and margin on calibration scenes only; held-out scenes are not used for selection."}

    branch_rows = {}
    timings = {}
    for mode in modes:
        branch_rows[mode] = {}
        for name, encoder in encoders.items():
            if mode == "polygon_masked":
                ref_crops = cached_reference_crops
            else:
                ref_crops = [crop_for_reference(Path(view["image_path"]), crop, mode)
                             for view, crop in zip(views, cached_reference_crops)]
            index, reference_seconds = make_index(encoder, products, views, ref_crops)
            scored, query_seconds = score_branch(rows, encoder, index, mode)
            branch_rows[mode][name] = scored
            calibration = [row for row in scored if row["split"] == "calibration"]
            settings = calibrate(calibration)
            per_split = {split: metric_summary(
                [row for row in scored if split == "all" or row["split"] == split],
                settings["minimum_similarity"], settings["minimum_margin"])
                for split in ("calibration", "held_out", "all")}
            key = f"{name}:{mode}"
            report["models"][key] = {"encoder": name, "crop_mode": mode,
                "thresholds": settings, "metrics": per_split,
                "reference_feature_seconds": reference_seconds,
                "query_feature_seconds": query_seconds,
                "note": "Reference and query images use the same encoder-specific built-in preprocessing."}
            write_csv(output / f"{name}_{mode}_per_object.csv",
                      serialize_rows(scored, settings["minimum_similarity"],
                                     settings["minimum_margin"], name, mode))

        if len(encoders) > 1 and "siglip2+mobilenetv2" in args.models:
            first_name, second_name = tuple(encoders)
            first_rows, second_rows = branch_rows[mode][first_name], branch_rows[mode][second_name]
            fused = []
            for left, right in zip(first_rows, second_rows):
                fused.append({**left, "scores": fuse_sku_scores({
                    first_name: left["scores"], second_name: right["scores"]
                }), "top": []})
                fused[-1]["top"] = [{"sku": sku, "similarity": score}
                    for sku, score in sorted(fused[-1]["scores"].items(),
                                             key=lambda pair: (-pair[1], pair[0]))[:5]]
            settings = calibrate([row for row in fused if row["split"] == "calibration"])
            per_split = {split: metric_summary(
                [row for row in fused if split == "all" or row["split"] == split],
                settings["minimum_similarity"], settings["minimum_margin"])
                for split in ("calibration", "held_out", "all")}
            key = f"siglip2+mobilenetv2:{mode}"
            report["models"][key] = {"encoder": "siglip2+mobilenetv2", "crop_mode": mode,
                "fusion": "equal arithmetic mean of branch-specific best-view cosine similarities",
                "thresholds": settings, "metrics": per_split}
            write_csv(output / f"siglip2_mobilenetv2_{mode}_per_object.csv",
                      serialize_rows(fused, settings["minimum_similarity"],
                                     settings["minimum_margin"], "siglip2+mobilenetv2", mode))

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# Crop Representation Ablation", "", f"Eligible objects: {len(rows)}; registered: {report['registered']}; undefined: {report['undefined']}",
             "", "All model/crop thresholds were calibrated on calibration scenes only. Held-out consists of two views of one arrangement, so it is not independent-scene evidence.", "", "| Model and crop | Split | Top-1 | Top-5 | Accepted correct | Accepted wrong | Registered unknown | Undefined rejected |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, model in report["models"].items():
        for split in ("calibration", "held_out"):
            metric = model["metrics"][split]
            lines.append(f"| {key} | {split} | {metric['raw_top1_correct']}/{metric['registered']} | {metric['raw_top5_contains_correct']}/{metric['registered']} | {metric['accepted_correct']} | {metric['accepted_wrong']} | {metric['registered_unknown']} | {metric['undefined_correctly_rejected']}/{metric['undefined']} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
