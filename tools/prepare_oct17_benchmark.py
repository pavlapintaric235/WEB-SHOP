"""Create a separate CVAT review package for the October 17 diagnostic photos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sam_vision import decode_mask
from tools.prepare_cvat_sku_attribute_zip import transform_coco

HISTORY = ROOT / "data/test_runs.jsonl"
OUTPUT = ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017"
RUN_IDS = [
    "20261017-014015-b2a337ec",
    "20261017-014239-235d7e51",
    "20261017-014423-e7b3fb42",
    "20261017-014554-b9412539",
    "20261017-014722-cc4ab8d3",
]

# Human visual recount, intentionally separate from the entered expectations.
# Units not independently visible are recorded in `unverified_expected_extras`.
VISIBLE_COUNTS = {
    RUN_IDS[0]: {
        "7DAYS-CAKE-30": 2, "CROCO-CRACKERS-100": 2,
        "ENERGIA-POWER-250": 1, "FRESHONA-BEANS-800": 1,
        "MAKS-CRISPY-50": 1, "OREO-ENROBED-41": 2,
        "VITAMINKA-STOBI-30": 1,
    },
    RUN_IDS[1]: {
        "DM-CURRY-325": 2, "DM-EINTOPF-400": 2,
        "DUKAT-MLIJEKO-1000": 3, "ENERGIA-POWER-250": 1,
        "FENTIMANS-GINGER-200": 3, "FRESHONA-BEANS-800": 2,
        "GARDENGOOD-PASSATA-200": 3, "JANA-ANTIOX-500": 1,
    },
    RUN_IDS[2]: {
        "DM-CURRY-325": 2, "DM-EINTOPF-400": 2,
        "DUKAT-MLIJEKO-1000": 3, "ENERGIA-POWER-250": 1,
        "FENTIMANS-GINGER-200": 3, "FRESHONA-BEANS-800": 2,
        "GARDENGOOD-PASSATA-200": 3, "JANA-ANTIOX-500": 1,
    },
    RUN_IDS[3]: {
        "DM-CURRY-325": 1, "DM-EINTOPF-400": 1,
        "ENERGIA-POWER-250": 3, "FENTIMANS-GINGER-200": 3,
        "FRESHONA-BEANS-800": 2, "GARDENGOOD-PASSATA-200": 3,
    },
    RUN_IDS[4]: {
        "DM-EINTOPF-400": 2, "ENERGIA-POWER-250": 4,
        "FENTIMANS-GINGER-200": 3,
    },
}

UNVERIFIED_EXTRAS = {
    RUN_IDS[0]: {},
    RUN_IDS[1]: {"ENERGIA-POWER-250": 2, "JANA-ANTIOX-500": 1},
    RUN_IDS[2]: {"ENERGIA-POWER-250": 2, "JANA-ANTIOX-500": 1},
    RUN_IDS[3]: {
        "DM-EINTOPF-400": 1, "ENERGIA-POWER-250": 1,
        "FRESHONA-BEANS-800": 1,
    },
    RUN_IDS[4]: {},
}

COUNT_NOTES = {
    RUN_IDS[0]: "The ten expected packages appear visible. Some are overlapped; inspect polygons individually.",
    RUN_IDS[1]: "Same cupboard arrangement as 01:44 from a wider view. Only one Energia and one Jana are independently visible; entered extras are not visibly verified.",
    RUN_IDS[2]: "Same cupboard arrangement as 01:42 from a closer view. Only one Energia and one Jana are independently visible; entered extras are not visibly verified.",
    RUN_IDS[3]: "Clearly visible: one Eintopf, three Energia, two Freshona. The extra expected units are not separately visible; possible occlusion is uncertain.",
    RUN_IDS[4]: "The expected two Eintopf, four Energia and three Fentimans are visually plausible, with crowding/partial occlusion in the Energia group.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runs() -> list[dict]:
    by_id = {}
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            run = json.loads(line)
            if run.get("run_id") in RUN_IDS:
                by_id[run["run_id"]] = run
    if set(by_id) != set(RUN_IDS):
        raise SystemExit("The expected five historical run IDs were not all found.")
    return [by_id[run_id] for run_id in RUN_IDS]


def normalized_photo(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def coco_polygons(mask: np.ndarray, image_size: tuple[int, int]) -> list[list[float]]:
    height, width = mask.shape
    image_width, image_height = image_size
    scale_x, scale_y = image_width / width, image_height / height
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(0.5, perimeter * 0.001), True)
        points = approx.reshape(-1, 2)
        if len(points) < 3 or cv2.contourArea(points) < 2:
            continue
        flat = []
        for x, y in points:
            flat.extend([round(float(x) * scale_x, 2), round(float(y) * scale_y, 2)])
        polygons.append(flat)
    return polygons


def make_preview(photo: Image.Image, detections: list[dict], output: Path) -> None:
    preview = photo.copy()
    preview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    pixels = np.asarray(preview).copy()
    sx, sy = preview.width / photo.width, preview.height / photo.height
    labels = []
    for number, detection in enumerate(detections, 1):
        encoded = detection.get("mask")
        if not encoded:
            continue
        mask = decode_mask(encoded)
        resized = cv2.resize(mask.astype(np.uint8), (preview.width, preview.height),
                             interpolation=cv2.INTER_NEAREST)
        color = (0, 180, 125) if detection.get("sku") else (245, 180, 0)
        contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pixels, contours, -1, color, 2)
        left, top = detection["box"][:2]
        x, y = round(left * sx), round(top * sy)
        labels.append((number, x, y, color))

    preview = Image.fromarray(pixels)
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default(size=18)
    for number, x, y, color in labels:
        label = str(number)
        box = draw.textbbox((0, 0), label, font=font)
        radius = max(box[2] - box[0], box[3] - box[1]) // 2 + 5
        center = (min(max(radius, x), preview.width - radius),
                  min(max(radius, y), preview.height - radius))
        draw.ellipse((center[0] - radius, center[1] - radius,
                      center[0] + radius, center[1] + radius), fill=color, outline="black", width=1)
        draw.text((center[0] - (box[2] - box[0]) // 2,
                   center[1] - (box[3] - box[1]) // 2 - 1), label, fill="black", font=font)
    preview.save(output, optimize=True)


def build(run_ids: list[str], output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing benchmark directory: {output}")
    output.mkdir(parents=True)
    image_dir = output / "cvat_review/images"
    preview_dir = output / "cvat_review/proposal_previews"
    image_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True)

    runs = load_runs()
    products_by_sku = {item["sku"]: item for item in runs[0]["products"]}
    skus = sorted(products_by_sku)
    if any({p["sku"] for p in run["products"]} != set(skus) for run in runs):
        raise SystemExit("Run catalogues differ; refusing to create inconsistent CVAT labels.")

    coco = {"info": {"description": "Unverified Web&Shop recognition suggestions for human review"},
            "licenses": [], "images": [], "annotations": [],
            "categories": [{"id": index + 1, "name": sku, "supercategory": "product"}
                           for index, sku in enumerate(skus)]}
    category_id = {sku: index + 1 for index, sku in enumerate(skus)}
    provisional = {
        "status": "provisional_visual_recount_not_human_approved",
        "counts_mean_visible_physical_instances_only": True,
        "fully_occluded_or_not_independently_visible_items_are_not_counted": True,
        "source_history": "data/test_runs.jsonl (read-only)",
        "scene_grouping": {RUN_IDS[1]: "same_cupboard_arrangement_01", RUN_IDS[2]: "same_cupboard_arrangement_01"},
        "scenes": [],
    }
    summary_rows, proposal_rows = [], []
    annotation_id = 1

    for image_id, run in enumerate(runs, 1):
        source = Path(run["original_image"]).resolve(strict=True)
        if ROOT.resolve() not in source.parents:
            raise SystemExit(f"Original image is outside the project: {source}")
        photo = normalized_photo(source)
        filename = source.name
        (image_dir / filename).write_bytes(source.read_bytes())
        width, height = photo.size
        coco["images"].append({"id": image_id, "file_name": filename,
                               "width": width, "height": height})

        predicted_annotations = {}
        final_detections = run.get("detections") or []
        for proposal_number, detection in enumerate(final_detections, 1):
            mask_payload = detection.get("mask")
            predicted_id = ""
            if detection.get("sku") and mask_payload:
                mask = decode_mask(mask_payload)
                polygons = coco_polygons(mask, photo.size)
                if polygons:
                    ys, xs = np.nonzero(mask)
                    scale_x, scale_y = width / mask.shape[1], height / mask.shape[0]
                    bbox = [float(xs.min() * scale_x), float(ys.min() * scale_y),
                            float((xs.max() + 1) * scale_x), float((ys.max() + 1) * scale_y)]
                    predicted_id = annotation_id
                    coco["annotations"].append({
                        "id": annotation_id, "image_id": image_id,
                        "category_id": category_id[detection["sku"]],
                        "segmentation": polygons,
                        "area": round(float(mask.sum() * scale_x * scale_y), 2),
                        "bbox": [round(bbox[0], 2), round(bbox[1], 2),
                                 round(bbox[2] - bbox[0], 2), round(bbox[3] - bbox[1], 2)],
                        "iscrowd": 0,
                    })
                    predicted_annotations[proposal_number] = annotation_id
                    annotation_id += 1
            matches = detection.get("matches") or []
            proposal_rows.append({
                "run_id": run["run_id"], "proposal_id": proposal_number,
                "status": detection.get("status"), "sku_suggestion": detection.get("sku") or "",
                "reason": detection.get("reason") or "",
                "similarity": detection.get("similarity", ""),
                "match_margin": detection.get("match_margin", ""),
                "box_xyxy_original_pixels": json.dumps(detection.get("box", [])),
                "top_candidates": json.dumps(matches[:3]),
                "coco_annotation_id": predicted_id,
                "review_state": "unverified_prediction" if predicted_id else "manual_review_required",
            })

        make_preview(photo, final_detections, preview_dir / f"{run['run_id']}-numbered.png")
        expected = {sku: int(value) for sku, value in (run.get("expected_counts") or {}).items() if value}
        visible = VISIBLE_COUNTS[run["run_id"]]
        extras = UNVERIFIED_EXTRAS[run["run_id"]]
        for sku in sorted(set(expected) | set(visible) | set(extras)):
            expected_count = expected.get(sku, 0)
            visible_count = visible.get(sku, 0)
            extra = extras.get(sku, 0)
            summary_rows.append({
                "run_id": run["run_id"], "created_at": run["created_at"], "sku": sku,
                "recorded_expected": expected_count, "provisional_visible_count": visible_count,
                "expected_not_visually_confirmed": extra,
                "difference_expected_minus_visible": expected_count - visible_count,
                "count_status": "provisional_visual_review_required",
            })
        provisional["scenes"].append({
            "run_id": run["run_id"], "created_at": run["created_at"],
            "image_filename": filename, "source_photo_hash": run["photo_hash"],
            "saved_original_sha256": sha256(source),
            "recorded_expected_counts": expected,
            "provisional_visible_counts": visible,
            "unverified_expected_extras": extras,
            "provisional_visible_total": sum(visible.values()),
            "count_notes": COUNT_NOTES[run["run_id"]],
            "same_arrangement_as": RUN_IDS[1] if run["run_id"] == RUN_IDS[2] else
                                  RUN_IDS[2] if run["run_id"] == RUN_IDS[1] else None,
            "polygons": "Not approved; import suggestions and complete/correct in CVAT.",
        })

    (output / "provisional_ground_truth.json").write_text(
        json.dumps(provisional, indent=2), encoding="utf-8")
    with (output / "provisional_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "cvat_review/proposal_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(proposal_rows[0]))
        writer.writeheader()
        writer.writerows(proposal_rows)

    coco_json = json.dumps(coco, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(output / "cvat_review/cvat_sku_suggestions.zip", "w",
                         compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("annotations/instances_default.json", coco_json)

    attribute_coco = transform_coco(coco)
    attribute_json = json.dumps(attribute_coco, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(output / "cvat_review/cvat_product_sku_attribute_suggestions.zip", "w",
                         compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("annotations/instances_default.json", attribute_json)
    (output / "cvat_review/sku_options.txt").write_text(
        "__undefined__\n" + "\n".join(skus) + "\n", encoding="utf-8")

    if len(coco["images"]) != 5 or len(coco["annotations"]) != 30:
        raise RuntimeError(
            f"Unexpected proposal package size: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations"
        )
    image_ids = {image["id"] for image in coco["images"]}
    category_ids = {category["id"] for category in coco["categories"]}
    if any(annotation["image_id"] not in image_ids or
           annotation["category_id"] not in category_ids
           for annotation in coco["annotations"]):
        raise RuntimeError("COCO image/category mapping validation failed.")
    if len({image["file_name"] for image in coco["images"]}) != 5:
        raise RuntimeError("COCO image filenames are not unique.")

    stage_summary = []
    for run in runs:
        reasons = Counter(item.get("reason") for item in (run.get("raw_detections") or []))
        stage_summary.append({"run_id": run["run_id"], "raw_mask_records": len(run.get("raw_detections") or []),
                              "kept_for_review": len(run.get("detections") or []),
                              "counted_sku_predictions": sum(bool(item.get("sku")) for item in (run.get("detections") or [])),
                              "unassigned_kept": run.get("unassigned_regions", 0),
                              "rejection_reasons": dict(reasons),
                              "timings": run.get("timings")})
    (output / "saved_run_summary.json").write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")

    readme = """# October 17 Recognition Benchmark Review Batch

This is a separate development/annotation package. It is not in the YOLO training dataset and nothing here is approved ground truth.

## Contents

- `images/`: copies of the five original photographs, retaining their run-based filenames.
- `cvat_product_sku_attribute_suggestions.zip`: preferred annotation-only COCO 1.0 import, with one `product` category and per-instance `attributes.sku` on the 30 MobileSAM suggestions. All are **unverified suggestions**.
- `cvat_sku_suggestions.zip`: alternate annotation-only COCO import with SKU codes as categories, retained for compatibility with the original review instructions.
- `sku_options.txt`: copy-ready values for the CVAT `sku` Select attribute, including the blank option.
- `proposal_previews/`: numbered overlays of the 121 retained proposals. Green numbers have a provisional SKU prediction; amber numbers are unassigned. The proposal number maps to `proposal_metadata.csv`.
- `proposal_metadata.csv`: status, reason, top SKU candidates, scores and COCO suggestion mapping for retained proposals.
- `../provisional_ground_truth.json` and `../provisional_counts.csv`: independent provisional visible-count recount, with uncertain expected extras separated.

## CVAT review

1. Create a new, separate CVAT task from all five images in `images/`. Do not add them to the training dataset.
2. Create one task label named `product`. Add a label attribute named `sku`, choose **Select**, and add the values from `sku_options.txt`. Set `__undefined__` as the default blank choice; then every object needs an explicit SKU selection.
3. Upload `cvat_product_sku_attribute_suggestions.zip` as COCO 1.0 annotations. Define the `product` label and its `sku` attribute before importing. Each imported shape carries its provisional SKU in the COCO annotation's `attributes.sku` field.
4. For every visible in-scope product, create or correct exactly one polygon around its visible pixels, choose `product`, and select that physical item's SKU in the `sku` dropdown. Label repeated units separately. Correct the 30 imported suggestions and add the other visible instances; suggestions are incomplete and unverified. Ignore unrelated background items outside this diagnostic inventory. Flag an in-scope item whose SKU cannot be verified instead of guessing.
5. For occlusion, trace only visible pixels. Do not add fully hidden expected units. Treat the 01:42 and 01:44 photos as one scene group, although annotate both views independently.
6. Export the reviewed whole task as COCO 1.0 with images included. Keep the export separate from the training dataset until the object-level and SKU audit is complete.

Only 30 unverified SKU suggestions are preloaded. The provisional recount is 66 visible target instances; add and verify the rest. Amber contours are unassigned proposals, not known objects. Never treat a model suggestion as ground truth.
"""
    (output / "cvat_review/README.md").write_text(readme, encoding="utf-8")

    print(f"Review package: {output}")
    print(f"Images: {len(coco['images'])}; unverified COCO suggestions: {len(coco['annotations'])}")
    print(f"Numbered previews: {len(runs)}; kept proposals indexed: {len(proposal_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed_root = (ROOT / "data/recognition_checks").resolve()
    if allowed_root not in output.parents:
        parser.error(f"Output must be below {allowed_root}")
    build(RUN_IDS, output)


if __name__ == "__main__":
    main()
