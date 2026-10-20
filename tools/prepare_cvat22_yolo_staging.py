"""Convert the reviewed 23-image COCO export to YOLO-seg labels without copying photos."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from segmentation_dataset import parse_yolo_label, polygon_area, polygon_self_intersections, yolo_line

SOURCE_ZIP = Path(r"C:\Users\U938\Downloads\inventoryLens_22_Scenes_Reviewed2.zip")
MAPPING_PATH = ROOT / "data/segmentation_dataset/reports/cvat_export_audit_20261016/mapping_report.json"
REVIEW_REPORT = ROOT / "data/segmentation_dataset/proposals/cvat22_final_review_20261016/validation_report.json"
OUT = ROOT / "data/segmentation_dataset/proposals/cvat22_yolo_staging_20261016"
EXPECTED_ARCHIVE_SHA256 = "7e14c071f1cc166ee8b3f3dc885b4d630cd96fa25d88ee4a10cc54365b56e4a1"
EXPECTED_IMAGES = 23
EXPECTED_ANNOTATIONS = 146
TINY_FRACTION = 0.002
PALETTE = [
    (25, 200, 245), (245, 65, 155), (125, 220, 75), (255, 185, 45),
    (160, 115, 245), (245, 105, 70), (45, 175, 125), (225, 215, 65),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def box_xyxy(polygons):
    points = [point for polygon in polygons for point in polygon]
    return [min(x for x, _ in points), min(y for _, y in points),
            max(x for x, _ in points), max(y for _, y in points)]


def canonical_polygon(points):
    ring = tuple((round(x, 5), round(y, 5)) for x, y in points)
    variants = []
    for candidate in (ring, tuple(reversed(ring))):
        variants.extend(candidate[i:] + candidate[:i] for i in range(len(candidate)))
    return min(variants)


def mask_iou(original, reconstructed):
    points = original + reconstructed
    x0 = max(0, math.floor(min(x for x, _ in points)) - 1)
    y0 = max(0, math.floor(min(y for _, y in points)) - 1)
    x1 = math.ceil(max(x for x, _ in points)) + 2
    y1 = math.ceil(max(y for _, y in points)) + 2
    shape = (max(1, y1 - y0), max(1, x1 - x0))
    masks = []
    for polygon in (original, reconstructed):
        mask = np.zeros(shape, dtype=np.uint8)
        coordinates = np.rint(np.asarray(polygon, dtype=np.float64) - (x0, y0)).astype(np.int32)
        cv2.fillPoly(mask, [coordinates], 1)
        masks.append(mask.astype(bool))
    intersection = int(np.logical_and(masks[0], masks[1]).sum())
    union = int(np.logical_or(masks[0], masks[1]).sum())
    return intersection / union if union else 0.0


def load_inputs():
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError(f"COCO export not found: {SOURCE_ZIP}")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite staging output: {OUT}")
    archive_hash = sha256_file(SOURCE_ZIP)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"Source ZIP hash changed: {archive_hash}")
    mapping_data = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    if mapping_data.get("primary_sha256") != archive_hash:
        raise ValueError("Verified image mapping belongs to a different ZIP export.")
    mapping = {row["file_name"]: row for row in mapping_data["image_records"]}
    review = json.loads(REVIEW_REPORT.read_text(encoding="utf-8"))
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            raise ValueError(f"ZIP CRC failure: {bad_entry}")
        coco = json.loads(archive.read("annotations/instances_default.json"))
    return archive_hash, mapping, review, coco


def convert():
    archive_hash, mapping, prior_review, coco = load_inputs()
    categories = coco.get("categories", [])
    if len(categories) != 1 or categories[0].get("name") != "product":
        raise ValueError(f"Expected exactly one 'product' category, got {categories!r}")
    coco_class_id = categories[0]["id"]
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    if len(images) != EXPECTED_IMAGES or len(annotations) != EXPECTED_ANNOTATIONS:
        raise ValueError(f"Unexpected COCO totals: {len(images)} images, {len(annotations)} annotations")
    if len({image["id"] for image in images}) != len(images):
        raise ValueError("Duplicate COCO image IDs.")
    if len({image["file_name"] for image in images}) != len(images):
        raise ValueError("Duplicate COCO image filenames.")
    if len({annotation["id"] for annotation in annotations}) != len(annotations):
        raise ValueError("Duplicate COCO annotation IDs.")

    image_by_id = {image["id"]: image for image in images}
    annotations_by_id = defaultdict(list)
    for annotation in annotations:
        if annotation.get("image_id") not in image_by_id:
            raise ValueError(f"Annotation {annotation.get('id')} refers to an unknown image.")
        annotations_by_id[annotation["image_id"]].append(annotation)

    image_rows = []
    label_contents = {}
    flagged = []
    geometry_errors = []
    duplicate_masks = []
    all_iou = []
    raster_differences = []
    max_coordinate_error = 0.0
    total_source_bytes = 0
    seen_stems = set()

    for image in images:
        filename = image["file_name"]
        record = mapping.get(filename)
        if not record or not record.get("pixel_identical"):
            raise ValueError(f"No pixel-verified original mapping for {filename}.")
        source = Path(record["matched_original"])
        if not source.is_file():
            raise FileNotFoundError(f"Mapped original is missing: {source}")
        if source.stem in seen_stems:
            raise ValueError(f"Filename stem collision would overwrite a YOLO label: {source.stem}")
        seen_stems.add(source.stem)
        with Image.open(source) as photograph:
            if photograph.size != (image["width"], image["height"]):
                raise ValueError(f"Original dimensions differ from COCO for {filename}: {photograph.size}")
        total_source_bytes += source.stat().st_size

        rows = []
        seen_shapes = set()
        image_annotations = annotations_by_id[image["id"]]
        for local_index, annotation in enumerate(image_annotations, 1):
            ann_id = annotation["id"]
            if annotation.get("category_id") != coco_class_id:
                geometry_errors.append({"image": filename, "annotation_id": ann_id, "error": "non-product class"})
                continue
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or len(segmentation) != 1:
                geometry_errors.append({"image": filename, "annotation_id": ann_id,
                                        "error": "RLE or multipart mask cannot be represented as one YOLO polygon"})
                continue
            flat = segmentation[0]
            if len(flat) < 6 or len(flat) % 2:
                geometry_errors.append({"image": filename, "annotation_id": ann_id, "error": "degenerate COCO polygon"})
                continue
            polygon = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
            width, height = image["width"], image["height"]
            if not all(math.isfinite(x) and math.isfinite(y) for x, y in polygon):
                geometry_errors.append({"image": filename, "annotation_id": ann_id, "error": "non-finite coordinate"})
                continue
            if any(x < -1e-6 or y < -1e-6 or x > width + 1e-6 or y > height + 1e-6 for x, y in polygon):
                geometry_errors.append({"image": filename, "annotation_id": ann_id, "error": "coordinate outside image bounds"})
                continue
            if polygon_self_intersections(polygon):
                geometry_errors.append({"image": filename, "annotation_id": ann_id, "error": "self-intersection"})
                continue
            shape_key = canonical_polygon(polygon)
            if shape_key in seen_shapes:
                duplicate_masks.append({"image": filename, "annotation_id": ann_id, "error": "exact duplicate polygon"})
            seen_shapes.add(shape_key)

            normalized = [(x / width, y / height) for x, y in polygon]
            row = yolo_line(normalized)
            parsed = parse_yolo_label(row, f"{filename} annotation {ann_id}")
            if len(parsed) != 1 or parsed[0][0] != 0:
                raise ValueError(f"YOLO class/row validation failed for annotation {ann_id}")
            reconstructed = [(x * width, y * height) for x, y in parsed[0][1]]
            max_coordinate_error = max(
                max_coordinate_error,
                max(abs(a - b) for point_a, point_b in zip(polygon, reconstructed)
                    for a, b in zip(point_a, point_b)),
            )
            raster_score = mask_iou(polygon, reconstructed)
            all_iou.append(raster_score)
            if raster_score < 1.0:
                raster_differences.append({
                    "image": filename,
                    "local_instance": local_index,
                    "annotation_id": ann_id,
                    "raster_mask_iou": raster_score,
                })
            rows.append(row)

            area = polygon_area(polygon)
            fraction = area / (width * height)
            bbox = box_xyxy([polygon])
            reason = "tiny mask; human confirmation" if fraction < TINY_FRACTION else None
            flagged.append({
                "image": filename,
                "local_instance": local_index,
                "annotation_id": ann_id,
                "category_id": annotation["category_id"],
                "reason": reason,
                "bbox_xyxy_px": [round(value, 3) for value in bbox],
                "bbox_xywh_px": [round(bbox[0], 3), round(bbox[1], 3),
                                 round(bbox[2] - bbox[0], 3), round(bbox[3] - bbox[1], 3)],
                "polygon_xy_px": [[round(x, 4), round(y, 4)] for x, y in polygon],
                "polygon_xy_normalized": [[round(x, 8), round(y, 8)] for x, y in normalized],
                "area_px2": round(area, 3),
                "image_fraction_pct": round(fraction * 100, 6),
            })

        if len(rows) != len(image_annotations):
            continue
        label_contents[f"{source.stem}.txt"] = "\n".join(rows) + "\n"
        image_rows.append({
            "coco_image_id": image["id"],
            "coco_filename": filename,
            "label_filename": f"{source.stem}.txt",
            "width": image["width"],
            "height": image["height"],
            "annotation_count": len(rows),
            "verified_original_path": str(source.resolve()),
            "original_file_bytes": source.stat().st_size,
            "original_sha256": sha256_file(source),
        })

    if geometry_errors or duplicate_masks:
        raise ValueError(json.dumps({"geometry_errors": geometry_errors, "duplicate_masks": duplicate_masks}, indent=2))
    if len(image_rows) != EXPECTED_IMAGES or sum(row["annotation_count"] for row in image_rows) != EXPECTED_ANNOTATIONS:
        raise ValueError("Converted totals differ from the COCO manifest.")

    by_file = {row["coco_filename"]: row for row in image_rows}
    flagged_by_key = {(row["image"], row["local_instance"]): row for row in flagged}
    overlay_findings = []
    for finding in prior_review.get("suspicious_overlaps", []):
        name = finding["image"]
        for local_index in finding["instances"]:
            item = flagged_by_key[(name, local_index)]
            item["reason"] = "member of a suspiciously overlapping pair; human confirmation"
            overlay_findings.append(item)
    tiny_findings = [row for row in flagged if row["reason"] == "tiny mask; human confirmation"]
    preview_targets = [
        ("IMG_4571.jpeg", [10, 13], "IMG_4571_overlap_instances_10_13.png", "Overlap review"),
        ("IMG_4571.jpeg", [16, 17, 18], "IMG_4571_tiny_instances_16_18.png", "Tiny-mask review"),
        ("IMG_4588.jpeg", [7], "IMG_4588_tiny_instance_7.png", "Tiny-mask review"),
    ]

    # All validation is complete before the output directory is created.
    OUT.mkdir(parents=True)
    label_dir = OUT / "labels" / "train"
    preview_dir = OUT / "review_previews"
    label_dir.mkdir(parents=True)
    preview_dir.mkdir()
    for filename, content in label_contents.items():
        (label_dir / filename).write_text(content, encoding="utf-8", newline="\n")

    with (OUT / "source_mapping.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    for filename, local_indices, output_name, title in preview_targets:
        image_info = next(row for row in images if row["file_name"] == filename)
        source_path = Path(by_file[filename]["verified_original_path"])
        anns = annotations_by_id[image_info["id"]]
        selected = [(idx, anns[idx - 1]) for idx in local_indices]
        details = [flagged_by_key[(filename, idx)] for idx in local_indices]
        x0 = min(item["bbox_xyxy_px"][0] for item in details)
        y0 = min(item["bbox_xyxy_px"][1] for item in details)
        x1 = max(item["bbox_xyxy_px"][2] for item in details)
        y1 = max(item["bbox_xyxy_px"][3] for item in details)
        span = max(x1 - x0, y1 - y0)
        margin = max(55, span * 0.18)
        left = max(0, math.floor(x0 - margin))
        top = max(0, math.floor(y0 - margin))
        right = min(image_info["width"], math.ceil(x1 + margin))
        bottom = min(image_info["height"], math.ceil(y1 + margin))
        with Image.open(source_path) as original:
            photo = original.convert("RGB").crop((left, top, right, bottom))
        scale = min(4.0, 2100 / max(1, photo.width), 1500 / max(1, photo.height))
        photo = photo.resize((max(1, round(photo.width * scale)), max(1, round(photo.height * scale))), Image.Resampling.LANCZOS)
        overlay = Image.new("RGBA", photo.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        try:
            label_font = ImageFont.truetype("arial.ttf", 20)
            body_font = ImageFont.truetype("arial.ttf", 15)
            title_font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            label_font = body_font = title_font = ImageFont.load_default()
        for palette_index, (local_idx, annotation) in enumerate(selected):
            ann_id = annotation["id"]
            polygon = flagged_by_key[(filename, local_idx)]["polygon_xy_px"]
            points = [(round((x - left) * scale), round((y - top) * scale)) for x, y in polygon]
            color = PALETTE[(local_idx - 1) % len(PALETTE)]
            draw.polygon(points, fill=(*color, 82), outline=(*color, 255), width=max(3, round(3 * scale)))
            coords = flagged_by_key[(filename, local_idx)]["bbox_xyxy_px"]
            label_x = min(photo.width - 10, max(4, round((coords[0] - left) * scale)))
            label_y = min(photo.height - 30, max(4, round((coords[1] - top) * scale)))
            label = f"instance {local_idx} | COCO annotation {ann_id}"
            bounds = draw.textbbox((label_x, label_y), label, font=label_font)
            draw.rounded_rectangle((bounds[0] - 5, bounds[1] - 3, bounds[2] + 5, bounds[3] + 3), radius=4, fill=(12, 18, 25, 235))
            draw.text((label_x, label_y), label, font=label_font, fill=(255, 255, 255, 255))
        composite = Image.alpha_composite(photo.convert("RGBA"), overlay).convert("RGB")
        footer_h = 88
        final = Image.new("RGB", (composite.width, composite.height + footer_h), (25, 33, 42))
        final.paste(composite, (0, 0))
        footer = ImageDraw.Draw(final)
        footer.text((12, composite.height + 9), f"{filename} — {title}; crop coordinates are full-image pixels.", font=title_font, fill="white")
        footer.text((12, composite.height + 42), "; ".join(
            f"inst {item['local_instance']} / ann {item['annotation_id']} bbox xyxy {item['bbox_xyxy_px']}"
            for item in details
        ), font=body_font, fill=(220, 227, 234))
        final.save(preview_dir / output_name, format="PNG", optimize=True)

    flagged_details = [row for row in flagged if row["reason"]]
    (OUT / "flagged_annotation_details.json").write_text(
        json.dumps(flagged_details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_lines = [
        "# Flagged Annotation Details", "",
        "Coordinates below are COCO full-resolution pixel coordinates. Tiny masks remain unchanged; visual confirmation is requested.", "",
    ]
    for item in flagged_details:
        md_lines.extend([
            f"## {item['image']} — instance {item['local_instance']} / annotation {item['annotation_id']}", "",
            f"Reason: {item['reason']}",
            f"Bounding box xyxy (px): `{item['bbox_xyxy_px']}`; xywh (px): `{item['bbox_xywh_px']}`",
            f"Polygon vertices xy (px): `{item['polygon_xy_px']}`",
            f"YOLO-normalized vertices xy: `{item['polygon_xy_normalized']}`", "",
        ])
    (OUT / "flagged_annotation_details.md").write_text("\n".join(md_lines), encoding="utf-8")

    image_iou_min = min(all_iou)
    image_iou_mean = sum(all_iou) / len(all_iou)
    total_annotations = sum(row["annotation_count"] for row in image_rows)
    report = {
        "status": "converted and validated; proposal only; not promoted; training not started",
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": archive_hash,
        "class_mapping": {str(coco_class_id): 0, "name": "product"},
        "image_records_converted": len(image_rows),
        "annotations_converted": total_annotations,
        "invalid_polygons": len(geometry_errors),
        "duplicate_masks": len(duplicate_masks),
        "coordinate_max_abs_roundtrip_error_px": max_coordinate_error,
        "raster_mask_iou_min": image_iou_min,
        "raster_mask_iou_mean": image_iou_mean,
        "instances_with_exact_raster_roundtrip": sum(iou == 1.0 for iou in all_iou),
        "raster_mask_rounding_differences": raster_differences,
        "tiny_annotations_for_review": [{key: item[key] for key in ("image", "local_instance", "annotation_id", "bbox_xyxy_px", "area_px2", "image_fraction_pct")}
                                         for item in tiny_findings],
        "overlap_annotations_for_review": [{key: item[key] for key in ("image", "local_instance", "annotation_id", "bbox_xyxy_px")}
                                            for item in overlay_findings],
        "source_photos": {
            "copied": False,
            "hardlinked": False,
            "symlinked": False,
            "mapped_original_bytes": total_source_bytes,
            "mapping_file": str(OUT / "source_mapping.csv"),
        },
        "yolo_label_directory": str(label_dir),
        "yolo_label_rows": total_annotations,
        "preview_paths": [str(preview_dir / item[2]) for item in preview_targets],
        "training_ready_note": "YOLO segmentation label files and image mapping are staged. No photos were duplicated; standard Ultralytics training requires a later approved materialization step or a custom image-path loader.",
    }
    (OUT / "conversion_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    readme = f"""# CVAT 22-image export conversion staging

This proposal contains {len(image_rows)} YOLO Segmentation label files and {total_annotations} class-0 (`product`) polygons converted from the reviewed COCO export. It does not modify or promote the approved seed dataset.

## Storage and image mapping

Photographs were not copied, hard-linked, or symlinked. The exact pixel-verified original paths and SHA-256 hashes are in `source_mapping.csv`; all mapped photos remain in `data/test_runs/`. This avoids another {total_source_bytes / (1024 * 1024):.1f} MiB of image duplication and prevents training tools from writing through a link into originals.

YOLO label rows are in `labels/train/`. The stage is not a self-contained Ultralytics dataset: before a future training run, an explicitly approved materialization or custom image-path loader is needed to pair labels with those mapped paths. The existing approved seed remains untouched.

## Validation

- Images: {len(image_rows)}; annotations: {total_annotations}; class mapping: COCO `product` to YOLO class `0`.
- Invalid polygons: {len(geometry_errors)}; exact duplicate masks: {len(duplicate_masks)}.
- Maximum normalized-coordinate round-trip error: {max_coordinate_error:.6f} px.
- Raster-mask IoU after serialization: min {image_iou_min:.6f}, mean {image_iou_mean:.6f}; exact raster round trips {sum(iou == 1.0 for iou in all_iou)}/{len(all_iou)}.
- Tiny masks and the overlap pair remain present and need the requested human check; details and close-ups are under `review_previews/` and in `flagged_annotation_details.md`.

No model training was started. See `conversion_validation.json` and `source_mapping.csv` for the machine-readable audit trail.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    convert()
