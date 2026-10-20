#!/usr/bin/env python3
"""Generate scene masks for the detected boxes of the 30 evaluated scenes.

The registration references already carry masks: MobileSAM proposed them, a human
inspected the boundary and approved, and the approved artifacts are committed under
`03_.../registration/`. The scene photographs had no masks at all, so the demo's
`masked` gallery could only ever change similarity numbers, never imagery.

This script produces the missing scene half of that dataset using the same backend
and the same artifact recipe as `review_app.py::_approve_mask`:

  * MobileSAM (`data/models/mobile_sam.pt`), CPU, 1024px working image
  * mask stored full-resolution as uint8 {0,1} in an `.npz`
  * polygons via `hybrid_pipeline.segmentation_models.mask_polygons`
  * box crop padded by 8%, masked crop with background set to (124, 124, 124)

It differs from registration in exactly one way, and the difference is recorded in
every row of the index: the prompt is the committed DINO detection box rather than a
whole-image proposal, and **no human has approved these masks**. Every row is written
with `review_status=unreviewed_proposal`. They are model output, not annotations, and
nothing downstream may present them as approved.

Run from the repository root:

    python experiments/new_product_scanning_20261020/build_scene_masks.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.hybrid_pipeline.segmentation_models import mask_polygons, render_masks  # noqa: E402

DATASET = ROOT / "03_newest_10_products_owner_employee_customer"
PHOTOS = DATASET / "photos"
MANIFEST = DATASET / "manifest_this_part.csv"
DETECTIONS = (ROOT / "experiments/new_product_scanning_20261020/results"
              / "imported_runpod_scene_results_20261020_v5_flat/detections.csv")
WEIGHTS = ROOT / "data/models/mobile_sam.pt"

OUT = DATASET / "scene_masks"
ARTIFACTS = OUT / "artifacts"
INDEX = OUT / "scene_mask_index.csv"
RUN_RECORD = OUT / "scene_mask_run.json"

WORKING_SIDE = 1024
CROP_PAD_FRACTION = 0.08
MASK_BACKGROUND = (124, 124, 124)
JPEG_QUALITY = 95
BACKEND = "mobile_sam.pt:box-prompt:cpu"
REVIEW_STATUS = "unreviewed_proposal"

FIELDS = [
    "source_photo", "source_sha256", "dataset_type", "detection_id",
    "detector_confidence", "mask_file", "mask_file_sha256",
    "polygon_file", "polygon_file_sha256", "box_crop_file", "box_crop_file_sha256",
    "masked_crop_file", "masked_crop_file_sha256",
    "mask_area_fraction", "leak_outside_box_fraction", "backend", "review_status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def photo_table() -> dict[str, dict[str, str]]:
    """Map filename -> manifest row, so every source photo is hash-verified."""
    table: dict[str, dict[str, str]] = {}
    for row in read_csv(MANIFEST):
        table[row["original_filename"]] = row
    return table


def load_verified(path: Path, expected_sha: str) -> Image.Image:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"Source image hash changed since the manifest: {path.name}")
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def write_artifacts(image: Image.Image, mask: np.ndarray, box: tuple[float, float, float, float],
                    photo_stem: str, detection_id: str, source_name: str,
                    source_sha: str) -> dict[str, object]:
    """Identical artifact recipe to review_app.py::_approve_mask."""
    key = f"{photo_stem}_{source_sha[:10]}_{detection_id}"

    mask_path = ARTIFACTS / "masks" / f"{key}.npz"
    polygon_path = ARTIFACTS / "polygons" / f"{key}.json"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    polygon_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
    polygon_path.write_text(json.dumps({
        "source_file": source_name,
        "source_sha256": source_sha,
        "detection_id": detection_id,
        "width": image.width,
        "height": image.height,
        "coordinate_space": "EXIF-transposed displayed-image pixels",
        "polygons": mask_polygons(mask),
    }, indent=2), encoding="utf-8")

    x1, y1, x2, y2 = [int(round(value)) for value in box]
    pad = max(2, round(max(x2 - x1, y2 - y1) * CROP_PAD_FRACTION))
    crop_box = (max(0, x1 - pad), max(0, y1 - pad),
                min(image.width, x2 + pad), min(image.height, y2 + pad))
    box_crop = image.crop(crop_box)
    pixels = np.asarray(image).copy()
    pixels[~mask] = MASK_BACKGROUND
    masked_crop = Image.fromarray(pixels).crop(crop_box)

    crop_dir = ARTIFACTS / "scene_crops" / photo_stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    box_crop_path = crop_dir / f"{detection_id}_box.jpg"
    masked_crop_path = crop_dir / f"{detection_id}_masked.jpg"
    box_crop.save(box_crop_path, quality=JPEG_QUALITY)
    masked_crop.save(masked_crop_path, quality=JPEG_QUALITY)

    def rel(path: Path) -> str:
        return path.relative_to(DATASET).as_posix()

    return {
        "mask_file": rel(mask_path), "mask_file_sha256": sha256_file(mask_path),
        "polygon_file": rel(polygon_path), "polygon_file_sha256": sha256_file(polygon_path),
        "box_crop_file": rel(box_crop_path), "box_crop_file_sha256": sha256_file(box_crop_path),
        "masked_crop_file": rel(masked_crop_path),
        "masked_crop_file_sha256": sha256_file(masked_crop_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N photographs (smoke runs).")
    args = parser.parse_args()

    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"MobileSAM checkpoint not found: {WEIGHTS}. "
                                "Run `python model_assets.py` first.")

    manifest = photo_table()
    paths = {path.name: path for folder in PHOTOS.iterdir() if folder.is_dir()
             for path in folder.glob("*.jpeg")}

    by_photo: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(DETECTIONS):
        by_photo.setdefault(row["photo_filename"], []).append(row)
    names = sorted(by_photo)
    if args.limit:
        names = names[:args.limit]

    from ultralytics import SAM
    model = SAM(str(WEIGHTS))

    records: list[dict[str, object]] = []
    for position, name in enumerate(names, start=1):
        detections = by_photo[name]
        if name not in manifest:
            raise KeyError(f"{name} is not in {MANIFEST.name}; refusing to guess its hash.")
        if name not in paths:
            raise FileNotFoundError(f"Source photograph is unavailable: {name}")
        source_sha = manifest[name]["sha256"]
        image = load_verified(paths[name], source_sha)

        working = image.copy()
        working.thumbnail((WORKING_SIDE, WORKING_SIDE))
        scale_x, scale_y = working.width / image.width, working.height / image.height
        boxes_small = [[float(d["x1"]) * scale_x, float(d["y1"]) * scale_y,
                        float(d["x2"]) * scale_x, float(d["y2"]) * scale_y] for d in detections]

        result = model.predict(source=np.asarray(working), bboxes=boxes_small,
                               device="cpu", verbose=False, save=False)
        if not result or result[0].masks is None:
            raise RuntimeError(f"MobileSAM returned no mask for {name}")
        small_masks = result[0].masks.data.detach().cpu().numpy() > 0.5
        if len(small_masks) != len(detections):
            raise RuntimeError(f"{name}: expected {len(detections)} masks, got {len(small_masks)}")

        print(f"[{position}/{len(names)}] {name}: {len(detections)} box(es)", flush=True)
        overlay_items: list[dict[str, object]] = []
        for detection, small, box_small in zip(detections, small_masks, boxes_small):
            mask = cv2.resize(small.astype(np.uint8), (image.width, image.height),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            if not mask.any():
                raise ValueError(f"{name} {detection['detection_id']}: mask is empty at full size.")

            bx1, by1, bx2, by2 = [int(round(value)) for value in box_small]
            inside = int(small[max(0, by1):by2, max(0, bx1):bx2].sum())
            total = int(small.sum())
            leak = 0.0 if total == 0 else round((total - inside) / total, 4)

            box_full = (float(detection["x1"]), float(detection["y1"]),
                        float(detection["x2"]), float(detection["y2"]))
            artifacts = write_artifacts(image, mask, box_full, Path(name).stem,
                                        detection["detection_id"], name, source_sha)
            overlay_items.append({
                "mask": mask, "box": box_full,
                "sku": f"{detection['detection_id']} {float(detection['detector_confidence']):.2f}",
            })
            records.append({
                "source_photo": f"photos/{paths[name].parent.name}/{name}",
                "source_sha256": source_sha,
                "dataset_type": detection["dataset_type"],
                "detection_id": detection["detection_id"],
                "detector_confidence": detection["detector_confidence"],
                **artifacts,
                "mask_area_fraction": round(float(mask.mean()), 4),
                "leak_outside_box_fraction": leak,
                "backend": BACKEND,
                "review_status": REVIEW_STATUS,
            })

        overlay_dir = ARTIFACTS / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        render_masks(image, overlay_items).save(
            overlay_dir / f"{Path(name).stem}_masks.jpg", quality=JPEG_QUALITY)

    INDEX.parent.mkdir(parents=True, exist_ok=True)
    with INDEX.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    RUN_RECORD.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backend": BACKEND,
        "weights_sha256": sha256_file(WEIGHTS),
        "detections_source": DETECTIONS.relative_to(ROOT).as_posix(),
        "detections_sha256": sha256_file(DETECTIONS),
        "working_side": WORKING_SIDE,
        "crop_pad_fraction": CROP_PAD_FRACTION,
        "mask_background_rgb": list(MASK_BACKGROUND),
        "jpeg_quality": JPEG_QUALITY,
        "photographs": len(names),
        "masks": len(records),
        "review_status": REVIEW_STATUS,
        "note": ("MobileSAM proposals prompted with the committed DINO detection boxes. "
                 "No human has reviewed these masks. Unlike the registration masks in "
                 "registration/mask_index.csv, they are not approved annotations."),
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(records)} masks for {len(names)} photographs")
    print(f"  index: {INDEX}")
    print(f"  run:   {RUN_RECORD}")
    print(f"  status: every row is {REVIEW_STATUS} — not approved annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
