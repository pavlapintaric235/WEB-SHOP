#!/usr/bin/env python3
"""Run the local CPU shape detector over the 30 evaluated scenes.

The committed scene proposals came from NVIDIA Retail DINO on a rented GPU. That path
cannot run on a machine without TensorRT/CUDA, so no new photograph could be processed
locally and the archived predictions could never be refreshed. This script reproduces the
whole proposal stage on CPU using `shape_recognition` (OWLv2 boxes, SAM 2.1 masks, both
Apache-2.0), and writes a result directory in the exact shape the existing consumers read:

  * `predictions.csv` carries the columns `review_app.py` displays *and* the nine columns
    `evaluate_scene_predictions.py` validates, so one file satisfies both.
  * `report.json` carries `experiment: new_product_scanning_20261020`, without which
    `review_app.py::_prediction_runs` silently ignores the directory.
  * masks follow `build_scene_masks.py`: full-resolution uint8 `.npz` under key `mask`,
    a polygon JSON in EXIF-transposed displayed-image pixels, and padded box/masked crops.

Two honesty rules are enforced in code rather than left to the caller:

  * Every row is written `acceptance_status=abstained` with an empty `sku_id`. Zero of the
    ten SKU identities have been approved, so no identity claim is defensible yet. The
    evaluator accepts this and scores detection quality only. SKU suggestions are reported
    separately in the `top1..top3` columns as provisional ranks.
  * `accuracy_metrics` stays null. Accuracy requires reviewed scene labels, which do not
    exist. Nothing here may be quoted as precision, recall or accuracy.

Run from the repository root:

    python experiments/new_product_scanning_20261020/run_local_shape_detector.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.hybrid_pipeline.segmentation_models import mask_polygons, render_masks  # noqa: E402
from shape_recognition import (  # noqa: E402
    BACKEND,
    DETECTOR_ID,
    MASK_ID,
    PROMPTS,
    REVIEW_STATUS,
    ShapeDetector,
    mask_leak_fraction,
)

EXPERIMENT = ROOT / "experiments/new_product_scanning_20261020"
DATASET = ROOT / "03_newest_10_products_owner_employee_customer"
PHOTOS = DATASET / "photos"
MANIFEST = EXPERIMENT / "audit/photo_manifest.csv"
CANDIDATES = EXPERIMENT / "audit/candidate_skus.csv"
REFERENCE_INDEX = DATASET / "registration/mask_index.csv"
RESULTS = EXPERIMENT / "results"

SCENE_GROUPS = {"employee", "customer"}
CROP_PAD_FRACTION = 0.08
MASK_BACKGROUND = (124, 124, 124)
JPEG_QUALITY = 95
RECOGNITION_STATUS = "unaccepted_provisional_rank_no_unknown_threshold"

PREDICTION_FIELDS = [
    # Consumed by evaluate_scene_predictions.py
    "photo_filename", "detection_id", "sku_id", "acceptance_status", "confidence",
    "x1", "y1", "x2", "y2",
    # Consumed by review_app.py
    "dataset_type", "detector_confidence",
    "top1_sku_id", "top1_name", "top1_similarity",
    "top2_sku_id", "top2_name", "top2_similarity",
    "top3_sku_id", "top3_name", "top3_similarity",
    # Provenance
    "detector_label", "mask_file", "mask_area_fraction", "leak_outside_box_fraction",
    "query_crop_sha256", "recognition_status", "review_status",
]

DETECTION_FIELDS = ["photo_filename", "dataset_type", "detection_id",
                    "detector_confidence", "x1", "y1", "x2", "y2"]

MASK_INDEX_FIELDS = [
    "source_photo", "source_sha256", "dataset_type", "detection_id", "detector_confidence",
    "mask_file", "mask_file_sha256", "polygon_file", "polygon_file_sha256",
    "box_crop_file", "box_crop_file_sha256", "masked_crop_file", "masked_crop_file_sha256",
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


def scene_rows() -> list[dict[str, str]]:
    """The evaluated scene photographs, from the hash-verified audit manifest."""
    rows = [row for row in read_csv(MANIFEST)
            if row["role"] == "scene" and row["group"] in SCENE_GROUPS]
    return sorted(rows, key=lambda row: row["filename"])


def load_verified(path: Path, expected_sha: str) -> Image.Image:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise ValueError(f"Source image hash changed since the manifest: {path.name}")
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def padded_crop_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    pad = max(2, round(max(x2 - x1, y2 - y1) * CROP_PAD_FRACTION))
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(width, x2 + pad), min(height, y2 + pad))


def masked_crop(image: Image.Image, mask: np.ndarray, box) -> Image.Image:
    pixels = np.asarray(image).copy()
    pixels[~mask] = MASK_BACKGROUND
    return Image.fromarray(pixels).crop(padded_crop_box(box, image.width, image.height))


class ReferenceGallery:
    """Approved owner reference crops, encoded once, for provisional SKU suggestions.

    This reuses the repository's existing local ResNet18 encoder rather than introducing a
    second recognizer. Its output is a provisional rank, never an identity: the ten SKU
    identities are unapproved, so callers must keep abstaining.
    """

    def __init__(self) -> None:
        from sam_vision import ReferenceEncoder

        names = {row["sku_id"]: row["proposed_name"] for row in read_csv(CANDIDATES)}
        crops, skus = [], []
        for row in read_csv(REFERENCE_INDEX):
            crop_path = DATASET / row["masked_crop_file"]
            if not crop_path.is_file():
                continue
            with Image.open(crop_path) as source:
                crops.append(source.convert("RGB").copy())
            skus.append(row["provisional_sku_id"])
        self.names = names
        self.skus = skus
        self.encoder = ReferenceEncoder()
        self.features = self.encoder.encode(crops) if crops else np.empty((0, 512), np.float32)

    def rank(self, crops: list[Image.Image]) -> list[list[tuple[str, str, float]]]:
        """Top three (sku_id, name, similarity) per crop, best reference view per SKU."""
        if not crops or len(self.features) == 0:
            return [[] for _ in crops]
        queries = self.encoder.encode(crops)
        ranked = []
        for row in queries @ self.features.T:
            best: dict[str, float] = {}
            for sku, score in zip(self.skus, row):
                best[sku] = max(best.get(sku, -1.0), float(score))
            top = sorted(best.items(), key=lambda item: -item[1])[:3]
            ranked.append([(sku, self.names.get(sku, ""), round(score, 6)) for sku, score in top])
        return ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N scenes.")
    parser.add_argument("--threshold", type=float, default=None, help="Detector score threshold.")
    parser.add_argument("--no-recognition", action="store_true",
                        help="Skip provisional SKU ranking and emit detection geometry only.")
    parser.add_argument("--output-name", default=None, help="Result directory name.")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / (args.output_name or f"local_owlv2_sam21_{stamp}")
    artifacts = out / "artifacts"
    (artifacts / "masks").mkdir(parents=True, exist_ok=True)
    (artifacts / "polygons").mkdir(parents=True, exist_ok=True)
    (artifacts / "overlays").mkdir(parents=True, exist_ok=True)

    rows = scene_rows()
    if args.limit:
        rows = rows[:args.limit]
    paths = {path.name: path for folder in PHOTOS.iterdir() if folder.is_dir()
             for path in folder.glob("*.jpeg")}

    detector = ShapeDetector(**({"threshold": args.threshold} if args.threshold else {}))
    gallery = None if args.no_recognition else ReferenceGallery()

    predictions: list[dict[str, object]] = []
    detections_flat: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    scenes: list[dict[str, object]] = []
    started = time.perf_counter()

    for position, row in enumerate(rows, start=1):
        name = row["filename"]
        if name not in paths:
            raise FileNotFoundError(f"Source photograph is unavailable: {name}")
        image = load_verified(paths[name], row["sha256"])
        stem = Path(name).stem
        short = row["sha256"][:10]

        elapsed = time.perf_counter()
        found = detector.detect(image, original_resolution=False)
        seconds = round(time.perf_counter() - elapsed, 2)

        # `detect` returned boxes and masks at the detector's working resolution so that
        # only one mask at a time is ever held at full size. Derive the scale from that
        # working image rather than from a returned mask, which may be absent.
        working = detector.working_image(image)
        working_width = working.width
        scale_x, scale_y = image.width / working.width, image.height / working.height

        crops: list[Image.Image] = []
        staged: list[dict[str, object]] = []
        overlay_items: list[dict[str, object]] = []

        for index, detection in enumerate(found, start=1):
            detection_id = f"det-{index}"
            box = (detection.box[0] * scale_x, detection.box[1] * scale_y,
                   detection.box[2] * scale_x, detection.box[3] * scale_y)
            mask = detection.mask
            if mask is not None and mask.shape != (image.height, image.width):
                mask = cv2.resize(mask.astype(np.uint8), (image.width, image.height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            if mask is None or not mask.any():
                raise ValueError(f"{name} {detection_id}: mask is empty at full size.")

            key = f"{stem}_{short}_{detection_id}"
            mask_path = artifacts / "masks" / f"{key}.npz"
            polygon_path = artifacts / "polygons" / f"{key}.json"
            np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
            polygon_path.write_text(json.dumps({
                "source_file": name,
                "source_sha256": row["sha256"],
                "detection_id": detection_id,
                "width": image.width,
                "height": image.height,
                "coordinate_space": "EXIF-transposed displayed-image pixels",
                "polygons": mask_polygons(mask),
            }, indent=2), encoding="utf-8")

            crop_dir = artifacts / "scene_crops" / stem
            crop_dir.mkdir(parents=True, exist_ok=True)
            box_crop_path = crop_dir / f"{detection_id}_box.jpg"
            masked_crop_path = crop_dir / f"{detection_id}_masked.jpg"
            image.crop(padded_crop_box(box, image.width, image.height)).save(
                box_crop_path, quality=JPEG_QUALITY)
            crop = masked_crop(image, mask, box)
            crop.save(masked_crop_path, quality=JPEG_QUALITY)
            crops.append(crop)

            leak = mask_leak_fraction(mask, box)
            area_fraction = round(float(mask.mean()), 4)
            overlay_items.append({"mask": mask, "box": box,
                                  "sku": f"{detection_id} {detection.score:.2f}"})
            staged.append({
                "detection_id": detection_id,
                "box": box,
                "score": detection.score,
                "label": detection.label,
                "mask_file": mask_path.relative_to(out).as_posix(),
                "mask_area_fraction": area_fraction,
                "leak": leak,
                "query_crop_sha256": sha256_file(masked_crop_path),
            })
            mask_rows.append({
                "source_photo": f"photos/{paths[name].parent.name}/{name}",
                "source_sha256": row["sha256"],
                "dataset_type": row["group"],
                "detection_id": detection_id,
                "detector_confidence": detection.score,
                "mask_file": mask_path.relative_to(out).as_posix(),
                "mask_file_sha256": sha256_file(mask_path),
                "polygon_file": polygon_path.relative_to(out).as_posix(),
                "polygon_file_sha256": sha256_file(polygon_path),
                "box_crop_file": box_crop_path.relative_to(out).as_posix(),
                "box_crop_file_sha256": sha256_file(box_crop_path),
                "masked_crop_file": masked_crop_path.relative_to(out).as_posix(),
                "masked_crop_file_sha256": sha256_file(masked_crop_path),
                "mask_area_fraction": area_fraction,
                "leak_outside_box_fraction": leak,
                "backend": BACKEND,
                "review_status": REVIEW_STATUS,
            })

        ranked = gallery.rank(crops) if gallery else [[] for _ in crops]
        for item, suggestions in zip(staged, ranked):
            box = item["box"]
            record = {
                "photo_filename": name,
                "detection_id": item["detection_id"],
                "sku_id": "",
                "acceptance_status": "abstained",
                "confidence": item["score"],
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "dataset_type": row["group"],
                "detector_confidence": item["score"],
                "detector_label": item["label"],
                "mask_file": item["mask_file"],
                "mask_area_fraction": item["mask_area_fraction"],
                "leak_outside_box_fraction": item["leak"],
                "query_crop_sha256": item["query_crop_sha256"],
                "recognition_status": RECOGNITION_STATUS,
                "review_status": REVIEW_STATUS,
            }
            for rank in range(3):
                sku, label, score = suggestions[rank] if rank < len(suggestions) else ("", "", "")
                record[f"top{rank + 1}_sku_id"] = sku
                record[f"top{rank + 1}_name"] = label
                record[f"top{rank + 1}_similarity"] = score
            predictions.append(record)
            detections_flat.append({
                "photo_filename": name, "dataset_type": row["group"],
                "detection_id": item["detection_id"], "detector_confidence": item["score"],
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            })

        overlay_name = f"{stem}_shapes.jpg"
        render_masks(image, overlay_items).save(
            artifacts / "overlays" / overlay_name, quality=JPEG_QUALITY)
        scenes.append({
            "photo_filename": name, "dataset_type": row["group"],
            "detection_count": len(found), "detector_seconds": seconds,
            "overlay": f"artifacts/overlays/{overlay_name}",
            "working_width": working_width,
        })
        print(f"[{position}/{len(rows)}] {name}: {len(found)} item(s) in {seconds:.1f}s", flush=True)

    def write_csv(path: Path, fields: list[str], records: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)

    write_csv(out / "predictions.csv", PREDICTION_FIELDS, predictions)
    write_csv(out / "detections.csv", DETECTION_FIELDS, detections_flat)
    write_csv(out / "scene_mask_index.csv", MASK_INDEX_FIELDS, mask_rows)

    (out / "report.json").write_text(json.dumps({
        "status": "PASS",
        "experiment": "new_product_scanning_20261020",
        "scope": "local CPU open-vocabulary detection and instance segmentation",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": "cpu",
        "backend": BACKEND,
        "detector": {"name": DETECTOR_ID, "license": "Apache-2.0", "prompts": list(PROMPTS),
                     "threshold": detector.threshold, "nms_iou": detector.nms_iou,
                     "working_side": detector.working_side},
        "segmenter": {"name": MASK_ID, "license": "Apache-2.0", "prompt": "detected box"},
        "recognizer": ({"name": "resnet18 reference encoder (in-repo)",
                        "gallery_images": len(gallery.skus),
                        "gallery_skus": len(set(gallery.skus)),
                        "similarities_are_calibrated_confidence": False,
                        "unknown_rejection_threshold": None} if gallery else None),
        "scene_images": len(rows),
        "detected_boxes": len(predictions),
        "total_seconds": round(time.perf_counter() - started, 1),
        "accuracy_metrics": None,
        "review_status": REVIEW_STATUS,
        "limitations": [
            "Every row is written acceptance_status=abstained with an empty sku_id because "
            "none of the ten SKU identities has been approved. The top1..top3 columns are "
            "provisional ranks from the in-repo ResNet18 encoder, not identities.",
            "accuracy_metrics is null by design. Scene labels do not exist, so no precision, "
            "recall, count-error or accuracy figure may be quoted from this run.",
            "Masks are box-prompted model output. No human has reviewed them; every row is "
            "review_status=unreviewed_proposal.",
            "Two identical items stacked vertically and touching are reported as one box. "
            "The detector does not separate them at any usable threshold, so a reviewer must "
            "split such a box by hand.",
            "A tall item can still attract more than one box when its parts score separately.",
        ],
        "scenes": scenes,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote {len(predictions)} proposals for {len(rows)} photographs")
    print(f"  results: {out}")
    print(f"  status:  every row is {REVIEW_STATUS} / abstained - not approved annotations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
