"""Build a portable, single-class YOLO detection training bundle for RunPod."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/runpod_yolo_detection"))
from dataset_utils import convert_segmentation_text  # noqa: E402


SOURCE = ROOT / "data/segmentation_dataset/experiments/initial_yolo26n_seg_20261017"
TEMPLATES = ROOT / "experiments/runpod_yolo_detection"
ARTIFACT_ROOT = ROOT / "data/segmentation_dataset/experiments/runpod_yolo11n_detection_20261017"
PACKAGE_ROOT = ARTIFACT_ROOT / "package"
ZIP_PATH = ARTIFACT_ROOT / "inventorylens_runpod_training.zip"
EXPECTED_COUNTS = {"train": (21, 130), "validation": (5, 33)}
EXPECTED_IMAGES = 26
EXPECTED_INSTANCES = 163
EXPECTED_RETAIL_PT_SHA256 = "4888d44cee1a3c55fa961599b762bad0e5aa31194237d8d10d717dff15e0cfdb"
EXCLUDED_DIAGNOSTIC_RUNS = [
    "20261017-014015-b2a337ec",
    "20261017-014239-235d7e51 (01:42 held-out arrangement view)",
    "20261017-014423-e7b3fb42 (01:44 held-out arrangement view)",
    "20261017-014554-b9412539",
    "20261017-014722-cc4ab8d3",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_sources() -> tuple[list[dict], dict]:
    rows = read_manifest(SOURCE / "dataset_manifest.csv")
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} approved rows, found {len(rows)}")
    counts = {split: {"images": 0, "instances": 0, "groups": set()} for split in EXPECTED_COUNTS}
    seen_names: set[str] = set()
    total = 0
    for row in rows:
        split = row["split"]
        if split not in counts:
            raise ValueError(f"Unexpected split in source manifest: {split}")
        image_path = (SOURCE / row["image_path"]).resolve(strict=True)
        label_path = (SOURCE / row["label_path"]).resolve(strict=True)
        if image_path.name != row["original_filename"] or image_path.stem != label_path.stem:
            raise ValueError(f"Image/label mapping mismatch for {image_path.name}")
        if image_path.name.lower() in {"img_4699.jpeg", "img_4699.png"}:
            raise ValueError("Final-test image leaked into training or validation")
        if image_path.name in seen_names:
            raise ValueError(f"Duplicate image filename: {image_path.name}")
        seen_names.add(image_path.name)
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
        if (width, height) != (int(row["width"]), int(row["height"])):
            raise ValueError(f"Image dimensions disagree with source manifest for {image_path.name}")
        if sha256(image_path) != row["image_sha256"] or sha256(label_path) != row["label_sha256"]:
            raise ValueError(f"Source image/label hash disagrees with approved manifest for {image_path.name}")
        converted, instances = convert_segmentation_text(label_path.read_text(encoding="utf-8"), str(label_path))
        if instances != int(row["annotation_count"]):
            raise ValueError(f"Polygon count mismatch for {image_path.name}: {instances} != {row['annotation_count']}")
        counts[split]["images"] += 1
        counts[split]["instances"] += instances
        counts[split]["groups"].add(row["group_id"])
        total += instances
        row["_source_image"] = str(image_path)
        row["_source_label"] = str(label_path)
        row["_converted_label"] = converted
        row["_width"] = str(width)
        row["_height"] = str(height)
    if counts["train"]["groups"] & counts["validation"]["groups"]:
        raise ValueError("A photographic arrangement group crosses the train/validation split")
    for split, expected in EXPECTED_COUNTS.items():
        observed = (counts[split]["images"], counts[split]["instances"])
        if observed != expected:
            raise ValueError(f"{split} expected {expected}, found {observed}")
    if total != EXPECTED_INSTANCES:
        raise ValueError(f"Expected {EXPECTED_INSTANCES} total instances, found {total}")
    return rows, {
        split: {"images": value["images"], "instances": value["instances"], "distinct_groups": len(value["groups"])}
        for split, value in counts.items()
    }


def validate_checkpoint(path: Path) -> str:
    path = path.expanduser().resolve(strict=True)
    if path.suffix.lower() != ".pt":
        raise ValueError("The fine-tuning checkpoint must be PyTorch .pt, not inference-only ONNX")
    digest = sha256(path)
    if digest != EXPECTED_RETAIL_PT_SHA256:
        raise ValueError(f"Retail YOLO11n checkpoint checksum mismatch: {digest}")
    return digest


def preflight_size(rows: list[dict], checkpoint: Path, test_image: Path) -> int:
    payload = checkpoint.stat().st_size + test_image.stat().st_size
    payload += sum(Path(row["_source_image"]).stat().st_size + len(row["_converted_label"].encode("utf-8")) for row in rows)
    payload += 8 * 1024 * 1024
    required = payload * 2 + 1024**3
    free = shutil.disk_usage(ROOT).free
    if free < required:
        raise OSError(f"Insufficient disk space: need {required:,} bytes including reserve; have {free:,}. No package files created.")
    return payload


def build_package(rows: list[dict], splits: dict, checkpoint: Path, checkpoint_sha: str, test_image: Path) -> dict:
    if PACKAGE_ROOT.exists() or ZIP_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing package/archive under {ARTIFACT_ROOT}")
    ARTIFACT_ROOT.mkdir(parents=True)
    for row in rows:
        split = row["split"]
        source_image = Path(row["_source_image"])
        out_image = PACKAGE_ROOT / "images" / split / source_image.name
        out_label = PACKAGE_ROOT / "labels" / split / f"{source_image.stem}.txt"
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, out_image)
        out_label.write_text(row["_converted_label"], encoding="utf-8")
        if sha256(out_image) != row["image_sha256"]:
            raise ValueError(f"Image copy hash check failed for {source_image.name}")

    (PACKAGE_ROOT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/validation\nnames:\n  0: product\n",
        encoding="utf-8",
    )
    checkpoint_out = PACKAGE_ROOT / "weights" / "sku110k-yolo11-n640.pt"
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, checkpoint_out)
    if sha256(checkpoint_out) != checkpoint_sha:
        raise ValueError("Checkpoint copy hash check failed")
    (PACKAGE_ROOT / "weights" / "sku110k-yolo11-n640.pt.sha256").write_text(
        f"{checkpoint_sha}  sku110k-yolo11-n640.pt\n", encoding="ascii"
    )

    test_image = test_image.expanduser().resolve(strict=True)
    if test_image.name.lower() != "img_4699.jpeg":
        raise ValueError(f"Expected the user-designated final-test image IMG_4699.jpeg, received {test_image.name}")
    test_out = PACKAGE_ROOT / "final_test" / "images" / "IMG_4699.jpeg"
    test_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test_image, test_out)
    count_truth = {
        "scope": "test-only; not included in data.yaml",
        "source_image": "IMG_4699.jpeg",
        "total_products": 6,
        "sku_counts": {
            "JANA-ANTIOX-500": 2,
            "FENTIMANS-GINGER-200": 2,
            "ENERGIA-POWER-250": 1,
            "DUKAT-MLIJEKO-1000": 1,
        },
        "box_annotations_available": False,
        "note": "The image has verified product counts only. Do not treat model proposal boxes in the confirmation JSON as ground-truth boxes.",
    }
    (PACKAGE_ROOT / "final_test" / "ground_truth_counts.json").write_text(json.dumps(count_truth, indent=2) + "\n", encoding="utf-8")
    (PACKAGE_ROOT / "final_test" / "manifest.json").write_text(json.dumps({
        "source_path_at_package_time": str(test_image),
        "image_sha256": sha256(test_out),
        "assignment": "untouched_final_test_only",
        "included_in_train_or_validation": False,
        "count_ground_truth": "final_test/ground_truth_counts.json",
        "bbox_ground_truth": None,
    }, indent=2) + "\n", encoding="utf-8")

    manifest_fields = ["split", "group_id", "record_type", "coco_filename", "original_filename", "image_path", "label_path", "width", "height", "instance_count", "source_polygon_sha256", "image_sha256"]
    with (PACKAGE_ROOT / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        for row in rows:
            source_label = Path(row["_source_label"])
            image_name = Path(row["_source_image"]).name
            writer.writerow({
                "split": row["split"],
                "group_id": row["group_id"],
                "record_type": row["record_type"],
                "coco_filename": row.get("coco_filename", ""),
                "original_filename": image_name,
                "image_path": f"images/{row['split']}/{image_name}",
                "label_path": f"labels/{row['split']}/{Path(image_name).stem}.txt",
                "width": row["_width"],
                "height": row["_height"],
                "instance_count": row["annotation_count"],
                "source_polygon_sha256": sha256(source_label),
                "image_sha256": row["image_sha256"],
            })

    for filename in ("train.py", "evaluate_detectors.py", "README.md", "requirements-runpod.txt"):
        shutil.copy2(TEMPLATES / filename, PACKAGE_ROOT / ("README_RUNPOD.md" if filename == "README.md" else filename))

    validation = {
        "status": "pass_for_experimental_detection_bundle",
        "classes": {"0": "product"},
        "images": sum(part["images"] for part in splits.values()),
        "instances": sum(part["instances"] for part in splits.values()),
        "splits": splits,
        "checks": [
            "source manifest image/label stem mapping and SHA-256",
            "source image decode and dimensions",
            "single class id 0",
            "normalized coordinate bounds",
            "polygon degeneracy and self-intersection",
            "exact duplicate polygon detection",
            "conversion from polygon coordinate extrema to normalized cx cy width height",
            "image/label instance counts",
            "no arrangement group leakage between train and validation",
            "final-test image excluded from train/validation",
        ],
        "known_review_flags": [
            {"image": "IMG_4571.jpeg", "items": "three tiny individual candy masks", "disposition": "retained; owner approved preserving these separate candies"},
            {"image": "IMG_4588.jpeg", "items": "one tiny wrapped-item mask", "disposition": "retained from owner-approved CVAT batch; small-object quality remains a visual caveat"},
            {"image": "IMG_4571.jpeg", "items": "one 3.6%-of-smaller-mask partial overlap", "disposition": "retained; prior reviewed overlap explicitly accepted by owner"},
            {"image": "all", "items": "semantic completeness and background leakage", "disposition": "cannot be fully established from geometry; source batch was human reviewed and approved"},
        ],
        "held_out_or_test_only_not_in_train_or_validation": EXCLUDED_DIAGNOSTIC_RUNS + ["IMG_4699.jpeg"],
        "training_started": False,
        "production_modified": False,
    }
    (PACKAGE_ROOT / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    (PACKAGE_ROOT / "review_notes.md").write_text(
        "# Label review notes\n\n"
        "No source annotations were repaired, removed, or overwritten. Detection boxes are bounding rectangles computed from each approved polygon.\n\n"
        "Retained review flags: IMG_4571 has three small individual candy instances and the previously reviewed partial overlap; IMG_4588 has one very small wrapped-item instance. The owner approved the 23-image CVAT batch and specifically requested that the candy instances and overlap be preserved. These are included, not silently discarded. The source geometry audit reported zero invalid polygons and zero duplicate-mask candidates.\n\n"
        "Automated checks do not prove scene completeness or semantic boundary accuracy. The CVAT batch received human visual review and owner approval; use fresh scene-grouped photographs for the final evaluation after this experiment.\n",
        encoding="utf-8",
    )

    package_files = sorted(path for path in PACKAGE_ROOT.rglob("*") if path.is_file())
    package_manifest = {
        "archive_name": ZIP_PATH.name,
        "classes": {"0": "product"},
        "training_and_validation": splits,
        "final_test_images": 1,
        "final_test_products": 6,
        "retail_checkpoint": {
            "path": "weights/sku110k-yolo11-n640.pt",
            "sha256": checkpoint_sha,
            "source": "https://huggingface.co/chistopat/sku110k-yolo11-object-detector",
        },
        "files": [path.relative_to(PACKAGE_ROOT).as_posix() for path in package_files] + ["package_manifest.json"],
    }
    (PACKAGE_ROOT / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")

    with zipfile.ZipFile(ZIP_PATH, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PACKAGE_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(PACKAGE_ROOT).as_posix())
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise zipfile.BadZipFile(f"Archive CRC check failed: {bad_member}")
        names = archive.namelist()
        if "data.yaml" not in names or "train.py" not in names or "weights/sku110k-yolo11-n640.pt" not in names:
            raise zipfile.BadZipFile("Required package file missing from archive")
    return {
        "zip_path": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": sha256(ZIP_PATH),
        "package_files": len(package_files) + 1,
        "training_and_validation": splits,
        "final_test_image_sha256": sha256(test_out),
        "checkpoint_sha256": checkpoint_sha,
        "archive_crc": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Retail SKU-110K YOLO11n PyTorch .pt")
    parser.add_argument("--final-test-image", type=Path, required=True, help="Original user-provided IMG_4699.jpeg")
    parser.add_argument("--preflight-only", action="store_true", help="Validate sources and size budget without writing the bundle")
    args = parser.parse_args()
    rows, splits = validate_sources()
    checkpoint_sha = validate_checkpoint(args.checkpoint)
    test_image = args.final_test_image.expanduser().resolve(strict=True)
    if test_image.name.lower() != "img_4699.jpeg":
        raise ValueError(f"Expected IMG_4699.jpeg, received {test_image.name}")
    payload = preflight_size(rows, args.checkpoint, test_image)
    preflight = {"source_status": "pass", "splits": splits, "expanded_payload_estimate_bytes": payload, "checkpoint_sha256": checkpoint_sha}
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return 0
    print(json.dumps(build_package(rows, splits, args.checkpoint, checkpoint_sha, test_image), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
