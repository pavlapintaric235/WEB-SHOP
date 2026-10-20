"""Materialize the approved grouped experiment set without modifying source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "data/segmentation_dataset/proposals/cvat22_yolo_staging_20261016"
SPLIT_CSV = PROPOSAL / "proposed_split.csv"
SOURCE_MAPPING = PROPOSAL / "source_mapping.csv"
DEFAULT_OUTPUT = ROOT / "data/segmentation_dataset/experiments/initial_yolo26n_seg_20261017"
EXPECTED = {"train": (21, 130), "validation": (5, 33)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Expected an original PNG image: {path}")
    return struct.unpack(">II", header[16:24])


def read_labels(path: Path) -> tuple[int, list[str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_no, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) < 7 or fields[0] != "0" or (len(fields) - 1) % 2:
            raise ValueError(f"Malformed/class-mismatched YOLO segmentation row {path}:{line_no}")
        coords = [float(value) for value in fields[1:]]
        if any(not 0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"Non-normalized/out-of-bounds polygon at {path}:{line_no}")
    return len(lines), lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    experiments_root = (ROOT / "data/segmentation_dataset/experiments").resolve()
    if experiments_root not in output.parents:
        raise SystemExit(f"Output must be a new child of {experiments_root}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite an existing path: {output}")

    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as stream:
        split_rows = list(csv.DictReader(stream))
    if len(split_rows) != 26:
        raise SystemExit(f"Expected 26 approved split rows, got {len(split_rows)}")
    with SOURCE_MAPPING.open(encoding="utf-8-sig", newline="") as stream:
        mapping_rows = {row["coco_filename"]: row for row in csv.DictReader(stream)}
    if len(mapping_rows) != 23:
        raise SystemExit("Expected exactly 23 verified new-image mappings")

    planned: list[dict[str, object]] = []
    split_totals = {name: {"images": 0, "instances": 0} for name in EXPECTED}
    source_names: set[str] = set()
    byte_budget = 0
    for row in split_rows:
        split = row["split"]
        if split not in EXPECTED:
            raise SystemExit(f"Unexpected split: {split}")
        source_image = Path(row["source_image_path"]).resolve(strict=True)
        source_label = Path(row["yolo_label_path"]).resolve(strict=True)
        if source_image.suffix.lower() != ".png":
            raise SystemExit(f"Unexpected original image format: {source_image}")
        if source_image.stem != source_label.stem:
            raise SystemExit(f"Image/label stem mismatch: {source_image.name} / {source_label.name}")
        if source_image.name in source_names:
            raise SystemExit(f"Output filename collision: {source_image.name}")
        source_names.add(source_image.name)
        image_count, label_lines = read_labels(source_label)
        expected_instances = int(row["annotation_count"])
        if image_count != expected_instances:
            raise SystemExit(f"Annotation count mismatch for {source_image.name}")
        width, height = png_dimensions(source_image)

        coco_filename = row["image_filename"] if row["record_type"] == "new_review_batch" else ""
        if coco_filename:
            mapping = mapping_rows.get(coco_filename)
            if not mapping:
                raise SystemExit(f"No source mapping for {coco_filename}")
            if Path(mapping["verified_original_path"]).resolve() != source_image:
                raise SystemExit(f"Verified original mismatch for {coco_filename}")
            if sha256(source_image) != mapping["original_sha256"]:
                raise SystemExit(f"Source image hash mismatch for {coco_filename}")
            if (width, height) != (int(mapping["width"]), int(mapping["height"])):
                raise SystemExit(f"Source dimensions mismatch for {coco_filename}")

        split_totals[split]["images"] += 1
        split_totals[split]["instances"] += image_count
        byte_budget += source_image.stat().st_size + source_label.stat().st_size
        planned.append({
            "split": split,
            "group_id": row["group_id"],
            "record_type": row["record_type"],
            "coco_filename": coco_filename,
            "source_image": source_image,
            "source_label": source_label,
            "width": width,
            "height": height,
            "annotation_count": image_count,
            "label_lines": label_lines,
        })

    expected_nested = {name: {"images": values[0], "instances": values[1]} for name, values in EXPECTED.items()}
    if split_totals != expected_nested:
        raise SystemExit(f"Split totals mismatch: {split_totals}")
    free_bytes = shutil.disk_usage(ROOT).free
    reserve_bytes = 2 * 1024**3
    if free_bytes < byte_budget + reserve_bytes:
        raise SystemExit(
            f"Insufficient free space: need {byte_budget + reserve_bytes:,} bytes including 2 GiB reserve; "
            f"have {free_bytes:,}. No files copied."
        )

    output.mkdir(parents=True)
    manifest_fields = [
        "split", "group_id", "record_type", "coco_filename", "original_filename", "image_path", "label_path",
        "width", "height", "annotation_count", "image_sha256", "label_sha256", "image_bytes", "label_bytes",
    ]
    manifest_rows: list[dict[str, object]] = []
    for item in planned:
        split = str(item["split"])
        source_image = item["source_image"]
        source_label = item["source_label"]
        assert isinstance(source_image, Path) and isinstance(source_label, Path)
        image_out = output / "images" / split / source_image.name
        label_out = output / "labels" / split / f"{source_image.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_out)
        shutil.copy2(source_label, label_out)
        if sha256(image_out) != sha256(source_image) or sha256(label_out) != sha256(source_label):
            raise SystemExit(f"Copy verification failed for {source_image.name}")
        manifest_rows.append({
            "split": split,
            "group_id": item["group_id"],
            "record_type": item["record_type"],
            "coco_filename": item["coco_filename"],
            "original_filename": source_image.name,
            "image_path": image_out.relative_to(output).as_posix(),
            "label_path": label_out.relative_to(output).as_posix(),
            "width": item["width"],
            "height": item["height"],
            "annotation_count": item["annotation_count"],
            "image_sha256": sha256(image_out),
            "label_sha256": sha256(label_out),
            "image_bytes": image_out.stat().st_size,
            "label_bytes": label_out.stat().st_size,
        })

    with (output / "dataset_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    data_yaml = (
        f'path: "{output.as_posix()}"\n'
        "train: images/train\n"
        "val: images/validation\n"
        "names:\n"
        "  0: product\n"
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    report = {
        "status": "materialized_for_approved_initial_experiment; training_not_authorized_or_started",
        "dataset_root": str(output),
        "source_split_manifest": str(SPLIT_CSV),
        "images": sum(value["images"] for value in split_totals.values()),
        "instances": sum(value["instances"] for value in split_totals.values()),
        "classes": {"0": "product"},
        "splits": split_totals,
        "copied_photo_and_label_bytes": byte_budget,
        "source_images_modified": False,
        "sqlite_or_application_data_modified": False,
    }
    (output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "Self-contained Web&Shop initial instance-segmentation experiment dataset.\n"
        "All images are byte-for-byte copies of verified originals; labels are copied to matching stems.\n"
        "Training and validation are grouped as recorded in dataset_manifest.csv.\n"
        "This dataset is not an untouched final test set. No model training was started.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
