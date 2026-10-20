"""Materialize a separate RunPod YOLO11-seg bundle from the reviewed polygon dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/segmentation_dataset/experiments/initial_yolo26n_seg_20261017"
TEMPLATES = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/segmentation_dataset/experiments/runpod_yolo11n_seg_20261017"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    with (SOURCE / "dataset_manifest.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def preflight(rows: list[dict[str, str]]) -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment artifacts: {OUTPUT}")
    total_bytes = 0
    for row in rows:
        for relative, expected in ((row["image_path"], row["image_sha256"]),
                                   (row["label_path"], row["label_sha256"])):
            path = (SOURCE / relative).resolve(strict=True)
            if SOURCE.resolve() not in path.parents:
                raise ValueError(f"Source manifest escaped the dataset root: {relative}")
            if sha256(path) != expected:
                raise ValueError(f"Source checksum mismatch: {path}")
            total_bytes += path.stat().st_size
    required = total_bytes * 2 + 512 * 1024**2
    free = shutil.disk_usage(ROOT).free
    if free < required:
        raise OSError(f"Insufficient space before package creation: need {required} bytes, have {free}")
    return total_bytes


def build() -> Path:
    rows = read_manifest()
    if len(rows) != 26 or sum(int(row["annotation_count"]) for row in rows) != 163:
        raise ValueError("Reviewed source must contain exactly 26 images and 163 product polygons")
    if any(row["original_filename"].lower() == "img_4699.jpeg" for row in rows):
        raise ValueError("Protected IMG_4699 appears in the training manifest")
    source_bytes = preflight(rows)
    package = OUTPUT / "package"
    dataset = package / "dataset"
    dataset.mkdir(parents=True)

    copied = []
    for row in rows:
        image_source = SOURCE / row["image_path"]
        label_source = SOURCE / row["label_path"]
        image_target = dataset / row["image_path"]
        label_target = dataset / row["label_path"]
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_source, image_target)
        shutil.copy2(label_source, label_target)
        if sha256(image_target) != row["image_sha256"] or sha256(label_target) != row["label_sha256"]:
            raise ValueError(f"Copied byte verification failed: {image_target.name}")
        with Image.open(image_target) as opened:
            opened.verify()
        copied.append({"filename": row["original_filename"], "split": row["split"],
                       "group_id": row["group_id"], "instances": int(row["annotation_count"]),
                       "image_sha256": row["image_sha256"], "label_sha256": row["label_sha256"]})

    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/validation\nnames:\n  0: product\n",
        encoding="utf-8",
    )
    shutil.copy2(SOURCE / "dataset_manifest.csv", dataset / "dataset_manifest.csv")
    for filename in ("train_yolo11_seg.py", "evaluate_segmentation.py", "requirements-runpod.txt", "README_RUNPOD.md"):
        shutil.copy2(TEMPLATES / filename, package / filename)

    groups = {split: len({row["group_id"] for row in rows if row["split"] == split})
              for split in ("train", "validation")}
    split_totals = {split: {"images": sum(row["split"] == split for row in rows),
                            "instances": sum(int(row["annotation_count"]) for row in rows if row["split"] == split),
                            "groups": groups[split]}
                    for split in ("train", "validation")}
    if split_totals != {"train": {"images": 21, "instances": 130, "groups": 17},
                        "validation": {"images": 5, "instances": 33, "groups": 5}}:
        raise ValueError(f"Scene-group split mismatch: {split_totals}")
    report = {
        "status": "pass_training_bundle_prepared_training_not_started",
        "source_dataset": str(SOURCE), "images": len(rows), "instances": 163,
        "classes": {"0": "product"}, "splits": split_totals,
        "copied_source_bytes": source_bytes,
        "original_polygon_label_bytes_preserved": True,
        "excluded": ["all five October 17 development diagnostic photos", "IMG_4699.jpeg",
                     "all other previously designated held-out photographs"],
        "checksum_verification": "all copied source image and label SHA-256 values match dataset manifest",
        "training_started": False,
        "production_database_or_code_modified": False,
        "limitations": [
            "Only 26 photographs across 22 groups; validation has five groups and is not a reliable generalization estimate.",
            "The package has no checkpoint: RunPod must download pretrained yolo11n-seg.pt; box-only checkpoints are invalid.",
            "Fine-tuning is a segmentation-specific experiment, not a repeat of detection-only training.",
        ],
    }
    (package / "dataset_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (package / "package_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump({"archive_name": "inventorylens_yolo11_seg_training.zip", "files": copied,
                   "split_totals": split_totals, "includes_protected_evaluation_photos": False},
                  stream, indent=2)
        stream.write("\n")

    archive = OUTPUT / "inventorylens_yolo11_seg_training.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zf.write(path, Path("inventorylens_yolo11_seg") / path.relative_to(package))
    with zipfile.ZipFile(archive) as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise ValueError(f"ZIP CRC validation failed at {bad_member}")
        if len(zf.namelist()) < 60:
            raise ValueError("Unexpectedly incomplete training archive")
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(build())
