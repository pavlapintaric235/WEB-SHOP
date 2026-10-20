"""Create a proposal-only grouped train/validation manifest; never copies data."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "data/segmentation_dataset/proposals/cvat22_yolo_staging_20261016"
MAPPING = STAGING / "source_mapping.csv"
APPROVED = ROOT / "data/segmentation_dataset/approved"
OUTPUT = STAGING / "proposed_split.csv"
SUMMARY = STAGING / "proposed_split_summary.json"

VALIDATION_IMAGES = {"IMG_4588.jpeg", "IMG_4634.jpeg", "IMG_4599.jpeg", "IMG_4607.jpeg", "IMG_4596.jpeg"}
PAIRED_GROUPS = {
    "IMG_4597.jpeg": "arrangement_pair_IMG_4597_IMG_4614",
    "IMG_4614.jpeg": "arrangement_pair_IMG_4597_IMG_4614",
    "IMG_4661.jpeg": "arrangement_pair_IMG_4661_IMG_4664",
    "IMG_4664.jpeg": "arrangement_pair_IMG_4661_IMG_4664",
}


def label_stats(path: Path) -> tuple[int, set[str]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows), {row[0] for row in rows}


def main() -> None:
    rows: list[dict[str, str | int]] = []
    with MAPPING.open(encoding="utf-8-sig", newline="") as source:
        mapped = list(csv.DictReader(source))
    if len(mapped) != 23 or sum(int(item["annotation_count"]) for item in mapped) != 146:
        raise SystemExit("Expected the verified 23-image / 146-instance mapping; refusing to write proposal.")

    mapped_names = {item["coco_filename"] for item in mapped}
    if not VALIDATION_IMAGES <= mapped_names:
        raise SystemExit("A proposed validation photograph is missing from the verified mapping.")

    for item in mapped:
        filename = item["coco_filename"]
        label = STAGING / "labels/train" / item["label_filename"]
        image = Path(item["verified_original_path"])
        count, classes = label_stats(label)
        if not image.is_file() or count != int(item["annotation_count"]) or classes != {"0"}:
            raise SystemExit(f"Mapping/label check failed for {filename}.")
        split = "validation" if filename in VALIDATION_IMAGES else "train"
        group = PAIRED_GROUPS.get(filename, f"scene_{Path(filename).stem}")
        rows.append({
            "split": split,
            "group_id": group,
            "record_type": "new_review_batch",
            "image_filename": filename,
            "annotation_count": count,
            "source_image_path": str(image),
            "yolo_label_path": str(label),
        })

    seed_rows = []
    for image_name in ("image1.png", "image2.png", "image3.png"):
        image = APPROVED / "images" / image_name
        label = APPROVED / "labels" / f"{Path(image_name).stem}.txt"
        count, classes = label_stats(label)
        if not image.is_file() or classes != {"0"}:
            raise SystemExit(f"Approved seed check failed for {image_name}.")
        seed_rows.append({
            "split": "train",
            "group_id": "approved_seed_batch_01",
            "record_type": "approved_seed",
            "image_filename": image_name,
            "annotation_count": count,
            "source_image_path": str(image),
            "yolo_label_path": str(label),
        })
    if sum(int(row["annotation_count"]) for row in seed_rows) != 17:
        raise SystemExit("Approved seed labels do not total the expected 17 instances.")
    rows.extend(seed_rows)

    fields = ["split", "group_id", "record_type", "image_filename", "annotation_count", "source_image_path", "yolo_label_path"]
    with OUTPUT.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["split"]), str(row["group_id"]), str(row["image_filename"]))))

    counts = {split: {"images": 0, "instances": 0} for split in ("train", "validation")}
    for row in rows:
        counts[str(row["split"])]["images"] += 1
        counts[str(row["split"])]["instances"] += int(row["annotation_count"])
    summary = {
        "status": "proposal_only_waiting_for_final_human_dataset_approval",
        "manifest": str(OUTPUT),
        "method": "scene/arrangement grouped; no images or labels copied",
        "total": {"images": len(rows), "instances": sum(int(row["annotation_count"]) for row in rows)},
        "splits": counts,
        "validation_fraction_images": counts["validation"]["images"] / len(rows),
        "validation_fraction_instances": counts["validation"]["instances"] / sum(int(row["annotation_count"]) for row in rows),
        "groups_kept_together": {
            "approved_seed_batch_01": ["image1.png", "image2.png", "image3.png"],
            "arrangement_pair_IMG_4597_IMG_4614": ["IMG_4597.jpeg", "IMG_4614.jpeg"],
            "arrangement_pair_IMG_4661_IMG_4664": ["IMG_4661.jpeg", "IMG_4664.jpeg"],
        },
        "validation_images": sorted(VALIDATION_IMAGES),
        "final_test_set": "none; collect fresh photographs after development",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
