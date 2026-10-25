"""Audit the first-two-set training pool and reserve the newest scenes for test.

This script reads source manifests and verified grocery labels. It writes only
small audit files next to itself; it never copies photographs or trains a model.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARTS = (
    ROOT / "01_early_k18_wow_coconut_and_grocery_history_a/manifest_this_part.csv",
    ROOT / "02_grocery_history_b_and_12_product_benchmark/manifest_this_part.csv",
)
GROCERY = ROOT / "data/segmentation_dataset/experiments/initial_yolo26n_seg_20261017"
NEWEST = ROOT / "experiments/new_product_scanning_20261020/audit/photo_manifest.csv"
EARLY_ARCHIVE = ROOT / "data/archives/20261014-223956-pre-grocery/test_runs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    first_two = [row for path in PARTS for row in read_csv(path)]
    scenes: dict[str, list[dict[str, str]]] = defaultdict(list)
    references: set[str] = set()
    for row in first_two:
        if row["role"] == "scene":
            scenes[row["sha256"]].append(row)
        elif row["role"] == "reference":
            references.add(row["sha256"])
        else:
            raise ValueError(f"Unexpected source role: {row['role']}")

    approved = read_csv(GROCERY / "dataset_manifest.csv")
    approved_by_hash: dict[str, dict[str, str]] = {}
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    for row in approved:
        image = GROCERY / row["image_path"]
        label = GROCERY / row["label_path"]
        if sha256(image) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError(f"Approved image or label checksum changed: {image.name}")
        if len([line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]) != int(row["annotation_count"]):
            raise ValueError(f"Approved annotation count changed: {image.name}")
        if row["image_sha256"] in approved_by_hash:
            raise ValueError(f"Duplicate approved photo hash: {image.name}")
        approved_by_hash[row["image_sha256"]] = row
        (train_groups if row["split"] == "train" else validation_groups).add(row["group_id"])
    if train_groups & validation_groups:
        raise ValueError("A grocery arrangement crosses train and validation")
    if not approved_by_hash.keys() <= scenes.keys():
        raise ValueError("An approved grocery image is absent from the first two sets")

    early = {digest: rows for digest, rows in scenes.items()
             if any(row["source_group"] == "early3_scene" for row in rows)}
    for digest, rows in early.items():
        found = False
        for row in rows:
            source = EARLY_ARCHIVE / row["original_filename"]
            if source.is_file() and sha256(source) == digest:
                found = True
                break
        if not found:
            raise ValueError(f"No byte-verified early scene for {digest}")

    newest = read_csv(NEWEST)
    newest_scenes = [row for row in newest if row["group"] in {"employee", "customer"}]
    newest_refs = [row for row in newest if row["group"] == "owner"]
    newest_hashes = {row["sha256"] for row in newest_scenes}
    if len(newest_hashes) != len(newest_scenes):
        raise ValueError("A newest scene has duplicate bytes")
    if newest_hashes & (set(scenes) | references):
        raise ValueError("Newest scene bytes overlap first-two training/reference data")
    if {row["sha256"] for row in newest_refs} & set(scenes):
        raise ValueError("A newest owner reference overlaps first-two scenes")

    records: list[dict[str, str | int]] = []
    for digest, rows in sorted(scenes.items()):
        approved_row = approved_by_hash.get(digest)
        records.append({
            "partition": approved_row["split"] if approved_row else "candidate_train_pending_labels",
            "source_set": "first_two",
            "sha256": digest,
            "source_groups": ";".join(sorted({row["source_group"] for row in rows})),
            "source_names": ";".join(sorted({row["original_filename"] for row in rows})),
            "arrangement_group": approved_row["group_id"] if approved_row else "unconfirmed",
            "label_status": "reviewed_instance_polygons" if approved_row else "no_approved_instance_labels",
            "instance_count": int(approved_row["annotation_count"]) if approved_row else "",
        })
    for row in newest_scenes:
        records.append({
            "partition": "newest_evaluation_pending_human_labels",
            "source_set": row["group"],
            "sha256": row["sha256"],
            "source_groups": row["group"],
            "source_names": row["filename"],
            "arrangement_group": "unconfirmed",
            "label_status": "no_approved_instance_labels",
            "instance_count": "",
        })

    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "status": "data_audit_only_no_training",
        "first_two_source_paths": len(first_two),
        "first_two_scene_paths": sum(row["role"] == "scene" for row in first_two),
        "first_two_distinct_scenes": len(scenes),
        "first_two_distinct_references": len(references),
        "approved_grocery_scene_images": len(approved),
        "approved_grocery_instances": sum(int(row["annotation_count"]) for row in approved),
        "train_images": sum(row["split"] == "train" for row in approved),
        "validation_images": sum(row["split"] == "validation" for row in approved),
        "first_two_scenes_needing_instance_labels": len(scenes) - len(approved),
        "early_distinct_scenes_needing_instance_labels": len(early),
        "newest_owner_references_reserved_for_gallery": len(newest_refs),
        "newest_scenes_reserved_for_evaluation": len(newest_scenes),
        "newest_employee_scenes": sum(row["group"] == "employee" for row in newest_scenes),
        "newest_customer_scenes": sum(row["group"] == "customer" for row in newest_scenes),
        "scene_hash_overlap_between_training_pool_and_newest": 0,
        "approved_grocery_group_leakage": False,
    }
    (HERE / "readiness.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
