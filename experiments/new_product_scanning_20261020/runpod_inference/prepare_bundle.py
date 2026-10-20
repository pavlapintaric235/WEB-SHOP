#!/usr/bin/env python3
"""Build a hash-verified, experiment-only RunPod inference input archive."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from experiment_io import load_manifest, latest_mask_events, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-name", default="runpod_scene_inference_input_20261020_v3.zip")
    args = parser.parse_args()
    if Path(args.archive_name).name != args.archive_name or not args.archive_name.endswith(".zip"):
        raise ValueError("--archive-name must be a ZIP basename within this experiment's results directory")
    output_dir = HERE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / args.archive_name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    manifest_rows = load_manifest()
    scenes = [row for row in manifest_rows if row["group"] in {"employee", "customer"}]
    if len(scenes) != 30:
        raise RuntimeError(f"Expected 30 employee/customer scenes, found {len(scenes)}")
    scene_records = []
    for row in scenes:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != row["sha256"]:
            raise RuntimeError(f"Scene photo missing or changed: {row['filename']}")
        scene_records.append({
            "photo_filename": row["filename"],
            "dataset_type": row["group"],
            "source_sha256": row["sha256"],
            "display_width": int(row["display_width"]),
            "display_height": int(row["display_height"]),
            "archive_path": f"input/scenes/{row['filename']}",
        })

    with (HERE / "audit" / "candidate_skus.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        sku_rows = {row["sku_id"]: row for row in csv.DictReader(stream)}
    latest = latest_mask_events()
    reference_records = []
    for filename, event in sorted(latest.items()):
        if event.get("decision") != "approved":
            continue
        sku_id = event["sku_id"]
        box_rel = event.get("box_crop_file")
        masked_rel = event.get("masked_crop_file")
        if sku_id not in sku_rows or not box_rel or not masked_rel:
            raise RuntimeError(f"Approved reference is missing an SKU or paired crops: {filename}")
        box_path, masked_path = HERE / box_rel, HERE / masked_rel
        if not box_path.is_file() or not masked_path.is_file():
            raise RuntimeError(f"Approved paired reference crops are missing: {filename}")
        reference_records.append({
            "sku_id": sku_id,
            "proposed_name": sku_rows[sku_id]["proposed_name"],
            "source_photo_filename": filename,
            "source_sha256": event["source_sha256"],
            "box_sha256": sha256_file(box_path),
            "masked_sha256": sha256_file(masked_path),
            "box_archive_path": f"input/references/{sku_id}/{filename.rsplit('.', 1)[0]}_box.jpg",
            "masked_archive_path": f"input/references/{sku_id}/{filename.rsplit('.', 1)[0]}_masked.jpg",
        })
    if len(reference_records) != 34 or len({row["sku_id"] for row in reference_records}) != 10:
        raise RuntimeError(f"Unexpected approved gallery: {len(reference_records)} crops / "
                           f"{len({row['sku_id'] for row in reference_records})} SKUs")

    registry = []
    for sku_id, row in sorted(sku_rows.items()):
        registry.append({
            "sku_id": sku_id,
            "proposed_name": row["proposed_name"],
            "brand": row["brand"],
            "variant_or_size": row["variant_or_size"],
            "identity_status": row["identity_status"],
            "owner_confirmation": row["owner_confirmation"],
            "status": "provisional_candidate_not_verified_catalogue_identity",
        })

    records = {
        "experiment": "new_product_scanning_20261020",
        "scope": "GPU inference only; no training and no use of scene labels as references",
        "scenes": scene_records,
        "reference_gallery": reference_records,
        "candidate_skus": registry,
    }

    builder_source = Path(__file__).resolve()
    runner_source = HERE / "runpod_inference" / "run_scene_inference.py"
    runner_shell = HERE / "runpod_inference" / "run_all.sh"
    readme_source = HERE / "runpod_inference" / "README_RUNPOD.md"
    aws_readme_source = HERE / "runpod_inference" / "README_AWS.md"
    files = [(builder_source, "scripts/prepare_bundle.py"),
             (runner_source, "scripts/run_scene_inference.py"),
             (runner_shell, "run_all.sh"),
             (readme_source, "README_RUNPOD.md")]
    if args.archive_name.startswith("aws_"):
        files.append((aws_readme_source, "README_AWS.md"))
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("input/manifest.json", json.dumps(records, indent=2) + "\n")
        for source, arcname in files:
            archive.write(source, arcname)
        for row, source_row in zip(scene_records, scenes):
            archive.write(Path(source_row["path"]), row["archive_path"])
        for record in reference_records:
            event = latest[record["source_photo_filename"]]
            for representation in ("box", "masked"):
                crop_path = HERE / event[f"{representation}_crop_file"]
                if sha256_file(crop_path) != record[f"{representation}_sha256"]:
                    raise RuntimeError(f"Reference crop changed while packaging: {crop_path.name}")
                archive.write(crop_path, record[f"{representation}_archive_path"])

    digest = sha256_file(output)
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP CRC failed for {bad_member}")
        members = archive.namelist()
    print(json.dumps({
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": digest,
        "zip_crc": "PASS",
        "scene_images": len(scene_records),
        "employee_images": sum(row["dataset_type"] == "employee" for row in scene_records),
        "customer_images": sum(row["dataset_type"] == "customer" for row in scene_records),
        "approved_paired_references": len(reference_records),
        "reference_crop_files": 2 * len(reference_records),
        "candidate_skus": len(registry),
        "members": len(members),
        "includes_model_weights": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
