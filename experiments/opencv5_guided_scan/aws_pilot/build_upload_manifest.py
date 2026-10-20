"""Freeze the approved original-photo set without copying or changing images."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PILOT_ROOT = Path(__file__).resolve().parent
OUTPUT = PILOT_ROOT / "photo_upload_manifest_20261020.csv"
RUN_ID = "inventorylens_20261020"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
EARLY_ROOT = PROJECT_ROOT / "data/archives/20261014-223956-pre-grocery"
GROCERY_CSV = (
    PROJECT_ROOT
    / "data/recognition_checks/final_comparative_gpu_benchmark_20261018"
    / "inventorylens_final_comparative_gpu_benchmark/data/source/photos.csv"
)
NEW_CSV = PROJECT_ROOT / "experiments/new_product_scanning_20261020/audit/photo_manifest.csv"
FIELDS = [
    "source_group", "role", "source_path", "original_filename", "bytes", "sha256",
    "manifest_sha256", "data_partition", "upload", "s3_key", "duplicate_of",
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def candidates() -> list[dict]:
    rows = []
    for path in sorted((EARLY_ROOT / "references").glob("TEST-*.png")):
        rows.append(dict(source_group="early3_ref", role="reference", path=path,
                         expected="", partition="experiment_only"))
    for path in sorted((EARLY_ROOT / "references/1").glob("TEST-*.png")):
        rows.append(dict(source_group="early3_ref", role="reference", path=path,
                         expected="", partition="experiment_only"))
    for path in sorted((EARLY_ROOT / "test_runs").glob("*-test-original.png")):
        rows.append(dict(source_group="early3_scene", role="scene", path=path,
                         expected="", partition="experiment_only"))
    for path in sorted((PROJECT_ROOT / "data/test_runs").glob("*-original.png")):
        rows.append(dict(source_group="grocery_historical", role="scene", path=path,
                         expected="", partition="experiment_only"))
    grocery_root = GROCERY_CSV.parent / "images"
    for row in csv_rows(GROCERY_CSV):
        rows.append(dict(source_group="grocery_benchmark", role=row["role"],
                         path=grocery_root / row["relative_path"],
                         expected=row["sha256"], partition=row["data_partition"]))
    for row in csv_rows(NEW_CSV):
        rows.append(dict(source_group=f"new10_{row['group']}",
                         role="reference" if row["group"] == "owner" else "scene",
                         path=Path(row["path"]), expected=row["sha256"],
                         partition="experiment_only"))
    return rows


def build() -> list[dict[str, str | int]]:
    first_by_hash: dict[str, str] = {}
    key_to_hash: dict[str, str] = {}
    rows = []
    for candidate in candidates():
        path = candidate["path"].resolve(strict=True)
        data = path.read_bytes()
        size = len(data)
        if size > MAX_IMAGE_BYTES:
            raise ValueError(f"Photo exceeds {MAX_IMAGE_BYTES} bytes: {path}")
        digest = hashlib.sha256(data).hexdigest()
        expected = candidate["expected"]
        if expected and digest != expected:
            raise ValueError(f"Source SHA-256 differs from source manifest: {path}")
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"Unsupported source extension: {path}")
        if digest in first_by_hash:
            key = first_by_hash[digest]
            upload = "no"
            duplicate_of = key
        else:
            image_id = f"{candidate['source_group']}-{digest[:20]}"
            key = f"incoming/{RUN_ID}/{image_id}.{suffix}"
            if key in key_to_hash and key_to_hash[key] != digest:
                raise ValueError(f"S3 key collision: {key}")
            first_by_hash[digest] = key
            key_to_hash[key] = digest
            upload = "yes"
            duplicate_of = ""
        rows.append({
            "source_group": candidate["source_group"],
            "role": candidate["role"],
            "source_path": str(path),
            "original_filename": path.name,
            "bytes": size,
            "sha256": digest,
            "manifest_sha256": expected,
            "data_partition": candidate["partition"],
            "upload": upload,
            "s3_key": key,
            "duplicate_of": duplicate_of,
        })
    return rows


def main() -> None:
    rows = build()
    with OUTPUT.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    unique = [row for row in rows if row["upload"] == "yes"]
    counts = Counter(row["source_group"] for row in rows)
    print(f"Wrote {OUTPUT}: {len(rows)} source paths, {len(unique)} unique uploads")
    print(f"Unique upload bytes: {sum(row['bytes'] for row in unique)}")
    print(f"Source groups: {dict(counts)}")


if __name__ == "__main__":
    main()
