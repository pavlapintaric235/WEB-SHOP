"""Build a self-contained, read-only RunPod inference bundle from verified inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_INPUT = Path(r"C:\Users\U938\Downloads\InventoryLens_Codex_experiment_READY.zip")
DEFAULT_XLSX = Path(r"C:\Users\U938\Downloads\InventoryLens_experiment_review_READY.xlsx")
DEFAULT_OUTPUT = ROOT / "data" / "recognition_checks" / "runpod_rtx3090_inference_20261018"
DB_PATH = ROOT / "data" / "inventorylens.sqlite3"
ARCHIVE_METADATA = (
    "photos.csv", "scenes.csv", "scene_counts_DRAFT.csv", "products.csv",
    "review_required.csv", "duplicates.csv", "package_manifest.json",
    "README_START_HERE.md", "CODEX_PROMPT.md",
)
PROJECT_MODULES = (
    "images.py", "local_features.py", "model_assets.py", "sam_vision.py", "vision.py",
    "experiments/hybrid_pipeline/__init__.py",
    "experiments/hybrid_pipeline/detectors.py",
    "experiments/hybrid_pipeline/hybrid_core.py",
    "experiments/hybrid_pipeline/hf_encoders.py",
    "experiments/hybrid_pipeline/mask_runtime.py",
    "experiments/hybrid_pipeline/segmentation_models.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = list(image.tobytes())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(values[row * 9 + col] > values[row * 9 + col + 1])
    return bits


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_crosswalk(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {row["experiment_sku"]: row for row in csv.DictReader(stream)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archive, workbook, output = args.archive.resolve(strict=True), args.workbook.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle output: {output}")
    output.mkdir(parents=True)
    bundle = output / "inventorylens_runpod_inference"
    bundle.mkdir()

    source_dir = bundle / "data" / "source"
    image_root = source_dir / "images"
    with zipfile.ZipFile(archive) as source_zip:
        bad = source_zip.testzip()
        if bad:
            raise ValueError(f"Input archive CRC failure at {bad}")
        members = set(source_zip.namelist())
        photos_name = "photos.csv"
        photos_rows = list(csv.DictReader(source_zip.read(photos_name).decode("utf-8-sig").splitlines()))
        unique_rows = [row for row in photos_rows if row["role"] in {"scene", "reference"}]
        if len(unique_rows) != 41 or len({row["sha256"] for row in unique_rows}) != 41:
            raise ValueError("Expected exactly 41 unique scene/reference photos from photos.csv")
        photo_hashes = {row["original_filename"]: row["sha256"] for row in unique_rows}
        if len(photo_hashes) != 41:
            raise ValueError("The input photo manifest has duplicate original filenames")
        for row in unique_rows:
            relative = Path(row["relative_path"])
            if relative.is_absolute() or ".." in relative.parts or row["relative_path"] not in members:
                raise ValueError(f"Unsafe or missing archive path for {row['original_filename']}")
            destination = image_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source_zip.open(row["relative_path"]) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256(destination) != row["sha256"]:
                raise ValueError(f"SHA256 mismatch for source photograph {row['original_filename']}")
            with Image.open(destination) as image:
                image.verify()
        for name in ARCHIVE_METADATA:
            if name in members:
                (source_dir / name).write_bytes(source_zip.read(name))
        embedded = source_zip.read("InventoryLens_experiment_review.xlsx")
        if embedded != workbook.read_bytes():
            raise ValueError("The standalone review workbook differs from the workbook inside the source archive")
        copy_file(workbook, source_dir / "InventoryLens_experiment_review.xlsx")

    crosswalk = load_crosswalk(Path(__file__).with_name("catalog_crosswalk.csv"))
    scene_hashes = {row["sha256"] for row in unique_rows if row["role"] == "scene"}
    scene_dhashes = {
        row["original_filename"]: dhash(image_root / row["relative_path"])
        for row in unique_rows if row["role"] == "scene"
    }
    photo_rows = {row["original_filename"]: row for row in unique_rows}
    products_csv = list(csv.DictReader((source_dir / "products.csv").open(encoding="utf-8-sig", newline="")))

    # Query the current registry without importing app code or opening a writable DB connection.
    db_uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        catalog_rows = connection.execute(
            "SELECT p.sku, p.name, r.image_path FROM products p "
            "JOIN product_references r ON r.product_id=p.id "
            "WHERE p.business_id=1 ORDER BY p.sku,r.id"
        ).fetchall()
    finally:
        connection.close()

    catalog_dir = source_dir / "catalog_references"
    cache_target = bundle / "project" / "data" / "models" / "reference_cache"
    cache_target.mkdir(parents=True, exist_ok=True)
    try:
        from sam_vision import REFERENCE_CACHE_VERSION
    except Exception as error:
        raise RuntimeError(f"Could not load reference-cache version from project: {error}") from error
    cache_source_dir = ROOT / "data" / "models" / "reference_cache"

    catalog_refs: dict[str, list[dict]] = {}
    exact_duplicates, near_duplicates, missing_gallery = [], [], []
    for row in catalog_rows:
        sku, name, original = row["sku"], row["name"], Path(row["image_path"])
        if not original.is_file():
            missing_gallery.append({"sku": sku, "path": str(original), "reason": "missing_file"})
            continue
        digest = sha256(original)
        if digest in scene_hashes:
            exact_duplicates.append({"sku": sku, "path": original.name, "sha256": digest})
            continue
        fingerprint = hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii") + original.read_bytes()).hexdigest()
        cache_source = cache_source_dir / f"{fingerprint}.npz"
        if not cache_source.is_file():
            missing_gallery.append({"sku": sku, "path": original.name, "reason": "missing_mobile_sam_reference_cache"})
            continue
        reference_dhash = dhash(original)
        close = [filename for filename, value in scene_dhashes.items()
                 if (reference_dhash ^ value).bit_count() <= 4]
        if close:
            near_duplicates.append({"sku": sku, "path": original.name, "possible_scene_matches": close,
                                    "dhash_distance_max": 4})
            continue
        safe_name = f"{sku}__{original.name}"
        destination = catalog_dir / sku / safe_name
        copy_file(original, destination)
        copy_file(cache_source, cache_target / f"{fingerprint}.npz")
        catalog_refs.setdefault(sku, []).append({
            "image_path": f"data/source/catalog_references/{sku}/{safe_name}",
            "sha256": digest, "reference_source": "registered_catalog_gallery",
            "catalog_sku": sku, "name": name, "cache_ready": True,
        })

    # Copy project-local code and compact pretrained checkpoints. No database is bundled.
    project_root = bundle / "project"
    for relative in PROJECT_MODULES:
        copy_file(ROOT / relative, project_root / relative)
    copy_file(ROOT / "data" / "models" / "mobile_sam.pt", project_root / "data" / "models" / "mobile_sam.pt")
    copy_file(ROOT / "data" / "models" / "resnet18-f37072fd.pth", project_root / "data" / "models" / "resnet18-f37072fd.pth")
    copy_file(ROOT / "experiments" / "hybrid_pipeline" / "weights" / "mobilenet_v2-7ebf99e0.pth",
              project_root / "experiments" / "hybrid_pipeline" / "weights" / "mobilenet_v2-7ebf99e0.pth")
    retail_weights = ROOT / "data" / "segmentation_dataset" / "experiments" / "runpod_yolo11n_detection_20261017" / "package" / "weights" / "sku110k-yolo11-n640.pt"
    copy_file(retail_weights, project_root / "experiments" / "hybrid_pipeline" / "weights" / "sku110k-yolo11-n640.pt")

    source_reference_rows = [row for row in unique_rows if row["role"] == "reference"]
    source_refs_by_sku: dict[str, list[dict]] = {}
    for row in source_reference_rows:
        source_refs_by_sku.setdefault(row["reference_sku"], []).append({
            "image_path": f"data/source/images/{row['relative_path']}",
            "sha256": row["sha256"], "reference_source": "provided_experiment_reference",
            "view_label": row["view_label"], "cache_ready": False,
        })

    product_entries = []
    for row in products_csv:
        ilx = row["experiment_sku"]
        map_row = crosswalk[ilx]
        catalog_sku = map_row["catalog_sku"]
        provided_refs = source_refs_by_sku.get(ilx, [])
        registered_refs = catalog_refs.get(catalog_sku, []) if catalog_sku else []
        refs = provided_refs + registered_refs
        product_entries.append({
            "sku": ilx,
            "name": row["product_name"],
            "variant_size": row["variant_size"],
            "catalog_sku_candidate": catalog_sku or None,
            "crosswalk_status": map_row["crosswalk_status"],
            "references": refs,
            "reference_count": len(refs),
            "reference_notes": map_row["scoring_note"],
        })
    (source_dir / "product_registry_experiment.json").write_text(
        json.dumps(product_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    (source_dir / "reference_audit.json").write_text(json.dumps({
        "source_reference_count": len(source_reference_rows),
        "registered_reference_count_copied": sum(map(len, catalog_refs.values())),
        "registered_references_by_sku": {key: len(value) for key, value in sorted(catalog_refs.items())},
        "exact_scene_duplicates_excluded": exact_duplicates,
        "possible_near_scene_duplicates_excluded_for_review": near_duplicates,
        "unavailable_catalog_references": missing_gallery,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    for filename in ("verify_cuda.py", "prepare_models.py", "prepare_reference_cache.py", "run_inference.py", "package_downloads.py", "README_RUNPOD.md"):
        copy_file(Path(__file__).with_name(filename), bundle / filename)
    test_path = Path(__file__).parent / "tests" / "test_runpod_inference.py"
    if test_path.is_file():
        copy_file(test_path, bundle / "tests" / test_path.name)
    copy_file(Path(__file__).with_name("requirements-runpod.txt"), bundle / "requirements-runpod.txt")
    copy_file(Path(__file__).with_name("catalog_crosswalk.csv"), source_dir / "catalog_crosswalk.csv")

    bundle_manifest = {
        "package": "InventoryLens RTX 3090 read-only inference experiment",
        "built_from_archive": str(archive), "archive_sha256": sha256(archive),
        "review_workbook_sha256": sha256(workbook),
        "source_photo_count": len(unique_rows),
        "scene_photo_count": sum(row["role"] == "scene" for row in unique_rows),
        "provided_reference_photo_count": len(source_reference_rows),
        "scene_groups": len({row["scene_id"] for row in unique_rows if row["role"] == "scene"}),
        "scene_photo_paths": [row["relative_path"] for row in unique_rows if row["role"] == "scene"],
        "provided_reference_paths": [row["relative_path"] for row in source_reference_rows],
        "model_sha256": {
            "mobile_sam.pt": sha256(project_root / "data" / "models" / "mobile_sam.pt"),
            "resnet18-f37072fd.pth": sha256(project_root / "data" / "models" / "resnet18-f37072fd.pth"),
            "mobilenet_v2-7ebf99e0.pth": sha256(project_root / "experiments" / "hybrid_pipeline" / "weights" / "mobilenet_v2-7ebf99e0.pth"),
            "sku110k-yolo11-n640.pt": sha256(project_root / "experiments" / "hybrid_pipeline" / "weights" / "sku110k-yolo11-n640.pt"),
        },
        "no_database_included": True,
        "validation_limits": [
            "No object-level boxes or polygons are supplied for these scenes; do not report detection precision, recall or mask IoU.",
            "Only user-confirmed scene totals and explicitly confirmed SKU counts are eligible for scoring.",
            "Candidate-only catalogue crosswalks are not verified registrations.",
            "Same-arrangement views must not be summed; group summaries use non-additive evidence only.",
        ],
    }
    (bundle / "bundle_manifest.json").write_text(json.dumps(bundle_manifest, indent=2), encoding="utf-8")

    zip_path = output / "inventorylens_runpod_inference.zip"
    # Photo, checkpoint, and NPZ payloads are already compressed; recompressing costs time and gains little.
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle))
    with zipfile.ZipFile(zip_path) as zf:
        if zf.testzip():
            raise RuntimeError("Generated upload ZIP failed CRC verification")
    print(json.dumps({"bundle_dir": str(bundle), "zip": str(zip_path),
                      "zip_bytes": zip_path.stat().st_size, "manifest": bundle_manifest,
                      "reference_audit": str(source_dir / "reference_audit.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
