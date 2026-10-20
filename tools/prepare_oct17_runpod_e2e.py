"""Build a small, hash-verified RunPod bundle for the five-photo Oct 17 diagnostic."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = [
    "20261017-014015-b2a337ec",
    "20261017-014239-235d7e51",
    "20261017-014423-e7b3fb42",
    "20261017-014554-b9412539",
    "20261017-014722-cc4ab8d3",
]
REMOTE_ROOT = "/workspace/webshop_benchmark"
REMOTE_INPUT = f"{REMOTE_ROOT}/runpod_e2e_20261019"
OUTPUT = ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017/runpod_e2e_20261019"
PACKAGE_ROOT = ROOT / "data/recognition_checks/final_comparative_gpu_benchmark_20261018/inventorylens_final_comparative_gpu_benchmark"
SOURCE_ZIP = Path.home() / "Downloads/inventory6.zip"
HISTORY = ROOT / "data/test_runs.jsonl"
ORIGINALS = ROOT / "data/test_runs"
VALIDATION = ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017/validation_inventory6_20261017/validation_report.json"
EVALUATOR = ROOT / "tools/evaluate_oct17_benchmark_runpod.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic package: {OUTPUT}")
    for path in (SOURCE_ZIP, HISTORY, VALIDATION, EVALUATOR):
        if not path.is_file():
            raise FileNotFoundError(path)

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        manifests = [name for name in archive.namelist()
                     if Path(name).name == "instances_default.json"]
        if len(manifests) != 1:
            raise ValueError("Inventory6 must have exactly one COCO instances_default.json.")
        coco_bytes = archive.read(manifests[0])
    coco = json.loads(coco_bytes)
    images = {int(item["id"]): item for item in coco.get("images", [])}
    categories = {int(item["id"]): item["name"] for item in coco.get("categories", [])}
    if len(images) != 5 or categories != {1: "product"}:
        raise ValueError(f"Unexpected reviewed export: {len(images)} images, categories={categories}.")

    snapshots = {}
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("run_id") in RUN_IDS:
            snapshots[record["run_id"]] = record
    if set(snapshots) != set(RUN_IDS):
        raise ValueError("Historical snapshots do not contain exactly the five expected runs.")

    gallery = [path for path in (PACKAGE_ROOT / "data/source/catalog_references").rglob("*")
               if path.is_file()]
    gallery_by_hash = {sha256(path): path for path in gallery}
    if len(gallery_by_hash) != len(gallery):
        raise ValueError("Duplicate reference bytes in the packaged reference gallery need review.")

    def remap_product(product: dict) -> dict:
        mapped = {"sku": product["sku"], "name": product["name"], "references": []}
        refs = product.get("references") or [{"id": None, "image_path": product["image_path"]}]
        for reference in refs:
            original = Path(reference["image_path"])
            if not original.is_file():
                raise FileNotFoundError(f"Saved reference is missing locally: {original}")
            digest = sha256(original)
            packaged = gallery_by_hash.get(digest)
            if packaged is None:
                raise ValueError(f"Reference bytes are not present in the archived gallery: {original.name}")
            relative = packaged.relative_to(PACKAGE_ROOT).as_posix()
            mapped["references"].append({
                "id": reference.get("id"),
                "image_path": f"{REMOTE_ROOT}/{relative}",
                "sha256": digest,
            })
        return mapped

    product_snapshot = snapshots[RUN_IDS[0]]["products"]
    products = [remap_product(product) for product in product_snapshot]
    if len(products) != 14 or sum(len(product["references"]) for product in products) != 50:
        raise ValueError("Saved five-run reference snapshot did not resolve to 14 SKUs / 50 references.")
    by_filename = {}
    for run_id in RUN_IDS:
        record = snapshots[run_id]
        filename = Path(record["original_image"]).name
        if filename in by_filename:
            raise ValueError(f"Duplicate original filename in run snapshots: {filename}")
        by_filename[filename] = (run_id, record)
    coco_filenames = {Path(item["file_name"]).name for item in images.values()}
    if coco_filenames != set(by_filename):
        raise ValueError(f"COCO-to-original mapping mismatch: {coco_filenames ^ set(by_filename)}")

    out_images = OUTPUT / "images"
    out_annotations = OUTPUT / "annotations"
    out_inputs = OUTPUT / "inputs"
    out_evaluator = OUTPUT / "evaluator"
    out_images.mkdir(parents=True)
    out_annotations.mkdir()
    out_inputs.mkdir()
    out_evaluator.mkdir()
    copied_images = []
    for filename, (run_id, record) in by_filename.items():
        source = ORIGINALS / filename
        coco_image = next(item for item in images.values() if Path(item["file_name"]).name == filename)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            from PIL import Image, ImageOps
            with Image.open(source) as image:
                size = ImageOps.exif_transpose(image).size
        except ImportError as error:
            raise RuntimeError("Pillow is required to verify original photo dimensions.") from error
        if size != (int(coco_image["width"]), int(coco_image["height"])):
            raise ValueError(f"COCO dimensions mismatch for {filename}: {size}")
        if not validation.get("image_mapping", {}).get(filename, {}).get("mapping_verified_by_filename_and_dimensions"):
            raise ValueError(f"Existing audit does not verify the original mapping for {filename}.")
        target = out_images / filename
        shutil.copy2(source, target)
        copied_images.append({"filename": filename, "run_id": run_id,
                              "dimensions": list(size), "sha256": sha256(target),
                              "source_sha256": sha256(source)})

    annotation_target = out_annotations / "instances_default.json"
    annotation_target.write_bytes(coco_bytes)
    shutil.copy2(SOURCE_ZIP, out_inputs / "inventory6.zip")
    shutil.copy2(EVALUATOR, out_evaluator / EVALUATOR.name)

    sanitized_history = []
    for run_id in RUN_IDS:
        source_record = snapshots[run_id]
        filename = Path(source_record["original_image"]).name
        sanitized_history.append({
            "run_id": run_id,
            "original_image": f"{REMOTE_INPUT}/images/{filename}",
            "confidence": float(source_record["confidence"]),
            "minimum_match_margin": float(source_record["minimum_match_margin"]),
            "products": products,
        })
    write_json(out_inputs / "history.json", sanitized_history)
    write_json(out_inputs / "registry.json", {"business_id": 1, "products": products})

    counts = Counter()
    per_image = {}
    for annotation in coco.get("annotations", []):
        sku = (annotation.get("attributes") or {}).get("sku", "__missing__")
        counts[sku] += 1
        filename = Path(images[int(annotation["image_id"])] ["file_name"]).name
        per_image.setdefault(filename, Counter())[sku] += 1
    if sum(counts.values()) != 79 or counts["__missing__"]:
        raise ValueError(f"Unexpected annotation labels/counts: {counts}")
    summary = {
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": sha256(SOURCE_ZIP),
        "images": len(images),
        "annotations": sum(counts.values()),
        "categories": categories,
        "annotation_counts_by_sku": dict(sorted(counts.items())),
        "annotations_by_image": {name: dict(sorted(value.items()))
                                 for name, value in sorted(per_image.items())},
        "historical_runs": RUN_IDS,
        "reference_skus": len(products),
        "reference_photos": sum(len(product["references"]) for product in products),
        "original_photo_mapping": copied_images,
        "important_limitations": [
            "These five photos are development diagnostics, not an untouched evaluation set.",
            "All 79 reviewed annotations are included provisionally; no disputed or tiny polygon was edited or silently dropped.",
            "The COCO YOLO11n-seg model is generic 80-class segmentation. In proposal mode all classes are treated as untyped product candidates; this is not a retail/product-trained detector.",
            "The existing MobileSAM, ResNet18 and SIFT execution remains on CPU; YOLO uses the specified CUDA device.",
            "SKU thresholds and reference identities come from the saved historical snapshots and are unchanged.",
        ],
    }
    write_json(OUTPUT / "ground_truth_summary.json", summary)
    readme = """# October 17 five-photo RunPod diagnostic\n\nThis package runs a read-only, isolated comparison on the five original October 17 development photos and the reviewed Inventory6 COCO annotations. It does not train a model, read SQLite, or modify production references/history.\n\nIncluded: original photographs, original Inventory6 ZIP plus extracted COCO manifest, five minimal historical configuration snapshots, the exact 14-SKU/50-reference registry mapped by SHA-256 to the already-installed RunPod gallery, and an isolated evaluator copy.\n\nArms: A) existing MobileSAM proposals plus the unchanged ResNet18/SIFT matcher; B) pretrained COCO YOLO11n-seg with every class treated as an untyped proposal plus the same matcher; C) human-reviewed masks passed to the same matcher. YOLO is not retail-trained and may propose non-products. Existing MobileSAM and matcher stages use CPU; YOLO is requested on CUDA. Fixed confidence/margin thresholds are read from each saved test snapshot; YOLO conf=0.25, image size=960, mask-match IoU=0.50.\n\nAll 79 annotations, including 7 `__undefined__`, are evaluated provisionally. Earlier review noted some small/boundary ambiguities; nothing was automatically corrected or excluded. These are development diagnostics, not a final test set.\n\n`bundle_manifest.json` records SHA-256 hashes for package files. Reference photos are not duplicated in this upload; their exact hashes were resolved to the gallery already present in `/workspace/webshop_benchmark/data/source/catalog_references/`.\n"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    manifest = {"files": [{"path": path.relative_to(OUTPUT).as_posix(),
                            "bytes": path.stat().st_size, "sha256": sha256(path)}
                       for path in files]}
    write_json(OUTPUT / "bundle_manifest.json", manifest)
    archive = OUTPUT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(path for path in OUTPUT.rglob("*") if path.is_file()):
            bundle.write(path, path.relative_to(OUTPUT.parent).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member:
            raise RuntimeError(f"Package ZIP CRC failed at {bad_member}")
    print(json.dumps({"directory": str(OUTPUT), "archive": str(archive),
                      "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive),
                      "files": len(files) + 1, "images": len(images), "annotations": sum(counts.values()),
                      "registered": sum(value for sku, value in counts.items() if sku != "__undefined__"),
                      "undefined": counts["__undefined__"], "reference_skus": len(products),
                      "reference_photos": sum(len(p["references"]) for p in products),
                      "mapping_verified": True}, indent=2))


if __name__ == "__main__":
    main()
