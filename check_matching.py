"""Replay saved real SAM proposals to regression-test matching, not segmentation."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps

import db
from images import product_references
from sam_vision import MAX_IMAGE_SIDE, RECOGNITION_VERSION, REFERENCE_CACHE_VERSION, decode_mask, render_detections
from vision import load_model


def replay_segments(snapshot, original):
    working = original.copy()
    working.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    records = snapshot["raw_detections"]
    masks = [decode_mask(record["mask"]) for record in records]
    if any(mask.shape != (working.height, working.width) for mask in masks):
        raise ValueError("Saved proposal resolution differs from the current working image.")
    signature = hashlib.sha256(working.tobytes()).digest()

    def segment(image):
        if image.size != working.size or hashlib.sha256(image.tobytes()).digest() != signature:
            raise ValueError("Saved proposals may only be used with their original inventory image.")
        return masks, [record["mask_quality"] for record in records]

    return segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=db.DATA_DIR / "test_runs.jsonl")
    parser.add_argument("--business-id", type=int, default=1)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--truth-overrides", type=Path)
    args = parser.parse_args()
    products = db.get_business_products(args.business_id)
    skus = {product["sku"] for product in products}
    snapshots = [json.loads(line) for line in args.history.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshots = [snapshot for snapshot in snapshots if snapshot.get("business_id") == args.business_id
                 and {product["sku"] for product in snapshot["products"]} == skus
                 and snapshot.get("recognition_engine") == "sam"
                 and (not args.run or snapshot["run_id"] in args.run)]
    if not snapshots or set(args.run) - {snapshot["run_id"] for snapshot in snapshots}:
        parser.error("No compatible saved runs found, or a requested run is missing.")
    overrides = json.loads(args.truth_overrides.read_text(encoding="utf-8")) if args.truth_overrides else {}
    if set(overrides) - {snapshot["run_id"] for snapshot in snapshots}:
        parser.error("Truth overrides contain runs outside this replay.")
    # Never substitute an inventory mask while preparing an uncached reference.
    for product in products:
        for reference in product_references(product):
            fingerprint = hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii")
                                         + Path(reference["image_path"]).read_bytes()).hexdigest()
            if not (db.DATA_DIR / "models/reference_cache" / (fingerprint + ".npz")).is_file():
                parser.error("Prepare all active references with the real model before replaying masks.")
    model = load_model("sam")
    for product in products:
        for reference in product_references(product):
            model.reference({**product, "image_path": reference["image_path"]})
    stamp = datetime.now(ZoneInfo("Europe/Zagreb")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    folder = db.DATA_DIR / "recognition_checks" / (stamp + "-matching-replay")
    folder.mkdir(parents=True)
    report = {"purpose": "matching regression with saved real SAM masks; NOT fresh segmentation or unseen evaluation",
              "recognition_version": RECOGNITION_VERSION, "reference_cache_version": REFERENCE_CACHE_VERSION,
              "catalogue_skus": sorted(skus), "truth_overrides": overrides,
              "reference_catalogue": [{"sku": product["sku"], "references": [
                  {**reference, "sha256": hashlib.sha256(Path(reference["image_path"]).read_bytes()).hexdigest()}
                  for reference in product_references(product)]} for product in products],
              "code_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                              for name in ("sam_vision.py", "local_features.py")}, "runs": []}
    original_segment = model.segment
    for snapshot in snapshots:
        source = Path(snapshot["original_image"])
        with Image.open(source) as photo:
            photo = ImageOps.exif_transpose(photo).convert("RGB")
        model.segment = replay_segments(snapshot, photo)
        start = perf_counter()
        try:
            detections, annotated, _, raw = model.recognize(
                photo, products, snapshot["confidence"], snapshot["minimum_match_margin"]
            )
        finally:
            model.segment = original_segment
        counts = dict(Counter(item["sku"] for item in detections if item.get("sku")))
        override = overrides.get(snapshot["run_id"])
        expected = override["counts"] if override else snapshot.get("expected_counts")
        if expected is not None:
            if not set(expected).issubset(skus):
                raise ValueError("Expected counts contain an unregistered SKU.")
            for quantity in expected.values():
                db.validate_quantity(quantity)
        exact = all(counts.get(sku, 0) == expected.get(sku, 0) for sku in skus) if expected is not None else None
        result = {"run_id": snapshot["run_id"], "source": str(source),
                  "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "expected_counts": expected, "saved_expected_counts": snapshot.get("expected_counts"),
                  "before_counts": snapshot["counts"], "counts": counts, "exact_counts": exact,
                  "seconds": round(perf_counter() - start, 3), "timings": model.last_timings,
                  "similarity_threshold": snapshot["confidence"], "minimum_match_margin": snapshot["minimum_match_margin"],
                  "geometric_recoveries": sum(item["reason"] == "geometric_reference_match" for item in detections),
                  "detections": detections, "raw_detections": raw}
        if counts != snapshot["counts"] or exact is False:
            annotated.save(folder / (snapshot["run_id"] + "-annotated.png"))
            spotlight, _ = render_detections(photo, detections, spotlight=True)
            spotlight.save(folder / (snapshot["run_id"] + "-spotlight.png"))
        report["runs"].append(result)
        print({key: result[key] for key in ("run_id", "before_counts", "counts", "exact_counts", "geometric_recoveries", "seconds")}, flush=True)
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
