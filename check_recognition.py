"""Run reproducible development checks without overwriting the user's latest app run."""

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
from sam_vision import RECOGNITION_VERSION, REFERENCE_CACHE_VERSION, render_detections
from vision import detect_products, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--engine", choices=["sam", "yoloe"], default="sam")
    parser.add_argument("--exclude-sku", action="append", default=[])
    parser.add_argument("--expect", nargs=2, action="append", metavar=("SKU", "QUANTITY"))
    parser.add_argument("--similarity", type=float, default=0.80)
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()
    expected = {sku: int(quantity) for sku, quantity in args.expect} if args.expect else None
    db.init_db()
    business = db.get_or_create_business()
    products = db.get_business_products(business["id"])
    products = [product for product in products if product["sku"] not in args.exclude_sku]
    if expected is not None:
        known_skus = {product["sku"] for product in products}
        if not set(expected).issubset(known_skus):
            parser.error("Expected counts contain unregistered SKUs.")
        for quantity in expected.values():
            db.validate_quantity(quantity)
    stamp = datetime.now(ZoneInfo("Europe/Zagreb")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    folder = db.DATA_DIR / "recognition_checks" / stamp
    folder.mkdir(parents=True)
    start = perf_counter()
    model = load_model(args.engine)
    print(f"Loaded {args.engine} in {perf_counter() - start:.1f}s", flush=True)
    report = {"purpose": "development check, not an unseen-image evaluation", "engine": args.engine,
              "reference_cache_version": REFERENCE_CACHE_VERSION if args.engine == "sam" else None,
              "recognition_version": RECOGNITION_VERSION if args.engine == "sam" else None,
              "catalogue_skus": [product["sku"] for product in products],
              "reference_catalogue": [{"sku": product["sku"], "references": [
                  {**reference, "sha256": hashlib.sha256(Path(reference["image_path"]).read_bytes()).hexdigest()}
                  for reference in product_references(product)]} for product in products],
              "similarity_threshold": args.similarity, "minimum_match_margin": args.margin,
              "images": [], "code_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                             for name in ("vision.py", "sam_vision.py", "local_features.py")}}
    for index, path in enumerate(args.image):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        print(f"Running {Path(path).name}...", flush=True)
        start = perf_counter()

        def show_progress(event):
            if event["stage"] == "references":
                label = f'References {event["completed"]}/{event["total"]}'
                if event.get("sku"):
                    label += f' | {event["sku"]}, view {event["view"]}'
                print(label, flush=True)
            else:
                print(f'Stage: {event["stage"]}', flush=True)

        detections, annotated, board, raw = detect_products(
            model, image, products, args.similarity if args.engine == "sam" else 0.25, args.margin,
            progress=show_progress,
        )
        counts = dict(Counter(item["sku"] for item in detections if item.get("sku")))
        spotlight, kind = render_detections(image, detections, spotlight=True)
        annotated.save(folder / f"{index}-annotated.png")
        spotlight.save(folder / f"{index}-spotlight.png")
        board.save(folder / f"{index}-references.png")
        result = {"source": str(Path(path).resolve()),
                  "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  "seconds": round(perf_counter() - start, 2),
                  "timings": getattr(model, "last_timings", None) if args.engine == "sam" else None,
                  "expected_counts": expected, "counts": counts,
                  "uncertain_regions": sum(not item.get("sku") for item in detections),
                  "raw_regions": len(raw), "spotlight_kind": kind,
                  "detections": detections, "raw_detections": raw}
        if expected is not None:
            result["exact_counts"] = all(counts.get(product["sku"], 0) == expected.get(product["sku"], 0)
                                         for product in products)
        report["images"].append(result)
        print({key: value for key, value in result.items() if key not in ("detections", "raw_detections")}, flush=True)
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
