"""Run the isolated YOLO + few-shot matcher on arbitrary photographs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageOps

from .detectors import RetailYOLODetector, configure_ultralytics
from .hybrid_core import (MobileNetV2Encoder, abstaining_decision,
                          build_reference_index, count_by_sku, load_catalog,
                          render_boxes, scores_for_features)


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Photographs to analyze")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to a timestamped run folder")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--minimum-similarity", type=float, default=0.525,
                        help="Calibration-only cutoff from the five-photo development benchmark")
    parser.add_argument("--minimum-margin", type=float, default=0.025,
                        help="Calibration-only runner-up margin from the five-photo development benchmark")
    args = parser.parse_args()

    for path in args.images:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.out or ROOT / "experiments/hybrid_pipeline/runs" / datetime.now().strftime("photos_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    products = load_catalog()
    encoder = MobileNetV2Encoder()
    index = build_reference_index(products, {encoder.name: encoder})
    configure_ultralytics()
    detector = RetailYOLODetector(device="cpu")

    records = []
    for number, path in enumerate(args.images, 1):
        with Image.open(path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        start = perf_counter()
        detections = detector.predict(photo, products, conf=args.confidence, image_size=args.image_size)
        detection_seconds = perf_counter() - start

        crops = []
        for item in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
            crops.append(photo.crop((max(0, x1), max(0, y1), min(photo.width, x2), min(photo.height, y2))))
        start = perf_counter()
        vectors = encoder.encode(crops)
        for item, vector in zip(detections, vectors):
            scores, _, references = scores_for_features({encoder.name: vector}, index)
            item.update(abstaining_decision(scores, args.minimum_similarity, args.minimum_margin))
            accepted = item.get("sku")
            item["best_reference"] = references[encoder.name].get(accepted) if accepted else None
        matching_seconds = perf_counter() - start

        counts = count_by_sku(detections)
        output_stem = f"{number:02d}_{path.stem}"
        overlay = render_boxes(photo, detections)
        overlay.save(output_dir / f"{output_stem}_hybrid.png", optimize=True)
        records.append({
            "source": str(path.resolve()),
            "overlay": f"{output_stem}_hybrid.png",
            "image_size": list(photo.size),
            "detector": detector.backend,
            "recognizer": "MobileNetV2 few-shot cosine retrieval",
            "thresholds": {"minimum_similarity": args.minimum_similarity,
                           "minimum_margin": args.minimum_margin},
            "counts": counts,
            "unknown_count": sum(item.get("sku") is None for item in detections),
            "timing_seconds": {"detection": detection_seconds,
                               "matching": matching_seconds,
                               "total": detection_seconds + matching_seconds},
            "detections": detections,
        })

    payload = {"purpose": "Unscored exploratory photos; no object-level ground truth supplied.",
               "settings": {"confidence": args.confidence, "image_size": args.image_size,
                            "device": "cpu"}, "images": records}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, default=lambda value: value.tolist()
                                                         if hasattr(value, "tolist") else str(value)),
                                             encoding="utf-8")
    with (output_dir / "counts.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "sku", "quantity"])
        for row in records:
            for sku, quantity in row["counts"].items():
                writer.writerow([row["source"], sku, quantity])
            writer.writerow([row["source"], "__unknown__", row["unknown_count"]])
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
