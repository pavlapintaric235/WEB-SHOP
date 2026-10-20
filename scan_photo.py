#!/usr/bin/env python3
"""Find every distinct object in one photograph and write its box, mask and outline.

This is the standalone entry point to `shape_recognition`. It needs no SQLite registry,
no reference photographs and no GPU, so it works on any picture:

    python scan_photo.py shelf.jpg
    python scan_photo.py shelf.jpg --output out --threshold 0.08
    python scan_photo.py shelf.jpg --prompt "a screw" --prompt "a washer"

It writes two files next to each other: an overlay image showing one colour per detected
item, and a JSON file carrying the boxes, the mask outlines and the detector's own wording
for what it thinks each shape is.

The output is model proposal, not annotation. Nothing here has been reviewed by a person,
and the JSON records that in `review_status` so downstream code cannot lose the distinction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from shape_recognition import BACKEND, PROMPTS, REVIEW_STATUS, ShapeDetector, mask_leak_fraction

OUTLINE_COLOURS = [
    (255, 80, 0), (0, 200, 255), (120, 255, 0), (255, 0, 180), (255, 220, 0),
    (0, 120, 255), (255, 120, 120), (160, 0, 255), (0, 255, 160), (255, 160, 0),
]


def render(image: Image.Image, detections) -> Image.Image:
    """Tint each mask in its own colour and outline its box."""
    pixels = np.asarray(image.convert("RGB")).copy()
    for index, detection in enumerate(detections):
        if detection.mask is None:
            continue
        colour = np.array(OUTLINE_COLOURS[index % len(OUTLINE_COLOURS)], dtype=np.float32)
        selected = detection.mask
        pixels[selected] = (pixels[selected] * 0.55 + colour * 0.45).astype(np.uint8)

    overlay = Image.fromarray(pixels)
    draw = ImageDraw.Draw(overlay)
    width = max(2, round(max(overlay.size) / 400))
    for index, detection in enumerate(detections):
        colour = OUTLINE_COLOURS[index % len(OUTLINE_COLOURS)]
        draw.rectangle(list(detection.box), outline=colour, width=width)
        draw.text((detection.box[0] + width + 2, detection.box[1] + width + 2),
                  f"{index + 1} {detection.label} {detection.score:.2f}", fill=(255, 255, 255))
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Photograph to scan.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory for the overlay and JSON (default: beside the image).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Detector score threshold; lower finds more and is noisier.")
    parser.add_argument("--prompt", action="append", default=None,
                        help="Replace the built-in vocabulary; repeat for several words.")
    parser.add_argument("--no-masks", action="store_true", help="Boxes only, which is faster.")
    parser.add_argument("--max-side", type=int, default=None,
                        help="Working resolution of the longest side.")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"No such image: {args.image}")

    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    options = {"with_masks": not args.no_masks}
    if args.threshold is not None:
        options["threshold"] = args.threshold
    if args.prompt:
        options["prompts"] = args.prompt
    if args.max_side is not None:
        options["working_side"] = args.max_side

    print(f"Scanning {args.image.name} ({image.width}x{image.height}) on CPU...", flush=True)
    started = perf_counter()
    detections = ShapeDetector(**options).detect(image)
    seconds = perf_counter() - started

    destination = args.output or args.image.parent
    destination.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    overlay_path = destination / f"{stem}_shapes.jpg"
    json_path = destination / f"{stem}_shapes.json"
    render(image, detections).save(overlay_path, quality=92)

    from experiments.hybrid_pipeline.segmentation_models import mask_polygons

    records = []
    for index, detection in enumerate(detections, start=1):
        record = {
            "detection_id": f"det-{index}",
            "box_xyxy": [round(value, 2) for value in detection.box],
            "score": round(detection.score, 4),
            "label": detection.label,
        }
        if detection.mask is not None:
            record["mask_area_fraction"] = round(float(detection.mask.mean()), 4)
            record["leak_outside_box_fraction"] = mask_leak_fraction(detection.mask, detection.box)
            record["polygons"] = mask_polygons(detection.mask)
        records.append(record)

    json_path.write_text(json.dumps({
        "source": args.image.name,
        "width": image.width,
        "height": image.height,
        "backend": BACKEND if not args.no_masks else BACKEND.split("+")[0],
        "prompts": list(args.prompt or PROMPTS),
        "coordinate_space": "EXIF-transposed displayed-image pixels",
        "seconds": round(seconds, 2),
        "review_status": REVIEW_STATUS,
        "detections": records,
    }, indent=2), encoding="utf-8")

    print(f"\nFound {len(detections)} item(s) in {seconds:.1f}s")
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        print(f"  {index:2d}. {detection.label:26s} {detection.score:.2f}  "
              f"box=({x1},{y1})-({x2},{y2})")
    print(f"\n  overlay: {overlay_path}")
    print(f"  details: {json_path}")
    print(f"  status:  {REVIEW_STATUS} - model output, not reviewed annotation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
