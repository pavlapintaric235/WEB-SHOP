"""Run the local retail detector on the four early scenes for annotation review.

Output is an unapproved draft. It must never be used as training ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / "01_early_k18_wow_coconut_and_grocery_history_a/manifest_this_part.csv"
ARCHIVE = ROOT / "data/archives/20261014-223956-pre-grocery/test_runs"
WEIGHTS = ROOT / "data/segmentation_dataset/experiments/runpod_yolo11n_detection_20261017/package/weights/sku110k-yolo11-n640.pt"
WEIGHTS_SHA256 = "4888d44cee1a3c55fa961599b762bad0e5aa31194237d8d10d717dff15e0cfdb"
OUTPUT = HERE / "draft_early_boxes.json"
PREVIEWS = HERE / "proposals" / "retail_detector_previews"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing draft: {OUTPUT}")
    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError("Retail detector checkpoint hash mismatch")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["source_group"] == "early3_scene"]
    by_hash: dict[str, dict[str, str]] = {}
    for row in rows:
        by_hash.setdefault(row["sha256"], row)
    if len(by_hash) != 4:
        raise ValueError(f"Expected four distinct early photos, got {len(by_hash)}")
    model = YOLO(str(WEIGHTS))
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    output = {"status": "unapproved_model_box_drafts", "weights_sha256": WEIGHTS_SHA256,
              "imgsz": 640, "confidence_threshold": 0.20, "scenes": []}
    for digest, row in sorted(by_hash.items(), key=lambda entry: entry[1]["original_filename"]):
        photo = ARCHIVE / row["original_filename"]
        if sha256(photo) != digest:
            raise ValueError(f"Early photo hash mismatch: {photo.name}")
        result = model.predict(source=str(photo), imgsz=640, conf=0.20, iou=0.5,
                               max_det=30, device="cpu", verbose=False)[0]
        boxes = []
        for index, box in enumerate(result.boxes, 1):
            boxes.append({"proposal_id": f"det-{index}",
                          "xyxy": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                          "confidence": round(float(box.conf[0]), 4)})
        with Image.open(photo) as opened:
            image = opened.convert("RGB")
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        width = max(5, image.width // 400)
        for box in boxes:
            draw.rectangle(box["xyxy"], outline=(255, 130, 0), width=width)
            x1, y1, _, _ = box["xyxy"]
            draw.text((x1 + width, y1 + width), box["proposal_id"], fill=(255, 130, 0),
                      stroke_width=2, stroke_fill=(0, 0, 0))
        preview_path = PREVIEWS / f"{photo.stem}_draft.jpg"
        preview.save(preview_path, quality=88)
        output["scenes"].append({"source_name": photo.name, "source_sha256": digest,
                                 "width": image.width, "height": image.height,
                                 "proposal_count": len(boxes), "boxes": boxes,
                                 "review_status": "unreviewed", "preview": str(preview_path.relative_to(HERE))})
        print(f"{photo.name}: {len(boxes)} unapproved boxes")
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Draft report: {OUTPUT}")


if __name__ == "__main__":
    main()
