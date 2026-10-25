"""Convert the reviewed, frozen YOLO generic-product split to TAO DINO COCO JSON.

This only creates a COCO-format copy under ignored training_data/tao_coco/.
It does not alter source images, labels, split membership, or train a model.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "training_data"
OUT = DATA / "tao_coco"


def convert(split: str) -> dict:
    image_dir = DATA / "images" / split
    label_dir = DATA / "labels" / split
    images, annotations = [], []
    ann_id = 1
    for image_id, image_path in enumerate(sorted(image_dir.iterdir()), start=1):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        with Image.open(image_path) as im:
            width, height = im.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        label_path = label_dir / f"{image_path.stem}.txt"
        lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in lines:
            if len(row) != 5 or row[0] != "0":
                raise ValueError(f"Expected one normalized class-0 YOLO box: {label_path}: {row}")
            _, cx, cy, bw, bh = map(float, row)
            x, y, w, h = (cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height
            if w <= 0 or h <= 0 or x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1:
                raise ValueError(f"Invalid box in {label_path}: {row}")
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,  # TAO DINO reserves category 0 for background.
                "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                "area": round(w * h, 3),
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "info": {"description": f"Reviewed generic product boxes; frozen {split} split", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "product", "supercategory": "product"}],
    }


def main() -> None:
    expected = {"train": (25, 144), "val": (5, 33)}
    summary = {}
    for split, (expected_images, expected_boxes) in expected.items():
        data = convert(split)
        got = (len(data["images"]), len(data["annotations"]))
        if got != (expected_images, expected_boxes):
            raise ValueError(f"{split} split changed: expected {(expected_images, expected_boxes)}, got {got}")
        dest = OUT / "annotations" / f"instances_{split}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        summary[split] = {"images": got[0], "boxes": got[1], "json": dest.relative_to(DATA).as_posix()}
    print(json.dumps({"status": "prepared_no_training_run", "class": "product", "category_id": 1, "splits": summary}, indent=2))


if __name__ == "__main__":
    main()
