"""Build an isolated generic-product YOLO detection dataset from reviewed labels.

The grocery masks are converted to enclosing boxes. Four early user-reviewed
box annotations are added to training. No source data is changed and no model
is trained by this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GROCERY = ROOT / "data/segmentation_dataset/experiments/initial_yolo26n_seg_20261017"
EARLY = HERE / "annotations/early_product_boxes_user_reviewed.json"
OUT = HERE / "training_data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def polygon_to_box(line: str) -> tuple[float, float, float, float]:
    values = [float(x) for x in line.split()]
    if len(values) < 7 or len(values) % 2 == 0:
        raise ValueError(f"Expected segmentation polygon label, got {line[:80]!r}")
    points = list(zip(values[1::2], values[2::2]))
    x1, x2 = min(x for x, _ in points), max(x for x, _ in points)
    y1, y2 = min(y for _, y in points), max(y for _, y in points)
    return x1, y1, x2, y2


def yolo_line(box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    return f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}"


def main() -> None:
    manifest = list(csv.DictReader((GROCERY / "dataset_manifest.csv").open(encoding="utf-8-sig", newline="")))
    early = json.loads(EARLY.read_text(encoding="utf-8"))
    if early.get("status") != "human_reviewed_boxes_imported":
        raise ValueError("Early annotations have not passed the import validator")
    if OUT.exists():
        shutil.rmtree(OUT)
    rows: list[dict[str, str]] = []
    counters = {"train": 0, "val": 0}

    def add_image(src: Path, split: str, group: str, boxes: list[tuple[float, float, float, float]], digest: str, label_source: str) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"Unexpected split {split}")
        i = counters[split]
        counters[split] += 1
        image_name = f"{split}_{i:03d}{src.suffix.lower()}"
        dst_image = OUT / "images" / split / image_name
        dst_label = OUT / "labels" / split / f"{split}_{i:03d}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_image)
        dst_label.write_text("\n".join(yolo_line(b) for b in boxes) + "\n", encoding="utf-8")
        rows.append({"split": split, "group_id": group, "image": dst_image.relative_to(OUT).as_posix(),
                     "label": dst_label.relative_to(OUT).as_posix(), "source": src.relative_to(ROOT).as_posix(),
                     "source_sha256": digest, "label_source": label_source, "instances": str(len(boxes))})

    for row in manifest:
        src = GROCERY / row["image_path"]
        src_label = GROCERY / row["label_path"]
        if sha256(src) != row["image_sha256"] or sha256(src_label) != row["label_sha256"]:
            raise ValueError(f"Approved grocery source hash changed: {src}")
        boxes = [polygon_to_box(line) for line in src_label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(boxes) != int(row["annotation_count"]):
            raise ValueError(f"Annotation count mismatch: {src_label}")
        split = "val" if row["split"] == "validation" else row["split"]
        add_image(src, split, row["group_id"], boxes, row["image_sha256"], "approved_grocery_polygon_envelopes")

    for scene in early["scenes"]:
        src = ROOT / scene["source_path"]
        if sha256(src) != scene["source_sha256"]:
            raise ValueError(f"Reviewed early source hash changed: {src}")
        w, h = scene["width"], scene["height"]
        boxes = [(x1/w, y1/h, x2/w, y2/h) for x1, y1, x2, y2 in scene["boxes_xyxy"]]
        add_image(src, "train", scene["arrangement_group"], boxes, scene["source_sha256"], "user_reviewed_xyxy")

    train_groups = {r["group_id"] for r in rows if r["split"] == "train"}
    val_groups = {r["group_id"] for r in rows if r["split"] == "val"}
    if train_groups & val_groups:
        raise ValueError(f"Arrangement leakage between train and validation: {train_groups & val_groups}")
    (OUT / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: product\n",
        encoding="utf-8")
    with (OUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "status": "prepared_no_training_run",
        "class": "generic product instance",
        "train_images": sum(r["split"] == "train" for r in rows),
        "train_instances": sum(int(r["instances"]) for r in rows if r["split"] == "train"),
        "validation_images": sum(r["split"] == "val" for r in rows),
        "validation_instances": sum(int(r["instances"]) for r in rows if r["split"] == "val"),
        "early_reviewed_train_images": sum(r["label_source"] == "user_reviewed_xyxy" for r in rows),
        "note": "Grocery segmentation masks are represented by their enclosing bounding boxes; no original annotations changed.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
