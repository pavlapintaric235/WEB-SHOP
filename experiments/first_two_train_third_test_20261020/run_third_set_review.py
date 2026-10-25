from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
OUT = EXP / "artifacts" / "third_set_review_20261020"
MANIFEST = ROOT / "03_newest_10_products_owner_employee_customer" / "manifest_this_part.csv"
SOURCES = {
    "new10_employee": Path(r"C:\Users\U938\Downloads\Employee_test"),
    "new10_customer": Path(r"C:\Users\U938\Downloads\Customer_test"),
}
MODELS = {
    "baseline": ROOT / "data/segmentation_dataset/experiments/runpod_yolo11n_detection_20261017/package/weights/sku110k-yolo11-n640.pt",
    "finetuned": EXP / "runs_cpu/retail_yolo11n_product_finetune_20261020-2/weights/best.pt",
}


def main() -> None:
    photos = OUT / "inputs"
    photos.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    scenes = [r for r in rows if r["role"] == "scene" and r["source_group"] in SOURCES]
    if len(scenes) != 30:
        raise RuntimeError(f"Expected 30 held-out scenes, found {len(scenes)}")
    image_paths: list[Path] = []
    for row in scenes:
        source = SOURCES[row["source_group"]] / row["original_filename"]
        target = photos / row["source_group"] / row["original_filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        import hashlib

        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != row["sha256"].lower():
            raise RuntimeError(f"Copied image hash mismatch: {target.name}")
        row["local_path"] = str(target.relative_to(OUT)).replace("\\", "/")
        image_paths.append(target)

    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "held_out_images": len(image_paths),
        "device": "cpu",
        "imgsz": 640,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.7,
        "class_note": "Generic product detections; class IDs are checkpoint-specific and do not identify SKUs.",
        "scenes": [{k: r[k] for k in ("source_group", "original_filename", "sha256", "local_path")} for r in scenes],
        "models": {},
    }
    for model_name, weights in MODELS.items():
        if not weights.is_file():
            raise FileNotFoundError(weights)
        model_out = OUT / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(weights))
        results = model.predict(
            source=[str(p) for p in image_paths],
            device="cpu",
            imgsz=640,
            conf=0.25,
            iou=0.7,
            workers=0,
            verbose=False,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(model_out),
            name="predictions",
            exist_ok=True,
        )
        detections = []
        for p, result in zip(image_paths, results, strict=True):
            boxes = []
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = [round(float(v), 2) for v in box.xyxy[0].tolist()]
                    cls = int(box.cls[0].item())
                    boxes.append({
                        "xyxy": xyxy,
                        "confidence": round(float(box.conf[0].item()), 5),
                        "class_id": cls,
                        "class_name": str(result.names.get(cls, cls)),
                    })
            detections.append({"image": str(p.relative_to(OUT)).replace("\\", "/"), "count": len(boxes), "boxes": boxes})
        report["models"][model_name] = {"weights": str(weights.relative_to(ROOT)), "detections": detections}
        del model
    (OUT / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote predictions for {len(image_paths)} scenes to {OUT}")


if __name__ == "__main__":
    main()
