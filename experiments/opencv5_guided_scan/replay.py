"""Replay frozen InventoryLens images and saved DINO/SigLIP2 output locally.

This module never invokes a model, AWS, or the production inventory service.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from capture_gate import capture_decision, quality


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "data/recognition_checks/final_comparative_gpu_benchmark_20261018"
    / "inventorylens_final_comparative_gpu_benchmark/data/source"
)
RUNS_ROOT = Path(__file__).resolve().parent / "runs"
ARCHIVE_SHA256 = "20b98a4b3a8bfe5ce0d1fb84e2dc78353fd695563b8e13cce4ae6d3a1b88bff3"
DETECTIONS_MEMBER = "detector_retry2/detections.csv"
PREDICTIONS_MEMBER = "recognition_retry1/per_detection_predictions.csv"
REPORT_MEMBER = "recognition_retry1/report.json"

# Scene, filename, manifest digest, workflow, human review context.
FIXTURES = [
    ("E01", "IMG_4703 (1).jpeg", "b0b797f715fa4fc42310f818e4e301fd7a39797012c1be3d67a37fef0fce7939", "employee", "Two Energia cans share a box; another was missed."),
    ("E02", "IMG_4707.jpeg", "a032caa5beefd214dd10efafd9106ba7e726cf333051074341d3b7ffc45c22d9", "employee", "Human review found acceptable masks. Dark products are not by themselves a bad exposure."),
    ("E06", "IMG_4717.jpeg", "d5338ce665094ba4abdec781a0dbcca7f6c2f8cf5d012e87c06eae1682cf8850", "employee", "Six wrapped items were confirmed; DINO returned no boxes. Add items manually."),
    ("B01", "IMG_4754.jpeg", "9f8f47fcaf840bd63cb8a56a5e00be5b0d384c96246f32545d05d1679b232cb0", "customer_basket", "Simple four-product scene; source of in-memory blur and dark controls."),
    ("B07", "IMG_4764.jpeg", "16f3fbf55a44395c79dc530b6f2ee149f3ef24e574385f9e2fdf06e2b1cbda57", "customer_basket", "One DINO box spans multiple SKUs. Mark it split and add separate items."),
    ("B08", "IMG_4768.jpeg", "4a54d28bf74b50957385a32e52a3a974b95014c1801b9ffd7005d98bc01691a9", "customer_basket", "One of two Oreos was missed. Add the missed item manually."),
    ("B10", "IMG_4770.jpeg", "78201d59483f66687e3c566a67649b2b16247f5e5c447ef447302613aeaf7379", "customer_basket", "First view of the same 10-unit arrangement, per the user's later correction. Do not sum views."),
    ("B10", "IMG_4771.jpeg", "a98ae0dde7df2202b9d2a9e0d3843c6de2e89e2d391fece38", "customer_basket", "Second view of the same 10-unit arrangement, per the user's later correction. Reconcile across views."),
]
EXPECTED_DETECTIONS = [6, 6, 0, 4, 1, 3, 4, 5]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def archive_rows(archive_path: Path) -> tuple[list[dict], list[dict], dict]:
    if file_digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("DINO archive SHA-256 does not match the audited artifact")
    with zipfile.ZipFile(archive_path) as archive:
        detections = csv_rows(archive.read(DETECTIONS_MEMBER))
        predictions = csv_rows(archive.read(PREDICTIONS_MEMBER))
        report = json.loads(archive.read(REPORT_MEMBER))
    return detections, predictions, report


def key(row: dict) -> tuple[str, str, str]:
    return row["scene_id"], row["image"], row["instance_id"]


def index_unique(rows: list[dict], label: str) -> dict[tuple[str, str, str], dict]:
    result = {}
    for row in rows:
        row_key = key(row)
        if row_key in result:
            raise ValueError(f"Duplicate {label} key: {row_key}")
        result[row_key] = row
    return result


def selected_predictions(detections: list[dict], predictions: list[dict]) -> tuple[dict, dict]:
    selected = {(scene, image) for scene, image, *_ in FIXTURES}
    det = [r for r in detections if (r["scene_id"], r["image"]) in selected]
    pred = [r for r in predictions if (r["scene_id"], r["image"]) in selected
            and r["crop_variant"] == "boxes" and r["recognizer"] == "siglip2"]
    det_by_key = index_unique(det, "detection")
    pred_by_key = index_unique(pred, "SigLIP2 box prediction")
    if det_by_key.keys() != pred_by_key.keys():
        missing = sorted(det_by_key.keys() - pred_by_key.keys())
        extra = sorted(pred_by_key.keys() - det_by_key.keys())
        raise ValueError(f"Saved output join mismatch; missing={missing}, extra={extra}")
    if len(det_by_key) != 29:
        raise ValueError(f"Expected 29 DINO boxes, got {len(det_by_key)}")
    return det_by_key, pred_by_key


def build_records(source_root: Path, detections: dict, predictions: dict) -> tuple[list[dict], list[np.ndarray]]:
    manifest = csv_rows((source_root / "photos.csv").read_bytes())
    by_photo = {(r["scene_id"], r["original_filename"]): r for r in manifest if r["role"] == "scene"}
    image_root = (source_root / "images").resolve(strict=True)
    records = []
    image_pixels = []
    for index, (scene, filename, expected_hash, workflow, human_note) in enumerate(FIXTURES):
        entry = by_photo.get((scene, filename))
        if entry is None or entry["sha256"] != expected_hash or entry["data_partition"] != "holdout_unmodified":
            raise ValueError(f"Source manifest mismatch for {scene}/{filename}")
        path = (image_root / entry["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(image_root):
            raise ValueError(f"Source path escapes images root: {path}")
        data = path.read_bytes()
        if digest(data) != expected_hash:
            raise ValueError(f"Source image SHA-256 mismatch: {path}")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        metrics = quality(image)
        if not metrics["decode_ok"] or (metrics["width"], metrics["height"]) != (
            int(entry["width"]), int(entry["height"])
        ):
            raise ValueError(f"Image decode or dimensions mismatch: {path}")
        decision = capture_decision(metrics)
        if decision["action"] != "proceed":
            raise ValueError(f"Frozen source image unexpectedly fails the demo capture gate: {path}")
        boxes = []
        related = [(k, r) for k, r in detections.items() if k[:2] == (scene, filename)]
        related.sort(key=lambda item: int(item[0][2]))
        if len(related) != EXPECTED_DETECTIONS[index]:
            raise ValueError(f"Unexpected box count for {scene}/{filename}: {len(related)}")
        for det_key, det in related:
            pred = predictions[det_key]
            xyxy = [float(det[field]) for field in ("x1", "y1", "x2", "y2")]
            predicted_box = json.loads(pred["box_xyxy"])
            if any(abs(a - b) > 1 for a, b in zip(xyxy, predicted_box)):
                raise ValueError(f"Box coordinates disagree for {det_key}")
            if not (0 <= xyxy[0] < xyxy[2] <= metrics["width"] + 1 and
                    0 <= xyxy[1] < xyxy[3] <= metrics["height"] + 1):
                raise ValueError(f"Invalid DINO box for {det_key}")
            boxes.append({
                "replay_id": f"{expected_hash}:{det_key[2]}",
                "instance_id": int(det_key[2]),
                "box_xyxy": xyxy,
                "detector_confidence": float(det["detector_confidence"]),
                "recognizer": "siglip2",
                "crop_variant": "boxes",
                "top1_sku": pred["top1_sku"],
                "top1_similarity": float(pred["top1_similarity"]),
                "top5": json.loads(pred["top5"]),
                "accepted_sku": pred["sku"] or None,
                "recognition_status": pred["status"],
                "recognition_reason": pred["reason"],
                "margin": float(pred["margin"]) if pred["margin"] else None,
                "thresholds": json.loads(pred["thresholds"]),
                "review_decision": "pending",
            })
        records.append({
            "photo_id": f"{scene}:{filename}",
            "scene_id": scene,
            "arrangement_id": scene,
            "workflow": workflow,
            "filename": filename,
            "relative_path": entry["relative_path"],
            "sha256": expected_hash,
            "quality": metrics,
            "capture_decision": decision,
            "human_review_context": human_note,
            "preview": f"previews/{index + 1:02d}_{scene}.jpg",
            "boxes": boxes,
            "review_state": "pending",
        })
        image_pixels.append(image)
    return records, image_pixels


def render_preview(image: np.ndarray, boxes: list[dict], target: Path) -> None:
    canvas = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = (round(value) for value in box["box_xyxy"])
        color = (0, 165, 255) if box["recognition_status"] == "unknown" else (40, 190, 60)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
        label = f"#{box['instance_id']} {box['accepted_sku'] or 'unknown'}"
        cv2.putText(canvas, label, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    height, width = canvas.shape[:2]
    if max(height, width) > 1400:
        scale = 1400 / max(height, width)
        canvas = cv2.resize(canvas, (round(width * scale), round(height * scale)),
                            interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 82]):
        raise OSError(f"Could not write preview: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Audited DINO result ZIP")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--run-name", required=True, help="New output folder name below experiment runs/")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name must contain only letters, digits, underscores or hyphens")
    output = RUNS_ROOT / args.run_name
    if output.exists():
        parser.error(f"Output already exists; refusing to overwrite: {output}")
    if cv2.__version__.split(".")[0] != "5":
        raise RuntimeError(f"OpenCV 5 is required; found {cv2.__version__}")

    source_root = args.source_root.resolve(strict=True)
    archive_path = args.archive.resolve(strict=True)
    raw_detections, raw_predictions, report = archive_rows(archive_path)
    detections, predictions = selected_predictions(raw_detections, raw_predictions)
    photos, pixels = build_records(source_root, detections, predictions)
    statuses = Counter(box["recognition_status"] for photo in photos for box in photo["boxes"])
    if statuses != {"recognized": 18, "unknown": 11}:
        raise ValueError(f"Unexpected SigLIP2 acceptance counts: {dict(statuses)}")

    b01 = pixels[3]
    controls = [
        {"name": "B01 blur sigma 6", "source": "in_memory_only",
         "quality": quality(cv2.GaussianBlur(b01, (0, 0), sigmaX=6))},
        {"name": "B01 dark alpha 0.07", "source": "in_memory_only",
         "quality": quality(cv2.convertScaleAbs(b01, alpha=0.07, beta=0))},
    ]
    for item in controls:
        item["capture_decision"] = capture_decision(item["quality"])
    if controls[0]["capture_decision"]["reason"] != "Hold the camera steadier":
        raise ValueError("Blur control did not take the expected retake branch")
    if controls[1]["capture_decision"]["reason"] != "Use more light":
        raise ValueError("Dark control did not take the expected retake branch")

    trace = {
        "kind": "inventorylens_opencv5_guided_scan_replay",
        "status": "review_only_no_inventory_commit",
        "generated_at": datetime.now(ZoneInfo("Europe/Zagreb")).isoformat(timespec="seconds"),
        "opencv_version": cv2.__version__,
        "opencv_python_distribution": version("opencv-python"),
        "source_manifest": str(source_root / "photos.csv"),
        "saved_inference_archive": str(archive_path),
        "saved_inference_archive_sha256": ARCHIVE_SHA256,
        "model_provenance": {
            "detector": report.get("detector"),
            "recognizer": "siglip2",
            "recognition_thresholds": report.get("fixed_thresholds", {}).get("siglip2"),
            "inference_mode": "saved_output_replay_no_model_execution",
        },
        "capture_rule": {"short_side_min_px": 720, "dark_pct_max": 35,
                         "bright_pct_max": 35, "lap_var_1024_min": 100,
                         "calibrated": False},
        "known_arrangement_totals_for_review_context_only": {"E06": 6, "B10": 10},
        "summary": {"photos": len(photos), "arrangements": len({p["arrangement_id"] for p in photos}),
                    "dino_boxes": len(detections), "siglip2_accepted": statuses["recognized"],
                    "siglip2_unknown": statuses["unknown"]},
        "controls": controls,
        "photos": photos,
        "scoring_limit": "No instance-level SKU accuracy, unknown-rejection accuracy, detector recall, or capture-gate accuracy can be scored from these fixtures.",
    }

    output.mkdir(parents=True)
    (output / "previews").mkdir()
    for photo, image in zip(photos, pixels):
        render_preview(image, photo["boxes"], output / photo["preview"])
    (output / "replay.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    template = (Path(__file__).resolve().parent / "review_template.html").read_text(encoding="utf-8")
    embedded = json.dumps(trace, ensure_ascii=False).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__REPLAY_JSON__", embedded), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": trace["summary"],
                      "controls": [c["capture_decision"]["reason"] for c in controls]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
