"""Experimental product detector adapters; no production recognition imports are edited."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .hybrid_core import (EXPERIMENT_DIR, RETAIL_YOLO_FILE, RETAIL_YOLO_SHA256,
                          ROOT, YOLOE_FILE)


def configure_ultralytics():
    config = EXPERIMENT_DIR / ".cache" / "ultralytics"
    config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config.resolve())


class RetailYOLODetector:
    """Single-class SKU-110K YOLO detector. Requires a compatible local checkpoint."""

    backend = "sku110k_yolo"

    def __init__(self, weights: Path = RETAIL_YOLO_FILE, device: str = "cpu"):
        configure_ultralytics()
        if not weights.is_file():
            raise FileNotFoundError(
                f"SKU-110K checkpoint is not installed: {weights}. See README for access/licensing notes."
            )
        if weights.resolve() == RETAIL_YOLO_FILE.resolve():
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            if digest != RETAIL_YOLO_SHA256:
                raise ValueError("The SKU-110K ONNX file does not match its published checksum.")
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.weights = str(weights)

    def predict(self, image: Image.Image, products: list[dict], conf=0.12,
                image_size=1280, max_det=300) -> list[dict]:
        results = self.model.predict(image, device=self.device, imgsz=image_size, conf=conf,
                                     iou=0.55, max_det=max_det, verbose=False, save=False)
        output = []
        if not results or results[0].boxes is None:
            return output
        for box in results[0].boxes:
            coords = [float(value) for value in box.xyxy[0].tolist()]
            output.append({"box": coords, "detector_confidence": float(box.conf.item()),
                           "detector_class": int(box.cls.item()), "backend": self.backend})
        return output


class YOLOEVisualPromptFallback:
    """Use the already-installed YOLOE segmentation checkpoint with one prompt per SKU.

    This is explicitly not a generic or SKU-110K-trained detector. It is an offline fallback
    that lets the embedding/matching experiment run while retail weights are unavailable.
    """

    backend = "yoloe_visual_prompt_fallback"

    def __init__(self, weights: Path = YOLOE_FILE, device: str = "cpu"):
        configure_ultralytics()
        if not weights.is_file():
            raise FileNotFoundError(f"YOLOE fallback checkpoint is missing: {weights}")
        from ultralytics import YOLOE
        self.model = YOLOE(str(weights))
        self.device = device

    def _prompts(self, products: list[dict]):
        from vision import crop_reference_to_foreground

        columns = min(4, len(products))
        rows = (len(products) + columns - 1) // columns
        tile_w, tile_h = 320, 360
        board = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
        boxes = []
        for index, product in enumerate(products):
            path = Path(product["image_path"])
            with Image.open(path) as opened:
                reference = crop_reference_to_foreground(ImageOps.exif_transpose(opened).convert("RGB"))
            reference.thumbnail((tile_w - 36, tile_h - 56))
            x, y = (index % columns) * tile_w, (index // columns) * tile_h
            left = x + (tile_w - reference.width) // 2
            top = y + 28 + (tile_h - 56 - reference.height) // 2
            board.paste(reference, (left, top))
            boxes.append([left, top, left + reference.width, top + reference.height])
        return board, np.asarray(boxes, dtype=np.float32)

    def predict(self, image: Image.Image, products: list[dict], conf=0.05,
                image_size=960, max_det=300) -> list[dict]:
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        board, boxes = self._prompts(products)
        prompts = {"bboxes": boxes, "cls": np.arange(len(products), dtype=np.int64)}
        results = self.model.predict(
            source=image, refer_image=board, visual_prompts=prompts,
            predictor=YOLOEVPSegPredictor, imgsz=image_size, conf=conf,
            agnostic_nms=False, retina_masks=False, max_det=max_det,
            device=self.device, verbose=False, save=False,
        )
        output = []
        if not results or results[0].boxes is None:
            return output
        result_masks = results[0].masks.xy if results[0].masks is not None else [None] * len(results[0].boxes)
        for result_box, polygon in zip(results[0].boxes, result_masks):
            class_id = int(result_box.cls.item())
            coords = [float(value) for value in result_box.xyxy[0].tolist()]
            mask = None
            if polygon is not None and len(polygon) >= 3:
                mask_image = Image.new("L", image.size, 0)
                from PIL import ImageDraw
                ImageDraw.Draw(mask_image).polygon([tuple(point) for point in polygon], fill=1)
                mask = np.asarray(mask_image, dtype=bool)
            output.append({
                "box": coords,
                "mask": mask,
                "detector_confidence": float(result_box.conf.item()),
                "detector_class": class_id,
                "prompt_sku": products[class_id]["sku"] if 0 <= class_id < len(products) else None,
                "backend": self.backend,
            })
        return deduplicate_boxes(output)


def deduplicate_boxes(detections: list[dict], iou_threshold=0.78) -> list[dict]:
    kept = []
    for item in sorted(detections, key=lambda row: row["detector_confidence"], reverse=True):
        x1, y1, x2, y2 = item["box"]
        duplicate = False
        for previous in kept:
            a1, b1, a2, b2 = previous["box"]
            ix1, iy1, ix2, iy2 = max(x1, a1), max(y1, b1), min(x2, a2), min(y2, b2)
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = max(0.0, x2 - x1) * max(0.0, y2 - y1) + max(0.0, a2 - a1) * max(0.0, b2 - b1) - intersection
            if union and intersection / union >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def load_detector(mode="auto", weights: Path | None = None):
    weights = weights or Path(os.environ.get("INVENTORYLENS_SKU110K_WEIGHTS", RETAIL_YOLO_FILE))
    if mode in ("auto", "sku110k") and weights.is_file():
        return RetailYOLODetector(weights)
    if mode == "sku110k":
        raise FileNotFoundError(f"SKU-110K weights requested but not found: {weights}")
    if mode in ("auto", "yoloe-fallback"):
        return YOLOEVisualPromptFallback()
    raise ValueError(f"Unsupported detector mode: {mode}")

