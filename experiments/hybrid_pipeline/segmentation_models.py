"""Isolated mask-proposal backends for the InventoryLens hackathon experiment."""

from __future__ import annotations

import colorsys
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detectors import RetailYOLODetector, configure_ultralytics
from .hybrid_core import RETAIL_YOLO_FILE, ROOT


DEFAULT_SAM_WEIGHTS = ROOT / "data" / "models" / "mobile_sam.pt"


def _synchronize_device(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(torch.device(device))


def _assert_model_device(wrapper, device: str, name: str) -> None:
    if not str(device).startswith("cuda"):
        return
    model = getattr(wrapper, "model", wrapper)
    parameters = getattr(model, "parameters", None)
    parameter = next(parameters(), None) if parameters else None
    if parameter is None or parameter.device.type != "cuda":
        actual = None if parameter is None else str(parameter.device)
        raise RuntimeError(f"{name} did not execute on CUDA (model parameter device: {actual}).")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (size[1], size[0]):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def duplicate_proposal_pairs(detections: list[dict], threshold: float = 0.985) -> list[list[int]]:
    """Flag near-identical proposals for review; never merge or discard them here."""
    pairs = []
    for first in range(len(detections)):
        for second in range(first + 1, len(detections)):
            if box_iou(detections[first]["box"], detections[second]["box"]) >= threshold:
                pairs.append([first + 1, second + 1])
    return pairs


def mask_polygons(mask: np.ndarray, simplify_fraction: float = 0.0015) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(0.5, perimeter * simplify_fraction), True)
        points = approx.reshape(-1, 2)
        if len(points) >= 3:
            polygons.append([[float(x), float(y)] for x, y in points])
    return polygons


def _instance_color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)


def render_masks(image: Image.Image, detections: list[dict]) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = source.copy()
    for index, item in enumerate(detections):
        mask = resize_mask(item["mask"], image.size)
        color = np.asarray(_instance_color(index), dtype=np.float32)
        overlay[mask] = 0.48 * overlay[mask] + 0.52 * color
    output = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=max(14, output.width // 110))
    line_width = max(2, output.width // 700)
    for index, item in enumerate(detections):
        color = _instance_color(index)
        for polygon in mask_polygons(item["mask"]):
            draw.line([tuple(point) for point in polygon + [polygon[0]]], fill=color, width=line_width)
        left, top, _, _ = item["box"]
        label = f"{index + 1} {item.get('sku') or 'UNKNOWN'}"
        y = max(0, int(top) - font.size - 4)
        width = min(output.width, int(left) + int(draw.textlength(label, font=font)) + 8)
        draw.rectangle((int(left), y, width, y + font.size + 4), fill=color)
        draw.text((int(left) + 3, y + 1), label, fill=(15, 20, 24), font=font)
    return output


class YOLOPromptedMobileSAM:
    """Retail YOLO boxes followed by one MobileSAM mask prompt per box."""

    backend = "sku110k_yolo_plus_mobile_sam"

    def __init__(self, detector_weights: Path = RETAIL_YOLO_FILE,
                 sam_weights: Path = DEFAULT_SAM_WEIGHTS, device: str = "cpu"):
        configure_ultralytics()
        if not sam_weights.is_file():
            raise FileNotFoundError(f"MobileSAM checkpoint not found: {sam_weights}")
        from ultralytics import SAM

        self.detector = RetailYOLODetector(detector_weights, device=device)
        self.segmenter = SAM(str(sam_weights))
        self.device = device

    def predict(self, image: Image.Image, conf: float = 0.12, imgsz: int = 1280,
                max_det: int = 300) -> tuple[list[dict], dict]:
        from time import perf_counter

        image = image.convert("RGB")
        working = image.copy()
        working.thumbnail((1280, 1280))
        scale_x = image.width / working.width
        scale_y = image.height / working.height
        _synchronize_device(self.device)
        detect_start = perf_counter()
        proposals = self.detector.predict(working, [], conf=conf, image_size=imgsz, max_det=max_det)
        _synchronize_device(self.device)
        _assert_model_device(self.detector.model, self.device, "Retail YOLO detector")
        detection_seconds = perf_counter() - detect_start
        if not proposals:
            return [], {"detector_seconds": detection_seconds, "sam_seconds": 0.0,
                        "duplicate_proposal_pairs": []}

        boxes = [item["box"] for item in proposals]
        _synchronize_device(self.device)
        sam_start = perf_counter()
        results = self.segmenter.predict(
            source=np.asarray(working), bboxes=boxes,
            device=self.device, imgsz=1024, conf=0.0, verbose=False, save=False,
        )
        _synchronize_device(self.device)
        _assert_model_device(self.segmenter, self.device, "MobileSAM segmenter")
        sam_seconds = perf_counter() - sam_start
        if not results or results[0].masks is None:
            raise RuntimeError("MobileSAM returned no masks for non-empty YOLO box prompts.")

        raw_masks = results[0].masks.data.detach().cpu().numpy()
        if len(raw_masks) != len(proposals):
            raise RuntimeError(
                f"Mask/prompt association mismatch: {len(proposals)} boxes produced {len(raw_masks)} masks."
            )
        output = []
        for index, (proposal, raw_mask) in enumerate(zip(proposals, raw_masks), 1):
            mask = resize_mask(raw_mask > 0.5, image.size)
            if not mask.any():
                continue
            scaled_box = [proposal["box"][0] * scale_x, proposal["box"][1] * scale_y,
                          proposal["box"][2] * scale_x, proposal["box"][3] * scale_y]
            output.append({**proposal, "box": scaled_box, "instance_id": index, "mask": mask,
                           "mask_polygons": mask_polygons(mask), "backend": self.backend,
                           "prompt_box": scaled_box, "association": "input_box_order"})
        return output, {
            "detector_seconds": detection_seconds,
            "sam_seconds": sam_seconds,
            "duplicate_proposal_pairs": duplicate_proposal_pairs(proposals),
            "proposal_count": len(proposals),
            "mask_count": len(output),
        }


class YOLO11Segmentation:
    """Pretrained or fine-tuned Ultralytics YOLO11 segmentation checkpoint."""

    backend = "yolo11_instance_segmentation"

    def __init__(self, weights: Path, device: str = "cpu"):
        configure_ultralytics()
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO11-seg checkpoint not found: {weights}")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if getattr(self.model, "task", None) != "segment":
            raise ValueError("Expected a YOLO segmentation checkpoint; detection-only weights are not accepted.")
        self.weights = str(weights.resolve())
        self.device = device

    def predict(self, image: Image.Image, conf: float = 0.25, imgsz: int = 960,
                max_det: int = 300) -> tuple[list[dict], dict]:
        from time import perf_counter

        image = image.convert("RGB")
        working = image.copy()
        working.thumbnail((1280, 1280))
        scale_x = image.width / working.width
        scale_y = image.height / working.height
        _synchronize_device(self.device)
        started = perf_counter()
        results = self.model.predict(source=working, imgsz=imgsz, conf=conf,
                                     max_det=max_det, device=self.device, retina_masks=True,
                                     verbose=False, save=False)
        _synchronize_device(self.device)
        _assert_model_device(self.model, self.device, "YOLO11 segmentation model")
        elapsed = perf_counter() - started
        if not results or results[0].boxes is None or results[0].masks is None:
            return [], {"model_seconds": elapsed, "proposal_count": 0, "mask_count": 0}
        result = results[0]
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        masks = result.masks.data.detach().cpu().numpy()
        if not (len(boxes) == len(scores) == len(masks)):
            raise RuntimeError("YOLO11-seg returned misaligned boxes, scores and masks.")
        output = []
        for index, (box, score, raw_mask) in enumerate(zip(boxes, scores, masks), 1):
            mask = resize_mask(raw_mask > 0.5, image.size)
            if not mask.any():
                continue
            scaled_box = [float(box[0] * scale_x), float(box[1] * scale_y),
                          float(box[2] * scale_x), float(box[3] * scale_y)]
            output.append({"box": scaled_box,
                           "detector_confidence": float(score), "detector_class": 0,
                           "instance_id": index, "mask": mask,
                           "mask_polygons": mask_polygons(mask), "backend": self.backend,
                           "association": "yolo_instance_order"})
        return output, {"model_seconds": elapsed, "proposal_count": len(boxes),
                        "mask_count": len(output)}
