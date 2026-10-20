"""Pure OpenCV 5 capture checks shared by the local replay and AWS pilot."""

from __future__ import annotations

import cv2
import numpy as np


def quality(image: np.ndarray | None) -> dict:
    if image is None:
        return {"decode_ok": False, "width": None, "height": None,
                "lap_var_1024": None, "dark_pct": None, "bright_pct": None}
    height, width = image.shape[:2]
    scale = min(1.0, 1024.0 / max(height, width))
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return {
        "decode_ok": True,
        "width": width,
        "height": height,
        "lap_var_1024": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "dark_pct": round(float(np.mean(gray <= 20) * 100), 2),
        "bright_pct": round(float(np.mean(gray >= 235) * 100), 2),
    }


def capture_decision(metrics: dict) -> dict[str, str]:
    if not metrics["decode_ok"]:
        return {"action": "retake", "reason": "Image could not be decoded"}
    if min(metrics["width"], metrics["height"]) < 720:
        return {"action": "retake", "reason": "Move closer or capture at higher resolution"}
    if metrics["dark_pct"] > 35:
        return {"action": "retake", "reason": "Use more light"}
    if metrics["bright_pct"] > 35:
        return {"action": "retake", "reason": "Reduce glare"}
    if metrics["lap_var_1024"] < 100:
        return {"action": "retake", "reason": "Hold the camera steadier"}
    return {"action": "proceed", "reason": "Capture checks passed; inspect every item"}
