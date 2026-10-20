"""Read-only few-shot recognition over experimental product masks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .hybrid_core import (MobileNetV2Encoder, abstaining_decision, fuse_sku_scores,
                          product_reference_paths, scores_for_reference_views)

CALIBRATED_MASK_THRESHOLDS = {
    frozenset({"mobilenetv2"}): (0.55, 0.00),
    frozenset({"siglip2"}): (0.55, 0.05),
    frozenset({"siglip2", "mobilenetv2"}): (0.65, 0.05),
}


def masked_crop(image: Image.Image, mask: np.ndarray, background=(124, 124, 124)) -> Image.Image:
    if mask.shape != (image.height, image.width):
        raise ValueError("Product mask dimensions must match the source image.")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot create a recognition crop from an empty mask.")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pixels = np.asarray(image.convert("RGB")).copy()
    pixels[~mask] = background
    return Image.fromarray(pixels).crop((left, top, right, bottom))


def masked_context_crop(image: Image.Image, mask: np.ndarray, fraction: float = 0.10,
                        background=(124, 124, 124)) -> Image.Image:
    """Mask within the product box while preserving a narrow real-image context ring."""
    if mask.shape != (image.height, image.width):
        raise ValueError("Product mask dimensions must match the source image.")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot create a recognition crop from an empty mask.")
    left, top, right, bottom = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad = max(1, round(max(right - left, bottom - top) * fraction))
    outer = (max(0, left - pad), max(0, top - pad),
             min(image.width, right + pad), min(image.height, bottom + pad))
    pixels = np.asarray(image.convert("RGB")).copy()[outer[1]:outer[3], outer[0]:outer[2]].copy()
    local_mask = mask[outer[1]:outer[3], outer[0]:outer[2]]
    inner = pixels[top - outer[1]:bottom - outer[1], left - outer[0]:right - outer[0]]
    inner_mask = local_mask[top - outer[1]:bottom - outer[1], left - outer[0]:right - outer[0]]
    inner[~inner_mask] = background
    return Image.fromarray(pixels)


def _load_cached_reference_crops(products: list[dict]) -> list[dict]:
    from model_assets import SAM_FILE
    from sam_vision import REFERENCE_CACHE_VERSION

    cache_dir = SAM_FILE.parent / "reference_cache"
    entries = []
    missing = []
    for product in products:
        for path in product_reference_paths(product):
            digest = hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii") + path.read_bytes()).hexdigest()
            cache_path = cache_dir / f"{digest}.npz"
            if not cache_path.is_file():
                missing.append(path.name)
                continue
            with np.load(cache_path, allow_pickle=False) as cached:
                crop = cached["crop"]
                if crop.ndim != 3 or crop.shape[2] != 3 or not np.isfinite(crop).all():
                    raise ValueError(f"Invalid read-only reference crop cache: {cache_path}")
                entries.append({"sku": product["sku"], "path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "crop": Image.fromarray(crop.astype(np.uint8))})
    if missing:
        raise RuntimeError(
            "The experimental matcher will not create or update production reference caches. "
            "Missing cached MobileSAM reference crops: " + ", ".join(missing[:8])
        )
    return entries


class MaskFewShotMatcher:
    """SigLIP2/MobileNetV2 retrieval against existing, cached product reference crops."""

    def __init__(self, products: list[dict], model_names: tuple[str, ...], device: str = "cpu",
                 local_files_only: bool = True, siglip_model_path: Path | None = None):
        if not model_names or not set(model_names) <= {"mobilenetv2", "siglip2"}:
            raise ValueError("Choose MobileNetV2, SigLIP2, or both.")
        try:
            self.minimum_similarity, self.minimum_margin = CALIBRATED_MASK_THRESHOLDS[
                frozenset(model_names)
            ]
        except KeyError as error:
            raise ValueError(f"No calibration cutoff exists for encoder set {model_names}.") from error
        self.products = products
        self.reference_rows = _load_cached_reference_crops(products)
        self.encoders = {}
        for name in model_names:
            if name == "mobilenetv2":
                self.encoders[name] = MobileNetV2Encoder(device=device)
            elif siglip_model_path is not None:
                from .hf_encoders import HuggingFaceImageEncoder, SIGLIP2_REVISION
                model_path = Path(siglip_model_path).expanduser().resolve(strict=True)
                self.encoders[name] = HuggingFaceImageEncoder(
                    str(model_path), "siglip2", device=device,
                    local_files_only=True, revision=SIGLIP2_REVISION,
                )
            else:
                from .hf_encoders import SigLIP2BaseEncoder
                self.encoders[name] = SigLIP2BaseEncoder(device=device, local_files_only=local_files_only)
        self.index = {product["sku"]: {name: [] for name in self.encoders} for product in products}
        crops = [row["crop"] for row in self.reference_rows]
        for name, encoder in self.encoders.items():
            features = encoder.encode(crops)
            for row, feature in zip(self.reference_rows, features):
                self.index[row["sku"]][name].append({
                    "feature": feature, "image": row["path"], "sha256": row["sha256"]
                })

    def match_crops(self, crops: list[Image.Image]) -> list[dict]:
        if not crops:
            return []
        features = {name: encoder.encode(crops) for name, encoder in self.encoders.items()}
        output = []
        for index in range(len(crops)):
            per_branch = {}
            best_refs = {}
            for name in self.encoders:
                branch_scores, _, references = scores_for_reference_views(
                    {name: features[name][index]}, self.index
                )
                per_branch[name] = branch_scores
                best_refs[name] = references[name]
            fused = fuse_sku_scores(per_branch)
            decision = abstaining_decision(
                fused, self.minimum_similarity, self.minimum_margin
            )
            decision["branch_scores"] = per_branch
            decision["best_reference_by_branch"] = best_refs
            decision["embedding_mode"] = "+".join(self.encoders)
            decision["thresholds"] = {
                "minimum_similarity": self.minimum_similarity,
                "minimum_margin": self.minimum_margin,
                "source": "calibration scenes only; polygon-masked crop comparison",
            }
            output.append(decision)
        return output

    def recognize(self, image: Image.Image, detections: list[dict]) -> list[dict]:
        crops = [masked_crop(image, item["mask"]) for item in detections]
        matches = self.match_crops(crops)
        output = []
        for item, crop, match in zip(detections, crops, matches):
            row = dict(item)
            row.update(match)
            row["crop_size"] = list(crop.size)
            row["mask_polygons"] = row.get("mask_polygons") or []
            output.append(row)
        return output
