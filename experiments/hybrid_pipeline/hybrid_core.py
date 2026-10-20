"""Few-shot SKU retrieval and isolated detector adapters for Web&Shop."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = EXPERIMENT_DIR / "weights"
MOBILENET_FILE = WEIGHTS_DIR / "mobilenet_v2-7ebf99e0.pth"
MOBILENET_SHA256 = "7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f"
YOLOE_FILE = ROOT / "yoloe-26s-seg.pt"
RETAIL_YOLO_FILE = WEIGHTS_DIR / "sku110k-yolo11-n640.onnx"
RETAIL_YOLO_SHA256 = "5810269bf9687ca93b0d4e1bc91cb83ac4311cd48d91c4f6091777721ba083c5"
DEFAULT_SIMILARITY = 0.55
DEFAULT_MARGIN = 0.04


def unit_vector(values: Any) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("Embedding must contain finite, non-zero values.")
    return vector / norm


def fuse_sku_scores(branch_scores: dict[str, dict[str, float]], weights=None) -> dict[str, float]:
    """Average independently normalized cosine scores, one vote per encoder."""
    if not branch_scores:
        raise ValueError("At least one embedding branch is required.")
    names = list(branch_scores)
    weights = weights or {name: 1.0 for name in names}
    denominator = sum(float(weights.get(name, 0.0)) for name in names)
    if denominator <= 0:
        raise ValueError("At least one embedding branch must have positive weight.")
    skus = set.intersection(*(set(branch_scores[name]) for name in names))
    if not skus:
        raise ValueError("Embedding branches do not share any SKU candidates.")
    return {
        sku: sum(float(weights.get(name, 0.0)) * float(branch_scores[name][sku])
                 for name in names) / denominator
        for sku in skus
    }


def abstaining_decision(scores: dict[str, float], minimum=DEFAULT_SIMILARITY,
                        margin=DEFAULT_MARGIN) -> dict:
    if not scores:
        return {"sku": None, "status": "unknown", "reason": "no_registered_references",
                "similarity": None, "runner_up_similarity": None, "margin": None}
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    sku, best = ordered[0]
    runner = ordered[1][1] if len(ordered) > 1 else None
    gap = best - runner if runner is not None else None
    accepted = best >= minimum and (gap is None or gap >= margin)
    return {
        "sku": sku if accepted else None,
        "status": "recognized" if accepted else "unknown",
        "reason": "reference_match" if accepted else (
            "below_similarity_threshold" if best < minimum else "ambiguous_margin"),
        "similarity": float(best),
        "runner_up_similarity": float(runner) if runner is not None else None,
        "margin": float(gap) if gap is not None else None,
        "ranked_scores": [{"sku": key, "similarity": float(value)}
                          for key, value in ordered[:5]],
    }


def visible_box(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def crop_masked(image: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.shape != (image.height, image.width):
        raise ValueError("Mask size does not match its source photograph.")
    box = visible_box(mask)
    if box is None:
        raise ValueError("Cannot crop an empty product mask.")
    pixels = np.asarray(image.convert("RGB")).copy()
    pixels[~mask] = (124, 124, 124)
    return Image.fromarray(pixels).crop(box)


class MobileNetV2Encoder:
    """ImageNet-pretrained MobileNetV2 pooled features; no classifier training."""

    name = "mobilenetv2"

    def __init__(self, weights_path: Path = MOBILENET_FILE, device: str = "cpu"):
        import torch
        from torchvision.models import mobilenet_v2
        from torchvision.models import MobileNet_V2_Weights

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Missing pretrained MobileNetV2 weights at {weights_path}. "
                "Run the download command in experiments/hybrid_pipeline/README.md."
            )
        if weights_path.resolve() == MOBILENET_FILE.resolve():
            digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
            if digest != MOBILENET_SHA256:
                raise ValueError("The local MobileNetV2 checkpoint does not match the official TorchVision file hash.")
        self.torch = torch
        self.device = torch.device(device)
        model = mobilenet_v2(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.classifier = torch.nn.Identity()
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()

    def encode(self, images: list[Image.Image], batch_size=32) -> np.ndarray:
        if not images:
            return np.empty((0, 1280), dtype=np.float32)
        outputs = []
        with self.torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                batch = self.torch.stack([
                    self.transform(image.convert("RGB")) for image in images[offset:offset + batch_size]
                ]).to(self.device)
                features = self.model(batch)
                features = self.torch.nn.functional.normalize(features, dim=1)
                outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)


class DinoV3Encoder:
    """Optional official DINOv3 ViT-S/16+ backbone loaded from user-provided files."""

    name = "dinov3"

    def __init__(self, repo: Path, weights: Path, device: str = "cpu"):
        import torch
        from torchvision import transforms

        if not repo.is_dir() or not (repo / "hubconf.py").is_file():
            raise FileNotFoundError("Set DINOV3_REPO to a local clone of facebookresearch/dinov3.")
        if not weights.is_file():
            raise FileNotFoundError("Set DINOV3_WEIGHTS to the gated ViT-S/16+ checkpoint you obtained from Meta.")
        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.hub.load(str(repo), "dinov3_vits16plus", source="local", weights=str(weights))
        self.model = self.model.eval().requires_grad_(False).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def encode(self, images: list[Image.Image], batch_size=16) -> np.ndarray:
        outputs = []
        with self.torch.inference_mode():
            for offset in range(0, len(images), batch_size):
                batch = self.torch.stack([
                    self.transform(image.convert("RGB")) for image in images[offset:offset + batch_size]
                ]).to(self.device)
                features = self.model.forward_features(batch)
                if isinstance(features, dict):
                    features = features.get("x_norm_clstoken", features.get("x_norm_patchtokens"))
                if features is None:
                    raise RuntimeError("DINOv3 forward_features did not return an embedding tensor.")
                if features.ndim == 3:
                    features = features.mean(dim=1)
                features = self.torch.nn.functional.normalize(features, dim=1)
                outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 384), dtype=np.float32)


def product_reference_paths(product: dict) -> list[Path]:
    references = product.get("references") or [{"image_path": product["image_path"]}]
    paths = [Path(item["image_path"]) for item in references]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing catalogue reference image(s): " + ", ".join(missing[:5]))
    return paths


def build_reference_index(products: list[dict], encoders: dict[str, Any]) -> dict:
    index = {product["sku"]: {name: [] for name in encoders} for product in products}
    pending = []
    for product in products:
        sku = product["sku"]
        for path in product_reference_paths(product):
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            pending.append((sku, path, digest, image))
    for name, encoder in encoders.items():
        vectors = encoder.encode([item[3] for item in pending])
        for (sku, path, digest, _), vector in zip(pending, vectors):
            index[sku][name].append({"feature": unit_vector(vector), "image": str(path), "sha256": digest})
    return index


def scores_for_features(features: dict[str, np.ndarray], index: dict,
                        branch_weights=None) -> tuple[dict, dict, dict]:
    branch_scores = {}
    best_references = {}
    for name, vector in features.items():
        query = unit_vector(vector)
        branch_scores[name] = {}
        best_references[name] = {}
        for sku, branches in index.items():
            views = branches.get(name, [])
            if not views:
                continue
            centroid = unit_vector(np.mean([view["feature"] for view in views], axis=0))
            branch_scores[name][sku] = float(np.dot(query, centroid))
            _, reference = max(((float(np.dot(query, view["feature"])), view) for view in views),
                               key=lambda pair: pair[0])
            best_references[name][sku] = {"path": reference["image"], "sha256": reference["sha256"]}
    return fuse_sku_scores(branch_scores, branch_weights), branch_scores, best_references


def scores_for_reference_views(features: dict[str, np.ndarray], index: dict,
                               branch_weights=None) -> tuple[dict, dict, dict]:
    """Score each SKU by its best individual reference view, matching production aggregation."""
    branch_scores = {}
    best_references = {}
    for name, vector in features.items():
        query = unit_vector(vector)
        branch_scores[name] = {}
        best_references[name] = {}
        for sku, branches in index.items():
            views = branches.get(name, [])
            if not views:
                continue
            scores = [float(np.dot(query, unit_vector(view["feature"]))) for view in views]
            winner = int(np.argmax(scores))
            branch_scores[name][sku] = scores[winner]
            best_references[name][sku] = {
                "path": views[winner]["image"],
                "sha256": views[winner]["sha256"],
                "similarity": scores[winner],
            }
    fused = fuse_sku_scores(branch_scores, branch_weights)
    return fused, branch_scores, best_references


def match_crop(crop: Image.Image, encoders: dict[str, Any], index: dict,
               minimum=DEFAULT_SIMILARITY, margin=DEFAULT_MARGIN,
               branch_weights=None) -> dict:
    branch_features = {}
    for name, encoder in encoders.items():
        branch_features[name] = encoder.encode([crop])[0]
    fused, branch_scores, best_references = scores_for_features(branch_features, index,
                                                                 branch_weights)
    decision = abstaining_decision(fused, minimum, margin)
    decision["branch_scores"] = branch_scores
    decision["best_reference_by_branch"] = best_references
    decision["embedding_mode"] = "+".join(encoders)
    return decision


def count_by_sku(detections: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        sku = item.get("sku") if item.get("status") == "recognized" else None
        if sku:
            counts[sku] = counts.get(sku, 0) + 1
    return dict(sorted(counts.items()))


def apply_manual_corrections(detections: list[dict], corrections: list[dict]):
    """Apply session-only SKU/count edits without changing the proposed predictions."""
    if len(detections) != len(corrections):
        raise ValueError("Each detection must have exactly one manual review row.")
    confirmed = []
    counts: dict[str, int] = {}
    for detection, correction in zip(detections, corrections):
        quantity = int(correction.get("quantity") or 0)
        if quantity < 0:
            raise ValueError("Manual quantities cannot be negative.")
        selected_sku = correction.get("sku") or "__unknown__"
        item = dict(detection)
        item["predicted_sku"] = (detection.get("sku")
                                 if detection.get("status") == "recognized" else None)
        item["confirmed_quantity"] = quantity
        item["confirmed_sku"] = selected_sku if quantity else "__unknown__"
        item["sku"] = selected_sku if quantity and selected_sku != "__unknown__" else None
        item["status"] = "recognized" if item["sku"] else "unknown"
        confirmed.append(item)
        if item["sku"]:
            counts[item["sku"]] = counts.get(item["sku"], 0) + quantity
    return confirmed, dict(sorted(counts.items()))


def render_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default(size=max(14, output.width // 100))
    for number, item in enumerate(detections, 1):
        left, top, right, bottom = [int(round(value)) for value in item["box"]]
        color = (0, 165, 110) if item.get("sku") else (230, 155, 20)
        draw.rectangle((left, top, right, bottom), outline=color, width=max(2, output.width // 500))
        identity = item.get("sku") or "UNKNOWN"
        similarity = item.get("similarity")
        label = f"{number} {identity}"
        if similarity is not None:
            label += f" {similarity:.2f}"
        label_width = draw.textlength(label, font=font)
        y = max(0, top - font.size - 5)
        draw.rectangle((left, y, min(output.width, left + label_width + 8), y + font.size + 4), fill=color)
        draw.text((left + 4, y + 2), label, fill="black", font=font)
    return output


def load_catalog(business_id: int = 1) -> list[dict]:
    """Read the existing registry only; this module never writes to its database."""
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import db
    products = db.get_business_products(business_id)
    if not products:
        raise RuntimeError(f"No registered products found for business {business_id}.")
    return products

