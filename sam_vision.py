"""MobileSAM object masks followed by learned reference-image matching."""

import base64
import hashlib
import os
import tempfile
import zlib
from pathlib import Path
from threading import RLock
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from images import product_references
from local_features import (MATCHER_VERSION, LabelVerifier, resolve_ambiguity,
                            resolve_low_score_geometry, should_verify_low_score_geometry)
from model_assets import ENCODER_FILE, SAM_FILE

DEFAULT_SIMILARITY = 0.80
DEFAULT_MARGIN = 0.05
MAX_IMAGE_SIDE = 1280
MIN_MASK_FRACTION = 0.002
REFERENCE_CACHE_VERSION = "mobile-sam12-resnet18-v3-bounded-reference"
RECOGNITION_VERSION = REFERENCE_CACHE_VERSION + ":" + MATCHER_VERSION


def mask_border_coverage(mask):
    """Use a narrow band because SAM can leave a one-pixel gap at the frame."""
    band = min(min(mask.shape), max(2, round(min(mask.shape) * 0.01)))
    return [float(edge.mean()) for edge in (
        mask[:band].any(axis=0), mask[-band:].any(axis=0),
        mask[:, :band].any(axis=1), mask[:, -band:].any(axis=1),
    )]


def encode_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask.reshape(-1)).tobytes()
    return {"encoding": "zlib-packbits", "shape": list(mask.shape),
            "data": base64.b64encode(zlib.compress(packed)).decode("ascii")}


def decode_mask(encoded, image_size=None):
    if encoded["encoding"] != "zlib-packbits":
        raise ValueError("Unsupported segmentation mask encoding.")
    height, width = encoded["shape"]
    packed = np.frombuffer(zlib.decompress(base64.b64decode(encoded["data"])), dtype=np.uint8)
    mask = np.unpackbits(packed, count=height * width).reshape(height, width).astype(bool)
    if image_size is not None and (width, height) != tuple(image_size):
        mask = cv2.resize(mask.astype(np.uint8), tuple(image_size), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def masked_crop(image, mask):
    box = mask_box(mask)
    if box is None:
        raise ValueError("Cannot match an empty mask.")
    pixels = np.asarray(image.convert("RGB")).copy()
    pixels[~mask] = (124, 124, 124)
    return Image.fromarray(pixels).crop(box)


def classify_scores(scores, products, minimum=DEFAULT_SIMILARITY, margin=DEFAULT_MARGIN):
    """Cosine similarity and runner-up gap are separate, uncalibrated checks."""
    if not products or len(scores) != len(products) or not np.isfinite(scores).all():
        raise ValueError("Reference scores must be finite and match the catalogue.")
    order = np.argsort(-np.asarray(scores), kind="stable")
    first = int(order[0])
    score = float(scores[first])
    gap = float(score - scores[int(order[1])]) if len(order) > 1 else None
    accepted = score >= minimum and (gap is None or gap >= margin)
    return {
        "sku": products[first]["sku"] if accepted else None,
        "name": products[first]["name"] if accepted else "Uncertain region",
        "status": "recognized" if accepted else "uncertain",
        "similarity": score,
        "confidence": score,
        "score_type": "cosine_similarity",
        "match_margin": gap,
        "reason": "reference_match" if accepted else (
            "below_similarity_threshold" if score < minimum else "ambiguous_references"
        ),
        "matches": [{"sku": products[int(index)]["sku"], "similarity": float(scores[int(index)])}
                    for index in order[:3]],
    }


def aggregate_reference_scores(view_scores, product_indices, product_count):
    """Compare SKUs by their best view, not by the number of uploaded photos."""
    view_scores = np.asarray(view_scores)
    if (view_scores.ndim != 2 or view_scores.shape[1] != len(product_indices)
            or not np.isfinite(view_scores).all()
            or set(product_indices) != set(range(product_count))):
        raise ValueError("Reference views must cover every product with finite scores.")
    scores = np.empty((view_scores.shape[0], product_count), dtype=view_scores.dtype)
    winners = np.empty(scores.shape, dtype=int)
    for product_index in range(product_count):
        columns = np.flatnonzero(np.asarray(product_indices) == product_index)
        best = np.argmax(view_scores[:, columns], axis=1)
        winners[:, product_index] = columns[best]
        scores[:, product_index] = view_scores[np.arange(len(view_scores)), columns[best]]
    return scores, winners


def prune_instances(records, masks):
    """Suppress duplicate/nested masks without using distances between box centers."""
    visible = [index for index, item in enumerate(records) if item["status"] != "rejected"]
    areas = [int(mask.sum()) for mask in masks]
    intersections = {}

    def overlap(first, second):
        key = tuple(sorted((first, second)))
        if key not in intersections:
            intersections[key] = int(np.count_nonzero(masks[first] & masks[second]))
        return intersections[key]

    # A mask enclosing several independently matched, disjoint instances cannot count as another unit.
    for parent in visible:
        if records[parent]["status"] != "recognized":
            continue
        children = [child for child in visible if child != parent
                    and records[child]["status"] == "recognized"
                    and 0.15 * areas[parent] < areas[child] < 0.8 * areas[parent]
                    and overlap(parent, child) / areas[child] > 0.90]
        selected = []
        for child in sorted(children, key=lambda index: areas[index], reverse=True):
            if all(overlap(child, other) / min(areas[child], areas[other]) < 0.2 for other in selected):
                selected.append(child)
        if len(selected) >= 2:
            union = np.logical_or.reduce([masks[index] for index in selected])
            if np.count_nonzero(union & masks[parent]) / areas[parent] > 0.65:
                records[parent].update(status="rejected", reason="mask_contains_multiple_instances", sku=None)

    kept = []
    for index in sorted(visible, key=lambda value: (
        records[value]["status"] == "recognized", areas[value], records[value]["mask_quality"]
    ), reverse=True):
        item = records[index]
        if item["status"] == "rejected":
            continue
        for other in kept:
            intersection = overlap(index, other)
            iou = intersection / (areas[index] + areas[other] - intersection)
            nested = intersection / min(areas[index], areas[other]) > 0.95
            if iou > 0.80 or (nested and records[other]["status"] == "recognized"):
                item.update(status="rejected", reason="duplicate_or_nested_mask", sku=None)
                break
        else:
            kept.append(index)
    return [records[index].copy() for index in kept]


def render_detections(image, detections, spotlight=False):
    """Use real instance masks; disclose rectangles when a legacy result has no mask."""
    image = image.convert("RGB")
    pixels = np.asarray(image).copy()
    recognized = np.zeros((image.height, image.width), dtype=bool)
    decoded = []
    rectangular = False
    for item in detections:
        mask = decode_mask(item["mask"], image.size) if item.get("mask") else None
        decoded.append(mask)
        if item.get("sku") and item.get("status", "recognized") == "recognized":
            if mask is not None:
                recognized |= mask
            else:
                rectangular = True
                left, top, right, bottom = [int(value) for value in item["box"]]
                recognized[max(0, top):min(image.height, bottom), max(0, left):min(image.width, right)] = True
    if spotlight:
        pixels[~recognized] = (pixels[~recognized] * 0.30).astype(np.uint8)
    thickness = max(2, image.width // 400)
    rendering_order = sorted(zip(detections, decoded), key=lambda pair: bool(pair[0].get("sku")))
    for item, mask in rendering_order:
        color = (0, 168, 120) if item.get("sku") else (245, 196, 48)
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pixels, contours, -1, color, thickness)
    annotated = Image.fromarray(pixels)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(12, image.width // 70))
    for item, mask in rendering_order:
        left, top, right, bottom = item["box"]
        color = "#00A878" if item.get("sku") else "#F5C430"
        if mask is None:
            draw.rectangle((left, top, right, bottom), outline=color, width=thickness)
        label = f'{item["name"]} ({item["sku"]})' if item.get("sku") else "Unassigned region"
        label_width = draw.textlength(label, font=font)
        x = min(max(0, left), max(0, image.width - label_width - 6))
        y = max(0, top - font.size - 8)
        draw.rectangle((x, y, min(image.width, x + label_width + 6), y + font.size + 6), fill=color)
        draw.text((x + 3, y + 2), label, fill="black", font=font)
    return annotated, "bounding_box" if rectangular else "instance_mask"


class ReferenceEncoder:
    def __init__(self):
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        self.torch = torch
        self.model = resnet18(weights=None)
        self.model.load_state_dict(torch.load(ENCODER_FILE, map_location="cpu", weights_only=True))
        self.model.fc = torch.nn.Identity()
        self.model.eval().requires_grad_(False)
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms(resize_size=224, crop_size=224)

    def encode(self, images):
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(images), 16):
                batch = self.torch.stack([
                    self.transform(ImageOps.pad(image.convert("RGB"), (224, 224), color=(124, 124, 124)))
                    for image in images[start:start + 16]
                ])
                features = self.torch.nn.functional.normalize(self.model(batch), dim=1)
                batches.append(features.cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, 512), dtype=np.float32)


class SAMRecognizer:
    def __init__(self):
        import torch
        from ultralytics import SAM
        from ultralytics.models.sam.predict import Predictor

        missing = [path.name for path in (SAM_FILE, ENCODER_FILE) if not path.is_file()]
        if missing:
            raise RuntimeError("Missing model files: " + ", ".join(missing)
                               + ". Run .\\.venv\\Scripts\\python.exe model_assets.py first.")

        class ProposalPredictor(Predictor):
            def generate(self, im, *args, **kwargs):
                kwargs.update(points_stride=12, points_batch_size=16, crop_n_layers=0,
                              conf_thres=0.80, stability_score_thresh=0.92)
                return super().generate(im, *args, **kwargs)

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.segmenter = SAM(str(SAM_FILE))
        self.predictor_class = ProposalPredictor
        self.encoder = ReferenceEncoder()
        self.label_verifier = LabelVerifier()
        self.references = {}
        self.reference_dir = SAM_FILE.parent / "reference_cache"
        self.lock = RLock()

    def segment(self, image):
        results = self.segmenter.predict(
            image, predictor=self.predictor_class, device="cpu", imgsz=1024,
            conf=0.0, iou=0.7, verbose=False, save=False,
        )
        if not results or results[0].masks is None:
            return [], []
        masks = results[0].masks.data.cpu().numpy().astype(bool)
        qualities = results[0].boxes.conf.cpu().tolist()
        return list(masks), qualities

    @staticmethod
    def rejection(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 2 or not mask.size:
            raise ValueError("Segmentation masks must be nonempty two-dimensional arrays.")
        fraction = float(mask.mean())
        if fraction < MIN_MASK_FRACTION:
            return "mask_too_small"
        touched = sum(bool(edge.any()) for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]))
        broad_edges = sum(coverage >= 0.10 for coverage in mask_border_coverage(mask))
        if fraction > 0.92 or (touched >= 3 and fraction > 0.5) or broad_edges >= 3:
            return "background_or_scene_mask"
        return None

    @classmethod
    def reference_rejection(cls, mask):
        reason = cls.rejection(mask)
        if reason:
            return reason
        # Registration photos require the whole product, unlike cropped inventory scenes.
        if any(coverage >= 0.10 for coverage in mask_border_coverage(mask)):
            return "reference_touches_frame"
        return None

    @classmethod
    def select_reference_mask(cls, masks, qualities):
        if len(masks) != len(qualities):
            raise ValueError("Reference masks and qualities must align.")
        valid = [index for index, mask in enumerate(masks)
                 if np.isfinite(qualities[index]) and qualities[index] > 0
                 and not cls.reference_rejection(mask)]
        if not valid:
            raise ValueError("No isolated product mask found.")
        # Prefer a full product over its label/cap, after excluding frame-connected scene regions.
        return max(valid, key=lambda index: int(masks[index].sum()) * qualities[index])

    def reference(self, product):
        path = Path(product["image_path"])
        stat = path.stat()
        key = (REFERENCE_CACHE_VERSION, str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        if key not in self.references:
            fingerprint = hashlib.sha256(REFERENCE_CACHE_VERSION.encode("ascii") + path.read_bytes()).hexdigest()
            cache_file = self.reference_dir / f"{fingerprint}.npz"
            if cache_file.is_file():
                with np.load(cache_file, allow_pickle=False) as cached:
                    feature, pixels = cached["feature"], cached["crop"]
                    if feature.shape != (512,) or not np.isfinite(feature).all() or pixels.ndim != 3 or pixels.shape[2] != 3:
                        raise ValueError(f"Invalid cached reference for {product['sku']}.")
                    self.references[key] = (feature, Image.fromarray(pixels))
                return self.references[key]
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            masks, qualities = self.segment(image)
            try:
                index = self.select_reference_mask(masks, qualities)
            except ValueError as error:
                raise ValueError(
                    f'No isolated product mask found in reference {path.name} for {product["sku"]}. '
                    'Keep the full product inside the frame with background visible on every side.'
                ) from error
            crop = masked_crop(image, masks[index])
            self.references[key] = (self.encoder.encode([crop])[0], crop)
            self.reference_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.reference_dir, suffix=".npz", delete=False) as temporary:
                    temporary_path = Path(temporary.name)
                    np.savez_compressed(temporary, feature=self.references[key][0], crop=np.asarray(crop))
                temporary_path.replace(cache_file)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        return self.references[key]

    def recognize(self, photo, products, minimum=DEFAULT_SIMILARITY, margin=DEFAULT_MARGIN, progress=None):
        if not products:
            raise ValueError("Register at least one product first.")
        with self.lock:
            start = perf_counter()
            self.last_timings = {}
            reference_views = []
            product_indices = []
            for index, product in enumerate(products):
                for view, reference in enumerate(product_references(product)):
                    reference_views.append({**product, "image_path": reference["image_path"],
                                            "reference_id": reference.get("id"), "view": view + 1})
                    product_indices.append(index)
            reference_data = []
            for index, reference in enumerate(reference_views):
                if progress:
                    progress({"stage": "references", "completed": index, "total": len(reference_views),
                              "sku": reference["sku"], "view": reference["view"]})
                reference_data.append(self.reference(reference))
            self.last_timings["references_seconds"] = round(perf_counter() - start, 3)
            if progress:
                progress({"stage": "references", "completed": len(reference_views), "total": len(reference_views)})
                progress({"stage": "photo"})
            vectors = np.stack([item[0] for item in reference_data])
            image = photo.convert("RGB")
            working = image.copy()
            working.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            stage_start = perf_counter()
            masks, qualities = self.segment(working)
            self.last_timings["photo_segmentation_seconds"] = round(perf_counter() - stage_start, 3)
            stage_start = perf_counter()
            if progress:
                progress({"stage": "matching"})
            records, crops, valid_indices = [], [], []
            scale_x, scale_y = image.width / working.width, image.height / working.height
            for index, (mask, quality) in enumerate(zip(masks, qualities)):
                reason = self.rejection(mask)
                box = mask_box(mask)
                if box is None:
                    continue
                record = {"sku": None, "name": "Uncertain region", "status": "rejected" if reason else "uncertain",
                          "reason": reason, "method": "mobile_sam_resnet18", "mask_quality": float(quality),
                          "box": [float(value * (scale_x if offset % 2 == 0 else scale_y)) for offset, value in enumerate(box)],
                          "box_area_ratio": (box[2] - box[0]) * (box[3] - box[1]) / (working.width * working.height),
                          "mask_area_ratio": float(mask.mean()), "border_coverage": mask_border_coverage(mask),
                          "mask": encode_mask(mask), "similarity_threshold": minimum, "margin_threshold": margin,
                          "score_type": "cosine_similarity", "confidence": 0.0}
                records.append(record)
                if not reason:
                    crops.append(masked_crop(working, mask))
                    valid_indices.append(len(records) - 1)
            view_scores = self.encoder.encode(crops) @ vectors.T
            scores, winners = aggregate_reference_scores(view_scores, product_indices, len(products))
            sku_indices = {product["sku"]: index for index, product in enumerate(products)}
            all_reference_crops = [(reference, reference_data[view][1])
                                   for view, reference in enumerate(reference_views)]
            for row, (index, similarities) in enumerate(zip(valid_indices, scores)):
                records[index].update(classify_scores(similarities, products, minimum, margin))
                for match in records[index]["matches"]:
                    reference = reference_views[winners[row, sku_indices[match["sku"]]]]
                    match.update(reference_id=reference["reference_id"],
                                 reference_image=reference["image_path"], reference_view=reference["view"])
                if records[index]["reason"] == "ambiguous_references":
                    # Include close fourth/later SKUs as well as the displayed top three.
                    candidates = set(np.flatnonzero(similarities >= similarities.max() - margin))
                    candidates.update(sku_indices[match["sku"]] for match in records[index]["matches"])
                    reference_crops = [(reference, reference_data[view][1])
                                       for view, reference in enumerate(reference_views)
                                       if product_indices[view] in candidates]
                    evidence = self.label_verifier.compare(crops[row], reference_crops)
                    resolve_ambiguity(records[index], evidence, [products[value]["sku"] for value in candidates])
                    if records[index]["sku"]:
                        records[index]["name"] = products[sku_indices[records[index]["sku"]]]["name"]
                        records[index]["method"] = "mobile_sam_resnet18_sift"
                elif should_verify_low_score_geometry(records[index], margin):
                    evidence = self.label_verifier.compare(crops[row], all_reference_crops)
                    resolve_low_score_geometry(records[index], evidence, sku_indices)
                    if records[index]["sku"]:
                        records[index]["name"] = products[sku_indices[records[index]["sku"]]]["name"]
                        records[index]["method"] = "mobile_sam_resnet18_sift"
            # Each record corresponds to one original SAM mask, including rejected masks.
            aligned_masks = [decode_mask(item["mask"]) for item in records]
            detections = prune_instances(records, aligned_masks)
            self.last_timings["matching_seconds"] = round(perf_counter() - stage_start, 3)
            stage_start = perf_counter()
            if progress:
                progress({"stage": "rendering"})
            annotated, _ = render_detections(image, detections)
            columns = min(3, len(reference_data))
            board = Image.new("RGB", (columns * 320, ((len(reference_data) + columns - 1) // columns) * 320), "white")
            draw = ImageDraw.Draw(board)
            for index, (_, crop) in enumerate(reference_data):
                tile = ImageOps.contain(crop, (288, 272))
                x, y = (index % columns) * 320, (index // columns) * 320
                board.paste(tile, (x + (320 - tile.width) // 2, y + 40 + (272 - tile.height) // 2))
                reference = reference_views[index]
                label = f'{reference["sku"]} - view {reference["view"]}'
                size = 16
                font = ImageFont.load_default(size=size)
                while draw.textlength(label, font=font) > 304 and size > 8:
                    size -= 1
                    font = ImageFont.load_default(size=size)
                draw.text((x + 8, y + 8), label, fill="black", font=font)
            self.last_timings["rendering_seconds"] = round(perf_counter() - stage_start, 3)
            self.last_timings["total_seconds"] = round(perf_counter() - start, 3)
            return detections, annotated, board, records
