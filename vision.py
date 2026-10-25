"""One-shot product detection with YOLOE visual prompts.

Reference images should contain one tightly cropped product each.
This is an experimental model, not a guaranteed SKU recognition system.
"""

from math import ceil
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sam_vision import SAMRecognizer, encode_mask, render_detections

MODEL_NAME = "yoloe-26s-seg.pt"
TILE = 320
ULTRALYTICS_CONFIG_DIR = Path(__file__).resolve().parent / "data" / "ultralytics"
IOU_DUPLICATE_THRESHOLD = 0.60
CONTAINMENT_DUPLICATE_THRESHOLD = 0.95
REFERENCE_FOREGROUND_THRESHOLD = 35
REFERENCE_CROP_PADDING = 12
PROPOSAL_FOREGROUND_THRESHOLD = 28
PROPOSAL_CROP_PADDING = 16
MIN_PROPOSAL_AREA_RATIO = 0.01
MAX_PROPOSAL_AREA_RATIO = 0.85
PROPOSAL_MATCH_THRESHOLD = 0.32
MIN_SPLIT_GAP_WIDTH = 18
MIN_SPLIT_SIDE_WIDTH = 40


def configure_ultralytics_environment():
    """Keep Ultralytics settings inside this project folder."""
    ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR.resolve()))


def load_model(engine="sam"):
    """Load SAM/reference matching by default; keep YOLOE as a comparison backend."""
    configure_ultralytics_environment()
    if engine == "sam":
        return SAMRecognizer()
    if engine != "yoloe":
        raise ValueError("Unknown recognition engine.")
    try:
        from ultralytics import YOLOE
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Install project dependencies first with "
            "`python -m pip install -r requirements.txt` inside the virtual environment."
        ) from error

    return YOLOE(str(Path(__file__).resolve().parent / MODEL_NAME))


def crop_reference_to_foreground(reference):
    """Crop reference photos around the product by comparing against the border."""
    image = reference.convert("RGB")
    array = np.asarray(image).astype(np.int16)
    height, width = array.shape[:2]
    border = np.concatenate(
        [
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(array - background, axis=2)
    mask = distance > REFERENCE_FOREGROUND_THRESHOLD
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image

    left = max(0, int(xs.min()) - REFERENCE_CROP_PADDING)
    top = max(0, int(ys.min()) - REFERENCE_CROP_PADDING)
    right = min(width, int(xs.max()) + 1 + REFERENCE_CROP_PADDING)
    bottom = min(height, int(ys.max()) + 1 + REFERENCE_CROP_PADDING)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def foreground_mask(image, threshold):
    array = np.asarray(image.convert("RGB")).astype(np.int16)
    border = np.concatenate(
        [
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ],
        axis=0,
    )
    background = np.median(border, axis=0)
    distance = np.linalg.norm(array - background, axis=2)
    mask = (distance > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def padded_box(x, y, w, h, image_size):
    width, height = image_size
    left = max(0, int(x) - PROPOSAL_CROP_PADDING)
    top = max(0, int(y) - PROPOSAL_CROP_PADDING)
    right = min(width, int(x + w) + PROPOSAL_CROP_PADDING)
    bottom = min(height, int(y + h) + PROPOSAL_CROP_PADDING)
    return [float(left), float(top), float(right), float(bottom)]


def split_box_on_vertical_gaps(mask, box, image_size):
    left, top, right, bottom = [int(round(value)) for value in box]
    crop = mask[top:bottom, left:right]
    if crop.size == 0:
        return [box]

    projection = np.count_nonzero(crop, axis=0)
    low_threshold = max(2, int(crop.shape[0] * 0.03))
    low_columns = projection <= low_threshold
    gaps = []
    start = None
    for index, is_low in enumerate(low_columns):
        if is_low and start is None:
            start = index
        elif not is_low and start is not None:
            if index - start >= MIN_SPLIT_GAP_WIDTH:
                gaps.append((start, index))
            start = None
    if start is not None and len(low_columns) - start >= MIN_SPLIT_GAP_WIDTH:
        gaps.append((start, len(low_columns)))

    candidates = []
    for gap_start, gap_end in gaps:
        split_at = (gap_start + gap_end) // 2
        if split_at < MIN_SPLIT_SIDE_WIDTH:
            continue
        if crop.shape[1] - split_at < MIN_SPLIT_SIDE_WIDTH:
            continue
        left_crop = crop[:, :split_at]
        right_crop = crop[:, split_at:]
        if np.count_nonzero(left_crop) == 0 or np.count_nonzero(right_crop) == 0:
            continue
        candidates = [
            [float(left), float(top), float(left + split_at), float(bottom)],
            [float(left + split_at), float(top), float(right), float(bottom)],
        ]
        break

    if not candidates:
        return [box]

    return [
        padded_box(
            candidate[0], candidate[1],
            candidate[2] - candidate[0], candidate[3] - candidate[1],
            image_size,
        )
        for candidate in candidates
    ]


def foreground_proposals(image):
    width, height = image.size
    image_area = width * height
    mask = foreground_mask(image, PROPOSAL_FOREGROUND_THRESHOLD)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []

    for component in range(1, count):
        x, y, w, h, area = stats[component]
        area_ratio = area / image_area if image_area else 0
        if area_ratio < MIN_PROPOSAL_AREA_RATIO:
            continue
        if area_ratio > MAX_PROPOSAL_AREA_RATIO:
            continue
        box = padded_box(x, y, w, h, image.size)
        boxes.extend(split_box_on_vertical_gaps(mask, box, image.size))
    return boxes


def crop_box(image, box):
    return image.crop(tuple(int(round(value)) for value in box))


def cv_gray(image):
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)


def orb_similarity(first, second):
    detector = cv2.ORB_create(nfeatures=800)
    first_keypoints, first_descriptors = detector.detectAndCompute(cv_gray(first), None)
    second_keypoints, second_descriptors = detector.detectAndCompute(cv_gray(second), None)
    if first_descriptors is None or second_descriptors is None:
        return 0.0
    if len(first_keypoints) < 4 or len(second_keypoints) < 4:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(first_descriptors, second_descriptors)
    if not matches:
        return 0.0
    good_matches = [match for match in matches if match.distance <= 64]
    denominator = max(12, min(len(first_keypoints), len(second_keypoints)))
    return min(1.0, len(good_matches) / denominator)


def color_similarity(first, second):
    first_hsv = cv2.cvtColor(np.asarray(first.convert("RGB")), cv2.COLOR_RGB2HSV)
    second_hsv = cv2.cvtColor(np.asarray(second.convert("RGB")), cv2.COLOR_RGB2HSV)
    first_hist = cv2.calcHist([first_hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    second_hist = cv2.calcHist([second_hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(first_hist, first_hist)
    cv2.normalize(second_hist, second_hist)
    score = cv2.compareHist(first_hist, second_hist, cv2.HISTCMP_CORREL)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def aspect_similarity(first, second):
    first_ratio = first.width / first.height if first.height else 0
    second_ratio = second.width / second.height if second.height else 0
    if first_ratio <= 0 or second_ratio <= 0:
        return 0.0
    return min(first_ratio, second_ratio) / max(first_ratio, second_ratio)


def crop_similarity(first, second):
    orb_score = orb_similarity(first, second)
    color_score = color_similarity(first, second)
    aspect_score = aspect_similarity(first, second)
    return 0.65 * orb_score + 0.25 * color_score + 0.10 * aspect_score


def reference_crop(product):
    with Image.open(Path(product["image_path"])) as source:
        return crop_reference_to_foreground(source)


def build_reference_board(products):
    """Make one reference image with one numbered visual prompt per SKU."""
    if not products:
        raise ValueError("Add at least one product to the catalogue.")

    columns = min(3, len(products))
    rows = ceil(len(products) / columns)
    board = Image.new("RGB", (columns * TILE, rows * TILE), "white")
    boxes = []

    for index, product in enumerate(products):
        image_path = Path(product["image_path"])
        with Image.open(image_path) as source:
            reference = crop_reference_to_foreground(source)
        reference.thumbnail((TILE - 32, TILE - 32))

        col, row = index % columns, index // columns
        left = col * TILE + (TILE - reference.width) // 2
        top = row * TILE + (TILE - reference.height) // 2
        board.paste(reference, (left, top))
        # This box assumes the uploaded reference is closely cropped to the product.
        boxes.append([left, top, left + reference.width, top + reference.height])

    return board, np.asarray(boxes, dtype=np.float32)


def build_single_reference_prompt(product):
    """Make a one-product visual prompt board."""
    board, boxes = build_reference_board([product])
    prompts = {
        "bboxes": boxes,
        "cls": np.array([0], dtype=np.int64),
    }
    return board, prompts


def box_area(box):
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    return width * height


def box_intersection(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def duplicate_overlap(first, second):
    """Only suppress overlapping boxes; adjacent tall products remain separate."""
    intersection = box_intersection(first, second)
    if intersection <= 0:
        return False

    first_area = box_area(first)
    second_area = box_area(second)
    union = first_area + second_area - intersection
    iou = intersection / union if union else 0.0
    smaller_area = min(first_area, second_area)
    containment = intersection / smaller_area if smaller_area else 0.0
    return (
        iou >= IOU_DUPLICATE_THRESHOLD
        or containment >= CONTAINMENT_DUPLICATE_THRESHOLD
    )


def better_duplicate(first, second):
    """Use the higher-scoring box, as in standard confidence-based NMS."""
    return first if first["confidence"] >= second["confidence"] else second


def plausible_inventory_box(detection, image_size, product_count, single_sku_mode=False):
    """Validate coordinates without rejecting valid close-up objects by area."""
    box = detection["box"]
    return bool(np.isfinite(box).all() and 0 <= box[0] < box[2] <= image_size[0]
                and 0 <= box[1] < box[3] <= image_size[1])


def suppress_duplicate_detections(detections):
    """Merge highly overlapping detections before counting."""
    kept = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate_index = None
        for index, existing in enumerate(kept):
            if duplicate_overlap(existing["box"], detection["box"]):
                duplicate_index = index
                break

        if duplicate_index is None:
            kept.append(detection)
        elif kept[duplicate_index]["sku"] != detection["sku"]:
            kept[duplicate_index] = dict(
                kept[duplicate_index], sku=None, name="Uncertain product",
                status="uncertain", reason="conflicting_prompt_labels",
            )
        else:
            kept[duplicate_index] = better_duplicate(kept[duplicate_index], detection)

    return kept


def run_product_prompt(model, image, product, confidence):
    try:
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    except ImportError as error:
        raise RuntimeError(
            "YOLOE visual prompting is unavailable because Ultralytics is not installed "
            "or the installed version does not include YOLOE support."
        ) from error

    board, prompts = build_single_reference_prompt(product)
    results = model.predict(
        source=image,
        refer_image=board,
        visual_prompts=prompts,
        predictor=YOLOEVPSegPredictor,
        imgsz=960,
        conf=confidence,
        agnostic_nms=False,
        retina_masks=True,
        verbose=False,
    )

    detections = []
    if not results or results[0].boxes is None:
        return detections

    for index, result_box in enumerate(results[0].boxes):
        class_id = int(result_box.cls.item())
        xyxy = [float(x) for x in result_box.xyxy[0].tolist()]
        score = float(result_box.conf.item())
        detection = {
                "sku": product["sku"] if class_id == 0 else None,
                "name": product["name"] if class_id == 0 else "Uncertain region",
                "confidence": score,
                "box": xyxy,
                "method": "yoloe_prompt",
                "status": "recognized" if class_id == 0 else "uncertain",
                "reason": "visual_prompt" if class_id == 0 else "unexpected_class_id",
                "score_type": "detector_confidence",
            }
        if results[0].masks is not None:
            detection["mask"] = encode_mask(results[0].masks.data[index].cpu().numpy() > 0)
        detections.append(detection)

    return detections


def run_proposal_match(image, products):
    proposals = foreground_proposals(image)
    references = [(product, reference_crop(product)) for product in products]
    detections = []

    for box in proposals:
        image_area = image.size[0] * image.size[1]
        area_ratio = box_area(box) / image_area if image_area else 0
        if area_ratio > MAX_PROPOSAL_AREA_RATIO:
            continue
        proposal_crop = crop_box(image, box)
        best_product = None
        best_score = 0.0
        for product, reference in references:
            score = crop_similarity(proposal_crop, reference)
            if score > best_score:
                best_score = score
                best_product = product

        if best_product and best_score >= PROPOSAL_MATCH_THRESHOLD:
            detections.append(
                {
                    "sku": best_product["sku"],
                    "name": best_product["name"],
                    "confidence": float(best_score),
                    "box": box,
                    "method": "proposal_match",
                    "raw": True,
                    "box_area_ratio": area_ratio,
                }
            )

    return detections


def detect_products(model, photo, products, confidence=0.80, min_margin=0.05, progress=None):
    """Return detections, annotated photo, and reference board.

    Each prediction's class index maps to the same index in products.
    """
    # Streamlit can retain a cached recognizer across source reloads. Checking the
    # concrete class then fails if the class was re-imported, even though the
    # cached object still exposes the SAM recognition API.
    recognize = getattr(model, "recognize", None)
    if callable(recognize):
        return recognize(photo, products, confidence, min_margin, progress=progress)

    board, boxes = build_reference_board(products)
    image = photo.convert("RGB")

    raw_detections = []
    for product in products:
        product_detections = run_product_prompt(model, image, product, confidence)
        for detection in product_detections:
            detection["raw"] = True
            detection["box_area_ratio"] = (
                box_area(detection["box"]) / (image.size[0] * image.size[1])
            )
        raw_detections.extend(product_detections)

    detections = [
        detection for detection in raw_detections
        if plausible_inventory_box(
            detection,
            image.size,
            len(products),
            single_sku_mode=len(raw_detections) == 1,
        )
    ]
    detections = suppress_duplicate_detections(detections)
    if not detections:
        proposal_detections = run_proposal_match(image, products)
        raw_detections.extend(proposal_detections)
        detections = suppress_duplicate_detections(proposal_detections)

    annotated, _ = render_detections(image, detections)

    return detections, annotated, board, raw_detections
