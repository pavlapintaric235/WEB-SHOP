"""Local open-vocabulary shape recognition: one box and one mask per physical item.

This module exists because the NVIDIA Retail DINO detector that proposed the committed
scene boxes needs TensorRT/CUDA. On a machine without a usable NVIDIA GPU it cannot run
at all, so scene proposals were frozen archived GPU output and no new photograph could be
processed locally. The path here runs end to end on CPU:

  * boxes   OWLv2 (`google/owlv2-base-patch16-ensemble`, Apache-2.0) proposes candidate
            objects from a short generic vocabulary. It is a dense per-patch predictor, so
            two adjacent identical products are predicted by different image patches and
            survive as separate boxes.
  * masks   SAM 2.1 Hiera-tiny (`facebook/sam2.1-hiera-tiny`, Apache-2.0) is prompted with
            each accepted box. Box-prompted masks stay inside their box, which is what keeps
            a mask off the neighbouring item. The image encoder runs once per photograph.

Both checkpoints are permissively licensed and load through `transformers`, so this path
adds no AGPL-3.0 dependency.

Geometry, not model size, is what removes the two failure modes that matter on a crowded
shelf. OWLv2 emits group boxes and part boxes alongside the real objects, and both score
measurably lower than the objects they span:

  * a *group box* encloses several whole products at once ("one box around two products").
    `suppress_group_boxes` drops any box that contains two or more smaller retained boxes.
  * a *part box* covers a cap, a label or a panel of one product. Class-agnostic NMS plus
    the minimum-size filters remove these.

Nothing here identifies a product. Identity remains the recognizer's job, and every caller
in this repository must keep treating these boxes and masks as unreviewed model proposals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DETECTOR_ID = "google/owlv2-base-patch16-ensemble"
MASK_ID = "facebook/sam2.1-hiera-tiny"

#: Generic vocabulary. These words describe packaging and object shapes, never brands, so
#: the same prompt set works for a catalogue it has never seen.
#:
#: An open-vocabulary detector finds only what it has been given a word for. Dropping the
#: last four entries makes the detector return *nothing at all* on a photograph whose only
#: subject is an unpackaged ornament, which is how the gold crown went missing from two of
#: the thirty evaluated scenes. Extending this tuple is the cheapest correction available
#: in the whole pipeline: it needs no retraining, no labelled data and no GPU. A catalogue
#: with unusual stock should add its own shape words here rather than lower the threshold.
PROMPTS: tuple[str, ...] = (
    "a single product package",
    "a tin can",
    "a plastic bottle",
    "a cardboard box",
    "a cosmetic tube",
    "a jar",
    "a sachet",
    "a cup",
    "a battery",
    "a candle",
    "a crown",
    "a figurine",
    "a small ornament",
    "a decorative object",
)

WORKING_SIDE = 1536
SCORE_THRESHOLD = 0.10
NMS_IOU = 0.55
MIN_SIDE_PX = 16
MIN_AREA_FRACTION = 0.0008
MAX_AREA_FRACTION = 0.90
CONTAINED_FRACTION = 0.80
MAX_CHILD_AREA_RATIO = 0.70
MINIMUM_CHILDREN = 2
PART_CONTAINED_FRACTION = 0.85

BACKEND = f"{DETECTOR_ID}+{MASK_ID}:cpu"
REVIEW_STATUS = "unreviewed_proposal"


@dataclass
class Detection:
    """One proposed physical item in a photograph.

    `box` is float xyxy in the coordinate space of the image handed to `detect`, which is
    the EXIF-transposed displayed image. `mask` is a full-size boolean array or None.
    """

    box: tuple[float, float, float, float]
    score: float
    label: str
    mask: np.ndarray | None = field(default=None, repr=False)


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_containment(inner: Sequence[float], outer: Sequence[float]) -> float:
    """Fraction of `inner` that lies inside `outer`, 0.0 when `inner` is degenerate."""
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if width <= 0 or height <= 0:
        return 0.0
    area = box_area(inner)
    return 0.0 if area <= 0 else (width * height) / area


def suppress_group_boxes(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    contained_fraction: float = CONTAINED_FRACTION,
    max_child_area_ratio: float = MAX_CHILD_AREA_RATIO,
    minimum_children: int = MINIMUM_CHILDREN,
) -> list[int]:
    """Return the indices to keep, dropping boxes that enclose several separate items.

    A box is a group box when at least `minimum_children` other surviving boxes are each
    mostly inside it and each clearly smaller than it. This is the rule that prevents one
    box being drawn around two adjacent products: the enclosing box is removed and the
    individual products are kept, never the other way round.

    Candidates are considered from the highest score down, so a confident real object is
    never dropped in favour of a weaker box that happens to span it.
    """
    order = sorted(range(len(boxes)), key=lambda index: -scores[index])
    dropped: set[int] = set()
    for index in order:
        if index in dropped:
            continue
        children = 0
        for other in order:
            if other == index or other in dropped:
                continue
            if box_area(boxes[other]) > max_child_area_ratio * box_area(boxes[index]):
                continue
            if box_containment(boxes[other], boxes[index]) >= contained_fraction:
                children += 1
                if children >= minimum_children:
                    break
        if children >= minimum_children:
            dropped.add(index)
    return [index for index in order if index not in dropped]


def suppress_contained_parts(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    contained_fraction: float = PART_CONTAINED_FRACTION,
) -> list[int]:
    """Return the indices to keep, dropping boxes that are a part of a stronger box.

    Run this *after* `suppress_group_boxes`, never before. The group rule has already
    removed boxes that span several whole items, so anything still lying almost entirely
    inside a surviving higher-scoring box is a cap, a label or a panel of that item rather
    than a separate product. This is what stops one tall item being reported several times.

    The threshold is deliberately high: a small product standing in front of a large one is
    also mostly inside its box, and keeping an extra proposal for a reviewer to reject costs
    less than silently deleting a real item.
    """
    order = sorted(range(len(boxes)), key=lambda index: -scores[index])
    kept: list[int] = []
    for index in order:
        if any(box_containment(boxes[index], boxes[parent]) >= contained_fraction for parent in kept):
            continue
        kept.append(index)
    return kept


def plausible_boxes(
    boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
    min_side_px: int = MIN_SIDE_PX,
    min_area_fraction: float = MIN_AREA_FRACTION,
    max_area_fraction: float = MAX_AREA_FRACTION,
) -> list[int]:
    """Indices of boxes that could be a photographed product rather than noise or the scene."""
    frame = float(width * height)
    keep: list[int] = []
    for index, box in enumerate(boxes):
        side_ok = (box[2] - box[0]) >= min_side_px and (box[3] - box[1]) >= min_side_px
        fraction = box_area(box) / frame if frame else 0.0
        if side_ok and min_area_fraction <= fraction <= max_area_fraction:
            keep.append(index)
    return keep


def mask_leak_fraction(mask: np.ndarray, box: Sequence[float]) -> float:
    """Fraction of mask pixels falling outside `box` — how far a mask spilled off its item."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    inside = int(mask[max(0, y1):max(0, y2), max(0, x1):max(0, x2)].sum())
    return round((total - inside) / total, 4)


class ShapeDetector:
    """OWLv2 box proposals refined into SAM 2.1 instance masks, on CPU.

    The models are loaded lazily on first use so that importing this module, and the pure
    geometry helpers above, stays free.
    """

    def __init__(
        self,
        detector_id: str = DETECTOR_ID,
        mask_id: str = MASK_ID,
        device: str = "cpu",
        prompts: Iterable[str] = PROMPTS,
        threshold: float = SCORE_THRESHOLD,
        nms_iou: float = NMS_IOU,
        working_side: int = WORKING_SIDE,
        with_masks: bool = True,
    ) -> None:
        self.detector_id = detector_id
        self.mask_id = mask_id
        self.device = device
        self.prompts = list(prompts)
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.working_side = working_side
        self.with_masks = with_masks
        self._detector = None
        self._detector_processor = None
        self._masker = None
        self._mask_processor = None

    def _load_detector(self) -> None:
        if self._detector is not None:
            return
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._detector_processor = Owlv2Processor.from_pretrained(self.detector_id)
        self._detector = Owlv2ForObjectDetection.from_pretrained(self.detector_id).to(self.device).eval()

    def _load_masker(self) -> None:
        if self._masker is not None:
            return
        from transformers import Sam2Model, Sam2Processor

        self._mask_processor = Sam2Processor.from_pretrained(self.mask_id)
        self._masker = Sam2Model.from_pretrained(self.mask_id).to(self.device).eval()

    def working_image(self, image):
        """Downscale a photograph to the detector's working side, preserving aspect ratio.

        OWLv2 resizes to 960px and SAM 2.1 to 1024px internally, so feeding a 4032px
        original buys no detail and makes every returned mask needlessly large.
        """
        working = image.copy()
        working.thumbnail((self.working_side, self.working_side))
        return working

    def propose(self, image) -> tuple[list[list[float]], list[float], list[str]]:
        """Candidate boxes in `image` pixel coordinates, after filtering and suppression."""
        import torch
        from torchvision.ops import nms

        self._load_detector()
        width, height = image.size
        # OWLv2 pads the image to a square on the bottom and right before resizing, so the
        # post-processor must be told the padded square size. Passing (height, width) here
        # yields systematically offset boxes on any non-square photograph.
        padded = max(width, height)
        text_labels = [self.prompts]
        inputs = self._detector_processor(text=text_labels, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._detector(**inputs)
        result = self._detector_processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(padded, padded)]),
            threshold=self.threshold,
            text_labels=text_labels,
        )[0]

        boxes, scores = result["boxes"], result["scores"]
        labels = result.get("text_labels") or ["object"] * len(boxes)
        if len(boxes) == 0:
            return [], [], []
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)

        keep = nms(boxes, scores, self.nms_iou).tolist()
        boxes = [boxes[index].tolist() for index in keep]
        scores = [float(scores[index]) for index in keep]
        labels = [labels[index] for index in keep]

        keep = plausible_boxes(boxes, width, height)
        boxes = [boxes[index] for index in keep]
        scores = [scores[index] for index in keep]
        labels = [labels[index] for index in keep]

        keep = suppress_group_boxes(boxes, scores)
        boxes = [boxes[index] for index in keep]
        scores = [scores[index] for index in keep]
        labels = [labels[index] for index in keep]

        keep = suppress_contained_parts(boxes, scores)
        order = sorted(keep, key=lambda index: -scores[index])
        return ([boxes[index] for index in order],
                [scores[index] for index in order],
                [labels[index] for index in order])

    def masks_for(self, image, boxes: Sequence[Sequence[float]]) -> list[np.ndarray]:
        """One boolean full-size mask per box, prompted by that box."""
        import torch

        if not boxes:
            return []
        self._load_masker()
        inputs = self._mask_processor(
            images=image, input_boxes=[[list(box) for box in boxes]], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self._masker(**inputs, multimask_output=False)
        masks = self._mask_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        return [np.asarray(mask[0], dtype=bool) for mask in masks]

    def detect(self, image, original_resolution: bool = True) -> list[Detection]:
        """Proposed items in `image`, highest scoring first.

        `image` must already be EXIF-transposed and RGB. Detection runs on a downscaled
        working copy, and a `Detection`'s box and mask are always returned in one shared
        coordinate space: `image` pixels by default, or the smaller working resolution
        when `original_resolution` is False and memory matters more than mask precision.
        """
        import cv2

        working = self.working_image(image)
        boxes, scores, labels = self.propose(working)
        masks = self.masks_for(working, boxes) if (self.with_masks and boxes) else [None] * len(boxes)

        scale_x = image.width / working.width
        scale_y = image.height / working.height
        detections: list[Detection] = []
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            full = (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
            if mask is not None and original_resolution and mask.shape != (image.height, image.width):
                mask = cv2.resize(mask.astype(np.uint8), (image.width, image.height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            detections.append(Detection(box=full if original_resolution else tuple(box),
                                        score=score, label=label, mask=mask))
        return detections
