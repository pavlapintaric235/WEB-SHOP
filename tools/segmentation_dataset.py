"""Prepare reviewed, class-agnostic product segmentation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = ROOT / "data" / "segmentation_dataset"
SEED = ROOT / "data" / "cvat_fixed" / "revision_02"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_IMAGE_SIDE = 1280
MIN_MASK_FRACTION = 0.002
PALETTE = {
    "suggested": (30, 190, 95),
    "uncertain": (246, 170, 35),
    "rejected": (225, 55, 55),
}


def utc_id(prefix="batch"):
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_area(points):
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2 if len(points) >= 3 else 0.0


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, eps=1e-10):
    return (
        abs(_orientation(a, b, p)) <= eps
        and min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _segments_intersect(a, b, c, d, eps=1e-10):
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return (
        (abs(o1) <= eps and _on_segment(a, b, c, eps))
        or (abs(o2) <= eps and _on_segment(a, b, d, eps))
        or (abs(o3) <= eps and _on_segment(c, d, a, eps))
        or (abs(o4) <= eps and _on_segment(c, d, b, eps))
    )


def polygon_self_intersections(points):
    """Return non-adjacent intersecting edge index pairs."""
    n = len(points)
    hits = []
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if _segments_intersect(a, b, points[j], points[(j + 1) % n]):
                hits.append((i, j))
    return hits


def parse_yolo_label(text, source="annotation"):
    rows = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
        except (ValueError, IndexError) as error:
            raise ValueError(f"{source}:{line_number}: not a YOLO polygon row") from error
        if len(values) < 6 or len(values) % 2:
            raise ValueError(f"{source}:{line_number}: polygon needs at least 3 x/y points")
        points = list(zip(values[::2], values[1::2]))
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            raise ValueError(f"{source}:{line_number}: coordinates must be finite")
        if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in points):
            raise ValueError(f"{source}:{line_number}: coordinates must be normalized to [0, 1]")
        if len(set(points)) < 3 or polygon_area(points) <= 1e-12:
            raise ValueError(f"{source}:{line_number}: polygon has fewer than 3 unique points or zero area")
        intersections = polygon_self_intersections(points)
        if intersections:
            raise ValueError(f"{source}:{line_number}: self-intersection at edges {intersections[0]}")
        rows.append((class_id, points))
    return rows


def yolo_line(points):
    fields = ["0"]
    for x, y in points:
        fields.extend((f"{x:.6f}", f"{y:.6f}"))
    return " ".join(fields)


def _manifest_path(dataset=DATASET):
    return Path(dataset) / "approved" / "review_manifest.json"


def _load_manifest(dataset=DATASET):
    path = _manifest_path(dataset)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("images", {})
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read review manifest {path}: {error}") from error


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def initialize_workspace(dataset=DATASET, seed=SEED):
    dataset = Path(dataset)
    folders = (
        "input_images", "proposals", "approved/images", "approved/labels",
        "previews", "uncertain", "rejected", "exports", "reports",
    )
    for relative in folders:
        (dataset / relative).mkdir(parents=True, exist_ok=True)

    approved_images = dataset / "approved" / "images"
    approved_labels = dataset / "approved" / "labels"
    manifest_path = _manifest_path(dataset)
    if manifest_path.exists() or any(approved_images.iterdir()) or any(approved_labels.iterdir()):
        raise FileExistsError(
            f"Approved data already exists under {dataset}; init will not overwrite it."
        )

    source_zip = seed / "PicturesCVAT_fixed_v2.zip"
    if not source_zip.is_file():
        raise FileNotFoundError(f"Approved seed archive not found: {source_zip}")
    records = {}
    seed_hash = sha256_file(source_zip)
    with zipfile.ZipFile(source_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("Approved seed ZIP failed its CRC check.")
        for index, expected_count in enumerate((3, 5, 9), 1):
            image_name = f"image{index}.png"
            label_name = f"image{index}.txt"
            image_bytes = archive.read(f"images/train/{image_name}")
            label_bytes = archive.read(f"labels/train/{label_name}")
            rows = parse_yolo_label(label_bytes.decode("utf-8"), label_name)
            if len(rows) != expected_count or any(class_id != 0 for class_id, _ in rows):
                raise ValueError(f"Approved seed {image_name} failed its count/class check.")
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
            (approved_images / image_name).write_bytes(image_bytes)
            (approved_labels / label_name).write_bytes(label_bytes)
            records[image_name] = {
                "reviewed": True,
                "all_visible_products_reviewed": True,
                "negative": False,
                "instances": expected_count,
                "group_id": "approved_cvat_seed_batch_01",
                "source": "User-approved CVAT revision 2",
                "source_archive_sha256": seed_hash,
            }

    _atomic_json(manifest_path, {
        "schema_version": 1,
        "images": records,
    })
    (dataset / "README.md").write_text(
        "# InventoryLens Segmentation Dataset\n\n"
        "`input_images/` is for new original scene photos. Copy files there; the tool never moves or edits them.\n\n"
        "Only manually reviewed CVAT exports imported with `--reviewed` go into `approved/`.\n"
        "The three approved seed photos are marked `approved_cvat_seed_batch_01`; keep them out of any future test split.\n\n"
        "Run `python tools/segmentation_dataset.py --help` from the project root for commands.\n",
        encoding="utf-8",
    )
    validation = validate_dataset(dataset)
    write_validation_report(dataset, validation, name="seed_validation")
    return validation


def _read_oriented_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def _mask_polygon(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.001), True)
    if len(approximation) < 3:
        return []
    height, width = binary.shape
    points = [(float(point[0][0]) / width, float(point[0][1]) / height) for point in approximation]
    if polygon_area(points) <= 1e-12 or polygon_self_intersections(points):
        return []
    return points


def _connected_masks(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] <= 0:
            continue
        components.append(labels == component_id)
    return components


def _mask_edge_coverage(mask):
    band = min(min(mask.shape), max(2, round(min(mask.shape) * 0.01)))
    return [float(edge.mean()) for edge in (
        mask[:band].any(axis=0), mask[-band:].any(axis=0),
        mask[:, :band].any(axis=1), mask[:, -band:].any(axis=1),
    )]


def _candidate(mask, quality, candidate_id, sam_id, inherited_reasons=(), rejected_reason=None):
    height, width = mask.shape
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    reasons = list(inherited_reasons)
    if rejected_reason:
        status = "rejected"
        reasons.append(rejected_reason)
    else:
        status = "suggested"
        fraction = area / (height * width)
        coverage = _mask_edge_coverage(mask)
        if any(value >= 0.01 for value in coverage):
            reasons.append("touches_image_edge")
        if fraction >= 0.50:
            reasons.append("large_mask_review")
        if fraction < 0.015:
            reasons.append("small_fragment_review")
        if math.isfinite(quality) and quality < 0.85:
            reasons.append("low_sam_score_review")
        box_width = int(xs.max() - xs.min() + 1)
        box_height = int(ys.max() - ys.min() + 1)
        fill = area / max(1, box_width * box_height)
        ratio = max(box_width / max(1, box_height), box_height / max(1, box_width))
        if fill < 0.12:
            reasons.append("sparse_or_fragmented_mask_review")
        if ratio > 10:
            reasons.append("extreme_aspect_ratio_review")
        if reasons:
            status = "uncertain"
    box = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    polygon = _mask_polygon(mask)
    if not polygon and status != "rejected":
        status = "uncertain"
        reasons.append("polygon_conversion_failed")
    return {
        "candidate_id": candidate_id,
        "source_sam_mask": sam_id,
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "sam_score": float(quality) if math.isfinite(quality) else None,
        "mask_area_ratio": area / (height * width),
        "box": box,
        "polygon": polygon,
        "_mask": mask,
    }


def _mark_duplicate_or_overlap(candidates):
    active = [item for item in candidates if item["status"] != "rejected" and item["_mask"].any()]
    for offset, first in enumerate(active):
        for second in active[offset + 1:]:
            a, b = first["_mask"], second["_mask"]
            intersection = int(np.count_nonzero(a & b))
            if not intersection:
                continue
            area_a, area_b = int(a.sum()), int(b.sum())
            iou = intersection / (area_a + area_b - intersection)
            containment = intersection / min(area_a, area_b)
            if iou >= 0.80 or containment >= 0.97:
                loser = second
                if (first["sam_score"] or 0) > (second["sam_score"] or 0):
                    loser = second
                elif (second["sam_score"] or 0) > (first["sam_score"] or 0):
                    loser = first
                elif area_a < area_b:
                    loser = first
                loser["status"] = "uncertain"
                loser["reasons"].append("possible_duplicate_or_nested_mask")
            elif iou >= 0.20:
                for item in (first, second):
                    item["status"] = "uncertain"
                    item["reasons"].append("overlapping_masks_need_review")

    # A broad mask containing multiple disjoint smaller masks may cover several products.
    for parent in active:
        parent_area = int(parent["_mask"].sum())
        children = []
        for child in active:
            if child is parent:
                continue
            child_area = int(child["_mask"].sum())
            if 0.15 * parent_area < child_area < 0.80 * parent_area:
                overlap = int(np.count_nonzero(parent["_mask"] & child["_mask"])) / child_area
                if overlap > 0.90:
                    children.append(child)
        if len(children) >= 2:
            disjoint = all(
                np.count_nonzero(a["_mask"] & b["_mask"]) / max(1, min(a["_mask"].sum(), b["_mask"].sum())) < 0.20
                for i, a in enumerate(children) for b in children[i + 1:]
            )
            if disjoint:
                parent["status"] = "uncertain"
                parent["reasons"].append("possible_mask_covers_multiple_products")


class DatasetSAMProposer:
    """Use the same MobileSAM predictor settings as recognition, without SKU matching."""

    def __init__(self):
        from vision import configure_ultralytics_environment
        configure_ultralytics_environment()
        from model_assets import SAM_FILE
        if not SAM_FILE.is_file():
            raise FileNotFoundError(
                f"MobileSAM weights are missing: {SAM_FILE}. "
                "The project setup command downloads this model; no download was started."
            )
        import torch
        from ultralytics import SAM
        from ultralytics.models.sam.predict import Predictor
        from sam_vision import SAMRecognizer

        class ProposalPredictor(Predictor):
            def generate(self, im, *args, **kwargs):
                kwargs.update(
                    points_stride=12,
                    points_batch_size=16,
                    crop_n_layers=0,
                    conf_thres=0.80,
                    stability_score_thresh=0.92,
                )
                return super().generate(im, *args, **kwargs)

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.recognizer = SAMRecognizer.__new__(SAMRecognizer)
        self.recognizer.segmenter = SAM(str(SAM_FILE))
        self.recognizer.predictor_class = ProposalPredictor
        self.rejection = SAMRecognizer.rejection

    def segment(self, image):
        return self.recognizer.segment(image)


def _discover_images(input_dir, single_image=None):
    if single_image:
        paths = [Path(single_image).resolve()]
    else:
        directory = Path(input_dir)
        paths = sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ) if directory.is_dir() else []
    if not paths:
        raise FileNotFoundError(
            f"No photos found. Copy original scene photos into {input_dir} or pass --image PATH."
        )
    stems = [path.stem.casefold() for path in paths]
    if len(set(stems)) != len(stems):
        raise ValueError("Input photo names must have unique file names (same-name JPG/PNG pairs are ambiguous).")
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Not a supported image: {path}")
    return paths


def _draw_preview(image, candidates, path):
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    thickness = max(3, width // 500)
    try:
        font = ImageFont.truetype("arial.ttf", max(24, width // 55))
        small_font = ImageFont.truetype("arial.ttf", max(18, width // 90))
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    for item in candidates:
        if item["status"] == "rejected":
            continue
        points = item["polygon"]
        mask = item["_mask"]
        color = PALETTE[item["status"]]
        if points:
            xy = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in points]
            if item["status"] != "rejected":
                draw_overlay.polygon(xy, fill=(*color, 55))
            draw_overlay.line(xy + [xy[0]], fill=(*color, 255), width=thickness, joint="curve")
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        else:
            resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(resized)
            if len(xs):
                contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2_image = np.asarray(canvas.convert("RGB")).copy()
                cv2.drawContours(cv2_image, contours, -1, color, thickness)
                canvas = Image.fromarray(cv2_image).convert("RGBA")
                draw = ImageDraw.Draw(canvas)
        if not len(xs):
            continue
        cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
        number = str(item["candidate_id"])
        box = draw.textbbox((0, 0), number, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        pad = max(5, thickness)
        draw.rectangle((cx - tw // 2 - pad, cy - th // 2 - pad,
                        cx + tw // 2 + pad, cy + th // 2 + pad), fill=(0, 0, 0, 230))
        draw.text((cx - tw // 2, cy - th // 2 - box[1]), number, font=font, fill=color)
    canvas = Image.alpha_composite(canvas, overlay)
    legend = "Green: suggestion   Amber: needs review   Rejected masks are saved separately"
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, max(56, small_font.size + 18)), fill=(255, 255, 255, 225))
    draw.text((12, 8), legend, font=small_font, fill=(25, 25, 25, 255))
    canvas.convert("RGB").save(path, quality=95)


def propose(input_dir=DATASET / "input_images", image=None, batch_id=None, output_dir=None, dataset=DATASET):
    dataset = Path(dataset)
    paths = _discover_images(input_dir, image)
    batch_id = batch_id or utc_id()
    proposal_dir = Path(output_dir) if output_dir else dataset / "proposals" / batch_id
    if not proposal_dir.is_absolute():
        proposal_dir = (ROOT / proposal_dir).resolve()
    if proposal_dir.exists() and any(path.is_file() for path in proposal_dir.rglob("*")):
        raise FileExistsError(f"Output is not empty; refusing to overwrite: {proposal_dir}")
    preview_dir = (dataset / "previews" / batch_id) if not output_dir else proposal_dir / "previews"
    uncertain_dir = (dataset / "uncertain" / batch_id) if not output_dir else proposal_dir / "uncertain"
    rejected_dir = (dataset / "rejected" / batch_id) if not output_dir else proposal_dir / "rejected"
    image_dir, label_dir = proposal_dir / "images", proposal_dir / "labels"
    for folder in (image_dir, label_dir, preview_dir, uncertain_dir, rejected_dir):
        folder.mkdir(parents=True, exist_ok=True)

    model = DatasetSAMProposer()
    approved_hashes = {}
    approved_dir = dataset / "approved" / "images"
    if approved_dir.is_dir():
        approved_hashes = {sha256_file(p): p.name for p in approved_dir.iterdir() if p.is_file()}
    seen_hashes = {}
    summary = {"batch_id": batch_id, "images": [], "model": "MobileSAM (proposal only; no SKU matching)",
               "warning": "Every suggestion needs human review. Suggestions are not approved labels."}

    for source_path in paths:
        source = _read_oriented_rgb(source_path)
        normalized_name = source_path.stem + ".png"
        image_out = image_dir / normalized_name
        source.save(image_out, format="PNG", optimize=True)
        digest = sha256_file(image_out)
        image_warnings = []
        if digest in seen_hashes:
            image_warnings.append(f"duplicate_photo_in_batch:{seen_hashes[digest]}")
        else:
            seen_hashes[digest] = source_path.name
        if digest in approved_hashes:
            image_warnings.append(f"duplicate_of_approved_photo:{approved_hashes[digest]}")

        working = source.copy()
        working.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        masks, qualities = model.segment(working)
        work_shape = (working.height, working.width)
        candidates = []
        next_id = 1
        for sam_id, (raw_mask, quality) in enumerate(zip(masks, qualities), 1):
            mask = np.asarray(raw_mask, dtype=bool)
            if mask.shape != work_shape:
                mask = cv2.resize(mask.astype(np.uint8), (working.width, working.height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            rejection = model.rejection(mask)
            if rejection:
                candidates.append(_candidate(mask, float(quality), next_id, sam_id, rejected_reason=rejection))
                next_id += 1
                continue
            parts = _connected_masks(mask)
            split_reason = ["disconnected_sam_mask_split_into_components"] if len(parts) > 1 else []
            for part in parts:
                part_rejection = "mask_too_small" if float(part.mean()) < MIN_MASK_FRACTION else None
                candidates.append(_candidate(part, float(quality), next_id, sam_id,
                                             inherited_reasons=split_reason,
                                             rejected_reason=part_rejection))
                next_id += 1
        _mark_duplicate_or_overlap(candidates)

        clean_lines = []
        for item in candidates:
            if item["status"] == "suggested" and item["polygon"]:
                clean_lines.append(yolo_line(item["polygon"]))
            elif item["status"] == "uncertain":
                mask_out = uncertain_dir / f"{source_path.stem}_candidate_{item['candidate_id']:03d}.png"
                Image.fromarray(item["_mask"].astype(np.uint8) * 255).save(mask_out)
            else:
                mask_out = rejected_dir / f"{source_path.stem}_candidate_{item['candidate_id']:03d}.png"
                Image.fromarray(item["_mask"].astype(np.uint8) * 255).save(mask_out)

        label_path = label_dir / f"{source_path.stem}.txt"
        label_path.write_text("\n".join(clean_lines) + ("\n" if clean_lines else ""), encoding="utf-8")
        preview_path = preview_dir / f"{source_path.stem}_preview.png"
        _draw_preview(source, candidates, preview_path)

        candidates_public = []
        for item in candidates:
            candidates_public.append({key: value for key, value in item.items() if key != "_mask"})
        summary["images"].append({
            "source_file": str(source_path), "output_image": normalized_name,
            "sha256": digest, "width": source.width, "height": source.height,
            "working_width": working.width, "working_height": working.height,
            "sam_masks": len(masks), "candidate_components": len(candidates),
            "suggested": sum(item["status"] == "suggested" for item in candidates),
            "uncertain": sum(item["status"] == "uncertain" for item in candidates),
            "rejected": sum(item["status"] == "rejected" for item in candidates),
            "input_warnings": image_warnings, "candidates": candidates_public,
        })

    # CVAT/Ultralytics YOLO segmentation import bundle contains only unflagged suggestions.
    (proposal_dir / "data.yaml").write_text("path: .\ntrain: train.txt\nnames:\n  0: product\n", encoding="utf-8")
    (proposal_dir / "train.txt").write_text(
        "".join(f"images/train/{path.name}\n" for path in sorted(image_dir.glob("*.png"))), encoding="utf-8"
    )
    with zipfile.ZipFile(proposal_dir / "cvat_import.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(proposal_dir / "data.yaml", "data.yaml")
        archive.write(proposal_dir / "train.txt", "train.txt")
        for path in sorted(image_dir.glob("*.png")):
            archive.write(path, f"images/train/{path.name}")
        for path in sorted(label_dir.glob("*.txt")):
            archive.write(path, f"labels/train/{path.name}")

    _atomic_json(proposal_dir / "proposal_report.json", summary)
    rows = [
        f"# Proposal batch `{batch_id}`", "",
        "These MobileSAM polygons are unapproved proposals. Review every preview and correct/add/remove polygons in CVAT.", "",
        "| Image | SAM masks | Clean suggestions | Uncertain | Rejected |", "|---|---:|---:|---:|---:|",
    ]
    for item in summary["images"]:
        rows.append(f"| `{item['output_image']}` | {item['sam_masks']} | {item['suggested']} | {item['uncertain']} | {item['rejected']} |")
    rows += ["", f"Preview folder: `{preview_dir}`", f"CVAT import archive: `{proposal_dir / 'cvat_import.zip'}`", "",
             "Uncertain and rejected masks are not placed in YOLO labels. Check `proposal_report.json` and the matching mask PNGs.",
             "A clean suggestion still is not ground truth; a human must verify every visible physical product."]
    (proposal_dir / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Proposal batch: {batch_id}")
    print(f"Images: {len(paths)} | report: {proposal_dir / 'proposal_report.json'}")
    print(f"Previews: {preview_dir}")
    print(f"CVAT import ZIP (suggestions only): {proposal_dir / 'cvat_import.zip'}")
    print("No annotations were added to approved/.")
    return summary


def _label_files_in_zip(archive):
    result = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.suffix.lower() != ".txt" or path.name.lower() in {"train.txt", "val.txt", "valid.txt", "test.txt"}:
            continue
        if not any(part.lower() in {"labels", "obj_train_data"} for part in path.parts[:-1]):
            continue
        stem = path.stem.casefold()
        if stem in result:
            raise ValueError(f"CVAT archive contains ambiguous duplicate label names for {path.stem}.")
        result[stem] = info.filename
    return result


def _image_files_in_zip(archive):
    result = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if not any(part.lower() in {"images", "obj_train_data"} for part in path.parts[:-1]):
            continue
        stem = path.stem.casefold()
        if stem in result:
            raise ValueError(f"CVAT archive contains ambiguous duplicate image names for {path.stem}.")
        result[stem] = info.filename
    return result


def _validate_cvat_archive(archive, batch_dir, explicit_negatives):
    names = [PurePosixPath(name) for name in archive.namelist()]
    if any(path.is_absolute() or ".." in path.parts for path in names):
        raise ValueError("CVAT archive contains an unsafe path.")
    label_entries = _label_files_in_zip(archive)
    image_entries = _image_files_in_zip(archive)
    source_images = {path.stem.casefold(): path for path in (batch_dir / "images").glob("*.png")}
    if not source_images:
        raise FileNotFoundError(f"Proposal batch has no source images: {batch_dir / 'images'}")
    if len(source_images) != len(list((batch_dir / "images").glob("*.png"))):
        raise ValueError("Proposal batch contains ambiguous image stems.")
    if set(image_entries) != set(source_images):
        missing = sorted(set(source_images) - set(image_entries))
        extra = sorted(set(image_entries) - set(source_images))
        raise ValueError(f"CVAT image mapping mismatch. Missing={missing}; unexpected={extra}.")
    for stem, entry in image_entries.items():
        with Image.open(BytesIO(archive.read(entry))) as exported:
            exported_rgb = ImageOps.exif_transpose(exported).convert("RGB")
        with Image.open(source_images[stem]) as expected:
            expected_rgb = expected.convert("RGB")
        if exported_rgb.size != expected_rgb.size or ImageChops.difference(exported_rgb, expected_rgb).getbbox() is not None:
            raise ValueError(f"CVAT image pixels do not match proposal source `{source_images[stem].name}`.")
    yaml_entries = [name for name in archive.namelist() if PurePosixPath(name).name.lower() == "data.yaml"]
    if yaml_entries and "product" not in archive.read(yaml_entries[0]).decode("utf-8", errors="replace").lower():
        raise ValueError("CVAT data.yaml does not declare the expected product class.")

    negative_stems = {Path(name).stem.casefold() for name in explicit_negatives}
    if any(Path(name).name != name for name in explicit_negatives):
        raise ValueError("--negative accepts image file names only, not paths.")
    if not negative_stems.issubset(source_images):
        raise ValueError(f"Unknown negative image(s): {sorted(negative_stems - set(source_images))}")
    missing_labels = set(source_images) - set(label_entries)
    if missing_labels != negative_stems:
        raise ValueError(
            "Each task photo needs a CVAT label file; for an intentionally empty/negative image, "
            f"repeat --negative FILENAME after reviewing it. Missing labels: {sorted(missing_labels)}"
        )
    validated = {}
    for stem, image_path in source_images.items():
        text = archive.read(label_entries[stem]).decode("utf-8") if stem in label_entries else ""
        rows = parse_yolo_label(text, label_entries.get(stem, f"{image_path.stem}.txt"))
        if any(class_id != 0 for class_id, _ in rows):
            raise ValueError(f"{image_path.stem}: expected only class 0 = product.")
        if stem in negative_stems and rows:
            raise ValueError(f"{image_path.name} was marked negative but has polygons.")
        validated[image_path.name] = (text, rows, stem in negative_stems)
    return validated


def import_cvat(zip_path, batch_id, reviewed=False, negatives=(), dataset=DATASET):
    if not reviewed:
        raise ValueError("Refusing to promote draft labels. Re-run only after CVAT review with --reviewed.")
    dataset = Path(dataset)
    batch_dir = dataset / "proposals" / batch_id
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"Proposal batch not found: {batch_dir}")
    zip_path = Path(zip_path).resolve()
    with zipfile.ZipFile(zip_path) as archive:
        validated = _validate_cvat_archive(archive, batch_dir, negatives)

    approved_images = dataset / "approved" / "images"
    approved_labels = dataset / "approved" / "labels"
    approved_images.mkdir(parents=True, exist_ok=True)
    approved_labels.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dataset)
    existing_hashes = {sha256_file(p): p.name for p in approved_images.iterdir() if p.is_file()}
    batch_hashes = {}
    for filename, (_, rows, negative) in validated.items():
        image_path = batch_dir / "images" / filename
        digest = sha256_file(image_path)
        if filename in manifest or (approved_images / filename).exists() or (approved_labels / f"{Path(filename).stem}.txt").exists():
            raise FileExistsError(f"Approved item already exists: {filename}; refusing to overwrite.")
        if digest in existing_hashes:
            raise ValueError(f"Duplicate of approved image {existing_hashes[digest]}: {filename}")
        if digest in batch_hashes:
            raise ValueError(f"Duplicate of another photo in this batch ({batch_hashes[digest]}): {filename}")
        batch_hashes[digest] = filename
        if negative != (len(rows) == 0):
            raise ValueError(f"Negative-image confirmation mismatch for {filename}.")

    written = []
    try:
        for filename, (text, rows, negative) in validated.items():
            image_path = batch_dir / "images" / filename
            label_path = approved_labels / f"{Path(filename).stem}.txt"
            shutil.copyfile(image_path, approved_images / filename)
            label_path.write_text(text, encoding="utf-8")
            written.extend((approved_images / filename, label_path))
            manifest[filename] = {
                "reviewed": True,
                "all_visible_products_reviewed": True,
                "negative": negative,
                "instances": len(rows),
                "group_id": batch_id,
                "source": f"Manually reviewed CVAT export: {zip_path.name}",
                "approved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        _atomic_json(_manifest_path(dataset), {"schema_version": 1, "images": manifest})
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    print(f"Imported {len(validated)} manually reviewed image(s) into approved/.")
    print("Run `python tools/segmentation_dataset.py validate` to check the full approved dataset.")
    return validated


def _bbox_iou(first, second):
    x1, y1, x2, y2 = first
    a1, b1, a2, b2 = second
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union else 0.0


def validate_dataset(dataset=DATASET):
    dataset = Path(dataset)
    image_dir, label_dir = dataset / "approved" / "images", dataset / "approved" / "labels"
    issues = []
    counts = {}
    if not image_dir.is_dir() or not label_dir.is_dir():
        return {"images": 0, "approved_instances": 0, "object_counts": {},
                "issues": [{"severity": "error", "code": "dataset_folders_missing", "detail": str(dataset)}]}
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    labels = {p.stem.casefold(): p for p in label_dir.glob("*.txt")}
    manifest = _load_manifest(dataset)
    hashes = {}
    total = 0
    for image_path in images:
        key = image_path.stem.casefold()
        label_path = labels.get(key)
        if label_path is None:
            issues.append({"severity": "error", "code": "missing_annotation_file", "image": image_path.name})
            counts[image_path.name] = None
            continue
        digest = sha256_file(image_path)
        if digest in hashes:
            issues.append({"severity": "warning", "code": "duplicate_image_content",
                           "image": image_path.name, "duplicate_of": hashes[digest]})
        else:
            hashes[digest] = image_path.name
        entry = manifest.get(image_path.name)
        if not entry or not entry.get("reviewed") or not entry.get("all_visible_products_reviewed"):
            issues.append({"severity": "error", "code": "human_review_not_confirmed", "image": image_path.name})
        try:
            with Image.open(image_path) as source:
                source.verify()
            rows = parse_yolo_label(label_path.read_text(encoding="utf-8"), label_path.name)
        except (OSError, ValueError, UnicodeDecodeError) as error:
            issues.append({"severity": "error", "code": "invalid_image_or_label", "image": image_path.name,
                           "detail": str(error)})
            counts[image_path.name] = None
            continue
        bad_classes = [i + 1 for i, (class_id, _) in enumerate(rows) if class_id != 0]
        if bad_classes:
            issues.append({"severity": "error", "code": "wrong_class_id", "image": image_path.name,
                           "rows": bad_classes})
        count = len(rows)
        counts[image_path.name] = count
        total += count
        if entry:
            if entry.get("instances") != count:
                issues.append({"severity": "error", "code": "manifest_count_mismatch", "image": image_path.name,
                               "manifest": entry.get("instances"), "labels": count})
            if entry.get("negative") and count:
                issues.append({"severity": "error", "code": "negative_image_has_instances", "image": image_path.name})
            if count == 0 and not entry.get("negative"):
                issues.append({"severity": "warning", "code": "empty_labels_not_confirmed_negative", "image": image_path.name})
        elif count == 0:
            issues.append({"severity": "warning", "code": "empty_labels_not_confirmed_negative", "image": image_path.name})

        canonical = set()
        boxes = []
        for row_index, (_, points) in enumerate(rows, 1):
            variants = []
            for ordered in (points, list(reversed(points))):
                variants.extend(tuple(ordered[i:] + ordered[:i]) for i in range(len(ordered)))
            signature = min(tuple((round(x, 7), round(y, 7)) for x, y in variant) for variant in variants)
            if signature in canonical:
                issues.append({"severity": "warning", "code": "duplicate_polygon_row", "image": image_path.name,
                               "instance": row_index})
            canonical.add(signature)
            boxes.append((min(x for x, _ in points), min(y for _, y in points),
                          max(x for x, _ in points), max(y for _, y in points)))
        for i, box in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                if _bbox_iou(box, boxes[j]) >= 0.70:
                    issues.append({"severity": "warning", "code": "instances_have_heavily_overlapping_boxes",
                                   "image": image_path.name, "instances": [i + 1, j + 1],
                                   "detail": "Review for duplicate masks or touching products; this is only a box heuristic."})

    for stem, label_path in labels.items():
        if not any(image.stem.casefold() == stem for image in images):
            issues.append({"severity": "error", "code": "orphan_annotation_file", "label": label_path.name})
    report = {
        "dataset": str(dataset),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "class_map": {"0": "product"},
        "images": len(images),
        "approved_instances": total,
        "flagged_issues": len(issues),
        "errors": sum(item["severity"] == "error" for item in issues),
        "warnings": sum(item["severity"] == "warning" for item in issues),
        "object_counts": counts,
        "issues": issues,
    }
    return report


def write_validation_report(dataset, report, name=None):
    dataset = Path(dataset)
    reports_dir = dataset / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    name = name or f"validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    _atomic_json(reports_dir / f"{name}.json", report)
    lines = [
        "# Segmentation Dataset Validation", "",
        f"- Images: {report['images']}",
        f"- Approved instances: {report['approved_instances']}",
        f"- Errors: {report['errors']}",
        f"- Warnings / review flags: {report['warnings']}",
        "", "## Object counts", "",
        "| Photograph | Instances |", "|---|---:|",
    ]
    for filename, count in report["object_counts"].items():
        lines.append(f"| `{filename}` | {count if count is not None else 'invalid'} |")
    lines += ["", "## Flags", ""]
    if not report["issues"]:
        lines.append("No structural or geometric issues found.")
    for issue in report["issues"]:
        lines.append(f"- **{issue['severity']} / {issue['code']}**: {issue.get('image', issue.get('label', 'dataset'))}"
                     + (f": {issue['detail']}" if issue.get("detail") else ""))
    lines += ["", "Geometry checks cannot confirm that every visible package was annotated correctly. Human visual review remains required."]
    (reports_dir / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return reports_dir / f"{name}.md"


def export_dataset(dataset=DATASET, export_id=None):
    dataset = Path(dataset)
    report = validate_dataset(dataset)
    validation_report = write_validation_report(dataset, report)
    if report["errors"]:
        raise ValueError(f"Export blocked by {report['errors']} validation error(s); see {validation_report}.")
    export_id = export_id or utc_id("yolo_segmentation")
    output = dataset / "exports" / export_id
    if output.exists():
        raise FileExistsError(f"Export already exists: {output}")
    images_out, labels_out = output / "images" / "train", output / "labels" / "train"
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)
    images = sorted(p for p in (dataset / "approved" / "images").iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for image in images:
        label = dataset / "approved" / "labels" / f"{image.stem}.txt"
        shutil.copyfile(image, images_out / image.name)
        shutil.copyfile(label, labels_out / label.name)
    (output / "train.txt").write_text(
        "".join(f"images/train/{image.name}\n" for image in images), encoding="utf-8"
    )
    (output / "data.yaml").write_text("path: .\ntrain: train.txt\nnames:\n  0: product\n", encoding="utf-8")
    shutil.copyfile(_manifest_path(dataset), output / "review_manifest.json")
    shutil.copyfile(validation_report, output / "validation_report.md")
    archive_path = output / "ultralytics_yolo_segmentation.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in (output / "data.yaml", output / "train.txt"):
            archive.write(path, path.name)
        for folder in (output / "images" / "train", output / "labels" / "train"):
            for path in sorted(folder.iterdir()):
                archive.write(path, path.relative_to(output).as_posix())
    print(f"Training-format export created: {output}")
    print(f"Images: {len(images)} | approved instances: {report['approved_instances']} | review flags: {report['warnings']}")
    print("This command only exports reviewed labels. It does not train a model or create a train/val/test split.")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="Create the workspace and copy in the approved 3-photo seed batch")
    init_parser.add_argument("--dataset", type=Path, default=DATASET)

    proposal_parser = subparsers.add_parser("propose", help="Generate MobileSAM draft masks for new scene photos")
    proposal_parser.add_argument("--input-dir", type=Path, default=DATASET / "input_images")
    proposal_parser.add_argument("--image", type=Path, help="Process exactly one photo, useful for a smoke test")
    proposal_parser.add_argument("--batch-id")
    proposal_parser.add_argument("--output-dir", type=Path,
                                 help="Separate output location; by default data/segmentation_dataset/proposals/<batch-id>")

    import_parser = subparsers.add_parser("import-cvat", help="Import a manually reviewed CVAT Ultralytics YOLO Segmentation ZIP")
    import_parser.add_argument("zip", type=Path)
    import_parser.add_argument("--batch-id", required=True)
    import_parser.add_argument("--reviewed", action="store_true", help="Confirm all visible products and outlines were checked")
    import_parser.add_argument("--negative", action="append", default=[], metavar="FILENAME",
                               help="Confirm an image has no product instances; repeat for each negative image")

    validate_parser = subparsers.add_parser("validate", help="Validate approved annotations and write a report")
    validate_parser.add_argument("--dataset", type=Path, default=DATASET)

    export_parser = subparsers.add_parser("export", help="Export reviewed labels in Ultralytics YOLO Segmentation format")
    export_parser.add_argument("--dataset", type=Path, default=DATASET)
    export_parser.add_argument("--export-id")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            report = initialize_workspace(args.dataset)
            print(f"Workspace initialized: {args.dataset}")
            print(f"Approved seed: {report['images']} images, {report['approved_instances']} product instances.")
            print(f"Seed report: {args.dataset / 'reports' / 'seed_validation.md'}")
        elif args.command == "propose":
            propose(args.input_dir, args.image, args.batch_id, args.output_dir)
        elif args.command == "import-cvat":
            import_cvat(args.zip, args.batch_id, args.reviewed, args.negative)
        elif args.command == "validate":
            report = validate_dataset(args.dataset)
            path = write_validation_report(args.dataset, report)
            print(f"Images: {report['images']} | approved instances: {report['approved_instances']} | flags: {report['flagged_issues']}")
            print(f"Validation report: {path}")
            if report["errors"]:
                return 1
        elif args.command == "export":
            export_dataset(args.dataset, args.export_id)
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
