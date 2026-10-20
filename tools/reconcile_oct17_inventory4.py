"""Compare reviewed inventory4 COCO SKU counts with the provisional recount."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017"
DEFAULT_ZIP = Path(r"C:\Users\U938\Downloads\inventory4.zip")
RUNS = [
    "20261017-014015-b2a337ec", "20261017-014239-235d7e51",
    "20261017-014423-e7b3fb42", "20261017-014554-b9412539",
    "20261017-014722-cc4ab8d3",
]


def load_manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist()
                   if Path(name).name == "instances_default.json"]
        if len(matches) != 1:
            raise ValueError("inventory4.zip must contain one COCO instances_default.json")
        return json.loads(archive.read(matches[0]))


def annotation_mask(annotation: dict, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for flat in annotation["segmentation"]:
        points = np.rint(np.asarray(flat, dtype=np.float64).reshape(-1, 2)).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_scene(photo_path: Path, annotations: list[dict], image: dict,
               provisional: dict, out_path: Path, candidate_skus: set[str]) -> None:
    with Image.open(photo_path) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    pixels = np.asarray(photo).copy()
    font = ImageFont.load_default(size=max(14, photo.width // 75))
    width, height = photo.size
    for index, ann in enumerate(annotations):
        mask = annotation_mask(ann, width, height)
        sku = ann["sku"]
        candidate = sku in candidate_skus
        color = (255, 116, 0) if candidate else ((0, 205, 190) if index % 2 else (40, 190, 90))
        pixels[mask] = (pixels[mask].astype(np.float32) * 0.66
                        + np.asarray(color, dtype=np.float32) * 0.34).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pixels, contours, -1, color, max(2, photo.width // 900))
    photo = Image.fromarray(pixels)
    draw = ImageDraw.Draw(photo)
    for ann in annotations:
        sku = ann["sku"]
        candidate = sku in candidate_skus
        color = (255, 116, 0) if candidate else ((0, 205, 190) if annotations.index(ann) % 2 else (40, 190, 90))
        bbox = ann.get("bbox") or [0, 0, width, height]
        x, y = int(bbox[0]), int(bbox[1])
        label = f"#{ann['id']} {sku}"
        label_width = int(draw.textlength(label, font=font))
        tx = min(max(0, x), max(0, width - label_width - 8))
        ty = max(0, y - font.size - 6)
        draw.rectangle((tx, ty, tx + label_width + 8, ty + font.size + 6), fill=color)
        draw.text((tx + 4, ty + 2), label, fill=(0, 0, 0), font=font)
    registered_total = sum(1 for ann in annotations if ann["sku"] != "__undefined__")
    provisional_total = sum(provisional.values())
    caption = (f"{Path(image['file_name']).name} | reviewed registered={registered_total}; "
               f"provisional={provisional_total}; orange=SKU groups above provisional")
    draw.rectangle((0, 0, width, font.size + 12), fill="white")
    draw.text((8, 4), caption, fill="black", font=font)
    photo.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--images", type=Path, default=DEFAULT_BASE / "cvat_review/images")
    parser.add_argument("--provisional", type=Path, default=DEFAULT_BASE / "provisional_counts.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_BASE / "count_reconciliation_inventory4")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reconciliation artifacts: {args.output}")

    coco = load_manifest(args.coco_zip)
    image_by_id = {item["id"]: item for item in coco["images"]}
    if len(image_by_id) != 5:
        raise ValueError(f"Expected five image records; found {len(image_by_id)}")
    counts = defaultdict(Counter)
    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        image = image_by_id[ann["image_id"]]
        sku = (ann.get("attributes") or {}).get("sku")
        if not sku:
            raise ValueError(f"Annotation {ann.get('id')} has no SKU attribute")
        item = {**ann, "sku": sku}
        annotations_by_image[image["id"]].append(item)
        counts[Path(image["file_name"]).name][sku] += 1

    provisional = defaultdict(Counter)
    with args.provisional.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            provisional[row["run_id"]][row["sku"]] = int(row["provisional_visible_count"])

    snapshots = json.loads((DEFAULT_BASE / "provisional_ground_truth.json").read_text(encoding="utf-8"))
    # The prepared summary records the exact image basename for each run.
    scene_by_run = {scene["run_id"]: scene for scene in snapshots["scenes"]}
    run_by_file = {scene["image_filename"]: run_id for run_id, scene in scene_by_run.items()}
    if set(run_by_file.values()) != set(RUNS):
        raise ValueError("Provisional run/image mapping is incomplete or unexpected")

    comparison_rows = []
    candidate_skus_by_file = defaultdict(set)
    for image in coco["images"]:
        filename = Path(image["file_name"]).name
        run_id = run_by_file.get(filename)
        if run_id is None:
            raise ValueError(f"No verified provisional mapping for {filename}")
        all_skus = set(counts[filename]) | set(provisional[run_id])
        for sku in sorted(all_skus):
            actual = counts[filename][sku]
            before = provisional[run_id].get(sku)
            delta = actual - before if before is not None else None
            comparison_rows.append({"run_id": run_id, "image": filename, "sku": sku,
                                    "provisional_count": before, "cvat_count": actual,
                                    "cvat_minus_provisional": delta,
                                    "comparison": ("not_in_provisional_registered_counts" if before is None
                                                   else "above_provisional" if delta > 0
                                                   else "below_provisional" if delta < 0 else "matches")})
            if before is not None and delta > 0:
                candidate_skus_by_file[filename].add(sku)

    args.output.mkdir(parents=True)
    write_csv(args.output / "counts_by_image_sku.csv", comparison_rows,
              ["run_id", "image", "sku", "provisional_count", "cvat_count",
               "cvat_minus_provisional", "comparison"])

    annotation_rows = []
    for image in coco["images"]:
        filename = Path(image["file_name"]).name
        width, height = int(image["width"]), int(image["height"])
        run_id = run_by_file[filename]
        for ann in annotations_by_image[image["id"]]:
            x, y, box_w, box_h = (ann.get("bbox") or [None] * 4)
            sku = ann["sku"]
            over_count_cell = sku in candidate_skus_by_file[filename]
            annotation_rows.append({
                "run_id": run_id, "image": filename, "annotation_id": ann["id"], "sku": sku,
                "bbox_x": x, "bbox_y": y, "bbox_width": box_w, "bbox_height": box_h,
                "center_x": round(x + box_w / 2, 2) if x is not None else None,
                "center_y": round(y + box_h / 2, 2) if y is not None else None,
                "image_width": width, "image_height": height,
                "candidate_in_above_provisional_sku_group": over_count_cell,
                "candidate_note": ("This entire repeated-SKU group is above the provisional count; "
                                   "the export alone cannot identify which member is excess."
                                   if over_count_cell else ""),
            })
    write_csv(args.output / "annotations_and_candidate_locations.csv", annotation_rows,
              ["run_id", "image", "annotation_id", "sku", "bbox_x", "bbox_y", "bbox_width",
               "bbox_height", "center_x", "center_y", "image_width", "image_height",
               "candidate_in_above_provisional_sku_group", "candidate_note"])

    preview_paths = []
    contact_tiles = []
    for image in coco["images"]:
        filename = Path(image["file_name"]).name
        if not candidate_skus_by_file[filename]:
            continue
        path = args.output / f"{Path(filename).stem}_count_review.png"
        draw_scene(args.images / filename, annotations_by_image[image["id"]], image,
                   provisional[run_by_file[filename]], path, candidate_skus_by_file[filename])
        preview_paths.append(path.name)
        with Image.open(path) as preview:
            tile = preview.copy()
            tile.thumbnail((900, 900))
            contact_tiles.append((filename, tile.copy()))

    tile_width = 900
    tile_height = 700
    cols = 2
    rows = (len(contact_tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_width, rows * tile_height), "#f5f5f5")
    sheet_draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default(size=18)
    for index, (filename, tile) in enumerate(contact_tiles):
        x = (index % cols) * tile_width
        y = (index // cols) * tile_height
        sheet_draw.text((x + 8, y + 6), filename, fill="black", font=label_font)
        tile.thumbnail((tile_width - 16, tile_height - 42))
        sheet.paste(tile, (x + 8, y + 32))
    sheet_path = args.output / "over_count_candidate_scenes_contact_sheet.png"
    sheet.save(sheet_path)

    registered_provisional = sum(sum(provisional[run].values()) for run in RUNS)
    registered_cvat = sum(count for filename, row in counts.items()
                          for sku, count in row.items() if sku != "__undefined__")
    positive_deltas = sum(row["cvat_minus_provisional"] for row in comparison_rows
                          if row["cvat_minus_provisional"] is not None and row["cvat_minus_provisional"] > 0)
    negative_deltas = sum(-row["cvat_minus_provisional"] for row in comparison_rows
                          if row["cvat_minus_provisional"] is not None and row["cvat_minus_provisional"] < 0)
    undefined = sum(count for row in counts.values() for sku, count in row.items()
                    if sku == "__undefined__")
    summary = {
        "reviewed_export": str(args.coco_zip.resolve()), "images": len(coco["images"]),
        "annotations": len(coco["annotations"]), "registered_cvat_annotations": registered_cvat,
        "explicit_undefined_annotations": undefined,
        "provisional_registered_count": registered_provisional,
        "net_registered_difference": registered_cvat - registered_provisional,
        "positive_cell_deltas": positive_deltas, "negative_cell_deltas": -negative_deltas,
        "interpretation": ("There is no unique one-extra annotation identifiable from the aggregate +1. "
                           "Multiple image/SKU groups are above provisional and others are below it; "
                           "all orange objects in those groups are candidates for human confirmation."),
        "candidate_preview_files": preview_paths, "contact_sheet": sheet_path.name,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "README.md").write_text(
        "# inventory4 Count Reconciliation\n\n"
        f"Reviewed export: `{args.coco_zip.resolve()}`.\n\n"
        f"Registered annotations: {registered_cvat}; provisional registered count: "
        f"{registered_provisional}; net difference: {registered_cvat - registered_provisional:+d}. "
        f"The positive per-cell differences total +{positive_deltas}; negative cells total "
        f"{negative_deltas}, so no single annotation can be identified as the unique extra from "
        "the overall +1. The full above-count repeated-SKU groups are highlighted orange.\n\n"
        "Review `counts_by_image_sku.csv` and `annotations_and_candidate_locations.csv` for the "
        "exact per-scene counts, annotation IDs and pixel bounding-box centers. The highlighted "
        "members are candidates, not a claim that each is wrong; the CVAT export is unchanged.\n\n"
        "Preview files: " + ", ".join(f"`{name}`" for name in preview_paths) +
        f". Contact sheet: `{sheet_path.name}`.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
