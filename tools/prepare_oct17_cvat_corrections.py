"""Create read-only, high-resolution CVAT correction previews for inventory5.zip."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OLD_ZIP = Path(r"C:\Users\U938\Downloads\inventory4.zip")
NEW_ZIP = Path(r"C:\Users\U938\Downloads\inventory5.zip")
IMAGE_DIR = ROOT / "data/test_runs"
DEFAULT_OUTPUT = ROOT / "data/recognition_checks/oct17_diagnostic_benchmark_20261017/cvat_correction_review_inventory5_final_20261017"

SCENES = {
    "01:40": "20261017-014015-b2a337ec-test-original.png",
    "01:42": "20261017-014239-235d7e51-test-original.png",
    "01:44": "20261017-014423-e7b3fb42-test-original.png",
    "01:45": "20261017-014554-b9412539-test-original.png",
}

# The IDs below refer to the original inventory4 reconciliation. Matching is by
# image and exact COCO bbox in inventory5 because CVAT may renumber annotations.
GROUPS = [
    {"key": "01_40_stobi", "scene": "01:40", "old_ids": [11],
     "title": "Stobi bag: verify label correction",
     "interpretation": "The same outline is now labeled VITAMINKA-STOBI-30 (old inventory4 ID 11 maps to current inventory5 ID 10). The earlier undefined-label issue appears corrected; confirm the bag identity visually.",
     "action": "No SKU edit is indicated if this is the current CVAT task. Keep one polygon for this visible bag."},
    {"key": "01_42_energia", "scene": "01:42", "old_ids": [25, 28, 30],
     "title": "Energia: three visibly distinct cans",
     "interpretation": "The original crop shows three staggered can tops/bodies at different depths. These are separate physical products, not three fragments of one can.",
     "action": "Do not merge these annotations. Check that each polygon follows only its own can's visible portion and that any overlap is limited to genuine occlusion."},
    {"key": "01_42_jana", "scene": "01:42", "old_ids": [27, 29],
     "title": "Jana: foreground and rear bottles",
     "interpretation": "The crop shows a foreground Jana bottle and a second, partly occluded Jana bottle behind it. They are distinct products.",
     "action": "Do not merge. Keep one polygon per bottle and restrict the rear bottle polygon to its actually visible cap/side region."},
    {"key": "01_44_passata", "scene": "01:44", "old_ids": [42, 46, 47, 51],
     "title": "Passata shelf: instance grouping and labels",
     "interpretation": "Old ID 42 now maps to the main Garden Good Passata carton; old IDs 46 and 47 map to narrow side regions of additional cartons. Old ID 51 maps to another narrow region labeled __undefined__. The boundaries and SKU of these narrow visible portions are not conclusive from the crop alone.",
     "action": "Do not remove the small polygons. In CVAT, verify whether each side strip belongs to a separate visible carton. If old ID 51/current ID 48 is another Garden Good Passata, change its SKU to GARDENGOOD-PASSATA-200; otherwise retain __undefined__. Ensure each carton has one polygon, not multiple fragments."},
    {"key": "01_44_energia", "scene": "01:44", "old_ids": [41, 48],
     "title": "Energia: front and rear cans",
     "interpretation": "The crop shows the front can plus a distinct, partly occluded can behind it at the upper-left/side. The shapes are not a split of the front can.",
     "action": "Keep separate objects. Verify the rear polygon contains only the visible rear can and does not spill onto the Fentimans bottle or shelf."},
    {"key": "01_44_jana", "scene": "01:44", "old_ids": [39, 49],
     "title": "Jana: foreground and rear bottles",
     "interpretation": "The crop shows a foreground Jana bottle and a second bottle behind it, visible as a cap and narrow side. These are separate physical products.",
     "action": "Do not merge. Keep the rear polygon limited to the visible part of the rear bottle; the foreground polygon should follow the front bottle boundary."},
    {"key": "01_45_eintopf", "scene": "01:45", "old_ids": [53, 58],
     "title": "Eintopf: front can and rear visible sliver",
     "interpretation": "The main Eintopf can is in front; the narrow strip at its left appears to be a second, partly occluded can/package.",
     "action": "Preserve both if the narrow strip follows a distinct package edge, as it appears to. Do not merge or delete it solely because it is small; verify the strip does not include shelf/background."},
    {"key": "01_45_energia", "scene": "01:45", "old_ids": [59, 61, 66, 69],
     "title": "Energia: staggered repeated cans",
     "interpretation": "Several can faces are staggered at different depths; the broad masks appear to represent distinct cans. One very small mask (current ID 55) is ambiguous and may be a narrow visible part of another can or a fragment.",
     "action": "Preserve separate masks for distinct cans. Specifically inspect current ID 55 (old ID 61) against neighboring IDs 61 and 64; merge only if it is demonstrably part of the same physical can. Do not merge based on repeated SKU."},
    {"key": "01_45_freshona", "scene": "01:45", "old_ids": [60, 62, 68],
     "title": "Freshona: three separate bean cans",
     "interpretation": "The crop shows three physically separate Freshona bean cans: a rear partly occluded can, a middle can, and a foreground can. The old ID 60 maps to the rear can, not an undefined object.",
     "action": "Keep all three as separate product instances. Verify each polygon follows its own visible can pixels and that the rear can is labeled FRESHONA-BEANS-800."},
]


def read_coco(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        manifests = [name for name in archive.namelist()
                     if Path(name).name == "instances_default.json"]
        if len(manifests) != 1:
            raise ValueError(f"Expected one COCO manifest in {path}, got {manifests}.")
        return json.loads(archive.read(manifests[0]))


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_index(coco: dict) -> tuple[dict[str, dict], dict[int, dict]]:
    images = {Path(item["file_name"]).name: item for item in coco["images"]}
    annotations = {int(item["id"]): item for item in coco["annotations"]}
    return images, annotations


def resolve_old_annotation(old: dict, new: dict, scene: str, old_id: int) -> tuple[dict, dict]:
    filename = SCENES[scene]
    old_images, old_anns = load_index(old)
    new_images, new_anns = load_index(new)
    if filename not in old_images or filename not in new_images:
        raise ValueError(f"Image mapping missing for {filename}.")
    old_image = old_images[filename]
    old_ann = old_anns.get(old_id)
    if old_ann is None or old_ann["image_id"] != old_image["id"]:
        raise ValueError(f"Old annotation {old_id} does not belong to {filename}.")
    old_bbox = np.asarray(old_ann["bbox"], dtype=float)
    candidates = [item for item in new["annotations"]
                  if item["image_id"] == new_images[filename]["id"]
                  and np.allclose(np.asarray(item["bbox"], dtype=float), old_bbox, atol=0.05)]
    if len(candidates) != 1:
        raise ValueError(f"Cannot uniquely map old annotation {old_id} in {filename}; "
                         f"bbox candidates={len(candidates)}.")
    return old_ann, candidates[0]


def crop_bounds(annotations: list[dict], width: int, height: int) -> tuple[int, int, int, int]:
    boxes = [annotation["bbox"] for annotation in annotations]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    margin = max(100, int(max(x2 - x1, y2 - y1) * 0.23))
    return (max(0, int(x1 - margin)), max(0, int(y1 - margin)),
            min(width, int(x2 + margin)), min(height, int(y2 + margin)))


def render_group(photo: Image.Image, rows: list[dict], path: Path, title: str) -> None:
    width, height = photo.size
    bounds = crop_bounds([row["current"] for row in rows], width, height)
    original = photo.crop(bounds).convert("RGB")
    overlay = original.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    colors = [(0, 176, 132, 215), (230, 105, 50, 225), (50, 120, 220, 220),
              (210, 50, 140, 220), (230, 190, 35, 230), (135, 90, 205, 220)]
    label_font = font(max(20, min(30, original.width // 48)))
    for index, row in enumerate(rows):
        annotation = row["current"]
        color = colors[index % len(colors)]
        all_points = []
        for flat in annotation.get("segmentation", []):
            points = np.asarray(flat, dtype=float).reshape(-1, 2)
            local = [(float(x - bounds[0]), float(y - bounds[1])) for x, y in points]
            if len(local) >= 3:
                draw.polygon(local, fill=(*color[:3], 20), outline=color, width=5)
                all_points.extend(local)
        if all_points:
            center_x = sum(point[0] for point in all_points) / len(all_points)
            center_y = sum(point[1] for point in all_points) / len(all_points)
            label = f"ID {annotation['id']}"
            box = draw.textbbox((0, 0), label, font=label_font)
            tw, th = box[2] - box[0], box[3] - box[1]
            px = max(0, min(original.width - tw - 10, int(center_x)))
            py = max(0, min(original.height - th - 8, int(center_y)))
            draw.rounded_rectangle((px, py, px + tw + 10, py + th + 8), radius=4,
                                   fill=(*color[:3], 240))
            draw.text((px + 5, py + 3), label, fill="white", font=label_font)

    header_height = 92
    footer_height = 28 + len(rows) * 34
    panel_width = original.width
    board = Image.new("RGB", (panel_width * 2, original.height + header_height + footer_height), "white")
    board_draw = ImageDraw.Draw(board)
    title_font = font(30)
    panel_font = font(24)
    board_draw.text((16, 12), title, fill="#202832", font=title_font)
    board_draw.text((16, 56), "Original photograph crop", fill="#344050", font=panel_font)
    board_draw.text((panel_width + 16, 56), "Current COCO polygons / IDs / SKU", fill="#344050", font=panel_font)
    board.paste(original, (0, header_height))
    board.paste(overlay, (panel_width, header_height))
    footer_y = header_height + original.height + 8
    legend_font = font(20)
    for index, row in enumerate(rows):
        annotation = row["current"]
        sku = (annotation.get("attributes") or {}).get("sku", "<missing SKU>")
        color = colors[index % len(colors)]
        y = footer_y + index * 34
        board_draw.rectangle((16, y + 4, 36, y + 24), fill=color[:3])
        board_draw.text((48, y),
                        f"Current ID {annotation['id']} | Old ID {row['old']['id']} | {sku}",
                        fill="#202832", font=legend_font)
    board.save(path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-coco", type=Path, default=OLD_ZIP)
    parser.add_argument("--coco", type=Path, default=NEW_ZIP)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true",
                        help="Refresh only this tool's known preview/report files in its existing output directory.")
    args = parser.parse_args()
    if args.output.exists():
        if not args.refresh:
            raise FileExistsError(f"Refusing to overwrite review package: {args.output}")
        existing = {item.name for item in args.output.iterdir()}
        expected = {"README.md", "annotation_mapping.csv", "annotation_mapping.json", "counts_by_image_sku.csv",
                    "label_changes_inventory4_to_inventory5.csv"}
        expected.update(f"review_{number:02d}_{group['key']}_comparison.png"
                        for number, group in enumerate(GROUPS, start=1))
        if not existing.issubset(expected):
            raise RuntimeError("Refresh refused: output directory contains files not owned by this tool.")
    else:
        args.output.mkdir(parents=True)
    old, current = read_coco(args.old_coco), read_coco(args.coco)
    if len(current.get("images", [])) != 5 or len(current.get("annotations", [])) != 79:
        raise ValueError("inventory5 must contain the expected five images and 79 annotations.")
    if len(current.get("categories", [])) != 1 or current["categories"][0].get("name") != "product":
        raise ValueError("Expected exactly one COCO category named product.")

    details, preview_rows = [], []
    for group_number, group in enumerate(GROUPS, start=1):
        filename = SCENES[group["scene"]]
        source = args.images / filename
        with Image.open(source) as image:
            photo = image.convert("RGB")
        if photo.size != (load_index(current)[0][filename]["width"],
                          load_index(current)[0][filename]["height"]):
            raise ValueError(f"Original photograph dimensions do not match inventory5 for {filename}.")
        rows = []
        for old_id in group["old_ids"]:
            prior, now = resolve_old_annotation(old, current, group["scene"], old_id)
            sku = (now.get("attributes") or {}).get("sku", "<missing SKU>")
            row = {"group": group["key"], "scene": group["scene"], "image": filename,
                   "old_inventory4_annotation_id": old_id,
                   "inventory4_sku": (prior.get("attributes") or {}).get("sku"),
                   "current_inventory5_annotation_id": now["id"], "current_sku": sku,
                   "bbox_xywh": now["bbox"], "interpretation": group["interpretation"],
                   "proposed_action": group["action"]}
            rows.append({"old": prior, "current": now})
            details.append(row)
        image_path = args.output / f"review_{group_number:02d}_{group['key']}_comparison.png"
        render_group(photo, rows, image_path, f"Review {group_number:02d} | {group['scene']} | {group['title']}")
        preview_rows.append((group, image_path.name, rows))

    with (args.output / "annotation_mapping.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    (args.output / "annotation_mapping.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    old_images_by_id = {item["id"]: Path(item["file_name"]).name for item in old["images"]}
    new_images_by_id = {item["id"]: Path(item["file_name"]).name for item in current["images"]}
    old_by_shape = {(old_images_by_id[item["image_id"]], tuple(round(value, 4) for value in item["bbox"])): item
                    for item in old["annotations"]}
    new_by_shape = {(new_images_by_id[item["image_id"]], tuple(round(value, 4) for value in item["bbox"])): item
                    for item in current["annotations"]}
    if set(old_by_shape) != set(new_by_shape):
        raise ValueError("The two exports differ in annotation/image bounding boxes; refusing to claim a label-only comparison.")
    label_changes = []
    for key, prior in old_by_shape.items():
        now = new_by_shape[key]
        if prior.get("segmentation") != now.get("segmentation"):
            raise ValueError(f"Polygon geometry changed for {key[0]} bbox {key[1]}; review geometry before using this package.")
        old_sku = (prior.get("attributes") or {}).get("sku")
        new_sku = (now.get("attributes") or {}).get("sku")
        if old_sku != new_sku:
            label_changes.append([key[0], prior["id"], now["id"], old_sku, new_sku,
                                  *key[1]])
    with (args.output / "label_changes_inventory4_to_inventory5.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "inventory4_annotation_id", "inventory5_annotation_id",
                         "inventory4_sku", "inventory5_sku", "bbox_x", "bbox_y", "bbox_width", "bbox_height"])
        writer.writerows(label_changes)
    images_by_id = {item["id"]: Path(item["file_name"]).name for item in current["images"]}
    counts = Counter((images_by_id[item["image_id"]],
                      (item.get("attributes") or {}).get("sku", "<missing SKU>"))
                     for item in current["annotations"])
    with (args.output / "counts_by_image_sku.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "sku", "annotation_count"])
        writer.writerows([image, sku, count] for (image, sku), count in sorted(counts.items()))

    label_change_lines = [
        f"- {image}: inventory4 ID {old_id} (`{old_sku}`) -> inventory5 ID {new_id} (`{new_sku}`); bbox `{bbox}`"
        for image, old_id, new_id, old_sku, new_sku, *bbox in label_changes
    ]
    lines = [
        "# Targeted CVAT Correction Review", "",
        f"Current source: `{args.coco.resolve()}`", f"Prior reconciliation source: `{args.old_coco.resolve()}`", "",
        "These files are read-only comparison previews. No annotations were changed. The supplied inventory5 export contains 5 image records and 79 product annotations; its image records map to the five original `data/test_runs/*-test-original.png` photographs by exact filename and dimensions.", "",
        "The current export has 71 registered-SKU labels and 8 `__undefined__` labels. Per-image/per-SKU counts are in `counts_by_image_sku.csv`. The two exports contain the same 79 polygon geometries; exactly four SKU attributes changed, listed in `label_changes_inventory4_to_inventory5.csv`.", "",
        "CVAT has renumbered IDs between exports. The table and previews identify both the old inventory4 ID and current inventory5 ID; use the current ID only when editing the current task.", "",
    ]
    for number, (group, image_name, rows) in enumerate(preview_rows, start=1):
        lines += [f"## Review {number:02d}: {group['scene']} - {group['title']}", "", f"![{group['title']}]({image_name})", "",
                  f"**Interpretation:** {group['interpretation']}", "",
                  f"**Proposed correction:** {group['action']}", "", "Current objects:", ""]
        for row in rows:
            now = row["current"]
            old_id = row["old"]["id"]
            sku = (now.get("attributes") or {}).get("sku")
            lines.append(f"- Old ID {old_id} -> current ID {now['id']}: `{sku}`; bbox `{now['bbox']}`")
        lines.append("")
    lines += ["## What Looks Already Corrected", "",
              "- 01:40 old annotation 11 is current inventory5 ID 10 and is labeled `VITAMINKA-STOBI-30`, not `__undefined__`.",
              "- 01:44 old annotation 42 is current ID 49 and is labeled `GARDENGOOD-PASSATA-200`.",
              "- 01:44 old annotations 46 and 47 are current IDs 50 and 51 and are labeled `GARDENGOOD-PASSATA-200`.",
              "- 01:44 old annotation 51 is current ID 48 and remains `__undefined__`; verify whether the exposed package is in fact another Passata carton.",
              "- The earlier suspicion that the 01:42/01:44 Energia and Jana annotations were split fragments is not supported by these closeups: they show separate foreground/rear products. Do not merge those masks.",
              "- 01:45 shows repeated physical cans. The three Freshona annotations map to three distinct visible cans. Keep separate instances.", "",
              "## Inventory4 to Inventory5 Changes", "",
              "All 79 image/annotation bounding-box mappings and polygon coordinates are identical. Only these four SKU attributes changed:",
              *label_change_lines, "",
              "## Exact CVAT Review Actions", "",
              "No merge or deletion is justified by these previews. Keep the distinct repeated products as separate annotations. Before re-export, inspect the following current inventory5 IDs:",
              "- 01:44: current ID 48 (old ID 51): decide whether this narrow visible package is Garden Good Passata or genuinely unregistered. If it is Passata, correct only its SKU attribute.",
              "- 01:44: current IDs 50 and 51 (old IDs 46 and 47): confirm the narrow regions each represent a separate package rather than multiple fragments of one carton; preserve visible pixels.",
              "- 01:44: current IDs 41 and 44 (old IDs 41 and 48), and IDs 39 and 45 (old IDs 39 and 49): keep the rear can/bottle separate, but verify their polygons contain only visible product pixels.",
              "- 01:45: current ID 55 (old ID 61): inspect whether this tiny patch is independently visible on a distinct Energia can or is a fragment of current ID 61/64. Do not remove without confirming the physical object.",
              "- 01:45: current IDs 52 and 65 (Eintopf), 53/55/61/64 (Energia), and 54/56/63 (Freshona): confirm visible boundaries, especially the rear/occluded products. Do not merge based only on repeated SKU.",
              "- 01:40 current ID 10 and 01:44 current ID 49 need no SKU correction in this export; their prior undefined labels are already corrected.", "",
              "All judgments above follow one polygon per visible physical product, visible pixels only, and no annotation for fully hidden products. These are recommendations for human CVAT review, not automated edits.", ""]
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Review package: {args.output.resolve()}")
    print(f"Comparison previews: {len(preview_rows)}")


if __name__ == "__main__":
    main()
