"""Render the shared edge of two reviewed COCO polygons without changing them."""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(r"C:\Users\U938\Downloads\inventoryLens_22_Scenes_Reviewed2.zip")
MAPPING = ROOT / "data/segmentation_dataset/reports/cvat_export_audit_20261016/mapping_report.json"
OUT = ROOT / "data/segmentation_dataset/proposals/cvat22_yolo_staging_20261016/review_previews/IMG_4571_overlap_intersection_detail.png"


def main():
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    with zipfile.ZipFile(ARCHIVE) as archive:
        coco = json.loads(archive.read("annotations/instances_default.json"))
    image = next(row for row in coco["images"] if row["file_name"] == "IMG_4571.jpeg")
    annotations = [row for row in coco["annotations"] if row["image_id"] == image["id"]]
    selected = [annotations[9], annotations[12]]
    source_path = Path(next(row for row in mapping["image_records"]
                            if row["file_name"] == image["file_name"])["matched_original"])
    masks = []
    for annotation in selected:
        polygon = np.asarray(annotation["segmentation"][0], dtype=np.float64).reshape(-1, 2)
        mask = np.zeros((image["height"], image["width"]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        masks.append(mask)
    intersection = cv2.bitwise_and(masks[0], masks[1])
    ys, xs = np.where(intersection > 0)
    if len(xs) == 0:
        raise ValueError("The reported overlap no longer exists in the source polygons.")
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
    margin = 90
    left, top = max(0, x0 - margin), max(0, y0 - margin)
    right, bottom = min(image["width"], x1 + margin), min(image["height"], y1 + margin)
    with Image.open(source_path) as original:
        crop = original.convert("RGB").crop((left, top, right, bottom))
    scale = min(4.0, 1800 / crop.width, 1600 / crop.height)
    crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [(30, 190, 255), (245, 45, 160)]
    for annotation, mask, color in zip(selected, masks, colors):
        polygon = np.asarray(annotation["segmentation"][0], dtype=np.float64).reshape(-1, 2)
        points = [(round((x - left) * scale), round((y - top) * scale)) for x, y in polygon]
        draw.polygon(points, fill=(*color, 48), outline=(*color, 255), width=4)
    overlap_crop = intersection[top:bottom, left:right]
    overlap_img = Image.fromarray(overlap_crop * 255).resize(crop.size, Image.Resampling.NEAREST)
    highlight = Image.new("RGBA", crop.size, (255, 245, 0, 0))
    highlight.putalpha(overlap_img.point(lambda value: 220 if value else 0))
    result = Image.alpha_composite(crop.convert("RGBA"), overlay)
    result = Image.alpha_composite(result, highlight).convert("RGB")
    footer_h = 82
    final = Image.new("RGB", (result.width, result.height + footer_h), (24, 32, 42))
    final.paste(result, (0, 0))
    try:
        title_font = ImageFont.truetype("arial.ttf", 21)
        detail_font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        title_font = detail_font = ImageFont.load_default()
    footer = ImageDraw.Draw(final)
    footer.text((10, result.height + 8), "IMG_4571 — original pixels; yellow = shared mask pixels", font=title_font, fill="white")
    footer.text((10, result.height + 42),
                f"COCO annotations {selected[0]['id']} (cyan), {selected[1]['id']} (magenta); shared pixels {int(intersection.sum())}; bbox xyxy [{x0}, {y0}, {x1}, {y1}]",
                font=detail_font, fill=(223, 229, 236))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    final.save(OUT, format="PNG", optimize=True)
    print(f"{OUT} | shared_pixels={int(intersection.sum())} | bbox_xyxy={[x0, y0, x1, y1]}")


if __name__ == "__main__":
    main()
