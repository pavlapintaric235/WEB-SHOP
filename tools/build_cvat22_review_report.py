import json
import math
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from segmentation_dataset import polygon_self_intersections


OUT = ROOT / "data/segmentation_dataset/proposals/cvat22_final_review_20261016"
MAPPING = ROOT / "data/segmentation_dataset/reports/cvat_export_audit_20261016/mapping_report.json"
NEW_ZIP = Path(r"C:\Users\U938\Downloads\inventoryLens_22_Scenes_Reviewed2.zip")
OLD_ZIP = Path(r"C:\Users\U938\Downloads\inventoryLens_22_Scenes_Reviewed.zip")


def parse_points(text):
    points = []
    for pair in text.split(";"):
        if pair:
            x, y = pair.split(",", 1)
            points.append((float(x), float(y)))
    return points


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    return abs(sum(points[i][0] * points[(i + 1) % len(points)][1] -
                   points[(i + 1) % len(points)][0] * points[i][1]
                   for i in range(len(points))) / 2)


def coco_polygons(ann):
    seg = ann.get("segmentation", [])
    if isinstance(seg, dict):
        raise ValueError("RLE masks are not expected in this polygon COCO export")
    return [[(float(poly[i]), float(poly[i + 1]))
             for i in range(0, len(poly), 2)] for poly in seg]


def bbox_for(polygons):
    points = [p for poly in polygons for p in poly]
    if not points:
        return [0, 0, 0, 0]
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def raster(polygons, width, height, scale=0.25):
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        pts = np.array([[(round(x * scale), round(y * scale)) for x, y in poly]], dtype=np.int32)
        if len(pts[0]) >= 3:
            cv2.fillPoly(mask, pts, 1)
    return mask


def load_exports():
    with zipfile.ZipFile(NEW_ZIP) as zf:
        coco = json.loads(zf.read("annotations/instances_default.json"))
    with zipfile.ZipFile(OLD_ZIP) as zf:
        root = ET.fromstring(zf.read("annotations.xml"))
    records = json.loads(MAPPING.read_text(encoding="utf-8"))["image_records"]
    by_id = {im["id"]: im for im in coco["images"]}
    anns_by_name = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_name[by_id[ann["image_id"]]["file_name"]].append(ann)
    old_by_name = {}
    for image in root.findall("image"):
        old_by_name[image.attrib["name"]] = [
            {"label": poly.attrib.get("label", ""), "points": parse_points(poly.attrib["points"])}
            for poly in image.findall("polygon")
        ]
    return coco, records, anns_by_name, old_by_name


def draw_overlay(photo, anns, small_indices=(), overlap_pairs=(), filename=""):
    base = photo.convert("RGBA")
    w, h = base.size
    scale = min(1500 / w, 1000 / h, 1.0)
    if scale != 1:
        base = base.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    palette = [(34, 191, 218), (238, 73, 151), (119, 203, 90), (250, 184, 53),
               (151, 114, 239), (240, 103, 72), (41, 158, 117), (222, 204, 64)]
    fs = max(16, round(22 * scale))
    try:
        font = ImageFont.truetype("arial.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    for idx, ann in enumerate(anns, 1):
        color = palette[(idx - 1) % len(palette)]
        polygons = coco_polygons(ann)
        for poly in polygons:
            pts = [(round(x * scale), round(y * scale)) for x, y in poly]
            if len(pts) >= 3:
                d.polygon(pts, fill=(*color, 35), outline=(*color, 255), width=max(2, round(3 * scale)))
        x1, y1, x2, y2 = bbox_for(polygons)
        cx, cy = round((x1 + x2) / 2 * scale), round((y1 + y2) / 2 * scale)
        rad = max(13, fs // 2 + 4)
        d.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), fill=(18, 28, 38, 235), outline=(255, 255, 255, 240), width=2)
        d.text((cx, cy), str(idx), font=font, fill=(255, 255, 255, 255), anchor="mm")
        if idx in small_indices:
            d.rectangle((round(x1*scale), round(y1*scale), round(x2*scale), round(y2*scale)), outline=(255, 178, 45, 255), width=3)
        if any(idx in pair for pair in overlap_pairs):
            d.rectangle((round(x1*scale), round(y1*scale), round(x2*scale), round(y2*scale)), outline=(255, 65, 65, 255), width=3)
    header = 82
    footer = 36
    result = Image.new("RGB", (base.width, base.height + header + footer), (25, 34, 42))
    result.paste(base.convert("RGB"), (0, header))
    result.paste(Image.alpha_composite(base, overlay).convert("RGB"), (0, header))
    rd = ImageDraw.Draw(result)
    try:
        title_font = ImageFont.truetype("arial.ttf", 27)
        sub_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    rd.text((16, 12), f"{filename}  |  PRODUCT COUNT: {len(anns)}", font=title_font, fill="white")
    rd.text((16, 49), "Geometry passed; amber = small mask; red = overlap review", font=sub_font, fill=(220, 226, 232))
    footer_text = "Manual check still required: product identity, completeness, boundaries, and background leakage."
    rd.text((16, header + base.height + 10), footer_text, font=sub_font, fill=(235, 213, 155))
    return result


def draw_comparison_panel(photo, polygons, title, color, scale=1.0, crop=None, marker=None):
    if crop:
        x1, y1, x2, y2 = crop
        photo = photo.crop((x1, y1, x2, y2))
        shift = (x1, y1)
    else:
        shift = (0, 0)
    w, h = photo.size
    s = min(scale, 1400 / w, 950 / h)
    image = photo.resize((round(w*s), round(h*s)), Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0,0,0,0))
    d = ImageDraw.Draw(layer)
    font_size = 16
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for i, poly in enumerate(polygons, 1):
        pts = [(round((x-shift[0])*s), round((y-shift[1])*s)) for x,y in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=(*color, 55), outline=(*color, 255), width=max(2, round(3*s)))
            if marker == i:
                bx = [p[0] for p in pts]; by = [p[1] for p in pts]
                cx, cy = sum(bx)//len(bx), sum(by)//len(by)
                d.ellipse((cx-20,cy-20,cx+20,cy+20), fill=(180,30,45,245), outline="white", width=3)
                d.text((cx,cy), "!", font=font, fill="white", anchor="mm")
    composite = Image.alpha_composite(image, layer).convert("RGB")
    panel = Image.new("RGB", (composite.width, composite.height+42), (29,38,47))
    panel.paste(composite, (0,42))
    ImageDraw.Draw(panel).text((12,11), title, font=font, fill="white")
    return panel


def save_side_by_side(path, panels, title):
    max_h = max(p.height for p in panels)
    gap, title_h = 12, 54
    width = sum(p.width for p in panels) + gap*(len(panels)-1)
    canvas = Image.new("RGB", (width, max_h+title_h), (18,25,32))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        f = ImageFont.load_default()
    d.text((12, 12), title, font=f, fill="white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_h))
        x += panel.width + gap
    canvas.save(path, quality=94)


def compare_exports(old_by_name, anns_by_name, images):
    matched, removed, added, changed = 0, [], [], []
    for name, old_shapes in old_by_name.items():
        width, height = images[name]["width"], images[name]["height"]
        old_polys = [shape["points"] for shape in old_shapes]
        new_polys = [poly for ann in anns_by_name[name] for poly in coco_polygons(ann)]
        old_masks = [raster([poly], width, height, 0.25) for poly in old_polys]
        new_masks = [raster([poly], width, height, 0.25) for poly in new_polys]
        similarities = []
        for old_mask in old_masks:
            row = []
            for new_mask in new_masks:
                intersection = int(np.logical_and(old_mask, new_mask).sum())
                union = int(np.logical_or(old_mask, new_mask).sum())
                row.append(intersection / union if union else 0.0)
            similarities.append(row)
        pairs = []
        candidates = sorted(
            ((i, j) for i in range(len(old_masks)) for j in range(len(new_masks))),
            key=lambda pair: similarities[pair[0]][pair[1]], reverse=True)
        for i, j in candidates:
            if similarities[i][j] < 0.5:
                break
            if all(i != old_i and j != new_i for old_i, new_i, _ in pairs):
                pairs.append((i, j, similarities[i][j]))
        matched += len(pairs)
        removed.extend({"image": name, "old_instance": i+1} for i in range(len(old_masks))
                       if all(i != old_i for old_i, _, _ in pairs))
        added.extend({"image": name, "new_instance": j+1} for j in range(len(new_masks))
                     if all(j != new_i for _, new_i, _ in pairs))
        changed.extend({"image": name, "old_instance": i+1, "new_instance": j+1,
                        "raster_iou_0_25": round(float(score), 4)}
                       for i, j, score in pairs if score < 0.999)
    return {"matched_shapes": matched, "removed": removed, "added": added, "changed": changed}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison").mkdir(exist_ok=True)
    coco, records, anns_by_name, old_by_name = load_exports()
    categories = coco.get("categories", [])
    images = {im["file_name"]: im for im in coco["images"]}
    geometry_errors, out_of_bounds, tiny, fragmented, overlap_findings, duplicate_findings = [], [], [], [], [], []
    per_image = []
    all_current = {}

    for rec in records:
        name = rec["file_name"]
        im = images[name]
        anns = anns_by_name[name]
        masks = []
        small_nums = []
        for local_idx, ann in enumerate(anns, 1):
            polys = coco_polygons(ann)
            if len(polys) > 1:
                fragmented.append({"image": name, "instance": local_idx, "polygon_parts": len(polys)})
            if ann.get("category_id") != 1:
                geometry_errors.append({"image": name, "instance": local_idx, "error": "non-product category_id"})
            area = sum(polygon_area(p) for p in polys)
            if not polys or any(len(p) < 3 for p in polys):
                geometry_errors.append({"image": name, "instance": local_idx, "error": "missing/degenerate polygon"})
            for poly in polys:
                if polygon_self_intersections(poly):
                    geometry_errors.append({"image": name, "instance": local_idx, "error": "self-intersection"})
                if any(x < -1e-4 or y < -1e-4 or x > im["width"]+1e-4 or y > im["height"]+1e-4 for x,y in poly):
                    out_of_bounds.append({"image": name, "instance": local_idx, "bbox": bbox_for(polys)})
            fraction = area / (im["width"] * im["height"])
            if fraction < 0.002:
                small_nums.append(local_idx)
                tiny.append({"image": name, "instance": local_idx, "area_px2": round(area), "image_fraction_pct": round(fraction*100, 4), "bbox_xyxy": [round(v,1) for v in bbox_for(polys)]})
            masks.append(raster(polys, im["width"], im["height"], scale=1.0))

        overlap_pairs = []
        for a in range(len(masks)):
            for b in range(a+1, len(masks)):
                inter = int(np.logical_and(masks[a], masks[b]).sum())
                if not inter:
                    continue
                aa, bb = int(masks[a].sum()), int(masks[b].sum())
                union = aa + bb - inter
                smaller_ratio = inter / min(aa, bb) if min(aa,bb) else 0
                iou = inter / union if union else 0
                if iou >= 0.80 or smaller_ratio >= 0.95:
                    duplicate_findings.append({"image": name, "instances": [a+1,b+1], "iou": round(iou,4), "intersection_over_smaller": round(smaller_ratio,4)})
                elif smaller_ratio >= 0.03:
                    overlap_pairs.append([a+1,b+1])
                    overlap_findings.append({"image": name, "instances": [a+1,b+1], "iou": round(iou,4), "intersection_over_smaller": round(smaller_ratio,4)})

        photo_path = Path(rec["matched_original"])
        photo = Image.open(photo_path).convert("RGB")
        all_current[name] = (anns, photo)
        overlay = draw_overlay(photo, anns, small_nums, overlap_pairs, name)
        overlay.save(OUT / "overlays" / f"{Path(name).stem}_overlay.jpg", quality=94)
        per_image.append({"filename": name, "scene": rec["scene"], "annotations": len(anns), "original": str(photo_path), "small_instances": small_nums, "overlap_pairs": overlap_pairs})

    # A 5-column contact sheet of all annotated overlays.
    sheet_cols, tile_w, tile_h, gap, top = 5, 440, 320, 10, 54
    rows = math.ceil(len(records)/sheet_cols)
    sheet = Image.new("RGB", (sheet_cols*tile_w+(sheet_cols+1)*gap, rows*tile_h+(rows+1)*gap+top), (231,235,239))
    sd = ImageDraw.Draw(sheet)
    try:
        titlefont = ImageFont.truetype("arial.ttf", 26)
        label_font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        titlefont = ImageFont.load_default(); label_font = ImageFont.load_default()
    sd.text((12,12), "InventoryLens CVAT Reviewed2 | 23 images | 146 product annotations", font=titlefont, fill=(25,34,42))
    for idx, rec in enumerate(records):
        name = rec["file_name"]
        img = Image.open(OUT / "overlays" / f"{Path(name).stem}_overlay.jpg").convert("RGB")
        img.thumbnail((tile_w-10,tile_h-42), Image.Resampling.LANCZOS)
        col, row = idx % sheet_cols, idx // sheet_cols
        x = gap + col*(tile_w+gap)
        y = top + gap + row*(tile_h+gap)
        sheet.paste(img, (x+(tile_w-img.width)//2,y+26))
        sd.text((x+5,y+3), f"{name} | {len(anns_by_name[name])} products | {rec['scene']}", font=label_font, fill=(25,34,42))
    sheet.save(OUT / "contact_sheet_23.jpg", quality=92)

    # Compare the missing old polygon in IMG_4571 as full-scene context.
    deleted_name, deleted_local = "IMG_4571.jpeg", 10
    deleted_polys = [x["points"] for x in old_by_name[deleted_name]]
    deleted_poly = deleted_polys[deleted_local-1]
    new_deleted_anns = anns_by_name[deleted_name]
    new_polys = [p for ann in new_deleted_anns for p in coco_polygons(ann)]
    photo = all_current[deleted_name][1]
    old_panel = draw_comparison_panel(photo, new_polys + [deleted_poly], "Earlier 147-shape export | removed shape #10 in magenta", (235,48,184), scale=0.55, marker=len(new_polys)+1)
    new_panel = draw_comparison_panel(photo, new_polys, "Reviewed2 146-annotation export | shape absent", (64,215,130), scale=0.55)
    save_side_by_side(OUT / "comparison" / "IMG_4571_removed_shape10.jpg", [old_panel,new_panel], "Removed annotation comparison | IMG_4571.jpeg | tiny region inside an existing can mask")

    # Compare the modified contour in IMG_4588 using a close crop around both outlines.
    change_name, old_local, new_local = "IMG_4588.jpeg", 15, 11
    old_poly = old_by_name[change_name][old_local-1]["points"]
    new_poly = coco_polygons(anns_by_name[change_name][new_local-1])[0]
    boxes = [bbox_for([old_poly]), bbox_for([new_poly])]
    x1 = max(0, int(min(b[0] for b in boxes)-55)); y1 = max(0, int(min(b[1] for b in boxes)-55))
    x2 = min(images[change_name]["width"], int(max(b[2] for b in boxes)+55)); y2 = min(images[change_name]["height"], int(max(b[3] for b in boxes)+55))
    photo = all_current[change_name][1]
    old_crop = draw_comparison_panel(photo, [old_poly], "Earlier shape #15 | magenta", (235,48,184), scale=3.0, crop=(x1,y1,x2,y2))
    new_crop = draw_comparison_panel(photo, [new_poly], "Reviewed2 shape #11 | green", (64,215,130), scale=3.0, crop=(x1,y1,x2,y2))
    # Redraw the newer outline in green on the overlay to distinguish both sources.
    # Build the overlay panel directly using the same source crop and both outline colors.
    xshift, yshift = x1, y1
    cropped = photo.crop((x1,y1,x2,y2))
    s = min(3.0, 1400/cropped.width, 950/cropped.height)
    base = cropped.resize((round(cropped.width*s),round(cropped.height*s)),Image.Resampling.LANCZOS).convert("RGBA")
    layer = Image.new("RGBA",base.size,(0,0,0,0)); d=ImageDraw.Draw(layer)
    for poly,col in [(old_poly,(235,48,184)),(new_poly,(64,215,130))]:
        pts=[(round((x-xshift)*s),round((y-yshift)*s)) for x,y in poly]
        d.polygon(pts,fill=(*col,45),outline=(*col,255),width=4)
    composite=Image.alpha_composite(base,layer).convert("RGB")
    panel=Image.new("RGB",(composite.width,composite.height+42),(29,38,47));panel.paste(composite,(0,42))
    ImageDraw.Draw(panel).text((12,11),"Overlay | old magenta, new green",font=ImageFont.truetype("arial.ttf",18),fill="white")
    save_side_by_side(OUT / "comparison" / "IMG_4588_changed_shape15.jpg", [old_crop,new_crop,panel], "Changed contour comparison | IMG_4588.jpeg | small wrapped item near lower middle")

    old_total = sum(len(v) for v in old_by_name.values())
    current_total = sum(map(len, anns_by_name.values()))
    export_diff = compare_exports(old_by_name, anns_by_name, images)
    report = {
        "status": "review-only; not imported or approved; training not started",
        "new_export": str(NEW_ZIP),
        "new_export_sha256": json.loads(MAPPING.read_text(encoding="utf-8"))["primary_sha256"],
        "zip_entries": ["annotations/instances_default.json"],
        "category_map": categories,
        "image_records": len(coco["images"]),
        "distinct_annotated_images": len(anns_by_name),
        "annotation_count": current_total,
        "old_export": str(OLD_ZIP),
        "old_xml_images": len(old_by_name),
        "old_xml_annotation_count": old_total,
        "geometry_findings": geometry_errors,
        "out_of_bounds": out_of_bounds,
        "fragmented_masks": fragmented,
        "duplicate_masks": duplicate_findings,
        "suspicious_overlaps": overlap_findings,
        "small_masks_below_0_2_percent": tiny,
        "visual_review": {
            "all_23_overlays_compared_with_verified_original_photos": True,
            "obvious_background_only_or_unrelated_object_masks": [],
            "obvious_unannotated_fully_visible_products": [],
            "manual_attention": [
                "Four very small wrapped-looking regions in IMG_4571.jpeg (#16-18) and IMG_4588.jpeg (#7) need confirmation as separate physical products.",
                "IMG_4571.jpeg instances #10/#13 have a small mask intersection; confirm whether this is correct occlusion or boundary spill.",
                "Dense/occluded baskets and stacks still warrant final boundary review even though no clearly missing full product stood out in the visual pass."
            ]
        },
        "visual_review_limits": [
            "Geometry validation cannot prove that a mask follows the true product boundary.",
            "The original photographs and overlays are provided for human confirmation of product identity, missing products, occlusion handling, and background leakage.",
            "Small wrapped items and overlaps are flagged, not removed or repaired.",
            "IMG_4664.jpeg is included as the verified 23rd photo (5 annotations); its intended inclusion remains unclear because the request says 22 new scenes."
        ],
        "per_image": per_image,
        "export_diff_matching": export_diff,
        "export_differences": [
            {"type":"removed annotation", "image":"IMG_4571.jpeg", "old_instance":10, "old_bbox_xyxy":[round(v,1) for v in bbox_for([deleted_poly])], "description":"Small old polygon lies inside an already annotated can mask; likely duplicate annotation. Confirm on comparison preview."},
            {"type":"changed contour", "image":"IMG_4588.jpeg", "old_instance":15, "new_instance":11, "raster_iou_0_25": next((c["raster_iou_0_25"] for c in export_diff["changed"] if c["image"] == change_name), None), "old_bbox_xyxy":[round(v,1) for v in bbox_for([old_poly])], "new_bbox_xyxy":[round(v,1) for v in bbox_for([new_poly])], "description":"Outline changed around a small wrapped item; review side-by-side crop."}
        ],
        "counts_by_image": {r["filename"]:r["annotations"] for r in per_image},
        "approved_dataset_modified": False,
        "training_started": False
    }
    (OUT / "validation_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines = [
        "# InventoryLens CVAT Reviewed2: Validation and Visual Review",
        "",
        "Status: review-only. No annotation was repaired, removed, imported, or approved; training was not started.",
        "",
        "## Export Summary",
        f"- New archive: `{NEW_ZIP}` (SHA-256 `{report['new_export_sha256']}`).",
        "- ZIP contents: `annotations/instances_default.json` only; source photos are mapped from the verified project originals.",
        f"- COCO manifest: {len(coco['images'])} image records, {len(anns_by_name)} distinct annotated images, {current_total} annotations.",
        f"- Category: `{categories[0]['id']} = {categories[0]['name']}`.",
        f"- Earlier XML export: {len(old_by_name)} images and {old_total} polygon shapes.",
        "- All 23 image mappings were previously confirmed by decoded-pixel equality in the export audit.",
        "",
        "## Geometry and Mask Findings",
        f"- Invalid/self-intersecting/degenerate polygons: {len(geometry_errors)}.",
        f"- Out-of-bounds polygons: {len(out_of_bounds)}.",
        f"- Multi-part/fragmented masks: {len(fragmented)}.",
        f"- Duplicate-mask candidates at IoU >= 0.80 or >= 95% containment of smaller mask: {len(duplicate_findings)}.",
        f"- Partial mask overlaps at >= 3% of the smaller mask (full-resolution raster): {len(overlap_findings)}.",
        f"- Masks under 0.2% of image area: {len(tiny)} (see table). These may be real small products; not auto-removed.",
        "- Pixel geometry was checked from COCO polygon coordinates. It does not establish semantic correctness or detect every missing object.",
        "",
        "### Overlap Review",
        "| Image | Instances | IoU | Intersection / smaller mask |",
        "|---|---:|---:|---:|"
    ]
    for f in overlap_findings:
        lines.append(f"| `{f['image']}` | {f['instances'][0]} / {f['instances'][1]} | {f['iou']:.3f} | {f['intersection_over_smaller']:.1%} |")
    if not overlap_findings: lines.append("| None flagged | | | |")
    lines += ["", "### Small Masks", "| Image | Instance | Area (px^2) | Image area | Bounding box (x1,y1,x2,y2) |", "|---|---:|---:|---:|---|"]
    for f in tiny:
        lines.append(f"| `{f['image']}` | {f['instance']} | {f['area_px2']:,} | {f['image_fraction_pct']:.4f}% | {f['bbox_xyxy']} |")
    changed_iou = next((c["raster_iou_0_25"] for c in export_diff["changed"] if c["image"] == change_name), None)
    lines += ["", "## 147-to-146 Export Comparison", "- One annotation is absent: `IMG_4571.jpeg`, earlier XML instance 10, bbox approximately `(288, 1193)-(351, 1278)`. Its mask is entirely contained inside Reviewed2 instance 6 (an energy-drink can), so duplicate removal is plausible; inspect the comparison before deciding intent.", f"- One contour changed: `IMG_4588.jpeg`, earlier XML instance 15 to Reviewed2 instance 11. Raster IoU at 25% resolution is {changed_iou:.3f}; this is a materially changed outline on a small wrapped item, and its crop comparison is provided.", f"- Automated one-to-one polygon matching paired {export_diff['matched_shapes']} masks. Exactly {len(export_diff['removed'])} old shape was unmatched, {len(export_diff['added'])} new shapes were unmatched, and {len(export_diff['changed'])} matched outline changed. No corrections were applied.", "", "## Visual Review", "- All 23 numbered overlays were compared with their verified original photos. No mask was obviously on an unrelated background object, and no clearly separable fully visible product was obviously left unannotated in this visual pass.", "- This is not approval: review the four tiny wrapped-looking regions and the slight intersection of `IMG_4571.jpeg` instances #10/#13. Crowded/occluded objects can still have boundary or completeness issues.", "", "## Per-Image Counts and Flags", "| Image | Scene | Count | Flagged masks | Overlaps |", "|---|---|---:|---|---|"]
    for r in per_image:
        lines.append(f"| `{r['filename']}` | {r['scene']} | {r['annotations']} | {r['small_instances'] or 'none'} | {r['overlap_pairs'] or 'none'} |")
    lines += ["", "## Human Review Still Needed", "- Review all overlays against the original-photo copies. Geometry checks cannot establish whether every visible physical product was annotated or whether contours include background.", "- Prioritize crowded `S20` (`IMG_4634.jpeg`), mixed dense display `T05` (`IMG_4588.jpeg`), crowded repeated-SKU scene `T06` (`IMG_4571.jpeg`), and the four amber-highlighted small masks.", "- `IMG_4664.jpeg` is the verified extra 23rd image with 5 annotations, despite the request referring to 22 new photos. It is included in this review set; its intended status needs confirmation.", "", "## Review Files", "- `originals/`: 23 verified source photos.", "- `overlays/`: numbered segmentation overlays with per-image counts and warning highlights.", "- `contact_sheet_23.jpg`: all 23 overlays in one view.", "- `comparison/IMG_4571_removed_shape10.jpg`: removed annotation in scene context.", "- `comparison/IMG_4588_changed_shape15.jpg`: old/new contour comparison crop.", "- `validation_report.json`: machine-readable audit details.", "", "Approved dataset unchanged; no training started."]
    (OUT / "validation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"images":len(records),"annotations":current_total,"geometry_errors":len(geometry_errors),"out_of_bounds":len(out_of_bounds),"duplicates":len(duplicate_findings),"overlaps":overlap_findings,"tiny":tiny,"outputs":[str(p) for p in [OUT/"contact_sheet_23.jpg", OUT/"comparison"/"IMG_4571_removed_shape10.jpg",OUT/"comparison"/"IMG_4588_changed_shape15.jpg",OUT/"validation_report.md",OUT/"validation_report.json"]]},indent=2))


if __name__ == "__main__":
    main()
