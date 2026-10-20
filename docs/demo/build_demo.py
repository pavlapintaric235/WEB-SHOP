#!/usr/bin/env python3
"""Build the visual walkthrough assets + data for docs/demo/index.html.

Reads only committed/verified experiment outputs. Resizes imagery for the web.
Nothing is invented: every number comes from predictions.csv, report.json,
candidate_skus.csv or the OpenCV 5 capture baseline CSV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("E:/PAVLA/WEB-SHOP")
RES = ROOT / "experiments/new_product_scanning_20261020/results/imported_runpod_scene_results_20261020_v5_flat"
EXP = ROOT / "experiments/new_product_scanning_20261020"
DATASET = ROOT / "03_newest_10_products_owner_employee_customer"
PHOTOS = DATASET / "photos"
SCENE_MASKS = DATASET / "scene_masks"
CAPCSV = ROOT / "experiments/opencv5_guided_scan/aws_pilot/local_expected_decisions_20261020.csv"
OUT = ROOT / "docs/demo"
ASSETS = OUT / "assets"

SCENE_W = 1100
CROP_W = 260
OBJECT_W = 150
Q = 76


def rows(p: Path):
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def shrink(src: Path, dst: Path, width: int) -> bool:
    from PIL import Image, ImageOps
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=Q, optimize=True, progressive=True)
    return True


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)

    report = json.loads((RES / "report.json").read_text(encoding="utf-8"))
    preds = rows(RES / "predictions.csv")
    skus = rows(EXP / "audit/candidate_skus.csv")
    cap = rows(CAPCSV)

    proceed = [r for r in cap if r["action"] == "proceed"]
    retake = [r for r in cap if r["action"] == "retake"]
    reasons: dict[str, int] = {}
    for r in retake:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    ex_ok = proceed[0] if proceed else None
    ex_no = retake[0] if retake else None
    keys = ("action", "reason", "width", "height", "lap_var_1024", "dark_pct", "bright_pct")
    capture = {
        "total": len(cap),
        "proceed": len(proceed),
        "retake": len(retake),
        "reasons": reasons,
        "example_proceed": {k: ex_ok[k] for k in keys} if ex_ok else None,
        "example_retake": {k: ex_no[k] for k in keys} if ex_no else None,
    }

    crops = EXP / "registration/artifacts/reference_crops"
    sku_out = []
    for s in skus:
        sid = s["sku_id"]
        pair = None
        d = crops / sid
        if d.is_dir():
            box = sorted(d.glob("*_box.jpg"))
            msk = sorted(d.glob("*_masked.jpg"))
            if box and shrink(box[0], ASSETS / f"sku_{sid}_box.jpg", CROP_W):
                pair = {"box": f"assets/sku_{sid}_box.jpg"}
                if msk and shrink(msk[0], ASSETS / f"sku_{sid}_masked.jpg", CROP_W):
                    pair["masked"] = f"assets/sku_{sid}_masked.jpg"
        sku_out.append({
            "id": sid,
            "name": s["proposed_name"],
            "brand": s["brand"],
            "variant": s["variant_or_size"],
            "refs": int(s["reference_count"]),
            "identity_status": s["identity_status"],
            "owner_confirmation": s["owner_confirmation"],
            "img": pair,
        })

    by_photo: dict[str, list[dict]] = {}
    for p in preds:
        by_photo.setdefault(p["photo_filename"], []).append(p)

    def g3(p, pre):
        out = []
        for n in ("top1", "top2", "top3"):
            sid = p.get(f"{pre}_{n}_sku_id") or ""
            if not sid:
                continue
            out.append({
                "sku": sid,
                "name": p.get(f"{pre}_{n}_name") or "",
                "sim": round(float(p.get(f"{pre}_{n}_similarity") or 0), 3),
            })
        return out

    det_map = {f.name.split("_", 1)[1].rsplit("_dino", 1)[0]: f for f in (RES / "overlays_detector").glob("*.jpg")}
    rec_map = {f.name.split("_", 1)[1].rsplit("_dino_siglip2", 1)[0]: f for f in (RES / "overlays_recognition").glob("*.jpg")}

    # Scene masks: MobileSAM prompted with the committed DINO boxes. These are
    # unreviewed proposals and were NOT part of the ranking run, so they are kept
    # separate from the gallery conditions, which vary references, not queries.
    mask_index = rows(SCENE_MASKS / "scene_mask_index.csv") if (SCENE_MASKS / "scene_mask_index.csv").is_file() else []
    mask_run = json.loads((SCENE_MASKS / "scene_mask_run.json").read_text(encoding="utf-8")) \
        if (SCENE_MASKS / "scene_mask_run.json").is_file() else {}
    mask_rows = {(Path(m["source_photo"]).name, m["detection_id"]): m for m in mask_index}

    scenes = []
    for photo in sorted(by_photo):
        stem = photo.rsplit(".", 1)[0]
        rec = by_photo[photo]
        dtype = rec[0]["dataset_type"]
        src_orig = None
        for sub in ("new10_employee", "new10_customer", "new10_owner"):
            cand = PHOTOS / sub / photo
            if cand.is_file():
                src_orig = cand
                break
        imgs = {}
        if src_orig and shrink(src_orig, ASSETS / f"orig_{stem}.jpg", SCENE_W):
            imgs["original"] = f"assets/orig_{stem}.jpg"
        if stem in det_map and shrink(det_map[stem], ASSETS / f"det_{stem}.jpg", SCENE_W):
            imgs["detector"] = f"assets/det_{stem}.jpg"
        if stem in rec_map and shrink(rec_map[stem], ASSETS / f"rec_{stem}.jpg", SCENE_W):
            imgs["recognition"] = f"assets/rec_{stem}.jpg"
        overlay = SCENE_MASKS / "artifacts/overlays" / f"{stem}_masks.jpg"
        if shrink(overlay, ASSETS / f"msk_{stem}.jpg", SCENE_W):
            imgs["masks"] = f"assets/msk_{stem}.jpg"

        boxes = []
        for p in sorted(rec, key=lambda r: r["detection_id"]):
            did = p["detection_id"]
            box = {
                "id": did,
                "conf": round(float(p["detector_confidence"]), 3),
                "status": p["recognition_status"],
                "crop_sha": (p.get("query_crop_sha256") or "")[:12],
                "original": g3(p, "original"),
                "masked": g3(p, "masked"),
                "combined": g3(p, "combined"),
            }
            m = mask_rows.get((photo, did))
            if m:
                crop = {}
                if shrink(DATASET / m["box_crop_file"], ASSETS / f"obj_{stem}_{did}_box.jpg", OBJECT_W):
                    crop["box"] = f"assets/obj_{stem}_{did}_box.jpg"
                if shrink(DATASET / m["masked_crop_file"], ASSETS / f"obj_{stem}_{did}_masked.jpg", OBJECT_W):
                    crop["masked"] = f"assets/obj_{stem}_{did}_masked.jpg"
                if crop:
                    box["crop"] = crop
                    box["mask_area"] = float(m["mask_area_fraction"])
                    box["mask_sha"] = m["mask_file_sha256"][:12]
            boxes.append(box)
        scenes.append({"photo": photo, "type": dtype, "img": imgs, "boxes": boxes})

    payload = {
        "meta": {
            "scene_images": report.get("scene_images"),
            "detected_boxes": report.get("detected_boxes"),
            "accuracy_metrics": report.get("accuracy_metrics"),
            "status": report.get("status"),
            "scope": report.get("scope"),
            "gpu": report.get("gpu"),
            "limitations": report.get("limitations", []),
            "gallery_conditions": ["original", "masked", "combined"],
        },
        "scene_masks": {
            "count": len(mask_index),
            "photographs": mask_run.get("photographs"),
            "backend": mask_run.get("backend"),
            "review_status": mask_run.get("review_status"),
            "generated_utc": mask_run.get("generated_utc"),
            "used_in_ranking_run": False,
        },
        "capture": capture,
        "skus": sku_out,
        "scenes": scenes,
    }
    (OUT / "data.js").write_text(
        "// Generated by build_demo.py from verified experiment outputs. Do not edit by hand.\n"
        "window.DEMO = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8", newline="\n")

    total = sum(f.stat().st_size for f in ASSETS.glob("*.jpg"))
    print(f"scenes: {len(scenes)}  boxes: {sum(len(s['boxes']) for s in scenes)}  skus: {len(sku_out)}")
    print(f"assets: {len(list(ASSETS.glob('*.jpg')))} jpgs, {total/1048576:.1f} MB")
    print(f"data.js: {(OUT / 'data.js').stat().st_size/1024:.0f} KB")
    print(f"capture: {capture['proceed']} proceed / {capture['retake']} retake of {capture['total']}")
    print(f"scene masks: {len(mask_index)} ({mask_run.get('review_status', 'absent')}), "
          f"used in ranking run: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
