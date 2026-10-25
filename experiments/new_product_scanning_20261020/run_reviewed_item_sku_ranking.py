"""Rank human-boxed test items against the provisional SigLIP2 reference gallery."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from experiment_io import ROOT, EXPERIMENT, load_registry, sha256_file, siglip_snapshot

sys.path.insert(0, str(ROOT))

GT_PATH = ROOT / "experiments" / "first_two_train_third_test_20261020" / "annotations" / "dino_item_boxes_review_20261023.json"
PREDICTIONS = ROOT / "experiments" / "first_two_train_third_test_20261020" / "artifacts" / "third_set_review_20261020" / "predictions.json"
IMAGE_ROOT = ROOT / "experiments" / "first_two_train_third_test_20261020" / "artifacts" / "third_set_review_20261020"
REF_ROOT = EXPERIMENT / "artifacts" / "reference_crops"
OUT_ROOT = EXPERIMENT / "results"


def topk(scores: dict[str, float], names: dict[str, str], k: int = 3) -> list[dict]:
    return [{"sku_id": sku, "name": names[sku], "similarity": float(score)}
            for sku, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def rank(query: np.ndarray, refs: np.ndarray, by_sku: dict[str, list[int]]) -> dict[str, float]:
    sims = refs @ query
    return {sku: float(np.mean(sorted((float(sims[i]) for i in idx), reverse=True)[:2]))
            for sku, idx in by_sku.items()}


def main() -> None:
    review = json.loads(GT_PATH.read_text(encoding="utf-8"))
    pred_data = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    pred_scenes = {Path(row["original_filename"]).name: row for row in pred_data["scenes"]}
    if len(review.get("scenes", [])) != 30 or not all(row.get("review_complete") for row in review["scenes"]):
        raise ValueError("Expected a complete item-box review for all 30 photos")

    registry = load_registry()
    names = {row["sku_id"]: row.get("proposed_name") or row["sku_id"] for row in registry}
    identity_confirmed = {row["sku_id"]: bool(row.get("identity_confirmed")) for row in registry}
    refs: dict[str, list[Path]] = {sku: [] for sku in names}
    for path in sorted(REF_ROOT.glob("IL-NEW-*/*_box.jpg")):
        if path.parent.name in refs:
            refs[path.parent.name].append(path)
    for path in sorted(REF_ROOT.glob("IL-NEW-*/*_masked.jpg")):
        if path.parent.name in refs:
            refs[path.parent.name].append(path)
    ref_paths = {"original": [], "masked": []}
    ref_skus: list[str] = []
    ref_digests: list[str] = []
    for path in sorted(REF_ROOT.glob("IL-NEW-*/*_box.jpg")):
        if path.parent.name not in names:
            continue
        ref_paths["original"].append(path)
        ref_skus.append(path.parent.name)
        ref_digests.append(sha256_file(path))
    for path in ref_paths["original"]:
        masked = path.with_name(path.name[:-8] + "_masked.jpg")
        if not masked.is_file():
            raise FileNotFoundError(f"Paired masked reference is missing: {masked}")
        ref_paths["masked"].append(masked)
    if len(ref_skus) != 34:
        raise ValueError(f"Expected 34 approved reference crops, found {len(ref_skus)}")

    ref_images: dict[str, list[Image.Image]] = {key: [] for key in ref_paths}
    for representation, paths in ref_paths.items():
        for path in paths:
            with Image.open(path) as image:
                ref_images[representation].append(ImageOps.exif_transpose(image).convert("RGB"))

    from experiments.hybrid_pipeline.hf_encoders import HuggingFaceImageEncoder, SIGLIP2_REVISION
    snapshot = siglip_snapshot()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pinned SigLIP2 model is not fully cached: {snapshot}")
    encoder = HuggingFaceImageEncoder(str(snapshot), "siglip2", device="cpu", local_files_only=True,
                                      revision=SIGLIP2_REVISION)
    ref_features = {key: encoder.encode(images, batch_size=1) for key, images in ref_images.items()}
    by_sku: dict[str, list[int]] = {}
    for i, sku in enumerate(ref_skus):
        by_sku.setdefault(sku, []).append(i)
    if len(by_sku) != 10 or min(map(len, by_sku.values())) < 2:
        raise ValueError("Expected at least two references for each of ten candidate SKUs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_ROOT / f"reviewed_item_sku_ranking_{run_id}"
    (out / "overlays").mkdir(parents=True, exist_ok=False)
    result_rows, scene_rows = [], []
    total_gt_items = 0
    for scene in review["scenes"]:
        name = scene["filename"]
        record = pred_scenes.get(name)
        if not record or record["sha256"] != scene["source_sha256"]:
            raise ValueError(f"Test-photo source hash mismatch: {name}")
        image_path = IMAGE_ROOT / record["local_path"]
        if sha256_file(image_path) != scene["source_sha256"]:
            raise ValueError(f"Packaged test photo hash mismatch: {name}")
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (scene["width"], scene["height"]):
            raise ValueError(f"Image dimensions do not match annotation export: {name}")

        crops = []
        for i, box in enumerate(scene["boxes_xyxy"], 1):
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                raise ValueError(f"Out-of-bounds box {i} in {name}")
            crop = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
            crops.append(crop)
        vectors = encoder.encode(crops, batch_size=1) if crops else np.empty((0, 768), dtype=np.float32)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("arial.ttf", max(18, round(min(image.size) / 55)))
        except OSError:
            font = ImageFont.load_default()
        row_scene = {"filename": name, "group": scene["group"], "items": []}
        for i, (box, vector) in enumerate(zip(scene["boxes_xyxy"], vectors), 1):
            scores_by_rep = {rep: rank(vector, ref_features[rep], by_sku) for rep in ("original", "masked")}
            scores_by_rep["combined"] = {sku: (scores_by_rep["original"][sku] + scores_by_rep["masked"][sku]) / 2 for sku in names}
            rankings = {rep: topk(scores, names) for rep, scores in scores_by_rep.items()}
            item = {"item_id": i, "box_xyxy": [round(float(x), 1) for x in box], "rankings": rankings}
            row_scene["items"].append(item)
            flat = {"photo_filename": name, "group": scene["group"], "item_id": i,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
            for rep, ranked in rankings.items():
                for place, candidate in enumerate(ranked, 1):
                    prefix = f"{rep}_top{place}_"
                    flat[prefix + "sku_id"] = candidate["sku_id"]
                    flat[prefix + "name"] = candidate["name"]
                    flat[prefix + "similarity"] = f"{candidate['similarity']:.6f}"
            result_rows.append(flat)
            total_gt_items += 1
            x1, y1, x2, y2 = map(lambda v: int(round(v)), box)
            draw.rectangle((x1, y1, x2, y2), outline="#13854f", width=max(3, min(image.size) // 500))
            caption = f"#{i} {rankings['combined'][0]['sku_id']} {rankings['combined'][0]['similarity']:.2f}"
            tb = draw.textbbox((x1, y1), caption, font=font)
            ty = max(0, y1 - (tb[3] - tb[1]) - 6)
            draw.rectangle((x1, ty, tb[2] + 6, ty + (tb[3] - tb[1]) + 6), fill="#13854f")
            draw.text((x1 + 3, ty + 2), caption, fill="white", font=font)
        overlay_name = f"{Path(name).stem}_sku_suggestions.jpg"
        overlay.save(out / "overlays" / overlay_name, quality=92)
        row_scene["overlay"] = f"overlays/{overlay_name}"
        scene_rows.append(row_scene)
        print(f"{name}: ranked {len(crops)} reviewed item crops", flush=True)

    csv_path = out / "item_sku_rankings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader(); writer.writerows(result_rows)
    report = {
        "status": "PROVISIONAL_SKU_RANKS_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_item_box_export_sha256": sha256_file(GT_PATH),
        "model": "google/siglip2-base-patch16-224",
        "model_revision": SIGLIP2_REVISION,
        "device": "cpu",
        "input": {"photos": len(scene_rows), "reviewed_item_crops": total_gt_items,
                  "reference_photo_crops_per_representation": len(ref_skus), "candidate_skus": len(names)},
        "gallery": {sku: {"name": names[sku], "reference_count": len(by_sku[sku]),
                          "identity_confirmed": identity_confirmed[sku]} for sku in names},
        "scoring": {"original": "approved unmasked box crop gallery",
                    "masked": "approved background-masked product crop gallery",
                    "combined": "arithmetic mean of original and masked per-SKU scores",
                    "per_sku": "mean of top two cosine similarities among that SKU's 34 paired reference crops",
                    "similarity_is_calibrated_confidence": False,
                    "unknown_rejection_threshold": None},
        "accuracy_metrics": None,
        "limitations": ["The ten SKU/product identities are not confirmed in the experimental registry.",
                        "The 291 test items have no reviewed SKU ground-truth labels, so these ranks cannot be scored for SKU accuracy.",
                        "This ranks human-drawn item crops; it does not test automatic DINO detection plus SKU recognition end to end.",
                        "Similarity scores are ranking signals, not calibrated confidence; no unknown rejection is applied."],
        "photos": scene_rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows_json = json.dumps(scene_rows, ensure_ascii=False).replace("</", "<\\/")
    page = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reviewed item SKU suggestions</title><style>
body{font:15px/1.5 system-ui,sans-serif;background:#f3f6f8;color:#17242f;margin:0}main{max-width:1250px;margin:auto;padding:24px}h1{margin:0}.muted{color:#596b78}.warn{padding:13px;background:#fff4dc;border:1px solid #e8d39f;border-radius:9px;margin:14px 0}.toolbar{display:flex;gap:8px;align-items:center;margin:12px 0}select,button{font:inherit;padding:8px;border:1px solid #bdc9d2;border-radius:7px;background:#fff}.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(380px,1fr);gap:12px}.panel{background:#fff;border:1px solid #dce3e8;border-radius:10px;padding:12px;min-width:0}.panel img{width:100%;height:auto;display:block;border-radius:6px}.tablewrap{overflow:auto;max-height:78vh}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #e3e9ed;padding:7px}th{position:sticky;top:0;background:#f6f8f9}.rank{white-space:nowrap}.itemhead{font-weight:700} @media(max-width:900px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style></head><body><main><h1>SKU suggestions for reviewed products</h1><p class="muted">Reference-photo rankings attached to your individually drawn product boxes.</p><div class="warn"><b>Provisional results:</b> these are similarity rankings, not verified SKU IDs or confidence scores. The ten candidate product identities remain unconfirmed, and this report has no SKU accuracy score. Check the candidate list against the packaging before treating any SKU as accepted.</div><div class="toolbar"><label for="scene">Photo</label><select id="scene"></select><span class="muted" id="count"></span></div><div class="grid"><section class="panel"><img id="overlay" alt="Reviewed item boxes with provisional top SKU labels"></section><section class="panel tablewrap"><table><thead><tr><th>Item</th><th>Combined top 3 suggestions (similarity)</th><th>Original top choice</th><th>Masked top choice</th></tr></thead><tbody id="rows"></tbody></table></section></div></main><script>
const scenes=__SCENES__;const sel=document.getElementById('scene');scenes.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${s.group} · ${s.filename} · ${s.items.length} items`;sel.append(o)});function esc(x){let d=document.createElement('div');d.textContent=x;return d.innerHTML}function show(){let s=scenes[+sel.value];document.getElementById('overlay').src=s.overlay;document.getElementById('overlay').alt=`${s.filename}: manually reviewed item boxes with top candidate`;document.getElementById('count').textContent=`${s.items.length} reviewed item boxes`;let b=document.getElementById('rows');b.replaceChildren();s.items.forEach(it=>{let tr=document.createElement('tr');let r=it.rankings;tr.innerHTML=`<td class="itemhead">Item ${it.item_id}<br><span class="muted">${it.box_xyxy.map(Math.round).join(', ')}</span></td><td>${r.combined.map((x,i)=>`<div class="rank">${i+1}. ${esc(x.sku_id)} · ${esc(x.name)} · ${x.similarity.toFixed(3)}</div>`).join('')}</td><td>${esc(r.original[0].sku_id)}<br>${r.original[0].similarity.toFixed(3)}</td><td>${esc(r.masked[0].sku_id)}<br>${r.masked[0].similarity.toFixed(3)}</td>`;b.append(tr)})}sel.onchange=show;show();
</script></body></html>'''.replace("__SCENES__", rows_json)
    (out / "sku_review.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "item_crops": total_gt_items,
                      "identity_confirmed_skus": sum(identity_confirmed.values()),
                      "accuracy_metrics": None}, indent=2))


if __name__ == "__main__":
    main()
