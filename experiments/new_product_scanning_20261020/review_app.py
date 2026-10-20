"""Local-only registration and annotation UI for the new-product experiment."""

from __future__ import annotations

import os
import csv
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from experiment_io import (
    ANNOTATIONS, ARTIFACTS, AUDIT, EXPERIMENT, RESULTS, RUNTIME,
    append_mask_event, atomic_json, latest_mask_events, load_manifest,
    load_registry, load_reviewed_mask, load_scene_labels, load_verified_image,
    read_rows, save_registry, save_scene_labels, siglip_snapshot, sku_by_reference,
)

st.set_page_config(page_title="InventoryLens | New SKU experiment", layout="wide")
MANIFEST_ROWS = load_manifest()
ROWS_BY_NAME = {row["filename"]: row for row in MANIFEST_ROWS}
OWNER_ROWS = [row for row in MANIFEST_ROWS if row["group"] == "owner"]
SCENE_ROWS = [row for row in MANIFEST_ROWS if row["group"] in {"employee", "customer"}]


def _thumbnail(image: Image.Image, max_side: int = 1100) -> Image.Image:
    result = image.copy()
    result.thumbnail((max_side, max_side))
    return result


def _overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    preview = _thumbnail(image)
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(preview.size, Image.Resampling.NEAREST)
    selected = np.asarray(resized) > 0
    pixels = np.asarray(preview).copy()
    pixels[selected] = (pixels[selected].astype(np.float32) * 0.42 + np.array([20, 198, 115]) * 0.58).astype(np.uint8)
    return Image.fromarray(pixels)


def _scene_overlay(image: Image.Image, instances: list[dict]) -> Image.Image:
    output = _thumbnail(image)
    sx, sy = output.width / image.width, output.height / image.height
    draw = ImageDraw.Draw(output)
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for index, item in enumerate(instances, 1):
        x1, y1, x2, y2 = item["box_xyxy"]
        box = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
        duplicate = tuple(item["box_xyxy"]) in seen_boxes
        seen_boxes.add(tuple(item["box_xyxy"]))
        draw.rectangle(box, outline=(230, 50, 70) if duplicate else (0, 210, 115),
                       width=max(3, output.width // 350))
        draw.text((box[0] + 3, box[1] + 3 + (index - 1) * 20),
                  f"{index}: {item['sku_id']}{' DUPLICATE BOX' if duplicate else ''}", fill="black",
                  stroke_width=2, stroke_fill="white")
    return output


def _prediction_overlay(image: Image.Image, predictions: list[dict]) -> Image.Image:
    output = _thumbnail(image)
    sx, sy = output.width / image.width, output.height / image.height
    draw = ImageDraw.Draw(output)
    for row in predictions:
        x1, y1, x2, y2 = (float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        box = (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
        draw.rectangle(box, outline=(255, 150, 25), width=max(3, output.width // 450))
        label = f"{row['detection_id']} {row['top1_sku_id']} sim {float(row['top1_similarity']):.2f}"
        draw.text((box[0] + 3, box[1] + 3), label, fill="black", stroke_width=2, stroke_fill="white")
    return output


def _prediction_runs() -> list[Path]:
    found = []
    for path in RESULTS.rglob("predictions.csv"):
        report_path = path.parent / "report.json"
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("experiment") == "new_product_scanning_20261020":
            found.append(path)
    return sorted(found, key=lambda path: str(path.parent), reverse=True)


@st.cache_resource(show_spinner=False)
def _sam_model(weights: str):
    from ultralytics import SAM
    return SAM(weights)


@st.cache_resource(show_spinner=False)
def _siglip_encoder():
    os.environ.setdefault("WEBSHOP_HF_CACHE", str(Path(__file__).resolve().parents[2] / ".venv" / "huggingface_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from experiments.hybrid_pipeline.hf_encoders import HuggingFaceImageEncoder, SIGLIP2_REVISION
    snapshot = siglip_snapshot()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pinned local SigLIP2 snapshot is incomplete: {snapshot}")
    return HuggingFaceImageEncoder(str(snapshot), "siglip2", device="cpu",
                                   local_files_only=True, revision=SIGLIP2_REVISION)


def _auto_mask_proposals(row: dict[str, str], image: Image.Image):
    """Create MobileSAM whole-image proposals; no proposal is accepted automatically."""
    from ultralytics.models.sam.predict import Predictor

    class ReviewPredictor(Predictor):
        def inference(self, im, bboxes=None, points=None, labels=None, masks=None,
                      multimask_output=False, *args, **kwargs):
            if all(value is None for value in (bboxes, points, masks)):
                return self.generate(
                    im, points_stride=8, points_batch_size=16,
                    conf_thres=0.86, stability_score_thresh=0.90,
                )
            return super().inference(im, bboxes, points, labels, masks,
                                     multimask_output, *args, **kwargs)

    digest = row["sha256"]
    cache = RUNTIME / "mask_proposals" / f"{digest}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as data:
            return data["masks"], data["boxes"], data["scores"]

    root = Path(__file__).resolve().parents[2]
    weights = root / "data" / "models" / "mobile_sam.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"Existing MobileSAM checkpoint not found: {weights}")
    working = image.copy()
    working.thumbnail((1024, 1024))
    with st.spinner("Generating provisional MobileSAM masks on CPU. This can take a few minutes."):
        results = _sam_model(str(weights)).predict(
            source=np.asarray(working), points=None, labels=None, imgsz=1024,
            predictor=ReviewPredictor, conf=0.20, iou=0.70,
            device="cpu", verbose=False, save=False,
        )
    if not results or results[0].masks is None:
        masks = np.empty((0, working.height, working.width), dtype=np.uint8)
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
    else:
        raw_masks = results[0].masks.data.detach().cpu().numpy() > 0.5
        raw_boxes = results[0].boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        raw_scores = results[0].boxes.conf.detach().cpu().numpy().astype(np.float32)
        areas = raw_masks.mean(axis=(1, 2))
        keep = np.flatnonzero((areas >= 0.005) & (areas <= 0.985))
        keep = sorted(keep, key=lambda index: float(raw_scores[index]), reverse=True)[:24]
        masks, boxes, scores = raw_masks[keep].astype(np.uint8), raw_boxes[keep], raw_scores[keep]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, masks=masks, boxes=boxes, scores=scores)
    return masks, boxes, scores


def _approve_mask(row: dict[str, str], image: Image.Image, mask_small: np.ndarray,
                  box_small: np.ndarray, score: float, sku_id: str, encode: bool) -> dict:
    import cv2
    from experiments.hybrid_pipeline.segmentation_models import mask_polygons

    stem = Path(row["filename"]).stem.replace(" ", "_")
    digest = row["sha256"]
    artifact_key = f"{stem}_{digest[:10]}"
    scale_x, scale_y = image.width / mask_small.shape[1], image.height / mask_small.shape[0]
    mask = cv2.resize(mask_small.astype(np.uint8), (image.width, image.height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not mask.any():
        raise ValueError("Selected mask is empty after resizing.")
    box = [float(box_small[0] * scale_x), float(box_small[1] * scale_y),
           float(box_small[2] * scale_x), float(box_small[3] * scale_y)]

    mask_path = ARTIFACTS / "masks" / f"{artifact_key}.npz"
    polygon_path = ARTIFACTS / "polygons" / f"{artifact_key}.json"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    polygon_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mask_path, mask=mask.astype(np.uint8))
    atomic_json(polygon_path, {
        "source_file": row["filename"], "source_sha256": digest,
        "width": image.width, "height": image.height,
        "coordinate_space": "EXIF-transposed displayed-image pixels",
        "polygons": mask_polygons(mask),
    })

    x1, y1, x2, y2 = [int(round(value)) for value in box]
    pad = max(2, round(max(x2 - x1, y2 - y1) * 0.08))
    crop_box = (max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad))
    box_crop = image.crop(crop_box)
    pixels = np.asarray(image).copy()
    pixels[~mask] = (124, 124, 124)
    masked_crop = Image.fromarray(pixels).crop(crop_box)
    crop_dir = ARTIFACTS / "reference_crops" / sku_id
    crop_dir.mkdir(parents=True, exist_ok=True)
    box_crop_path = crop_dir / f"{artifact_key}_box.jpg"
    masked_crop_path = crop_dir / f"{artifact_key}_masked.jpg"
    box_crop.save(box_crop_path, quality=95)
    masked_crop.save(masked_crop_path, quality=95)

    embedding_path = ARTIFACTS / "embeddings" / sku_id / f"{artifact_key}_siglip2.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_status, embedding_error = "not_requested", None
    if encode:
        try:
            features = _siglip_encoder().encode([box_crop, masked_crop], batch_size=1)
            np.savez_compressed(embedding_path, box_crop=features[0], masked_crop=features[1])
            embedding_status = "saved"
        except Exception as error:
            embedding_status = "unavailable"
            embedding_error = f"{type(error).__name__}: {error}"

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": row["filename"], "source_path": row["path"], "source_sha256": digest,
        "sku_id": sku_id, "decision": "approved", "mask_score": float(score),
        "bbox_xyxy": box, "mask_file": str(mask_path.relative_to(EXPERIMENT)),
        "polygon_file": str(polygon_path.relative_to(EXPERIMENT)),
        "box_crop_file": str(box_crop_path.relative_to(EXPERIMENT)),
        "masked_crop_file": str(masked_crop_path.relative_to(EXPERIMENT)),
        "embedding_file": str(embedding_path.relative_to(EXPERIMENT)) if embedding_status == "saved" else None,
        "embedding_status": embedding_status, "embedding_error": embedding_error,
    }
    append_mask_event(event)
    return event


def _registration_page() -> None:
    st.header("Product registration")
    st.caption("Experimental review only. No production SKUs or inventory records are changed.")
    records = load_registry()
    priority_ids = ["IL-NEW-004", "IL-NEW-008", "IL-NEW-009", "IL-NEW-010", "IL-NEW-007"]
    priority_notes = {
        "IL-NEW-004": "Tealight: scent, unit and sellable pack unresolved",
        "IL-NEW-008": "Battery: exact subtype, chemistry and pack unresolved",
        "IL-NEW-009": "Cup: material, size and pack unresolved",
        "IL-NEW-010": "Crown: exact product identity unresolved",
        "IL-NEW-007": "Glue stick: exact retail variant needs confirmation",
    }
    st.dataframe([{
        "Candidate ID": row["sku_id"],
        "Assigned ID": row.get("assigned_sku_id", row["sku_id"]),
        "Proposed name": row["proposed_name"],
        "Current variant / size note": row.get("variant_or_size", ""),
        "Identity status": "Confirmed by owner" if row.get("identity_confirmed") else "Unresolved",
        "Priority": "Review first" if row["sku_id"] in priority_ids else "",
    } for row in records], hide_index=True, width="stretch")
    st.info("Review the five priority identities first. Leave unreadable names, variants, sizes or SKU IDs unresolved.")
    order = sorted(range(len(records)), key=lambda index: (
        priority_ids.index(records[index]["sku_id"]) if records[index]["sku_id"] in priority_ids else 100 + index))
    selected = st.selectbox("Candidate product (priority first)", order,
                            format_func=lambda index: f"{records[index]['sku_id']} - {records[index]['proposed_name']}")
    item = dict(records[selected])
    if item["sku_id"] in priority_notes and not item.get("identity_confirmed"):
        st.warning(priority_notes[item["sku_id"]])

    st.subheader("Reference photographs")
    refs = [row for row in OWNER_ROWS if row["filename"] in item["reference_files"]]
    mask_events = latest_mask_events()
    columns = st.columns(min(3, len(refs))) if refs else []
    for index, row in enumerate(refs):
        with columns[index % len(columns)]:
            st.image(_thumbnail(load_verified_image(row), 500), caption=f"Original: {row['filename']}")
            event = mask_events.get(row["filename"])
            if event and event.get("decision") == "approved" and event.get("masked_crop_file"):
                with Image.open(EXPERIMENT / event["masked_crop_file"]) as masked:
                    st.image(_thumbnail(masked.convert("RGB"), 350), caption="Approved masked reference")
            else:
                st.caption("No approved paired mask; excluded from the 34-pair GPU gallery")
    if refs:
        enlarged = st.selectbox("Enlarge an original reference", [row["filename"] for row in refs])
        st.image(_thumbnail(load_verified_image(ROWS_BY_NAME[enlarged]), 1000), caption=enlarged)

    st.subheader("Approve or correct registration")
    with st.form("sku_review_form"):
        item["assigned_sku_id"] = st.text_input("Assigned SKU ID", item.get("assigned_sku_id", item["sku_id"]),
                                                 help="The candidate ID remains the experiment key until a reviewed mapping is applied.")
        item["proposed_name"] = st.text_input("Product name", item["proposed_name"])
        item["brand"] = st.text_input("Brand, if visible", item.get("brand", ""))
        st.caption(f"Existing unverified variant / size note: {item.get('variant_or_size', '') or 'none'}")
        item["confirmed_variant"] = st.text_input("Variant, if known", item.get("confirmed_variant", ""))
        item["confirmed_package_size"] = st.text_input("Package size, if known", item.get("confirmed_package_size", ""))
        item["identity_confirmed"] = st.checkbox("I confirm this is one distinct sellable SKU",
                                                  bool(item.get("identity_confirmed", False)))
        item["angle_status"] = st.text_input("Reference-angle notes", item.get("angle_status", ""))
        item["owner_notes"] = st.text_area("Unresolved identity / packaging notes", item.get("owner_notes", ""))
        submitted = st.form_submit_button("Save experimental registration", type="primary")
    if submitted:
        assigned = item["assigned_sku_id"].strip()
        if not assigned or any(index != selected and assigned == other.get("assigned_sku_id", other["sku_id"])
                               for index, other in enumerate(records)):
            st.error("Assigned SKU ID must be nonempty and unique across this experimental review.")
        elif item["identity_confirmed"] and not item["proposed_name"].strip():
            st.error("A confirmed product needs a product name. Leave it unresolved if identity is uncertain.")
        else:
            item["assigned_sku_id"] = assigned
            records[selected] = item
            save_registry(records)
            st.success(f"Saved {item['sku_id']} in the experiment and appended a review event.")
            st.rerun()
    st.caption("Source photographs and candidate SKU CSV remain unchanged.")


def _mask_page() -> None:
    st.header("Automatic reference-mask review")
    st.caption("MobileSAM runs locally on CPU. Proposals are not ground truth; approve only a mask covering exactly one product.")
    sku_map = sku_by_reference()
    sku_options = [item["sku_id"] for item in load_registry()]
    filter_sku = st.selectbox("Candidate SKU", ["All"] + sku_options)
    available = [row for row in OWNER_ROWS if row["filename"] in sku_map and (filter_sku == "All" or sku_map[row["filename"]] == filter_sku)]
    if not available:
        st.warning("No owner references match this selection.")
        return
    filename = st.selectbox("Reference photograph", [row["filename"] for row in available])
    row, image = ROWS_BY_NAME[filename], load_verified_image(ROWS_BY_NAME[filename])
    st.write(f"Candidate: **{sku_map[filename]}** · hash checked: `{row['sha256'][:16]}…`")
    left, right = st.columns(2)
    with left:
        st.image(_thumbnail(image), caption=f"Original · {image.width}×{image.height}")
    with right:
        if st.button("Generate / load MobileSAM proposals", type="primary"):
            try:
                st.session_state[f"proposals_{filename}"] = _auto_mask_proposals(row, image)
            except Exception as error:
                st.error(f"Mask proposal failed: {type(error).__name__}: {error}")
        proposals = st.session_state.get(f"proposals_{filename}")
        if proposals is not None:
            masks, boxes, scores = proposals
            if len(masks):
                chosen = st.selectbox("Candidate mask", range(len(masks)), format_func=lambda i: f"Proposal {i + 1} · score {scores[i]:.3f}", key=f"proposal_{filename}")
                st.image(_overlay(image, masks[chosen]), caption="Green overlay is the proposed object mask.")
                st.caption(f"Proposed box in displayed-image pixels: {np.round(boxes[chosen], 1).tolist()}")
                encode = st.checkbox("Also create SigLIP2 embeddings from original-box and masked crops", True, key=f"encode_{filename}")
                col_a, col_b = st.columns(2)
                if col_a.button("Approve mask and save crops", key=f"approve_{filename}"):
                    try:
                        event = _approve_mask(row, image, masks[chosen], boxes[chosen], float(scores[chosen]), sku_map[filename], encode)
                        st.success(f"Saved approved mask, polygon and both crops. SigLIP2 embeddings: {event['embedding_status']}.")
                        if event.get("embedding_error"):
                            st.warning(event["embedding_error"])
                    except Exception as error:
                        st.error(f"Could not save approved mask: {type(error).__name__}: {error}")
                if col_b.button("Reject this proposal", key=f"reject_{filename}"):
                    append_mask_event({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "source_file": filename, "source_path": row["path"], "source_sha256": row["sha256"],
                        "sku_id": sku_map[filename], "decision": "rejected", "proposal_index": int(chosen),
                        "mask_score": float(scores[chosen]),
                    })
                    st.info("Rejection recorded. Source photos and prior approved artifacts are preserved.")
            else:
                st.warning("No proposal passed the broad size filter. This is a mask-generation failure, not evidence that the SKU is absent.")
    current = latest_mask_events().get(filename)
    if current:
        st.caption(f"Latest saved decision: {current['decision']} · {current.get('timestamp_utc', '')}")


def _scene_page() -> None:
    st.header("Employee / customer scene review")
    st.caption("Add one box per physical item, including detector misses. Never combine touching objects into one ground-truth box.")
    if not SCENE_ROWS:
        st.warning("No scene images are present in the audit manifest.")
        return
    dataset = st.selectbox("Capture type", ["employee", "customer"])
    rows = sorted((row for row in SCENE_ROWS if row["group"] == dataset),
                  key=lambda item: (item["filename"] != "IMG_4857.jpeg", item["filename"]))
    filename = st.selectbox("Scene photograph", [row["filename"] for row in rows])
    row, image = ROWS_BY_NAME[filename], load_verified_image(ROWS_BY_NAME[filename])
    st.image(_thumbnail(image), caption=f"Original photograph: {filename}")
    prediction_files = _prediction_runs()
    model_rows = []
    if prediction_files:
        prediction_file = st.selectbox(
            "RunPod model prediction run",
            prediction_files,
            format_func=lambda path: str(path.parent.relative_to(RESULTS)),
            key="scene_prediction_run",
        )
        with prediction_file.open("r", encoding="utf-8-sig", newline="") as stream:
            model_rows = [item for item in csv.DictReader(stream) if item["photo_filename"] == filename]
        st.subheader("Model suggestions")
        st.caption("NVIDIA DINO boxes with provisional SigLIP2 top-3 rankings. Similarity is not calibrated confidence; these are not human labels.")
        if model_rows:
            st.image(_prediction_overlay(image, model_rows), caption=f"{filename} · orange boxes are model detections")
            registry_names = {entry["sku_id"]: entry["proposed_name"] for entry in load_registry()}
            st.caption("Model predictions use stable candidate IDs; check Product registration for any corrected assigned SKU ID. These ranks remain unconfirmed.")
            display_rows = [{
                "Detection": item["detection_id"],
                "DINO score": round(float(item["detector_confidence"]), 3),
                "Top 1": f"{item['top1_sku_id']} · {registry_names.get(item['top1_sku_id'], item['top1_name'])}",
                "Similarity 1": round(float(item["top1_similarity"]), 3),
                "Top 2": f"{item['top2_sku_id']} · {registry_names.get(item['top2_sku_id'], item['top2_name'])}",
                "Similarity 2": round(float(item["top2_similarity"]), 3) if item["top2_similarity"] else None,
                "Top 3": f"{item['top3_sku_id']} · {registry_names.get(item['top3_sku_id'], item['top3_name'])}",
                "Similarity 3": round(float(item["top3_similarity"]), 3) if item["top3_similarity"] else None,
            } for item in model_rows]
            st.dataframe(display_rows, hide_index=True, use_container_width=True)
            if "masked_top1_sku_id" in model_rows[0]:
                def candidates(item: dict, gallery: str) -> str:
                    return ", ".join(
                        f"{item[f'{gallery}_top{rank}_sku_id']} ({float(item[f'{gallery}_top{rank}_similarity']):.3f})"
                        for rank in (1, 2, 3)
                    )

                st.dataframe([{
                    "Detection": item["detection_id"],
                    "Original top 3": candidates(item, "original"),
                    "Masked top 3": candidates(item, "masked"),
                    "Combined top 3": candidates(item, "combined"),
                    "Top choice changed": item["original_top1_sku_id"] != item["masked_top1_sku_id"],
                    "State": "Unaccepted; human review required",
                } for item in model_rows], hide_index=True, use_container_width=True)
        else:
            st.info("No DINO boxes were returned for this scene. Review the full photo for missed products.")
    else:
        st.info("No RunPod scene prediction results have been imported into this experiment yet.")
    saved = load_scene_labels()
    record = saved.get(filename, {
        "photo_filename": filename, "dataset_type": dataset, "source_path": row["path"],
        "source_sha256": row["sha256"], "arrangement_group": f"single:{filename}",
        "arrangement_confirmed": False, "review_complete": False, "instances": [],
    })
    draft_key = f"draft_{filename}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = list(record.get("instances", []))
    current_boxes = [tuple(item["box_xyxy"]) for item in st.session_state[draft_key]]
    duplicate_boxes = sorted({box for box in current_boxes if current_boxes.count(box) > 1})
    if duplicate_boxes:
        st.error(f"Duplicate annotation coordinates: {duplicate_boxes}. These are overlapping label records, not evidence of separate products. Redraw one distinct box for each visible physical item before marking review complete.")
        if filename == "IMG_4857.jpeg":
            st.info(f"This {image.width} x {image.height} photo visibly shows two stacked cans. Both stored boxes are [0, 0, 476, 570], about one quarter of the image width and height. Keep two physical instances, then correct their separate full-resolution boxes after checking the photo.")
    st.image(_scene_overlay(image, st.session_state[draft_key]), caption=f"{filename} · {image.width}×{image.height}")
    st.caption(f"SHA-256 verified · {len(st.session_state[draft_key])} reviewed object boxes")

    groups = [f"single:{filename}"]
    for group in read_rows(AUDIT / "arrangement_groups.csv"):
        if group["confidence"] == "high" and filename in group["images"].split(";"):
            groups.append(group["group_id"])
    old_group = record.get("arrangement_group", groups[0])
    if old_group not in groups:
        old_group = groups[0]
    chosen_group = st.selectbox("Provisional arrangement group", groups, index=groups.index(old_group), key=f"group_{filename}")
    confirmed_group = st.checkbox("I confirm these are the same unchanged physical products and arrangement", bool(record.get("arrangement_confirmed", False)), key=f"group_confirm_{filename}") if not chosen_group.startswith("single:") else False

    sku_names = {item["sku_id"]: item["proposed_name"] for item in load_registry()}
    sku_options = ["__unknown__"] + list(sku_names)
    source_detection_options = ["None"] + [item["detection_id"] for item in model_rows]
    with st.form(f"add_instance_{filename}"):
        st.markdown("**Add one human-verified physical item**")
        sku = st.selectbox("Candidate SKU / identity", sku_options,
                           format_func=lambda value: "Unknown / unresolved" if value == "__unknown__" else f"{value} · {sku_names[value]} (provisional)",
                           key=f"sku_{filename}")
        cols = st.columns(4)
        x1 = cols[0].number_input("Left x", min_value=0, max_value=image.width, value=0, step=max(1, image.width // 100), key=f"x1_{filename}")
        y1 = cols[1].number_input("Top y", min_value=0, max_value=image.height, value=0, step=max(1, image.height // 100), key=f"y1_{filename}")
        x2 = cols[2].number_input("Right x", min_value=0, max_value=image.width, value=max(1, image.width // 4), step=max(1, image.width // 100), key=f"x2_{filename}")
        y2 = cols[3].number_input("Bottom y", min_value=0, max_value=image.height, value=max(1, image.height // 4), step=max(1, image.height // 100), key=f"y2_{filename}")
        visibility = st.selectbox("Human-reviewed visibility", ["clear", "partly_occluded", "heavily_occluded", "insufficient_to_assess"], key=f"visibility_{filename}")
        relation_options = (["matched_detection", "missed_by_detector", "one_of_multiple_in_merged_detection"]
                            if model_rows else ["unscored_pre_inference"])
        relation = st.selectbox("How did the detector handle this item?", relation_options,
                                key=f"relation_{filename}")
        source_detection = st.selectbox("Related model box", source_detection_options,
                                        key=f"source_detection_{filename}")
        notes = st.text_input("Notes (optional)", key=f"notes_{filename}")
        add = st.form_submit_button("Add instance")
    if add:
        if x2 <= x1 or y2 <= y1:
            st.error("The box must have positive width and height.")
        elif model_rows and relation != "missed_by_detector" and source_detection == "None":
            st.error("Select the related model box, or mark this item missed by the detector.")
        elif relation == "missed_by_detector" and source_detection != "None":
            st.error("A missed item must have no related model box.")
        else:
            st.session_state[draft_key].append({
                "instance_id": f"{filename}:{uuid.uuid4().hex[:12]}",
                "sku_id": sku, "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "visibility": visibility, "eligible_for_bbox_scoring": visibility != "insufficient_to_assess",
                "detection_relation": relation,
                "source_detection_id": None if source_detection == "None" else source_detection,
                "notes": notes,
            })
            st.rerun()

    for index, item in enumerate(st.session_state[draft_key]):
        a, b, c = st.columns([5, 2, 1])
        a.write(f"{index + 1}. `{item['sku_id']}` · {item['box_xyxy']} · {item.get('visibility', '')}")
        b.write("scoreable" if item.get("eligible_for_bbox_scoring", True) else "not scoreable")
        if c.button("Remove", key=f"remove_{filename}_{index}"):
            st.session_state[draft_key].pop(index)
            st.rerun()

        with st.expander(f"Edit physical item {index + 1} box and identity", expanded=filename == "IMG_4857.jpeg" and bool(duplicate_boxes)):
            st.caption(f"Use original-image pixels: width {image.width}, height {image.height}. Editing the draft does not save until you click Save scene review.")
            with st.form(f"edit_instance_{filename}_{index}"):
                edited_sku = st.selectbox("Identity", sku_options,
                                          index=sku_options.index(item["sku_id"]) if item["sku_id"] in sku_options else 0,
                                          key=f"edit_sku_{filename}_{index}")
                box_cols = st.columns(4)
                edited_coords = [
                    box_cols[0].number_input("Left x", min_value=0, max_value=image.width,
                                             value=int(item["box_xyxy"][0]), key=f"edit_x1_{filename}_{index}"),
                    box_cols[1].number_input("Top y", min_value=0, max_value=image.height,
                                             value=int(item["box_xyxy"][1]), key=f"edit_y1_{filename}_{index}"),
                    box_cols[2].number_input("Right x", min_value=0, max_value=image.width,
                                             value=int(item["box_xyxy"][2]), key=f"edit_x2_{filename}_{index}"),
                    box_cols[3].number_input("Bottom y", min_value=0, max_value=image.height,
                                             value=int(item["box_xyxy"][3]), key=f"edit_y2_{filename}_{index}"),
                ]
                visibility_options = ["clear", "partly_occluded", "heavily_occluded", "insufficient_to_assess"]
                edited_visibility = st.selectbox("Visibility", visibility_options,
                                                  index=visibility_options.index(item.get("visibility", "clear")),
                                                  key=f"edit_visibility_{filename}_{index}")
                edited_notes = st.text_input("Correction note", item.get("notes", ""),
                                             key=f"edit_notes_{filename}_{index}")
                apply_edit = st.form_submit_button("Apply to draft")
            if apply_edit:
                if edited_coords[2] <= edited_coords[0] or edited_coords[3] <= edited_coords[1]:
                    st.error("Each physical item needs a positive-area box.")
                else:
                    st.session_state[draft_key][index] = {
                        **item, "sku_id": edited_sku, "box_xyxy": [int(value) for value in edited_coords],
                        "visibility": edited_visibility,
                        "eligible_for_bbox_scoring": edited_visibility != "insufficient_to_assess",
                        "notes": edited_notes,
                    }
                    st.rerun()

    saved_boxes = [tuple(item["box_xyxy"]) for item in record.get("instances", [])]
    saved_has_duplicates = len(saved_boxes) != len(set(saved_boxes))
    complete = st.checkbox("I reviewed the whole image, including products a detector may have missed",
                           bool(record.get("review_complete", False)) and not duplicate_boxes and not saved_has_duplicates,
                           key=f"complete_{filename}")
    if st.button("Save scene review", type="primary", key=f"save_scene_{filename}"):
        if complete and duplicate_boxes:
            st.error("Resolve duplicate annotation boxes before approving this scene as review-complete.")
            return
        saved[filename] = {
            **record, "photo_filename": filename, "dataset_type": dataset,
            "source_path": row["path"], "source_sha256": row["sha256"],
            "arrangement_group": chosen_group, "arrangement_confirmed": bool(confirmed_group),
            "review_complete": bool(complete), "instances": st.session_state[draft_key],
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_scene_labels(saved)
        st.success(f"Saved {len(st.session_state[draft_key])} instances for {filename}.")


def _evaluation_page() -> None:
    st.header("Evaluation")
    st.warning("No scene accuracy is reported until verified per-object boxes and detector predictions are both supplied.")
    if st.button("Run owner-reference SigLIP2 leave-one-photo-out evaluation", type="primary"):
        script = EXPERIMENT / "evaluate_references.py"
        with st.spinner("Encoding fixed owner queries on CPU. The first run may take several minutes."):
            result = subprocess.run([sys.executable, str(script)], cwd=EXPERIMENT, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            st.success("Completed exploratory owner-photo retrieval. This is not employee/customer scene accuracy.")
            st.code(result.stdout[-4000:])
        else:
            st.error("Evaluation could not run. Check the local model cache and mask approvals.")
            st.code((result.stderr or result.stdout)[-5000:])

    st.subheader("Scene prediction evaluation")
    st.write("Upload one row per predicted box using EXIF-corrected image pixels: `photo_filename,detection_id,sku_id,acceptance_status,confidence,x1,y1,x2,y2`. Acceptance status must be `accepted` or `abstained`; provisional ranks alone cannot be scored as accepted predictions.")
    uploaded = st.file_uploader("Detector predictions CSV", type=["csv"])
    if uploaded and st.button("Score reviewed scenes at IoU 0.50"):
        target = EXPERIMENT / "imports" / f"scene_predictions_{int(time.time())}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getvalue())
        script = EXPERIMENT / "evaluate_scene_predictions.py"
        result = subprocess.run([sys.executable, str(script), "--predictions", str(target)], cwd=EXPERIMENT, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            st.success("Scene scoring completed using only review-complete images.")
            st.code(result.stdout[-4000:])
        else:
            st.error("Scene evaluation failed.")
            st.code((result.stderr or result.stdout)[-5000:])
    st.caption("The reference evaluation excludes each query from its own gallery. Genuine unknown-SKU rejection needs reviewed unregistered objects. No acceptance threshold is fitted or lowered here.")


st.title("InventoryLens · New product experiment")
st.caption("Local and isolated · read-only source photos · no production records")
page = st.sidebar.radio("Workspace", ["Product registration", "Reference masks", "Scene review", "Evaluation"])
if page == "Product registration":
    _registration_page()
elif page == "Reference masks":
    _mask_page()
elif page == "Scene review":
    _scene_page()
else:
    _evaluation_page()

