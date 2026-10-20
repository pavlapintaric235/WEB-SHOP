"""Local employee-review demo for experimental masks and few-shot SKU candidates."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".cache" / "ultralytics"))
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / ".venv" / "huggingface_cache"))

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

try:
    from .hybrid_core import ROOT, load_catalog
    from .mask_runtime import MaskFewShotMatcher
    from .segmentation_models import (DEFAULT_SAM_WEIGHTS, YOLO11Segmentation,
                                      YOLOPromptedMobileSAM, render_masks)
except ImportError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.hybrid_pipeline.hybrid_core import load_catalog
    from experiments.hybrid_pipeline.mask_runtime import MaskFewShotMatcher
    from experiments.hybrid_pipeline.segmentation_models import (
        DEFAULT_SAM_WEIGHTS, YOLO11Segmentation, YOLOPromptedMobileSAM, render_masks)


@st.cache_resource(show_spinner="Loading detector, reference photos and visual encoders...")
def load_runtime(segmenter_kind: str, segmenter_weights: str, embedding_mode: str):
    from experiments.hybrid_pipeline.detectors import configure_ultralytics

    configure_ultralytics()
    products = load_catalog()
    if segmenter_kind == "Retail YOLO + MobileSAM refinement":
        segmenter = YOLOPromptedMobileSAM(device="cpu")
    else:
        segmenter = YOLO11Segmentation(Path(segmenter_weights), device="cpu")
    model_names = {
        "MobileNetV2": ("mobilenetv2",),
        "SigLIP2": ("siglip2",),
        "SigLIP2 + MobileNetV2": ("siglip2", "mobilenetv2"),
    }[embedding_mode]
    matcher = MaskFewShotMatcher(products, model_names, device="cpu", local_files_only=True)
    return products, segmenter, matcher


def _add_manual_box(image: Image.Image, x1: int, y1: int, x2: int, y2: int) -> dict:
    left, right = sorted((max(0, min(image.width, x1)), max(0, min(image.width, x2))))
    top, bottom = sorted((max(0, min(image.height, y1)), max(0, min(image.height, y2))))
    if right - left < 2 or bottom - top < 2:
        raise ValueError("The manual region must be at least 2 pixels wide and tall.")
    mask = np.zeros((image.height, image.width), dtype=bool)
    mask[top:bottom, left:right] = True
    return {"box": [left, top, right, bottom], "mask": mask,
            "mask_polygons": [[[left, top], [right, top], [right, bottom], [left, bottom]]],
            "detector_confidence": 0.0, "backend": "employee_manual_rectangle",
            "association": "manual_rectangle"}


def _edited_rows(value) -> list[dict]:
    if hasattr(value, "to_dict"):
        return value.to_dict(orient="records")
    if isinstance(value, list):
        return value
    raise TypeError(f"Unsupported review table result: {type(value).__name__}")


def main() -> None:
    st.set_page_config(page_title="InventoryLens | Mask Review", layout="wide")
    st.title("InventoryLens | Product Mask Review")
    st.caption("Experimental employee review. Predictions stay local and are not written to inventory.")

    segmenter_kind = st.selectbox("Instance proposals", [
        "Retail YOLO + MobileSAM refinement", "YOLO11n-seg checkpoint"
    ])
    segmenter_weights = ""
    if segmenter_kind == "YOLO11n-seg checkpoint":
        segmenter_weights = st.text_input("YOLO11-seg weights path", value="yolo11n-seg.pt")
    embedding_mode = st.selectbox("SKU visual matching", [
        "SigLIP2 + MobileNetV2", "SigLIP2", "MobileNetV2"
    ])

    samples = sorted((ROOT / "data" / "test_runs").glob("20261017-*-test-original.png"))
    sample_names = [path.name for path in samples]
    selected_sample = st.selectbox("Saved October 17 diagnostic photograph", ["Choose file", *sample_names])
    upload = st.file_uploader("Or upload a product photograph", type=["jpg", "jpeg", "png", "webp"])
    photo = None
    source_name = ""
    if upload is not None:
        photo = ImageOps.exif_transpose(Image.open(upload)).convert("RGB")
        source_name = upload.name
    elif selected_sample != "Choose file":
        source_path = next(path for path in samples if path.name == selected_sample)
        with Image.open(source_path) as opened:
            photo = ImageOps.exif_transpose(opened).convert("RGB")
        source_name = source_path.name

    if photo is None:
        st.info("Choose or upload a photograph to begin.")
        return

    if st.button("Propose product masks and SKU matches", type="primary"):
        try:
            products, segmenter, matcher = load_runtime(segmenter_kind, segmenter_weights, embedding_mode)
            with st.spinner("Detecting objects, refining masks and ranking registered SKUs..."):
                detections, timings = segmenter.predict(photo)
                recognized = matcher.recognize(photo, detections)
            st.session_state["mask_review_run"] = {
                "photo": photo, "source": source_name, "detections": recognized,
                "timings": timings, "products": products,
                "segmenter_kind": segmenter_kind, "embedding_mode": embedding_mode,
                "segmenter_weights": segmenter_weights,
                "recognition_thresholds": {
                    "minimum_similarity": matcher.minimum_similarity,
                    "minimum_margin": matcher.minimum_margin,
                },
            }
        except Exception as error:
            st.error(f"The experimental run could not complete: {error}")
            st.stop()

    run = st.session_state.get("mask_review_run")
    if not run or run["source"] != source_name:
        st.image(photo, caption=source_name, use_container_width=True)
        return

    detections = [dict(item) for item in run["detections"]]
    st.image(render_masks(photo, detections), caption=f"{source_name} | {len(detections)} proposed instances",
              use_container_width=True)
    st.caption(f"Proposal model: {run['segmenter_kind']} | SKU features: {run['embedding_mode']}")
    thresholds = run["recognition_thresholds"]
    st.caption(
        f"Automatic acceptance cutoff: similarity >= {thresholds['minimum_similarity']:.2f} "
        f"and margin >= {thresholds['minimum_margin']:.2f}; employee confirmation required."
    )
    st.json(run["timings"])

    sku_options = [product["sku"] for product in run["products"]]
    manual_key = f"manual_masks:{source_name}"
    if manual_key not in st.session_state:
        st.session_state[manual_key] = []
    with st.expander("Add a missed product region"):
        st.caption("A manual region starts as a rectangle. Add one region per visible physical product.")
        with st.form("manual_region_form"):
            col1, col2, col3, col4 = st.columns(4)
            x1 = col1.number_input("Left X", min_value=0, max_value=photo.width, value=0)
            y1 = col2.number_input("Top Y", min_value=0, max_value=photo.height, value=0)
            x2 = col3.number_input("Right X", min_value=0, max_value=photo.width, value=max(1, photo.width // 4))
            y2 = col4.number_input("Bottom Y", min_value=0, max_value=photo.height, value=max(1, photo.height // 4))
            submitted = st.form_submit_button("Add product region")
        if submitted:
            try:
                manual = _add_manual_box(photo, int(x1), int(y1), int(x2), int(y2))
                products, _, matcher = load_runtime(run["segmenter_kind"],
                    str(run.get("segmenter_weights", "")), run["embedding_mode"])
                manual_match = matcher.recognize(photo, [manual])[0]
                manual_match["instance_id"] = len(detections) + len(st.session_state[manual_key]) + 1
                st.session_state[manual_key].append(manual_match)
                st.rerun()
            except Exception as error:
                st.error(str(error))

    detections.extend(st.session_state[manual_key])
    if st.button("Clear manually added regions"):
        st.session_state[manual_key] = []
        st.rerun()

    initial_rows = []
    for index, item in enumerate(detections, 1):
        proposed = item.get("sku") if item.get("status") == "recognized" else "__unknown__"
        initial_rows.append({"Keep": True, "Object": index,
            "Proposed SKU": proposed, "Confirmed SKU": proposed,
            "Similarity": item.get("similarity"), "Reason": item.get("reason", "manual_region")})
    edited = st.data_editor(initial_rows, hide_index=True, use_container_width=True,
        disabled=["Object", "Proposed SKU", "Similarity", "Reason"],
        column_config={
            "Keep": st.column_config.CheckboxColumn(required=True),
            "Confirmed SKU": st.column_config.SelectboxColumn(options=["__unknown__", *sku_options], required=True),
            "Similarity": st.column_config.NumberColumn(format="%.3f"),
        }, key=f"mask-review-{source_name}")
    edited_rows = _edited_rows(edited)

    confirmation = []
    counts: dict[str, int] = {}
    for item, row in zip(detections, edited_rows):
        if not bool(row.get("Keep", True)):
            continue
        sku = row.get("Confirmed SKU") or "__unknown__"
        confirmation.append({"object": item.get("instance_id"), "confirmed_sku": sku,
            "predicted_sku": item.get("sku"), "similarity": item.get("similarity"),
            "detector_confidence": item.get("detector_confidence"),
            "backend": item.get("backend"), "box_xyxy": item.get("box"),
            "mask_polygons_xy": item.get("mask_polygons", []),
            "top_five": item.get("ranked_scores", [])})
        if sku != "__unknown__":
            counts[sku] = counts.get(sku, 0) + 1

    with st.expander("Top-five SKU candidates by product", expanded=False):
        for index, item in enumerate(detections, 1):
            candidates = item.get("ranked_scores", [])
            st.markdown(f"**Object {index}**")
            st.dataframe(candidates, hide_index=True, use_container_width=True)
    st.subheader("Employee-confirmed counts")
    st.dataframe([{"SKU": sku, "Quantity": quantity} for sku, quantity in sorted(counts.items())],
                 hide_index=True, use_container_width=True)

    payload = {"source_image": source_name, "segmenter": run["segmenter_kind"],
        "embedding_mode": run["embedding_mode"], "proposal_timings": run["timings"],
        "recognition_thresholds": run["recognition_thresholds"],
        "reviewed_objects": confirmation, "confirmed_counts": counts,
        "scope": "experimental local review; inventory database not updated"}
    csv_buffer = io.StringIO()
    import csv
    fields = ["object", "confirmed_sku", "predicted_sku", "similarity", "detector_confidence", "backend"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    for row in confirmation:
        writer.writerow({key: row.get(key) for key in fields})
    col_json, col_csv = st.columns(2)
    col_json.download_button("Download confirmed JSON", json.dumps(payload, indent=2),
        file_name=f"{Path(source_name).stem}_reviewed.json", mime="application/json")
    col_csv.download_button("Download confirmed CSV", csv_buffer.getvalue(),
        file_name=f"{Path(source_name).stem}_counts.csv", mime="text/csv")


if __name__ == "__main__":
    main()
