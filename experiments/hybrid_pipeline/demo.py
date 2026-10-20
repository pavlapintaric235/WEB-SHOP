"""Local, isolated Streamlit demo for exemplar-based product counting."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from time import perf_counter

import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((EXPERIMENT_DIR / ".cache" / "ultralytics").resolve()))
os.environ.setdefault("HF_HOME", str(ROOT / ".venv" / "huggingface_cache"))

try:
    from .detectors import RetailYOLODetector, YOLOEVisualPromptFallback, configure_ultralytics
    from .hf_encoders import DinoV2SmallEncoder, SigLIP2BaseEncoder
    from .hybrid_core import (DinoV3Encoder, MobileNetV2Encoder, abstaining_decision,
                              build_reference_index, crop_masked, load_catalog,
                              product_reference_paths, render_boxes, scores_for_features,
                              apply_manual_corrections)
except ImportError:
    from experiments.hybrid_pipeline.detectors import (RetailYOLODetector,
        YOLOEVisualPromptFallback, configure_ultralytics)
    from experiments.hybrid_pipeline.hf_encoders import DinoV2SmallEncoder, SigLIP2BaseEncoder
    from experiments.hybrid_pipeline.hybrid_core import (DinoV3Encoder,
        MobileNetV2Encoder, abstaining_decision, build_reference_index, crop_masked,
        load_catalog, product_reference_paths, render_boxes, scores_for_features,
        apply_manual_corrections)


@st.cache_resource(show_spinner="Loading local models and registered SKU references...")
def base_resources(detector_choice: str, embedding_mode: str):
    products = load_catalog()
    encoders = {}
    if embedding_mode in {"MobileNetV2 only", "DINOv2 small + MobileNetV2"}:
        encoders["mobilenetv2"] = MobileNetV2Encoder()
    if embedding_mode in {"DINOv2 small", "DINOv2 small + MobileNetV2"}:
        encoders["dinov2"] = DinoV2SmallEncoder(local_files_only=True)
    if embedding_mode == "SigLIP2 Base":
        encoders["siglip2"] = SigLIP2BaseEncoder(local_files_only=True)
    if embedding_mode in {"DINOv3 only", "DINOv3 + MobileNetV2"}:
        repo = Path(os.environ.get("DINOV3_REPO", ""))
        weights = Path(os.environ.get("DINOV3_WEIGHTS", ""))
        dino = DinoV3Encoder(repo, weights)
        encoders = {dino.name: dino} if embedding_mode == "DINOv3 only" else {**encoders, dino.name: dino}
    index = build_reference_index(products, encoders)
    configure_ultralytics()
    detector = (YOLOEVisualPromptFallback(device="cpu") if detector_choice == "YOLOE visual-prompt fallback"
                else RetailYOLODetector(device="cpu"))
    return products, encoders, index, detector


def run_recognition(photo, source_name, products, encoders, base_index, detector,
                    confidence, minimum, margin, reference_uploads, new_sku, new_name):
    active_products = list(products)
    index = {sku: dict(branches) for sku, branches in base_index.items()}
    with tempfile.TemporaryDirectory(prefix="webshop-hybrid-") as temp_dir:
        if reference_uploads:
            sku = new_sku.strip()
            if not sku:
                raise ValueError("Enter a temporary SKU before uploading its reference photographs.")
            if sku in index:
                raise ValueError("That SKU is already registered. Use a different temporary SKU.")
            paths, images = [], []
            for number, uploaded in enumerate(reference_uploads, 1):
                image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                path = Path(temp_dir) / f"reference_{number}_{Path(uploaded.name).name}"
                image.save(path)
                paths.append(path)
                images.append(image)
            index[sku] = {}
            for name, encoder in encoders.items():
                vectors = encoder.encode(images)
                index[sku][name] = [{"feature": vector, "image": str(path), "sha256": "session-only"}
                                    for vector, path in zip(vectors, paths)]
            active_products.append({"sku": sku, "name": new_name.strip() or sku,
                                    "image_path": str(paths[0]),
                                    "references": [{"id": None, "image_path": str(path)} for path in paths]})

        start = perf_counter()
        detections = detector.predict(photo, active_products, conf=confidence)
        detection_seconds = perf_counter() - start
        crops = []
        for item in detections:
            if item.get("mask") is not None and item["mask"].any():
                crops.append(crop_masked(photo, item["mask"]))
            else:
                x1, y1, x2, y2 = [int(round(value)) for value in item["box"]]
                crops.append(photo.crop((max(0, x1), max(0, y1), min(photo.width, x2), min(photo.height, y2))))
        start = perf_counter()
        vectors = {name: encoder.encode(crops) for name, encoder in encoders.items()}
        for position, item in enumerate(detections):
            features = {name: values[position] for name, values in vectors.items()}
            scores = scores_for_features(features, index)[0]
            item.update(abstaining_decision(scores, minimum, margin))
        matching_seconds = perf_counter() - start
    return {"id": uuid.uuid4().hex[:10], "photo": photo, "source": source_name,
            "detector": detector.backend, "recognizer": ", ".join(encoders) + " few-shot cosine retrieval",
            "detections": detections, "detection_seconds": detection_seconds,
            "matching_seconds": matching_seconds, "minimum": minimum, "margin": margin}


def main() -> None:
    st.set_page_config(page_title="Web&Shop Hybrid Lab", layout="wide")
    st.title("Web&Shop Hybrid Lab")
    st.caption("Isolated few-shot experiment. Nothing is written to the production catalogue or inventory.")
    st.warning("The SKU-110K detector is research-only. YOLOE is a visual-prompt fallback, not the SKU-110K model.")

    retail_weights = EXPERIMENT_DIR / "weights" / "sku110k-yolo11-n640.onnx"
    detector_options = (["SKU-110K YOLO11n ONNX (research-only)"] if retail_weights.is_file() else []) + ["YOLOE visual-prompt fallback"]
    detector_choice = st.selectbox("Detector", detector_options)
    if detector_choice.startswith("SKU-110K"):
        st.error("Research-only checkpoint. Do not deploy commercially without written rights clearance.")
    else:
        st.info("YOLOE fallback uses registered image prompts. It is not a general retail detector.")

    embedding_modes = ["MobileNetV2 only", "DINOv2 small", "DINOv2 small + MobileNetV2", "SigLIP2 Base"]
    dino3_ready = bool(os.environ.get("DINOV3_REPO") and os.environ.get("DINOV3_WEIGHTS"))
    if dino3_ready:
        embedding_modes += ["DINOv3 only", "DINOv3 + MobileNetV2"]
    embedding_mode = st.selectbox("SKU embedding", embedding_modes)
    if embedding_mode != "MobileNetV2 only":
        st.caption("Pretrained weights load from the local Hugging Face cache only. No weights are downloaded automatically.")

    try:
        products = load_catalog()
    except Exception as error:
        st.error(f"Could not read the registered SKU catalogue: {error}")
        st.stop()
    sku_options = [product["sku"] for product in products]
    with st.expander("Registered SKUs", expanded=False):
        st.dataframe([{"SKU": product["sku"], "Product": product["name"],
                       "Reference photos": len(product_reference_paths(product))} for product in products],
                     use_container_width=True, hide_index=True)

    new_sku = st.text_input("Temporary SKU to try", placeholder="e.g. DEMO-CHIPS-01")
    new_name = st.text_input("Temporary product name", placeholder="Optional display name")
    reference_uploads = st.file_uploader("Reference photos for this SKU", type=["jpg", "jpeg", "png", "webp"],
                                         accept_multiple_files=True, key="new_sku_refs")
    st.caption("Temporary registrations exist only for this analysis session.")
    uploaded_photo = st.file_uploader("Product photograph", type=["jpg", "jpeg", "png", "webp"], key="scene")
    sample_paths = sorted((ROOT / "data/test_runs").glob("20261017-*-test-original.png"))
    sample_labels = ["Upload a photograph"] + [path.name for path in sample_paths]
    selected_sample = st.selectbox("Or choose a saved October 17 development photograph", sample_labels)
    minimum = st.slider("Minimum exemplar similarity", 0.0, 1.0, 0.55, 0.01)
    margin = st.slider("Minimum top-two margin", 0.0, 0.5, 0.04, 0.01)
    confidence = st.slider("Detector proposal confidence", 0.01, 0.50, 0.05, 0.01)
    run = st.button("Run recognition", type="primary",
                    disabled=(uploaded_photo is None and selected_sample == "Upload a photograph"))

    if run:
        try:
            with st.spinner("Loading the detector and registered SKU references..."):
                products, encoders, base_index, detector = base_resources(detector_choice, embedding_mode)
            if uploaded_photo is not None:
                photo = ImageOps.exif_transpose(Image.open(uploaded_photo)).convert("RGB")
                source_name = uploaded_photo.name
            else:
                photo_path = next(path for path in sample_paths if path.name == selected_sample)
                with Image.open(photo_path) as source:
                    photo = ImageOps.exif_transpose(source).convert("RGB")
                source_name = photo_path.name
            with st.spinner("Detecting objects and matching SKU references..."):
                st.session_state["hybrid_last_run"] = run_recognition(
                    photo, source_name, products, encoders, base_index, detector,
                    confidence, minimum, margin, reference_uploads, new_sku, new_name)
        except Exception as error:
            st.error(f"Recognition could not run: {error}")
            if "transformers" in str(error).lower() or "local cache" in str(error).lower():
                st.caption("Install the optional encoder requirements and download its pretrained checkpoint explicitly when disk space is available.")
            return

    result = st.session_state.get("hybrid_last_run")
    if not result:
        return
    detections = result["detections"]
    proposed = [{**item, "sku": item.get("sku") if item.get("status") == "recognized" else None}
                for item in detections]
    predicted_counts = Counter(item["sku"] for item in proposed if item.get("sku"))
    initial = [{"Object": number, "Proposed SKU": item.get("sku") or "__unknown__",
                "Confirmed SKU": item.get("sku") or "__unknown__", "Quantity": 1,
                "Similarity": round(item["similarity"], 4) if item.get("similarity") is not None else None,
                "Detector score": round(item.get("detector_confidence", 0.0), 4)}
               for number, item in enumerate(proposed, 1)]
    st.subheader(f"Review: {result['source']}")
    edited = st.data_editor(initial, key=f"manual_review_{result['id']}", hide_index=True,
        use_container_width=True, num_rows="fixed",
        column_config={
            "Object": st.column_config.NumberColumn(disabled=True),
            "Proposed SKU": st.column_config.TextColumn(disabled=True),
            "Confirmed SKU": st.column_config.SelectboxColumn(options=["__unknown__", *sku_options], required=True),
            "Quantity": st.column_config.NumberColumn(min_value=0, max_value=100, step=1, required=True),
            "Similarity": st.column_config.NumberColumn(disabled=True, format="%.4f"),
            "Detector score": st.column_config.NumberColumn(disabled=True, format="%.4f"),
        })

    edited_records = edited.to_dict("records") if hasattr(edited, "to_dict") else edited
    correction_rows = [{"sku": row.get("Confirmed SKU"), "quantity": row.get("Quantity", 1)}
                       for row in edited_records]
    final_detections, confirmed_counts = apply_manual_corrections(detections, correction_rows)
    exports = [{"object": index + 1, "predicted_sku": item["predicted_sku"],
                "confirmed_sku": item["confirmed_sku"], "quantity": item["confirmed_quantity"],
                "similarity": item.get("similarity"),
                "runner_up_similarity": item.get("runner_up_similarity"),
                "margin": item.get("margin"),
                "detector_confidence": item.get("detector_confidence"),
                "box_xyxy": item.get("box")}
               for index, item in enumerate(final_detections)]

    st.image(render_boxes(result["photo"], final_detections), caption=result["source"], use_container_width=True)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Counts")
        st.caption("Automatic proposal counts")
        st.dataframe([{"SKU": sku, "Quantity": count} for sku, count in sorted(predicted_counts.items())],
                     use_container_width=True, hide_index=True)
        st.caption("Confirmed counts")
        st.dataframe([{"SKU": sku, "Quantity": count} for sku, count in sorted(confirmed_counts.items())],
                     use_container_width=True, hide_index=True)
        st.metric("Unknown / excluded detections", sum(row["confirmed_sku"] == "__unknown__" and row["quantity"] > 0 for row in exports))
        st.caption(f"Detector {result['detection_seconds']:.2f}s | Matching {result['matching_seconds']:.2f}s | Total {result['detection_seconds'] + result['matching_seconds']:.2f}s (CPU)")
    with right:
        st.subheader("Run details")
        st.write(f"{result['detector']} · {result['recognizer']} · thresholds {result['minimum']:.2f}/{result['margin']:.2f}")
        st.dataframe([{ "Object": number, "Proposed SKU": row["predicted_sku"] or "__unknown__",
                        "Confirmed SKU": row["confirmed_sku"], "Quantity": row["quantity"],
                        "Reason": detections[number - 1].get("reason"),
                        "Box XYXY": [round(float(value), 1) for value in detections[number - 1]["box"]]}
                       for number, row in enumerate(exports, 1)], use_container_width=True, hide_index=True)

    payload = {"source": result["source"], "detector": result["detector"],
               "recognizer": result["recognizer"], "minimum_similarity": result["minimum"],
               "minimum_margin": result["margin"], "automatic_counts": dict(sorted(predicted_counts.items())),
               "confirmed_counts": dict(sorted(confirmed_counts.items())),
               "timing_seconds": {"detection": result["detection_seconds"],
                                  "matching": result["matching_seconds"],
                                  "total": result["detection_seconds"] + result["matching_seconds"]},
               "detections": exports}
    json_bytes = json.dumps(payload, indent=2).encode("utf-8")
    csv_buffer = io.StringIO()
    fields = ["object", "predicted_sku", "confirmed_sku", "quantity", "similarity",
              "runner_up_similarity", "margin", "detector_confidence", "box_xyxy"]
    writer = csv.DictWriter(csv_buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(exports)
    stem = Path(result["source"]).stem
    st.download_button("Export confirmed JSON", json_bytes,
                       file_name=f"{stem}-confirmed.json", mime="application/json")
    st.download_button("Export confirmed CSV", csv_buffer.getvalue(),
                       file_name=f"{stem}-confirmed.csv", mime="text/csv")


if __name__ == "__main__":
    main()
