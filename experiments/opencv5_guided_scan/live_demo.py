"""Guided-scan experiment: local OpenCV 5, optional AWS capture, human review.

No recognition model is executed. AWS uploads require an explicit operator click.
Review decisions stay local until the reviewer downloads review-only JSON.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from replay import SOURCE_ROOT, capture_decision, digest, quality


RUN_ROOT = Path(__file__).resolve().parent / "runs/local_replay_20261020"
TRACE_PATH = RUN_ROOT / "replay.json"
NEW_SCENE_MANIFEST = (Path(__file__).resolve().parents[1]
                      / "new_product_scanning_20261020/audit/photo_manifest.csv")
MAX_IMAGE_BYTES = 15 * 1024 * 1024
ACTION_LABELS = {
    "pending": "Pending",
    "accept": "Confirm one item",
    "overlap": "Overlap/partial - exclude from count",
    "split": "Split into manual items",
    "remove": "Remove false box",
}


def load_trace(path: Path = TRACE_PATH) -> dict:
    trace = json.loads(path.read_text(encoding="utf-8"))
    if trace.get("kind") != "inventorylens_opencv5_guided_scan_replay":
        raise ValueError("Unexpected replay trace")
    photos = trace.get("photos", [])
    hashes = [photo["sha256"] for photo in photos]
    if len(photos) != 8 or len(set(hashes)) != len(hashes):
        raise ValueError("Replay trace must contain eight unique source hashes")
    return trace


def inspect_image(data: bytes) -> tuple[np.ndarray | None, dict, dict, str]:
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image is larger than the 15 MB local demo limit")
    try:
        image = (cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                 if data else None)
    except cv2.error:
        image = None
    metrics = quality(image)
    return image, metrics, capture_decision(metrics), digest(data)


def match_saved_photo(image_sha256: str, trace: dict) -> dict | None:
    return next((photo for photo in trace["photos"] if photo["sha256"] == image_sha256), None)


def match_new_scene(image_sha256: str) -> dict | None:
    """Identify an audited new scene by bytes, without treating labels as model output."""
    if not NEW_SCENE_MANIFEST.is_file():
        return None
    with NEW_SCENE_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        return next(({"filename": row["filename"], "group": row["group"]}
                     for row in csv.DictReader(handle)
                     if row["group"] in {"employee", "customer"}
                     and row["sha256"] == image_sha256), None)


def frozen_sample_bytes(photo: dict) -> bytes:
    image_root = (SOURCE_ROOT / "images").resolve(strict=True)
    image_path = (image_root / photo["relative_path"]).resolve(strict=True)
    if not image_path.is_relative_to(image_root):
        raise ValueError("Sample path leaves the frozen image root")
    data = image_path.read_bytes()
    if digest(data) != photo["sha256"]:
        raise ValueError("Frozen sample no longer matches its audited SHA-256")
    return data


def overlay_bytes(image: np.ndarray, boxes: list[dict]) -> bytes:
    canvas = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = (round(value) for value in box["box_xyxy"])
        color = (0, 165, 255) if box["recognition_status"] == "unknown" else (40, 190, 60)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4)
        label = f"#{box['instance_id']} {box['accepted_sku'] or 'unknown'}"
        cv2.putText(canvas, label, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    height, width = canvas.shape[:2]
    if max(height, width) > 1400:
        scale = 1400 / max(height, width)
        canvas = cv2.resize(canvas, (round(width * scale), round(height * scale)),
                            interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("Could not render the model overlay")
    return encoded.tobytes()


def review_export(*, image_sha256: str, source_name: str, input_mode: str,
                  metrics: dict, decision: dict, override: bool, photo: dict | None,
                  box_decisions: list[dict], manual_items: list[dict],
                  b10_total: int | None, b10_note: str,
                  aws_capture: dict | None = None) -> dict:
    reviewed_photo = {
        "photo_id": photo["photo_id"] if photo else None,
        "source_sha256": image_sha256,
        "source_name": source_name,
        "arrangement_id": photo["arrangement_id"] if photo else None,
        "reviewed": True,
        "box_decisions": box_decisions,
        "manual_items": manual_items,
    }
    return {
        "kind": "inventorylens_guided_scan_live_review",
        "status": "review_only_no_inventory_commit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv2.__version__,
        "opencv_python_distribution": version("opencv-python"),
        "input_mode": input_mode,
        "capture": {"quality": metrics, "decision": decision,
                    "human_override": override, "effective_action": "proceed"},
        "aws_capture": aws_capture,
        "inference_mode": ("saved_output_replay_no_model_execution" if photo
                           else "manual_only_no_model_execution"),
        "model_results_are_accuracy_scores": False,
        "photos": [reviewed_photo],
        "arrangements": ({"B10": {"reviewed_total": b10_total,
                                  "reconciliation_note": b10_note,
                                  "source_view_count": 2,
                                  "included_view_count": 1}}
                         if photo and photo["arrangement_id"] == "B10" else {}),
    }


def run_app() -> None:
    st.set_page_config(page_title="InventoryLens guided scan", layout="wide")
    st.title("InventoryLens guided scan")
    st.caption("OpenCV 5 capture guidance, saved DINO/SigLIP2 proposals, human review. AWS upload requires an explicit click. No inventory write.")
    if cv2.__version__.split(".")[0] != "5":
        st.error(f"This demo requires OpenCV 5; found {cv2.__version__}.")
        st.stop()
    try:
        trace = load_trace()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        st.error(f"Could not load the frozen replay: {exc}")
        st.stop()

    input_mode = st.radio("Photo source", ["Upload", "Camera", "Frozen sample"], horizontal=True)
    raw: bytes | None = None
    source_name = ""
    if input_mode == "Upload":
        upload = st.file_uploader("Upload a JPEG or PNG", type=["jpg", "jpeg", "png"])
        if upload is not None:
            raw, source_name = upload.getvalue(), upload.name
    elif input_mode == "Camera":
        capture = st.camera_input("Take a product photo")
        if capture is not None:
            raw, source_name = capture.getvalue(), capture.name
    else:
        labels = {photo["photo_id"]: photo for photo in trace["photos"]}
        selected = st.selectbox("Frozen replay photo", list(labels))
        try:
            raw = frozen_sample_bytes(labels[selected])
            source_name = labels[selected]["filename"]
        except (OSError, ValueError) as exc:
            st.error(f"Cannot read sample: {exc}")
            st.stop()
        if labels[selected]["scene_id"] == "B01":
            variation = st.selectbox("B01 capture control", ["Original", "Blur in memory", "Dark in memory"])
            if variation != "Original":
                original = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                transformed = (cv2.GaussianBlur(original, (0, 0), sigmaX=6)
                               if variation == "Blur in memory"
                               else cv2.convertScaleAbs(original, alpha=0.07, beta=0))
                ok, encoded = cv2.imencode(".jpg", transformed)
                if not ok:
                    st.error("Could not create the in-memory capture control")
                    st.stop()
                raw = encoded.tobytes()
                source_name = f"{source_name} ({variation})"
                st.caption("Control image exists only in memory. Its different SHA-256 prevents archived model proposals from attaching to it.")

    if raw is None:
        st.info("Choose a photo to run the OpenCV 5 capture check.")
        st.stop()
    try:
        image, metrics, decision, image_sha256 = inspect_image(raw)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    photo = match_saved_photo(image_sha256, trace)
    new_scene = match_new_scene(image_sha256) if photo is None else None

    st.subheader("1. Capture guidance")
    cols = st.columns(4)
    cols[0].metric("Size", f"{metrics['width']} x {metrics['height']}" if metrics["decode_ok"] else "Unreadable")
    cols[1].metric("Focus cue", f"{metrics['lap_var_1024']:.1f}" if metrics["decode_ok"] else "-")
    cols[2].metric("Dark pixels", f"{metrics['dark_pct']:.1f}%" if metrics["decode_ok"] else "-")
    cols[3].metric("Bright pixels", f"{metrics['bright_pct']:.1f}%" if metrics["decode_ok"] else "-")
    st.caption("Illustrative thresholds, not calibrated capture-quality accuracy. A person may override a retake prompt.")
    if not metrics["decode_ok"]:
        st.error(decision["reason"])
        st.stop()
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
             caption="Captured photo for the retake decision", width=420)
    if new_scene:
        st.info(f"{new_scene['filename']} is an audited {new_scene['group']} photo from the newer 30-scene set. "
                "AWS checks capture quality here. DINO/SigLIP2 scene predictions have not been generated for this set; "
                "product counting and SKU selection will need human review.")
    elif photo is None:
        st.info("AWS checks capture quality for this photo. Live DINO/SigLIP2 recognition is unavailable; "
                "product counting and SKU selection will need human review.")
    aws_result = None
    use_aws = st.checkbox("Use AWS OpenCV 5 capture check for this photo",
                          key=f"aws-mode-{image_sha256}")
    if use_aws:
        st.caption("Click below to send these exact photo bytes to the private WEB&SHOP pilot in eu-north-1. A one-day S3 lifecycle rule schedules the photo and result for deletion.")
        aws_state_key = f"aws-result-{image_sha256}"
        if st.button("Run AWS capture check", key=f"aws-run-{image_sha256}"):
            try:
                from aws_capture_client import run_capture

                with st.spinner("Waiting for the AWS capture result..."):
                    st.session_state[aws_state_key] = run_capture(raw, image_sha256)
            except Exception as exc:
                st.error(f"AWS capture check failed: {exc}")
                st.stop()
        aws_result = st.session_state.get(aws_state_key)
        if aws_result is None:
            st.info("Click Run AWS capture check to continue with AWS guidance.")
            st.stop()
        if aws_result["image_sha256"] != image_sha256:
            st.error("The AWS result does not match the selected photo")
            st.stop()
        if aws_result["capture_decision"] != decision:
            st.error("AWS and local capture decisions disagree; inspect this photo before review")
            st.stop()
        st.success(f"AWS OpenCV {aws_result['runtime']['cv2_version']}: {aws_result['capture_decision']['reason']}")
        st.caption(f"Matching SHA-256: {image_sha256[:16]}... | AWS processing: {aws_result['processing_ms']:.0f} ms")
    effective_decision = aws_result["capture_decision"] if aws_result else decision
    override = False
    if effective_decision["action"] == "retake":
        st.warning(f"Retake requested: {effective_decision['reason']}")
        override = st.checkbox("I inspected this photo and want to proceed anyway", key=f"override-{image_sha256}")
        if not override:
            st.stop()
    else:
        if not use_aws:
            st.success(effective_decision["reason"])

    st.subheader("2. Product proposals")
    if photo:
        st.info(f"Exact SHA-256 match: {photo['photo_id']}. DINO boxes and SigLIP2 candidates are saved results replayed from the audited archive; no model runs now.")
        st.image(overlay_bytes(image, photo["boxes"]), caption="Green: accepted model suggestion; orange: unknown. All boxes still need human review.")
    else:
        if new_scene:
            st.info(f"{new_scene['filename']} has no imported DINO/SigLIP2 scene predictions yet. "
                    "Inspect each physical product and add it manually below; leave uncertain SKUs unresolved.")
        else:
            st.info("This photo does not match a frozen source image. Live DINO/SigLIP2 inference is unavailable; inspect products and add them manually.")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded photo")

    st.subheader("3. Human review")
    st.caption("Each physical product counts once. Use overlap for a box covering already represented products; use split for a standalone merged box. Nothing commits to inventory.")
    box_decisions = []
    boxes = photo["boxes"] if photo else []
    for box in boxes:
        with st.container(border=True):
            candidates = ", ".join(f"{item['sku']} ({item['similarity']:.2f})" for item in box["top5"][:3])
            st.markdown(f"**Box #{box['instance_id']}** · {box['recognition_status']} · candidates: {candidates}")
            key = f"{image_sha256}-{box['instance_id']}"
            col1, col2, col3 = st.columns([2, 1, 1])
            action = col1.selectbox("Review action", list(ACTION_LABELS),
                                    format_func=ACTION_LABELS.get, key=f"action-{key}")
            sku = col2.text_input("SKU", value=box["accepted_sku"] or "", key=f"sku-{key}").strip() or None
            quantity = col3.number_input("Units", min_value=1, max_value=50, value=1,
                                         disabled=action != "accept", key=f"qty-{key}")
            note = st.text_input("Reason or correction note", key=f"note-{key}").strip()
            box_decisions.append({"replay_id": box["replay_id"], "decision": action,
                                  "countable": action == "accept", "sku": sku,
                                  "quantity": int(quantity), "note": note})

    manual_count = st.number_input("Manual item rows", min_value=0, max_value=30, value=0,
                                   help="Add missed products or physical items from a merged box.",
                                   key=f"manual-count-{image_sha256}")
    manual_items = []
    for index in range(int(manual_count)):
        cols = st.columns([1, 1, 2])
        sku = cols[0].text_input(f"Manual #{index + 1} SKU", key=f"manual-sku-{image_sha256}-{index}").strip() or None
        quantity = cols[1].number_input(f"Manual #{index + 1} units", min_value=1, max_value=50,
                                        value=1, key=f"manual-qty-{image_sha256}-{index}")
        note = cols[2].text_input(f"Manual #{index + 1} note", key=f"manual-note-{image_sha256}-{index}").strip()
        manual_items.append({"manual_id": f"{photo['photo_id'] if photo else image_sha256}:manual:{index + 1}",
                             "countable": True, "sku": sku, "quantity": int(quantity), "note": note})

    b10_total: int | None = None
    b10_note = ""
    if photo and photo["arrangement_id"] == "B10":
        st.info("B10 has two views of one arrangement. The latest user review confirms 10 units; never sum the views.")
        total_text = st.text_input("One reconciled B10 arrangement total", key=f"b10-total-{image_sha256}").strip()
        b10_note = st.text_input("How were the two views reconciled?", key=f"b10-note-{image_sha256}").strip()
        if total_text.isdigit():
            b10_total = int(total_text)

    reviewed = st.checkbox("I checked every visible product in this photo", key=f"reviewed-{image_sha256}")
    if not reviewed:
        st.info("Finish the review to enable the review-only JSON download.")
        return
    if any(item["decision"] == "pending" for item in box_decisions):
        st.warning("Resolve every model box before exporting.")
        return
    if any(item["decision"] == "split" for item in box_decisions) and not manual_items:
        st.warning("Add the physical products represented by the split box.")
        return
    if photo and photo["arrangement_id"] == "B10" and b10_total is None:
        st.warning("Enter one B10 arrangement total before exporting.")
        return
    if photo and photo["arrangement_id"] == "B10" and not b10_note:
        st.warning("Explain how the two B10 views were reconciled before exporting.")
        return

    export = review_export(image_sha256=image_sha256, source_name=source_name,
                           input_mode=input_mode.lower().replace(" ", "_"),
                           metrics=metrics, decision=effective_decision, override=override,
                           photo=photo, box_decisions=box_decisions,
                           manual_items=manual_items, b10_total=b10_total,
                           b10_note=b10_note, aws_capture=aws_result)
    count = sum(item["quantity"] for item in box_decisions + manual_items if item["countable"])
    st.success(f"Review-only count for this photo: {count} units. Check the SKU list before using it elsewhere.")
    st.download_button("Download review-only JSON",
                       data=json.dumps(export, indent=2, ensure_ascii=False),
                       file_name=f"inventorylens_guided_scan_{image_sha256[:12]}.json",
                       mime="application/json")


if __name__ == "__main__":
    run_app()
