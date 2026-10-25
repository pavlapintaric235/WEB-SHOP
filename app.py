"""InventoryLens: reference-photo onboarding -> detected counts -> approved CSV.

Run with: streamlit run app.py
"""

import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime
from threading import Lock
from uuid import uuid4
from zoneinfo import ZoneInfo

import streamlit as st

from vision import detect_products, load_model
import auth
import db
from inventory import InventoryService
from owner_ui import dashboard
from images import product_references, read_image
from sam_vision import (DEFAULT_MARGIN, DEFAULT_SIMILARITY, RECOGNITION_VERSION,
                        REFERENCE_CACHE_VERSION, render_detections)

DATA_DIR = db.DATA_DIR
LAST_DETECTION_FILE = DATA_DIR / "last_detection.json"
LAST_ANNOTATED_FILE = DATA_DIR / "last_annotated.png"
TEST_RUNS_DIR = DATA_DIR / "test_runs"
TEST_RUNS_FILE = DATA_DIR / "test_runs.jsonl"
TEST_RUNS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="WEB&SHOP", layout="wide")


def make_csv(products, approved_counts, existing_file=None):
    """Overwrite the pictured AREA counts by SKU, not whole-store inventory."""
    for value in approved_counts.values():
        db.validate_quantity(value)
    if existing_file is None:
        fields = ["sku", "name", "quantity"]
        rows = [
            {"sku": p["sku"], "name": p["name"],
             "quantity": str(approved_counts[p["sku"]])}
            for p in products
        ]
    else:
        try:
            text = existing_file.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("CSV must use UTF-8 encoding.") from error
        fields, rows = db.validate_csv_text(text)
        by_sku = {p["sku"]: p for p in products}
        seen = set()
        for row in rows:
            sku = row.get("sku", "").strip().upper()
            if sku in approved_counts:
                row["quantity"] = str(approved_counts[sku])
                seen.add(sku)
        for sku, count in approved_counts.items():
            if sku not in seen:
                row = {field: "" for field in fields}
                row["sku"] = sku
                row["quantity"] = str(count)
                if "name" in fields:
                    row["name"] = by_sku[sku]["name"]
                rows.append(row)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def safe_filename(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    return value.strip("-")[:60] or "test"


def read_test_runs(business_id, limit=10):
    if not TEST_RUNS_FILE.exists():
        return []
    runs = []
    for line in TEST_RUNS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            run = json.loads(line)
            if run.get("business_id", demo_business["id"]) == business_id:
                runs.append(run)
        except json.JSONDecodeError:
            continue
    return runs[-limit:]


def compare_counts(products, predicted_counts, expected_counts):
    rows = []
    for product in products:
        sku = product["sku"]
        expected = int(expected_counts.get(sku, 0))
        predicted = int(predicted_counts.get(sku, 0))
        rows.append(
            {
                "sku": sku,
                "name": product["name"],
                "expected": expected,
                "predicted": predicted,
                "error": predicted - expected,
                "correct": predicted == expected,
            }
        )
    return rows


def save_detection_snapshot(
    detections, annotated, products, confidence, photo_hash,
    test_name="", expected_counts=None, notes="", raw_detections=None,
    user=None, original=None, engine="sam", min_margin=DEFAULT_MARGIN, timings=None
):
    predicted_counts = dict(Counter(detection["sku"] for detection in detections if detection.get("sku")))
    created_at = datetime.now(ZoneInfo("Europe/Zagreb")).isoformat(timespec="seconds")
    run_id = datetime.now(ZoneInfo("Europe/Zagreb")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    label = safe_filename(test_name)
    annotated_path = TEST_RUNS_DIR / f"{run_id}-{label}.png"

    annotated.save(LAST_ANNOTATED_FILE)
    annotated.save(annotated_path)
    original_path = TEST_RUNS_DIR / f"{run_id}-{label}-original.png"
    if original is not None:
        original.save(original_path)
    snapshot = {
        "run_id": run_id,
        "created_at": created_at,
        "test_name": test_name.strip() or "Untitled test",
        "notes": notes.strip(),
        "confidence": confidence,
        "recognition_engine": engine,
        "reference_cache_version": REFERENCE_CACHE_VERSION if engine == "sam" else None,
        "recognition_version": RECOGNITION_VERSION if engine == "sam" else None,
        "score_type": "cosine_similarity" if engine == "sam" else "detector_confidence",
        "minimum_match_margin": min_margin if engine == "sam" else None,
        "timings": timings,
        "unassigned_regions": sum(not detection.get("sku") for detection in detections),
        "business_id": user["business_id"] if user else None,
        "user_id": user["id"] if user else None,
        "photo_hash": photo_hash,
        "products": [
            {"sku": product["sku"], "name": product["name"],
             "references": product_references(product)}
            for product in products
        ],
        "expected_counts": expected_counts,
        "counts": predicted_counts,
        "comparison": compare_counts(products, predicted_counts, expected_counts)
        if expected_counts is not None else [],
        "detections": detections,
        "raw_detections": detections if raw_detections is None else raw_detections,
        "annotated_image": str(LAST_ANNOTATED_FILE),
        "run_annotated_image": str(annotated_path),
        "original_image": str(original_path) if original is not None else None,
    }
    LAST_DETECTION_FILE.write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )
    with TEST_RUNS_FILE.open("a", encoding="utf-8") as history:
        history.write(json.dumps(snapshot) + "\n")
    return snapshot


@st.cache_resource(show_spinner="Loading recognition models...")
def cached_model(engine, recognition_version=RECOGNITION_VERSION):
    """Version the resource key so hot reload cannot retain the old reference selector."""
    return load_model(engine)


@st.cache_resource
def inference_lock():
    return Lock()


st.title("WEB&SHOP")
st.caption('InventoryLens · visual product registration and inventory review')
db.init_db()
demo_business = auth.seed_demo_users()
db.default_location(demo_business["id"])
db.migrate_catalog_json(demo_business["id"])

if "user_id" not in st.session_state:
    entry = st.radio('Role view', ['Owner / employee sign in', 'Buyer'])
    if entry == 'Buyer':
        from buyer_ui import buyer_view
        buyer_view()
        st.stop()
    with st.form("login"):
        username = st.text_input("Email or username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.form_submit_button("Sign in", type="primary"):
            user = auth.authenticate(username, password)
            if user:
                st.session_state["user_id"] = user["id"]
                st.rerun()
            else:
                st.error("Incorrect email, username or password.")
    st.stop()

service = InventoryService(st.session_state["user_id"])
try:
    user = service.user()
except PermissionError as error:
    st.session_state.clear()
    st.error(str(error))
    st.stop()
is_owner = user["role"] == "owner"
with st.sidebar:
    st.write(db.get_business(user["business_id"])["name"])
    st.caption(f'{user["username"]} | {user["role"]}')
    if st.button("Sign out", icon=":material/logout:"):
        st.session_state.clear()
        st.rerun()
    page = st.radio("Workspace", ["Owner dashboard", "Fresh-photo SKU scan", "Buyer", "Scanner"] if is_owner
                    else ["Scanner", "Fresh-photo SKU scan", "Buyer"])

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
if page == "Owner dashboard":
    dashboard(service, read_image)
    st.stop()
if page == 'Buyer':
    from buyer_ui import buyer_view
    buyer_view()
    st.stop()
if page == "Fresh-photo SKU scan":
    from scanner_ui import fresh_scan
    fresh_scan(service)
    st.stop()

products = service.products()
locations = service.locations()
if not products or not locations:
    st.info("An owner must register products and an inventory location before scanning.")
    st.stop()

st.subheader("Inventory scan")
location_id = st.selectbox(
    "Count location", [item["id"] for item in locations],
    format_func=lambda value: next(item["name"] for item in locations if item["id"] == value),
)
source = st.radio("Photo source", ["Upload photo", "Camera"], horizontal=True)
if source == "Camera":
    photo_upload = st.camera_input("Capture inventory photo", key="camera_photo")
else:
    photo_upload = st.file_uploader(
        "Inventory photograph", type=["png", "jpg", "jpeg"], key="inventory_photo"
    )
engine, confidence, min_margin = "sam", DEFAULT_SIMILARITY, DEFAULT_MARGIN
if is_owner:
    with st.expander("Recognition settings"):
        engine = st.selectbox("Recognition engine", ["sam", "yoloe"], format_func=lambda value: {
            "sam": "MobileSAM + reference matching", "yoloe": "YOLOE comparison (primary reference only)"
        }[value])
        if engine == "sam":
            confidence = st.slider("Minimum reference similarity", 0.50, 0.99, DEFAULT_SIMILARITY, 0.01)
            min_margin = st.slider("Minimum gap to second-best product", 0.0, 0.30, DEFAULT_MARGIN, 0.01)
        else:
            confidence = st.slider("Minimum YOLOE confidence", 0.05, 0.9, 0.25, 0.05)
test_name, test_notes, expected_counts = "", "", None
if is_owner:
    with st.expander("Test tracker"):
        test_name = st.text_input(
            "Test name", value="", placeholder="Test A - one of each at 0.40",
        )
        test_notes = st.text_area(
            "Notes", value="", placeholder="Lighting, spacing, angle", height=80,
        )
        if st.checkbox("Record expected counts for this test"):
            expected_counts = {}
            photo_key = hashlib.sha256(photo_upload.getvalue()).hexdigest()[:12] if photo_upload else "no-photo"
            for product in products:
                expected_counts[product["sku"]] = st.number_input(
                    f'{product["name"]} ({product["sku"]}) expected',
                    min_value=0, value=0, step=1, key=f'expected_{photo_key}_{product["sku"]}',
                )
            if not any(expected_counts.values()):
                test_key = hashlib.sha256(f"{photo_key}:{test_name.strip()}".encode()).hexdigest()[:12]
                confirmed_negative = st.checkbox(
                    "Confirm this is an intentional test with zero products",
                    key=f"confirm_empty_expected_{test_key}",
                )
                if not confirmed_negative:
                    expected_counts = None
                    st.warning(
                        "All expected quantities are zero. This run will not be scored unless you "
                        "confirm it is an intentional empty-scene test."
                    )
        elif test_name.strip():
            st.info("This run will be saved without expected counts and will not be scored.")

if photo_upload is not None:
    try:
        current_photo = read_image(photo_upload)
        st.image(current_photo, caption="Inventory photo", use_container_width=True)
    except ValueError as error:
        st.error(str(error))
        current_photo = None
else:
    current_photo = None

if current_photo is None:
    st.info("Detection is disabled until you upload a readable inventory photo in this section.")
else:
    st.success("Ready for detection: catalogue product(s) and inventory photo are loaded.")

photo_hash = hashlib.sha256(photo_upload.getvalue()).hexdigest() if photo_upload else ""
catalog_hash = hashlib.sha256(json.dumps(products, sort_keys=True).encode()).hexdigest()
current_signature = (photo_hash, catalog_hash, engine, confidence, min_margin, location_id,
                     user["id"], RECOGNITION_VERSION if engine == "sam" else None)

if st.button("Run real AI detection", type="primary",
             disabled=not products or current_photo is None):
    st.session_state.pop("analysis", None)
    reference_progress = st.empty()
    stage_status = st.empty()

    def show_progress(event):
        if event["stage"] == "references":
            count, total = event["completed"], event["total"]
            label = f"Preparing references: {count}/{total}"
            if event.get("sku"):
                label += f' | {event["sku"]}, view {event["view"]}'
            reference_progress.progress(count / total, text=label)
        else:
            reference_progress.empty()
            stage_status.text({"photo": "Segmenting inventory photo...",
                               "matching": "Matching product regions...",
                               "rendering": "Rendering results..."}[event["stage"]])

    try:
        with st.spinner("Analyzing product regions and reference matches..."):
            with inference_lock():
                model = cached_model(engine, RECOGNITION_VERSION)
                detections, annotated, board, raw_detections = detect_products(
                    model, current_photo, products, confidence, min_margin, progress=show_progress
                )
                timings = getattr(model, "last_timings", None) if engine == "sam" else None
        spotlight, spotlight_kind = render_detections(current_photo, detections, spotlight=True)
        snapshot = None
        try:
            snapshot = save_detection_snapshot(
                detections, annotated, products, confidence, photo_hash,
                test_name, expected_counts, test_notes, raw_detections, user, current_photo, engine, min_margin, timings,
            )
        except OSError as error:
            st.warning(f"Detection completed but the diagnostic log could not be saved: {error}")
        st.session_state["analysis"] = {
            "signature": current_signature,
            "detections": detections,
            "annotated": annotated,
            "board": board,
            "snapshot": snapshot,
            "raw_detections": raw_detections,
            "submission_key": uuid4().hex,
            "spotlight": spotlight,
            "spotlight_kind": spotlight_kind,
            "timings": timings,
        }
        for product in products:
            st.session_state.pop(f'approved_{product["sku"]}', None)
    except Exception as error:
        st.error(f"AI inference failed: {error}")
        st.info("Check the terminal output, model download, and available memory.")
    finally:
        reference_progress.empty()
        stage_status.empty()

analysis = st.session_state.get("analysis")
if analysis and analysis["signature"] != current_signature:
    st.info("The photo, catalogue or threshold changed, or recognition was updated. Run detection again.")
elif analysis:
    st.subheader("3. Review detections")
    original_view = st.toggle("Original photo")
    spotlight_view = st.toggle("Darken background", value=True, disabled=original_view)
    shown = current_photo if original_view else analysis.get("spotlight", analysis["annotated"]) if spotlight_view else analysis["annotated"]
    st.image(shown, use_container_width=True)
    if not original_view and spotlight_view:
        st.caption("Segmentation spotlight" if analysis.get("spotlight_kind") == "instance_mask"
                   else "Bounding-box spotlight")
        st.caption("Darkened areas may include missed products.")
    with st.expander("See visual reference board"):
        st.image(analysis["board"])
    counts = Counter(d["sku"] for d in analysis["detections"] if d.get("sku"))
    st.write(f"Proposed registered-product count: {sum(counts.values())}")
    timings = analysis.get("timings")
    if timings:
        st.caption(f'Analysis: {timings["total_seconds"]:.1f}s | '
                   f'References: {timings["references_seconds"]:.1f}s | '
                   f'Photo segmentation: {timings["photo_segmentation_seconds"]:.1f}s')
    uncertain = sum(not d.get("sku") for d in analysis["detections"])
    if uncertain:
        st.warning(f"{uncertain} uncertain region(s) are excluded from proposed SKU counts. Review before approval.")
    with st.expander("Detection details"):
        raw_detections = analysis.get("raw_detections", analysis["detections"])
        st.caption(
            f"Raw model boxes: {len(raw_detections)} | "
            f"Matched products: {sum(counts.values())} | Uncertain regions: {uncertain}"
        )
        if analysis["detections"]:
            st.dataframe(
                [
                    {
                        "SKU": detection["sku"],
                        "Product": detection["name"],
                        "Method": detection.get("method", "unknown"),
                        "Status": detection.get("status", "recognized"),
                        "Score": round(detection["confidence"], 3),
                        "Score type": detection.get("score_type", "legacy_score"),
                        "Match gap": detection.get("match_margin"),
                        "Best match SKU": (detection.get("matches") or [{}])[0].get("sku"),
                        "Reference view": (detection.get("matches") or [{}])[0].get("reference_view"),
                        "Reason": detection.get("reason", ""),
                        "Label verification": detection.get("geometric_verification", {}).get("status", "not used"),
                        "Verified label points": detection.get("geometric_verification", {}).get("inliers"),
                        "Box area": round(detection.get("box_area_ratio", 0), 3),
                        "Box": [round(value, 1) for value in detection["box"]],
                    }
                    for detection in analysis["detections"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No detections were returned at this confidence threshold.")
    snapshot = analysis.get("snapshot")
    if snapshot and snapshot["comparison"] and is_owner:
        with st.expander("Test result comparison", expanded=True):
            st.dataframe(
                [
                    {
                        "SKU": row["sku"],
                        "Product": row["name"],
                        "Expected": row["expected"],
                        "Predicted": row["predicted"],
                        "Error": row["error"],
                        "Correct": row["correct"],
                    }
                    for row in snapshot["comparison"]
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f'Saved run: {snapshot["run_id"]}')
    scope = st.radio("Count scope", ["Partial: selected SKUs", "Complete: all registered SKUs"])
    complete = scope.startswith("Complete")
    st.caption("Approved quantities replace selected SKU totals at this location. Other locations remain unchanged.")
    included = st.multiselect(
        "SKUs included in this count", [product["sku"] for product in products],
        default=[sku for sku in counts if counts[sku] > 0],
        key=f'include_{analysis["submission_key"]}',
    ) if not complete else [product["sku"] for product in products]
    approved = {}
    for product in products:
        if product["sku"] not in included:
            continue
        approved[product["sku"]] = st.number_input(
            f'{product["name"]} ({product["sku"]})', min_value=0, max_value=2_147_483_647,
            value=int(counts.get(product["sku"], 0)), step=1,
            key=f'approved_{product["sku"]}'
        )

    notes = st.text_input("Count notes")
    confirmed = st.checkbox("I reviewed these quantities and the count scope", key=analysis["submission_key"])
    saved_id = analysis.get("saved_id")
    if saved_id:
        st.success(f"Count #{saved_id} recorded successfully.")
    if st.button("Approve and save count", type="primary", disabled=not confirmed or not approved or bool(saved_id)):
        try:
            session_id = service.approve(
                location_id, photo_hash, analysis["submission_key"], approved, dict(counts),
                "complete" if complete else "partial", notes,
            )
            analysis["saved_id"] = session_id
            st.rerun()
        except (ValueError, PermissionError) as error:
            st.error(str(error))

    if is_owner and approved:
        with st.expander("Reviewed count CSV"):
            existing_csv = st.file_uploader(
                "Existing UTF-8 CSV template", type=["csv"], key="template_csv"
            )
            try:
                selected_products = [product for product in products if product["sku"] in approved]
                csv_bytes = make_csv(selected_products, approved, existing_csv)
                st.download_button(
                    "Download reviewed count CSV", data=csv_bytes,
                    file_name="reviewed_count.csv", mime="text/csv",
                    icon=":material/download:",
                )
            except (ValueError, csv.Error) as error:
                st.error(str(error))

if not is_owner:
    st.stop()
st.subheader("Recent test runs")
recent_runs = read_test_runs(user["business_id"], limit=8)
if recent_runs:
    st.dataframe(
        [
            {
                "Time": run["created_at"],
                "Test": run["test_name"],
                "Confidence": run["confidence"],
                "Expected": run["expected_counts"],
                "Predicted": run["counts"],
                "Annotated": run["run_annotated_image"],
            }
            for run in reversed(recent_runs)
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Full log: {TEST_RUNS_FILE}")
else:
    st.info("No tracked test runs yet.")
