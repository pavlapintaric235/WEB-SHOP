"""Fresh-photo scanning through the shared authenticated inventory service."""
import hashlib
import io
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from experiments.opencv5_guided_scan.capture_gate import capture_decision, quality
from live_scan import ShopMatcher, normalize_photo, render_items, request_dino, replay_dino, review_counts, reference_fingerprint, COORDINATES, SIGLIP_REVISION
import db


@st.cache_resource(show_spinner='Preparing shop references…')
def shop_matcher(catalog_json, references_sha256):
    return ShopMatcher(json.loads(catalog_json))


def _json_default(value):
    if isinstance(value, (np.integer, np.bool_)):
        if isinstance(value, np.bool_):
            return bool(value)
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot save {type(value).__name__} in scan evidence.')


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.saving-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _save_scan_evidence(folder, original, normalized, annotated, evidence):
    """Keep each analyzed photo and its evolving review beside the local database."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, contents in ((evidence['original_filename'], original), ('upright.png', normalized)):
        if not (folder / name).exists():
            _atomic_write(folder / name, contents)
    previous = {}
    if (folder / 'analysis.json').exists():
        previous = json.loads((folder / 'analysis.json').read_text(encoding='utf-8'))
    evidence['annotation_revision'] = evidence.get('draft', {}).get('revision', -1)
    if not (folder / 'analysis.png').exists() or previous.get('annotation_revision') != evidence['annotation_revision']:
        image_buffer = io.BytesIO()
        annotated.save(image_buffer, format='PNG')
        _atomic_write(folder / 'analysis.png', image_buffer.getvalue())
    serialized = json.dumps(evidence, indent=2, ensure_ascii=False, default=_json_default).encode('utf-8')
    _atomic_write(folder / 'analysis.json', serialized)


def fresh_scan(service, buyer=False):
    user, products, locations = service.user(), service.products(), service.locations()
    st.header('Scan and review products')
    st.caption('New photo → capture check → product boxes → shop SKU suggestions → your review.')
    if not products or not locations:
        st.info('The owner must register products and a shelf before scanning.')
        return
    pending_identity = [p['sku'] for p in products if 'pending identity' in (p.get('registration_status') or '')
                        or p['sku'] == 'MAKS-CRISPY-50']
    if pending_identity:
        st.warning('Package identity is pending for: ' + ', '.join(pending_identity) +
                   '. Suggestions do not verify the actual variant or count unit; check the package before confirming.')
    location = st.selectbox('Shelf / count location', [r['id'] for r in locations],
        format_func=lambda key: next(r['name'] for r in locations if r['id'] == key), key='fresh_location')
    source = st.radio('Photo source', ['Upload photo', 'Camera', 'Archived photo · REPLAY MODE'], horizontal=True, key='fresh_source')
    replay = source == 'Archived photo · REPLAY MODE'
    archive = Path(__file__).resolve().parent / 'data/webshop_demo/live_worker_smoke.json'
    if replay:
        st.warning('REPLAY MODE: archived DINO boxes for this historical photo. SKU matching runs against the selected shop now. This is not fresh-photo detection or a benchmark.')
        try:
            recorded = json.loads(archive.read_text(encoding='utf-8'))
            data = Path(recorded['source']).read_bytes()
        except (OSError, KeyError, ValueError) as error:
            st.error(f'Local replay evidence unavailable: {error}')
            return
    else:
        upload = (st.camera_input('Take a shelf photo', key='fresh_camera') if source == 'Camera'
                  else st.file_uploader('Shelf photograph', type=['jpg', 'jpeg', 'png'], key='fresh_upload'))
        if upload is None:
            st.info('Take a new photograph of your registered products to begin.')
            return
        data = upload.getvalue()
    source_name = getattr(upload, 'name', '') if not replay else 'archived-replay.jpg'
    preprocessing_started = time.perf_counter()
    try:
        image, normalized = normalize_photo(data)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return
    catalog_json = json.dumps(products, sort_keys=True)
    try:
        references_sha256 = reference_fingerprint(products)
    except OSError:
        st.error('A registered reference photo is unavailable. Ask the owner to replace it before scanning.')
        return
    model_signature = (os.environ.get('WEBSHOP_DINO_SHA256', ''), os.environ.get('WEBSHOP_DINO_URL', ''),
                       os.environ.get('WEBSHOP_SIGLIP_PATH', ''), SIGLIP_REVISION)
    signature = hashlib.sha256(data + catalog_json.encode() +
        str((user['id'], location, references_sha256, model_signature, replay, buyer)).encode()).hexdigest()
    if st.session_state.get('fresh_signature') != signature:
        for key in ['fresh_capture', 'fresh_result', 'fresh_approval', 'fresh_saved']:
            st.session_state.pop(key, None)
        st.session_state['fresh_signature'] = signature
    st.image(image, caption='Upright source photograph', width='stretch')
    if cv2.__version__.split('.')[0] != '5':
        st.error('This scan requires the project’s OpenCV 5 environment.')
        return
    metrics = quality(np.asarray(image)[:, :, ::-1].copy())
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    mode = st.radio('Capture check', ['On this computer', 'AWS capture service'], horizontal=True, key='fresh_mode')
    if st.button('Check this photo', type='primary', key='fresh_check'):
        st.session_state.pop('fresh_capture', None)
        st.session_state.pop('fresh_result', None)
        try:
            check_started = time.perf_counter()
            if mode == 'AWS capture service':
                from experiments.opencv5_guided_scan.aws_capture_client import run_capture
                with st.spinner('Checking with the existing AWS service…'):
                    evidence = run_capture(data, hashlib.sha256(data).hexdigest())
                if [evidence['quality']['width'], evidence['quality']['height']] != list(image.size):
                    raise ValueError('AWS and local image orientation differ; resolve this before detection.')
                decision, label = evidence['capture_decision'], 'AWS OpenCV 5'
            else:
                evidence, decision, label = metrics, capture_decision(metrics), 'Local OpenCV 5'
            codes = {'Image could not be decoded': 'DECODE_FAILED',
                     'Move closer or capture at higher resolution': 'LOW_RESOLUTION',
                     'Use more light': 'UNDEREXPOSED', 'Reduce glare': 'OVEREXPOSED',
                     'Hold the camera steadier': 'LOW_FOCUS',
                     'Capture checks passed; inspect every item': 'PROCEED_REVIEW'}
            st.session_state['fresh_capture'] = dict(decision=decision, evidence=evidence, source=label, mode=mode,
                reason_code=codes.get(decision['reason'], 'CAPTURE_REVIEW'),
                elapsed_seconds=round(time.perf_counter() - check_started, 3))
        except Exception as error:
            st.error(f'Capture check failed: {error}. Retry, or explicitly select the local check.')
    capture = st.session_state.get('fresh_capture')
    if not capture or capture['mode'] != mode:
        st.info('Check this photograph before running recognition.')
        return
    st.caption(capture['source'])
    decision = capture['decision']
    with st.expander('Capture metrics and provenance'):
        st.json(capture)
    if decision['action'] != 'proceed':
        st.warning(decision['reason'] + '. Please take another photo.')
        return
    st.success(decision['reason'])
    connected = bool(os.environ.get('WEBSHOP_DINO_URL') and os.environ.get('WEBSHOP_DINO_TOKEN'))
    if not connected and not replay:
        st.warning('Live product detection is not connected yet. This photo has no model predictions.')
    if st.button('Find products in this photo', disabled=not (connected or replay), key='fresh_detect'):
        st.session_state.pop('fresh_result', None)
        st.session_state.pop('fresh_saved', None)
        try:
            started = time.perf_counter()
            with st.spinner('Finding products and comparing shop references…'):
                detector = replay_dino(archive, normalized, image.size) if replay else request_dino(normalized, image.size)
                detection_seconds = time.perf_counter() - started
                items = shop_matcher(catalog_json, references_sha256).rank_boxes(image, detector['detections'])
            st.session_state['fresh_result'] = dict(detector=detector, items=items,
                elapsed_seconds=round(time.perf_counter() - started, 2),
                preprocessing_seconds=round(preprocessing_seconds, 3),
                detector_request_seconds=round(detection_seconds, 3),
                matching_seconds=round(time.perf_counter() - started - detection_seconds, 3),
                source_sha256=hashlib.sha256(data).hexdigest(), coordinate_space=COORDINATES,
                catalog_sha256=hashlib.sha256(catalog_json.encode()).hexdigest(), mode='REPLAY' if replay else 'LIVE',
                evidence_id=uuid4().hex,
                created_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            st.session_state['fresh_approval'] = uuid4().hex
        except Exception as error:
            st.error(f'Product recognition failed: {error}')
    result = st.session_state.get('fresh_result')
    if result is None:
        return
    result.setdefault('evidence_id', uuid4().hex)
    result.setdefault('created_at_utc', datetime.now(timezone.utc).isoformat(timespec='seconds'))
    extension = Path(source_name).suffix.lower()
    if extension not in {'.jpg', '.jpeg', '.png'}:
        with Image.open(io.BytesIO(data)) as original:
            extension = '.png' if original.format == 'PNG' else '.jpg'
    archive_folder = db.DATA_DIR / 'fresh_scans' / str(user['business_id']) / result['evidence_id']
    archive_relative = archive_folder.relative_to(db.DATA_DIR).as_posix()
    original_filename = 'original' + extension
    evidence = {
        'schema_version': 1,
        'scan_id': result['evidence_id'],
        'created_at_utc': result['created_at_utc'],
        'business_id': user['business_id'],
        'user': {'id': user['id'], 'username': user.get('username', 'buyer'), 'role': user['role']},
        'location': {'id': location, 'name': next(r['name'] for r in locations if r['id'] == location)},
        'mode': result['mode'],
        'original_filename': original_filename,
        'source_photo': {'file': original_filename, 'sha256': result['source_sha256']},
        'upright_photo': {'file': 'upright.png', 'sha256': hashlib.sha256(normalized).hexdigest(),
                          'size': list(image.size)},
        'analysis_image': 'analysis.png',
        'coordinate_space': result['coordinate_space'],
        'catalog_sha256': result['catalog_sha256'],
        'siglip_revision': SIGLIP_REVISION,
        'capture_check': capture,
        'detector': result['detector'],
        'sku_suggestions': result['items'],
        'timings_seconds': {
            'preprocessing': result['preprocessing_seconds'],
            'detector_request': result['detector_request_seconds'],
            'matching': result['matching_seconds'],
            'recognition_total': result['elapsed_seconds'],
        },
        'products': [{'sku': p['sku'], 'name': p['name'], 'unit': p.get('unit'),
                      'brand_variant_size': p.get('brand_variant_size'),
                      'registration_status': p.get('registration_status')} for p in products],
        'reviewed_items': None,
        'reviewed_counts': None,
        'inventory_receipt_id': st.session_state.get('fresh_saved'),
    }
    initial_overlay = [dict(box=item['box'], sku=item.get('sku'), item_id=item.get('item_id'))
                       for item in result['items']]
    try:
        if not (archive_folder / 'analysis.json').exists():
            _save_scan_evidence(archive_folder, data, normalized, render_items(image, initial_overlay), evidence)
        st.session_state['fresh_archive_relative'] = archive_relative
    except OSError as error:
        st.error(f'This scan was analyzed, but local evidence could not be saved: {error}')
    else:
        st.caption(f"Scan evidence saved locally in `{archive_relative}` (original photo, upright photo, annotated analysis and JSON).")
    from grouped_scan_ui import review_panel, overlay_rows
    from scan_review import new_draft, grouped, reviewed_counts
    approval_id = st.session_state['fresh_approval']
    draft_key = f'fresh_draft_{approval_id}'
    if draft_key not in st.session_state:
        saved_evidence = json.loads((archive_folder / 'analysis.json').read_text(encoding='utf-8')) if (archive_folder / 'analysis.json').exists() else {}
        st.session_state[draft_key] = saved_evidence.get('draft') or new_draft(result['items'])
    draft = st.session_state[draft_key]
    locked = bool(st.session_state.get('fresh_saved'))
    st.caption(f"{result.get('mode', 'LIVE')} ? {result['elapsed_seconds']} seconds ? {len(result['items'])} model proposals")
    review_panel(image, products, draft, str(user.get('username', user['id'])), approval_id, locked=locked)
    evidence['draft'] = draft
    evidence['reviewed_items'] = draft['rows']
    evidence['correction_provenance'] = draft['events']
    counts = {sku: group['quantity'] for sku, group in grouped(draft).items()}
    if buyer:
        from buyer_ui import show_price_estimate
        show_price_estimate(products, counts, f'cart_{approval_id}')
        if any(not group['reviewed'] for group in grouped(draft).values()):
            st.warning('Cart estimate includes unreviewed suggestions. Unknown or unpriced items are excluded from the priced subtotal.')
    if not buyer and not locked:
        with st.expander('Unknown product ? request owner registration'):
            description = st.text_input('Product description / package details', key=f'unknown_description_{approval_id}')
            if st.button('Request owner registration', key=f'unknown_request_{approval_id}', disabled=not description.strip()):
                service.request_registration(description, result['evidence_id'])
                from scan_review import event
                event(draft, 'registration_request', user['id'], description=description)
                st.success('Request sent to this shop owner. This does not add a unit or register a recognition reference.')
    approved = None
    review_error = None
    try:
        approved = reviewed_counts(draft, [p['sku'] for p in products], image.size, allow_empty=True)
        evidence['reviewed_counts'] = approved
    except (ValueError, TypeError, KeyError) as error:
        review_error = str(error)
    def persist():
        evidence['inventory_receipt_id'] = st.session_state.get('fresh_saved')
        evidence['stock_actions'] = [e for e in draft['events'] if e['action'] == 'inventory_action']
        try:
            _save_scan_evidence(archive_folder, data, normalized, render_items(image, overlay_rows(draft)), evidence)
        except OSError as error:
            st.error(f'Could not update local scan evidence: {error}')
    persist()
    st.download_button('Download scan evidence', json.dumps(evidence, indent=2, default=_json_default),
                       file_name=f'webshop_scan_{result["evidence_id"]}.json', mime='application/json')
    if locked:
        st.success(f"Saved inventory receipt #{st.session_state['fresh_saved']}. This draft is locked.")
        return
    if review_error:
        st.info('Confirm or exclude every item before action confirmation. ' + review_error)
        return
    if buyer:
        st.success('Reviewed cart preview. Customer actions never change stock.')
        st.download_button('Download reviewed cart', json.dumps(approved), 'cart_preview.json')
        return
    st.subheader('Confirmation')
    st.dataframe([{'SKU': sku, 'Product': next(p['name'] for p in products if p['sku'] == sku), 'Units': n}
                  for sku, n in approved.items()], hide_index=True, width='stretch')
    operation = st.radio('What should happen next?', ['review', 'audit', 'add', 'subtract'],
        format_func=lambda v: {'review':'Review only ? save analysis, no stock changes',
            'audit':'Shelf audit ? set counts for explicit scope', 'add':'Receiving / ADD', 'subtract':'Removal / SUBTRACT'}[v],
        key=f'action_{approval_id}')
    if operation == 'review':
        if st.button('Save reviewed analysis', type='primary'):
            from scan_review import event
            event(draft, 'review_only', user['id'])
            persist()
            st.success('Reviewed analysis saved. Stock was not changed.')
        return
    action_counts = dict(approved)
    if operation == 'audit':
        scope = st.multiselect('Explicit audit scope (selected absent products will be set to zero)',
                               [p['sku'] for p in products], key=f'audit_scope_{approval_id}')
        if not scope:
            st.info('Choose the exact SKU scope for this shelf audit.')
            return
        action_counts = {sku: approved.get(sku, 0) for sku in scope}
    if not action_counts:
        st.info('Choose at least one reviewed product for a stock action.')
        return
    notes = st.text_input('Transaction notes / prior receipt for a correction', key=f'fresh_notes_{approval_id}')
    try:
        preview = service.preview_action(location, action_counts, operation)
    except (ValueError, PermissionError) as error:
        st.error(str(error))
        return
    st.dataframe(preview, hide_index=True, width='stretch')
    evidence['stock_action_preview'] = {'operation': operation, 'counts': action_counts, 'preview': preview, 'notes': notes}
    persist()
    decision_hash = hashlib.sha256(json.dumps([operation, action_counts, notes, preview], sort_keys=True).encode()).hexdigest()[:12]
    confirmed = st.checkbox('I checked the scope and before/after stock; apply this inventory action', key=f'fresh_confirm_{approval_id}_{decision_hash}')
    if st.button('Approve inventory action', type='primary', disabled=not confirmed):
        if not confirmed:
            st.error('Confirm the current scope and before/after stock before applying this action.')
            return
        try:
            proposed = {}
            for item in result['items']:
                if item['sku']:
                    proposed[item['sku']] = proposed.get(item['sku'], 0) + 1
            saved = service.approve(location, result['source_sha256'], approval_id, action_counts, proposed,
                'partial', f"{result.get('mode', 'LIVE')} DINO/SigLIP2 reviewed scan. Evidence {result['evidence_id']}. {notes}", operation=operation)
            st.session_state['fresh_saved'] = saved
            from scan_review import event
            event(draft, 'inventory_action', user['id'], receipt_id=saved, operation=operation,
                  counts=action_counts, preview=preview, notes=notes)
            persist()
            st.rerun()
        except (ValueError, PermissionError) as error:
            st.error(str(error))
