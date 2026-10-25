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
from judge_release import user_error
from judge_release import judge_mode,shareable_evidence
from PIL import Image

from experiments.opencv5_guided_scan.capture_gate import capture_decision, quality
from live_scan import normalize_photo, render_items, request_dino, replay_dino, review_counts, reference_fingerprint, COORDINATES, SIGLIP_REVISION
import db


@st.cache_resource(show_spinner='Preparing shop references…')
def _shop_matcher(gallery_json, references_sha256):
    from gallery_8509 import CachedShopMatcher
    return CachedShopMatcher(json.loads(gallery_json))


def shop_matcher(catalog_json,references_sha256):
    # Prices, stock and shelf edits do not change recognition reference vectors.
    products=json.loads(catalog_json)
    gallery=[{k:p[k] for k in ('business_id','sku','name','image_path','references') if k in p} for p in products]
    return _shop_matcher(json.dumps(gallery,sort_keys=True),references_sha256)


@st.cache_data(max_entries=2,show_spinner=False)
def prepared_photo(data):
    image,normalized=normalize_photo(data)
    metrics=quality(np.asarray(image)[:,:,::-1].copy())
    return image,normalized,metrics


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
        annotated.save(image_buffer, format='PNG',compress_level=1)
        _atomic_write(folder / 'analysis.png', image_buffer.getvalue())
    serialized = json.dumps(evidence, indent=2, ensure_ascii=False, default=_json_default).encode('utf-8')
    _atomic_write(folder / 'analysis.json', serialized)


def saved_review(user, source_hash, upright_hash, allowed_skus):
    folder=db.DATA_DIR/'fresh_scans'/str(user['business_id'])
    if not folder.exists():return None
    for path in sorted(folder.glob('*/analysis.json'),key=lambda p:p.stat().st_mtime,reverse=True):
        try:
            if path.stat().st_size>10*1024*1024:continue
            record=json.loads(path.read_text(encoding='utf-8'))
            if (record.get('business_id')==user['business_id'] and record.get('user',{}).get('id')==user['id']
                    and record.get('source_photo',{}).get('sha256')==source_hash
                    and record.get('upright_photo',{}).get('sha256')==upright_hash
                    and record.get('draft',{}).get('rows')
                    and all(not r.get('sku') or r['sku'] in allowed_skus for r in record['draft']['rows'])
                    and record.get('detector') is not None and record.get('sku_suggestions') is not None):
                return record
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


def fresh_scan(service, buyer=False):
    user, products, locations = service.user(), service.products(), service.locations()
    st.header('Scan and review products')
    st.caption('New photo → capture check → product boxes → shop SKU suggestions → your review.')
    if not products or not locations:
        st.info('The owner must register products and a shelf before scanning.')
        return
    pending_identity = [p['sku'] for p in products if 'pending identity' in (p.get('registration_status') or '')]
    if pending_identity:
        st.warning('Package identity is pending for: ' + ', '.join(pending_identity) +
                   '. Suggestions do not verify the actual variant or count unit; check the package before confirming.')
    with st.expander('Prepare scanner before recording'):
        st.caption('Loads existing local reference vectors and the model once. Every LIVE scan still runs fresh DINO inference.')
        if st.button('Prepare live scanner',key='prepare_live_scanner'):
            try:
                from gallery_8509 import gallery_fingerprint
                started=time.perf_counter()
                shop_matcher(json.dumps(products,sort_keys=True),gallery_fingerprint(products)).prepare()
                st.success(f'Scanner prepared in {time.perf_counter()-started:.1f}s. Keep this app running for your recording.')
            except Exception as error:st.error(f'Scanner preparation failed: {user_error(error)}')
    location = st.selectbox('Shelf / count location', [r['id'] for r in locations],
        format_func=lambda key: next(r['name'] for r in locations if r['id'] == key), key='fresh_location')
    source_options=['Upload photo','Camera'] if judge_mode() else ['Upload photo','Camera','Archived photo · REPLAY MODE']
    source = st.radio('Photo source', source_options, horizontal=True, key='fresh_source')
    replay = source == 'Archived photo · REPLAY MODE'
    archive = Path(__file__).resolve().parent / 'data/webshop_demo/live_worker_smoke.json'
    if replay:
        if judge_mode():
            st.error('Private archived photos are unavailable in the judge sandbox.');return
        st.warning('REPLAY MODE: archived DINO boxes for this historical photo. SKU matching runs against the selected shop now. This is not fresh-photo detection or a benchmark.')
        try:
            recorded = json.loads(archive.read_text(encoding='utf-8'))
            data = Path(recorded['source']).read_bytes()
        except (OSError, KeyError, ValueError) as error:
            st.error(f'Local replay evidence unavailable: {user_error(error)}')
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
        image, normalized, metrics = prepared_photo(data)
    except (ValueError, OSError) as error:
        st.error(user_error(error))
        return
    catalog_json = json.dumps(products, sort_keys=True)
    try:
        from gallery_8509 import gallery_fingerprint
        references_sha256 = gallery_fingerprint(products)
    except (OSError,ValueError):
        st.error('A registered reference photo is unavailable. Ask the owner to replace it before scanning.')
        return
    model_signature = (os.environ.get('WEBSHOP_DINO_SHA256', ''), os.environ.get('WEBSHOP_DINO_URL', ''),
                       os.environ.get('WEBSHOP_SIGLIP_PATH', ''), SIGLIP_REVISION)
    signature = hashlib.sha256(data + catalog_json.encode() +
        str((user['id'], location, references_sha256, model_signature, replay, buyer)).encode()).hexdigest()
    if st.session_state.get('fresh_signature') != signature:
        for key in ['fresh_capture', 'fresh_result', 'fresh_approval', 'fresh_saved', 'fresh_receipts', 'fresh_capture_error']:
            st.session_state.pop(key, None)
        st.session_state['fresh_signature'] = signature
    if st.session_state.get('fresh_result'):
        with st.expander('Original input photo'):
            st.image(image, caption='Upright source photograph', width=480)
    else:
        st.image(image, caption='Upright source photograph', width=480)
    if cv2.__version__.split('.')[0] != '5':
        st.error('This scan requires the project’s OpenCV 5 environment.')
        return
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    mode = st.radio('Capture check', ['AWS capture service', 'On this computer'], horizontal=True, key='fresh_mode')
    st.caption('AWS uploads this photo only when you select AWS and click Check this photo. LOCAL is an explicit fallback.')
    if st.button('Check this photo', type='primary', key='fresh_check'):
        st.session_state.pop('fresh_capture', None)
        st.session_state.pop('fresh_capture_error', None)
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
            st.session_state['fresh_capture_error'] = dict(provider='UNAVAILABLE', error=user_error(error),
                image_sha256=hashlib.sha256(data).hexdigest(), elapsed_seconds=round(time.perf_counter()-check_started,3))
            st.error(f'Capture check failed: {user_error(error)}. Retry, or explicitly select the local check.')
    capture = st.session_state.get('fresh_capture')
    if not capture or capture['mode'] != mode:
        if st.session_state.get('fresh_capture_error'):
            from quality_8509 import diagnostics
            diagnostics(st.session_state['fresh_capture_error'], 'Capture service unavailable')
        st.info('Check this photograph before running recognition.')
        return
    st.caption(capture['source'])
    decision = capture['decision']
    from quality_8509 import diagnostics
    raw = capture['evidence']
    cloud = capture['source'] == 'AWS OpenCV 5'
    diagnostic = dict(provider='AWS' if cloud else 'LOCAL', cv2_version=raw.get('runtime',{}).get('cv2_version',cv2.__version__) if cloud else cv2.__version__, image_sha256=hashlib.sha256(data).hexdigest(), metrics=raw.get('quality',raw), decision=capture['decision'], elapsed_seconds=capture['elapsed_seconds'])
    with st.expander('Scan diagnostics', expanded=not bool(st.session_state.get('fresh_result'))):
        diagnostics(diagnostic, 'OpenCV capture guidance')
        if replay: st.warning('Detector provider: REPLAY ? archived DINO boxes. Capture analysis above is separate.')
    if decision['action'] != 'proceed':
        st.warning(decision['reason'] + '. Please take another photo.')
        from quality_8509 import accepts
        override = st.text_input('Exposure-only override reason (blur/low resolution cannot be overridden)', key='scan_override_reason')
        authorized = st.checkbox('I reviewed the exposure and accept a manual override', key='scan_override_confirm')
        if not (authorized and accepts(diagnostic, override)):
            return
        capture['override'] = {'reason': override, 'user_id': user['id'], 'at_utc': datetime.now(timezone.utc).isoformat()}
    st.success(decision['reason'])
    if not buyer and not replay and not st.session_state.get('fresh_result'):
        saved=saved_review(user,hashlib.sha256(data).hexdigest(),hashlib.sha256(normalized).hexdigest(),{p['sku'] for p in products})
        if saved:
            st.info('A saved review exists for this exact photograph. Resume it to keep your corrections and earlier inference results.')
            if st.button('Resume saved review for this photo',key='resume_saved_review'):
                timings=saved.get('timings_seconds',{})
                st.session_state['fresh_result']=dict(detector=saved['detector'],items=saved['sku_suggestions'],
                    elapsed_seconds=timings.get('recognition_total',0),preprocessing_seconds=timings.get('preprocessing',0),
                    detector_request_seconds=timings.get('detector_request',0),matching_seconds=timings.get('matching',0),
                    source_sha256=hashlib.sha256(data).hexdigest(),coordinate_space=saved['coordinate_space'],
                    catalog_sha256=saved['catalog_sha256'],mode='RESUMED',original_inference_mode=saved.get('original_inference_mode',saved['mode']),
                    evidence_id=saved['scan_id'],created_at_utc=saved['created_at_utc'],box_quality=saved.get('box_quality'))
                st.session_state['fresh_approval']=saved.get('submission_key') or 'resume-'+saved['scan_id']
                if saved.get('inventory_receipt_id'):
                    st.session_state['fresh_saved']=saved['inventory_receipt_id']
                    st.session_state['fresh_receipts']=saved.get('inventory_receipt_ids') or [saved['inventory_receipt_id']]
                st.rerun()
    connected = bool(os.environ.get('WEBSHOP_DINO_URL') and os.environ.get('WEBSHOP_DINO_TOKEN'))
    if not connected and not replay:
        st.warning('Live product detection is not connected yet. This photo has no model predictions.')
    if st.button('Find products in this photo', disabled=not (connected or replay), key='fresh_detect'):
        st.session_state.pop('fresh_result', None)
        st.session_state.pop('fresh_saved', None)
        st.session_state.pop('fresh_receipts', None)
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
            from quality_8509 import box_quality
            st.session_state['fresh_result']['box_quality'] = box_quality(image, detector['detections'])
            st.session_state['fresh_approval'] = uuid4().hex
        except Exception as error:
            st.error(f'Product recognition failed: {user_error(error)}')
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
        'original_inference_mode':result.get('original_inference_mode',result['mode']),
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
        st.error(f'This scan was analyzed, but local evidence could not be saved: {user_error(error)}')
    else:
        st.caption(f"Photo, analysis and review saved · scan {result['evidence_id'][:8]}")
    from review_8509 import review_panel
    from grouped_scan_ui import overlay_rows
    from scan_review import new_draft, grouped, reviewed_counts
    approval_id = st.session_state['fresh_approval']
    evidence['submission_key']=approval_id
    draft_key = f'fresh_draft_{approval_id}'
    if draft_key not in st.session_state:
        saved_evidence = json.loads((archive_folder / 'analysis.json').read_text(encoding='utf-8')) if (archive_folder / 'analysis.json').exists() else {}
        st.session_state[draft_key] = saved_evidence.get('draft') or new_draft(result['items'])
    draft = st.session_state[draft_key]
    locked = bool(st.session_state.get('fresh_saved'))
    st.caption(f"{result.get('mode', 'LIVE')} · {result['elapsed_seconds']} seconds · {len(result['items'])} model proposals")
    if not buyer:
        st.markdown('[Go to Excel preview and approval](#scan-approval)')
    st.subheader('2. Review detected products')
    review_panel(image, products, draft, str(user.get('username', user['id'])), approval_id, locked=locked)
    if result.get('box_quality'):
        with st.expander('OpenCV box guidance', expanded=False):
            for guidance in result['box_quality']['guidance']: st.info(guidance)
            st.caption(result['box_quality']['limitation'])
            flagged = [r for r in result['box_quality']['boxes'] if r['near_frame'] or r['overlaps']]
            if flagged: st.dataframe(flagged, hide_index=True, width='stretch')
    evidence['box_quality'] = result.get('box_quality')
    evidence['capture_diagnostics'] = diagnostic
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
        review_error = user_error(error)
    def persist():
        evidence['inventory_receipt_id'] = st.session_state.get('fresh_saved')
        evidence['inventory_receipt_ids'] = st.session_state.get('fresh_receipts',[])
        evidence['stock_actions'] = [e for e in draft['events'] if e['action'] == 'inventory_action']
        try:
            _save_scan_evidence(archive_folder, data, normalized, render_items(image, overlay_rows(draft)), evidence)
        except OSError as error:
            st.error(f'Could not update local scan evidence: {user_error(error)}')
    persist()
    st.download_button('Download scan evidence', json.dumps(shareable_evidence(evidence), indent=2, default=_json_default),
                       file_name=f'webshop_scan_{result["evidence_id"]}.json', mime='application/json')
    if locked:
        receipt_labels=', '.join(str(r) for r in st.session_state.get('fresh_receipts',[st.session_state['fresh_saved']]))
        st.success(f"Saved inventory receipt(s) #{receipt_labels}. This draft is locked.")
        saved_action=next((e for e in reversed(draft['events']) if e['action']=='inventory_action'),{})
        st.dataframe(saved_action.get('preview',[]),hide_index=True,width='stretch')
        from catalog_workbook import export_scan_workbook
        st.download_button('Download scan review Excel',export_scan_workbook(service,draft,saved_action.get('preview',[]),saved_action.get('operation','review')),
            file_name='webshop_saved_scan.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key=f'saved_excel_{approval_id}')
        return
    if buyer:
        if review_error:
            st.info('Confirm or exclude every item before action confirmation. ' + review_error)
            return
        st.success('Reviewed cart preview. Customer actions never change stock.')
        st.download_button('Download reviewed cart', json.dumps(approved), 'cart_preview.json')
        return
    st.markdown('<div id="scan-approval"></div>', unsafe_allow_html=True)
    st.subheader('3. Excel preview & stock approval')
    operation = st.radio('What should happen next?', ['review', 'audit', 'add', 'subtract'],
        format_func=lambda v: {'review':'Review only (no stock changes)',
            'audit':'Shelf audit (set stock)', 'add':'Receiving / ADD', 'subtract':'Removal / SUBTRACT'}[v],
        key=f'action_{approval_id}', horizontal=True)
    assigned_possible=bool(counts) and all(p.get('shelf_id') in {l['name'] for l in locations} for p in products if p['sku'] in counts)
    use_assigned=False
    if operation in ('add','subtract'):
        use_assigned=st.checkbox("Use each product's assigned shelf",value=assigned_possible,key=f'assigned_shelves_{approval_id}')
        if use_assigned:st.caption('Mixed-shelf photo: each SKU uses its own catalogue shelf. All shelf changes save together after approval.')
    zero_counts={sku:0 for sku in counts if sku}
    stock_error=None
    try:
        stock_rows=(service.preview_catalog_action(zero_counts,'add') if use_assigned and zero_counts else service.preview_action(location,zero_counts,'add'))
    except (ValueError,PermissionError) as error:
        stock_rows=[];stock_error=user_error(error)
    stock = {r['sku']:r['before'] for r in stock_rows}
    stock_shelves={r['sku']:r.get('shelf_id',next(l['name'] for l in locations if l['id']==location)) for r in stock_rows}
    st.dataframe([{'SKU':sku or 'Unknown',
                   'Product':next((p['name'] for p in products if p['sku']==sku),'Assign a SKU above'),
                   'Scanned units':group['quantity'],'Stock shelf':stock_shelves.get(sku),'Current stock':stock.get(sku),
                   'Review status':'Reviewed' if group['reviewed'] else 'Needs review'}
                  for sku,group in grouped(draft).items()], hide_index=True, width='stretch')
    from catalog_workbook import export_scan_workbook
    st.download_button('Download scan review Excel',export_scan_workbook(service,draft,[],operation),
        file_name='webshop_scan_review.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key=f'scan_excel_{approval_id}')
    st.caption('Excel is a preview. Stock changes only after you approve an inventory action below.')
    def blocked_approval(message):
        st.warning(message)
        st.button('Approve inventory action', type='primary',disabled=True,key=f'blocked_approval_{approval_id}')
    if stock_error:
        blocked_approval(stock_error)
        return
    if review_error:
        blocked_approval('Approval is locked: assign a SKU to each unknown item, then mark every group reviewed or remove false detections. '+review_error)
        return
    if operation == 'review':
        if st.button('Save reviewed analysis', type='primary'):
            from scan_review import event
            event(draft, 'review_only', user['id'])
            persist()
            st.success('Reviewed analysis saved. Stock was not changed.')
        blocked_approval('Choose Receiving / ADD, Removal / SUBTRACT or Shelf audit to update stock.')
        return
    action_counts = dict(approved)
    if operation == 'audit':
        scope = st.multiselect('Explicit audit scope (selected absent products will be set to zero)',
                               [p['sku'] for p in products], key=f'audit_scope_{approval_id}')
        if not scope:
            blocked_approval('Choose the exact SKU scope for this shelf audit.')
            return
        action_counts = {sku: approved.get(sku, 0) for sku in scope}
    if not action_counts:
        blocked_approval('Choose at least one reviewed product for a stock action.')
        return
    notes = st.text_input('Transaction notes / prior receipt for a correction', key=f'fresh_notes_{approval_id}')
    try:
        preview = service.preview_catalog_action(action_counts,operation) if use_assigned else service.preview_action(location, action_counts, operation)
    except (ValueError, PermissionError) as error:
        blocked_approval(user_error(error))
        return
    st.markdown('**Stock before → after approval**')
    st.dataframe(preview, hide_index=True, width='stretch')
    st.download_button('Download stock change Excel',export_scan_workbook(service,draft,preview,operation),
        file_name='webshop_stock_preview.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',key=f'stock_excel_{approval_id}')
    evidence['stock_action_preview'] = {'operation': operation, 'counts': action_counts, 'preview': preview, 'notes': notes,'assigned_shelves':use_assigned}
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
            action_notes=f"{result.get('mode', 'LIVE')} DINO/SigLIP2 reviewed scan. Evidence {result['evidence_id']}. {notes}"
            if use_assigned:
                receipt_ids=service.approve_catalog_action(result['source_sha256'],approval_id,action_counts,proposed,
                    action_notes,operation,expected_preview=preview)
                saved=receipt_ids[0]
            else:
                saved=service.approve(location, result['source_sha256'], approval_id, action_counts, proposed,
                    'partial', action_notes, operation=operation)
                receipt_ids=[saved]
            st.session_state['fresh_saved'] = saved
            st.session_state['fresh_receipts'] = receipt_ids
            from scan_review import event
            event(draft, 'inventory_action', user['id'], receipt_id=saved, operation=operation,
                  counts=action_counts, preview=preview, notes=notes,receipt_ids=receipt_ids)
            persist()
            st.rerun()
        except (ValueError, PermissionError) as error:
            st.error(user_error(error))
