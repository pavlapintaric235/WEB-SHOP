"""Actual, explicitly requested owner verification; no stock writes."""
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import streamlit as st
from judge_release import user_error

import db
from images import product_references
from live_scan import normalize_photo, request_dino, render_items
from quality_8509 import analyze, aws_analyze, diagnostics, box_quality


def test_recognition(service, product):
    service.user(owner=True)
    st.subheader('Test Recognition')
    st.caption('Use a distinct new photo, not any reference image. This test never changes stock.')
    upload=st.file_uploader('New test photo',type=['jpg','jpeg','png'],key=f'test_photo_{product["id"]}')
    if not upload:return
    data=upload.getvalue();image,normalized=normalize_photo(data)
    source_hash=hashlib.sha256(data).hexdigest()
    query_hash=hashlib.sha256(normalized).hexdigest()
    for reference in product_references(product):
        reference_data=Path(reference['image_path']).read_bytes()
        if hashlib.sha256(reference_data).hexdigest()==source_hash or hashlib.sha256(normalize_photo(reference_data)[1]).hexdigest()==query_hash:
            st.error('This is a registered reference photograph. Use a distinct test photo.');return
    key=f'owner_test_{product["id"]}_{source_hash}'
    provider=st.radio('Test photo quality provider',['AWS','LOCAL'],horizontal=True,key=key+'_provider')
    if st.button('Check test photo quality',key=key+'_check'):
        try:st.session_state[key+'_quality']=aws_analyze(data) if provider=='AWS' else analyze(data)
        except Exception as error:st.error(f'AWS unavailable. Retry or explicitly select LOCAL: {user_error(error)}')
    quality=st.session_state.get(key+'_quality')
    if not quality or quality['provider']!=provider:return
    diagnostics(quality)
    if quality['decision']['action']!='proceed':st.warning('Retake this test image before inference.');return
    if st.button('Run LIVE DINO + SigLIP2 test',type='primary',key=key+'_run'):
        try:
            started=time.perf_counter()
            with st.spinner('Running the existing RunPod detector and current-shop gallery…'):
                detector=request_dino(normalized,image.size)
                from scanner_8509 import shop_matcher
                from gallery_8509 import gallery_fingerprint
                products=service.products()
                matcher=shop_matcher(json.dumps(products,sort_keys=True),gallery_fingerprint(products))
                items=matcher.rank_boxes(image,detector['detections'])
            result=dict(mode='LIVE',kind='actual_owner_recognition_test',expected_sku=product['sku'],source_sha256=source_hash,
                detector=detector,items=items,quality=quality,box_quality=box_quality(image,detector['detections']),
                elapsed_seconds=round(time.perf_counter()-started,3),found=any(r['sku']==product['sku'] for r in items),
                candidate_in_top3=any(product['sku'] in [r['sku'] for r in item['ranked_scores'][:3]] for item in items))
            folder=db.DATA_DIR/'owner_recognition_tests'/str(service.user()['business_id'])/uuid4().hex
            folder.mkdir(parents=True);(folder/'original_photo').write_bytes(data)
            (folder/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            render_items(image,items).save(folder/'analysis.png')
            st.session_state[key+'_result']=result
        except Exception as error:st.error(f'Registration is separate from this test. LIVE recognition test pending/failed: {user_error(error)}')
    result=st.session_state.get(key+'_result')
    if result:
        st.image(render_items(image,result['items']),width='stretch')
        st.metric('LIVE processing time',f"{result['elapsed_seconds']:.2f}s")
        st.success('Selected SKU suggested by actual inference.' if result['found'] else 'Selected SKU was not assigned; inspect alternatives and unknowns.')
        st.dataframe([dict(box=i['item_id'],best=i['sku'] or 'Unknown',alternatives=', '.join(f"{r['sku']} ({r['similarity']:.3f})" for r in i['ranked_scores'][:3])) for i in result['items']],hide_index=True,width='stretch')
        st.caption('One test is evidence for this query, not a reliability estimate.')
