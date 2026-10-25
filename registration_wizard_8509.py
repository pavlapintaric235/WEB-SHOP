"""Guided owner registration using real OpenCV decisions and cached SigLIP2."""
import hashlib
import io
import json
from pathlib import Path

import streamlit as st
from judge_release import user_error
from visual_8509 import registration_steps
from PIL import Image

from quality_8509 import analyze, aws_analyze, accepts, diagnostics
from registration_8509 import register, details_checked, assets_for
from catalog_workbook import workbook_panel


def collect_photos(state, uploads):
    for upload in uploads or []:
        data=upload.getvalue();digest=hashlib.sha256(data).hexdigest()
        if digest not in state['photos'] and digest not in state['ignored']:
            try:state['photos'][digest]=dict(data=data,name=upload.name,quality=analyze(data),angle='unknown',approved=False,override='')
            except (ValueError,OSError) as error:st.error(f'{upload.name}: {user_error(error)}')


def wizard(service, product=None):
    service.user(owner=True)
    key='wizard_'+str(product['id'] if product else 'new')
    state=st.session_state.setdefault(key,dict(step=0,details={},photos={},ignored=[]))
    if product and not state.get('assets_loaded'):
        for asset in assets_for(service,product['id']):
            if asset['active']: continue
            record=json.loads(asset['quality_json'])
            state['photos'][asset['source_sha256']]=dict(data=Path(asset['original_path']).read_bytes(),name=Path(asset['original_path']).name,
                quality=record['analysis'],angle=asset['angle'],approved=True,override=record.get('override',''))
        state['assets_loaded']=True
    st.subheader('+ Add Product' if not product else f"Add approved photos · {product['name']}")
    st.caption('Product details → 3–5 useful views → quality approval → visual registration')
    registration_steps(state['step'])
    st.progress(min(1,(state['step']+1)/4))
    if state.get('result'):
        r=state['result']
        st.success(f"{r['sku']} saved. " + ('References embedded and available to this shop’s retrieval gallery.' if r['ready'] else 'Incomplete draft — not in the active recognition gallery.'))
        st.info('Embedding generation is not proof of recognition. Open this product’s Test Recognition action with a distinct new photo.')
        workbook_panel(service, key+'_workbook')
        if st.button('Register another product',key=key+'_reset'):
            del st.session_state[key];st.rerun()
        return
    defaults=product or {}
    if state['step']==0:
        uploads=st.file_uploader('Reference photographs',type=['png','jpg','jpeg'],accept_multiple_files=True,key=key+'_upload')
        collect_photos(state, uploads)
        if state['photos']:
            st.image([p['data'] for p in state['photos'].values()],width=110)
        st.caption('Upload 3–5 views of a new product, enter its details below, then review image quality. Existing models are reused without retraining.')
        with st.form(key+'_details'):
            left,right=st.columns(2)
            with left:
                name=st.text_input('Product name',value=defaults.get('name',''))
                brand=st.text_input('Brand',value=defaults.get('brand') or '')
                variant=st.text_input('Variant / flavor',value=defaults.get('variant') or '')
                size=st.text_input('Package size',value=defaults.get('package_size') or '')
                units=['piece','bottle','package','can','carton','jar','bag']
                unit=st.selectbox('Quantity unit',units,index=units.index(defaults['unit']) if defaults.get('unit') in units else 0)
            with right:
                price=st.number_input('Price in EUR',min_value=0.0,value=float(defaults.get('price') or 0),step=.10)
                stock=st.number_input('Initial stock quantity',min_value=0,max_value=100000,value=int(defaults.get('opening_stock') or 0),disabled=bool(product))
                shelves=[r['name'] for r in service.locations()]
                shelf=st.selectbox('Shelf / location',shelves,index=shelves.index(defaults['shelf_id']) if defaults.get('shelf_id') in shelves else 0)
                sku=st.text_input('SKU (leave blank to generate a stable ID)',value=defaults.get('sku',''),disabled=bool(product))
                barcode=st.text_input('Barcode / EAN (optional)',value=defaults.get('barcode') or '')
            description=st.text_area('Description (optional)',value=defaults.get('description') or '')
            if st.form_submit_button('Continue to photographs',type='primary'):
                try:
                    state['details']=details_checked(dict(name=name,brand=brand,variant=variant,package_size=size,unit=unit,price=price,initial_quantity=int(stock),shelf_id=shelf,sku=sku,barcode=barcode,description=description))
                    state['step']=1;st.rerun()
                except ValueError as error:st.error(user_error(error))
        return
    details=state['details']
    st.caption(f"{details['name']} · {details['sku']} · €{details['price']:.2f} · {details['shelf_id']}")
    if state['step']==1:
        st.markdown('### Upload 3–5 photographs of this product')
        st.caption('Mark the clearest front photograph as “front”; it becomes the first catalogue reference and product card image.')
        existing_count=len(product.get('references',[])) if product else 0
        st.caption(f'Recommendation: front, side and back/angled. Existing registered views: {existing_count}. No polygons or model training required.')
        uploads=st.file_uploader('Reference photographs',type=['png','jpg','jpeg'],accept_multiple_files=True,key=key+'_upload')
        with st.expander('Use camera capture'):
            shot=st.camera_input('Capture a product reference',key=key+'_camera')
            if st.button('Add camera photo',key=key+'_camera_add',disabled=shot is None):uploads=list(uploads or [])+[shot]
        collect_photos(state, uploads)
        columns=st.columns(2)
        for index,(digest,photo) in enumerate(list(state['photos'].items())):
            with columns[index%2],st.container(border=True):
                st.image(photo['data'],width=240,caption=photo['name'])
                angles=['front','side','back','angled','unknown']
                photo['angle']=st.selectbox('Reference view',angles,index=angles.index(photo['angle']),key=key+digest+'_angle')
                diagnostics(photo['quality'],'Reference quality')
                if st.button('Analyze this photo on AWS',key=key+digest+'_aws'):
                    photo['approved']=False
                    try:
                        with st.spinner('Explicit upload to existing AWS OpenCV Lambda…'):photo['quality']=aws_analyze(photo['data'])
                        st.rerun()
                    except Exception as error:
                        photo['quality']=dict(provider='UNAVAILABLE',error=user_error(error),image_sha256=digest)
                        st.rerun()
                if photo['quality'].get('provider')=='UNAVAILABLE':
                    if st.button('Use local quality check explicitly',key=key+digest+'_fallback'):
                        photo['quality']=analyze(photo['data']);st.rerun()
                photo['override']=st.text_input('Exposure-only review override reason (optional)',value=photo['override'],key=key+digest+'_override')
                eligible=accepts(photo['quality'],photo['override'])
                if not eligible:st.warning('Replace this image if blurry, too small or unusable. Exposure-only review needs a specific override reason.')
                quality_key=hashlib.sha256(json.dumps(photo['quality'],sort_keys=True).encode()).hexdigest()[:10]
                photo['approved']=st.checkbox('I checked the image, framing and product identity',value=photo['approved'],disabled=not eligible,key=key+digest+quality_key+'_approved') and eligible
                if st.button('Remove / replace photograph',key=key+digest+'_remove'):
                    state['ignored'].append(digest);del state['photos'][digest];st.rerun()
        views={p['angle'] for p in state['photos'].values() if p['approved']}
        st.caption('Suggested views still missing: '+(', '.join(v for v in ('front','side','back') if v not in views) or 'none'))
        approved=[p for p in state['photos'].values() if p['approved']]
        st.info(f'{len(approved)} new approved references · {existing_count} existing · maximum total 5')
        if st.button('Review registration summary',type='primary',disabled=existing_count+len(approved)>5):
            state['step']=2;st.rerun()
        return
    approved=[p for p in state['photos'].values() if p['approved']]
    total=len(approved)+(len(product.get('references',[])) if product else 0)
    st.markdown('### Confirm this product')
    st.write(f"**{details['name']}** · {details['brand']} {details['variant']} {details['package_size']}")
    st.write(f"SKU **{details['sku']}** · **€{details['price']:.2f}** · initial stock **{details['initial_quantity']}** · shelf **{details['shelf_id']}**")
    if approved:st.image([p['data'] for p in approved],width=110)
    st.caption(f'{total}/3–5 approved references. Original images are preserved; optional legacy masked crops are separate. New references use original upright images.')
    complete=3<=total<=5
    if not complete:st.warning('Incomplete draft: this SKU will not enter the active reference gallery.')
    summary_hash=hashlib.sha256(json.dumps([details,[(hashlib.sha256(p['data']).hexdigest(),p['angle'],p['quality'],p['override']) for p in approved]],sort_keys=True).encode()).hexdigest()[:16]
    accepted=st.checkbox('I approve this identity, references, price, shelf and initial stock',key=key+'_final_confirm_'+summary_hash)
    if st.button('Register Product' if complete else 'Save incomplete draft',type='primary',disabled=not accepted):
        try:
            with st.spinner('Caching SigLIP2 references and saving the catalogue atomically…'):
                state['result']=register(service,details,approved,confirmed=accepted,incomplete=not complete,existing_id=product['id'] if product else None)
            st.rerun()
        except Exception as error:st.error(f'Registration was not completed: {user_error(error)}')
    if st.button('Back to photographs',key=key+'_back'):
        state['step']=1;st.rerun()
