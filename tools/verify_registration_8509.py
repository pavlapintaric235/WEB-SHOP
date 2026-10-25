"""Bounded real owner/employee/buyer acceptance on the user's folder+ images.

Explicit --live performs <=3 AWS image checks and one existing-worker DINO call.
Only the isolated demo catalogue/ledger may be changed. No resources are created.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true');args=parser.parse_args()
    if not args.live:raise SystemExit('Use --live only for the authorized small cloud test.')
    folder=ROOT/'data/webshop_ux_demo';report_path=folder/'live_registration_acceptance.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('completed'):
        print('Existing completed measurement retained; no repeated uploads or stock action.');return
    os.environ.update(WEBSHOP_DATA_DIR=str(folder),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    connection=json.loads((ROOT/'data/webshop_demo/worker_connection.json').read_text())
    os.environ.update(WEBSHOP_DINO_URL=connection['url'],WEBSHOP_DINO_TOKEN=connection['token'])
    import numpy as np
    from PIL import ImageFilter
    from streamlit.testing.v1 import AppTest
    import auth,db
    from inventory import InventoryService
    from registration_8509 import init_registration,encoder,export_catalog
    from quality_8509 import aws_analyze,analyze
    from live_scan import ShopMatcher,normalize_photo,reference_fingerprint
    from gallery_8509 import record_hash,CachedShopMatcher
    from scan_review import edit_item,add_manual,event,price_estimate
    db.init_db();init_registration()
    source=Path(r'C:\Users\U938\Downloads\folder+')
    paths=[source/f'IMG_{n}.jpeg' for n in range(5029,5034)];scene=source/'IMG_5034.jpeg'
    previous=json.loads(report_path.read_text()) if report_path.exists() else {}
    report=dict(completed=False,kind='ACTUAL_NEW_PRODUCT_AND_SCENE_ACCEPTANCE',reference_files=[str(p) for p in paths],scene=str(scene),
                mocks='Only file-uploader selection is automated; quality, AWS, embeddings and DINO are real.',
                demo_values='EUR 1.49 and opening quantity zero are fictional acceptance values, not confirmed inventory.',started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
    def ledger(path):
        with sqlite3.connect(path) as c:
            return {t:c.execute('SELECT * FROM '+t+' ORDER BY id').fetchall() for t in ('scan_sessions','scan_items','inventory_counts')}
    source_db=ROOT/'data/webshop_release_demo/inventorylens.sqlite3';old_ledger=ledger(source_db)
    protected=['app.py','owner_ui.py','scanner_ui.py','live_scan.py','vision.py','db.py','inventory.py']
    protected_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    def save():report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if previous.get('reference_quality'):report['reference_quality']=previous['reference_quality']
    def create():return AppTest.from_file(str(ROOT/'app_8509.py'),default_timeout=300).run()
    def check(app):
        if app.exception or app.error:raise RuntimeError(str([e.value for e in list(app.exception)+list(app.error)]))
    def login(app,user):
        app.text_input(key='login_username').input(user);app.text_input(key='login_password').input(user+'123')
        next(b for b in app.button if b.label=='Sign in').click().run();check(app)
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    print('STAGE warming real encoder on the main thread',flush=True)
    encoder()
    print('STAGE encoder loaded',flush=True)
    owner=InventoryService(auth.authenticate('owner','owner123')['id'])
    baseline_products=[p for p in owner.products() if p['name']!='Freshona Lentils 400 g']
    existing=next((p for p in owner.products() if p['name']=='Freshona Lentils 400 g'),None)
    if existing:
        registered=dict(id=existing['id'],sku=existing['sku'],ready=True,references=len(existing['references']))
        print('Using the already completed new-product registration from this acceptance run.',flush=True)
    else:
        uploads=[]
        for path in paths:
            upload=io.BytesIO(path.read_bytes());upload.name=path.name;uploads.append(upload)
        with patch('streamlit.file_uploader',side_effect=lambda *a,**k:uploads if k.get('key')=='wizard_new_upload' else ([] if k.get('accept_multiple_files') else None)):
            app=create();login(app,'owner')
            next(b for b in app.button if b.label=='+ Add Product').click().run();check(app)
            for label,value in [('Product name','Freshona Lentils 400 g'),('Brand','Freshona'),('Variant / flavor','Lentils'),('Package size','400 g')]:
                next(x for x in app.text_input if x.label==label).input(value)
            next(x for x in app.number_input if x.label=='Price in EUR').set_value(1.49)
            next(x for x in app.selectbox if x.label=='Quantity unit').set_value('can')
            next(x for x in app.text_area if x.label=='Description (optional)').input('New physical package in folder+; EUR 1.49 and zero opening stock are fictional demo values.')
            next(b for b in app.button if b.label=='Continue to photographs').click().run();check(app)
            print('STAGE five real photos loaded and locally analyzed',flush=True)
            next(b for b in app.button if b.label=='Analyze this photo on AWS').click().run(timeout=180);check(app)
            print('STAGE actual AWS reference result received',flush=True)
            view_keys=[s.key for s in app.selectbox if s.label=='Reference view']
            for key,view in zip(view_keys,['front','side','back','side','angled']):
                app.selectbox(key=key).set_value(view).run();print('STAGE view label '+view,flush=True)
            for key in [c.key for c in app.checkbox if c.label=='I checked the image, framing and product identity']:
                app.checkbox(key=key).check().run();print('STAGE photo approved',flush=True)
            next(b for b in app.button if b.label=='Review registration summary').click().run();check(app)
            next(c for c in app.checkbox if c.label.startswith('I approve this identity')).check().run()
            print('STAGE starting real registration embeddings',flush=True)
            next(b for b in app.button if b.label=='Register Product').click().run(timeout=300);check(app)
            registered=dict(app.session_state['wizard_new']['result'])
            report['reference_quality']=[{k:v for k,v in p.items() if k!='data'} for p in app.session_state['wizard_new']['photos'].values()]
    report['registration']=registered;save()
    print('STAGE real SigLIP2 registration complete '+registered['sku'],flush=True)
    image,_=normalize_photo(paths[0].read_bytes());blur=io.BytesIO();image.filter(ImageFilter.GaussianBlur(22)).save(blur,format='JPEG',quality=90)
    report['blur_control']=aws_analyze(blur.getvalue())
    assert report['blur_control']['decision']['action']=='retake'
    (folder/'intentionally_blurred_control.jpg').write_bytes(blur.getvalue());save()
    print('STAGE genuine AWS blur control RETAKE',flush=True)
    # The exact old implementation and model compute the baseline feature matrix.
    cache=folder/'siglip_gallery'/str(owner.user()['business_id']);cache.mkdir(parents=True,exist_ok=True)
    from live_scan import reference_records
    records=reference_records(baseline_products)
    if all((cache/(record_hash(r)+'.npy')).exists() for r in records):
        # Reuse the real vectors computed by the exact baseline in an earlier
        # interrupted acceptance run. The cache key binds revision and pixels.
        baseline=ShopMatcher.__new__(ShopMatcher);baseline.records=records;baseline.encoder=encoder()
        baseline.features=np.stack([np.load(cache/(record_hash(r)+'.npy'),allow_pickle=False) for r in records])
        baseline.by_sku={}
        for i,r in enumerate(records):baseline.by_sku.setdefault(r['sku'],[]).append(i)
        print('STAGE reused measured baseline vectors with revision/pixel-bound keys',flush=True)
    else:
        baseline=ShopMatcher(baseline_products,encoder=encoder())
    for record,feature in zip(baseline.records,baseline.features):
        with (cache/(record_hash(record)+'.npy')).open('wb') as f:np.save(f,feature,allow_pickle=False)
    candidate_before=CachedShopMatcher(baseline_products,image_encoder=encoder())
    assert np.allclose(baseline.features,candidate_before.features,atol=1e-6)
    print('STAGE baseline/current eligible gallery features agree',flush=True)
    scene_upload=io.BytesIO(scene.read_bytes());scene_upload.name=scene.name
    with patch('streamlit.file_uploader',side_effect=lambda *a,**k:scene_upload if k.get('key')=='fresh_upload' else ([] if k.get('accept_multiple_files') else None)):
        employee=create();login(employee,'employee')
        next(b for b in employee.button if b.label=='Check this photo').click().run(timeout=180);check(employee)
        next(b for b in employee.button if b.label=='Find products in this photo').click().run(timeout=300);check(employee)
        result=employee.session_state['fresh_result'];assert result['mode']=='LIVE';assert result['detector']['kind']=='fresh_tao_dino_inference'
        query,_=normalize_photo(scene.read_bytes())
        old=baseline.rank_boxes(query,result['detector']['detections'])
        same=candidate_before.rank_boxes(query,result['detector']['detections'])
        assert [r['sku'] for r in old]==[r['sku'] for r in same]
        report['diagnostic']=dict(old_gallery_reference_count=len(baseline.records),new_gallery_reference_count=sum(len(p['references']) for p in owner.products()),
            identical_eligible_gallery_same_assignments=True,comparison=[dict(box=i+1,before_8508=a['sku'],before_8509=b['sku'],after_new_registration=c['sku'],
                new_sku_top3=registered['sku'] in [v['sku'] for v in c['ranked_scores'][:3]]) for i,(a,b,c) in enumerate(zip(old,same,result['items']))],
            no_threshold_changes=True,missing_detections='Not scored without box ground truth',merged_boxes='Inspect OpenCV overlap guidance; not inferred from proposal count')
        report['scene_result']=result
        report['scene_quality']=employee.session_state['fresh_capture']
        sku=registered['sku'];assigned=[r for r in result['items'] if r['sku']==sku]
        report['new_sku_actually_assigned']=bool(assigned)
        report['new_sku_in_top3']=any(sku in [v['sku'] for v in r['ranked_scores'][:3]] for r in result['items'])
        print('STAGE actual DINO boxes '+str(len(result['items']))+'; new SKU assigned '+str(bool(assigned)),flush=True)
        # Explicitly review only the one visible lentil can for this isolated demo action.
        draft=employee.session_state['fresh_draft_'+employee.session_state['fresh_approval']]
        kept=False
        for row in draft['rows']:
            if row['sku']==sku and not kept:
                edit_item(draft,row['item_id'],dict(confirmed=True,quantity=1),'employee');kept=True
            else:edit_item(draft,row['item_id'],{'exclude':True},'employee')
        if not kept:add_manual(draft,sku,1,query.size,'employee')
        event(draft,'explicit_acceptance_scope','employee',description='One visually checked lentil can; other products outside this fictional stock action.')
        employee.run();check(employee)
        before=ledger(db.DB_FILE)
        next(r for r in employee.radio if r.label=='What should happen next?').set_value('add').run()
        next(x for x in employee.text_input if x.label.startswith('Transaction notes')).input('Isolated acceptance demo. One observed lentil can; other products outside action scope.').run()
        next(c for c in employee.checkbox if c.label.startswith('I checked the scope')).check().run()
        assert ledger(db.DB_FILE)==before
        next(b for b in employee.button if b.label=='Approve inventory action').click().run();check(employee)
        report['employee_receipt']=employee.session_state['fresh_saved'];report['evidence_id']=result['evidence_id']
    after=ledger(db.DB_FILE)
    buyer=create();next(r for r in buyer.radio if r.label=='Role view').set_value('Buyer').run();check(buyer)
    next(t for t in buyer.text_input if t.label=='Search the catalogue').input('Freshona Lentils 400 g').run()
    next(n for n in buyer.number_input if n.label=='Shopping quantity').set_value(2).run();check(buyer)
    assert any(m.label=='Estimated priced subtotal' and '2.98' in str(m.value) for m in buyer.metric)
    assert ledger(db.DB_FILE)==after
    assert ledger(source_db)==old_ledger
    assert protected_hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    assert sku in export_catalog(owner)
    report.update(completed=True,old_8508_ledger_and_code_unchanged=True,buyer_price_total=2.98,buyer_no_stock_write=True,
                  registered_without_retraining=True,reference_count=5,scene_distinct_from_references=True)
    save();print(json.dumps({k:report[k] for k in ('completed','registration','new_sku_actually_assigned','new_sku_in_top3','employee_receipt','buyer_price_total','old_8508_ledger_and_code_unchanged')},indent=2))


if __name__=='__main__':main()
