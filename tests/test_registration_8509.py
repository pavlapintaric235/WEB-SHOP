"""8509 registration acceptance on temporary DBs; embeddings are explicitly mocked."""
import io
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

import numpy as np
from PIL import Image,ImageFilter
from streamlit.testing.v1 import AppTest

import db
from inventory import InventoryService
from tests import test_workflow as workflow
from registration_8509 import init_registration,register,assets_for,update_details,export_catalog
from quality_8509 import analyze,accepts,box_quality
from gallery_8509 import CachedShopMatcher,gallery_fingerprint
from shop_catalog import BuyerCatalog


class RegistrationTests(unittest.TestCase):
    setUp=workflow.WorkflowTests.setUp

    def test_owner_adds_email_employee_and_employee_signs_in(self):
        with patch.dict('os.environ', {'WEBSHOP_DATA_DIR': str(self.root)}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app_8509.py'), default_timeout=60).run()
            app.text_input(key='login_username').input('owner')
            app.text_input(key='login_password').input('owner123')
            next(b for b in app.button if b.label == 'Sign in').click().run()
            app.radio(key='owner_section_8509').set_value('Employees').run()
            next(t for t in app.text_input if t.label == 'Employee email or username').input('Video+staff@Example.org')
            next(t for t in app.text_input if t.label == 'Initial password').input('password123')
            next(b for b in app.button if b.label == 'Add employee').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            next(b for b in app.button if b.label == 'Sign out').click().run()
            app.text_input(key='login_username').input(' VIDEO+STAFF@example.org ')
            app.text_input(key='login_password').input('password123')
            next(b for b in app.button if b.label == 'Sign in').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any('video+staff@example.org' in c.value for c in app.caption))
            self.assertEqual(next(r for r in app.radio if r.label == 'Workspace').value, 'Employee Scan')

    def prepare(self):
        init_registration()
        data=[]
        for index in range(5):
            image=Image.fromarray(np.random.default_rng(index+80).integers(40,210,(800,800,3),dtype=np.uint8))
            stream=io.BytesIO();image.save(stream,format='PNG');raw=stream.getvalue()
            data.append(dict(data=raw,name=f'view{index}.png',quality=analyze(raw),angle='front' if not index else 'angled',approved=True))
        encoder=Mock();encoder.encode.side_effect=lambda images,**k:np.tile(np.eye(1,768,0,dtype=np.float32),(len(images),1))
        details=dict(name='New lentil can',brand='Freshona',variant='Lentils',package_size='400 g',unit='can',price=1.49,initial_quantity=0,shelf_id='Main Shelf',sku='')
        return details,data,encoder

    def test_five_references_persist_unique_export_gallery_and_edit(self):
        details,photos,encoder=self.prepare()
        photos[0]['angle']='back'
        photos[3]['angle']='front'
        result=register(self.owner,details,photos,confirmed=True,image_encoder=encoder)
        self.assertTrue(result['ready']);self.assertTrue(result['sku'].startswith('P-'))
        db.init_db();init_registration()
        product=db.get_product_by_sku(self.business['id'],result['sku'])
        full=next(p for p in self.owner.products() if p['id']==result['id'])
        self.assertEqual(len(full['references']),5)
        self.assertEqual(Path(full['image_path']).read_bytes(),photos[3]['data'])
        self.assertEqual(full['references'][0]['image_path'],full['image_path'])
        self.assertEqual(len(assets_for(self.owner,result['id'])),5)
        matcher=CachedShopMatcher([full],image_encoder=encoder)
        self.assertEqual(matcher.features.shape,(5,768))
        self.assertTrue(all(r['feature_source']=='registered_embedding' for r in matcher.manifest))
        with self.assertRaises(ValueError):register(self.owner,details,photos,confirmed=True,image_encoder=encoder)
        update_details(self.owner,result['id'],2.25,'Main Shelf')
        self.assertEqual(next(p for p in BuyerCatalog(self.business['id']).products() if p['id']==result['id'])['price'],2.25)
        self.assertIn(result['sku'],export_catalog(self.owner))
        from catalog_workbook import export_workbook
        from shop_catalog import read_catalog_bundle
        sheet=read_catalog_bundle(export_workbook(self.owner),'current.xlsx')['products']
        exported=next(p for p in sheet if p['sku']==result['sku'])
        self.assertEqual(exported['reference_count'],'5')
        self.assertEqual(exported['price'],'2.25')
        self.assertEqual(db.get_inventory(self.business['id'],self.location)[-1]['quantity'],0)

    def test_incomplete_draft_is_not_an_active_gallery_and_can_resume(self):
        details,photos,encoder=self.prepare()
        result=register(self.owner,details,photos[:2],confirmed=True,incomplete=True,image_encoder=encoder)
        product=next(p for p in self.owner.products() if p['id']==result['id'])
        self.assertFalse(result['ready']);self.assertEqual(product['references'],[]);self.assertEqual(product['image_path'],'')
        self.assertFalse(any(a['active'] for a in assets_for(self.owner,result['id'])))
        details['sku']=result['sku']
        completed=register(self.owner,details,photos[:3],confirmed=True,existing_id=result['id'],image_encoder=encoder)
        self.assertTrue(completed['ready']);self.assertEqual(completed['id'],result['id'])

    def test_encoding_failure_rolls_back_product_and_files(self):
        details,photos,encoder=self.prepare();encoder.encode.side_effect=RuntimeError('simulated encoder failure')
        before=self.owner.products()
        with self.assertRaises(RuntimeError):register(self.owner,details,photos[:3],confirmed=True,image_encoder=encoder)
        self.assertEqual(self.owner.products(),before)
        self.assertFalse(list((self.root/'registered_references').rglob('*_original*')))

    def test_actual_opencv_blur_rejects_and_geometry_flags_overlap(self):
        details,photos,encoder=self.prepare()
        image=Image.open(io.BytesIO(photos[0]['data'])).filter(ImageFilter.GaussianBlur(20))
        stream=io.BytesIO();image.save(stream,format='PNG')
        record=analyze(stream.getvalue())
        self.assertEqual(record['decision']['action'],'retake');self.assertFalse(accepts(record,'I approve anyway'))
        photos[0].update(data=stream.getvalue(),quality=record)
        with self.assertRaises(ValueError):register(self.owner,details,photos[:3],confirmed=True,image_encoder=encoder)
        report=box_quality(Image.new('RGB',(800,800)),[dict(box=[0,0,400,400]),dict(box=[50,50,390,390])])
        self.assertTrue(report['boxes'][0]['near_frame']);self.assertTrue(report['boxes'][0]['overlaps'])

    def test_permissions_cross_shop_and_cached_representation_freshness(self):
        details,photos,encoder=self.prepare()
        with self.assertRaises(PermissionError):register(self.employee,details,photos[:3],confirmed=True,image_encoder=encoder)
        other=db.get_or_create_business('Other shop')
        __import__('auth').create_user(other['id'],'other-owner','long-password','owner')
        user=__import__('auth').authenticate('other-owner','long-password')['id']
        with self.assertRaises(PermissionError):update_details(InventoryService(user),self.owner.products()[0]['id'],2.0,'Main Shelf')
        with self.assertRaises(PermissionError):CachedShopMatcher([dict(business_id=1),dict(business_id=2)],image_encoder=encoder)
        from sam_vision import REFERENCE_CACHE_VERSION
        data=Path(self.owner.products()[0]['image_path']).read_bytes()
        cache=self.root/'mask_cache';cache.mkdir()
        filename=hashlib.sha256(REFERENCE_CACHE_VERSION.encode('ascii')+data).hexdigest()+'.npz'
        with patch('gallery_8509.REFERENCE_CACHE_ROOT',cache):
            first=gallery_fingerprint(self.owner.products())
            (cache/filename).write_bytes(b'first cached representation')
            second=gallery_fingerprint(self.owner.products())
            (cache/filename).write_bytes(b'changed cached representation')
            third=gallery_fingerprint(self.owner.products())
        self.assertNotEqual(first,second);self.assertNotEqual(second,third)

    def test_wizard_five_views_owner_to_buyer_with_mocked_embeddings(self):
        details,photos,encoder=self.prepare()
        uploads=[]
        for i,p in enumerate(photos):
            stream=io.BytesIO(p['data']);stream.name=f'photo{i}.png';uploads.append(stream)
        def uploader(*a,**k):return uploads if k.get('key')=='wizard_new_upload' else ([] if k.get('accept_multiple_files') else None)
        with patch.dict('os.environ',{'WEBSHOP_DATA_DIR':str(self.root)}),patch('streamlit.file_uploader',side_effect=uploader),patch('registration_8509.encoder',return_value=encoder):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_8509.py'),default_timeout=60).run()
            app.text_input(key='login_username').input('owner');app.text_input(key='login_password').input('owner123')
            next(b for b in app.button if b.label=='Sign in').click().run()
            next(b for b in app.button if b.label=='+ Add Product').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.session_state['wizard_new']['photos']),5)
            for label,value in [('Product name','Brand new package'),('Brand','Demo brand'),('Package size','400 g')]:
                next(x for x in app.text_input if x.label==label).input(value)
            next(x for x in app.number_input if x.label=='Price in EUR').set_value(1.49)
            next(b for b in app.button if b.label=='Continue to photographs').click().run()
            for key in [c.key for c in app.checkbox if c.label=='I checked the image, framing and product identity']:
                app.checkbox(key=key).check().run()
            next(b for b in app.button if b.label=='Review registration summary').click().run()
            next(c for c in app.checkbox if c.label.startswith('I approve this identity')).check().run()
            next(b for b in app.button if b.label=='Register Product').click().run()
            self.assertEqual(len(app.exception),0);self.assertFalse(app.error)
            self.assertTrue(any('embedded' in m.value for m in app.success))
            self.assertTrue(any(b.label=='Download current Excel workbook' for b in app.get('download_button')))
            self.assertEqual(len(BuyerCatalog(self.business['id']).products()),3)

    def test_image_tower_matches_full_model_image_feature_math(self):
        import torch
        from transformers import SiglipConfig,SiglipModel,SiglipVisionModel
        torch.set_num_threads(2)
        configuration=SiglipConfig(vision_config=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,image_size=32,patch_size=16),
                                  text_config=dict(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,vocab_size=64))
        full=SiglipModel(configuration).eval()
        vision=SiglipVisionModel(configuration.vision_config).eval()
        vision.vision_model.load_state_dict(full.vision_model.state_dict(),strict=True)
        inputs=dict(pixel_values=torch.randn(1,3,32,32))
        with torch.inference_mode():
            expected=full.get_image_features(**inputs)
            actual=vision(**inputs).pooler_output
        self.assertTrue(torch.equal(expected,actual))
        from siglip_8509 import load_image_weights
        checkpoint=self.root/'tiny_equivalence_checkpoint'
        full.save_pretrained(checkpoint)
        loaded=load_image_weights(checkpoint,SiglipVisionModel,configuration.vision_config)
        with torch.inference_mode():
            self.assertTrue(torch.equal(expected,loaded(**inputs).pooler_output))

    def test_employee_aws_unavailable_is_visible_and_local_is_explicit(self):
        from tests.test_fresh_scan_ui import FreshScanUITests
        self.prepare()
        photo=FreshScanUITests.photo(self)
        with patch.dict('os.environ',{'WEBSHOP_DATA_DIR':str(self.root),'WEBSHOP_DINO_URL':'','WEBSHOP_DINO_TOKEN':''}), \
             patch('streamlit.file_uploader',return_value=photo), \
             patch('experiments.opencv5_guided_scan.aws_capture_client.run_capture',side_effect=RuntimeError('simulated AWS unavailable')) as aws:
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_8509.py'),default_timeout=60).run()
            app.text_input(key='login_username').input('employee');app.text_input(key='login_password').input('employee123')
            next(b for b in app.button if b.label=='Sign in').click().run()
            self.assertEqual(next(r for r in app.radio if r.label=='Capture check').value,'AWS capture service')
            next(b for b in app.button if b.label=='Check this photo').click().run()
            self.assertTrue(any(m.label=='Analysis provider' and m.value=='UNAVAILABLE' for m in app.metric))
            self.assertEqual(aws.call_count,1)
            next(r for r in app.radio if r.label=='Capture check').set_value('On this computer').run()
            next(b for b in app.button if b.label=='Check this photo').click().run()
            self.assertTrue(any(m.label=='Analysis provider' and m.value=='LOCAL' for m in app.metric))
            self.assertEqual(aws.call_count,1)
            self.assertEqual(self.owner.history(),[])

    def test_gallery_snapshot_reuses_real_stored_vectors_without_decoding_references(self):
        details,photos,image_encoder=self.prepare()
        result=register(self.owner,details,photos,confirmed=True,image_encoder=image_encoder)
        product=next(p for p in self.owner.products() if p['id']==result['id'])
        first=CachedShopMatcher([product],image_encoder=image_encoder)
        with patch('gallery_8509.reference_records',side_effect=AssertionError('Cached galleries must not decode reference photos')):
            second=CachedShopMatcher([product],image_encoder=image_encoder)
        np.testing.assert_array_equal(first.features,second.features)
        self.assertEqual(first.by_sku,second.by_sku)
        from live_scan import ShopMatcher
        baseline=ShopMatcher.__new__(ShopMatcher)
        baseline.__dict__.update(first.__dict__)
        query=Image.new('RGB',(800,800),'white')
        boxes=[dict(box=[0,0,400,400],confidence=.8),dict(box=[400,400,800,800],confidence=.7)]
        self.assertEqual(baseline.rank_boxes(query,boxes),second.rank_boxes(query,boxes))
        self.assertEqual(image_encoder.encode.call_args.kwargs['batch_size'],4)
        # An updated authoritative embedding must invalidate the snapshot.
        asset=assets_for(self.owner,result['id'])[0]
        with np.load(asset['embedding_path'],allow_pickle=False) as saved:
            metadata={k:saved[k] for k in saved.files}
        metadata['feature']=np.eye(1,768,1,dtype=np.float32)[0]
        np.savez_compressed(asset['embedding_path'],**metadata)
        third=CachedShopMatcher([product],image_encoder=image_encoder)
        self.assertFalse(np.array_equal(second.features,third.features))

    def test_price_change_keeps_gallery_and_buyer_map_keeps_list_without_stock_writes(self):
        details,photos,image_encoder=self.prepare()
        result=register(self.owner,details,photos,confirmed=True,image_encoder=image_encoder)
        products=[p for p in self.owner.products() if p['id']==result['id']]
        from scanner_8509 import shop_matcher
        fingerprint=gallery_fingerprint(products)
        first=shop_matcher(json.dumps(products),fingerprint)
        products[0]['price']=3.99
        self.assertIs(first,shop_matcher(json.dumps(products),fingerprint))
        before=self.owner.history()
        with db.connect() as c:
            c.execute('INSERT INTO shop_shelves(business_id,shelf_id,zone,aisle,map_x,map_y) VALUES (?,?,?,?,?,?)',(self.business['id'],'Main Shelf','','',1,1))
        with patch.dict('os.environ',{'WEBSHOP_DATA_DIR':str(self.root)}):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_8509.py'),default_timeout=60).run()
            next(r for r in app.radio if r.label=='Role view').set_value('Buyer').run()
            next(t for t in app.text_input if t.label=='Search the catalogue').input('New lentil can').run()
            next(n for n in app.number_input if n.label=='Shopping quantity').set_value(2).run()
            next(r for r in app.radio if r.label=='Buyer section').set_value('Shopping list & map').run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label=='Estimated priced subtotal' and '2.98' in str(m.value) for m in app.metric))
            self.assertTrue(any('Ordered shelf list only' in c.value for c in app.caption))
            next(r for r in app.radio if r.label=='Buyer section').set_value('Products').run()
            self.assertEqual(next(n for n in app.number_input if n.label=='Shopping quantity').value,2)
        self.assertEqual(self.owner.history(),before)


if __name__=='__main__':unittest.main()
