"""Fictional workbook and UI checks on temporary DBs; no cloud inference."""
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import db
from scan_review import new_draft, add_manual, edit_item, grouped, reviewed_counts, set_quantity, split_item, price_estimate
from shop_catalog import read_catalog_bundle, BuyerCatalog, validate_catalog, validate_shelves
from shop_map import visit_plan, valid_grid
from tests import test_workflow as workflow
from tests import test_fresh_scan_ui as fresh

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'demo_data/WEB_SHOP_mock_catalog_prices_shelves_map.xlsx'


class PracticalUXTests(unittest.TestCase):
    setUp = workflow.WorkflowTests.setUp
    quantities = workflow.WorkflowTests.quantities

    def test_real_workbook_schema_demo_guard_and_pending_refs(self):
        bundle = read_catalog_bundle(SAMPLE.read_bytes(), SAMPLE.name)
        self.assertEqual((len(validate_catalog(bundle['products'])), len(validate_shelves(bundle['shelves']))), (14, 8))
        with self.assertRaises(ValueError):
            self.owner.import_catalog(bundle['products'], True, shelves=bundle['shelves'], fictional=True)
        with db.connect() as connection:
            connection.execute('UPDATE businesses SET is_demo=1 WHERE id=?', (self.business['id'],))
        before = self.owner.products()[0]['references']
        self.owner.import_catalog(bundle['products'], True, True, bundle['shelves'], True)
        self.assertEqual(self.owner.products(), self.employee.products())
        self.assertEqual(self.owner.products(), BuyerCatalog(self.business['id']).products())
        provisional = next(p for p in self.owner.products() if p['sku'] == 'DEMO-JANA-001')
        self.assertEqual(provisional['image_path'], '')
        self.assertEqual(provisional['references'], [])
        self.assertIn('pending', provisional['registration_status'])
        self.assertEqual(next(p for p in self.owner.products() if p['sku']=='A')['references'], before)
        with db.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM shop_shelves WHERE business_id=?', (self.business['id'],)).fetchone()[0], 8)
        with self.assertRaises(ValueError):
            self.owner.import_catalog(bundle['products'], True, True, bundle['shelves'], True)

    def test_duplicate_mapping_shelf_errors_and_reference_preservation(self):
        record = dict(sku='A', name='Edited name', price='1.23', shelf_id='X')
        refs = self.owner.products()[0]['references']
        with self.assertRaises(ValueError):
            self.owner.import_catalog([record, record], True)
        with self.assertRaises(ValueError):
            self.owner.import_catalog([record], True, shelves=[dict(shelf_id='Y')])
        self.owner.import_catalog([record], True, shelves=[dict(shelf_id='X', map_x=1,map_y=2)])
        self.assertEqual(self.owner.products()[0]['references'], refs)
        with self.assertRaises(PermissionError):
            self.employee.import_catalog([record], True)
        other = db.get_or_create_business('Other shop')
        self.assertEqual(BuyerCatalog(other['id']).products(), [])

    def test_group_add_remove_split_stable_ids_provenance(self):
        draft = new_draft([dict(sku='A',box=[0,0,40,80]), dict(sku='A',box=[40,0,80,80])])
        self.assertEqual(grouped(draft)['A']['quantity'], 2)
        add_manual(draft,'B',3,(100,100),'employee')
        set_quantity(draft,'A',3,(100,100),'employee')
        self.assertEqual(grouped(draft)['A']['quantity'],3)
        set_quantity(draft,'A',2,(100,100),'employee')
        split_item(draft,'D1',[dict(sku='A',box=[0,0,20,80]),dict(sku='B',box=[20,0,40,80])],(100,100),'employee')
        self.assertTrue(next(r for r in draft['rows'] if r['item_id']=='D1')['exclude'])
        self.assertEqual(len({r['item_id'] for r in draft['rows']}),len(draft['rows']))
        with self.assertRaises(ValueError):
            reviewed_counts(draft,['A','B'],(100,100))
        for row in draft['rows']:
            if not row['exclude']:
                edit_item(draft,row['item_id'],{'confirmed':True},'employee')
        self.assertEqual(reviewed_counts(draft,['A','B'],(100,100)), {'A':2,'B':4})
        self.assertTrue(any(e['action']=='split' for e in draft['events']))
        self.assertEqual(self.owner.history(), [])

    def test_unknown_is_not_a_known_unit_and_request_is_scoped(self):
        draft = new_draft([dict(sku=None,box=[0,0,20,20])])
        with self.assertRaises(ValueError):
            reviewed_counts(draft,['A'],(100,100))
        self.employee.request_registration('Unknown bottle / check package', 'scan-fixture')
        self.assertEqual(len(self.owner.registration_requests()),1)
        self.assertEqual(self.quantities(), {'A':None,'B':None})
        with self.assertRaises(PermissionError):
            self.employee.registration_requests()

    def test_prices_exact_cents_unknown_missing_and_zero(self):
        products=[dict(sku='A',name='A',price=.1),dict(sku='B',name='B',price=None),dict(sku='FREE',name='Free',price=0)]
        total=price_estimate(products,{'A':3,'B':2,'UNKNOWN':1,'FREE':2})
        self.assertEqual(total['total'],.30)
        self.assertFalse(total['complete'])
        self.assertEqual(total['excluded'],['B','UNKNOWN'])
        self.assertEqual(total['lines'][-1]['subtotal'],0)

    def test_stock_preview_explicit_scope_and_idempotency(self):
        self.employee.approve(self.location,'photo','initial',{'A':4,'B':2})
        before=self.quantities()
        self.assertEqual(self.employee.preview_action(self.location,{'A':3},'add')[0]['after'],7)
        self.assertEqual(self.quantities(),before)
        with self.assertRaises(ValueError):
            self.employee.preview_action(self.location,{'B':3},'subtract')
        identity=self.employee.approve(self.location,'photo','action',{'A':3},operation='add')
        self.assertEqual(self.employee.approve(self.location,'photo','action',{'A':3},operation='add'),identity)
        self.assertEqual(self.quantities(),{'A':7,'B':2})
        with self.assertRaises(ValueError):
            self.employee.approve(self.location,'photo','action',{'A':4},operation='add')
        other=db.get_or_create_business('Other'); other_location=db.default_location(other['id'])['id']
        with self.assertRaises(PermissionError):
            self.employee.preview_action(other_location,{'A':1},'audit')

    def test_map_valid_route_and_honest_fallback(self):
        layout=dict(grid='E...C\n.#.#.\n.....',shelves=[dict(shelf_id='A',map_x=1,map_y=1),dict(shelf_id='B',map_x=3,map_y=1)])
        plan=visit_plan(layout,['A','B'])
        self.assertTrue(plan['routed'])
        self.assertEqual(set(plan['order']),{'A','B'})
        for x,y in plan['path']:
            self.assertNotEqual(layout['grid'].splitlines()[y][x],'#')
        self.assertFalse(visit_plan(dict(layout,grid=''),['A'])['routed'])
        self.assertFalse(visit_plan(layout,['MISSING'])['routed'])
        self.assertFalse(visit_plan(dict(layout,grid='E#..C\n#####\n.....'),['A','B'])['routed'])

    def test_owner_upload_preview_requires_confirmation(self):
        from streamlit.testing.v1 import AppTest
        upload=io.BytesIO(SAMPLE.read_bytes()); upload.name=SAMPLE.name
        def uploader(*args,**kwargs):
            return upload if kwargs.get('key')=='catalog_import' else ([] if kwargs.get('accept_multiple_files') else None)
        with patch('streamlit.file_uploader',side_effect=uploader):
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
            app.text_input(key='login_username').input('owner'); app.text_input(key='login_password').input('owner123')
            next(b for b in app.button if b.label=='Sign in').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertTrue(any('isolated UX demo' in e.value for e in app.error))
            self.assertTrue(next(b for b in app.button if b.label=='Import catalogue').disabled)
            self.assertEqual(len(self.owner.products()),2)
            with db.connect() as connection:
                connection.execute('UPDATE businesses SET is_demo=1 WHERE id=?', (self.business['id'],))
            app.run()
            next(c for c in app.checkbox if c.label.startswith('I reviewed SKU mappings')).check().run()
            self.assertFalse(next(b for b in app.button if b.label=='Import catalogue').disabled)
            next(b for b in app.button if b.label=='Import catalogue').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(self.owner.products()),16)
            self.assertEqual(self.owner.products(),self.employee.products())


class GroupedUITests(unittest.TestCase):
    setUp = workflow.WorkflowTests.setUp
    quantities = workflow.WorkflowTests.quantities

    def test_draft_manual_review_confirmation_archive_and_buyer_no_write(self):
        photo=fresh.FreshScanUITests.photo(self)
        matcher=Mock(); matcher.rank_boxes.return_value=[dict(sku='A',box=[1,2,300,400],ranked_scores=[dict(sku='A',similarity=.8)])]
        with patch('streamlit.file_uploader',return_value=photo), patch.dict(os.environ,{'WEBSHOP_DINO_URL':'http://127.0.0.1:8765','WEBSHOP_DINO_TOKEN':'mock-only'}), patch('scanner_ui.request_dino',return_value={'detections':[]}), patch('scanner_ui.shop_matcher',return_value=matcher):
            app=fresh.FreshScanUITests.open_scan(self)
            next(b for b in app.button if b.label=='Check this photo').click().run()
            next(b for b in app.button if b.label=='Find products in this photo').click().run()
            self.assertEqual(len(app.exception),0)
            next(b for b in app.button if b.label=='Mark group reviewed').click().run()
            self.assertEqual(next(r for r in app.radio if r.label=='What should happen next?').value,'review')
            next(b for b in app.button if b.label=='+').click().run()
            self.assertEqual(self.owner.history(),[])
            next(b for b in app.button if b.label=='Add missed product').click().run()
            self.assertEqual(len(app.exception),0)
            archives=list((self.root/'fresh_scans').rglob('analysis.json'))
            self.assertEqual(len(archives),1)
            record=json.loads(archives[0].read_text(encoding='utf-8'))
            self.assertEqual(record['reviewed_counts'],{'A':3})
            self.assertEqual(record['draft']['rows'][0]['item_id'],'D1')
            self.assertTrue((archives[0].parent/'analysis.png').exists())
            self.assertTrue((archives[0].parent/record['original_filename']).exists())
            next(r for r in app.radio if r.label=='What should happen next?').set_value('add').run()
            self.assertTrue(next(b for b in app.button if b.label=='Approve inventory action').disabled)
            next(c for c in app.checkbox if c.label.startswith('I checked the scope')).check().run()
            next(b for b in app.button if b.label=='Approve inventory action').click().run()
            self.assertEqual(len(app.exception),0)
            self.assertEqual(self.quantities()['A'],3)
            before=self.quantities()
            next(r for r in app.radio if r.label=='Workspace').set_value('Buyer').run()
            next(c for c in app.checkbox if c.label=='Camera-assisted cart preview').check().run()
            next(b for b in app.button if b.label=='Check this photo').click().run()
            next(b for b in app.button if b.label=='Find products in this photo').click().run()
            self.assertEqual(len(app.exception),0)
            next(b for b in app.button if b.label=='Mark group reviewed').click().run()
            self.assertFalse(any(b.label=='Approve inventory action' for b in app.button))
            self.assertEqual(self.quantities(),before)
            self.assertTrue(any('Estimated priced subtotal'==m.label for m in app.metric))


if __name__ == '__main__':
    unittest.main()
