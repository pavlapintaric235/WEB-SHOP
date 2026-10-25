"""Release checks on isolated storage; simulated inference is explicitly mocked."""
import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

import db
from inventory import InventoryService
from live_scan import normalize_photo, replay_dino, review_counts, render_items, ShopMatcher
from shop_catalog import BuyerCatalog, read_catalog, shopping_list
from tests import test_workflow as workflow
from tools.prepare_webshop_demo import write_workbook


class DeliveryTests(unittest.TestCase):
    setUp = workflow.WorkflowTests.setUp
    quantities = workflow.WorkflowTests.quantities

    def record(self, **extra):
        return dict(sku='A', name='Owner product', price='1.25', unit='package', shelf_id='A1',
                    initial_quantity='8', registration_status='owner confirmed', **extra)

    def test_A_B_K_shared_workbook_and_shop_catalogue(self):
        workbook = self.root / 'catalog.xlsx'
        row = self.record()
        write_workbook(workbook, [('Products', [list(row), list(row.values())]), ('Shelves', [['shelf_id'], ['A1']])])
        records = read_catalog(workbook.read_bytes(), workbook.name)
        self.owner.import_catalog(records, confirmed=True, opening=True)
        buyer = BuyerCatalog(self.business['id'])
        self.assertEqual(self.owner.products(), self.employee.products())
        self.assertEqual(self.owner.products(), buyer.products())
        p = buyer.products()[0]
        self.assertEqual((p['price'], p['shelf_id'], p['unit']), (1.25, 'A1', 'package'))
        self.assertEqual(len(p['references']), 1)

    def test_import_rollback_invalid_rows_and_existing_opening_stock(self):
        with self.assertRaises(ValueError):
            self.owner.import_catalog([self.record(), dict(sku='C', name='C', price='nan')], True)
        self.assertEqual(self.owner.products()[0]['name'], 'Product A')
        self.owner.import_catalog([self.record()], True, True)
        with self.assertRaises(ValueError):
            self.owner.import_catalog([dict(sku='C', name='New'), self.record()], True, True)
        self.assertEqual([p['sku'] for p in self.owner.products()], ['A', 'B'])

    def test_I_J_audit_add_subtract_and_duplicate(self):
        for key, operation, count in [('audit', 'audit', 8), ('receive', 'add', 3), ('remove', 'subtract', 2)]:
            result = self.employee.approve(self.location, 'photo', key, {'A': count}, operation=operation)
            self.assertEqual(self.employee.approve(self.location, 'photo', key, {'A': count}, operation=operation), result)
        self.assertEqual(self.quantities()['A'], 9)
        history = self.owner.history()
        self.assertEqual([(r['operation'], r['before_quantity'], r['after_quantity']) for r in history],
                         [('subtract', 11, 9), ('add', 8, 11), ('audit', 0, 8)])
        with self.assertRaises(ValueError):
            self.employee.approve(self.location, 'photo', 'receive', {'A': 3}, operation='subtract')

    def test_negative_removal_rolls_back_whole_transaction(self):
        self.employee.approve(self.location, 'photo', 'start', {'A': 5, 'B': 1})
        with self.assertRaises(ValueError):
            self.employee.approve(self.location, 'photo', 'bad', {'A': 1, 'B': 2}, operation='subtract')
        self.assertEqual(self.quantities(), {'A': 5, 'B': 1})
        self.assertEqual(len(self.owner.history()), 2)

    def test_L_buyer_list_and_preview_have_no_writer(self):
        self.owner.import_catalog([self.record()], True)
        buyer = BuyerCatalog(self.business['id'])
        before = self.quantities()
        items = shopping_list(buyer.products(), {'A': 3})
        self.assertEqual((items[0]['quantity'], items[0]['price']), (3, 1.25))
        self.assertFalse(hasattr(buyer, 'approve'))
        self.assertEqual(self.quantities(), before)
        self.assertEqual(self.owner.history(), [])

    def test_N_employee_cannot_import_or_write_other_shop(self):
        other = db.get_or_create_business('Other')
        location = db.default_location(other['id'])['id']
        with self.assertRaises(PermissionError):
            self.employee.import_catalog([self.record()], True)
        with self.assertRaises(PermissionError):
            self.employee.approve(location, 'photo', 'other', {'A': 2}, operation='add')
        with self.assertRaises(ValueError):
            shopping_list(BuyerCatalog(other['id']).products(), {'A': 3})

    def test_pending_product_references_can_be_registered_later(self):
        self.owner.import_catalog([dict(sku='NEW', name='Pending', registration_status='pending')], True)
        self.owner.add_references('NEW', self.image)
        p = next(p for p in self.owner.products() if p['sku'] == 'NEW')
        self.assertTrue(Path(p['image_path']).is_file())
        db.init_db()
        self.assertEqual(len(next(p for p in self.owner.products() if p['sku'] == 'NEW')['references']), 1)

    def test_renamed_demo_shop_is_not_duplicated_on_role_reload(self):
        import auth
        with db.connect() as connection:
            connection.execute('UPDATE businesses SET name=? WHERE id=?', ('Fictional demo', self.business['id']))
        self.assertEqual(auth.seed_demo_users()['id'], self.business['id'])
        with db.connect() as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM businesses').fetchone()[0], 1)

    def test_zero_opening_quantity_and_price_remain_explicit_zero(self):
        record = self.record(); record.update(price=0, initial_quantity=0)
        self.owner.import_catalog([record], True, True)
        self.assertEqual(self.owner.products()[0]['price'], 0)

    def test_O_three_roles_without_database_reset(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=30).run()
        app.text_input(key='login_username').input('owner')
        app.text_input(key='login_password').input('owner123')
        next(b for b in app.button if b.label == 'Sign in').click().run()
        self.owner.import_catalog([self.record()], True)
        app.run()
        self.assertEqual(len(app.exception), 0)
        next(b for b in app.button if b.label == 'Sign out').click().run()
        app.text_input(key='login_username').input('employee')
        app.text_input(key='login_password').input('employee123')
        next(b for b in app.button if b.label == 'Sign in').click().run()
        self.assertEqual(self.employee.products()[0]['price'], 1.25)
        next(r for r in app.radio if r.label == 'Workspace').set_value('Buyer').run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any('Owner product' in str(frame.value) for frame in app.dataframe))
        next(n for n in app.number_input if n.label.startswith('A ·')).set_value(3).run()
        next(t for t in app.text_input if t.label == 'Search products').input('Product B').run()
        app.run()
        next(t for t in app.text_input if t.label == 'Search products').input('').run()
        self.assertEqual(next(n for n in app.number_input if n.label.startswith('A ·')).value, 3)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(self.owner.history(), [])


class GeometryTests(unittest.TestCase):
    def test_D_exif_rotated_crop_and_overlay_share_upright_pixels(self):
        source = Image.new('RGB', (80, 40), 'blue')
        source.paste('red', (0, 0, 40, 40))
        exif = Image.Exif(); exif[274] = 6
        stream = io.BytesIO(); source.save(stream, format='JPEG', exif=exif)
        image, data = normalize_photo(stream.getvalue())
        self.assertEqual(image.size, (40, 80))
        crops = []
        class Encoder:
            def encode(self, images, batch_size=1):
                crops.extend(images)
                return np.array([[1., 0.] for _ in images])
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'ref.png'; Image.new('RGB', (40, 40), 'red').save(path)
            matcher = ShopMatcher([dict(sku='A', name='A', image_path=str(path))], Encoder())
            matcher.rank_boxes(image, [dict(box=[0, 0, 40, 40], confidence=.9)])
        self.assertGreater(crops[-1].getpixel((20, 20))[0], 230)
        overlay = render_items(image, [dict(box=[0, 0, 40, 40], sku='A')])
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(overlay.getpixel((20, 70)), image.getpixel((20, 70)))
        self.assertNotEqual(overlay.getpixel((0, 0)), image.getpixel((0, 0)))

    def test_E_F_G_H_repetition_correction_missing_split_unknown(self):
        row = dict(x1=0, y1=0, x2=20, y2=30, sku='A', quantity=1, confirmed=True)
        self.assertEqual(review_counts([row, row, row], {'A'}, (100, 100)), {'A': 3})
        self.assertEqual(review_counts([row, {**row, 'sku': 'B'}], {'A', 'B'}, (100, 100)), {'A': 1, 'B': 1})
        self.assertEqual(review_counts([{**row, 'exclude': True}, row, {**row, 'x1': 30, 'x2': 50}], {'A'}, (100, 100)), {'A': 2})
        with self.assertRaises(ValueError):
            review_counts([{**row, 'sku': '', 'confirmed': False}], {'A'}, (100, 100))

    def test_replay_rejects_different_image(self):
        archive = Path(__file__).resolve().parents[1] / 'data/webshop_demo/live_worker_smoke.json'
        if not archive.exists():
            self.skipTest('Private replay evidence is local only')
        with self.assertRaisesRegex(ValueError, 'different photo'):
            replay_dino(archive, b'wrong photo', (4032, 3024))


class CaptureUITests(unittest.TestCase):
    setUp = workflow.WorkflowTests.setUp

    def test_confirmed_repeated_items_add_only_after_explicit_approval(self):
        import pandas as pd
        import os
        from tests.test_fresh_scan_ui import FreshScanUITests
        detector = {'detections': [{'box': [1, 2, 300, 400], 'confidence': .9}] * 3}
        item = {'sku': 'A', 'box': [1, 2, 300, 400], 'ranked_scores': [{'sku': 'A', 'similarity': .8}]}
        matcher = Mock(); matcher.rank_boxes.return_value = [item, item, item]
        with patch('streamlit.file_uploader', return_value=FreshScanUITests.photo(self)), \
             patch.dict(os.environ, {'WEBSHOP_DINO_URL': 'http://127.0.0.1:8765', 'WEBSHOP_DINO_TOKEN': 'mock-only'}), \
             patch('scanner_ui.request_dino', return_value=detector), \
             patch('scanner_ui.shop_matcher', return_value=matcher):
            app = FreshScanUITests.open_scan(self)
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            next(b for b in app.button if b.label == 'Find products in this photo').click().run()
            self.assertEqual(self.owner.history(), [])
            next(b for b in app.button if b.label == 'Mark group reviewed').click().run()
            next(s for s in app.radio if s.label == 'What should happen next?').set_value('add').run()
            self.assertTrue(next(b for b in app.button if b.label == 'Approve inventory action').disabled)
            next(c for c in app.checkbox if c.label.startswith('I checked the scope')).check().run()
            next(b for b in app.button if b.label == 'Approve inventory action').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(self.owner.history()[0]['approved_quantity'], 3)
            self.assertEqual(self.owner.history()[0]['operation'], 'add')
            self.assertFalse(any(b.label == 'Approve inventory action' for b in app.button))
            next(r for r in app.radio if r.label == 'Workspace').set_value('Buyer').run()
            next(c for c in app.checkbox if c.label == 'Camera-assisted cart preview').check().run()
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            next(b for b in app.button if b.label == 'Find products in this photo').click().run()
            self.assertEqual(len(app.exception), 0)
            next(b for b in app.button if b.label == 'Mark group reviewed').click().run()
            self.assertTrue(any('Reviewed cart preview' in s.value for s in app.success))
            self.assertFalse(any(b.label == 'Approve inventory action' for b in app.button))
            self.assertEqual(len(self.owner.history()), 1)

    def test_M_aws_unavailable_requires_explicit_local_fallback_and_dark_retake(self):
        from tests.test_fresh_scan_ui import FreshScanUITests
        # The AWS failure is simulated; this test performs no upload or cloud invocation.
        with patch('streamlit.file_uploader', return_value=FreshScanUITests.photo(self)), \
             patch('experiments.opencv5_guided_scan.aws_capture_client.run_capture', side_effect=RuntimeError('simulated unavailable')) as cloud:
            app = FreshScanUITests.open_scan(self)
            next(r for r in app.radio if r.label == 'Capture check').set_value('AWS capture service').run()
            self.assertEqual(cloud.call_count, 0)
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            self.assertEqual(cloud.call_count, 1)
            self.assertTrue(any('explicitly select the local check' in e.value for e in app.error))
            next(r for r in app.radio if r.label == 'Capture check').set_value('On this computer').run()
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            self.assertTrue(any('Local OpenCV 5' in c.value for c in app.caption))
            self.assertEqual(cloud.call_count, 1)
        stream = io.BytesIO(); Image.new('RGB', (800, 800), 'black').save(stream, format='PNG')
        with patch('streamlit.file_uploader', return_value=stream):
            app = FreshScanUITests.open_scan(self)
            next(b for b in app.button if b.label == 'Check this photo').click().run()
            self.assertTrue(any('more light' in w.value for w in app.warning))
            self.assertFalse(any(b.label == 'Find products in this photo' for b in app.button))


if __name__ == '__main__':
    unittest.main()
