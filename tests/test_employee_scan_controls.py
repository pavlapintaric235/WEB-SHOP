"""Visible review and approval controls, using simulated inference and a temporary shop."""
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest
from tests import test_workflow as workflow
from tests.test_fresh_scan_ui import FreshScanUITests


class EmployeeScanControlsTests(unittest.TestCase):
    setUp = workflow.WorkflowTests.setUp

    def test_excel_and_locked_approval_visible_then_explicit_add(self):
        import db
        with db.connect() as c:
            c.execute("UPDATE products SET shelf_id='Main Shelf' WHERE business_id=?",(self.business['id'],))
        photo = FreshScanUITests.photo(self)
        matcher = Mock()
        matcher.rank_boxes.return_value = [
            dict(sku='A',box=[5,5,300,400],ranked_scores=[dict(sku='A',similarity=.8)]),
            dict(sku='',box=[400,5,700,400],ranked_scores=[]),
        ]
        detector = dict(detections=[dict(box=[5,5,300,400],confidence=.9),dict(box=[400,5,700,400],confidence=.9)])
        with patch.dict(os.environ, {'WEBSHOP_DATA_DIR':str(self.root),'WEBSHOP_DINO_URL':'http://127.0.0.1:8765','WEBSHOP_DINO_TOKEN':'simulated-only'}), \
             patch('streamlit.file_uploader',return_value=photo), \
             patch('scanner_8509.request_dino',return_value=detector) as worker, \
             patch('scanner_8509.shop_matcher',return_value=matcher):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app_8509.py'),default_timeout=60).run()
            app.text_input(key='login_username').input('employee')
            app.text_input(key='login_password').input('employee123')
            next(b for b in app.button if b.label=='Sign in').click().run()
            next(r for r in app.radio if r.label=='Capture check').set_value('On this computer').run()
            next(b for b in app.button if b.label=='Check this photo').click().run()
            next(b for b in app.button if b.label=='Find products in this photo').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Excel preview' in h.value for h in app.subheader))
            self.assertTrue(any(b.label=='Download scan review Excel' for b in app.get('download_button')))
            self.assertTrue(next(b for b in app.button if b.label=='Approve inventory action').disabled)
            self.assertEqual(self.owner.history(),[])
            next(b for b in app.button if b.label=='Mark group reviewed').click().run()
            [s for s in app.selectbox if s.label=='Product for this box'][1].set_value('B')
            [c for c in app.checkbox if c.label=='I reviewed this instance'][1].check()
            [b for b in app.button if b.label=='Save box correction'][1].click().run()
            next(r for r in app.radio if r.label=='What should happen next?').set_value('add').run()
            self.assertTrue(next(c for c in app.checkbox if c.label=="Use each product's assigned shelf").value)
            self.assertFalse(app.exception)
            self.assertTrue(any(b.label=='Download stock change Excel' for b in app.get('download_button')))
            self.assertTrue(next(b for b in app.button if b.label=='Approve inventory action').disabled)
            next(c for c in app.checkbox if c.label.startswith('I checked the scope')).check().run()
            next(b for b in app.button if b.label=='Approve inventory action').click().run()
            self.assertFalse(app.exception)
            self.assertEqual({r['sku']:r['quantity'] for r in self.owner.inventory(self.location)}, {'A':1,'B':1})
            self.assertEqual(len({r['id'] for r in self.owner.history()}),1)
            self.assertTrue(any(b.label=='Download scan review Excel' for b in app.get('download_button')))
            self.assertFalse(any(b.label=='Approve inventory action' for b in app.button))
            next(b for b in app.button if b.label=='Sign out').click().run()
            app.text_input(key='login_username').input('employee')
            app.text_input(key='login_password').input('employee123')
            next(b for b in app.button if b.label=='Sign in').click().run()
            next(r for r in app.radio if r.label=='Capture check').set_value('On this computer').run()
            next(b for b in app.button if b.label=='Check this photo').click().run()
            next(b for b in app.button if b.label=='Resume saved review for this photo').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Saved inventory receipt' in s.value for s in app.success))
            self.assertEqual(worker.call_count,1)
            self.assertEqual(len({r['id'] for r in self.owner.history()}),1)
